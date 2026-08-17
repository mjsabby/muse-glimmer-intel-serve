# Build plan — Muse Glimmer 30B on Intel Arc

Target: the same deliverable the two sibling repos already ship — a float64 CPU
oracle that defines the model function exactly, a deterministic bf16/f16 twin
that defines the acceptable deviation band, a SYCL engine that is tensor-parallel
across two Arc Pro B70s, and a Python serving frontend speaking OpenAI Chat
Completions / OpenAI Responses / Anthropic Messages with streaming, guided JSON,
tool calling, speculative decoding, and image/video input.

The two references are vendored as submodules and are the primary design input:

| submodule | what to lift from it |
|---|---|
| `vendor/gemma4-intel-serve` | sliding-window ring KV caches, Q8_0 tier, GGUF+mmproj ingest, MTP/draft-model serving, grammar-forced tool names, three-protocol server, `--trace-dir`, PIL-exact image prep |
| `vendor/qwen35-intel-serve` | the oracle skeleton (`simd.hpp`, `fmath.hpp`, `rounding.hpp`, `bf16exec.hpp`), the instrumented HF reference harness (`py/ref_forward.py`), allocation planning and seal modes, ViT window attention + 2-D RoPE + pixel-shuffle merge, video preprocessing |

Muse Glimmer's text stack is closer to gemma4 (Gemma-style sandwich norms,
sliding/global mix, softcapped logits) and its vision tower is closer to qwen35
(window attention, `cu_seqlens`, 2-D RoPE, pixel shuffle). Neither one's
speculative-decoding code transfers directly — see Phase 5.

Architecture facts, tensor names, and the verified traps are in
[ARCHITECTURE.md](../ARCHITECTURE.md). This document is only the order of work.

---

## Corrections to the original framing

Two assumptions in the task as stated do not survive contact with the
checkpoint, and they change Phase 5 materially:

1. **Muse Glimmer does have a separate drafter model.**
   `meta-models/Muse-Glimmer-30B-assistant` (5 layers, ~2.6 B) is published
   alongside the base model, exactly like gemma4's `-assistant` repos.
2. **There is no in-checkpoint MTP layer.** The base checkpoint's 1436 tensors
   contain no `mtp.*` group — unlike Qwen3.6, which carries its NextN head
   inside the main file.

The drafter is also not EAGLE: it is **DFlash block diffusion**, proposing
`block_size = 16` tokens per forward pass from the target's hidden states at
layers {1, 13, 25, 37, 49}, with bidirectional attention inside the block. So
"MTP support" for this model means *block-drafter support*, which is a different
verification story (a whole block accepted/rejected per round, not a chain) and a
different engine shape (one drafter call per 16 candidate tokens, and a target
verification pass over 16 rows). Phase 5 is written against DFlash.

One more scope correction: **Muse Glimmer has no audio tower.** The config has
no `audio_config`, there are no audio tensors, and the model card states
"Input: text + image, Output: text". The gemma4 audio path has nothing to port
to. `input_audio` gets a capability error, the way qwen35 already does it. Video
*is* supported (there is a `video_token_id`, a `MuseGlimmerVideoProcessor`, and
the reference routes video through `get_image_features`), so the multimodal
scope is image + video.

---

## Phase 0 — Repo skeleton and reference environment

Cloud-friendly; no GPU needed.

- `CMakeLists.txt` with two isolated build trees, following qwen35: `build/`
  (plain g++, oracle + tests, no oneAPI) and `build-gpu/` (icpx, AOT for
  `bmg-g31`). `build.sh --cpu-only` must work on a machine with no oneAPI.
- Port verbatim from `vendor/qwen35-intel-serve/src`: `simd.hpp`, `fmath.hpp`,
  `rounding.hpp`, `json.{h,cpp}`, `json_grammar.h`, `json_schema.h`,
  `safetensors.{h,cpp,hpp}`, `sha256.*`, `hf_resolver.*`, `tensor.*`,
  `tokenizer.*`. These are model-independent and already gated in that repo.
- Reference environment: a venv with a `transformers` new enough to expose
  `MuseGlimmerForConditionalGeneration` and `MuseGlimmerAssistantModel`
  (checkpoint says `5.15.0.dev0`), NumPy, Pillow, FastAPI/Uvicorn, the HF
  client. Pin and record the exact version and file path in `VERIFICATION.md`
  the way both siblings do — the oracle contract is only meaningful relative to
  a pinned reference.
- `py/ref_forward.py`: fork of qwen35's instrumented reference, retargeted at
  Muse Glimmer. It must support `--pure` (lift every stock f32 cast to f64) and
  `--dtype bf16|f16` (round at exactly the points a stock low-precision run
  materializes), and dump per-layer hidden states.
- `py/encode_prompt.py`, `py/diff_logits.py`, `py/diff_lp.py` port unchanged.

**Exit gate**: `ref_forward.py --pure` runs the real 30B checkpoint on CPU and
dumps logits + per-layer hiddens for a fixed token id list.

## Phase 1 — Tiny-model harness

Do not debug a 30 B model. qwen35's `py/make_tiny.py` builds randomly-initialized
checkpoints with the real architecture at toy dimensions; the oracle reaches
bitwise agreement there long before the real weights are downloaded.

Build three tiny configs:

- `tiny_text` — 4 layers `[sliding, sliding, sliding, full]`, H=64, D=16, 2 q
  heads / 1 kv head, I=128, V=512, window 8. Exercises the sandwich norms, the
  two eps values, weight-less QK-norm + `qk_scale_factor`, the attention output
  gate, NoPE-on-global, and the softcapped scaled head.
- `tiny_vision` — 4-layer tower, hidden 32, 2 heads, `[window, window, window,
  full]`, pos grid 4×4, patch 14, merge 2, plus adapter and projection.
- `tiny_dflash` — 2 drafter layers over the tiny text model, block_size 4,
  target_layer_ids [0, 2].

**Exit gate**: `oracle` matches `ref_forward.py --pure` to 0 differing bytes on
all three tiny models, and `--dtype bf16` matches `ref_forward.py --pure --dtype
bf16` bitwise.

## Phase 2 — f64 text oracle

This is the core deliverable and the thing the whole stack is refereed against.
Implement §"Text ops" of ARCHITECTURE.md in `src/muse_glimmer.hpp` with fixed
reduction orders, `-ffp-contract=off`, and identical scalar / AVX-512 operation
sequences.

Order of implementation, each with its own golden:

1. Safetensors loader + config parse (`layer_types`, `layer_rope_theta`, the two
   eps values, `qk_scale_factor`, `output_multiplier`, `final_logit_softcapping`).
2. Weight-less RMSNorm, centered RMSNorm, plain RMSNorm — three call sites, and
   the single most likely place to be silently wrong. Assert at load time that
   the final `norm` is read as plain and the four per-layer norms as centered.
3. Embedding + embed-norm. Keep the raw embedding reachable for DFlash.
4. RoPE tables (θ=500000, full 128-dim rotation, `rotate_half`), computed in f64.
5. Attention: QK-norm → `q *= 3.87` → optional RoPE → `scores · 128^-0.5` →
   masked softmax → GQA gather (16:1) → `⊙ sigmoid(gate_proj(x))` → `o_proj`.
   Sliding mask and full mask as separate paths from day one.
6. SwiGLU MLP and the sandwich residual structure.
7. Final head: plain norm → `lm_head` → `· output_multiplier` → `20·tanh(z/20)`.

CLI mirrors the siblings: `--model`, `--ids`, `--out`, `--dump-hidden`,
`--threads`, `--kernels scalar|avx512`, `--attn eager|flash`, `--dtype
f64|bf16|f16`.

**Exit gates**
- Bitwise vs `ref_forward.py --pure` on tiny (already from Phase 1) and
  layer-by-layer agreement on the real 30 B checkpoint at the documented f64
  noise floor (target: max abs logit delta ~1e-13, exact argmax and exact top-64
  at every position — the bar both siblings hold).
- `--kernels scalar` and the AVX-512 path bitwise identical.
- Thread-count invariance: 1 vs N threads bitwise identical.
- Measured and published gap to *stock* unpatched HF-f64.

## Phase 3 — bf16/f16 twin

The referee target for every GPU kernel. Rules are in ARCHITECTURE.md
§"Numerics policy"; the implementation is a port of qwen35's `bf16exec.hpp` +
`rounding.hpp` retargeted at this op list. Watch the Muse-Glimmer-specific
constants: `qk_scale_factor`, `output_multiplier`, and `final_logit_softcapping`
are Python floats and take their f32 opmath value.

**Exit gate**: bitwise vs `ref_forward.py --pure --dtype bf16` (and `f16`) on all
tiny models and on the real checkpoint's first N layers; `--attn flash` twin
defined (rounds only the attention output, skipping the S and P materializations)
so fused GPU kernels have a legal target.

## Phase 4 — GGUF ingest

The official `meta-models/Muse-Glimmer-30B-GGUF` repo ships three GGUF families,
and they are the fastest route to a Q8-class serving tier:

| file | `general.architecture` | notes |
|---|---|---|
| `Muse-Glimmer-30B-KQuant-*.gguf` | `muse-glimmer` | 731 tensors, `blk.*` naming, embedded chat template and tokenizer |
| `mmproj-Muse-Glimmer-30B-*.gguf` | `clip` (`clip.projector_type: muse-glimmer`) | 809 tensors, `v.blk.*` + `mm.*` |
| `dflash-Muse-Glimmer-30B-*.gguf` | `dflash` | 58 tensors, `dflash.block_size`, `dflash.target_layers` |

Port gemma4's `src/gguf.{h,cpp}` and `gguf_weights.{h,cpp}` and add the
`muse-glimmer` mapping. Two conversions are **mandatory** and are the whole point
of gating this phase against the oracle:

- **Subtract 1** from `blk.*.{attn_norm, post_attention_norm, ffn_norm,
  post_ffw_norm}` to recover the zero-centered weights. Do **not** touch
  `output_norm` — verified identical to safetensors.
- Recognize `blk.*.attn_q_norm` = constant 3.87 and `blk.*.attn_k_norm` =
  constant 1.0 as the GGUF spelling of weight-less QK-norm + `qk_scale_factor`.
  Assert the constancy at load rather than assuming it; a checkpoint that ever
  ships a genuinely non-constant q_norm should fail loudly, not silently.
- `post_norm_eps = 1e-8` is absent from GGUF metadata — supply it from a
  built-in per-architecture default, and record that llama.cpp does the same.

Also read `tokenizer.chat_template`, `tokenizer.ggml.*`, and
`tokenizer.ggml.mask_token_id` (present in the dflash GGUF) straight out of the
file, as gemma4 does, so a GGUF-only deployment needs no HF snapshot.

**Exit gate**: an fp16/Q8_0 GGUF and the safetensors snapshot produce logits from
the oracle that agree to the quantization band; the Q4_K tiers land in the same
band llama.cpp's own Q4_K does, measured against the f64 oracle
(gemma4's `docs/q8_vs_llamacpp.md` is the template).

## Phase 5 — DFlash drafter in the oracle

Before any GPU work, the drafter has to be pinned down in f64, because it is the
piece with no prior art in either sibling repo.

- Implement `MuseGlimmerAssistantModel` per ARCHITECTURE.md §"DFlash": context
  projection `fc [33280 → 6656]` + `output_norm_enc`, 5 sliding pre-norm layers
  with q/k norm and GQA 32/8, bidirectional intra-block mask, causal to context,
  final `norm`, then the **target's** `lm_head` + `output_multiplier` + softcap.
- Resolve the open item first: **how many denoising iterations per block, and
  what the acceptance rule is.** `MuseGlimmerAssistantModel.forward` is one pass;
  the loop lives in the generation utility. Read it out of the reference
  generation code and cross-check against llama.cpp's `dflash` implementation
  before writing the gate. Everything downstream depends on getting this exactly
  right, and getting it wrong produces plausible-looking but wrong acceptance
  rates rather than an obvious failure.
- Oracle CLI: `oracle draft --assistant <repo> --block 16`, dumping per-block
  draft tokens *and* per-position logits (draft tokens are discrete argmax
  decisions; near-ties diverge between implementations, so gates compare both,
  the way qwen35's MTP gate does).

**Exit gate**: block-level token agreement and per-position logit agreement with
the reference drafter on tiny and real checkpoints; a measured acceptance rate on
a fixed prompt set to serve as the regression baseline.

## Phase 6 — Vision tower in the oracle

- PIL-exact preprocessing in C++: `resample: 1` (LANCZOS), rescale `1/255`,
  normalize `mean = std = 0.5`, patch 14 with temporal pairing 2, merge 2,
  `max_image_tokens = 4096`. gemma4 already has a byte-exact PIL bicubic port
  (`src/imageprep.cpp`); this needs the LANCZOS kernel added and the
  smart-resize grid logic from qwen35's Qwen-style processor.
- Tower: learned 32×32 position grid resampled with `grid_sample(bilinear,
  align_corners=False, padding_mode=zeros)`; window reorder with `window_size =
  32·14 = 448`; **2-D RoPE with `(w, h)` flipped position ids offset by +1** and
  frequencies laid out `cat[freq_w, freq_h, freq_w, freq_h]` over
  `spatial_dim = 48`; window vs full attention by `layer_types`; reverse the
  window permutation; `ln_post`; pixel shuffle to `[N/4, 6144]`.
- Projector: `gelu(fc2(gelu(fc1(x))))` → `vision_projection` → weight-less
  RMSNorm, then scatter into `inputs_embeds` at token 200092 / 200091.
- Video: same path with `t > 1`; per-frame offsets in the pixel-shuffle index;
  2 fps sampling, `num_frames 96`, `max_video_frame_tokens 144`.

**Exit gate**: `tools/vision_prep_parity.py` shows byte-exact preprocessing vs
the checkpoint processor on a fixed image set; tower + projector outputs match
the f64 reference; end-to-end image-prompt logits match.

## Phase 7 — SYCL engine, single GPU

Now the oracle exists and everything below is refereed against it. Port
gemma4's `src/gpu/` backend abstraction, then build kernels bottom-up, each gated
bitwise against the bf16 twin where the twin is bitwise and envelope-gated at the
logit level elsewhere.

Kernel inventory, in dependency order:

1. RMSNorm ×3 flavors, SwiGLU, RoPE, sigmoid gate, softcapped tail.
2. GEMV / GEMM (custom SYCL for the latency-bound decode path; oneDNN/XMX for
   prefill, following both siblings' "oneDNN where it wins" split).
3. Attention: sliding-window decode, global decode, flash prefill. The GQA ratio
   here is 16:1 with only **2 KV heads** and `head_dim` 128 — the KV cache is
   tiny (2 × 128 × 2 bytes = 512 B per layer per token for k, same for v), which
   makes long context unusually cheap and makes `k = v` dedup irrelevant. Expect
   the decode bottleneck to be weight bandwidth, not attention.
4. Ring KV cache for the 39 sliding layers (holds `sliding_window + chunk` rows
   regardless of `--max-seq`), linear cache for the 13 global layers. This is
   gemma4's design and it maps 1:1.
5. Chunked prefill with a bounded activation scratch.
6. Q8_0 weight tier with activation-quantized MMVQ decode.

Memory sketch, BF16, per B70 (31.9 GiB): weights are ~59.6 GB for the full
model, so **BF16 does not fit on two cards** — Q8 (~30 GB) is the two-card
configuration and Q4_K (~17–20 GB) is the one-card configuration. This is the
opposite balance from gemma4's 31B, and it means the Q8/Q4 tiers are not an
optimization, they are the product. Plan the allocation planner accordingly and
prewarm-then-seal as both siblings do.

**Exit gates**: per-kernel bitwise/envelope gates vs the twin; decode
bit-reproducible across reruns and thread counts; `run_gpu_gates.sh` green.

## Phase 8 — Dual-GPU tensor parallelism

Head/vocab/FFN split with deterministic device-to-device all-reduce, per
gemma4's design. Muse Glimmer specifics:

- 32 q heads over 2 cards splits cleanly (16 each); **2 KV heads** also split
  1 each, so the KV cache shards without replication.
- `nq·D = 4096` vs `H = 6656`: the attention block is narrower than the
  residual, so the o_proj all-reduce moves 6656 floats per token while the
  q/k/v/gate split is over 4096 — sharding boundaries differ between the two
  halves of the block and the planner must not assume they match.
- `I = 19968` splits 9984 per card.
- `V = 202048` splits 101024 per card for the vocab-parallel head; the softcap
  and multiplier are elementwise and commute with the split.
- The Arc driver pitfall gemma4 documents is load-bearing: kernels must never
  dereference peer-GPU USM (silent zeros / `DEVICE_LOST`); all inter-GPU traffic
  is copy-engine D2D, self-checked at startup.

**Exit gate**: 1-GPU and 2-GPU decode bit-identical on the same prompt; measured
scaling published per model tier.

## Phase 9 — Serving frontend

Port `serve/` from gemma4 (it is the more complete of the two: Responses API,
grammar-forced tool names, trace dumps) and retarget the model-specific parts.

- **Chat templating**: use the checkpoint's own `chat_template.jinja` via HF
  `apply_chat_template` so rendering is byte-exact by construction. Deserialize
  `tool_calls[].function.arguments` at the wire boundary — the template
  `raise_exception`s on a JSON string, so every agentic second turn 400s
  otherwise. Repair tool schemas missing `type` the way gemma4 does.
- **Parsing**: drive the output parser from `tokenizer_config.response_template`
  rather than hand-written regexes — it defines the open/close patterns for
  `content` (`to=user<|message|>` … `<|eot|>`/`<|eom|>`), `reasoning_content`
  (`to=self<|message|>` … `<|eom|>`), and `tool_calls` (the `<atem:invoke>` /
  `<atem:parameter>` XML-inline form, `allow_non_json` values).
- **Reasoning**: `Reasoning strength: low|medium|high|xhigh` in the system block,
  default `high`. Surface as `reasoning_content` (chat), `thinking` blocks
  (Anthropic), `reasoning` items (Responses); never leak into `content`.
  `--no-thinking` sets `low` rather than removing the line — the template always
  emits one.
- **Tool-call forcing**: the generation prompt is bare `<|start|>assistant` and
  the model emits ` to=<recipient>` itself. So `tool_choice` maps onto a
  recipient-token grammar: `required` clamps the recipient to the declared tool
  set, a forced choice clamps it to one name, `none` clamps to `user`. This is
  cleaner than gemma4's `<|tool_call>`-opener FSM and should be a hard guarantee,
  not a prompt nudge. Argument bodies stay free-form (or PDA-masked under
  `json_schema`).
- **Guided JSON**: `json_object` and `json_schema` via the byte-level PDA already
  in `src/json_schema.h` / `json_grammar.h`. Mask cost scales with vocab; at
  202048 expect ~0.3 ms/step, in line with gemma4's measured 0.39 ms at 262 k.
- **Sampling**: full-vocab temperature / top-k / top-p / min-p / penalties /
  seed / logit bias / logprobs / stop strings. Defaults from
  `generation_config.json` (1.0 / 0.95 / 64). Stop on both eos ids
  (200001 `<|end_of_text|>`, 200008 `<|eot|>`).
- **Media**: `image_url` and `video_url` content parts, `data:`/http/https, 64 MiB
  cap, content-addressed decode cache. **`input_audio` returns a capability
  error** — the model has no audio tower.
- **Endpoints**: `/v1/chat/completions`, `/v1/completions`, `/v1/responses`,
  `/v1/messages`, `/v1/messages/count_tokens`, `/v1/models`, `/health`,
  `/metrics`; SSE on all three chat protocols; `--api-key`, `--cors-origins`,
  cancel-on-disconnect, `--trace-dir`.
- **Cross-turn prefix reuse**: rolls the KV cache back to the longest common
  prefix, bounded on sliding layers by `window + chunk` — same caveat gemma4
  documents, and worth surfacing in `--chunk` guidance for agent workloads.

**Exit gates**: `tests/serve_tests.py` (no GPU) green; `tests/live_api_tests.py
--spawn` green on a card; serving greedy decode token-identical to the native
engine on the same rendered prompt.

## Phase 10 — Speculative serving with DFlash

The engine work the block drafter needs, once Phase 5 has pinned the semantics:

- Drafter resident alongside the target (separate 5-layer model, own sliding KV
  cache) plus a **hidden-state tap** on target layers {1, 13, 25, 37, 49} — a
  requirement neither sibling engine has, since EAGLE taps one layer and Qwen's
  MTP taps only the pre-final-norm hidden. The tap must survive tensor-parallel
  sharding: those hiddens are the full 6656-wide residual, so they need a gather
  across cards before the drafter's `fc`.
- One drafter call proposes 16 tokens; the target verifies all 16 rows in one
  batched forward; rejection rolls back to the first mismatch and replays.
  Distribution-preserving acceptance (llama.cpp-style target sampling) so greedy
  output stays bit-identical to plain greedy decode — the property both siblings
  guarantee and gate (`tools/spec_parity.py`).
- Report `spec_accept_rate` / `spec_accepted` / `spec_drafted` / `spec_rounds`
  in `usage` and `/metrics`.

**Exit gate**: `spec_parity.py` shows greedy spec output bit-identical to plain
greedy; measured acceptance and tok/s published per tier.

## Phase 11 — Benchmarks and comparison

`tools/bench.py` on the same split-final-token prefill / logits download / host
sampling / decode path as serving; `tools/longctx.py` for depth curves and
needle retrieval. Head-to-head vs llama.cpp's SYCL backend on identical token
streams, per `vendor/qwen35-intel-serve/docs/comparison.md`. Publish per
model tier, GPU count, quantization, and depth.

---

## What can be done in the cloud, and what needs the box

| phase | cloud | needs the B70 box |
|---|---|---|
| 0 Skeleton, reference env | ✅ | |
| 1 Tiny harness | ✅ | |
| 2 f64 text oracle | ✅ (CPU only; 30 B f64 forward is slow but runnable) | |
| 3 bf16/f16 twin | ✅ | |
| 4 GGUF ingest | ✅ | |
| 5 DFlash oracle | ✅ | |
| 6 Vision oracle | ✅ | |
| 7 SYCL single-GPU | design only | ✅ |
| 8 Dual-GPU TP | design only | ✅ |
| 9 Serving frontend | ✅ except live gates | ✅ for `live_api_tests.py` |
| 10 DFlash serving | design only | ✅ |
| 11 Benchmarks | | ✅ |

Phases 0–6 and most of 9 are pure CPU work and are the natural first session.
Everything from Phase 7 on needs the hardware, but its *design* is fixed by the
oracle that Phases 2–6 produce — which is the whole reason the oracle comes
first.

## Risk register

| risk | why it bites | mitigation |
|---|---|---|
| DFlash denoising loop semantics unknown | acceptance rate is plausible but wrong; no loud failure | resolve against the reference generation utility **and** llama.cpp before writing the gate (Phase 5, first task) |
| Norm convention mix-up (centered vs plain, two eps) | logits drift subtly; passes smoke tests | load-time assertions + a dedicated tiny-model gate per norm site (Phase 1) |
| GGUF `-1` on the wrong tensor set | GGUF and safetensors paths silently disagree | byte-probe assertion at load: GGUF `output_norm` must equal safetensors `norm`; the four layer norms must differ by exactly 1.0 |
| Vision RoPE `(w, h)` flip and `+1` offset | every vision logit wrong, image captions merely "worse" | fixture test against reference `position_ids` before the tower is written |
| BF16 does not fit two B70s | a BF16-first plan strands at Phase 7 | Q8 is the two-card target and Q4_K the one-card target from the start; BF16 stays an oracle/twin concept |
| `transformers` version drift | the oracle contract is version-relative | pin the exact version and record the modeling-file hash in `VERIFICATION.md` |

## Deliverable docs (mirroring the siblings)

- `ARCHITECTURE.md` — the contract. **Written.**
- `docs/plan.md` — this file.
- `docs/oracle.md` — how to build and run the f64 oracle, once it exists.
- `VERIFICATION.md` — methodology, per-layer error tables, measured gates.
- `docs/gpu.md` — engine design, memory model, hardware pitfalls, kernel tracker.
- `docs/serving.md` — protocols, tools, JSON, media.
- `docs/comparison.md` — head-to-head vs llama.cpp and the sibling engines.
