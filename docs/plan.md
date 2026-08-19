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
6. Q8_0 weight tier, quantized at load from the same BF16 checkpoint the oracle
   binds (see "The Q8 tier and MMVQ" below).

Allocation planning and prewarm-then-seal follow both siblings: prewarm the
widest prefill and deepest decode at startup, then arm allocation seal mode 2 so
memory failure is a startup event, not a mid-request event.

**Exit gates**: per-kernel bitwise/envelope gates vs the twin; decode
bit-reproducible across reruns and thread counts; `run_gpu_gates.sh` green.

### Memory budget and weight tiers

Two Arc Pro B70s = 63.8 GiB. Target 29.777 B params, drafter 2.556 B (both
counts exact, from the checkpoint indices).

| tier | target | /card | drafter | free/card after both |
|---|---:|---:|---:|---:|
| BF16 | 55.46 GiB | 27.73 | 4.76 GiB | 1.79 |
| Q8_0 | 29.46 GiB | 14.73 | 2.53 GiB | 15.90 |
| Q4_K_M | 16.81 GiB | 8.41 | 1.44 GiB | 22.77 |

**BF16 fits two cards.** 27.73 GiB/card leaves 4.17 GiB/card before caches. The
KV cache is unusually cheap here: 2 KV heads × head_dim 128 in bf16 is
**1 KiB per layer per token**, and only the 13 global layers grow — the 39
sliding layers hold a fixed `sliding_window + chunk` ring.

| context | global KV | sliding rings | per card |
|---|---:|---:|---:|
| 32 768 | 0.406 GiB | 0.114 GiB | 0.26 |
| 65 536 | 0.812 GiB | 0.114 GiB | 0.46 |
| 131 072 (full window) | 1.625 GiB | 0.114 GiB | 0.87 |

So BF16 at the **full 131 K window** still leaves ~3.3 GiB/card. The binding
constraint on BF16 is not the cache — it is prefill chunk scratch, the vision
tower's patch reserve, oneDNN workspaces, and (decisively) a resident drafter:
a BF16 drafter costs 2.38 GiB/card and drops the margin to ~0.9 GiB, which is
not enough. Meta ships `dflash-Muse-Glimmer-30B-Q4_K_M.gguf` and
`dflash-kquant.gguf` for exactly this reason, so **BF16 target + quantized
drafter** (0.72 GiB/card) is the configuration to plan for.

Q8 target on **one card** is 29.46 GiB against 31.9 — 2.44 GiB free, viable at
modest context, the same shape as qwen35's one-card 27B Q8 at 16 K. Q4_K_M on
one card leaves 15 GiB and is the obvious consumer-hardware tier.

Planned defaults, to be replaced by measured prewarm depths:

| config | cards | expected context |
|---|---:|---|
| BF16 plain | 2 | full 131 072 (validate by prewarm) |
| BF16 + Q4 drafter | 2 | 131 072, tighter scratch |
| Q8 plain | 2 | 131 072, comfortable |
| Q8 plain | 1 | ~16 384–32 768 |
| Q4_K_M | 1 | 131 072 |

Every row above is paper arithmetic. gemma4's hard-won lesson applies verbatim:
**`max ctx` is an allocation ceiling, not a usable-prefill one** — the number
that counts is the deepest position that prewarms *and* decodes, and that has to
be measured on the box.

### The Q8 tier and MMVQ

MMVQ ("mat-vec quantized") is the decode-path GEMV that keeps the activation
quantized instead of dequantizing the weight to float first: the activation
vector is quantized once per 32-element block, and the block dot runs
`int8 × int8 → int32` (the dp4a idiom, pattern-matched by IGC on Arc). It is
not specific to Q8 — llama.cpp uses the same scheme for every quantized weight
format, Q4_K included; Q8_0 is just the easiest case because the weight blocks
are already int8 + f16 scale.

Both siblings implement it, and they reached **opposite defaults** — this is the
single most useful thing to inherit here:

- qwen35 runs MMVQ for Q8 decode by default.
- gemma4 measured it and turned it **off** (`docs/gpu.md`, 2026-07-16). Variant
  A — dequantize per-element in-register into the standard lane-strided f32-fma
  structure — beat MMVQ by 2–5% on every model at 12718 depth (E2B 89.8 vs 85.7,
  31B 17.7 vs 17.5). The reasoning: decode is **weight-bandwidth-bound**, both
  variants stream identical weight bytes, and MMVQ adds an activation-quant
  launch per GEMV while its integer dots buy nothing at these shapes.

Muse Glimmer's decode shapes are dense and wide (no MoE, `H = 6656`,
`I = 19968`), which is exactly the regime where gemma4's negative result was
measured. So: **implement variant A first, keep MMVQ behind
`ORACLE_GPU_MMVQ=1`, and re-measure rather than assume.** If MMVQ ever wins it
will be on the Q4_K tier, where the bits-per-weight is lower and the dequant
cost per streamed byte is higher — that is the one configuration neither sibling
tested, and it is worth a bench line of its own.

Q8 is a **separate accuracy tier, not the bf16 twin band**: quantize at load
with `quantize_row_q8_0_ref` semantics so the weights are bit-identical to a
llama.cpp Q8_0 GGUF, then gate against llama.cpp-Q8_0 and against the f64
oracle, per gemma4's `docs/q8_vs_llamacpp.md`.

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
- **The drafter must support its own weight tier, independent of the target's.**
  A BF16 target plus a BF16 drafter leaves ~0.9 GiB/card, which is not enough
  (see the memory budget). Meta publishes `dflash-Muse-Glimmer-30B-Q4_K_M.gguf`
  and `dflash-kquant.gguf` precisely for this; the loader should accept a
  quantized drafter against a BF16 target, and the drafter's accuracy tier is
  gated separately (against the f64 drafter oracle from Phase 5) since a lossier
  drafter costs acceptance rate, not correctness.
- One drafter call proposes 16 tokens; the target verifies all 16 rows in one
  batched forward; rejection rolls back to the first mismatch and replays.
  Distribution-preserving acceptance (llama.cpp-style target sampling) so greedy
  output stays bit-identical to plain greedy decode — the property both siblings
  guarantee and gate (`tools/spec_parity.py`).
- Report `spec_accept_rate` / `spec_accepted` / `spec_drafted` / `spec_rounds`
  in `usage` and `/metrics`.

**Exit gate**: `spec_parity.py` shows greedy spec output bit-identical to plain
greedy; measured acceptance and tok/s published per tier.

### Write the block-drafter engine model-generically

DFlash is a *trained* companion network, not an architecture trick — there is no
DFlash drafter for Gemma 4 or Qwen3.6 today, and one cannot be derived from
their weights; it would have to be trained against each target. But nothing in
the **serving machinery** is Muse-Glimmer-specific: a multi-layer hidden-state
tap, a context projection, a bidirectional block of `B` mask tokens, one target
verification pass over `B` rows, and block-level rollback are all generic. The
model-specific parts are just `block_size`, `target_layer_ids`, `mask_token_id`,
and the drafter's own layer stack.

The siblings would benefit if such drafters ever existed: Gemma 4 uses EAGLE
heads at k=6 and Qwen3.6 an in-checkpoint MTP at k=4, both of which draft a
*chain* one token at a time, where DFlash proposes 16 in a single pass (Meta
measures 3.1× on a 5090 vs 2.77× for gemma4's 31B EAGLE at k=6). So the block
path should be built behind the same draft-backend interface the gemma4 engine
already uses for `--draft-assistant` / `--draft-model`, with the block
parameters read from config rather than hardcoded. That costs nothing here and
makes the work portable if a DFlash head is ever trained for either sibling.
It is explicitly **not** in this repo's scope to train one.

## Phase 11 — Benchmarks and comparison

`tools/bench.py` on the same split-final-token prefill / logits download / host
sampling / decode path as serving; `tools/longctx.py` for depth curves and
needle retrieval. Head-to-head vs llama.cpp's SYCL backend on identical token
streams, per `vendor/qwen35-intel-serve/docs/comparison.md`. Publish per
model tier, GPU count, quantization, and depth.

---

## What can be done in the cloud, and what needs the box

| phase | cloud | needs the B70 box | status |
|---|---|---|---|
| 0 Skeleton, reference env | ✅ | | **done** |
| 1 Tiny harness | ✅ | | **done** |
| 2 f64 text oracle | ✅ (CPU only; the 30 B f64 forward runs in ~17 s for 6 tokens) | | **done** |
| 3 bf16/f16 twin | ✅ | | **done** (text; drafter/tower twins written, not gated) |
| 4 GGUF ingest | ✅ | | not started (the Q8_0 tier quantizes at load from BF16 instead; see [gpu.md](gpu.md)) |
| 5 DFlash oracle | ✅ | | **done** |
| 6 Vision oracle | ✅ | | tower + projector **done** on CPU and GPU (20.8x, see [gpu.md](gpu.md)); pixel ingestion and video not started |
| 7 SYCL single-GPU | design only | ✅ | **done** (text; see [gpu.md](gpu.md), incl. the opt-in `--flash-prefill` matrix-engine attention tier. No Q8 tier yet, no vision/DFlash on GPU) |
| 8 Dual-GPU TP | design only | ✅ | **done** (text; head/FFN/vocab split, KV replicated — see [gpu.md](gpu.md)). 14.81 tok/s decode, 1030 tok/s prefill, full 131 072 context fits |
| 9 Serving frontend | ✅ except live gates | ✅ for `live_api_tests.py` | **done** — three protocols, guided JSON, grammar-forced recipients, media; `serve_tests.py` 41/41 with no GPU, `live_api_tests.py` 30/30 on the box. See [serving.md](serving.md) |
| 10 DFlash serving | design only | ✅ | **done** as an engine loop: `--draft-rounds N`, gated on producing the f64 oracle's sequence. Up to 11 tokens/round and 119.9 tok/s on code. The HTTP serving layer around it is Phase 9. |
| 11 Benchmarks | | ✅ | **done** — head-to-head vs llama.cpp's SYCL backend, both stacks at their best: BF16 ahead on all 12 tests, Q8 level on decode and ahead at depth. Six fixes came out of it. See [comparison.md](comparison.md) |

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
| BF16 + BF16 drafter does not fit two B70s | speculative BF16 serving strands late, after the drafter engine work is done | plan the drafter as a **quantized** resident from the start (Meta ships Q4_K and kquant drafter GGUFs); BF16 target + Q4 drafter is the configuration |
| Paper memory budget vs. measured prewarm depth | `max ctx` is an allocation ceiling, not a usable-prefill one — a config that allocates can still die in the forward | publish only depths that prewarm *and* decode, per gemma4's 31B experience |
| `transformers` version drift | the oracle contract is version-relative | pin the exact version and record the modeling-file hash in `VERIFICATION.md` |

## Handoff — start here

**Status: Phases 0, 1, 2, 3, 5 and the model-function half of 6 are
implemented and gated.** Run `./build.sh --cpu-only && ./run_tiny.sh` — 22
checks, all green — and read [VERIFICATION.md](../VERIFICATION.md) for what each
gate proves and every measured number. Everything below the line is what is
left.

What exists now: `src/muse_glimmer.hpp` (f64 text oracle + bf16/f16 twins),
`src/dflash.hpp` (the block drafter and the greedy speculative loop),
`src/vision.hpp` (tower + adapter + projection), the instrumented references
`py/ref_forward.py` / `py/ref_dflash.py` / `py/ref_vision.py`, the tiny-model
factory `py/make_tiny.py`, and `tools/spec_baseline.sh` for the acceptance-rate
regression baseline.

**Next, in order of value:**

1. **Q8 prefill into an empty cache.** The only shape llama.cpp still wins
   (0.57x at pp512, 0.92x at pp2048, level by pp8192, and ours by 1.15x once
   there is depth): the Q8 tier dequantizes each weight into a BF16 scratch
   once per BLOCK, an extra HBM round trip a fused path does not pay. The cost
   is per block and the benefit is per token, so it vanishes as the block
   fills. Two ways out, neither started: a fused int8→bf16 GEMM feeding DPAS,
   or layer-major prefill — loop layers outside blocks, so a weight is
   dequantized once per prefill rather than once per block.
   See [comparison.md](comparison.md).
2. **Phase 4, GGUF ingest** — nothing about it changed; the three conversions in
   that section are still the whole job, and `assert_norm_flavours()` in
   `src/muse_glimmer.hpp` documents why the +1 check has to be the cross-source
   one rather than a per-tensor statistic.
3. **The rest of Phase 6**: C++ pixel ingestion (see the correction below) and a
   video (`t > 1`) gate. The tower takes `t` already, and the server reaches it
   through the checkpoint's own video processor, but no video gate has been run.
4. Q8 for the vision tower and for the embedding table (~1.2 GiB/card, possibly
   enough to fit the 30B on one card).

Three things that are already known and must not be rediscovered the hard way:

- The four per-layer norms are `(1+w)`, the final norm is `w`. Byte-verified.
- Global layers are NoPE; only the 39 sliding layers rotate.
- QK-norm is weight-less; `qk_scale_factor = 3.87` multiplies **q only** and is
  *in addition to* `head_dim^-0.5`.

### Corrections this plan needed, found while implementing it

- **Phase 10's exit gate as written is not achievable.** "greedy spec output
  bit-identical to plain greedy" cannot hold on a stack whose decode and prefill
  paths are different kernels: a verified token's logits come from a 16-row
  forward (oneDNN matmul, tile-softmax attention) and a plain decode's from a
  1-row forward (hand-written GEMV, split-K attention). Measured, every
  divergence is a near-tie — in each observed case one path saw an EXACT tie
  where the other saw 0.06-0.25 logprob. The gate now checks that property,
  which is the one that can hold. See [serving.md](serving.md).
- **The same request twice can differ**, because the cache is part of the
  input: prefix reuse changes the prefill chunking and the fast tiers are
  envelope-level. `--no-prefix-reuse` makes the answer a function of the prompt
  alone; the gates ask for "same state, same answer".

- **The DFlash open item is closed.** One pass per block, no denoising loop;
  a round proposes `block_size - 1` = **15** tokens, not 16; the head is the
  **bare** `lm_head` with no `output_multiplier` and no softcap; acceptance is
  HF's ordinary assisted-decoding rule. And the block's queries are placed at
  absolute positions `n … n+B-1` by `DFlashCache.get_query_offset()`, not by
  `position_ids` — get that wrong and the drafter runs without erroring and
  proposes different tokens. See ARCHITECTURE.md §"The drafting loop".
- **Preprocessing is not PIL.** `MuseGlimmerImageProcessor` is a
  `TorchvisionBackend`; `resample: 1` runs **torchvision's** antialiased
  LANCZOS, and only on torchvision ≥ 0.27 (below that it silently substitutes
  BICUBIC). gemma4's PIL port is the wrong target. VERIFICATION.md §6c.
- **The norm-flavour guard cannot be a per-tensor statistic.** A zero-centered
  norm in this checkpoint reaches mean +2.09 with no negative entries, while the
  plain final norm has mean +0.017 and 49.9% negatives. The sound check is the
  aggregate (`min(w) ≥ -1` everywhere, negatives present in most of the 208).
  VERIFICATION.md §6.
- **`-march=native` is required for the scalar-vs-vector gate to mean
  anything.** Without it `__AVX512F__` compiles out and the gate compares the
  scalar path with itself. Both sibling repos' CMakeLists have the same gap.
- **torch's f64 `sqrt` is the approximate one, not its division.** `rsqrt` and
  `pow(·, -0.5)` are correctly rounded and equal C's `1.0/std::sqrt`;
  `1/torch.sqrt(x)` is not. VERIFICATION.md §3.

`tools/probe/README.md` lists three checkpoint checks worth re-running if the
checkpoint is ever revised.

## Deliverable docs (mirroring the siblings)

- `ARCHITECTURE.md` — the contract. **Written.**
- `docs/plan.md` — this file.
- `docs/oracle.md` — how to build and run the f64 oracle. **Written.**
- `VERIFICATION.md` — methodology, per-layer error tables, measured gates.
  **Written** for Phases 0–3, 5, 6.
- `docs/gpu.md` — engine design, memory model, hardware pitfalls, kernel tracker.
- `docs/serving.md` — protocols, tools, JSON, media. **Written.**
- `docs/comparison.md` — head-to-head vs llama.cpp. **Written.**
