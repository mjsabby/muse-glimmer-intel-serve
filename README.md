# muse-glimmer-intel-serve

An OpenAI- and Anthropic-compatible serving stack for **Muse Glimmer 30B** on
Intel Arc GPUs (SYCL / Level-Zero, 1–2× Arc Pro B70), built oracle-first: a
deterministic float64 CPU implementation defines the model function exactly, a
bf16/f16 twin defines the deviation band any correct low-precision kernel must
stay inside, and every GPU kernel is gated against them.

**Status: it serves, and it has been measured against llama.cpp.** The f64 oracle and its bf16/f16 twins, the DFlash block
drafter and the vision tower (Phases 0–3, 5, most of 6); the SYCL engine on two
Arc Pro B70s with tensor parallelism, a Q8_0 weight tier, int8-DPAS speculative
drafting and a prewarmed-then-sealed static allocation (7, 8, 10); and the
three-protocol HTTP frontend with guided JSON, grammar-forced tool recipients
and image input (9). GGUF ingest and the published benchmark sweep are not
started.

```bash
./build.sh --cpu-only                     # g++ only, no oneAPI needed
.venv/bin/python py/make_tiny.py --out tiny
./run_tiny.sh                             # the bitwise + noise-floor + determinism gates

./build/muse-oracle --model meta-models/Muse-Glimmer-30B \
    --ids 200000,954,7963,323,11698,373 --out out/run --dump-hidden
```

On the released 30B checkpoint the oracle agrees with a precision-lifted HF
reference to **1.1e-13** max abs on logits with **exact argmax and exact top-64**
at every position; with an image in the prompt, **1.3e-11** and still exact
argmax and top-64 at all 46 positions. Its `--dtype bf16`/`f16` twins are
**bitwise** against the rounding-instrumented reference, and on the tiny models
the whole path — text, drafter, and image-to-logits — is bitwise. Full numbers,
methodology and the traps found along the way are in
[VERIFICATION.md](VERIFICATION.md).

| phase | | |
|---|---|---|
| 0 | repo skeleton, pinned reference env, instrumented HF reference | ✅ |
| 1 | tiny-model harness (`tiny_text` / `tiny_vision` / `tiny_dflash`) | ✅ text gate green |
| 2 | f64 text oracle | ✅ |
| 3 | bf16/f16 twin | ✅ |
| 4 | GGUF ingest | ⬜ |
| 5 | DFlash drafter in the oracle | ✅ |
| 6 | vision tower + projector | ✅ · pixel ingestion ⬜ |
| 7–8 | SYCL engine, dual-GPU tensor parallelism | ✅ 28 GPU gates green |
| 9 | serving frontend | ✅ 41 offline + 30 live gates green |
| 10 | DFlash speculative serving | ✅ 15 → 98 tok/s |
| 11 | benchmark sweep vs llama.cpp | ✅ BF16 12/12, Q8 9/12 + 2 ties [comparison.md](docs/comparison.md) |

- [ARCHITECTURE.md](ARCHITECTURE.md) — the exact model specification the oracle
  implements: config, tensor inventory, op-by-op semantics for the text
  backbone, vision tower and DFlash drafter, the serving/chat contract, and the
  numerics policy. Derived from the `transformers` reference and verified
  against the shipped safetensors and GGUF bytes.
- [VERIFICATION.md](VERIFICATION.md) — what each gate compares, the pinned
  reference environment, and every measured number.
- [docs/oracle.md](docs/oracle.md) — how to build and run the oracle, and what
  each flag means.
- [docs/gpu.md](docs/gpu.md) — the SYCL engine: layout decisions, tensor
  parallelism, the attention and Q8 tiers, the memory model, prewarm and the
  allocation seal, and every hardware trap found the hard way.
- [docs/serving.md](docs/serving.md) — the HTTP stack: the channel protocol,
  `tool_choice` as a grammar, guided JSON, what speculative decoding actually
  guarantees, prefix reuse, and media.
- [docs/comparison.md](docs/comparison.md) — head-to-head with llama.cpp's SYCL
  backend on the same two cards, each in its best configuration: BF16 ahead on
  all twelve tests (1.01–1.27×), Q8 ahead on nine and level on decode, and the
  seven fixes the sweep produced — two of which were the same mistake in
  different kernels, reading quantized weights one byte at a time.
- [docs/plan.md](docs/plan.md) — the build plan, phase by phase, with exit gates
  and a cloud-vs-hardware split.

The plan listed one genuinely open question — DFlash's denoising iteration count
and acceptance rule. It is **resolved and implemented**: one pass per block,
**15** proposed tokens per round rather than 16 (the anchor row's logits are
dropped), a **bare** `lm_head` with no `output_multiplier` and no softcap, and
HF's ordinary assisted-decoding acceptance rule. On the released 30B + drafter
the oracle's block matches the reference token for token with per-position
logits agreeing to 3.7e-13. See
[ARCHITECTURE.md](ARCHITECTURE.md#the-drafting-loop--resolved).

```bash
./build/muse-oracle --model meta-models/Muse-Glimmer-30B \
    --assistant meta-models/Muse-Glimmer-30B-assistant \
    --ids tools/prompts/fact.ids --out out/draft --draft-rounds 6
tools/spec_baseline.sh            # the acceptance-rate regression baseline
```

## Serving

```bash
./build.sh                                # oracle + SYCL engine + the .so
source /opt/intel/oneapi/setvars.sh

ZES_ENABLE_SYSMAN=1 .venv/bin/python -m serve.server \
    --model meta-models/Muse-Glimmer-30B \
    --assistant meta-models/Muse-Glimmer-30B-assistant --q8-assistant \
    --gpus 2 --max-seq 8192 --vision cpu --port 8123

curl localhost:8123/v1/chat/completions -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"What is the capital of France?"}]}'
```

OpenAI Chat Completions / Completions / Responses and Anthropic Messages, SSE
on all three chat protocols, ATEM tool calling with grammar-forced recipients,
guided `json_object` / `json_schema`, image and video input, `Reasoning
strength` control, prefix reuse across turns, cancellation, auth, CORS,
`/metrics` and request tracing. Measured on two B70s with a BF16 target and a
Q8 drafter: **17.7 tok/s plain decode, 31–97 tok/s speculative** depending on
how predictable the text is, retrieval verified at the full 131 072-token
window, and a footprint that does not move after startup.
Details and the gate list are in [docs/serving.md](docs/serving.md).

## The model

`meta-models/Muse-Glimmer-30B` — 29.6 B params, dense causal transformer with a
~1.8 B ViT-G/14 perception encoder.

| | |
|---|---|
| layers / hidden / FFN | 52 / 6656 / 19968 (SwiGLU) |
| attention | 32 q heads / 2 kv heads (GQA 16:1), head_dim 128, gated |
| layer pattern | `[local, local, local, global]` ×13, sliding window 2048 |
| position encoding | RoPE θ=500000 on **local layers only** — global layers are NoPE |
| norms | Gemma-style sandwich, zero-centered `(1+w)`; plain `w` on the final norm |
| head | `20·tanh(0.196116·z / 20)` — pre-scaled, softcapped |
| vocab / context | 202048 / 131072 |
| modalities | text + image + video in, text out (**no audio tower**) |
| drafter | `-assistant`: DFlash block diffusion, 5 layers, 16-token blocks |

Weights load from Hugging Face safetensors or from the official GGUFs
(`muse-glimmer` main, `clip` mmproj, `dflash` drafter). BF16 weights are
55.5 GiB and **do** fit two B70s (27.7 GiB/card, 4.2 GiB/card free); with only
2 KV heads the cache is 13 KiB/token, so even the full 131 K window is under
0.9 GiB/card. Q8 is the comfortable tier and the one that leaves room for a
resident drafter. See [the memory budget](docs/plan.md#memory-budget-and-weight-tiers).

## References

Two prior stacks by the same author were the design input for this one. What
they contributed has been ported in-tree, so they are no longer vendored —
consult them upstream:

| repo | model | what it contributed |
|---|---|---|
| [`gemma4-intel-serve`](https://github.com/mjsabby/gemma4-intel-serve) | Gemma 4 (E2B…31B) | sliding-window ring KV caches, Q8_0 tier, GGUF + mmproj ingest, three-protocol server, grammar-forced tool names, speculative decoding |
| [`qwen35-intel-serve`](https://github.com/mjsabby/qwen35-intel-serve) | Qwen3.5 / Qwen3.6 | the f64 oracle skeleton and instrumented HF reference, the bf16/f16 twin, allocation planning, ViT window attention + 2-D RoPE + pixel-shuffle merge, video preprocessing |

## Planned scope

Feature parity with the two siblings, adjusted for what this checkpoint
actually ships:

- f64 CPU oracle over safetensors **and** GGUF, with deterministic bf16/f16 twins
- dual-GPU tensor parallelism (heads, KV, FFN, vocab) with deterministic D2D
  reductions
- Q8_0 and Q4_K weight tiers; ring caches for the 39 sliding layers, linear for
  the 13 global
- vision + video: PE-style tower, window attention, 2-D RoPE, pixel-shuffle
  merge, byte-exact preprocessing
- DFlash block-drafter speculative decoding, distribution-preserving
- OpenAI Chat Completions / Completions / Responses and Anthropic Messages, SSE
  streaming, cancellation, auth, CORS, metrics, request tracing
- ATEM tool calling with grammar-forced recipients, guided `json_object` /
  `json_schema` via a byte-level PDA, and `Reasoning strength` control

Two scope notes carried over from the plan: there is **no in-checkpoint MTP
layer** (speculation is the separate DFlash drafter), and there is **no audio
path** (`input_audio` returns a capability error). Both are explained in
[docs/plan.md](docs/plan.md#corrections-to-the-original-framing).
