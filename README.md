# muse-glimmer-intel-serve

An OpenAI- and Anthropic-compatible serving stack for **Muse Glimmer 30B** on
Intel Arc GPUs (SYCL / Level-Zero, 1–2× Arc Pro B70), built oracle-first: a
deterministic float64 CPU implementation defines the model function exactly, a
bf16/f16 twin defines the deviation band any correct low-precision kernel must
stay inside, and every GPU kernel is gated against them.

**Status: planning.** Nothing is implemented yet. This repository currently
contains the verified architecture contract, the phased build plan, and the two
sibling stacks it is derived from.

- [ARCHITECTURE.md](ARCHITECTURE.md) — the exact model specification the oracle
  implements: config, tensor inventory, op-by-op semantics for the text
  backbone, vision tower and DFlash drafter, the serving/chat contract, and the
  numerics policy. Derived from the `transformers` reference and verified
  against the shipped safetensors and GGUF bytes.
- [docs/plan.md](docs/plan.md) — the build plan, phase by phase, with exit gates
  and a cloud-vs-hardware split.

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
(`muse-glimmer` main, `clip` mmproj, `dflash` drafter). BF16 for the full model
is ~59.6 GB and does not fit two B70s — **Q8 is the two-card configuration and
Q4_K the one-card configuration.**

## References

Two prior stacks by the same author are vendored as submodules and are the
design input for this one:

| path | model | what it contributes |
|---|---|---|
| [`vendor/gemma4-intel-serve`](https://github.com/mjsabby/gemma4-intel-serve) | Gemma 4 (E2B…31B) | sliding-window ring KV caches, Q8_0 tier, GGUF + mmproj ingest, three-protocol server, grammar-forced tool names, speculative decoding |
| [`vendor/qwen35-intel-serve`](https://github.com/mjsabby/qwen35-intel-serve) | Qwen3.5 / Qwen3.6 | the f64 oracle skeleton and instrumented HF reference, the bf16/f16 twin, allocation planning, ViT window attention + 2-D RoPE + pixel-shuffle merge, video preprocessing |

```bash
git clone --recurse-submodules https://github.com/mjsabby/muse-glimmer-intel-serve
# or, in an existing clone:
git submodule update --init --depth 1
```

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
