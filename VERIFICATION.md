# Verification — methodology, pinned reference, measured gates

The oracle's contract is *relative to a pinned reference*. This file records
which reference, what "agreement" means for each gate, and every number
measured so far. Everything below is reproducible with `./run_tiny.sh` and the
commands quoted in each section.

Status: Phases 0–3 of [docs/plan.md](docs/plan.md) are complete (repo skeleton,
tiny harness, f64 text oracle, bf16/f16 twin). Phases 4–11 are not started; the
sections for their gates are marked *not yet measured*.

---

## 1. Pinned reference environment

The oracle is refereed against `transformers`' own modules, executed with the
precision patches described in §3. Both the version and the modeling-file
content are pinned, because a change to either changes what "correct" means.

| | |
|---|---|
| python | 3.14.4 |
| torch | 2.13.0+cpu |
| transformers | **5.15.0** (checkpoint declares `transformers_version` `5.15.0.dev0`) |
| numpy | 2.5.2 |
| host | AMD Ryzen 9 9950X (16C/32T), AVX-512 incl. `avx512_bf16` / `avx512_vnni` |
| compiler | g++ (Ubuntu 15.2.0-16ubuntu1) 15.2.0, `-O3 -ffp-contract=off -fno-math-errno -march=native` |

Reference environment: `.venv` in the repo root (git-ignored), created with
`python3 -m venv .venv` and

```bash
.venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch
.venv/bin/pip install transformers==5.15.0 numpy pillow safetensors huggingface_hub \
                      fastapi uvicorn sse-starlette jinja2 tokenizers requests
```

SHA-256 of the modeling files the oracle is written against
(`.venv/lib/python3.14/site-packages/transformers/`):

| file | sha256 |
|---|---|
| `models/muse_glimmer/modeling_muse_glimmer.py` | `fd285c1dc9f4dd996e875250c2b76016cbc359aa85d55aea2102527c88fa7050` |
| `models/muse_glimmer/configuration_muse_glimmer.py` | `67900ce9164c509c9067687b1ed60e37270510d028c848ff2397ab92edfb3c16` |
| `models/muse_glimmer/image_processing_muse_glimmer.py` | `827b93a32e8a4b3c80c35395cc261f798a3af3b17665116b0b30d40906a960a9` |
| `models/muse_glimmer/video_processing_muse_glimmer.py` | `3423198b4db57e95761d7417e50e29e2f3318bc0bdd40e4bf419d6ae68c0b86c` |
| `models/muse_glimmer_assistant/modeling_muse_glimmer_assistant.py` | `fc4f276d8e473ccea3c826bba7da8c84cbb6d557e3f6bbb2b4c2fed01b126dc2` |
| `models/muse_glimmer_assistant/configuration_muse_glimmer_assistant.py` | `4898e4b1eef8b8be5201c6daf7b16a9c07afc8f1d6cefec4cb0c272689caec2b` |
| `generation/candidate_generator.py` (the DFlash drafting loop) | `cf3ba7f85407bd24f5a6016135809137d57492fca9d4373f7813242c1b7b75c2` |

Checkpoints:

| repo | revision |
|---|---|
| `meta-models/Muse-Glimmer-30B` | `a4e59da52a7bc87ae7251dd5545c0dd437c44b68` |
| `meta-models/Muse-Glimmer-30B-assistant` | `e8192f3a8f617f74be2ce220360c89ef4789f39f` |

---

## 2. What each gate compares

Four distinct claims, deliberately kept apart. Conflating them is how an oracle
ends up "bitwise" against a reference that shares its own bugs.

| gate | compares | strength |
|---|---|---|
| **A. structural** | `muse-oracle` vs `ref_forward.py --pure --fixed-reduce` | **bitwise** |
| **B. numeric** | `muse-oracle` vs `ref_forward.py --pure` | f64 noise floor |
| **C. twin** | `muse-oracle --dtype X` vs `ref_forward.py --pure --dtype X` | **bitwise** |
| **D. determinism** | oracle vs itself across `--kernels` and `--threads` | **bitwise** |

**Gate A** holds the arithmetic fixed on both sides and varies only the model
structure. `--fixed-reduce` runs HF's own modules — its masks, its layer order,
its eps and scale placement, its gating — but routes every reduction and
transcendental through the oracle's kernels, loaded from
`libmuse_refkernels.so` (`tools/oracle_kernels.cpp`, compiled from
`src/simd.hpp`, `src/fmath.hpp` and `src/muse_glimmer.hpp`). This is the gate
that catches a wrong norm flavour, a swapped eps, a missing rope, a gate fed
from the wrong tensor. It is bitwise because the only remaining degrees of
freedom are structural.

That the oracle *defines* the reduction order is not a loophole; it is
ARCHITECTURE.md §"Numerics policy". The order itself is gated separately by
**Gate D**, and the transcendentals by `tests/fmath_test.cpp`.

**Gate B** removes the shared kernels: the reference uses BLAS' (unspecified)
dot order and libm's `exp`/`tanh`/`sin`/`cos`. Agreement can then only be at the
f64 noise floor, and the measured floor is the published quality number.

**Gate C** is bitwise even though the two sides accumulate in different orders,
because every materialization rounds to an 8- or 11-bit mantissa: an f64
accumulation difference of ~1e-16 relative lands on the same bf16 grid point
except with probability ~1e-13 per element.

---

## 3. The precision patch (`--pure`)

Stock HF rounds through f32 at hard-coded casts even in an f64 run. `--pure`
lifts exactly these, changing precision only — never a formula or an op order:

| site | stock | `--pure` |
|---|---|---|
| `MuseGlimmerRMSNorm.forward` (final `norm`, embed-norm, both qk-norms, perception norm) | `x.float()`, `weight.float()`, `.type_as(x)` | all f64 |
| `MuseGlimmerTextCenteredRMSNorm.forward` (the four sandwich norms) | same three casts | all f64 |
| `compute_default_rope_parameters` | `torch.arange(..., dtype=torch.float)`, so `theta ** e` is an **f32** power | f64 arange and power |
| `MuseGlimmerTextRotaryEmbedding.forward` | forced-f32 autocast region around the freq matmul and the trig | f64 throughout |
| `eager_attention_forward` | `softmax(..., dtype=torch.float32).to(query.dtype)` | dtype-preserving softmax |

### A torch quirk that decides how the norms are written

Measured with `py/probe_torch_ops.py` on torch 2.13.0+cpu, against a
correctly-rounded `1.0/math.sqrt(x)` ground truth over 20 000 samples:

| expression | mismatches vs correctly rounded | max ulp |
|---|---:|---:|
| `torch.rsqrt(x)` | 0 | 0 |
| `torch.pow(x, -0.5)` | 0 | 0 |
| `1.0 / torch.sqrt(x)` | 209 | 2 |
| `torch.div(ones, torch.sqrt(x))` | 209 | 2 |
| `torch.reciprocal(torch.sqrt(x))` | 209 | 2 |

It is **`torch.sqrt` that is approximate** (271/20 000 at 1 ulp), not the
division: `torch.div` and `torch.mul` are exact. So `torch.rsqrt` and
`torch.pow(·, -0.5)` are interchangeable and both equal C++'s
`1.0 / std::sqrt(x)` bit for bit — which is what `src/muse_glimmer.hpp` uses.
A referee written as `1 / torch.sqrt(ms)` would disagree with the oracle on ~1%
of norm rows and would look like an oracle bug.

Corollary for the spec: the reference's use of `torch.pow(ms, -0.5)` in
`MuseGlimmerRMSNorm` (commented as chosen to match JAX) versus `torch.rsqrt` in
`MuseGlimmerTextCenteredRMSNorm` is **not** a numerical difference between the
two norm kinds.

### Reduction orders

The mean-of-squares inside every norm is summed with 8 interleaved accumulators
and the fixed tree `((a0+a1)+(a2+a3)) + ((a4+a5)+(a6+a7))`, plus the `n mod 8`
tail. On a 1-D f64 row this happens to be exactly what torch does (verified for
n ∈ {128, 1536, 6656}); on the 2-D `[T, H]` tensors the model actually uses,
torch's inner-dim reduction matches no simple blocked order, which is why
Gate A exists and Gate B is a noise-floor gate rather than a bitwise one.

---

## 4. Measured gates — Phase 1 (tiny models)

`./run_tiny.sh`, on `tiny/tiny_text` (4 layers `[sliding, sliding, sliding,
full]`, H=64, D=16, 2 q heads / 1 kv head, I=128, V=512, window 8), T=16.

### Gate A — bitwise, f64

| artefact | differing bytes |
|---|---:|
| `logits.bin` (16 × 512) | **0** |
| `hidden_00` … `hidden_04` | **0** each |

Localized per op with `--trace-dir` on both sides (`input_ln`, `attn_gated`,
`attn_out`, `post_attn_resid`, `pre_ff_ln`, `swiglu`, `mlp_out` for all four
layers): **0 differing bytes at every trace point**.

That trace is what found the one real discrepancy during bring-up:
`ACT2FN["silu"]` is `transformers.activations.SiLUActivation`, not `nn.SiLU`, so
an activation patch aimed at `nn.SiLU` silently did nothing and layer 0 diverged
first at `swiglu` (70/1536 elements, 1.1e-16). `ref_forward.py` now patches the
class the registry actually returns and asserts the routing took effect.

### Gate B — noise floor, f64

| quantity | value |
|---|---|
| max abs logit delta | 9.68e-16 |
| mean abs logit delta | 1.18e-16 |
| argmax agreement | 16/16 |
| top-20 overlap | 100.00% |
| per-layer max abs | 0.0 (embedding) → 3.3e-14 (layer 3) |

### Gate C — bitwise, bf16 and f16 twins

| dtype | differing logit elements | argmax agreement | hidden flips |
|---|---:|---|---:|
| bf16 | **0** / 8192 | 16/16 | 0 at every layer |
| f16 | **0** / 8192 | 16/16 | 0 at every layer |

### Gate D — determinism

Six configurations — `--kernels {scalar, avx512} × --threads {1, 4, 32}` — all
**bit-identical**.

> During bring-up this gate was passing vacuously: without `-march=native` the
> `__AVX512F__` paths in `src/simd.hpp` are compiled out entirely, so
> `--kernels avx512` had nothing to select and the comparison was the scalar
> path against itself. `CMakeLists.txt` now adds `-march=native` by default
> (`-DMUSE_NATIVE=OFF` to disable) and `--kernels avx512` **errors** rather than
> falling back when the binary has no AVX-512. The sibling repos' CMakeLists do
> not set `-march`, so the same gate is worth re-checking there.

---

## 5. Measured gates — Phase 2 (the real 30B checkpoint)

`meta-models/Muse-Glimmer-30B`, ids `200000,954,7963,323,11698,373`
("`<|begin_of_text|>The capital of France is`"), T=6.

```bash
./build/muse-oracle --model meta-models/Muse-Glimmer-30B \
    --ids 200000,954,7963,323,11698,373 --out out/real_oracle --dump-hidden
.venv/bin/python py/ref_forward.py --model meta-models/Muse-Glimmer-30B \
    --ids 200000,954,7963,323,11698,373 --out out/real_ref \
    --pure --load streamed --dump-hidden --threads 32
.venv/bin/python py/diff_logits.py out/real_oracle out/real_ref --topk 64
```

Sanity: the oracle's top-1 at the last position is token 13796 = `" Paris"`.

### Gate B on the full model

| quantity | value | plan's target |
|---|---|---|
| max abs logit delta | **1.11e-13** | ~1e-13 |
| mean abs logit delta | 1.35e-14 | |
| max rel logit delta | 1.71e-09 | |
| argmax agreement | **6/6** | exact |
| top-64 overlap | **100.00%** | exact |

Per-layer max abs delta (53 hidden dumps: 0 = embedding, 1…51 = decoder layer
outputs, 52 = final-norm output):

| layer | 0 | 1 | 10 | 20 | 30 | 40 | 46 | 51 | 52 |
|---|---|---|---|---|---|---|---|---|---|
| max abs | **0.0** | 4.55e-13 | 3.41e-13 | 4.12e-13 | 1.02e-12 | 2.05e-12 | 2.27e-12 (worst) | 1.82e-12 | 1.07e-13 |

The embedding + weight-less embed-norm is **bitwise** identical; the error grows
monotonically with depth to 2.3e-12 and the final plain norm renormalizes it
back to 1.1e-13. That shape — a smooth accumulation with no step change at any
layer boundary — is itself evidence that no single op is wrong.

### Timing (this host, 32 threads)

| run | wall |
|---|---|
| `muse-oracle` f64, T=6, 52 layers | **16.9 s** |
| `ref_forward.py --pure --load streamed` | 53.7 s |

The streamed reference materializes one decoder layer at a time in f64 (484 M
parameters, 3.9 GiB each) and keeps `embed_tokens` and `lm_head` in BF16,
converting at the use site — exact, and 2.7 GiB each instead of 10.8.

### Gap to *stock* unpatched HF-f64

| comparison | max abs | mean abs | argmax | top-64 |
|---|---|---|---|---|
| pure oracle vs stock HF-f64 | **4.61e-06** | 6.81e-07 | 6/6 | 100% |
| oracle `--hf-f32-compat` vs stock HF-f64 | 9.48e-06 | 8.69e-07 | 6/6 | 100% |

So stock HF-f64 sits ~4.6e-06 away from the true f64 model function in logit
units — seven orders of magnitude above the oracle's own noise floor, and
entirely due to the f32 casts listed in §3. It does not move the argmax or the
top-64 here.

**Honest limitation of `--hf-f32-compat`.** It replicates the *placement* of
stock's f32 casts, so it shows what those casts cost, but it is **not** a
bitwise model of stock: it cannot be, because torch's f32 inner-dim reduction
matches no simple blocked order (probed for 16/32-lane variants at n = 128 and
n = 6656: no match), and stock's f32 softmax and f32 rope have the same problem.
It lands in the same 1e-5 band as stock but slightly further from it than the
pure oracle is. Use the pure-vs-stock row above as "the gap to stock"; treat
`--hf-f32-compat` as an order-of-magnitude instrument only.

---

## 6. Norm flavours — why the load-time guard is what it is

docs/plan.md calls the centered-vs-plain mix-up "the single highest-value guard
in the whole build" and suggests load-time assertions. The obvious form of that
assertion does not work, and the released checkpoint is the counter-example.

Statistics of the released 30B's norm weights:

| tensor | mean | min | max | % negative |
|---|---:|---:|---:|---:|
| `layers.0.input_layernorm` (centered) | +0.3398 | -1.0000 | 3.5000 | 1.7% |
| `layers.41.input_layernorm` (centered) | +0.8012 | -0.0054 | 1.7969 | 0.0% |
| `layers.51.post_feedforward_layernorm` (centered) | **+2.0938** | -0.4727 | 6.7500 | 0.0% |
| `norm.weight` (**plain**) | **+0.0169** | -4.9375 | 4.3750 | **49.9%** |

A zero-centered norm reaches mean +2.09 with no negative entries at all, while
the plain final norm has mean +0.017 and half its entries negative. **The two
flavours are not separable from one tensor's statistics** — the plain one looks
more "centered" than many of the centered ones. (A first version of the guard
asserted `mean <= 0.75` for centered sites and rejected the real checkpoint at
layer 41.)

What *is* sound is the aggregate, because the centered parameterisation keeps
`1 + w >= 0`:

* every per-layer norm has `min(w) >= -1` — measured minimum over all 208 is
  exactly `-1.000000`;
* 200 of 208 per-layer norms contain a negative entry, and a +1-shifted
  (GGUF-style) set would have **none**, since the originals never go below -1.

`assert_norm_flavours()` therefore checks exactly those two things: no
`min(w) < -1` (the +1 subtracted twice) and at least half the per-layer norms
containing a negative (the +1 not subtracted at all). Anything sharper needs the
cross-source evidence — GGUF `blk.*.attn_norm` == safetensors + 1 exactly, GGUF
`output_norm` == safetensors exactly — which belongs to the GGUF loader
(Phase 4) and is the assertion docs/plan.md's risk register actually describes.

---

## 7. DFlash — the plan's one open item, resolved

docs/plan.md leaves one question genuinely open ("the number of denoising
iterations per block and the acceptance rule ... blocks Phase 5"). It is
answered by `generation/candidate_generator.py::DFlashTokenCandidateGenerator`
and `generation/utils.py::_assisted_decoding` in the pinned transformers, hashed
in §1. The four findings are written up in
[ARCHITECTURE.md §"The drafting loop — resolved"](ARCHITECTURE.md#the-drafting-loop--resolved);
in brief:

1. **One pass per block** — no iterative denoising loop.
2. **15 tokens per round, not 16**: the block is `[anchor] + MASK × 15` and the
   candidate logits are `lm_head(h)[:, 1:]` — row 0 is dropped.
3. **The drafter's head is the bare `lm_head`** — no `output_multiplier`, no
   softcap. Invisible under greedy drafting (both are monotone), but it changes
   the sampled distribution.
4. **Acceptance is HF's ordinary assisted-decoding rule** — first-mismatch
   rollback when greedy, `_speculative_sampling` when sampling.

Not yet measured: the drafter's own gates (block-level token agreement,
per-position logit agreement, acceptance rate). Phase 5 is not implemented.

---

## 8. Not yet measured

| phase | gate | status |
|---|---|---|
| 4 | GGUF ingest, Q8/Q4_K bands vs llama.cpp | not started |
| 5 | DFlash oracle: block token + per-position logit agreement, acceptance rate | semantics resolved (§7), not implemented |
| 6 | vision: byte-exact preprocessing, tower + projector parity | not started |
| 7–8 | SYCL kernels, dual-GPU TP | needs the B70 box |
| 9 | serving: `serve_tests.py`, `live_api_tests.py` | not started |
| 10 | DFlash serving: `spec_parity.py` | needs the B70 box |
| 11 | benchmarks vs llama.cpp | needs the B70 box |

The tiny `tiny_vision` and `tiny_dflash` checkpoints exist (`py/make_tiny.py`)
but their gates are Phase 5/6 work and have not been run.
