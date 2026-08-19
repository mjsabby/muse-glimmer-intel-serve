# Verification — methodology, pinned reference, measured gates

The oracle's contract is *relative to a pinned reference*. This file records
which reference, what "agreement" means for each gate, and every number
measured so far. Everything below is reproducible with `./run_tiny.sh` and the
commands quoted in each section.

Status: every phase of [docs/plan.md](docs/plan.md) except **4 (GGUF ingest)**
is implemented and gated — the oracle and its twins (0–3), the DFlash drafter
(5), the vision tower (6, model function only), the SYCL engine and dual-GPU
tensor parallelism (7–8), the serving frontend (9), speculative serving (10)
and the head-to-head against llama.cpp (11). §8 lists what is still not
measured, which is shorter than it was and no less honest for it.

Gate counts, all green as of this writing:

| suite | checks | needs |
|---|---:|---|
| `./run_tiny.sh` | 30 | CPU only |
| `./run_gpu_gates.sh` | 30 | the B70 box |
| `.venv/bin/python -m tests.serve_tests` | 42 | CPU only |
| `.venv/bin/python -m tests.live_api_tests` | 33 | the box, a running server |

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
| torchvision | **0.28.0+cpu** — load-bearing: `MuseGlimmerImageProcessor` is a `TorchvisionBackend`, and below **0.27** its `resize` silently substitutes BICUBIC for the LANCZOS the checkpoint asks for (§6c) |
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

## 6b. DFlash drafter gates (Phase 5)

`src/dflash.hpp` implements `MuseGlimmerAssistantModel` and one drafting round;
`py/ref_dflash.py` is the matching HF reference.

### tiny_dflash — bitwise

`tiny/tiny_text` + `tiny/tiny_dflash` (2 drafter layers, `block_size` 4,
`target_layer_ids` [0, 2]), T=12:

| artefact | differing bytes |
|---|---:|
| `draft/logits.bin` (3 × 512, bare head) | **0** |
| `draft/hidden.bin` (4 × 64, post-norm) | **0** |
| draft tokens | identical |

### The released 30B + released drafter

`meta-models/Muse-Glimmer-30B` + `meta-models/Muse-Glimmer-30B-assistant`,
prompt `"<|begin_of_text|>The capital of France is"`, one round:

| quantity | value |
|---|---|
| anchor (target's bonus token) | 13796 = `" Paris"` |
| proposals per round | **15** (block_size 16, anchor row dropped) |
| draft token agreement | **15/15 identical** |
| per-position max abs logit delta | **3.70e-13** (mean 5.17e-14) |
| per-position argmax agreement | **15/15** |
| per-position top-64 overlap | **100.00%** |
| drafter post-norm hidden, max abs | 4.35e-14 |

Decoded: `" Paris" → ", and is is of of, to to to to to to to to"` — the block
degrades with distance from the anchor, which is what a one-pass block-diffusion
drafter does.

Both the tokens *and* the per-position logits are gated, per docs/plan.md: draft
tokens are discrete argmax decisions, so a near-tie can legitimately flip
between two correct implementations, and a token-only gate would eventually pass
something wrong or fail something right.

### Acceptance-rate baseline

`tools/spec_baseline.sh 6` — greedy speculative decoding over the four prompts
in `tools/prompts/`, 6 rounds each, measured with the f64 oracle so the number
is a property of the **model**, not of a kernel's precision. Phase 10's engine
must reproduce it; a lossier drafter tier costs acceptance rate, not
correctness, and this is what that cost is measured against.

| prompt | rounds | drafted | accepted | accept_rate | tokens/round |
|---|---:|---:|---:|---:|---:|
| `code` (a Python function body) | 6 | 90 | 46 | **0.5111** | **8.667** |
| `list` (an enumerated list) | 6 | 90 | 16 | 0.1778 | 3.667 |
| `fact` ("The capital of France is") | 6 | 90 | 8 | 0.0889 | 2.333 |
| `prose` (open-ended narrative) | 6 | 90 | 8 | 0.0889 | 2.333 |
| **all** | **24** | **360** | **78** | **0.2167** | **4.250** |

A round always yields at least one token (the target's own bonus) and at most
`block_size` = 16, so tokens/round is `1 + accepted/round`. Two of the six code
rounds accepted all 15 proposals.

Caveats to carry into Phase 10 rather than quote as a headline: this is 24
rounds on four short prompts of a base (not instruction-tuned) checkpoint, and
the spread by prompt type — 8.7 tokens/round on code versus 2.3 on prose — is
much larger than the sampling noise. Compare per-prompt, not just the aggregate.
Wall time was 8m10s: the oracle has no KV cache, so each round re-runs the whole
f64 target forward.

### The mask offset that only the cache knows

The drafter's mask is built without `position_ids`; the query index for the
`abs(q_idx - kv_idx) <= sliding_window` overlay comes from
`DFlashCache.get_query_offset()`, which adds `previous_accepted_tokens`. A
first version of `py/ref_dflash.py` called the drafter with `use_cache=False`
and no cache, which silently placed the block at positions `0 … B-1` instead of
`n … n+B-1`. Nothing errored; the tiny model simply drafted `[113, 94, 94]`
instead of `[356, 356, 356]`. This is the kind of failure docs/plan.md warns
about for DFlash — "plausible-looking but wrong acceptance rates rather than an
obvious failure" — and it is why the gate compares against the oracle rather
than against a smoke test.

## 6c. Vision tower gates (Phase 6, partial)

`src/vision.hpp` implements `MuseGlimmerVisionModel` + `MuseGlimmerVisionAdapter`
+ `vision_projection` + `perception_emb_norm`; `py/ref_vision.py` is the
matching reference. The oracle consumes the **reference's own** `pixel_values`,
so these gates cover the tower and the projector, not pixel ingestion — see
"Not implemented" below.

### tiny_vision — bitwise, end to end

4-layer tower, hidden 32, 2 heads, `[window, window, window, full]`, pos grid
4×4, patch 14, merge 2. A 120×150 synthetic image becomes an 8×10 patch grid
(80 patches → 20 merged tokens):

| artefact | differing bytes |
|---|---:|
| `vision.bin` (20 × 64 projected features) | **0** |
| `logits.bin` (image + text, 22 × 512) | **0** |

The end-to-end row is the stronger one: it covers the scatter into
`inputs_embeds` at the placeholder positions as well as the tower.

### The released 30B tower

A 137×211 synthetic image → `smart_resize` 140×224 → grid 1×10×16 (160 patches
→ 40 merged tokens), 50 tower layers:

| quantity | value |
|---|---|
| max abs delta on the projected features | **1.42e-12** |
| mean abs delta | 1.19e-14 |
| max rel delta | 2.70e-09 |

End to end (40 image placeholders + a 6-token text tail, 46 positions through
the full 52-layer text stack):

| quantity | value |
|---|---|
| max abs logit delta | **1.26e-11** |
| mean abs logit delta | 2.73e-13 |
| argmax agreement | **46/46** |
| top-64 overlap | **100.00%** |

The end-to-end logit delta is two orders above the text-only 1.11e-13 because
the tower's 1.4e-12 feature error enters at the embedding and is then amplified
through 52 decoder layers. Worth remembering when setting a gate threshold for
image prompts: it is not the same number as the text one.

Structure is pinned by the bitwise tiny gate; this number is the f64 noise floor
of a 50-layer tower at the real dimensions, where the window path differs from
the tiny one (32-patch windows, and the `pad = win - n % win` quirk that emits a
whole extra all-padding window when `n` is already a multiple of `win`).

### Not implemented: pixel ingestion

The plan says "PIL-exact preprocessing in C++ … gemma4 already has a byte-exact
PIL bicubic port; this needs the LANCZOS kernel added". That is the wrong target
for this checkpoint, and the correction matters more than the missing code:

* `MuseGlimmerImageProcessor` is a **`TorchvisionBackend`**. `resample: 1` is
  `PILImageResampling.LANCZOS`, but it is mapped to
  `tvF.InterpolationMode.LANCZOS` and executed by
  `torchvision.transforms.v2.functional.resize(..., antialias=True)`.
* Torchvision only supports LANCZOS **for tensors** from 0.27; below that,
  `TorchvisionBackend.resize` warns once and silently substitutes **BICUBIC**.
  The resample kernel is a property of the installed torchvision and is pinned
  in §1 for that reason.
* Measured on this build (torchvision 0.28.0+cpu): resize runs on **uint8** in,
  uint8 out, and agrees with PIL's LANCZOS to ≤ 1/255 (0.49% of channels differ
  by 1). The same call on a **float** tensor diverges from PIL by up to 46/255,
  because nothing clamps Lanczos' negative lobes — so "which dtype does the
  resize see" is a real question, not a detail.

So a C++ port must reproduce torchvision's antialiased Lanczos on uint8, and a
PIL port would be off by up to 1/255 per channel before the model even starts.
`smart_resize`, rescale/normalize and `patchify` are deterministic integer/affine
work and are the easy part.

## 6d. GPU engine gates (Phases 7–8) — `./run_gpu_gates.sh`

The engine is a **candidate under the referee stack, not a second oracle**. The
four claim classes of §2 map onto it directly, and the mapping is the point:
where a GPU kernel performs the twin's operations in the twin's order the gate
is bitwise, and where a reduction order necessarily differs the gate is an
envelope against the f64 oracle with the twin's own deviation as the budget.
Calling the second kind "close enough" without naming the budget is how a
quantization bug hides.

28 checks on `tiny/tiny_text`, `tiny/tiny_vision` and `tiny/tiny_dflash`:

| gate | compares | strength |
|---|---|---|
| gpu (1 shard) == bf16 twin | `muse-gpu` vs `muse-oracle --exec bf16` | **bitwise** |
| gpu (2 shards) == bf16 twin | the same at 2 shards | **bitwise** |
| chunk 1 == 2 == 4 == 8 == 16 | prefill block width | **bitwise** |
| oneDNN == hand-written SYCL GEMM | `--no-dnnl` A/B | **bitwise** |
| decode path == prefill path | N decode steps vs prefilling the same sequence | **bitwise** |
| rerun bit-identical | the engine against itself | **bitwise** |
| shards 2: 1 card == 2 cards, prefill **and** decode | placement is not arithmetic | **bitwise** |
| 2-card rerun bit-identical | the all-reduce is order-fixed | **bitwise** |
| flash tier inside the twin's envelope | `--flash-prefill` | envelope (max abs 3.906e-03, argmax 1/1, top-20 90%) |
| flash tier differs from the exact tier | the tier is actually engaged | inequality |
| flash tier rerun + 1 card == 2 cards | looser numerics, not sloppy ones | **bitwise** |
| flash-decode inside the exact path's envelope | `--flash-decode` split-K | envelope (max abs 0.000e+00 on this model) |
| flash-decode rerun bit-identical | | **bitwise** |
| drafter proposals == the f64 oracle's | DFlash on GPU | **token-exact** |
| q8 drafter (int8 DPAS) proposals == the f64 oracle's | quantized activations | **token-exact** |
| drafter: 1 card == 2 cards | | **bitwise** |
| spec sequence == the f64 oracle's | the accept rule and the rollback | **token-exact** |
| spec loop rerun deterministic | oneDNN's atomic split-K, which one forward cannot see | **bitwise** |
| vision features vs the f64 oracle | max 1.958e-02 against the **twin's own** 2.419e-02 | envelope |
| image+text logits | argmax 189 vs 189, top-20 100% | envelope |
| q8 vs the f64 oracle | argmax 47 vs 47, top-20 100%, max 1.200e-02 | separate tier |
| q8 chunk-invariant (1 == 16), q8 rerun | | **bitwise** |
| sealed engine: 15 prompt lengths | no post-seal allocation, no VRAM drift | exact (0 bytes) |
| the seal fires without prewarm | a tripwire nobody has seen trip is untested code | behaviour |

Three of these deserve their reason recorded, because each was written after a
bug got past the others:

* **decode == prefill** exists because the decode path collapses the leading
  dimension to 1 and takes a different GEMM route. It is a second
  implementation of the same function and it once produced a stuck token with
  no error ([gpu.md](docs/gpu.md)'s one-column oneDNN bug).
* **shards 2: 1 card == 2 cards, at DECODE** was added after the prefill-only
  version passed while `hand_off` assumed a contiguity that only holds for
  prefill.
* **spec loop rerun deterministic** was added after speculative decoding
  produced 220 tokens on one run and 176 on the next, with every single-forward
  gate green. oneDNN's atomic split-K; fixed with `set_deterministic(true)`.

Numbers on the real 30B are in [gpu.md](docs/gpu.md) and
[comparison.md](docs/comparison.md); this file records what the gates prove,
not how fast it is.

---

## 6e. Serving gates (Phase 9)

Two suites, split by what they can prove without the weights.

**`tests/serve_tests.py` — 41 checks, no GPU.** The engine is faked by a
deterministic target *addressed by cache position*: the distribution after a
cache of length L is a point mass on `script[L - prompt_len]`. That detail is
the test. A fake that advanced a cursor per call passes the speculative accept
path and diverges on the reject path — the one the test exists for.

Covers: the generation prompt is a bare `<|start|>assistant` (the premise
`serve/recipient.py` is built on); the reasoning-strength line is always
emitted and a system prompt's own line wins; `tool_calls[].function.arguments`
must be deserialized or the template `raise_exception`s; `<|eom|>` is not an
eos id; the ATEM channel machine (reasoning never leaks into content, tool
calls whole, truncated turns still emit); the recipient grammar (a forced tool
name is the only legal recipient, `none` leaves no tool addressable, `self` is
withdrawn after one use, the mask is `None` inside the body); greedy
speculative output identical to greedy plain on the fake; raw `/v1/completions`
using the prompt verbatim; and all three wire formats end to end, including the
second turn of a tool call, an api key, and the errors that must stay errors.

**`tests/live_api_tests.py` — 30 checks, on the box.** What only the weights
can show: the channel split on a real turn, determinism from a known cache
state, the speculative tie property (§6f), `tool_choice` forcing a recipient
the model did not want, guided JSON parsing, streamed == non-streamed,
prefix reuse being both a speed-up and the same answer, both other protocols,
an image changing the answer, and context overflow / `input_audio` /
empty-messages returning 400.

---

## 6f. What speculative decoding guarantees, and what it does not

The tiny-model gate is **token-exact against the f64 oracle**, and that is the
strongest statement available. It does **not** generalize to "greedy
speculative output is bit-identical to greedy plain output on the 30B", and the
plan's Phase 10 exit gate as written asked for exactly that. It cannot hold:

* a speculatively verified token's logits come from a **16-row** forward —
  oneDNN matmul, tile-softmax attention;
* a plain decode's come from a **1-row** forward — hand-written GEMV, split-K
  attention.

Different arithmetic, both inside the twin's envelope, neither of them "the"
answer. Making them identical would mean verifying 16 tokens with 16 separate
decode steps, which is the non-speculative path.

Measured on the 30B: identical on 1–3 of 3 gate prompts depending on cache
state, and **every observed first divergence was a tie** — in each case one of
the two paths saw an *exact* tie (0.0000 logprob gap between the top two
candidates) where the other saw 0.06–0.25. The live gate checks that property
instead of equality, because a real acceptance bug diverges at a comfortable
margin.

The same caveat applies to prefix reuse: it changes the prefill chunking, and
the fast attention tiers are envelope-level, so the same request from a
different cache state can land on the other side of a tie.
`--no-prefix-reuse` makes the answer a function of the prompt alone.

---

## 6g. Video and long-context retrieval

Two gaps that were closed after the rest of the stack shipped, both because
"it is implemented" and "it has ever been run" are different statements.

### Video: the same tower, a temporal grid

`get_video_features` is `get_image_features` with a `t > 1` grid, so the tower
is not the risk — the grid handling is (window index, 2-D position taps and
pixel-shuffle offsets are all computed per frame), and so is the placeholder
id, which is `video_token_id` and not `image_token_id`.

| gate | grid | strength |
|---|---|---|
| `run_tiny.sh` video features, 4 frames | 2,8,12 | **bitwise** (0 differing bytes) |
| `run_tiny.sh` video+text logits, 4 frames | 2,8,12 | **bitwise** |
| `run_tiny.sh` video features + logits, 8 frames | 4,6,10 | **bitwise** |
| `run_gpu_gates.sh` video features vs the f64 oracle | 2,8,12 | envelope (2.218e-02 against the twin's own 2.459e-02) |
| `run_gpu_gates.sh` video+text logits | 2,8,12 | argmax + top-20 |
| `live_api_tests.py` frame-list video | | temporal order (red shot then blue) |

Writing the reference's video path found the first bug immediately: the
end-to-end call handed video pixels in through the *image* argument, which
scatters nothing (`tokens: 0, features: 48`) because the mask is keyed on the
placeholder id.

Serving decodes containers through the **ffmpeg binary** rather than a Python
decoder — `pyav`, `decord`, `opencv` and torchvision's video reader are all
absent from the pinned environment, and ffmpeg is what all four of them wrap.
A 2-second `testsrc2` clip end to end returns "SMPTE color-bars … timed from
00:00:00.500 to 00:00:01.500", which is the frames *and* their order arriving
intact. A frame-list form (`video_url: {frames: [...]}`) needs no decoder at
all and is what the gate uses.

### Retrieval at depth

Speed at depth is not retrieval at depth. `tools/longctx.py` places one
distinctive sentence at a known fraction of a haystack of numbered filler
paragraphs and asks for it back; the needle is a random word and number drawn
per run, so a pass cannot come from the checkpoint having memorized anything.

| prompt tokens | 5% | 50% | 95% |
|---:|---|---|---|
| 4 161 | found | found | found |
| 16 491 | found | found | found |
| 65 632 | found | found | found |
| **131 517** | **found** | **found** | **found** |

12 of 12, the last three at 111 s of prefill each on two B70s. The answer is
scored across both channels: a code recalled in `reasoning_content` and not
repeated is still retrieval, and scoring only `content` would call it a miss.

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
| 4 | GGUF ingest, Q8/Q4_K bands vs llama.cpp | **not started** — the only untouched phase |
| 6 | vision: byte-exact **preprocessing** in C++ | **not started**. The server reaches the checkpoint's own `MuseGlimmerImageProcessor` through Python, so serving is byte-exact by construction; what is missing is a C++ ingestion path for a Python-free deployment. §6c explains why the plan's PIL target is wrong |
| 6 | video: 2 fps sampling from a container | **gated for the grid, not for the sampling.** `t > 1` is now bitwise on the CPU, envelope-gated on the GPU and checked end to end through the server (§6g); what is *not* gated is the frame selection ffmpeg does before the processor sees the frames |

Three more honest gaps inside what *is* implemented:

* the **bf16/f16 twins of the drafter and the tower** are written (`--dtype`
  threads through `src/dflash.hpp` and `src/vision.hpp`) but only the text
  twin is gated. `py/ref_dflash.py` and `py/ref_vision.py` have no
  low-precision instrumentation yet.
* `--hf-f32-compat` covers the vision tower's f32 sites (the
  `apply_rotary_pos_emb_vision` downcast, the f32 softmax) but that path has not
  been compared against stock, only the text one (§5).
* the GPU gates run on the **tiny** models. The real 30B is checked against the
  f64 oracle at the logit level (argmax and top-k, §5) and against llama.cpp
  for throughput, but there is no bitwise 30B GPU gate and there cannot be one:
  the f64 oracle's 30B forward takes ~17 s for 6 tokens, and the twin is a CPU
  implementation whose reduction order the GPU does not share at that scale.
