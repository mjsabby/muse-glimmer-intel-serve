# The f64 oracle — build, run, and what each flag means

`muse-oracle` is the deterministic float64 CPU implementation of Muse Glimmer's
text path. It defines the model function; every other implementation in this
repo (the bf16/f16 twin, and later the SYCL kernels) is refereed against it.

What it is *not*: a fast path. A 52-layer f64 forward over the 30B checkpoint
takes ~17 s for 6 tokens on a 32-thread desktop and streams 55 GiB of weights
per call. It exists to be right, reproducibly.

## Build

```bash
./build.sh --cpu-only          # g++, no oneAPI required
```

or directly:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)" --target muse-oracle fmath-test unit_tests muse_refkernels
ctest --test-dir build --output-on-failure
```

`-march=native` is on by default (`-DMUSE_NATIVE=OFF` to turn it off). It
changes **speed only** — `src/simd.hpp`'s AVX-512 and scalar executions of the
8-lane fma reduction are the same abstract operation sequence — but without it
the `__AVX512F__` paths are compiled out and `--kernels avx512` has nothing to
select, which silently turns the determinism gate into a no-op.

## Run

```bash
./build/muse-oracle --model meta-models/Muse-Glimmer-30B \
                    --ids 200000,954,7963,323,11698,373 \
                    --out out/run --dump-hidden
```

`--model` takes a snapshot directory or an `org/repo` id resolved through the
**local** HF cache (`$HF_HUB_CACHE` → `$HF_HOME/hub` → `~/.cache/huggingface/hub`);
it never touches the network. `--ids` takes a comma/space separated list or a
file containing one — `py/encode_prompt.py` produces both.

Output in `--out`:

| file | contents |
|---|---|
| `logits.bin` | f64 `[T, V]`, row-major |
| `meta.json` | ids, `T`, `V`, mode, dtype, kernel selection, hidden count |
| `hidden_XX.bin` | with `--dump-hidden`: f64 `[T, H]` per index |

Hidden-state indices follow HF's `output_hidden_states` convention:
`0` = the embedding (**after** the weight-less embed-norm), `i+1` = the output
of decoder layer `i` for `i < L-1`, and `L` = the final-norm output. The last
decoder layer's raw output is not exposed on either side, so the two dumps line
up index for index.

## Flags

| flag | meaning |
|---|---|
| `--dtype f64\|bf16\|f16` | storage dtype of the executed model. `f64` is the oracle; `bf16`/`f16` are the deterministic twins (see below). |
| `--attn eager\|flash` | `--dtype != f64` only. `eager` materializes the attention scores and probabilities at the storage dtype, as a stock eager run does. `flash` skips both materializations, which is the legal target for a fused GPU kernel. |
| `--kernels auto\|scalar\|avx512` | selects the execution of the fixed 8-lane fma reduction order. Results are **bit-identical** across all three; this exists to prove that. `avx512` errors if the binary has no AVX-512 rather than falling back. |
| `--threads N` | OpenMP threads. Results are bit-identical for every N — parallelism is only across independent output elements. |
| `--hf-f32-compat` | replicate the *placement* of stock HF's hard-coded f32 casts instead of running pure f64. An instrument for sizing those casts, not a bitwise model of stock — see VERIFICATION.md §5. |
| `--trace-dir DIR` | dump every named intermediate of every layer (`L03.swiglu.bin`, …). The fastest way to localize a disagreement to one op. |
| `--top K` | print the last position's top-K to stdout. |

## The bf16 / f16 twins

`--dtype bf16` does **not** compute in bf16. Every operator stays exact f64
internally; what changes is that the result is rounded to the storage dtype at
exactly the points a stock low-precision run materializes a tensor:

* once per `nn.Linear` output (an ideal accumulator, then one rounding);
* once per residual add, per activation, per elementwise product;
* the rope tables through the stock f32 chain, then rounded at `.to(x.dtype)`;
* rope application as three ops (`x*cos`, `rotate_half(x)*sin`, their sum);
* attention: matmul output and `* scaling` (eager only), probabilities at
  `.to(query.dtype)` (eager only), and the `P@V` result;
* the four output-tail ops (`* output_multiplier`, `/ softcap`, `tanh`,
  `* softcap`) each round separately;
* Python-scalar constants (`head_dim^-0.5`, `qk_scale_factor`,
  `output_multiplier`, `final_logit_softcapping`) take their **f32 opmath**
  value, which is what torch's scalar promotion does.

The rounding chain itself is `RNE(f64→f32)` then `RNE(f32→bf16/f16)` — torch's
cast is a double rounding through f32, verified on this torch build
(`py/probe_torch_ops.py`).

This makes the twin a *specification* of the deviation band rather than a
simulation of it: a GPU kernel that matches the twin bitwise is correct by
construction, and one that does not can be diffed against it position by
position with `py/diff_lp.py`.

## The vision tower

```bash
# the reference produces the pixels AND the features
.venv/bin/python py/ref_vision.py --model meta-models/Muse-Glimmer-30B \
    --images cat.jpg --out out/vis --pure --load streamed --threads 32

# the oracle consumes exactly those pixels
./build/muse-oracle --model meta-models/Muse-Glimmer-30B --ids 1,2,3 \
    --out out/vis_oracle --pixels out/vis/pixel_values.bin --grid 1,10,16
```

`--pixels` takes raw f64 `[N, patch_dim]` (`patch_dim` = 2·3·14² = 1176) and
`--grid` the `t,h,w` patch grid per image, semicolon-separated for several. The
tower's projected output is written to `vision.bin` as f64
`[N / merge_size², text hidden]`; if `--ids` contains image (200092) or video
(200091) placeholder tokens, those features are scattered into `inputs_embeds`
and the run continues into the text stack, giving end-to-end multimodal logits.

Pixels come from the reference on purpose. Pixel ingestion and the tower are two
different correctness problems — the first is a bit-exactness question about
**torchvision's** antialiased Lanczos kernel (not PIL's; see
ARCHITECTURE.md §"Vision ops"), the second is about the model's architecture.
Feeding the oracle the reference's own `pixel_values` gates the second cleanly
and does not pretend the first is solved. C++ preprocessing is not implemented.

The scatter happens **after** the weight-less embed-norm: the reference embeds
the placeholder ids as 0 and then `masked_scatter`s, and the features already
carry `perception_emb_norm`. Normalizing them again would be wrong, and would
look almost right.

## The DFlash drafter

```bash
./build/muse-oracle --model meta-models/Muse-Glimmer-30B \
                    --assistant meta-models/Muse-Glimmer-30B-assistant \
                    --ids 200000,954,7963,323,11698,373 --out out/run
```

`--assistant` taps the target's hidden states at the drafter's
`target_layer_ids` on the way through the forward, then runs **one** drafting
round — the same round `DFlashTokenCandidateGenerator` performs on its first
call. It writes `out/run/draft/{logits.bin,hidden.bin,meta.json}`:
`block_size - 1` rows of **bare** `lm_head` logits (no `output_multiplier`, no
softcap — that is what the reference does), the drafter's post-norm hidden
states for all `block_size` rows, and the argmax tokens.

Both are dumped on purpose: the tokens are discrete argmax decisions, so a
near-tie can flip between two correct implementations. A gate that compares only
tokens will eventually pass something wrong, or fail something right.

`--draft-rounds N` runs the greedy speculative loop for N rounds and reports
`accept_rate` and tokens/round. The acceptance rule is HF's ordinary
assisted-decoding one, so a round always yields at least 1 token (the target's
own bonus) and at most `block_size`. The oracle has no KV cache — it is a
prefill referee — so every round re-runs the whole target forward; the round
*count* matches a real engine, the cost per round does not.
`tools/spec_baseline.sh` sweeps the fixed prompt set in `tools/prompts/` and
prints the baseline table.

`py/ref_dflash.py` is the matching reference. It has one non-obvious
requirement: the drafter must be called with a `DFlashCache` on which
`set_previous_accepted_tokens(n)` has been set, because
`DFlashCache.get_query_offset()` is what places the block's queries at absolute
positions `n … n+B-1` for the sliding-window mask. Without it the block is
masked as if it sat at positions `0 … B-1`; nothing errors, and the drafted
tokens are simply different.

## Refereeing against the HF reference

```bash
# structural gate: bitwise (shared arithmetic, HF's structure)
.venv/bin/python py/ref_forward.py --model tiny/tiny_text --ids ... \
    --out out/ref --pure --fixed-reduce --dump-hidden
cmp out/oracle/logits.bin out/ref/logits.bin

# numeric gate: HF's own BLAS/libm, compared at the f64 noise floor
.venv/bin/python py/ref_forward.py --model meta-models/Muse-Glimmer-30B --ids ... \
    --out out/ref --pure --load streamed --dump-hidden --threads 32
.venv/bin/python py/diff_logits.py out/oracle out/ref --topk 64

# twin gate: bitwise
.venv/bin/python py/ref_forward.py ... --pure --dtype bf16 --dump-hidden
.venv/bin/python py/diff_lp.py out/oracle_bf16 out/ref_bf16 --dtype bf16
```

`./run_tiny.sh` runs all of these on the tiny models plus the determinism sweep,
and is the gate to run before touching anything numeric. VERIFICATION.md
explains why there are three separate gates and what each one does and does not
prove.

`--fixed-reduce` loads `build/libmuse_refkernels.so`, which is compiled from the
oracle's own headers. It is for toy dimensions only — a scalar C loop, not BLAS
— and `ref_forward.py` will happily take minutes if pointed at a real
checkpoint with it.

## Reading the code

`src/muse_glimmer.hpp` is the whole model, in execution order:

| section | what |
|---|---|
| `parse_config` | including the checks that refuse to guess: missing `post_norm_eps`, a per-layer theta that disagrees with `rope_parameters`, `attention_bias`, a non-silu activation |
| `bind_weights` / `assert_norm_flavours` | tensor binding and the +1-shift guard (VERIFICATION.md §6 explains why the guard is an aggregate one) |
| `gemm*` | the 8-lane fma reduction, panelled for cache, bit-identical to `dot8` |
| `mean_sq`, `rmsnorm_rows` | the three norm flavours behind one `NormKind` enum |
| `build_rope_table`, `apply_rope` | full-head rotation, `rotate_half` pairing |
| `attention_forward` | QK-norm → `q *= 3.87` → optional rope → `· 128^-0.5` → masked softmax → GQA gather → `⊙ sigmoid(gate_proj(x))` → `o_proj` |
| `apply_output_tail` | `· output_multiplier`, then `20·tanh(z/20)`, in the reference's four-op order |
| `forward` | the Gemma-style sandwich, both eps values, and the hidden/trace hooks |

The four things most likely to be got wrong are called out at the top of that
file, and each has a gate: three norm flavours, two eps values assigned by
position, NoPE on the global layers, and a weight-less QK-norm whose
`qk_scale_factor` multiplies **q only**.
