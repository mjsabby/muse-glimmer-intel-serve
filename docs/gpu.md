# The SYCL / oneDNN engine on Intel Arc

`src/gpu/gpu_engine.cpp`, built by `build-gpu/` with icpx, AOT for `bmg-g31`.
It runs Muse Glimmer 30B across two Arc Pro B70s and is refereed against the
CPU stack: the f64 oracle defines the model function, `src/bf16exec.hpp` (the
bf16 twin) defines the deviation band, and this engine is a *candidate* gated
against them by `run_gpu_gates.sh`.

```
source /opt/intel/oneapi/setvars.sh
cmake -B build-gpu -DMUSE_GPU=ON -DCMAKE_CXX_COMPILER=icpx -DMUSE_NATIVE=OFF
cmake --build build-gpu -j
./run_gpu_gates.sh                        # the Phase 7 gate, on the tiny model
./build-gpu/muse-gpu --model meta-models/Muse-Glimmer-30B \
    --ids "$(cat prompt.ids)" --shards 2 --gpus 2 --chunk 512 --decode 16
```

## The hardware, measured rather than assumed

| | |
|---|---|
| 2 × Intel Arc Pro B70 | 31.89 GiB each, 256 EUs, 2.8 GHz, Level Zero |
| oneDNN | 3.11.2, GPU engine works (the **CPU** engine on this box does not) |
| fp64 | supported and correct, emulated and slow |
| sub-group sizes | 16, 32 (this engine uses 32 throughout) |
| largest single allocation | 31.75 GiB; ~30 GiB cumulative in chunks |

### Bandwidth: the number everyone quotes is wrong

A `memset`-filled buffer is uniform bytes, and the Xe memory subsystem
compresses it losslessly. Measured on the same 8 GiB buffer and the same kernel:

| fill | read bandwidth |
|---|---:|
| `memset` (uniform) | **1592 GB/s** |
| random (incompressible) | **599 GB/s** |

Weights are incompressible, so **599 GB/s is the real figure** and every decode
projection must be judged against it. The inflated number briefly made decode
look 6× off target when it was 2.7×. Any future bandwidth claim in this repo
must come from a randomly-filled buffer.

Bandwidth holds as the card fills, so weights are genuinely resident and decode
is not secretly paying PCIe:

| footprint | 1 GiB | 4 | 12 | 20 | 26 |
|---|---:|---:|---:|---:|---:|
| GB/s | 599 | 599 | 595 | 588 | 577 |

## Two layout decisions

**Weights are never transposed.** They stay in checkpoint order `W[out, in]`
row-major and oneDNN is driven as `Y[out, T] = W[out, in] · X[in, T]`, which
makes the weight the natural row-major operand. Measured at the real MLP shape:

| form | TFLOP/s |
|---|---:|
| `Y[T,out] = X · Wᵀ` (weight read through a transposed stride) | 133–153 |
| weight pre-transposed at load | 174 |
| **`Y[out,T] = W · X` (this engine)** | **173** |

Full speed with no 55 GiB shuffle at load.

**Every activation is dim-major `[dim, T]`.** It follows from the GEMM form, and
it also suits the elementwise kernels: one work-item per token means adjacent
lanes touch adjacent addresses.

The leading dimension is **per block, not per allocation**: buffers are
allocated `[dim, block]`, but a block with `n == 1` uses a leading dimension of
1. That one line is worth ~30% of decode — with a stride, oneDNN writes each of
~20 000 outputs 512 bytes apart and every norm and elementwise kernel reads
scattered.

## Traps, all of them found the hard way

### oneDNN silently ignores a B-handle offset when the matrix has one column

`matmul` with `b_md` dims `{in, 1}` returns wrong results when the B memory
handle is offset from its allocation base. No error, no NaN — just different
numbers. Measured (in = 64, out = 256):

| n | offset | verdict |
|---:|---:|---|
| 1 | 0 | ok |
| 1 | ld−1 | **WRONG** (row stride 8 and 16 alike) |
| ≥ 2 | any | ok |

The destination stride is irrelevant. This is exactly the shape of the
last-token head GEMM (`x + (n-1)`) and of every decode step, so the symptom was
a model whose 52 layers were **bitwise correct** while the logits were wrong.
It cost a full bisect to localize.

Two defences, both permanent:
* `n == 1` never goes to oneDNN through an offset pointer — the column is packed
  into a dedicated buffer first, so the handle is always at offset 0;
* `self_check()` runs at startup and compares the oneDNN route against the
  hand-written SYCL GEMM on the shapes the forward pass actually issues,
  including `(n=1, offset=ld-1)`. It throws rather than returning bad logits.

### A dim-major live region is contiguous only when `n == block`

The residual stream is `[H, block]`, so the live part of a block is `H`
segments of `n` floats at stride `block`. Copying the first `n*H` floats is
correct for a full prefill chunk and wrong for every decode step. This bit the
cross-card handoff, and the tiny-model gate missed it because its 2-GPU check
was prefill-only. `run_gpu_gates.sh` now gates 1-GPU vs 2-GPU **decode**
specifically.

### `ext_oneapi_memcpy2d` degenerates at `n == 1`

The obvious fix to the above — a 2D copy of the strided region — becomes
`width = 4 bytes, height = 6656` at decode: 6656 separate four-byte transfers,
twice per token. It measured slower than every GEMM in the model put together.
The handoff now packs on-device and does one contiguous copy.

### Default arguments desynchronized the leading dimension

`gemm()` used to default `ldx`/`ldy` to the allocated block. The moment decode
started collapsing the leading dimension to 1, the GEMMs addressed a stride the
rest of the kernels no longer used, and decode emitted a stuck token with no
error. Both are now required parameters.

## Kernel notes

* **RMSNorm** has two implementations chosen by shape alone (never by a runtime
  measurement, so a rerun cannot pick differently). The narrow one gives a row
  to one sub-group — right for prefill, where there are hundreds of rows. At
  decode there is exactly one row, and walking 6656 elements 32 at a time
  measured **68 µs**, ×4 per layer ×52 layers per token. The wide kernel spreads
  a row over 256 threads and is worth ~19% of decode. Sum of squares stays in
  f64 to match the twin; the B70 emulates it, and it is still not the
  bottleneck.
* **Decode GEMV** is oneDNN on a packed column: 776 GB/s against 700 for the
  best hand-written variant, on a card that streams 599–790 depending on the
  access pattern. The hand-written `k_gemv1` is kept as the `--no-dnnl`
  reference and as the self-check's second opinion.
* **Attention** is flash-style (S and P never materialize at bf16), one
  sub-group per (token, head). It is correct and naive; see below.
* Launch overhead on this stack is **1.3 µs**, so kernel count is not the
  problem and fusing for its own sake is not worth it.

## Tensor parallelism

`--shards N` splits every layer across N tensor-parallel shards; `--gpus M`
places them (shard `s` runs on card `s % M`). The two flags are deliberately
separate, and that separation *is* the dual-GPU gate: the shard count is what
changes the arithmetic — o_proj and mlp_down become partial sums — while the
card count only changes where it happens. So `--shards 2 --gpus 1` and
`--shards 2 --gpus 2` must agree **bitwise**, and `run_gpu_gates.sh` checks it
for prefill and decode.

| tensor | split | why |
|---|---|---|
| q_proj, gate, mlp_gate, mlp_up | rows (output dim) | contiguous slices, no communication |
| o_proj, mlp_down | columns (input dim) | each shard makes a partial `[H, n]` → all-reduce |
| lm_head | rows (vocab) | multiplier and softcap are elementwise and commute with the split |
| norms, embedding | replicated | 13 KB each; a broadcast costs more than the duplicate |
| k_proj, v_proj, KV cache | replicated | see below |

**KV is replicated, not sharded.** Muse Glimmer has 2 KV heads and the tiny gate
model has 1, so sharding by KV head either fails or needs a second code path.
Replicating costs 3.4 MB per layer per shard and ~0.7% duplicated work. Each
shard still runs attention only for its own q heads — `k_attention` takes a
global head offset, because the GQA group index has to come from the global head
number while the cache holds all heads.

The residual stream is replicated on every shard, which is what the replicated
embedding buys: no broadcast after the lookup, and every replicated op
downstream is computed from identical inputs and so stays identical by
construction. The all-reduce sums in **shard order** on every shard for the same
reason — the norms that follow are replicated, so a one-ulp disagreement between
shards would compound rather than average out.

### The transport is host-staged, and that is not the obvious choice

The all-reduce fires twice per layer, 104 times per token. Direct card-to-card
`memcpy` in one shared SYCL context is the obvious transport and benchmarks at
**48 GB/s** standalone. Inside this engine it measured **0.4 GB/s** — 70× slower
— which made TP prefill *worse* than no TP at all (70 tok/s against 549).

What it is not: VRAM pressure (49.8 GB/s with 26 GiB of ballast per card), a
kernel-written source surface (47.9 GB/s), or missing peer access (reported
supported, and explicit enabling changes nothing). It does not reproduce
standalone, and the cause is still open.

Staging through **pinned host memory** — one hop longer — runs at 28.2 GB/s in
the engine, essentially the PCIe ceiling, and is the default. `MUSE_GPU_P2P=1`
selects the direct path for re-measuring when the driver changes.

| transport | exchange | prefill 512 |
|---|---:|---:|
| direct card-to-card | 0.4 GB/s | 70 tok/s |
| **pinned host staging** | **28.2 GB/s** | **1030 tok/s** |

### Memory

27.36 GiB of weights per card, against 31.89 GiB physical and ~30 GiB usable.
Replicating the embedding and the KV projections costs ~1.4 GiB per card over a
layer split, and buys both cards streaming concurrently.

| context | KV total | per card | fits |
|---|---:|---:|---|
| 4 096 | 0.26 GiB | 0.13 | yes |
| 32 768 | 0.97 GiB | 0.49 | yes |
| 131 072 (full window) | 3.41 GiB | 1.71 | yes |

## Measured, 30B BF16, two cards

Prompt of pseudo-random ids; CPU column is `muse-oracle --exec bf16` on the
same box (32-thread Zen 5, AVX-512, weights resident).

| | tensor parallel | layer split | CPU | vs CPU |
|---|---:|---:|---:|---:|
| prefill 128 | 644 tok/s | 549 | 60.7 | 10.6× |
| prefill 512 | **1030 tok/s** | 941 | 67.2 | 15.3× |
| prefill 2048 | **867 tok/s** | 623 | 64.1 | 13.5× |
| decode | **14.81 tok/s** | 8.89 | 1.13 | 13.1× |
| weight upload | 54.72 GiB in ~22 s | 51.88 in ~15 s | — | — |

Tensor parallelism is worth **1.67× on decode** and 1.39× on deep prefill over
the layer split, for ~1.4 GiB more per card.

Agreement with the bf16 twin after prefill 128 + 8 decode steps: **all 8
generated tokens identical**, argmax equal, top-64 overlap 98.4%, max abs logit
difference 0.125 — one bf16 ulp at that magnitude, which is what a different
reduction order costs.

### Where decode still loses

A GEMM-only benchmark of one decode token (52 layers × 8 projections, real
shapes, cached primitives) runs in **87.9 ms at 572 GB/s** on one card —
essentially its ceiling. Under TP each card streams half of that, so ~44 ms is
the floor, i.e. **~22 tok/s**. At 14.81 the engine is at 67% of it; the gap is
the per-layer small kernels and the 104 all-reduce round trips (1.5 ms of
transfer per token, plus their synchronization).

Prefill still falls from 1030 to 867 tok/s between 512 and 2048 because the
attention kernel is O(T²) with a serial key loop per (token, head). A
tiled/flash prefill attention is the fix, and it is now the largest single item
left.

## Gates (`run_gpu_gates.sh`)

| gate | what it protects |
|---|---|
| gpu == bf16 twin | the referee contract |
| chunk invariance 1…16 | the sliding ring holds `window + chunk` rows |
| oneDNN == hand-written GEMM | the one-column bug above, permanently |
| decode == prefill | the decode path is not a second implementation |
| rerun bit-identical | determinism |
| shards 2: 1 card == 2 cards, prefill **and decode** | tensor-parallel placement is not arithmetic |
| 2-card rerun bit-identical | the all-reduce is order-fixed |

`MUSE_GPU_TRACE=<dir>` dumps the residual stream per layer (row-major `[n, H]`),
which is how the oneDNN bug was localized to the head. `MUSE_GPU_PROFILE=1`
prints per-stage attribution; it inserts a queue wait per stage, so it changes
what it measures and is a diagnostic, not a throughput report.
