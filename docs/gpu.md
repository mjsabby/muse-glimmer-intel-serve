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

### The exchange must be serialized, and peer access enabled before the context

The all-reduce fires twice per layer, 104 times per token. Getting it wrong is
expensive: the first working version ran the exchange at **0.4 GB/s**, which
made TP prefill *worse* than no TP at all (70 tok/s against 549). Three
separate things were wrong, and only the third is obvious in hindsight.

1. **Peer access has to be enabled BEFORE the spanning context is created.**
   The Level-Zero V2 adapter establishes its cross-device mappings at context
   creation; enabling afterwards is not an error, it just silently leaves them
   absent. (gemma4 documents the same trap.)
2. **The copy is a PULL**, issued on the destination's queue, not a push from
   the source.
3. **The two legs must be serialized.** Letting both cards pull from each other
   concurrently is *bimodal* on this driver: the same 13.6 MB disjoint exchange
   measures 0.56 ms (48 GB/s) on one run and 639 ms on the next, with no change
   in size, alignment or aliasing. Serializing gives up the overlap and is
   stable at 27 GB/s — and PCIe is the limit either way, so nothing is lost.

| transport | exchange | prefill 512 |
|---|---:|---:|
| concurrent bidirectional pulls | 0.4 GB/s (bimodal) | 70 tok/s |
| **serialized peer pulls** (default) | **26.6 GB/s** | **1008 tok/s** |
| pinned host staging (`MUSE_GPU_XFER=host`) | 28.1 GB/s | 984 tok/s |

Host staging is kept as a fallback because it is immune to the concurrency
bug, but it is not needed: serialized peer copies are marginally faster
end-to-end and avoid the host hop entirely. `MUSE_GPU_PEER_PROBE=1` times the
peer path at two points in startup, which is how the bimodality was found.

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
| prefill 128 | 638 tok/s | 549 | 60.7 | 10.5× |
| prefill 512 | 1008 tok/s | 941 | 67.2 | 15.0× |
| prefill 2048 | 860 tok/s | 623 | 64.1 | 13.4× |
| prefill 8192 | 551 tok/s | — | — | — |
| decode | **14.6 tok/s** | 8.89 | 1.13 | 12.9× |

With `--flash-prefill` (looser contract, see below):

| | flash tier | exact | vs CPU |
|---|---:|---:|---:|
| prefill 512 | **1173 tok/s** | 1008 | 17.5× |
| prefill 2048 | **1624 tok/s** | 860 | 25.3× |
| prefill 8192 | **1651 tok/s** | 551 | — |
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

### Prefill attention: what was tried, and why it did not help

Attention is the dominant prefill cost at depth — 49% at T=2048 and **70% at
T=8192** — so it was the obvious target. Three restructurings were implemented
and measured, all of them bitwise-identical to the kernel they replaced (same
keys, same order, same per-key sub-group reduction, same online-softmax
updates; only the memory source and the scheduling changed):

| variant | attn @ 2048 | attn @ 8192 |
|---|---:|---:|
| untiled (baseline) | 1.176 s | 10.623 s |
| tiled through local memory | 1.202 s | **10.391 s** |
| tiled, flat loader with `e/D`, `e%D` | 1.447 s | — |
| tiled, reductions batched ahead of the softmax | 2.029 s | — |

Two findings worth keeping:

* **Integer division is not free.** The first tile loader indexed elements
  flat and computed `e / D`, `e % D` and `(jt+jj) % cap` per element. Xe has no
  hardware integer divide, and that alone made the tiled kernel *slower* than
  the untiled one. Hoisting to one division per thread recovered all of it.
* **Attention here is not bandwidth-bound.** Staging K/V in local memory is the
  textbook fix and it bought 2% at 8192 and nothing at 2048 — the L2 was already
  absorbing the reuse. Batching the reductions ahead of the softmax (they are
  independent; only the softmax updates are ordered) was worse still, because
  the local-memory round trip costs more than the pipelining saves.

The limiter is the **cross-lane reduction per (query, key) pair**, which the
exact-arithmetic contract requires: one sub-group reduction per key is what
reproduces the twin's dot product. No amount of staging or reordering removes
it.

What does remove it is computing `S = Q·Kᵀ` and `P·V` on the matrix engines —
which is the `--flash-prefill` tier below.

The tiled kernel stays the default for the exact tier: bitwise-equivalent,
marginally better at depth. `MUSE_GPU_ATTN=plain` selects the untiled one.

## `--flash-prefill`: the matrix-engine attention tier

Opt-in, prefill only, and **a different numerical contract** — which is exactly
why it is opt-in.

The exact kernel folds keys into the running softmax one at a time; that per-key
schedule is what reproduces the twin, and it is also what forces a cross-lane
reduction per (query, key). A matrix engine can only be used if a whole tile of
scores is produced at once, so this tier takes the max and the sum over a
**tile** of keys and rescales the output accumulator once per tile instead of
once per key. Same function, different summation schedule. It cannot be bitwise
against the eager twin *or* the flash twin and is gated on the logit envelope —
asserting bitwise here would be asserting something no fused kernel can meet,
which is the trap the eager/flash twin split already exists to avoid.

Structure, per (query tile, key run):

1. `S[heads, rows, keys] = Q · Kᵀ` — one batched oneDNN matmul. Every query head
   in a shard reads the same KV head, so K broadcasts across the batch and all
   16 heads go in one call. (Startup refuses the tier if a shard would span more
   than one GQA group.)
2. a SYCL kernel applies the scale and the causal/window mask, folds the tile
   into the running `(max, sumexp)`, rescales the accumulator, and emits the
   bf16 probability tile;
3. `O += P · V` — a second batched matmul with a `sum` post-op, so the tile
   folds into the accumulator without a separate add.

Key runs are split at the ring's wrap point: a run that crossed it would leave
the keys non-contiguous and the GEMM's strides would quietly lie.

### Measured, 30B, two cards

| | attention | prefill |
|---|---:|---:|
| T=2048 exact | 1.181 s | 856 tok/s |
| T=2048 **flash** | **0.073 s** (16×) | **1624 tok/s** |
| T=8192 exact | 10.333 s | 551 tok/s |
| T=8192 **flash** | **0.513 s** (20×) | **1651 tok/s** |

Better than the 13× the GEMM-only floor predicted, because the tier skips the
masked half of the square that the floor measurement included. Attention falls
from 70% of prefill to 10%, and prefill stops degrading with depth — 1624 tok/s
at 2048 against 1651 at 8192.

Envelope against the bf16 twin on the 30B (prefill 128 + 8 decode steps): all 8
generated tokens identical, argmax equal, top-64 overlap 98.4%, max abs logit
difference 0.156 — against 0.094 for the exact tier, so the looser schedule
costs about one extra bf16 ulp.

Decode is untouched: it has one query and nothing to batch, so it keeps the
exact kernel and the exact contract.

## DFlash drafter

`--assistant DIR` binds the block-diffusion drafter onto the same shards as the
target and runs one drafting round after prefill: `block_size - 1` = **15
proposals** from a block of an anchor plus 15 MASK tokens.

Verified against the **f64 oracle**, not just the twin — on the tiny model and
on the real 30B, all 15 proposed tokens identical, on one card and on two.

The drafter is a different model with conventions that are each a way to be
quietly wrong, so they get their own code rather than flags on the target's:

* its RMSNorm rounds **before** the weight multiply (`w · bf16(x·rs)`, against
  the target's `bf16(x·rs·w)`), and every one of its norms is **plain** —
  including the ones whose names match the target's zero-centered sandwich
  norms;
* attention is **bidirectional** (no causal mask at all, only
  `|q_pos − kv_pos| ≤ sliding_window`) with an ordinary materialized
  max/exp/sum rather than an online update;
* the block's tokens are embedded with the target's **raw** table, and the head
  is the target's **bare** `lm_head` — no output multiplier, no softcap;
* q/k norms carry weights, unlike the target's weight-less pair.

The target's hidden states at `target_layer_ids` are captured on the way
through the forward pass (after each layer's final residual) rather than
recomputed. `h` is replicated across shards, so every shard captures its own
tap and the taps need no exchange.

`encoder.fc` is `[H, taps·H]`. Concatenating the taps into an `[n, taps·H]`
buffer is the obvious reading of the reference and it grows with the prompt, so
instead the fc is kept as `taps` separate `[H, H]` blocks and accumulated —
each block column-sharded on its input, one all-reduce at the end.

### Speculative decoding (`--draft-rounds N`)

Per round: one drafter forward proposes 15 tokens, then **one** target forward
over `[bonus ++ candidates]` verifies them all at once. Acceptance is HF's
ordinary assisted-decoding rule. Unlike the oracle's loop, which re-runs the
whole target forward over the full candidate sequence each round, this keeps
the KV cache and forwards only the new rows — rolling back after a partial
accept is just `len_`, since the rejected positions get overwritten next round.
Gated on producing the **same sequence** as the oracle's loop, which is what
actually checks the rollback.

30B, BF16, two cards, 20 rounds:

| prompt | tokens/round | tok/s | vs plain decode (14.76) |
|---|---:|---:|---:|
| fact | 2.85 | 33.4 | 2.3× |
| **code** | **11.0** | **119.9** | **8.1×** |
| prose | 2.15 | 25.2 | 1.7× |

Acceptance is what decides it, and it is extremely prompt-dependent: code
drafts 11 of 15 tokens per round, prose barely 2.

Cost per round is **draft 16 ms, verify 75 ms** — the 5-layer drafter costing
~20% of the 52-layer target, which is the right proportion. Getting there took
one fix worth recording: the taps were first stored **f32**, making the context
projection an `f32 × bf16` matmul, which oneDNN can only serve from a reference
kernel. That single GEMM was **435 ms per round — 98% of the drafter's cost, and
1000× its bandwidth budget**. The target's hidden states only ever hold
bf16-representable values (every residual rounds), so storing the taps as bf16
row-major is exact and makes it an ordinary DPAS GEMM: 8.696 s → 0.026 s over 20
rounds, and speculative decoding went from *slower* than plain decode to 8×
faster.

A second, smaller one: oneDNN JITs per shape, and the drafter's context length
grows every round, so every round was a fresh JIT. Row counts are now rounded up
to power-of-two buckets (`row_bucket`), which collapses that to ~log₂(max_seq)
shapes. The extra rows are computed and discarded — free of consequence, since
GEMM rows are independent, KV rows past the live length are never read, and
context rows past `n` are never copied forward.

### Memory

| | per card |
|---|---:|
| target, BF16, TP | 27.36 GiB |
| + drafter, BF16, sharded | **29.80 GiB** |
| + drafter, Q8_0 (`--q8-assistant`) | **28.75 GiB** |

`--q8-assistant` quantizes the drafter independently of the target — the build
plan's recommended shape. Its dimensions all divide by 32, so it quantizes
cleanly: 4.89 → **2.79 GiB**, with *identical* drafted tokens and an identical
acceptance pattern on the test prompts. It costs throughput, though, because
the drafter's GEMMs are only 16 rows wide and go through the same
dequantize-to-scratch path prefill uses — reading the weight, writing bf16, and
reading it back, i.e. 3× the traffic of just reading bf16:

| drafter tier | VRAM | draft/round | spec tok/s (code) |
|---|---:|---:|---:|
| BF16 | 4.89 GiB | 16.4 ms | **119.9** |
| Q8_0 | **2.79 GiB** | 48.0 ms | 89.3 |

A dedicated small-M Q8 GEMM (one sub-group per output row, accumulating all 16
activation rows from a single pass over the weight) would read 1.06 bytes per
weight instead of expanding it, and should make the Q8 drafter *faster* than
BF16 rather than slower. Not written.

Against 31.89 GiB physical, so BF16 target **and** BF16 drafter do fit two
cards — which the build plan predicted would not happen (it budgeted 2.38
GiB/card for the drafter and expected the margin to run out). Sharding the
drafter the same way as the target is what buys it. It is still tight, and a
quantized drafter remains the comfortable configuration.

## Vision tower

`--pixels FILE --grid t,h,w` runs the 50-layer ViT on the cards and scatters its
output at the image/video placeholder tokens, after the embedding norm.

| 1024 patches, 30B tower | |
|---|---:|
| CPU (bf16, 32 threads, AVX-512) | 11.391 s |
| **GPU (2 cards)** | **0.547 s** |
| | **20.8×** |

Everything that is an **index** rather than an arithmetic result is computed on
the host by the already-gated code in `vision.hpp` — the window permutation, the
`cu_seqlens` segment bounds, the bilinear position taps, the 2-D rope table, the
pixel-shuffle source map. Those are the parts carrying the fiddly conventions
(the w/h flip and +1, the `[fw|fh|fw|fh]` frequency layout, the channel-major
merge), and reimplementing them on the device would be reimplementing the traps.
The device does arithmetic only.

Two things the tower does that the text stack does not, and both are separate
kernels rather than flags:

* **LayerNorm, not RMSNorm** — it subtracts a mean and carries a bias.
* **Every projection has a bias.** On the column-sharded ones (`o_proj`,
  `fc2`) the bias is added *after* the all-reduce; adding it per shard would add
  it `nshard` times.

### Accuracy: read this against the twin, not the oracle

The tower in bf16 is genuinely noisy — 50 LayerNorm layers accumulate — and the
right reference is the CPU bf16 twin's own deviation, not zero. On the 30B with
80 patches:

| | max abs | cos (min per row) |
|---|---:|---:|
| CPU bf16 twin vs f64 oracle | 3.386 | 0.9668 |
| **GPU vs f64 oracle** | **3.347** | **0.9808** |
| GPU vs CPU bf16 twin | 1.875 | 0.9965 |

The GPU is *closer to the oracle than the CPU twin is*, so it has introduced
nothing beyond bf16. GPU and twin also agree on the top-3 ordering where the f64
oracle differs — bf16 flips a near-tie, on both of them, the same way. At 1024
patches, GPU vs twin is cosine 0.9843 min / 0.9990 mean.

The gate encodes exactly this: the GPU's distance from the **oracle** must not
exceed 1.5× the **twin's** distance from the oracle. Comparing GPU to twin
directly and calling the difference an error would be comparing two reduction
orders and blaming one of them.

## VRAM

The footprint is **static and checked**. Everything is allocated during
construction and `bind_*`; `--seal N` then arms a tripwire (1 = log, 2 =
refuse) so any later device allocation is a startup-shaped failure rather than
a mid-request OOM. Verified: a full prefill + decode after `--seal 2` reports
no violations.

What is already saving VRAM:

* **Sliding-window KV ring.** The 39 sliding layers hold `sliding_window +
  chunk` rows, not `max_seq` — their cache is independent of context length.
  Only the 13 global layers grow, and at 2 KV heads that is 1 KiB per layer per
  token, so the full 131 072 window costs 1.71 GiB/card.
* **No S-sized attention scratch on the text path.** The exact kernels carry the
  online softmax in registers, so there is no per-work-group score buffer and
  therefore no SLM-bound context ceiling. (gemma4 hit exactly that: their global
  attention keeps all `S` scores in 128 KiB of SLM and aborts past ~32 K. The
  `--flash-prefill` tier does materialize a tile, but a bounded one — `FBQ x
  FBK`, not `S`.)
* **`--vision cpu`.** The tower is ~1.86 GiB/card once sharded. Running it on
  the CPU keeps that out of VRAM entirely, and the CPU path is the
  bitwise-gated one:

  | | tower forward | VRAM |
  |---|---:|---:|
  | `--vision gpu` | **0.545 s** | 29.24 GiB/card |
  | `--vision cpu` | 13.071 s | **27.37 GiB/card** |

  For one image per request against a long generation, trading 12.5 s for 1.86
  GiB is often the right way round; for image-heavy serving it is not. Both are
  one flag.

Two techniques from gemma4 that do **not** transfer, checked rather than
assumed:

* **`k = v` dedup.** Muse Glimmer's `k_proj` and `v_proj` are different tensors
  (max abs difference 6.8e-2 on layer 0), so there is nothing to dedup. It would
  not be worth much anyway: 2 KV heads make the cache tiny to begin with.
* **oneDNN Q8 matmul.** See the Q8 section — grouped scales are unsupported in
  this build.

## How much context fits

Measured end to end — prefilled and decoded, not just allocated. `--flash-decode`
is required past ~2K (see below); `--ids-file` reads long prompts, which exceed
the argv limit.

| configuration | max context | prefill | decode @ full |
|---|---:|---:|---:|
| BF16 | **131 072** (the model's max) | 1143 tok/s | 5.67 tok/s |
| Q8_0 | **131 072** | 946 tok/s | 5.70 tok/s |
| BF16 + BF16 drafter | ~16 384 | — | — |
| BF16 + Q8_0 drafter | ~65 536 | — | — |
| Q8_0 + Q8_0 drafter | **131 072** | — | — |

So **the 30B runs its full 131 072-token window on two cards, on either tier** —
114.7 s to prefill 131 072 tokens. The KV cache is only 3.56 GiB there, because
39 of 52 layers are sliding and hold `window + chunk` rows regardless of context;
only the 13 global layers grow, at 1 KiB per layer per token.

**The drafter used to cap context, and it was the taps rather than its weights.**
DFlash reads the target's hidden states at 5 layers, and holding them for the
whole context costs `5 × max_seq × H × 2` bytes per shard — 4.4 GB at 64 K.

But the drafter's own attention is sliding-window 2048 and its block sits at
`pos0 == n`, so any context position older than `n − window` is masked out and
contributes **exactly zero**. Truncating to that window is therefore
arithmetically identical to the reference, not an approximation — and it lets
the taps live in a `window + chunk` ring exactly like the KV cache, at a
**constant 272 MB per shard** regardless of context.

Two consequences of the truncation, both load-bearing: a key ROW is no longer
its absolute position, so the drafter's rope and its attention mask both take a
`kbase = n − W` offset. And the window is gathered out of the ring into a
contiguous buffer before the context GEMM — reading the ring directly would
split every round's range at the wrap point into runs whose lengths change every
round, and oneDNN JITs per shape, so that just trades the memory back for JIT
churn.

That moved the ceiling from ~8 K to 16 K; sizing the Q8 dequant scratch by which
tier is *actually* quantized (a Q8 drafter never expands the vocab head — that
is the target's) took BF16+Q8 to 64 K, and Q8+Q8 now reaches the **full 131 072
window with a drafter attached**.

## Decode falls off with context unless you split the keys

At decode there is ONE query, so the exact attention kernel launches `nqs`
sub-groups — 16 of ~256 Xe-cores — and each walks the whole key range serially.
Cost is linear in context and the card is idle. Measured as **72-82% of decode
time** past 2 K.

`--flash-decode` splits the key range across work-groups, each producing a
partial `(max, sumexp, accumulator)`, then merges them. Parallelism goes from 16
to 16 × splits:

| context | exact | `--flash-decode` |
|---|---:|---:|
| 2 048 | 4.48 tok/s | **10.27** |
| 8 192 | 2.89 | **10.19** |
| 16 384 | 1.95 | **10.05** |
| 32 768 | — | 9.05 |
| 131 072 | — | 5.67 |

Decode goes from degrading linearly to **flat out to 16 K**. It is the same
class of contract as `--flash-prefill` — reordering the softmax rescalings is a
different schedule of the same function — so it is envelope-gated, not bitwise,
and therefore opt-in. It produced identical tokens to the exact path on the test
prompts.

The split count is chosen from the context length alone, never from a runtime
measurement, so a rerun cannot pick differently.

## Are the matrix engines actually used?

Yes. oneDNN reports `jit:gemm:any` for the prefill shapes, which names the
dispatcher rather than the kernel — but **173 TFLOP/s bf16** is not reachable
without XMX/DPAS (the vector ALUs would give roughly 20-40), so the systolic
path is what runs. The `--flash-prefill` tier puts attention's `Q·Kᵀ` and `P·V`
on the same engines, which is where its 16-20x comes from.

## Q8_0 weight tier (`--q8`)

Quantized at load from the same BF16 checkpoint with llama.cpp's
`quantize_row_q8_0_ref` semantics (per 32 elements: `d = amax/127` in f16,
`q = round(x/d)`). Quants and scales are kept in **separate** arrays rather than
llama.cpp's interleaved 34-byte block, so the GEMV reads a whole row of quants
contiguously and touches scales 32× less often.

| 30B, 2 cards | VRAM/card | decode | prefill |
|---|---:|---:|---:|
| BF16 | 27.36 GiB | 14.76 tok/s | 635 tok/s |
| **Q8_0** | **15.71 GiB** | 15.07 tok/s | 245 tok/s |
| Q8_0, `MUSE_GPU_Q8_BLOCKDOT=1` | 15.71 GiB | **21.10 tok/s** | 245 tok/s |

**1.74× less weight VRAM**, and it generates the same 16 tokens as BF16 on the
test prompt.

### Prefill is slower, and that is oneDNN's doing

oneDNN on this stack has **no grouped-scale path at all** — measured, not
assumed:

| configuration | accepted? |
|---|---|
| bf16 × s8, per-32 grouped scales (f16 or f32) | **no** |
| f16 × s8, per-32 grouped scales | **no** |
| bf16 × s8, per-column scales | yes, but `ocl:ref` (the reference kernel) |
| bf16 × s8, grouped, transposed strides | no (`unsupported format tag`) |

Q8_0 needs per-32 scales by construction, so the matrix engines cannot consume
it directly. Prefill therefore expands each weight into a BF16 scratch tile and
runs the ordinary DPAS GEMM on it — which costs a write and a re-read of the
tile, hence 245 vs 635 tok/s. **Q8 here is a memory-and-decode tier, not a
prefill one.**

### Two dot arithmetics, and the choice is real

llama.cpp's Q8_0 dot sums a block of products and scales once at the end. The
prefill path *cannot* do that — it materializes `bf16(q·d)` per element so the
matrix engines can run. Taking the fast form at decode makes decode disagree
with prefill **inside one tier**, so a token this engine generated would not be
reproduced by prefilling the same sequence. The default therefore matches
prefill and is chunk-invariant; `MUSE_GPU_Q8_BLOCKDOT=1` selects llama.cpp's
semantics and buys 40% of decode back.

(`BLOCKDOT` is a template parameter, not a flag. As a runtime branch inside the
unrolled inner loop it cost **12×** — 16.2 → 1.31 tok/s — because both arms stay
in the loop body and the accumulators spill.)

### The bug worth remembering

The first version hand-rolled `f32_to_f16` and **flushed subnormals to zero**.
That is not a corner case: up to **1.25% of Q8_0 blocks** in this checkpoint
have a scale below f16's smallest normal (6.1e-5), so ~1% of every weight matrix
was silently zeroed. The model still ran, still put the right token first, and
still looked plausible — it had just lost most of its accuracy (logits differed
from BF16 by **9.98 on a scale of 11**, top-64 overlap 41%). Using the
compiler's `_Float16` conversion instead fixed it: max 0.31, top-64 89%, and the
generated tokens became identical to BF16.

Gating is against the **f64 oracle**, never the bf16 twin bitwise: 8-bit weights
are a separate accuracy tier, roughly 3× wider than bf16, and asserting the twin's
band would be asserting a contract Q8 is not trying to meet.

### Constraint

Every quantized matrix needs its input dimension divisible by 32. The tiny gate
model must therefore run Q8 at `--shards 1`: at 2 shards its `o_proj` column
slice is 16 wide. The 30B's slices are 2048 and 9984, both fine.

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
| flash tier inside the twin's envelope | the looser tier is still the same function |
| flash tier differs from the exact tier | the tier is actually active |
| flash tier rerun + 1 card == 2 cards, bitwise | looser numerics, not sloppy ones |
| drafter proposals == the f64 oracle's | the drafter's four inverted conventions |
| drafter: 1 card == 2 cards | the drafter shards like the target |
| spec sequence == the f64 oracle's | the accept rule and the cache rollback |
| spec loop rerun deterministic | oneDNN's atomic split-K, which a single-forward gate cannot see |
| flash-decode inside the exact path's envelope + rerun bitwise | split-K is a schedule, not a different function |
| vision features within the twin's own deviation of the oracle | the tower, judged against bf16 rather than against zero |
| q8 argmax + top-k vs the f64 oracle | a separate accuracy tier, gated on its own terms |
| q8 chunk-invariant + rerun bitwise | decode and prefill agree inside the tier |
| image+text logits argmax + top-20 | the features land at the right token positions |

`MUSE_GPU_TRACE=<dir>` dumps the residual stream per layer (row-major `[n, H]`),
which is how the oneDNN bug was localized to the head. `MUSE_GPU_PROFILE=1`
prints per-stage attribution; it inserts a queue wait per stage, so it changes
what it measures and is a diagnostic, not a throughput report.
