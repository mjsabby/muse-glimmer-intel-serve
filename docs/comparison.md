# Head to head with llama.cpp on the same two cards

Same model, same hardware, same shapes, **both stacks in their best
configuration**. `llama-bench` is the reference harness and `tools/bench.py`
copies its semantics exactly: `ppN` prefills N tokens into an empty cache,
`tgN` generates N tokens, `-d D` runs either against a cache already holding D
tokens, and both report mean ± stddev of tokens per second.

```
llama.cpp  b10480 (01818e495), SYCL backend, icpx 2026.0
           -ngl 99 -sm tensor -fa on -b 4096, -ub 1024 (BF16) / 2048 (Q8_0)
           unsloth/Muse-Glimmer-30B-GGUF (BF16 51.89 GiB / Q8_0 27.57 GiB, text only)
ours       --flash-prefill --flash-decode, chunk 2048, safetensors, vision off
hardware   2x Intel Arc Pro B70, 31.89 GiB each
```

Both of those flag sets are the *other* side's best, found by sweeping rather
than assumed. `-sm tensor` is not llama.cpp's default (that is `-sm layer`,
which pipeline-splits) and it is much faster here; `-ub 1024`/`2048` beat its
default 512 by 3% and 18%. Comparing against a default nobody would run is not
a comparison.

| llama.cpp configuration | pp512 | tg32 |
|---|---:|---:|
| `-sm layer -fa off` (near-default) | 1485 | 9.67 |
| `-sm layer -fa on` | 1625 | 9.71 |
| **`-sm tensor -fa on`** | **1871** | **16.41** |

## BF16, two cards

| test | llama.cpp | ours | ours / llama.cpp |
|---|---:|---:|---:|
| pp512 | 1864.30 ± 9.88 | 1885.61 ± 44.55 | **1.01** |
| pp2048 | 1831.20 ± 1.70 | 2077.62 ± 7.67 | **1.13** |
| pp8192 | 1726.31 ± 0.07 | 1966.92 ± 2.78 | **1.14** |
| tg128 | 16.38 ± 0.00 | 17.66 ± 0.01 | **1.08** |
| pp512 @ d4096 | 1619.25 ± 14.92 | 1748.22 ± 34.08 | **1.08** |
| pp2048 @ d4096 | 1686.41 ± 1.04 | 1915.98 ± 3.39 | **1.14** |
| pp8192 @ d4096 | 1638.99 ± 0.68 | 1871.89 ± 2.73 | **1.14** |
| tg128 @ d4096 | 16.12 ± 0.00 | 17.13 ± 0.02 | **1.06** |
| pp512 @ d16384 | 1240.26 ± 11.34 | 1573.26 ± 27.45 | **1.27** |
| pp2048 @ d16384 | 1490.71 ± 1.08 | 1734.63 ± 6.60 | **1.16** |
| pp8192 @ d16384 | 1452.06 ± 3.31 | 1695.45 ± 2.34 | **1.17** |
| tg128 @ d16384 | 15.90 ± 0.01 | 16.93 ± 0.07 | **1.06** |

**Twelve tests out of twelve.** 1.01x at the shallowest prefill and 1.27x where
attention rather than the GEMMs sets the pace.

## Q8_0, two cards

llama.cpp reads unsloth's Q8_0 GGUF; this engine quantizes the same BF16
weights at load with `quantize_row_q8_0_ref` semantics, so the weight *bytes*
are the same format.

| test | llama.cpp | ours | ours / llama.cpp |
|---|---:|---:|---:|
| pp512 | 1378.69 ± 4.66 | 789.29 ± 9.67 | 0.57 |
| pp2048 | 1622.36 ± 1.55 | 1498.88 ± 7.03 | 0.92 |
| pp8192 | 1504.32 ± 1.54 | 1443.12 ± 0.50 | 0.96 |
| tg128 | 26.48 ± 0.01 | 26.71 ± 0.04 | **1.01** |
| pp512 @ d4096 | 724.93 ± 13.87 | 763.84 ± 3.09 | **1.05** |
| pp2048 @ d4096 | 1242.66 ± 0.72 | 1429.89 ± 0.59 | **1.15** |
| pp8192 @ d4096 | 1209.95 ± 0.66 | 1395.76 ± 2.24 | **1.15** |
| tg128 @ d4096 | 25.79 ± 0.01 | 25.58 ± 0.02 | 0.99 |
| pp512 @ d16384 | 633.30 ± 8.47 | 733.19 ± 1.76 | **1.16** |
| pp2048 @ d16384 | 1135.74 ± 4.94 | 1317.67 ± 1.74 | **1.16** |
| pp8192 @ d16384 | 1101.76 ± 0.49 | 1295.13 ± 1.31 | **1.18** |
| tg128 @ d16384 | 25.27 ± 0.02 | 25.16 ± 0.06 | 1.00 |

Decode is a dead heat (1.01x, 0.99x, 1.00x — inside the run-to-run spread),
prefill at depth is ours by 1.15-1.18x, and **prefill into an empty cache is
llama.cpp's**: 0.57x at 512 tokens, 0.92x at 2048.

That last row is one structural difference, and it is not mysterious. This
engine dequantizes each Q8 weight into a BF16 scratch and runs the ordinary
DPAS GEMM, which costs an extra HBM round trip **per block** — read q8, write
bf16, read bf16, or about 2.5x the weight traffic of a fused path. llama.cpp
fuses the dequantization into its matmul. The cost is per block and the benefit
is per token, so it disappears as the block fills (pp512 0.57x, pp2048 0.92x,
pp8192 0.96x) and is more than repaid once attention dominates. Two ways out,
both identified and neither started: a fused int8→bf16 GEMM feeding DPAS, or
layer-major prefill (loop layers outside blocks, so a weight is dequantized
once per prefill rather than once per block).

## Speculative decoding, which llama.cpp cannot do here

DFlash is a trained companion network; llama.cpp has no drafter for this model,
so this is a capability difference rather than a race.

| | tok/s | accept |
|---|---:|---:|
| plain decode, BF16, 2 cards | 17.66 | — |
| **+ Q8 DFlash drafter, on code** | **72.67** | 0.39 |
| the same on a list of primes | 97 | 0.56 |
| the same on prose | 31 | 0.13 |

4.1x on code at BF16 weights, against llama.cpp's best decode of 26.48 at Q8.

## What the sweep found

Benchmarking against another implementation is worth doing precisely because it
turns "this seems fine" into a number with something to compare it to. This one
produced four fixes, in the order they were found.

**1. Split-K decode slices were four times too large.** One sub-group runs one
slice, so the slice width *is* the occupancy: at 16 query heads a 512-key slice
means 128 work-groups of 32 lanes, about 6% of the card. Decode at 16384 went
10.02 → 14.85 tok/s. (Retuned later to 128-key slices once the kernel stopped
being latency-bound; see below.)

**2. The peer exchange was serialized.** It had been, deliberately, because
concurrent pulls were bimodal on this driver — 0.56 ms one run and 639 ms the
next. Re-measured over 200 reps at 1/6/13/26 MiB that pathology is gone and
concurrency is now the *steadier* of the two (13 MiB/leg: median 0.551 ms, max
0.555, against serialized 0.963 / 5.862). It matters because the all-reduce is
33% of prefill. +17% prefill, same bytes.

**3. The attention accumulators lived in private memory.** `na` — head_dim over
the sub-group width — was counted by a runtime loop, and an array indexed under
a runtime bound cannot be held in registers, so every key paid a
private-memory round trip. Making it a compile-time constant is the same
arithmetic in the same order (the bitwise gates still pass) and was worth +7.5%
on decode at 16384, more than every scheduling change tried around it. With the
slices retuned afterwards: 2048 → 16.71, 4096 → 16.75, 16384 → 16.42, 65536 →
15.65 tok/s.

**4. The bf16 conversion helpers were the entire Q8 story.** `f2bf()` was a
memcpy, a NaN branch and three integer ops; `bf2f()` a shift through a memcpy.
That is nothing in a kernel that runs it once per output and it is the whole
workload in one that runs it once per weight *byte*. Replacing both with
`sycl::ext::oneapi::bfloat16` — the same round-to-nearest-even, one instruction
— took the Q8 decode GEMV from 235 to 500 GB/s in isolation and 332 to 513
GB/s in the engine, and Q8 decode from 18.7 to 26.9 tok/s.

Finding it is the part worth recording. The kernel measured ~500 GB/s in a
standalone harness and ~332 in the engine, and the following were each measured
and each *not* the cause: rows per work-group, vector versus scalar loads, the
compiler's FP flags, per-kernel device-code split, the size of the working set,
the number of separate allocations, VRAM pressure (28 GiB of ballast changed
nothing), sustained-throughput throttling, interleaved small kernels,
producer-consumer dependencies on the activation buffer, two cards running
concurrently, and the all-reduce. What settled it was lifting the engine's
kernel *verbatim* into the standalone harness — 389 GB/s there, against 559 for
a reimplementation I had assumed was equivalent — and then bisecting the
difference between the two texts. `rb()` alone was 2x.

The lesson is the one this repo keeps relearning: a microbenchmark of "the same
kernel" is only evidence if it is the same *text*.

**5. Prefill chunk 512 → 2048** (default changed). BF16 prefill +11%, Q8
prefill +91% — the Q8 tier dequantizes per block, so a wider block amortizes
it. ~0.8 GiB/card more scratch.

**6. The all-reduce no longer host-syncs.** The pack's event is a cross-queue
dependency for the transport instead: 19.0 µs per all-reduce becomes 11.3, and
there are 104 of them per decode token.

## Reproducing

```bash
# ours
.venv/bin/python tools/bench.py --gpus 2 -p 512,2048,8192 -n 128 -d 0,4096,16384 -r 3
.venv/bin/python tools/bench.py --gpus 2 --q8 -p 512,2048,8192 -n 128 -d 0,4096,16384 -r 3

# llama.cpp, each tier at its best ubatch
llama-bench -m Muse-Glimmer-30B-BF16-00001-of-00002.gguf \
    -p 512,2048,8192 -n 128 -d 0,4096,16384 -r 2 -ngl 99 -sm tensor -fa on -b 4096 -ub 1024
llama-bench -m Muse-Glimmer-30B-Q8_0.gguf \
    -p 512,2048,8192 -n 128 -d 0,4096,16384 -r 2 -ngl 99 -sm tensor -fa on -b 4096 -ub 2048
```

Both harnesses were run with the other stack's process not resident: 55 GiB of
weights leaves no room for a second copy, and a run that quietly falls back to
a smaller allocation is not a measurement. `ZES_ENABLE_SYSMAN=1` for free-VRAM
reporting on both sides.

`MUSE_GPU_KTIME=1` reports device-side timings of the Q8 decode GEMVs, per
shape — built for finding #4 above, and kept because "fast in isolation, slow
in the engine" is a question that deserves timestamps rather than subtraction.
