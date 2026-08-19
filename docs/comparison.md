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
| pp512 | 1378.69 ± 4.66 | 1317.52 ± 12.11 | 0.96 |
| pp2048 | 1622.36 ± 1.55 | 1859.65 ± 8.87 | **1.15** |
| pp8192 | 1504.32 ± 1.54 | 1764.49 ± 2.82 | **1.17** |
| tg128 | 26.48 ± 0.01 | 26.83 ± 0.07 | **1.01** |
| pp512 @ d4096 | 724.93 ± 13.87 | 1239.33 ± 13.96 | **1.71** |
| pp2048 @ d4096 | 1242.66 ± 0.72 | 1727.46 ± 5.17 | **1.39** |
| pp8192 @ d4096 | 1209.95 ± 0.66 | 1688.55 ± 3.54 | **1.40** |
| tg128 @ d4096 | 25.79 ± 0.01 | 25.59 ± 0.07 | 0.99 |
| pp512 @ d16384 | 633.30 ± 8.47 | 1145.88 ± 12.45 | **1.81** |
| pp2048 @ d16384 | 1135.74 ± 4.94 | 1574.21 ± 3.86 | **1.39** |
| pp8192 @ d16384 | 1101.76 ± 0.49 | 1548.09 ± 3.05 | **1.40** |
| tg128 @ d16384 | 25.27 ± 0.02 | 25.12 ± 0.10 | 0.99 |

Nine of twelve ours, up to 1.81x; decode is a tie within ±1%; and the one
llama.cpp still holds is prefilling 512 tokens into an empty cache, at 0.96x.

**The fix was not the one that had been planned.** This document previously
said the Q8 prefill gap needed a fused int8→bf16 GEMM or layer-major prefill,
because the tier dequantizes each weight into a BF16 scratch per block. The
extra round trip is real, but it was not the cost. Timing the two halves
separately said the dequantize pass was running at **108 GB/s** against a card
that streams 550, and the reason was the same one the decode GEMV had already
been caught by: it read the int8 quants **one byte at a time**. Two 16-byte
vector loads and four 16-byte stores later it runs at 320 GB/s, prefill went

| | before | after |
|---|---:|---:|
| pp512 | 789 | **1318** |
| pp2048 | 1499 | **1860** |
| pp8192 | 1443 | **1764** |

and the dequantize pass went from 74% of Q8 prefill to 50%. Two blocks per
work-item measured the same 316 GB/s, so at a 1:2 read:write ratio that is the
pattern's rate rather than the thread count's — the remaining round trip is
what pp512 is still paying, and the fused GEMM is now worth roughly the 4%
that separates it from llama.cpp rather than the 43% it looked like.

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
produced seven fixes, in the order they were found.

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

**5. The Q8 dequantize pass read bytes one at a time.** The same trap as #4 in
a different kernel, and the one that had been mistaken for a structural
limitation: 108 GB/s where the card streams 550, because
`ob[k] = f2bf(float(wb[k]) * sc)` loads one int8 per iteration. Two 16-byte
vector loads and four 16-byte stores take it to 320 GB/s, Q8 pp512 from 789 to
1318 tok/s and pp2048 from 1499 to 1860. The lesson generalizes: every kernel
in this engine that touches quantized weights element-wise is worth checking
for the access width before anything cleverer is designed for it.

**6. Prefill chunk 512 → 2048** (default changed). BF16 prefill +11%, Q8
prefill +91% — the Q8 tier dequantizes per block, so a wider block amortizes
it. ~0.8 GiB/card more scratch.

**7. The all-reduce no longer host-syncs.** The pack's event is a cross-queue
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
