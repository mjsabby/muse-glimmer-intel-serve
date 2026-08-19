# Head to head with llama.cpp on the same two cards

Same model, same hardware, same shapes, both stacks given their best
configuration. `llama-bench` is the reference harness and `tools/bench.py`
copies its semantics exactly: `ppN` prefills N tokens into an empty cache,
`tgN` generates N tokens, `-d D` runs either against a cache already holding D
tokens, and both report mean ± stddev of tokens per second over the
repetitions.

```
llama.cpp  b10480 (01818e495), SYCL backend, icpx 2026.0, -ngl 99 -sm tensor -fa on
           unsloth/Muse-Glimmer-30B-GGUF (BF16 51.89 GiB / Q8_0 27.57 GiB, text only)
ours       --flash-prefill --flash-decode, chunk 512, safetensors, vision off
hardware   2x Intel Arc Pro B70, 31.89 GiB each
```

`-sm tensor` is not llama.cpp's default (that is `-sm layer`, which
pipeline-splits by layer). It is used here because it is **much** faster on this
model and it is the honest comparison: it splits tensors across the cards, which
is what this engine does.

| llama.cpp split mode | pp512 | tg32 |
|---|---:|---:|
| `layer`, `-fa off` | 1485.40 | 9.67 |
| `layer`, `-fa on` | 1625.35 | 9.71 |
| **`tensor`, `-fa on`** | **1870.59** | **16.41** |

(`-sm tensor -fa off` and `-sm row` fail to create a context on this build.)

## BF16, two cards

| test | llama.cpp | ours | ours / llama.cpp |
|---|---:|---:|---:|
| pp512 | 1864.19 ± 1.10 | 1662.68 ± 7.09 | 0.89 |
| pp2048 | 1777.42 ± 1.69 | 1627.62 ± 2.36 | 0.92 |
| pp8192 | 1675.76 ± 0.52 | 1553.70 ± 1.03 | 0.93 |
| tg128 | 16.34 ± 0.00 | 16.12 ± 0.00 | 0.99 |
| pp512 @ d4096 | 1652.38 ± 16.10 | 1533.03 ± 3.38 | 0.93 |
| pp2048 @ d4096 | 1635.94 ± 0.56 | 1529.47 ± 0.58 | 0.93 |
| pp8192 @ d4096 | 1602.40 ± 0.17 | 1499.26 ± 2.25 | 0.94 |
| tg128 @ d4096 | 16.09 ± 0.02 | 15.70 ± 0.01 | 0.98 |
| pp512 @ d16384 | 1255.66 ± 17.38 | 1399.53 ± 3.44 | **1.11** |
| pp2048 @ d16384 | 1404.17 ± 2.18 | 1402.88 ± 1.37 | 1.00 |
| pp8192 @ d16384 | 1414.85 ± 0.84 | 1375.84 ± 0.23 | 0.97 |
| tg128 @ d16384 | 15.89 ± 0.00 | 14.79 ± 0.00 | 0.93 |
| pp2048 @ d65536 | 902.75 ± 0.74 | 1049.69 ± 1.87 | **1.16** |
| tg128 @ d65536 | 15.16 ± 0.00 | 13.00 ± 0.02 | 0.86 |

Parity, roughly: 7-11% behind on shallow prefill, ahead once the cache is deep
enough that attention rather than the GEMMs sets the pace, and within 2% on
decode until 16 K.

## Q8_0, two cards

llama.cpp reads unsloth's Q8_0 GGUF; this engine quantizes the same BF16
weights at load with `quantize_row_q8_0_ref` semantics, so the weight *bytes*
are the same format.

| test | llama.cpp | ours | ours / llama.cpp |
|---|---:|---:|---:|
| pp512 | 1377.68 ± 3.81 | 732.66 ± 6.35 | 0.53 |
| pp2048 | 1328.68 ± 1.45 | 728.40 ± 2.02 | 0.55 |
| pp8192 | 1268.70 ± 0.46 | 712.16 ± 0.90 | 0.56 |
| tg128 | 26.40 ± 0.01 | 16.28 ± 0.01 | 0.62 |
| pp512 @ d4096 | 735.13 ± 2.59 | 706.95 ± 0.38 | 0.96 |
| pp2048 @ d4096 | 734.13 ± 1.28 | 705.91 ± 0.62 | 0.96 |
| pp8192 @ d4096 | 727.50 ± 0.39 | 699.11 ± 0.52 | 0.96 |
| tg128 @ d4096 | 25.73 ± 0.02 | 15.88 ± 0.00 | 0.62 |
| pp512 @ d16384 | 643.17 ± 1.32 | 676.07 ± 1.61 | **1.05** |
| pp2048 @ d16384 | 679.45 ± 1.56 | 676.83 ± 0.82 | 1.00 |
| pp8192 @ d16384 | 685.38 ± 0.38 | 670.21 ± 0.36 | 0.98 |
| tg128 @ d16384 | 25.24 ± 0.01 | 14.96 ± 0.01 | 0.59 |

**This is where the engine loses, and the reason is structural rather than
incidental.** Two findings, both worth acting on:

* **Q8 decode is 1.6x slower.** The tell is that our Q8 decode (16.28) is the
  same speed as our BF16 decode (16.12) while reading half the bytes — so it is
  not weight-bandwidth-bound, and halving the weights bought nothing. llama.cpp
  goes 16.34 → 26.40 on the same change, which is what a bandwidth-bound decode
  does. Our Q8 GEMV runs at roughly 330 GB/s where the BF16 path reaches ~450+.
* **Q8 prefill is 1.9x slower at depth 0 — and at parity by 4 K.** The engine
  dequantizes each weight into a BF16 scratch and runs the ordinary DPAS GEMM,
  which costs an extra HBM round trip per block (read q8, write bf16, read
  bf16 ≈ 2.5x the weight traffic of a fused path). llama.cpp fuses the dequant
  into its matmul. Interestingly llama.cpp's own Q8 prefill *collapses* at depth
  (1378 → 735 at d4096) while ours barely moves (733 → 707), so the two meet.

The fix for both is the same one already written for the drafter: VNNI-packed
weights and the int8 DPAS kernel with per-32-block scales
([gpu.md](gpu.md#q8_0-weight-tier---q8)), extended from the drafter's ≤16 rows
to arbitrary M. That is the next optimization, not a mystery.

## Q8_0, one card

| test | llama.cpp | ours | ours / llama.cpp |
|---|---:|---:|---:|
| pp512 | 1090.13 ± 7.73 | 442.14 ± 2.19 | 0.41 |
| pp2048 | 1082.20 ± 0.79 | 325.74 ± 0.20 | 0.30 |
| tg128 | 16.25 ± 0.00 | 9.63 ± 0.01 | 0.59 |
| pp512 @ d4096 | 502.53 ± 0.78 | 207.33 ± 0.21 | 0.41 |
| pp2048 @ d4096 | 503.44 ± 0.47 | 198.72 ± 0.04 | 0.39 |
| tg128 @ d4096 | 15.88 ± 0.00 | 9.37 ± 0.00 | 0.59 |

The single-card gap is bigger than the Q8 gap alone, because **the
`--flash-prefill` tier is unavailable at one shard**. It broadcasts one KV head
across a shard's query heads, which is only valid if they all sit in one GQA
group: 32 query heads over 2 KV heads means 16 per group, so two shards land
exactly on the boundary and one shard straddles it. The engine refuses rather
than computing the wrong thing, and single-card prefill therefore runs the exact
attention tier.

`--gpus 1 --shards 2` would satisfy the constraint, and does not fit: two shards
each carry a Q8 dequantization scratch sized by the vocabulary head (1.34 GB),
and 27.7 GiB of weights plus 2.7 GB of scratch is over the card.

This engine is built for the two-card configuration. One card is a Q8-only
fallback and is documented as one.

## Speculative decoding, which llama.cpp cannot do here

DFlash is a trained companion network; llama.cpp has no drafter for this model,
so this is a capability difference rather than a race.

| | tok/s | accept |
|---|---:|---:|
| plain decode, BF16, 2 cards | 15.79 | — |
| **+ Q8 DFlash drafter, on code** | **72.67** | 0.39 |
| the same on a list of primes (serving gate) | 97 | 0.56 |
| the same on prose | 31 | 0.13 |

**4.6x on code, and it beats llama.cpp's Q8 decode (26.40) at BF16 weights.**
The measured range is wide because acceptance is a property of the text: highly
predictable output (lists, code scaffolding) accepts more than half the block,
prose accepts an eighth. What it is not allowed to do is change the answer, and
what it actually guarantees is in
[serving.md](serving.md#speculative-decoding-and-what-it-actually-guarantees).

## What the sweep found and fixed

Benchmarking against another implementation is worth doing precisely because it
turns "this seems fine" into a number with something to compare it to. This
round produced one fix immediately:

**Split-K decode attention was cutting the key range into slices four times too
large.** One sub-group runs one slice, so at 16 query heads a 512-key slice
means 8 slices and 128 work-groups of 32 lanes — about 6% of the card, with a
dependent chain per key and nothing to hide its latency behind. Cutting the
slice to 64 keys:

| context | 512-key slices | 64-key slices |
|---|---:|---:|
| 2 048 | 10.31 | **15.80** |
| 4 096 | 10.22 | **15.71** |
| 16 384 | 10.02 | **14.85** |
| 65 536 | 7.53 | **12.79** |

+53%, +54%, +48%, +70%. Against llama.cpp that moved BF16 decode at depth from
0.62x to 0.93-0.98x. Past ~256 slices the merge kernel — one work-group per
head, walking the partials serially — starts taking it back, which is where the
cap comes from.

## Reproducing

```bash
# ours
.venv/bin/python tools/bench.py --gpus 2 -p 512,2048,8192 -n 128 -d 0,4096,16384 -r 3
.venv/bin/python tools/bench.py --gpus 2 --q8 -p 512,2048,8192 -n 128 -d 0,4096,16384 -r 3
.venv/bin/python tools/bench.py --gpus 2 \
    --assistant meta-models/Muse-Glimmer-30B-assistant --q8-assistant \
    --spec-prompt "Write a Python function that merges two sorted lists." -r 2

# llama.cpp
llama-bench -m Muse-Glimmer-30B-BF16-00001-of-00002.gguf \
    -p 512,2048,8192 -n 128 -d 0,4096,16384 -r 3 -ngl 99 -sm tensor -fa on
```

Both harnesses were run with the other stack's process not resident: 55 GiB of
weights leaves no room for a second copy, and a run that quietly falls back to
a smaller allocation is not a measurement. `ZES_ENABLE_SYSMAN=1` for free-VRAM
reporting on both sides.
