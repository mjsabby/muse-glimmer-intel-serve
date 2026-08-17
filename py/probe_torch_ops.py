#!/usr/bin/env python
"""Pin down the torch CPU behaviours the f64 oracle has to reproduce bitwise.

Three questions, all of which change the oracle's inner loops:

  1. Does `torch.pow(x, -0.5)` equal `torch.rsqrt(x)` bitwise in f64?
     (`MuseGlimmerRMSNorm._norm` uses `torch.pow(ms, -0.5)`, the centered kind
     uses `torch.rsqrt` — the two must be either identical or handled apart.)
  2. What reduction order does `x.pow(2).mean(-1)` use in f64, i.e. which
     fixed-lane blocked sum reproduces it bitwise?
  3. Do the bf16/f16 storage casts go through f32 (double rounding)?

Run it against the pinned reference environment and record the answers in
VERIFICATION.md; the oracle's kernels are written against them.
"""

import itertools
import sys

import torch


def blocked_sum(x, lanes):
    """Sum with `lanes` interleaved accumulators, then a fixed pairwise tree,
    plus a sequential tail — the shape simd.hpp / the norm kernels use."""
    n = x.shape[-1]
    nb = (n // lanes) * lanes
    acc = [torch.zeros((), dtype=x.dtype) for _ in range(lanes)]
    for i in range(0, nb, lanes):
        for l in range(lanes):
            acc[l] = acc[l] + x[i + l]
    tail = torch.zeros((), dtype=x.dtype)
    for i in range(nb, n):
        tail = tail + x[i]
    # balanced tree over the lanes, then the tail
    cur = acc
    while len(cur) > 1:
        cur = [cur[i] + cur[i + 1] for i in range(0, len(cur), 2)]
    return cur[0] + tail


def sequential_sum(x):
    s = torch.zeros((), dtype=x.dtype)
    for v in x:
        s = s + v
    return s


def probe_pow():
    print("== 1. pow(x, -0.5) vs rsqrt(x), f64 ==")
    g = torch.Generator().manual_seed(7)
    for n in (1000,):
        x = torch.rand(n, generator=g, dtype=torch.float64) * 10.0 + 1e-8
        p = torch.pow(x, -0.5)
        r = torch.rsqrt(x)
        d = torch.ones_like(x) / torch.sqrt(x)
        print(f"   pow == rsqrt      : {bool((p == r).all())}")
        print(f"   pow == 1/sqrt     : {bool((p == d).all())}")
        print(f"   rsqrt == 1/sqrt   : {bool((r == d).all())}")
        # and in f32 (the stock, unlifted path)
        x32 = x.to(torch.float32)
        print(f"   f32 pow == rsqrt  : {bool((torch.pow(x32, -0.5) == torch.rsqrt(x32)).all())}")


def probe_mean(dims):
    print("== 2. x.pow(2).mean(-1) reduction order, f64 ==")
    g = torch.Generator().manual_seed(11)
    for n in dims:
        x = torch.randn(n, generator=g, dtype=torch.float64)
        ref = x.pow(2).mean(-1)
        sq = x * x
        cands = {}
        for lanes in (1, 2, 4, 8, 16):
            cands[f"blocked{lanes}"] = blocked_sum(sq, lanes) / n
        cands["sequential"] = sequential_sum(sq) / n
        cands["torch_sum"] = sq.sum(-1) / n
        hits = [k for k, v in cands.items() if bool((v == ref).all())]
        print(f"   n={n:6d}  exact match: {hits if hits else 'NONE'}")
        if not hits:
            for k, v in cands.items():
                print(f"      {k:12s} delta {float(v - ref):+.3e}")


def probe_dot(dims):
    """The GEMM contract: does an 8-lane fma-blocked dot reproduce
    torch.nn.functional.linear in f64?  (It is not expected to — the oracle
    *defines* the order and the reference is patched to accept it — but the
    size of the gap is worth knowing.)"""
    print("== 2b. F.linear vs the 8-lane fma dot, f64 (informational) ==")
    g = torch.Generator().manual_seed(13)
    for n in dims:
        a = torch.randn(n, generator=g, dtype=torch.float64)
        b = torch.randn(n, generator=g, dtype=torch.float64)
        ref = torch.dot(a, b)
        lanes = 8
        nb = (n // lanes) * lanes
        acc = [torch.zeros((), dtype=torch.float64) for _ in range(lanes)]
        for i in range(0, nb, lanes):
            for l in range(lanes):
                acc[l] = torch.addcmul(acc[l], a[i + l], b[i + l])  # not fma, close enough
        cur = acc
        while len(cur) > 1:
            cur = [cur[i] + cur[i + 1] for i in range(0, len(cur), 2)]
        got = cur[0]
        for i in range(nb, n):
            got = got + a[i] * b[i]
        print(f"   n={n:6d}  |dot - blocked8| = {float(abs(got - ref)):.3e}  "
              f"rel {float(abs(got - ref) / max(abs(float(ref)), 1e-300)):.3e}")


def probe_casts():
    print("== 3. f64 -> bf16 / f16 double rounding through f32 ==")
    g = torch.Generator().manual_seed(17)
    x = torch.randn(200000, generator=g, dtype=torch.float64) * 1e3
    for rdt in (torch.bfloat16, torch.float16):
        direct = x.to(rdt)
        viaf32 = x.to(torch.float32).to(rdt)
        same = bool((direct == viaf32).all())
        print(f"   {str(rdt):16s} f64->{rdt} == f64->f32->{rdt}: {same}")


def main():
    torch.set_num_threads(int(sys.argv[1]) if len(sys.argv) > 1 else 1)
    print(f"torch {torch.__version__}, threads {torch.get_num_threads()}")
    probe_pow()
    print()
    # Muse Glimmer's norm widths: H=6656 (text), D=128 (qk-norm), 1536 (vision),
    # plus the tiny-model widths.
    probe_mean([16, 64, 128, 1536, 6656])
    print()
    probe_dot([64, 128, 6656])
    print()
    probe_casts()


if __name__ == "__main__":
    main()
