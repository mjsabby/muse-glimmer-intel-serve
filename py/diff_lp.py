#!/usr/bin/env python
"""Compare two low-precision dump dirs (f64 files holding bf16/f16-valued numbers).

Reports element counts that differ, ulp distance in the storage dtype's grid,
per-position argmax agreement, and (with hidden dumps on both sides) the
per-layer flip table that localizes where two runs part ways.

Usage: diff_lp.py A B [--dtype bf16|f16]
"""

import argparse
import glob
import json
import os

import numpy as np
import torch


def load(d, name):
    p = os.path.join(d, name)
    return np.fromfile(p, dtype=np.float64) if os.path.exists(p) else None


def ulp_dist(a, b, rdt):
    """Distance in storage-grid steps between two arrays of storage-valued f64."""
    ta = torch.tensor(a).to(rdt).view(torch.int16).numpy().astype(np.int32)
    tb = torch.tensor(b).to(rdt).view(torch.int16).numpy().astype(np.int32)
    # sign-magnitude bit patterns -> monotonic integer line
    ua = np.where(ta < 0, -(ta & 0x7FFF), ta)
    ub = np.where(tb < 0, -(tb & 0x7FFF), tb)
    return np.abs(ua - ub)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--dtype", choices=["bf16", "f16"], default="bf16")
    args = p.parse_args()
    rdt = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    meta = json.load(open(os.path.join(args.a, "meta.json")))
    T, V = meta["T"], meta["V"]
    la, lb = load(args.a, "logits.bin"), load(args.b, "logits.bin")
    assert la is not None and lb is not None and la.size == lb.size == T * V

    n = int(np.sum(la != lb))
    ud = ulp_dist(la, lb, rdt)
    print(f"logits: {n}/{la.size} elements differ  max|d|={np.abs(la - lb).max():.6g}  "
          f"max ulp={ud.max()}  mean ulp (differing)={ud[la != lb].mean() if n else 0:.2f}")
    A, B = la.reshape(T, V), lb.reshape(T, V)
    agree = sum(int(np.argmax(A[t]) == np.argmax(B[t])) for t in range(T))
    print(f"argmax agreement: {agree}/{T} positions")
    for t in range(T):
        ia, ib = int(np.argmax(A[t])), int(np.argmax(B[t]))
        if ia != ib:
            print(f"  pos {t}: argmax {ia} ({A[t, ia]:.6g}) vs {ib} ({B[t, ib]:.6g}); "
                  f"cross: A[{ib}]={A[t, ib]:.6g} B[{ia}]={B[t, ia]:.6g}")

    ha = sorted(glob.glob(os.path.join(args.a, "hidden_*.bin")))
    if ha and os.path.exists(os.path.join(args.b, os.path.basename(ha[0]))):
        print("per-layer flips (count / total, max ulp):")
        for pa in ha:
            xa = np.fromfile(pa, dtype=np.float64)
            xb = np.fromfile(os.path.join(args.b, os.path.basename(pa)), dtype=np.float64)
            d = int(np.sum(xa != xb))
            u = ulp_dist(xa, xb, rdt).max() if d else 0
            print(f"  {os.path.basename(pa)}: {d}/{xa.size}  max ulp={u}")


if __name__ == "__main__":
    main()
