#!/usr/bin/env python
"""Compare two dump dirs produced by ref_forward.py / the C++ oracle.

Reports max/mean abs diff, relative diff, per-position argmax agreement and
top-k overlap on logits; per-layer max-abs table when both sides dumped hidden
states. --assert-max-abs makes it exit non-zero when the logits gate fails."""

import argparse
import json
import os
import sys

import numpy as np


def load(d):
    meta = json.load(open(os.path.join(d, "meta.json")))
    logits = np.fromfile(os.path.join(d, "logits.bin"), dtype=np.float64)
    logits = logits.reshape(meta["T"], meta["V"])
    hidden = []
    for i in range(meta.get("n_hidden") or 0):
        hidden.append(np.fromfile(os.path.join(d, f"hidden_{i:02d}.bin"), dtype=np.float64))
    return meta, logits, hidden


def main():
    p = argparse.ArgumentParser()
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--topk", type=int, default=20)
    p.add_argument("--assert-max-abs", type=float, default=None)
    args = p.parse_args()

    ma, la, ha = load(args.a)
    mb, lb, hb = load(args.b)
    assert ma["ids"] == mb["ids"], "compared dumps used different token ids"
    assert la.shape == lb.shape

    d = np.abs(la - lb)
    denom = np.maximum(np.abs(la), np.abs(lb))
    rel = d / np.where(denom > 0, denom, 1.0)
    T = la.shape[0]

    argmax_eq = int(np.sum(la.argmax(1) == lb.argmax(1)))
    k = args.topk
    overlap = np.mean([
        len(set(np.argpartition(la[t], -k)[-k:]) & set(np.argpartition(lb[t], -k)[-k:])) / k
        for t in range(T)
    ])

    print(f"A: {args.a}  ({ma.get('kind','?')}, pure={ma.get('pure','?')})")
    print(f"B: {args.b}  ({mb.get('kind','?')}, pure={mb.get('pure','?')})")
    print(f"logits [T={T}, V={la.shape[1]}]")
    print(f"  max abs diff : {d.max():.3e}   (mean {d.mean():.3e})")
    print(f"  max rel diff : {rel.max():.3e}")
    print(f"  worst position: t={int(d.max(1).argmax())} (max abs {d.max(1).max():.3e})")
    print(f"  argmax equal : {argmax_eq}/{T}")
    print(f"  top-{k} overlap: {overlap*100:.2f}%")

    if ha and hb:
        n = min(len(ha), len(hb))
        print(f"per-layer max abs diff over {n} hidden dumps:")
        worst = []
        for i in range(n):
            m = float(np.abs(ha[i] - hb[i]).max())
            worst.append((m, i))
            print(f"  layer {i:2d}: {m:.3e}")
        worst.sort(reverse=True)
        print("  worst layers:", [(i, f"{m:.2e}") for m, i in worst[:5]])

    summary = dict(max_abs=float(d.max()), mean_abs=float(d.mean()),
                   max_rel=float(rel.max()), argmax_eq=argmax_eq, T=T,
                   topk_overlap=float(overlap))
    print("SUMMARY " + json.dumps(summary))

    if args.assert_max_abs is not None and d.max() > args.assert_max_abs:
        print(f"FAIL: max abs {d.max():.3e} > gate {args.assert_max_abs:.1e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
