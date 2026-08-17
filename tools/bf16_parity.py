#!/usr/bin/env python
"""Envelope gate for the fast BF16 engine against the oracle's bf16 twin.

`muse-oracle --exec bf16` is a candidate implementation, not a referee. It
accumulates in f32 (that is what `vdpbf16ps` does, and what llama.cpp does),
while `--dtype bf16 --attn flash` specifies an *ideal* accumulator with one
BF16 rounding per materialization. So the two cannot be bitwise, and the gate is
an envelope on the logits: how far apart, in BF16 ulps, and do they still agree
on what the model would say.

Usage:
  tools/bf16_parity.py --model tiny/tiny_text --ids 1,5,7,... [--chunk 256]
"""

import argparse
import json
import os
import subprocess
import sys

import numpy as np


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"failed: {' '.join(cmd)}\n{r.stderr}")
    return r.stderr


def bf16_ulp(a, b):
    """Distance in BF16 grid steps between two arrays of BF16-valued floats."""
    def bits(x):
        u = np.asarray(x, dtype=np.float32).view(np.uint32) >> 16
        u = u.astype(np.int32)
        return np.where(u & 0x8000, -(u & 0x7FFF), u)
    return np.abs(bits(a) - bits(b))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--ids", required=True)
    p.add_argument("--out", default="out/bf16_parity")
    p.add_argument("--oracle", default="build/muse-oracle")
    p.add_argument("--chunk", type=int, default=256)
    p.add_argument("--topk", type=int, default=64)
    p.add_argument("--assert-argmax", action="store_true",
                   help="exit non-zero unless the two agree on the top-1 token")
    a = p.parse_args()

    fast = os.path.join(a.out, "fast")
    twin = os.path.join(a.out, "twin")
    run([a.oracle, "--model", a.model, "--ids", a.ids, "--out", fast,
         "--exec", "bf16", "--chunk", str(a.chunk)])
    run([a.oracle, "--model", a.model, "--ids", a.ids, "--out", twin,
         "--dtype", "bf16", "--attn", "flash"])

    fm = json.load(open(os.path.join(fast, "meta.json")))
    tm = json.load(open(os.path.join(twin, "meta.json")))
    V = fm["V"]
    lf = np.fromfile(os.path.join(fast, "logits.bin"), dtype=np.float64)
    lt = np.fromfile(os.path.join(twin, "logits.bin"), dtype=np.float64).reshape(tm["T"], V)[-1]
    assert lf.size == V, f"{lf.size} != {V}"

    d = np.abs(lf - lt)
    ulp = bf16_ulp(lf, lt)
    ka = set(np.argpartition(lf, -a.topk)[-a.topk:])
    kb = set(np.argpartition(lt, -a.topk)[-a.topk:])
    same_top1 = int(lf.argmax()) == int(lt.argmax())

    print(f"fast --exec bf16  vs  oracle --dtype bf16 --attn flash   (last position, V={V})")
    print(f"  elements differing : {int(np.sum(lf != lt))}/{V}")
    print(f"  max abs diff       : {d.max():.6g}   (mean {d.mean():.3g})")
    print(f"  max BF16 ulp       : {int(ulp.max())}   (mean {ulp.mean():.2f}, "
          f"p99 {int(np.percentile(ulp, 99))})")
    print(f"  top-1              : {int(lf.argmax())} vs {int(lt.argmax())} "
          f"{'AGREE' if same_top1 else 'DIFFER'}")
    print(f"  top-{a.topk} overlap      : {100 * len(ka & kb) / a.topk:.2f}%")
    print("SUMMARY " + json.dumps(dict(
        max_abs=float(d.max()), max_ulp=int(ulp.max()), mean_ulp=float(ulp.mean()),
        top1_agree=bool(same_top1), topk_overlap=len(ka & kb) / a.topk,
        prefill_tok_s=fm.get("prefill_tok_s"), V=V)))

    if a.assert_argmax and not same_top1:
        sys.exit("FAIL: the fast engine and the twin disagree on the top-1 token")


if __name__ == "__main__":
    main()
