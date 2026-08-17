#!/usr/bin/env bash
# The DFlash acceptance-rate baseline (docs/plan.md Phase 5's third exit gate).
#
# Greedy speculative decoding over the fixed prompt set in tools/prompts/,
# measured with the f64 oracle so the number is a property of the MODEL, not of
# a kernel's precision. Phase 10's engine must reproduce it; a lossier drafter
# tier costs acceptance rate, not correctness, and this is what that cost is
# measured against.
#
# Each round is one full f64 target forward (~20 s for these lengths) plus one
# drafter forward, so a 4-prompt x 6-round sweep takes roughly 10 minutes.
#
# Usage: tools/spec_baseline.sh [rounds] [out-dir]
set -euo pipefail
cd "$(dirname "$0")/.."

ROUNDS="${1:-6}"
OUT="${2:-out/spec_baseline}"
TARGET="${MUSE_TARGET:-meta-models/Muse-Glimmer-30B}"
DRAFTER="${MUSE_DRAFTER:-meta-models/Muse-Glimmer-30B-assistant}"
PY="${MUSE_PYTHON:-.venv/bin/python}"

mkdir -p "$OUT"
for f in tools/prompts/*.ids; do
    name="$(basename "$f" .ids)"
    echo "== $name =="
    ./build/muse-oracle --model "$TARGET" --assistant "$DRAFTER" \
        --ids "$f" --out "$OUT/$name" --draft-rounds "$ROUNDS" 2>/dev/null \
        | grep -E '^spec:'
done

echo
echo "== summary =="
$PY - "$OUT" <<'EOF'
import glob, json, os, sys

rows = []
for p in sorted(glob.glob(os.path.join(sys.argv[1], "*", "draft", "spec.json"))):
    d = json.load(open(p))
    rows.append((os.path.basename(os.path.dirname(os.path.dirname(p))), d))
if not rows:
    sys.exit("no spec.json found")
w = max(len(n) for n, _ in rows)
print(f"{'prompt':{w}s} {'rounds':>6s} {'drafted':>8s} {'accepted':>9s} "
      f"{'accept_rate':>12s} {'tok/round':>10s}")
tot_d = tot_a = tot_r = tot_t = 0
for n, d in rows:
    print(f"{n:{w}s} {d['rounds']:6d} {d['drafted']:8d} {d['accepted']:9d} "
          f"{d['accept_rate']:12.4f} {d['tokens_per_round']:10.3f}")
    tot_d += d["drafted"]
    tot_a += d["accepted"]
    tot_r += d["rounds"]
    tot_t += d["tokens_per_round"] * d["rounds"]
print(f"{'ALL':{w}s} {tot_r:6d} {tot_d:8d} {tot_a:9d} "
      f"{tot_a / tot_d:12.4f} {tot_t / tot_r:10.3f}")
EOF
