#!/usr/bin/env bash
# The Phase 7 gate: the SYCL/oneDNN engine against the bf16 twin, plus the
# GPU engine's own internal contracts.
#
#   1. vs the twin      muse-gpu == muse-oracle --exec bf16   (the referee)
#   2. chunk invariance the prefill block width must not change the answer
#   3. GEMM routes      oneDNN and the hand-written SYCL GEMM must agree
#   4. decode == prefill  decoding N tokens must match prefilling the same seq
#   5. determinism      reruns bit-identical
#   6. flash tier      --flash-prefill is envelope-gated, never bitwise
#   7. tensor parallel  --shards N fixes the arithmetic, --gpus M only places it,
#                       so 1 card and 2 cards must agree bitwise (Phase 8 exit gate)
#
# Usage: ./run_gpu_gates.sh [tiny-dir]
set -euo pipefail
cd "$(dirname "$0")"

TINY="${1:-tiny}"
GPU=build-gpu/muse-gpu
CPU=build/muse-oracle
OUT=out/gpu_gate
IDS="1,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59"
MODEL="$TINY/tiny_text"

if [[ ! -x "$GPU" ]]; then
    echo "build the GPU engine first:" >&2
    echo "  source /opt/intel/oneapi/setvars.sh" >&2
    echo "  cmake -B build-gpu -DMUSE_GPU=ON -DCMAKE_CXX_COMPILER=icpx && cmake --build build-gpu -j" >&2
    exit 1
fi
[[ -x "$CPU" ]] || { echo "build the CPU referee first: ./build.sh --cpu-only" >&2; exit 1; }

rc=0
pass() { printf '  \033[32mPASS\033[0m %s\n' "$1"; }
fail() { printf '  \033[31mFAIL\033[0m %s\n' "$1"; rc=1; }
same() { # name, dirA, dirB
    if cmp -s "$2/logits.bin" "$3/logits.bin"; then pass "$1"; else
        fail "$1 ($(cmp -l "$2/logits.bin" "$3/logits.bin" 2>/dev/null | wc -l) differing bytes)"
    fi
}
run_gpu() { # outdir, extra args...
    local o="$1"; shift
    $GPU --model "$MODEL" --ids "$IDS" --out "$o" --max-seq 256 "$@" >/dev/null 2>&1
}

mkdir -p "$OUT"
ngpu=$($GPU --list-devices 2>/dev/null | head -1 | awk '{print $1}')
echo "== visible GPUs: ${ngpu:-0} =="
[[ "${ngpu:-0}" -ge 1 ]] || { echo "  no GPU: nothing to gate" >&2; exit 1; }

echo "== vs the bf16 twin (the referee) =="
$CPU --model "$MODEL" --ids "$IDS" --out "$OUT/twin" --exec bf16 --chunk 16 --max-seq 256 \
    >/dev/null 2>&1
run_gpu "$OUT/g16" --shards 1 --gpus 1 --chunk 16
same "gpu (1 shard) == bf16 twin" "$OUT/g16" "$OUT/twin"
run_gpu "$OUT/g16_tp" --shards 2 --gpus 1 --chunk 16
same "gpu (2 shards) == bf16 twin" "$OUT/g16_tp" "$OUT/twin"

echo "== chunk invariance (the ring must hold window + chunk rows) =="
for ch in 1 2 4 8 16; do run_gpu "$OUT/ch_$ch" --shards 2 --gpus 1 --chunk "$ch"; done
ok=1
for ch in 1 2 4 8; do
    cmp -s "$OUT/ch_16/logits.bin" "$OUT/ch_$ch/logits.bin" || { ok=0; fail "chunk $ch differs from 16"; }
done
(( ok )) && pass "chunk 1 == 2 == 4 == 8 == 16"

echo "== the two GEMM routes agree =="
# oneDNN 3.11.2 silently drops the B-handle offset when the matrix has one
# column, which made the head GEMM and every decode step wrong while every
# layer stayed bitwise correct. This gate is why that is not still true.
run_gpu "$OUT/nodnnl" --shards 1 --gpus 1 --chunk 16 --no-dnnl
same "oneDNN == hand-written SYCL GEMM" "$OUT/g16" "$OUT/nodnnl"

echo "== decode == prefill on the same sequence =="
SHORT="1,5,7,11,13,17,19,23"
$GPU --model "$MODEL" --ids "$SHORT" --out "$OUT/dec" --max-seq 256 --shards 2 --gpus 1 \
    --chunk 8 --decode 8 >/dev/null 2>&1
GEN=$(.venv/bin/python -c "import json;print(','.join(map(str,json.load(open('$OUT/dec/meta.json'))['generated'])))")
$GPU --model "$MODEL" --ids "$SHORT,$GEN" --out "$OUT/pre" --max-seq 256 --shards 2 \
    --gpus 1 --chunk 16 >/dev/null 2>&1
# prefilling prompt+generated must land on the same next-token distribution the
# last decode step produced
if cmp -s "$OUT/dec/logits.bin" "$OUT/pre/logits.bin"; then
    pass "decode path == prefill path (generated: $GEN)"
else
    fail "decode diverges from prefill on the same sequence"
fi

echo "== determinism =="
run_gpu "$OUT/rep1" --shards 2 --gpus 1 --chunk 16
run_gpu "$OUT/rep2" --shards 2 --gpus 1 --chunk 16
same "rerun bit-identical" "$OUT/rep1" "$OUT/rep2"

echo "== tensor parallelism: the split is arithmetic, the cards are placement =="
# Sharding fixes the reduction order (o_proj and mlp_down become partial sums
# that are added in shard order). WHERE a shard runs must therefore not change
# a single bit — so `--shards 2` on one card and on two cards have to agree
# exactly. That equality is Phase 8's exit gate, and it only means anything
# because the shard count is what varies the arithmetic, not the card count.
run_gpu "$OUT/tp2x1" --shards 2 --gpus 1 --chunk 16
if [[ "${ngpu:-0}" -ge 2 ]]; then
    run_gpu "$OUT/tp2x2" --shards 2 --gpus 2 --chunk 16
    same "shards 2: 1 card == 2 cards (prefill)" "$OUT/tp2x1" "$OUT/tp2x2"

    # Prefill alone does NOT cover the cross-card path. The residual stream is
    # dim-major [H, block], so its live region is contiguous only when
    # n == block — true for a full prefill chunk, false for every decode step.
    # A prefill-only gate passed here while decode silently read the wrong
    # columns on the 30B.
    $GPU --model "$MODEL" --ids "$IDS" --out "$OUT/d1" --max-seq 256 --shards 2 --gpus 1 \
        --chunk 16 --decode 8 >/dev/null 2>&1
    $GPU --model "$MODEL" --ids "$IDS" --out "$OUT/d2" --max-seq 256 --shards 2 --gpus 2 \
        --chunk 16 --decode 8 >/dev/null 2>&1
    same "shards 2: 1 card == 2 cards (decode, n < block)" "$OUT/d1" "$OUT/d2"
    g1=$(.venv/bin/python -c "import json;print(json.load(open('$OUT/d1/meta.json'))['generated'])")
    g2=$(.venv/bin/python -c "import json;print(json.load(open('$OUT/d2/meta.json'))['generated'])")
    [[ "$g1" == "$g2" ]] && pass "generated tokens agree $g1" \
                         || fail "generated differ: $g1 vs $g2"

    # every shard must hold the SAME reduced residual stream: the norms after
    # an all-reduce are replicated, so a one-ulp disagreement between shards
    # would compound instead of averaging out
    run_gpu "$OUT/tp4" --shards 2 --gpus 2 --chunk 16
    same "2-card run is reproducible" "$OUT/tp2x2" "$OUT/tp4"
else
    echo "  (only one card visible: cross-card gates skipped)"
fi

echo "== --flash-prefill tier (matrix-engine attention, LOOSER contract) =="
# This tier takes the softmax max/sum over a TILE of keys rather than folding
# keys in one at a time, because that is the only shape a matrix engine can
# compute. It is a different summation schedule for the same function, so it
# is gated on the logit ENVELOPE — asserting bitwise equality here would be
# asserting something no fused kernel can meet, which is the trap the eager/
# flash twin split in run_tiny.sh exists to avoid.
$GPU --model "$MODEL" --ids "$IDS" --out "$OUT/fl" --max-seq 256 --shards 2 --gpus 1 \
    --chunk 16 --flash-prefill >/dev/null 2>&1
envelope() { # name, dirA, dirB, max-abs budget
    .venv/bin/python - "$2" "$3" "$4" "$1" <<'PY'
import json, subprocess, sys
a, b, budget, name = sys.argv[1:5]
out = subprocess.run([".venv/bin/python", "py/diff_logits.py", a, b, "--topk", "20"],
                     capture_output=True, text=True).stdout
line = [l for l in out.splitlines() if l.startswith("SUMMARY")]
if not line:
    print("  \033[31mFAIL\033[0m %s (diff_logits produced no summary)" % name); sys.exit(1)
d = json.loads(line[0][len("SUMMARY "):])
ok = d["argmax_eq"] == d["T"] and d["topk_overlap"] >= 0.85 and d["max_abs"] <= float(budget)
print("  %s %s (max abs %.3e, argmax %d/%d, top-20 %.0f%%)" %
      ("\033[32mPASS\033[0m" if ok else "\033[31mFAIL\033[0m", name,
       d["max_abs"], d["argmax_eq"], d["T"], 100 * d["topk_overlap"]))
sys.exit(0 if ok else 1)
PY
}
# top-k overlap is a weak signal on the tiny model (random init, so the
# logits are near-uniform and the tail order is noise); argmax equality and
# the absolute budget are what carry this gate.
envelope "flash tier inside the envelope of the twin" "$OUT/fl" "$OUT/twin" 0.5 || rc=1

# the tiers must actually DIFFER: if they did not, the looser contract would
# be buying nothing and something is not being exercised
if cmp -s "$OUT/fl/logits.bin" "$OUT/g16_tp/logits.bin"; then
    fail "flash tier is bit-identical to the exact tier — the tier is not active"
else
    pass "flash tier differs from the exact tier (distinct contracts)"
fi

# and it must still be deterministic and placement-independent
$GPU --model "$MODEL" --ids "$IDS" --out "$OUT/fl2" --max-seq 256 --shards 2 --gpus 1 \
    --chunk 16 --flash-prefill >/dev/null 2>&1
same "flash tier rerun bit-identical" "$OUT/fl" "$OUT/fl2"
if [[ "${ngpu:-0}" -ge 2 ]]; then
    $GPU --model "$MODEL" --ids "$IDS" --out "$OUT/fl3" --max-seq 256 --shards 2 --gpus 2 \
        --chunk 16 --flash-prefill >/dev/null 2>&1
    same "flash tier: 1 card == 2 cards" "$OUT/fl" "$OUT/fl3"
fi

exit $rc
