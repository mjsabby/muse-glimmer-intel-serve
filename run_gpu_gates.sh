#!/usr/bin/env bash
# The Phase 7 gate: the SYCL/oneDNN engine against the bf16 twin, plus the
# GPU engine's own internal contracts.
#
#   1. vs the twin      muse-gpu == muse-oracle --exec bf16   (the referee)
#   2. chunk invariance the prefill block width must not change the answer
#   3. GEMM routes      oneDNN and the hand-written SYCL GEMM must agree
#   4. decode == prefill  decoding N tokens must match prefilling the same seq
#   5. determinism      reruns bit-identical
#   6. device split     1 GPU == 2 GPUs (a layer split is not a numerics change)
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
run_gpu "$OUT/g16" --gpus 1 --chunk 16
same "gpu == bf16 twin" "$OUT/g16" "$OUT/twin"

echo "== chunk invariance (the ring must hold window + chunk rows) =="
for ch in 1 2 4 8 16; do run_gpu "$OUT/ch_$ch" --gpus 1 --chunk "$ch"; done
ok=1
for ch in 1 2 4 8; do
    cmp -s "$OUT/ch_16/logits.bin" "$OUT/ch_$ch/logits.bin" || { ok=0; fail "chunk $ch differs from 16"; }
done
(( ok )) && pass "chunk 1 == 2 == 4 == 8 == 16"

echo "== the two GEMM routes agree =="
# oneDNN 3.11.2 silently drops the B-handle offset when the matrix has one
# column, which made the head GEMM and every decode step wrong while every
# layer stayed bitwise correct. This gate is why that is not still true.
run_gpu "$OUT/nodnnl" --gpus 1 --chunk 16 --no-dnnl
same "oneDNN == hand-written SYCL GEMM" "$OUT/g16" "$OUT/nodnnl"

echo "== decode == prefill on the same sequence =="
SHORT="1,5,7,11,13,17,19,23"
$GPU --model "$MODEL" --ids "$SHORT" --out "$OUT/dec" --max-seq 256 --gpus 1 --chunk 8 \
    --decode 8 >/dev/null 2>&1
GEN=$(.venv/bin/python -c "import json;print(','.join(map(str,json.load(open('$OUT/dec/meta.json'))['generated'])))")
$GPU --model "$MODEL" --ids "$SHORT,$GEN" --out "$OUT/pre" --max-seq 256 --gpus 1 --chunk 16 \
    >/dev/null 2>&1
# prefilling prompt+generated must land on the same next-token distribution the
# last decode step produced
if cmp -s "$OUT/dec/logits.bin" "$OUT/pre/logits.bin"; then
    pass "decode path == prefill path (generated: $GEN)"
else
    fail "decode diverges from prefill on the same sequence"
fi

echo "== determinism =="
run_gpu "$OUT/rep1" --gpus 1 --chunk 16
run_gpu "$OUT/rep2" --gpus 1 --chunk 16
same "rerun bit-identical" "$OUT/rep1" "$OUT/rep2"

if [[ "${ngpu:-0}" -ge 2 ]]; then
    echo "== device split is not a numerics change =="
    run_gpu "$OUT/g2" --gpus 2 --chunk 16
    same "1 GPU == 2 GPUs (prefill)" "$OUT/g16" "$OUT/g2"

    # Prefill alone does NOT cover the layer-split handoff. The residual stream
    # is dim-major [H, block], so its live region is contiguous only when
    # n == block — true for a full prefill chunk, false for every decode step.
    # A prefill-only cross-card gate passes while decode silently reads the
    # wrong columns, which is exactly what happened here on the 30B.
    $GPU --model "$MODEL" --ids "$IDS" --out "$OUT/d1" --max-seq 256 --gpus 1 --chunk 16 \
        --decode 8 >/dev/null 2>&1
    $GPU --model "$MODEL" --ids "$IDS" --out "$OUT/d2" --max-seq 256 --gpus 2 --chunk 16 \
        --decode 8 >/dev/null 2>&1
    same "1 GPU == 2 GPUs (decode, n < block)" "$OUT/d1" "$OUT/d2"
    g1=$(.venv/bin/python -c "import json;print(json.load(open('$OUT/d1/meta.json'))['generated'])")
    g2=$(.venv/bin/python -c "import json;print(json.load(open('$OUT/d2/meta.json'))['generated'])")
    [[ "$g1" == "$g2" ]] && pass "generated tokens agree $g1" \
                         || fail "generated differ: $g1 vs $g2"
else
    echo "== device split: skipped (needs 2 cards) =="
fi

exit $rc
