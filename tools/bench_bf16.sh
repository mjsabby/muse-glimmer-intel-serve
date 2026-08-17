#!/usr/bin/env bash
# BF16 CPU benchmark for the AVX-512 engine: prefill tok/s at several depths and
# decode tok/s, in the same shape llama.cpp's `llama-bench -p N -n M` reports so
# the two are directly comparable.
#
# The number this produces is dominated by ONE question: does the 55.5 GiB of
# BF16 weights fit in page cache? Streaming from NVMe caps decode at ~0.07 tok/s
# on this box; resident, the ceiling is DRAM read bandwidth (measured 62.7 GB/s
# => ~1.05 tok/s). The script prints MemAvailable before it starts and warms the
# cache first, so a run that was secretly disk-bound is visible in the output
# rather than silently reported as an engine result.
#
# Usage: tools/bench_bf16.sh [out-dir]
set -euo pipefail
cd "$(dirname "$0")/.."

OUT="${1:-out/bench_bf16}"
MODEL="${MUSE_TARGET:-meta-models/Muse-Glimmer-30B}"
THREADS="${MUSE_THREADS:-$(nproc)}"
DECODE="${MUSE_DECODE:-16}"
PY="${MUSE_PYTHON:-.venv/bin/python}"
mkdir -p "$OUT"

avail_gb() { awk '/^MemAvailable:/{printf "%.1f", $2/1048576}' /proc/meminfo; }
# MemAvailable counts reclaimable page cache, so it barely moves when the
# weights land in cache — `Cached` is what shows whether they are resident.
cached_gb() { awk '/^Cached:/{printf "%.1f", $2/1048576}' /proc/meminfo; }
WEIGHTS_GB=59.6

echo "host      : $(nproc) threads, $(awk '/^model name/{sub(/^.*: /,"");print;exit}' /proc/cpuinfo)"
echo "threads   : $THREADS"
echo "MemAvail  : $(avail_gb) GB   (BF16 weights need ${WEIGHTS_GB} GB resident)"
if (( $(echo "$(avail_gb) < $WEIGHTS_GB" | bc -l) )); then
    echo "WARNING   : weights will NOT fit in page cache — this measures your NVMe,"
    echo "            not the engine. Free ~${WEIGHTS_GB} GB for a meaningful number."
fi
echo

# ids: a deterministic pseudo-prompt, long enough for the deepest run
$PY - "$OUT" <<'EOF'
import random, sys
random.seed(7)
ids = ['200000'] + [str(random.randrange(1000, 200000)) for _ in range(4095)]
open(sys.argv[1] + "/ids.txt", "w").write(",".join(ids))
EOF

head_ids() { $PY -c "
import sys
ids=open('$OUT/ids.txt').read().split(',')
print(','.join(ids[:int(sys.argv[1])]))" "$1"; }

echo "warming the page cache (one full pass over the weights)..."
./build/muse-oracle --model "$MODEL" --ids "$(head_ids 8)" --out "$OUT/warm" \
    --exec bf16 --max-seq 64 --threads "$THREADS" >/dev/null 2>&1
echo "page cache after warm: $(cached_gb) GB  (want >= ${WEIGHTS_GB})"
echo

printf '%-22s %12s %12s\n' "run" "tok/s" "wall (s)"
for n in 128 512 2048; do
    ids="$(head_ids $n)"
    log=$(./build/muse-oracle --model "$MODEL" --ids "$ids" --out "$OUT/pre_$n" \
          --exec bf16 --chunk "$n" --max-seq $((n + 64)) --threads "$THREADS" 2>&1)
    ts=$(echo "$log" | awk '/prefill/{print $8}')
    ws=$(echo "$log" | awk '/prefill/{print $5}')
    printf '%-22s %12s %12s\n' "prefill $n (1 pass)" "$ts" "$ws"
done

log=$(./build/muse-oracle --model "$MODEL" --ids "$(head_ids 32)" --out "$OUT/dec" \
      --exec bf16 --decode "$DECODE" --max-seq 256 --threads "$THREADS" 2>&1)
printf '%-22s %12s %12s\n' "decode $DECODE" \
    "$(echo "$log" | awk '/decode /{print $8}')" "$(echo "$log" | awk '/decode /{print $5}')"

echo
echo "page cache at end: $(cached_gb) GB"
echo "llama.cpp equivalent:"
echo "  llama-bench -m <muse-bf16.gguf> -p 512 -n $DECODE -t $THREADS"
