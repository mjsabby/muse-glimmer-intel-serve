#!/usr/bin/env bash
# The Phase 1 / Phase 3 gate: the oracle against the instrumented HF reference
# on the tiny models, plus the oracle's own determinism contract.
#
#   1. f64, bitwise      oracle == ref_forward --pure --fixed-reduce
#                        (same model structure, same reduction order)
#   2. f64, noise floor  oracle vs ref_forward --pure
#                        (HF's own BLAS/libm; publishes the gap)
#   3. bf16/f16, bitwise oracle --dtype X == ref_forward --pure --dtype X
#   4. determinism       --kernels scalar == avx512, 1 thread == N threads
#
# Usage: ./run_tiny.sh [tiny-dir] [python]
set -euo pipefail
cd "$(dirname "$0")"

TINY="${1:-tiny}"
PY="${2:-.venv/bin/python}"
ORACLE=build/muse-oracle
OUT=out/gate
IDS="1,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59"

if [[ ! -x "$ORACLE" ]]; then
    echo "build the oracle first: ./build.sh --cpu-only" >&2
    exit 1
fi
if [[ ! -d "$TINY/tiny_text" ]]; then
    echo "make the tiny models first: $PY py/make_tiny.py --out $TINY" >&2
    exit 1
fi

rc=0
pass() { printf '  \033[32mPASS\033[0m %s\n' "$1"; }
fail() { printf '  \033[31mFAIL\033[0m %s\n' "$1"; rc=1; }
bitwise() { # name, dirA, dirB
    if cmp -s "$2/logits.bin" "$3/logits.bin"; then pass "$1"; else
        fail "$1 ($(cmp -l "$2/logits.bin" "$3/logits.bin" | wc -l) differing bytes)"
    fi
}

mkdir -p "$OUT"

echo "== tiny_text: f64 bitwise (oracle order on both sides) =="
$ORACLE --model "$TINY/tiny_text" --ids "$IDS" --out "$OUT/f64_oracle" --dump-hidden >/dev/null
$PY py/ref_forward.py --model "$TINY/tiny_text" --ids "$IDS" --out "$OUT/f64_fixed" \
    --pure --fixed-reduce --dump-hidden --threads 1 >/dev/null
bitwise "logits" "$OUT/f64_oracle" "$OUT/f64_fixed"
for i in 00 01 02 03 04; do
    if [[ -f "$OUT/f64_oracle/hidden_$i.bin" ]]; then
        cmp -s "$OUT/f64_oracle/hidden_$i.bin" "$OUT/f64_fixed/hidden_$i.bin" \
            && pass "hidden_$i" || fail "hidden_$i"
    fi
done

echo "== tiny_text: f64 vs stock-reduction HF (noise floor, published) =="
$PY py/ref_forward.py --model "$TINY/tiny_text" --ids "$IDS" --out "$OUT/f64_ref" \
    --pure --dump-hidden --threads 1 >/dev/null
$PY py/diff_logits.py "$OUT/f64_oracle" "$OUT/f64_ref" --topk 20 --assert-max-abs 1e-12 \
    | sed 's/^/  /'

for dt in bf16 f16; do
    for at in eager flash; do
        echo "== tiny_text: $dt twin, --attn $at, bitwise =="
        $ORACLE --model "$TINY/tiny_text" --ids "$IDS" --out "$OUT/${dt}_${at}_oracle" \
            --dtype "$dt" --attn "$at" --dump-hidden >/dev/null
        $PY py/ref_forward.py --model "$TINY/tiny_text" --ids "$IDS" \
            --out "$OUT/${dt}_${at}_ref" --pure --dtype "$dt" --attn "$at" \
            --dump-hidden --threads 1 >/dev/null
        bitwise "$dt/$at logits" "$OUT/${dt}_${at}_oracle" "$OUT/${dt}_${at}_ref"
    done
    # the flash twin must differ from the eager one: it is a DIFFERENT
    # (looser) contract, and a flash kernel gated against the eager twin would
    # be gated against something no fused kernel can meet
    if cmp -s "$OUT/${dt}_eager_oracle/logits.bin" "$OUT/${dt}_flash_oracle/logits.bin"; then
        fail "$dt eager and flash twins are identical — the S/P materializations are not being applied"
    else
        pass "$dt eager != flash (the twins are distinct contracts)"
    fi
done

echo "== tiny_dflash: one drafting round, bitwise =="
$ORACLE --model "$TINY/tiny_text" --ids "$IDS" --out "$OUT/draft_oracle" \
    --assistant "$TINY/tiny_dflash" >/dev/null
$PY py/ref_dflash.py --model "$TINY/tiny_text" --assistant "$TINY/tiny_dflash" \
    --ids "$IDS" --out "$OUT/draft_ref" --pure --fixed-reduce --threads 1 >/dev/null
if cmp -s "$OUT/draft_oracle/draft/logits.bin" "$OUT/draft_ref/logits.bin"; then
    pass "draft logits"; else fail "draft logits"; fi
if cmp -s "$OUT/draft_oracle/draft/hidden.bin" "$OUT/draft_ref/hidden.bin"; then
    pass "draft hidden"; else fail "draft hidden"; fi
ta=$($PY -c "import json;print(json.load(open('$OUT/draft_oracle/draft/meta.json'))['draft_tokens'])")
tb=$($PY -c "import json;print(json.load(open('$OUT/draft_ref/meta.json'))['draft_tokens'])")
if [[ "$ta" == "$tb" ]]; then pass "draft tokens $ta"; else fail "draft tokens $ta vs $tb"; fi

echo "== tiny_vision: tower + projector + end-to-end, bitwise =="
VIDS="$($PY -c "print(','.join(['1'] + ['500'] * 20 + ['7']))")"
$PY py/ref_vision.py --model "$TINY/tiny_vision" --images SYNTH:120,150,5 \
    --out "$OUT/vision_ref" --pure --fixed-reduce --threads 1 --ids "$VIDS" >/dev/null
VGRID=$($PY -c "import json;print(json.load(open('$OUT/vision_ref/meta.json'))['grid_arg'])")
$ORACLE --model "$TINY/tiny_vision" --ids "$VIDS" --out "$OUT/vision_oracle" \
    --pixels "$OUT/vision_ref/pixel_values.bin" --grid "$VGRID" >/dev/null
if cmp -s "$OUT/vision_oracle/vision.bin" "$OUT/vision_ref/vision.bin"; then
    pass "vision features (grid $VGRID)"; else fail "vision features"; fi
if cmp -s "$OUT/vision_oracle/logits.bin" "$OUT/vision_ref/logits.bin"; then
    pass "image+text logits"; else fail "image+text logits"; fi

echo "== determinism: kernels x thread count =="
ref=""
for k in scalar avx512; do
    for t in 1 4 32; do
        d="$OUT/det_${k}_${t}"
        $ORACLE --model "$TINY/tiny_text" --ids "$IDS" --out "$d" \
            --kernels "$k" --threads "$t" >/dev/null
        if [[ -z "$ref" ]]; then ref="$d"; else bitwise "$k/$t threads" "$ref" "$d"; fi
    done
done

exit $rc
