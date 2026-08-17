#!/usr/bin/env bash
# Build the deterministic CPU referee/tests and, unless --cpu-only is given,
# the Arc Pro B70 SYCL backend in a compiler-isolated build directory.
#
# --cpu-only must work on a machine with no oneAPI installed at all.
set -euo pipefail
cd "$(dirname "$0")"

jobs="$(nproc)"
cpu_only=0
if [[ "${1:-}" == "--cpu-only" ]]; then
    cpu_only=1
fi

cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$jobs" --target muse-oracle fmath-test unit_tests
ctest --test-dir build --output-on-failure

if (( cpu_only )); then
    exit 0
fi

oneapi_vars=/opt/intel/oneapi/setvars.sh
if [[ ! -f "$oneapi_vars" ]]; then
    echo "error: $oneapi_vars not found; use --cpu-only or install oneAPI" >&2
    exit 1
fi
set +u
source "$oneapi_vars" --force >/dev/null
set -u
cmake -B build-gpu -DMUSE_GPU=ON -DCMAKE_CXX_COMPILER=icpx
cmake --build build-gpu -j"$jobs" --target muse-gpu
