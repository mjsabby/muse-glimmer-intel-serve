cmake_minimum_required(VERSION 3.20)
project(muse_glimmer_intel_serve CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# -ffp-contract=off is load-bearing: the oracle's reduction order is only
# reproducible if the compiler never fuses a*b+c on its own (simd.hpp uses
# std::fma explicitly where fusion is part of the contract).
add_compile_options(-Wall -Wextra -O3 -ffp-contract=off -fno-math-errno)
find_package(OpenMP REQUIRED)

# ISA selection changes SPEED ONLY: simd.hpp's AVX-512 and scalar executions of
# the 8-lane fma reduction are the same abstract operation sequence, and
# `muse-oracle --kernels scalar` vs the default is a bitwise gate (run_tiny.sh).
# Without this the __AVX512F__ paths are compiled out entirely and the gate
# silently degenerates to comparing the scalar path with itself.
option(MUSE_NATIVE "Build the CPU referee for the host ISA (-march=native)" ON)
if(MUSE_NATIVE)
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag(-march=native MUSE_HAS_MARCH_NATIVE)
  if(MUSE_HAS_MARCH_NATIVE)
    add_compile_options(-march=native)
  endif()
endif()

# The float64 referee is a compact, header-oriented target built with plain
# g++, independent of the SYCL production runtime and of oneAPI being present.
add_executable(muse-oracle src/main.cpp)
target_include_directories(muse-oracle PRIVATE src)
target_link_libraries(muse-oracle PRIVATE OpenMP::OpenMP_CXX)

add_executable(fmath-test tests/fmath_test.cpp)
target_include_directories(fmath-test PRIVATE src)
target_link_libraries(fmath-test PRIVATE OpenMP::OpenMP_CXX m)

# The oracle's reduction order and transcendentals, exposed to
# py/ref_forward.py --fixed-reduce so the tiny-model gate can be bitwise
# instead of noise-floor. See tools/oracle_kernels.cpp for why.
add_library(muse_refkernels SHARED tools/oracle_kernels.cpp)
target_include_directories(muse_refkernels PRIVATE src)
target_link_libraries(muse_refkernels PRIVATE OpenMP::OpenMP_CXX)
set_target_properties(muse_refkernels PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Model-neutral infrastructure shared with the (future) GPU engine and serving
# library.
add_library(muse_core STATIC
  src/json.cpp
  src/hf_resolver.cpp
  src/safetensors.cpp
  src/sha256.cpp
  src/tensor.cpp
  src/tokenizer.cpp
)
target_include_directories(muse_core PUBLIC src)
target_link_libraries(muse_core PUBLIC OpenMP::OpenMP_CXX)

enable_testing()
add_executable(unit_tests
  tests/test_main.cpp
  tests/test_json.cpp
  tests/test_npy.cpp
  tests/test_resolver.cpp
  tests/test_json_schema.cpp
  tests/test_sha256.cpp
  tests/test_muse_config.cpp
)
target_compile_definitions(unit_tests PRIVATE MUSE_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
target_link_libraries(unit_tests PRIVATE muse_core)
add_test(NAME unit_tests COMMAND unit_tests)
add_test(NAME fmath_test COMMAND fmath-test)

# Intel Arc production backend (Phase 7+). Kept in a separate icpx build
# directory so the g++ float64 referee's compiler/numerics contract cannot be
# changed by the GPU toolchain.
option(MUSE_GPU "Build the SYCL/Level-Zero backend for Intel Arc" OFF)
if(MUSE_GPU)
  # icpx's own OpenMP codegen (-fiopenmp, which is what find_package(OpenMP)
  # selects under this compiler) segfaults the clang frontend on
  # muse::vision::forward. The LLVM OpenMP (-fopenmp) compiles it. Scoped to
  # this build tree only: the bitwise referee is the g++ build, so which
  # OpenMP runtime the GPU tree uses is not part of any contract.
  set_property(TARGET OpenMP::OpenMP_CXX PROPERTY INTERFACE_COMPILE_OPTIONS -fopenmp)
  set_property(TARGET OpenMP::OpenMP_CXX PROPERTY INTERFACE_LINK_OPTIONS -fopenmp)
  set(GPU_SYCL_COMPILE
      -fsycl
      -fsycl-targets=spir64_gen
      -ffp-model=precise
      -ffp-contract=off
      -fsycl-device-code-split=per_kernel)
  set(GPU_AOT_LINK "SHELL:-Xsycl-target-backend=spir64_gen \"-device bmg-g31\"")

  if(NOT EXISTS "${CMAKE_SOURCE_DIR}/src/gpu")
    message(FATAL_ERROR "MUSE_GPU=ON but src/gpu/ does not exist yet (Phase 7)")
  endif()

  add_library(muse_gpu_backend STATIC
    src/gpu/gpu_engine.cpp
  )
  target_include_directories(muse_gpu_backend PUBLIC src)
  target_compile_options(muse_gpu_backend PRIVATE ${GPU_SYCL_COMPILE})

  option(MUSE_GPU_DNNL "Use oneDNN/XMX for the fast prefill tier" ON)
  if(MUSE_GPU_DNNL)
    find_package(dnnl CONFIG REQUIRED)
    target_link_libraries(muse_gpu_backend PUBLIC DNNL::dnnl)
    target_compile_definitions(muse_gpu_backend PUBLIC ORACLE_GPU_DNNL=1)
  else()
    target_compile_definitions(muse_gpu_backend PUBLIC ORACLE_GPU_DNNL=0)
  endif()

  target_link_libraries(muse_gpu_backend PUBLIC muse_core)
  target_link_options(muse_gpu_backend INTERFACE ${GPU_SYCL_COMPILE} ${GPU_AOT_LINK})

  # icpx defaults HOST code to -fp-model=fast, which reassociates the f32 sums
  # the CLI and the loader evaluate; the g++ referee does not, so pin it.
  add_executable(muse-gpu src/gpu/gpu_main.cpp)
  target_compile_options(muse-gpu PRIVATE -fp-model=precise)
  target_link_libraries(muse-gpu PRIVATE muse_gpu_backend)
  target_link_options(muse-gpu PRIVATE ${GPU_SYCL_COMPILE} ${GPU_AOT_LINK})

  set_target_properties(muse_gpu_backend muse_core
                        PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()
