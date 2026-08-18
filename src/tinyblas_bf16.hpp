// tinyBLAS — the BF16/AVX-512 GEMM from llamafile, vendored.
//
// Copyright 2024 Mozilla Foundation
//
// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
// BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
// ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
// ---------------------------------------------------------------------------
// Derived from ggml/src/ggml-cpu/llamafile/sgemm.cpp in llama.cpp
// (build 01818e495). Changes made here:
//
//   * only the __AVX512BF16__ BF16 x BF16 -> F32 path is kept; every other
//     type, ISA and architecture is dropped;
//   * ggml's threadpool coordination (params->ith/nth, ggml_barrier,
//     ggml_threadpool_chunk_set/add) is replaced by an OpenMP dynamic loop
//     over the same job space, which this repo's engine already runs under;
//   * ggml_bf16_t becomes a plain uint16_t.
//
// The arithmetic — the RM x RN register tile, the k-order, and the per-output
// horizontal sum — is unmodified, and is bit-identical to llama.cpp's on the
// shapes this repo uses (verified against the upstream kernel on
// W[19968,6656] x X[2048,6656]).
//
// Why vendor rather than write our own: measured on this machine at the real
// MLP shape, this kernel does 4.14 TFLOP/s where the engine's own hand-written
// tile does 3.13. The difference is not the instruction or the tile shape,
// which are the same — it is that gemm_bloc's bounds are compile-time
// constants, so the RM*RN accumulators stay in registers, while a kernel that
// carries runtime `min()` edge guards through the hot loop spills them. Ragged
// edges are handled by instantiating a narrower tile instead.
//
// MKL was measured first and is not an option here: cblas_gemm_bf16bf16f32
// runs at 0.76 TFLOP/s on this Zen 5 host, 4x slower than even the naive
// kernel. The oneDNN that ships with oneAPI 2026.0 is built against the DPC++
// runtime and has no usable CPU engine on this box.
#pragma once

#include <algorithm>
#include <cstdint>

#if defined(__AVX512BF16__)
#include <immintrin.h>
#endif

namespace tinyblas
{

#if defined(__AVX512BF16__)

    inline __m512bh load_bh(const uint16_t *p)
    {
        return (__m512bh)_mm512_loadu_ps((const float *)p);
    }
    inline __m512 madd_bh(__m512bh a, __m512bh b, __m512 c)
    {
        return _mm512_dpbf16_ps(c, a, b);
    }
    inline float hsum(__m512 x) { return _mm512_reduce_add_ps(x); }

    template <int M>
    inline int64_t block_size(int64_t m)
    {
        const int64_t nb = (m + M - 1) / M;
        return (m % nb == 0) ? m / nb : (m / nb) + 1;
    }
    constexpr inline int64_t bloc_pos(int64_t ib, int64_t ibN, int64_t bs)
    {
        return ib < ibN ? ib * bs : ibN * bs + (ib - ibN) * (bs - 1);
    }

    // C[n, m] = A[m, k] * B[n, k]^T, all row-major, A/B bf16, C f32.
    // In this repo's terms: A = weights [out, in], B = activations [T, in],
    // C = output [T, out], so m = out, n = T, k = in.
    struct TinyBLAS
    {
        static constexpr int64_t KN = 32; // bf16 lanes per vdpbf16ps operand

        const uint16_t *A;
        int64_t lda;
        const uint16_t *B;
        int64_t ldb;
        float *C;
        int64_t ldc;
        int64_t k;

        // The hot kernel: an RM x RN tile with COMPILE-TIME bounds, so the
        // RM*RN accumulators live in registers for the whole k sweep.
        template <int RM, int RN>
        inline void gemm_bloc(int64_t ii, int64_t jj) const
        {
            __m512 Cv[RN][RM] = {};
            for (int64_t l = 0; l < k; l += KN)
            {
                // help the compiler with the op order
                if constexpr (RM <= RN)
                {
                    __m512bh Av[RM];
                    for (int64_t i = 0; i < RM; ++i)
                        Av[i] = load_bh(A + lda * (ii + i) + l);
                    for (int64_t j = 0; j < RN; ++j)
                    {
                        __m512bh Bv = load_bh(B + ldb * (jj + j) + l);
                        for (int64_t i = 0; i < RM; ++i)
                            Cv[j][i] = madd_bh(Av[i], Bv, Cv[j][i]);
                    }
                }
                else
                {
                    __m512bh Bv[RN];
                    for (int64_t j = 0; j < RN; ++j)
                        Bv[j] = load_bh(B + ldb * (jj + j) + l);
                    for (int64_t i = 0; i < RM; ++i)
                    {
                        __m512bh Av = load_bh(A + lda * (ii + i) + l);
                        for (int64_t j = 0; j < RN; ++j)
                            Cv[j][i] = madd_bh(Av, Bv[j], Cv[j][i]);
                    }
                }
            }
            for (int64_t j = 0; j < RN; ++j)
                for (int64_t i = 0; i < RM; ++i)
                    C[ldc * (jj + j) + (ii + i)] = hsum(Cv[j][i]);
        }

        template <int RM, int RN, int BM>
        void gemm(int64_t m, int64_t n, int64_t BN) const
        {
            const int64_t ytiles = m / (RM * BM);
            const int64_t xtiles = (n + RN - 1) / RN;
            const int64_t jj_RN = (xtiles - (xtiles * RN - n));
            const int64_t NB_BN = xtiles < BN ? 1 : (xtiles + BN / 2) / BN;
            const int64_t SIZE_BN = xtiles % NB_BN == 0 ? xtiles / NB_BN : xtiles / NB_BN + 1;
            const int64_t jj_BN = (NB_BN - (NB_BN * SIZE_BN - xtiles));
            const int64_t nb_job = ytiles * NB_BN;

            // upstream hands jobs out through ggml's threadpool chunk counter;
            // a dynamic OpenMP loop over the same job space is equivalent and
            // does not change which (ii, jj) tiles exist
#pragma omp parallel for schedule(dynamic)
            for (int64_t job = 0; job < nb_job; ++job)
            {
                const int64_t ii = (job % ytiles) * RM * BM;
                const int64_t jb = job / ytiles;
                const int64_t jr0 = bloc_pos(jb, jj_BN, SIZE_BN);
                const int64_t jrN = bloc_pos(jb + 1, jj_BN, SIZE_BN);
                const int64_t jj0 = bloc_pos(jr0, jj_RN, RN);
                const int64_t jj2 = bloc_pos(jrN, jj_RN, RN);
                const int64_t jj1 = jj2 < jj_RN * RN ? jj2 : jj_RN * RN;
                for (int64_t bi = 0; bi < BM * RM; bi += RM)
                {
                    int64_t jj = jj0;
                    for (; jj < jj1; jj += RN)
                        gemm_bloc<RM, RN>(ii + bi, jj);
                    if constexpr (RN > 1)
                        for (; jj < jj2; jj += RN - 1)
                            gemm_bloc<RM, RN - 1>(ii + bi, jj);
                }
            }
        }

        template <int RM, int RN, int BM>
        inline void mnpack(int64_t m, int64_t n, int64_t SIZE_N, int64_t BN) const
        {
            if (SIZE_N == RN)
                return gemm<RM, RN, BM>(m, n, BN);
            if constexpr (RN > 1)
                return mnpack<RM, RN - 1, BM>(m, n, SIZE_N, BN);
        }

        // false => this kernel does not handle the shape; the caller falls back
        bool matmul(int64_t m, int64_t n, int nth) const
        {
            if (k % KN != 0)
                return false;
            const int64_t SIZE_N = block_size<6>(n);
            if (SIZE_N < 1 || SIZE_N > 6)
                return false;
            if (m % 16 == 0 && (m / 16 >= nth))
            {
                mnpack<4, 6, 4>(m, n, SIZE_N, 12);
                return true;
            }
            if (m % 8 == 0)
            {
                mnpack<4, 6, 2>(m, n, SIZE_N, 12);
                return true;
            }
            if (m % 4 == 0)
            {
                mnpack<4, 6, 1>(m, n, SIZE_N, 12);
                return true;
            }
            return false;
        }
    };

    // Y[t*out + o] = sum_i W[o,i] * X[t,i]. Returns false if the shape is not
    // supported and the caller should use its own kernel.
    inline bool gemm_bf16(const uint16_t *W, const uint16_t *X, float *Y, int64_t T, int64_t in,
                          int64_t out, int nth)
    {
        TinyBLAS tb{W, in, X, in, Y, out, in};
        return tb.matmul(out, T, nth);
    }

#else
    inline bool gemm_bf16(const uint16_t *, const uint16_t *, float *, int64_t, int64_t,
                          int64_t, int)
    {
        return false;
    }
#endif

} // namespace tinyblas
