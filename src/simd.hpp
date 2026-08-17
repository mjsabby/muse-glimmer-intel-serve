// The GEMM-class reduction order of the oracle, and its two executions.
//
// dot8(a, b, n) = 8 interleaved accumulator lanes, each accumulating with the
// IEEE-754 fused multiply-add (lane l takes elements i ≡ l mod 8):
//     lane_l = fma(a[i], b[i], lane_l)      for i = l, l+8, l+16, …
// combined in the fixed tree
//     ((l0+l1) + (l2+l3)) + ((l4+l5) + (l6+l7))
// plus a sequential fma tail for n mod 8 leftovers.
//
// The AVX-512 path (_mm512_fmadd_pd) and the scalar path (std::fma) execute
// the SAME abstract operation sequence — fma and add are exactly specified by
// IEEE 754, so the two paths are bitwise identical on any platform; the ISA
// only affects speed. Explicit fma here does not conflict with
// -ffp-contract=off, which only stops the COMPILER from fusing plain a*b+c
// expressions on its own.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

#if defined(__AVX512F__)
#include <immintrin.h>
#endif

namespace simd
{

    // set once at startup (--kernels scalar); read-only afterwards
    inline bool &force_scalar()
    {
        static bool v = false;
        return v;
    }

    inline bool avx512_compiled()
    {
#if defined(__AVX512F__)
        return true;
#else
        return false;
#endif
    }

    inline double dot8_scalar(const double *a, const double *b, int64_t n)
    {
        double l0 = 0, l1 = 0, l2 = 0, l3 = 0, l4 = 0, l5 = 0, l6 = 0, l7 = 0;
        int64_t i = 0;
        for (; i + 8 <= n; i += 8)
        {
            l0 = std::fma(a[i + 0], b[i + 0], l0);
            l1 = std::fma(a[i + 1], b[i + 1], l1);
            l2 = std::fma(a[i + 2], b[i + 2], l2);
            l3 = std::fma(a[i + 3], b[i + 3], l3);
            l4 = std::fma(a[i + 4], b[i + 4], l4);
            l5 = std::fma(a[i + 5], b[i + 5], l5);
            l6 = std::fma(a[i + 6], b[i + 6], l6);
            l7 = std::fma(a[i + 7], b[i + 7], l7);
        }
        double tail = 0;
        for (; i < n; ++i)
            tail = std::fma(a[i], b[i], tail);
        return (((l0 + l1) + (l2 + l3)) + ((l4 + l5) + (l6 + l7))) + tail;
    }

#if defined(__AVX512F__)
    inline double dot8_avx512(const double *a, const double *b, int64_t n)
    {
        __m512d acc = _mm512_setzero_pd();
        int64_t i = 0;
        for (; i + 8 <= n; i += 8)
            acc = _mm512_fmadd_pd(_mm512_loadu_pd(a + i), _mm512_loadu_pd(b + i), acc);
        alignas(64) double l[8];
        _mm512_store_pd(l, acc);
        double tail = 0;
        for (; i < n; ++i)
            tail = std::fma(a[i], b[i], tail);
        return (((l[0] + l[1]) + (l[2] + l[3])) + ((l[4] + l[5]) + (l[6] + l[7]))) + tail;
    }
#endif

    inline double dot8(const double *a, const double *b, int64_t n)
    {
#if defined(__AVX512F__)
        if (!force_scalar())
            return dot8_avx512(a, b, n);
#endif
        return dot8_scalar(a, b, n);
    }

#if defined(__AVX512F__)
    // Panelled whole-row-block driver: computes out[r*T + c] = dot8(w_r, x_c, n)
    // for ALL T columns of an R-row block, blocking the i dimension into
    // panels of PANEL doubles so the R w-rows' panel slice (R*PANEL*8 bytes)
    // stays L1-resident across every token group, with the C-column groups'
    // accumulators PARKED in memory between panels. The park store/reload is
    // exact (no arithmetic), panels advance in ascending i, and each lane
    // still accumulates elements i = lane (mod 8) strictly ascending — the
    // result is bitwise-identical to unpanelled dot8_tile calls per element.
    template <int R, int C>
    inline void dot8_block_panel_avx512(const double *const w[], const double *X,
                                        int64_t T, int64_t in, double *out /*[R*T]*/,
                                        double *park /*[ceil(T/C)*R*C*8]*/)
    {
        // panel length in doubles. R*PANEL*8 bytes of w stay cache-hot across
        // all token groups (80KB at R=5 — L2-hot; smaller L1-sized panels
        // measured no better here: the win is bounded by DRAM/page-cache
        // pressure when several models contend for RAM — measured ~6% overall
        // on the 27B vs the unpanelled tile, bitwise-identical)
        constexpr int64_t PANEL = 2048;
        const int64_t ngroups = (T + C - 1) / C;
        const int64_t n8 = in & ~int64_t(7);
        for (int64_t g = 0; g < ngroups * R * C * 8; ++g)
            park[g] = 0.0;
        for (int64_t p0 = 0; p0 < n8; p0 += PANEL)
        {
            const int64_t pend = std::min(n8, p0 + PANEL);
            for (int64_t g = 0; g < ngroups; ++g)
            {
                const int64_t t0 = g * C, nc = std::min<int64_t>(C, T - t0);
                double *pk = park + g * R * C * 8;
                if (nc == C)
                {
                    __m512d acc[R][C];
                    for (int r = 0; r < R; ++r)
                        for (int c = 0; c < C; ++c)
                            acc[r][c] = _mm512_loadu_pd(pk + (r * C + c) * 8);
                    for (int64_t i = p0; i < pend; i += 8)
                    {
                        __m512d xv[C];
                        for (int c = 0; c < C; ++c)
                            xv[c] = _mm512_loadu_pd(X + (t0 + c) * in + i);
                        for (int r = 0; r < R; ++r)
                        {
                            const __m512d wv = _mm512_loadu_pd(w[r] + i);
                            for (int c = 0; c < C; ++c)
                                acc[r][c] = _mm512_fmadd_pd(wv, xv[c], acc[r][c]);
                        }
                    }
                    for (int r = 0; r < R; ++r)
                        for (int c = 0; c < C; ++c)
                            _mm512_storeu_pd(pk + (r * C + c) * 8, acc[r][c]);
                }
                else
                { // ragged final token group: same op order, one column at a time
                    for (int64_t c = 0; c < nc; ++c)
                        for (int r = 0; r < R; ++r)
                        {
                            __m512d a = _mm512_loadu_pd(pk + (r * C + c) * 8);
                            for (int64_t i = p0; i < pend; i += 8)
                                a = _mm512_fmadd_pd(_mm512_loadu_pd(w[r] + i),
                                                    _mm512_loadu_pd(X + (t0 + c) * in + i), a);
                            _mm512_storeu_pd(pk + (r * C + c) * 8, a);
                        }
                }
            }
        }
        // lane-combine tree + sequential fma tail, per output element (dot8's exact epilogue)
        for (int64_t g = 0; g < ngroups; ++g)
        {
            const int64_t t0 = g * C, nc = std::min<int64_t>(C, T - t0);
            const double *pk = park + g * R * C * 8;
            for (int r = 0; r < R; ++r)
                for (int64_t c = 0; c < nc; ++c)
                {
                    const double *l = pk + (r * C + c) * 8;
                    double tail = 0;
                    const double *x = X + (t0 + c) * in;
                    for (int64_t j = n8; j < in; ++j)
                        tail = std::fma(w[r][j], x[j], tail);
                    out[r * T + (t0 + c)] =
                        (((l[0] + l[1]) + (l[2] + l[3])) + ((l[4] + l[5]) + (l[6] + l[7]))) +
                        tail;
                }
        }
    }

    // R-row x C-column register tile of dot8s: out[r*C + c] = dot8(w_r, x_c, n).
    // Every output element keeps its own 8-lane fma chain in exactly dot8's
    // order — the tile only SHARES the w/x vector loads across elements
    // (R*C fmas per R+C loads instead of 1 per 2, and each x line is reused by
    // R rows), so the results are bitwise identical to R*C separate dot8
    // calls. Register budget: R*C accumulators + C xv + 1 wv zmm registers.
    template <int R, int C>
    inline void dot8_tile_avx512(const double *const w[], const double *const x[],
                                 int64_t n, double *out)
    {
        __m512d acc[R][C];
        for (int r = 0; r < R; ++r)
            for (int c = 0; c < C; ++c)
                acc[r][c] = _mm512_setzero_pd();
        int64_t i = 0;
        for (; i + 8 <= n; i += 8)
        {
            __m512d xv[C];
            for (int c = 0; c < C; ++c)
                xv[c] = _mm512_loadu_pd(x[c] + i);
            for (int r = 0; r < R; ++r)
            {
                const __m512d wv = _mm512_loadu_pd(w[r] + i);
                for (int c = 0; c < C; ++c)
                    acc[r][c] = _mm512_fmadd_pd(wv, xv[c], acc[r][c]);
            }
        }
        auto hsum = [](__m512d v)
        {
            alignas(64) double l[8];
            _mm512_store_pd(l, v);
            return (((l[0] + l[1]) + (l[2] + l[3])) + ((l[4] + l[5]) + (l[6] + l[7])));
        };
        for (int r = 0; r < R; ++r)
            for (int c = 0; c < C; ++c)
            {
                double tail = 0;
                for (int64_t j = i; j < n; ++j)
                    tail = std::fma(w[r][j], x[c][j], tail);
                out[r * C + c] = hsum(acc[r][c]) + tail;
            }
    }
#endif

} // namespace simd
