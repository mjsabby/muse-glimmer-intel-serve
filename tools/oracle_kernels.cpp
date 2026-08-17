// The oracle's arithmetic kernels, exposed to py/ref_forward.py --fixed-reduce.
//
// ARCHITECTURE.md §"Numerics policy" says the ORACLE defines the reduction
// order: an 8-lane fma-blocked dot for every GEMM-class reduction, a blocked-8
// mean of squares for the norms, and FDLIBM transcendentals. BLAS' dot order is
// unspecified and libm's exp/tanh/sin/cos are platform-dependent, so a
// reference built on them can only agree with the oracle at the f64 noise
// floor. Linking the reference against these functions makes the tiny-model
// gate BITWISE, and narrows what that gate proves to exactly the right thing:
// the model structure — which module runs when, with which weights, mask, eps
// and scale — with the arithmetic held fixed on both sides.
//
// The kernels themselves are gated separately:
//   * the reduction order, by muse-oracle --kernels scalar vs the AVX-512 path
//     and by 1-vs-N thread invariance (both bitwise);
//   * the transcendentals, by tests/fmath_test.cpp against libm bounds.
//
// Deliberately scalar and single-threaded: this is a correctness instrument for
// toy dimensions, not a fast path.
#include <cstdint>

#include "fmath.hpp"
#include "muse_glimmer.hpp"
#include "simd.hpp"
#include "vision.hpp"

extern "C"
{

    // Y[t*out + o] = dot8(W[o,:], X[t,:]) — src/simd.hpp's fixed 8-lane order.
    void ok_gemm(const double *W, const double *X, double *Y, int64_t T, int64_t in,
                 int64_t out)
    {
        for (int64_t t = 0; t < T; ++t)
            for (int64_t o = 0; o < out; ++o)
                Y[t * out + o] = simd::dot8(W + o * in, X + t * in, in);
    }

    // out[r] = mean(X[r,:]^2) — src/muse_glimmer.hpp's blocked-8 order.
    void ok_meansq(const double *X, double *out, int64_t rows, int64_t dim)
    {
        for (int64_t r = 0; r < rows; ++r)
            out[r] = muse::mean_sq(X + r * dim, dim);
    }

    void ok_expv(const double *x, double *y, int64_t n)
    {
        for (int64_t i = 0; i < n; ++i)
            y[i] = fmath::exp(x[i]);
    }
    void ok_tanhv(const double *x, double *y, int64_t n)
    {
        for (int64_t i = 0; i < n; ++i)
            y[i] = fmath::tanh(x[i]);
    }
    void ok_sinv(const double *x, double *y, int64_t n)
    {
        for (int64_t i = 0; i < n; ++i)
            y[i] = fmath::sin(x[i]);
    }
    void ok_cosv(const double *x, double *y, int64_t n)
    {
        for (int64_t i = 0; i < n; ++i)
            y[i] = fmath::cos(x[i]);
    }
    void ok_siluv(const double *x, double *y, int64_t n)
    {
        for (int64_t i = 0; i < n; ++i)
            y[i] = muse::silu(x[i]);
    }
    void ok_sigmoidv(const double *x, double *y, int64_t n)
    {
        for (int64_t i = 0; i < n; ++i)
            y[i] = muse::sigmoid(x[i]);
    }
    // nn.LayerNorm with the oracle's blocked-8 mean and variance. torch's
    // f64 layer_norm uses its own (Welford-ish) accumulation, so this is the
    // vision tower's equivalent of ok_meansq for the text path.
    void ok_layernorm(const double *X, const double *w, const double *b, double eps, double *Y,
                      int64_t rows, int64_t dim)
    {
        muse::vision::layernorm_rows(X, w, b, eps, Y, rows, dim, prec::Dtype::F64);
    }

    void ok_erfv(const double *x, double *y, int64_t n)
    {
        for (int64_t i = 0; i < n; ++i)
            y[i] = fmath::erf(x[i]);
    }
    // the vision tower's exact-erf gelu (ACT2FN["gelu"], not the tanh form)
    void ok_geluv(const double *x, double *y, int64_t n)
    {
        for (int64_t i = 0; i < n; ++i)
            y[i] = fmath::gelu(x[i]);
    }
    void ok_powv(double base, const double *e, double *y, int64_t n)
    {
        for (int64_t i = 0; i < n; ++i)
            y[i] = fmath::pow(base, e[i]);
    }

} // extern "C"
