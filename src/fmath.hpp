// Self-contained deterministic f64 math for the oracle — no libm.
//
// Every function below is a fixed sequence of IEEE-754 exactly-rounded
// operations (+, -, *, /, sqrt, fma, comparisons, bit manipulation), so given
// the same inputs it produces bit-identical results on any IEEE-754 double
// platform, independent of libc/libm version — provided FP contraction is off
// (-ffp-contract=off; std::fma is used only where written explicitly, and fma
// itself is exactly specified by IEEE 754).
//
// exp/log/expm1/tanh/sin/cos are derived from FDLIBM 5.3 (developed at SunSoft,
// a Sun Microsystems business: "Permission to use, copy, modify, and distribute
// this software is freely granted, provided that this notice is preserved."),
// accuracy < 1 ulp for exp/sin/cos, ~1 ulp for log/expm1/tanh.
// pow(a,b) = exp(b·log a) with an fma cross-term correction; relative error
// ~ (1 + |b·ln a|/2) ulp — used only for RoPE inverse-frequency construction,
// where the resulting phase error is < 1e-11 rad for every position the model
// supports (see muse_glimmer.hpp).
//
// sin/cos argument reduction implements the FDLIBM medium path, valid for
// |x| < 2^20·π/2 ≈ 1.647e6 and guarded by an exception beyond it (RoPE angles
// are ≤ max_position_embeddings = 131072, far inside).
#pragma once

#include <cmath> // std::sqrt/fabs/fma only — compiled to hardware instructions
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace fmath
{

    inline uint64_t d2u(double x)
    {
        uint64_t u;
        std::memcpy(&u, &x, 8);
        return u;
    }
    inline double u2d(uint64_t u)
    {
        double x;
        std::memcpy(&x, &u, 8);
        return x;
    }
    inline uint32_t hi_word(double x) { return uint32_t(d2u(x) >> 32); }
    inline uint32_t lo_word(double x) { return uint32_t(d2u(x) & 0xffffffffu); }
    inline double set_hi(double x, uint32_t h)
    {
        return u2d((uint64_t(h) << 32) | (d2u(x) & 0xffffffffull));
    }
    inline double from_hl(uint32_t h, uint32_t l) { return u2d((uint64_t(h) << 32) | l); }
    // multiply by 2^k via exponent-field arithmetic (valid while result stays normal)
    inline double scale_exp(double y, int k)
    {
        return u2d(d2u(y) + (uint64_t(int64_t(k)) << 52));
    }

    // ------------------------------------------------------------------ exp (FDLIBM e_exp.c)

    inline double exp(double x)
    {
        static const double one = 1.0, halF[2] = {0.5, -0.5},
                            o_threshold = 7.09782712893383973096e+02,
                            u_threshold = -7.45133219101941108420e+02,
                            ln2HI[2] = {6.93147180369123816490e-01, -6.93147180369123816490e-01},
                            ln2LO[2] = {1.90821492927058770002e-10, -1.90821492927058770002e-10},
                            invln2 = 1.44269504088896338700e+00,
                            P1 = 1.66666666666666019037e-01, P2 = -2.77777777770155933842e-03,
                            P3 = 6.61375632143793436117e-05, P4 = -1.65339022054652515390e-06,
                            P5 = 4.13813679705723846039e-08, huge = 1.0e+300,
                            twom1000 = 9.33263618503218878990e-302;
        double hi = 0.0, lo = 0.0, c, t, y;
        int k = 0, xsb;
        uint32_t hx = hi_word(x);
        xsb = (hx >> 31) & 1;
        hx &= 0x7fffffff;
        if (hx >= 0x40862E42)
        { // |x| >= 709.78...
            if (hx >= 0x7ff00000)
            {
                if (((hx & 0xfffff) | lo_word(x)) != 0)
                    return x + x;          // NaN
                return xsb == 0 ? x : 0.0; // +inf / -inf
            }
            if (x > o_threshold)
                return huge * huge; // overflow -> inf
            if (x < u_threshold)
                return twom1000 * twom1000; // underflow -> 0
        }
        if (hx > 0x3fd62e42)
        { // |x| > 0.5 ln2
            if (hx < 0x3FF0A2B2)
            { // |x| < 1.5 ln2
                hi = x - ln2HI[xsb];
                lo = ln2LO[xsb];
                k = 1 - xsb - xsb;
            }
            else
            {
                k = int(invln2 * x + halF[xsb]);
                t = k;
                hi = x - t * ln2HI[0];
                lo = t * ln2LO[0];
            }
            x = hi - lo;
        }
        else if (hx < 0x3e300000)
        { // |x| < 2^-28
            if (huge + x > one)
                return one + x;
        }
        else
        {
            k = 0;
        }
        t = x * x;
        c = x - t * (P1 + t * (P2 + t * (P3 + t * (P4 + t * P5))));
        if (k == 0)
            return one - ((x * c) / (c - 2.0) - x);
        y = one - ((lo - (x * c) / (2.0 - c)) - hi);
        if (k >= -1021)
            return scale_exp(y, k);
        return scale_exp(y, k + 1000) * twom1000;
    }

    // ------------------------------------------------------------------ log (FDLIBM e_log.c)

    inline double log(double x)
    {
        static const double ln2_hi = 6.93147180369123816490e-01,
                            ln2_lo = 1.90821492927058770002e-10,
                            two54 = 1.80143985094819840000e+16,
                            Lg1 = 6.666666666666735130e-01, Lg2 = 3.999999999940941908e-01,
                            Lg3 = 2.857142874366239149e-01, Lg4 = 2.222219843214978396e-01,
                            Lg5 = 1.818357216161805012e-01, Lg6 = 1.531383769920937332e-01,
                            Lg7 = 1.479819860511658591e-01;
        double hfsq, f, s, z, R, w, t1, t2, dk;
        int k = 0, i, j;
        uint32_t hx = hi_word(x), lx = lo_word(x);
        if (hx < 0x00100000)
        { // x < 2^-1022
            if (((hx & 0x7fffffff) | lx) == 0)
                return -HUGE_VAL; // log(0)
            if (hx >> 31)
                return std::numeric_limits<double>::quiet_NaN(); // log(<0)
            k -= 54;
            x *= two54;
            hx = hi_word(x);
        }
        if (hx >= 0x7ff00000)
            return x + x; // inf/nan
        k += int(hx >> 20) - 1023;
        hx &= 0x000fffff;
        i = int((hx + 0x95f64) & 0x100000);
        x = set_hi(x, hx | (uint32_t(i) ^ 0x3ff00000));
        k += i >> 20;
        f = x - 1.0;
        if ((0x000fffff & (2 + hx)) < 3)
        { // |f| < 2^-20
            if (f == 0.0)
            {
                if (k == 0)
                    return 0.0;
                dk = double(k);
                return dk * ln2_hi + dk * ln2_lo;
            }
            R = f * f * (0.5 - 0.33333333333333333 * f);
            if (k == 0)
                return f - R;
            dk = double(k);
            return dk * ln2_hi - ((R - dk * ln2_lo) - f);
        }
        s = f / (2.0 + f);
        dk = double(k);
        z = s * s;
        i = int(hx) - 0x6147a;
        w = z * z;
        j = 0x6b851 - int(hx);
        t1 = w * (Lg2 + w * (Lg4 + w * Lg6));
        t2 = z * (Lg1 + w * (Lg3 + w * (Lg5 + w * Lg7)));
        i |= j;
        R = t2 + t1;
        if (i > 0)
        {
            hfsq = 0.5 * f * f;
            if (k == 0)
                return f - (hfsq - s * (hfsq + R));
            return dk * ln2_hi - ((hfsq - (s * (hfsq + R) + dk * ln2_lo)) - f);
        }
        if (k == 0)
            return f - s * (f - R);
        return dk * ln2_hi - ((s * (f - R) - dk * ln2_lo) - f);
    }

    // ------------------------------------------------------------------ log1p (FDLIBM s_log1p.c)

    inline double log1p(double x)
    {
        static const double ln2_hi = 6.93147180369123816490e-01,
                            ln2_lo = 1.90821492927058770002e-10,
                            two54 = 1.80143985094819840000e+16,
                            Lp1 = 6.666666666666735130e-01, Lp2 = 3.999999999940941908e-01,
                            Lp3 = 2.857142874366239149e-01, Lp4 = 2.222219843214978396e-01,
                            Lp5 = 1.818357216161805012e-01, Lp6 = 1.531383769920937332e-01,
                            Lp7 = 1.479819860511658591e-01;
        double hfsq, f = 0.0, c = 0.0, s, z, R, u;
        int32_t hx = int32_t(hi_word(x)), hu = 0, ax = hx & 0x7fffffff;
        int k = 1;
        if (hx < 0x3FDA827A)
        { // x < 0.41422
            if (ax >= 0x3ff00000)
            { // x <= -1.0
                if (x == -1.0)
                    return -HUGE_VAL; // log1p(-1) = -inf
                return std::numeric_limits<double>::quiet_NaN();
            }
            if (ax < 0x3e200000)
            { // |x| < 2^-29
                if (two54 + x > 0.0 && ax < 0x3c900000)
                    return x; // |x| < 2^-54
                return x - x * x * 0.5;
            }
            if (hx > 0 || hx <= int32_t(0xbfd2bec3))
            { // -0.2929 < x < 0.41422
                k = 0;
                f = x;
                hu = 1;
            }
        }
        if (hx >= 0x7ff00000)
            return x + x; // inf/nan
        if (k != 0)
        {
            if (hx < 0x43400000)
            {
                u = 1.0 + x;
                hu = int32_t(hi_word(u));
                k = (hu >> 20) - 1023;
                c = (k > 0) ? 1.0 - (u - x) : x - (u - 1.0); // exact correction term
                c /= u;
            }
            else
            {
                u = x;
                hu = int32_t(hi_word(u));
                k = (hu >> 20) - 1023;
                c = 0.0;
            }
            hu &= 0x000fffff;
            if (hu < 0x6a09e)
            {
                u = set_hi(u, uint32_t(hu) | 0x3ff00000); // normalize u
            }
            else
            {
                k += 1;
                u = set_hi(u, uint32_t(hu) | 0x3fe00000); // normalize u/2
                hu = (0x00100000 - hu) >> 2;
            }
            f = u - 1.0;
        }
        hfsq = 0.5 * f * f;
        if (hu == 0)
        { // |f| < 2^-20
            if (f == 0.0)
            {
                if (k == 0)
                    return 0.0;
                c += double(k) * ln2_lo;
                return double(k) * ln2_hi + c;
            }
            R = hfsq * (1.0 - 0.66666666666666666 * f);
            if (k == 0)
                return f - R;
            return double(k) * ln2_hi - ((R - (double(k) * ln2_lo + c)) - f);
        }
        s = f / (2.0 + f);
        z = s * s;
        R = z * (Lp1 + z * (Lp2 + z * (Lp3 + z * (Lp4 + z * (Lp5 + z * (Lp6 + z * Lp7))))));
        if (k == 0)
            return f - (hfsq - s * (hfsq + R));
        return double(k) * ln2_hi - ((hfsq - (s * (hfsq + R) + (double(k) * ln2_lo + c))) - f);
    }

    // ------------------------------------------------------------------ pow (via exp/log)

    // Domain restricted to what the model needs (a > 0 finite); error bound in the
    // file header. b·log(a) is corrected with an exact fma cross term.
    inline double pow(double a, double b)
    {
        if (b == 0.0 || a == 1.0)
            return 1.0;
        if (!(a > 0.0) || a == HUGE_VAL)
            throw std::runtime_error("fmath::pow: unsupported base (need finite a > 0)");
        double l = fmath::log(a);
        double p = l * b;
        double e = std::fma(l, b, -p); // exact residual of l*b
        double y = fmath::exp(p);
        return y + y * e;
    }

    // ------------------------------------------------------------------ expm1 (FDLIBM s_expm1.c)

    inline double expm1(double x)
    {
        static const double one = 1.0, huge = 1.0e+300, tiny = 1.0e-300,
                            o_threshold = 7.09782712893383973096e+02,
                            ln2_hi = 6.93147180369123816490e-01,
                            ln2_lo = 1.90821492927058770002e-10,
                            invln2 = 1.44269504088896338700e+00,
                            Q1 = -3.33333333333331316428e-02, Q2 = 1.58730158725481460165e-03,
                            Q3 = -7.93650757867487942473e-05, Q4 = 4.00821782732936239552e-06,
                            Q5 = -2.01099218183624371326e-07;
        double y, hi, lo, c = 0.0, t, e, hxs, hfx, r1;
        int k, xsb;
        uint32_t hx = hi_word(x);
        xsb = int(hx & 0x80000000u);
        hx &= 0x7fffffff;
        if (hx >= 0x4043687A)
        { // |x| >= 56 ln2
            if (hx >= 0x40862E42)
            {
                if (hx >= 0x7ff00000)
                {
                    if (((hx & 0xfffff) | lo_word(x)) != 0)
                        return x + x;           // NaN
                    return xsb == 0 ? x : -1.0; // ±inf
                }
                if (x > o_threshold)
                    return huge * huge; // overflow
            }
            if (xsb != 0)
            { // x < -56 ln2 -> -1
                if (x + tiny < 0.0)
                    return tiny - one;
            }
        }
        if (hx > 0x3fd62e42)
        { // |x| > 0.5 ln2
            if (hx < 0x3FF0A2B2)
            {
                if (xsb == 0)
                {
                    hi = x - ln2_hi;
                    lo = ln2_lo;
                    k = 1;
                }
                else
                {
                    hi = x + ln2_hi;
                    lo = -ln2_lo;
                    k = -1;
                }
            }
            else
            {
                k = int(invln2 * x + (xsb == 0 ? 0.5 : -0.5));
                t = k;
                hi = x - t * ln2_hi;
                lo = t * ln2_lo;
            }
            x = hi - lo;
            c = (hi - x) - lo;
        }
        else if (hx < 0x3c900000)
        { // |x| < 2^-54
            return x;
        }
        else
        {
            k = 0;
        }
        hfx = 0.5 * x;
        hxs = x * hfx;
        r1 = one + hxs * (Q1 + hxs * (Q2 + hxs * (Q3 + hxs * (Q4 + hxs * Q5))));
        t = 3.0 - r1 * hfx;
        e = hxs * ((r1 - t) / (6.0 - x * t));
        if (k == 0)
            return x - (x * e - hxs);
        e = (x * (e - c) - c);
        e -= hxs;
        if (k == -1)
            return 0.5 * (x - e) - 0.5;
        if (k == 1)
        {
            if (x < -0.25)
                return -2.0 * (e - (x + 0.5));
            return one + 2.0 * (x - e);
        }
        if (k <= -2 || k > 56)
        {
            y = one - (e - x);
            if (k == 1024)
                y = y * 2.0 * 8.98846567431157953865e+307; // 2^1023
            else
                y = scale_exp(y, k);
            return y - one;
        }
        t = one;
        if (k < 20)
        {
            t = from_hl(uint32_t(0x3ff00000 - (0x200000 >> k)), 0); // 1 - 2^-k
            y = t - (e - x);
            y = scale_exp(y, k);
        }
        else
        {
            t = from_hl(uint32_t((0x3ff - k) << 20), 0); // 2^-k
            y = x - (e + t);
            y += one;
            y = scale_exp(y, k);
        }
        return y;
    }

    // ------------------------------------------------------------------ tanh (FDLIBM s_tanh.c)

    inline double tanh(double x)
    {
        static const double one = 1.0, two = 2.0, tiny = 1.0e-300;
        double t, z;
        uint32_t jx = hi_word(x), ix = jx & 0x7fffffff;
        if (ix >= 0x7ff00000)
        {
            if ((ix & 0xfffff) | lo_word(x))
                return x + x;               // NaN
            return (jx >> 31) ? -one : one; // ±inf
        }
        if (ix < 0x40360000)
        { // |x| < 22
            if (ix < 0x3c800000)
                return x * (one + x); // |x| < 2^-55
            if (ix >= 0x3ff00000)
            { // |x| >= 1
                t = fmath::expm1(two * std::fabs(x));
                z = one - two / (t + two);
            }
            else
            {
                t = fmath::expm1(-two * std::fabs(x));
                z = -t / (t + two);
            }
        }
        else
        {
            z = one - tiny; // |x| >= 22
        }
        return (jx >> 31) ? -z : z;
    }

    // ------------------------------------------------------------------ sin/cos kernels (FDLIBM)

    inline double k_sin(double x, double y, int iy)
    {
        static const double half = 5.00000000000000000000e-01,
                            S1 = -1.66666666666666324348e-01, S2 = 8.33333333332248946124e-03,
                            S3 = -1.98412698298579493134e-04, S4 = 2.75573137070700676789e-06,
                            S5 = -2.50507602534068634195e-08, S6 = 1.58969099521155010221e-10;
        double z = x * x;
        double v = z * x;
        double r = S2 + z * (S3 + z * (S4 + z * (S5 + z * S6)));
        if (iy == 0)
            return x + v * (S1 + z * r);
        return x - ((z * (half * y - v * r) - y) - v * S1);
    }

    inline double k_cos(double x, double y)
    {
        static const double one = 1.00000000000000000000e+00,
                            C1 = 4.16666666666666019037e-02, C2 = -1.38888888888741095749e-03,
                            C3 = 2.48015872894767294178e-05, C4 = -2.75573143513906633035e-07,
                            C5 = 2.08757232129817482790e-09, C6 = -1.13596475577881948265e-11;
        double a, hz, z, r, qx;
        uint32_t ix = hi_word(x) & 0x7fffffff;
        z = x * x;
        r = z * (C1 + z * (C2 + z * (C3 + z * (C4 + z * (C5 + z * C6)))));
        if (ix < 0x3fd33333)
            return one - (0.5 * z - (z * r - x * y)); // |x| < 0.3
        if (ix > 0x3fe90000)
            qx = 0.28125; // |x| > 0.78125
        else
            qx = from_hl(ix - 0x00200000, 0); // x/4
        hz = 0.5 * z - qx;
        a = one - qx;
        return a - (hz - (z * r - x * y));
    }

    // FDLIBM e_rem_pio2.c medium path: |x| < 2^20·π/2. Beyond that the full
    // Payne–Hanek machinery would be needed — outside the oracle's angle range,
    // so it throws instead of silently degrading.
    inline int rem_pio2(double x, double *y)
    {
        static const double half = 5.00000000000000000000e-01,
                            invpio2 = 6.36619772367581382433e-01,
                            pio2_1 = 1.57079632673412561417e+00,
                            pio2_1t = 6.07710050650619224932e-11,
                            pio2_2 = 6.07710050630396597660e-11,
                            pio2_2t = 2.02226624879595063154e-21,
                            pio2_3 = 2.02226624871116645580e-21,
                            pio2_3t = 8.47842766036889956997e-32;
        uint32_t hx = hi_word(x), ix = hx & 0x7fffffff;
        if (ix >= 0x413921fb)
            throw std::runtime_error("fmath::rem_pio2: |x| >= 2^20*pi/2 unsupported");
        double t = std::fabs(x);
        int n = int(t * invpio2 + half);
        double fn = double(n);
        double r = t - fn * pio2_1;
        double w = fn * pio2_1t; // 1st round good to 85 bits
        y[0] = r - w;
        uint32_t high = hi_word(y[0]);
        int j = int(ix >> 20);
        int i = j - int((high >> 20) & 0x7ff);
        if (i > 16)
        { // 2nd iteration, good to 118 bits
            t = r;
            w = fn * pio2_2;
            r = t - w;
            w = fn * pio2_2t - ((t - r) - w);
            y[0] = r - w;
            high = hi_word(y[0]);
            i = j - int((high >> 20) & 0x7ff);
            if (i > 49)
            { // 3rd iteration, 151 bits
                t = r;
                w = fn * pio2_3;
                r = t - w;
                w = fn * pio2_3t - ((t - r) - w);
                y[0] = r - w;
            }
        }
        y[1] = (r - y[0]) - w;
        if (hx >> 31)
        {
            y[0] = -y[0];
            y[1] = -y[1];
            return -n;
        }
        return n;
    }

    inline double sin(double x)
    {
        uint32_t ix = hi_word(x) & 0x7fffffff;
        if (ix <= 0x3fe921fb)
        { // |x| <= pi/4
            if (ix < 0x3e500000)
                return x; // |x| < 2^-26
            return k_sin(x, 0.0, 0);
        }
        if (ix >= 0x7ff00000)
            return x - x; // inf/nan
        double y[2];
        int n = rem_pio2(x, y);
        switch (n & 3)
        {
        case 0:
            return k_sin(y[0], y[1], 1);
        case 1:
            return k_cos(y[0], y[1]);
        case 2:
            return -k_sin(y[0], y[1], 1);
        default:
            return -k_cos(y[0], y[1]);
        }
    }

    inline double cos(double x)
    {
        uint32_t ix = hi_word(x) & 0x7fffffff;
        if (ix <= 0x3fe921fb)
        { // |x| <= pi/4
            if (ix < 0x3e46a09e)
                return 1.0; // |x| < 2^-27
            return k_cos(x, 0.0);
        }
        if (ix >= 0x7ff00000)
            return x - x; // inf/nan
        double y[2];
        int n = rem_pio2(x, y);
        switch (n & 3)
        {
        case 0:
            return k_cos(y[0], y[1]);
        case 1:
            return -k_sin(y[0], y[1], 1);
        case 2:
            return -k_cos(y[0], y[1]);
        default:
            return k_sin(y[0], y[1], 1);
        }
    }

} // namespace fmath
