// Low-precision execution modes for the oracle (--dtype bf16|f16).
//
// The oracle stays f64 inside every operator; these helpers implement the
// *storage* rounding that a bf16/f16 execution of the model performs at each
// tensor materialization point. The rounding chain deliberately mirrors
// torch's cast implementation, measured on torch 2.x CPU (py/probe_bf16.py):
//
//   f64 -> bf16  ==  RNE(f64->f32) then RNE(f32->bf16)   (double rounding via
//   f32; 0/4001 crafted boundary traps deviate). Same for f16.
//
// Key consequence (also measured, 0/500k): for any single arithmetic op
// (+,-,*,/) on bf16 operands, torch's result == round(exact f64 result)
// under this chain, because the correctly-rounded f32 op equals the f32
// rounding of the exact value. So "exact f64 between materializations +
// this rounding at them" reproduces stock op-for-op wherever stock performs
// one correctly-rounded operation per materialization.
//
// NaN canonicalization matches c10: bf16 NaN -> 0x7FC0 (sign dropped),
// f16 NaN -> sign | 0x7E00. Infinities and RNE overflow-to-inf behave as in
// hardware casts.
#pragma once

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

namespace prec
{

    enum class Dtype : uint8_t
    {
        F64,
        BF16,
        F16
    };

    inline Dtype parse_dtype(const std::string &s)
    {
        if (s == "f64" || s == "float64")
            return Dtype::F64;
        if (s == "bf16" || s == "bfloat16")
            return Dtype::BF16;
        if (s == "f16" || s == "float16" || s == "fp16")
            return Dtype::F16;
        throw std::runtime_error("unknown --dtype: " + s + " (want f64|bf16|f16)");
    }

    inline const char *dtype_name(Dtype d)
    {
        switch (d)
        {
        case Dtype::BF16:
            return "bf16";
        case Dtype::F16:
            return "f16";
        default:
            return "f64";
        }
    }

    // f64 -> f32 -> bf16 -> f64 (RNE both steps; the f64->f32 step is the
    // hardware cast, the f32->bf16 step is c10's round_to_nearest_even)
    inline double round_bf16(double x)
    {
        float f = static_cast<float>(x);
        uint32_t u;
        std::memcpy(&u, &f, 4);
        if (f != f)
            u = 0x7fc00000u; // c10: NaN -> 0x7FC0, sign dropped
        else
            u = (u + 0x7fffu + ((u >> 16) & 1u)) & 0xffff0000u;
        std::memcpy(&f, &u, 4);
        return double(f);
    }

    // f64 -> f32 -> f16 -> f64 (RNE, subnormal-correct, overflow -> inf)
    inline double round_f16(double x)
    {
        float f = static_cast<float>(x);
        uint32_t u;
        std::memcpy(&u, &f, 4);
        const uint32_t sign = u & 0x80000000u;
        uint32_t abs = u & 0x7fffffffu;
        uint16_t h;
        if (abs > 0x7f800000u)
            h = uint16_t((sign >> 16) | 0x7e00u); // NaN
        else if (abs >= 0x477ff000u)
        { // >= 65520 rounds to inf (0x477ff000 = 65520)
            h = uint16_t((sign >> 16) | 0x7c00u);
        }
        else if (abs < 0x38800000u)
        { // subnormal or zero (< 2^-14); result grid ulp = 2^-24
            if (abs < 0x33000000u)
                h = uint16_t(sign >> 16); // < 2^-25: RNE to 0
            else
            {
                // h = round(mant24 * 2^(E-126)) with E the f32 exponent field
                const int shift = 126 - int(abs >> 23); // in [14, 24]
                const uint32_t mant = (abs & 0x007fffffu) | 0x00800000u;
                uint32_t q = mant >> shift;
                const uint32_t rem = mant & ((1u << shift) - 1u);
                const uint32_t halfway = 1u << (shift - 1);
                if (rem > halfway || (rem == halfway && (q & 1u)))
                    ++q;
                h = uint16_t((sign >> 16) | q);
            }
        }
        else
        {
            uint32_t val = abs + 0xc8000000u; // rebias exponent: -112 << 23
            uint32_t q = val >> 13;
            uint32_t rem = val & 0x1fffu;
            if (rem > 0x1000u || (rem == 0x1000u && (q & 1u)))
                ++q;
            h = uint16_t((sign >> 16) | q);
        }
        // decode h back to double
        const uint32_t hs = uint32_t(h & 0x8000u) << 16;
        const uint32_t he = (h >> 10) & 0x1fu;
        const uint32_t hm = h & 0x3ffu;
        uint32_t out;
        if (he == 0x1fu)
            out = hs | 0x7f800000u | (hm << 13);
        else if (he == 0)
        {
            if (hm == 0)
                out = hs;
            else
            { // normalize subnormal: value = hm * 2^-24 = (m/2^10) * 2^(113-s-127)
                int s = 0;
                uint32_t m = hm;
                while (!(m & 0x400u))
                {
                    m <<= 1;
                    ++s;
                }
                out = hs | (uint32_t(113 - s) << 23) | ((m & 0x3ffu) << 13);
            }
        }
        else
            out = hs | ((he + 112u) << 23) | (hm << 13);
        float g;
        std::memcpy(&g, &out, 4);
        return double(g);
    }

    // f64 -> f32 -> f64 (materialization at f32: rope freq grids, audio
    // attention internals, vision pooler region, MoE routing weights)
    inline double round_f32(double x) { return double(static_cast<float>(x)); }

    inline double round_act(Dtype d, double x)
    {
        switch (d)
        {
        case Dtype::BF16:
            return round_bf16(x);
        case Dtype::F16:
            return round_f16(x);
        default:
            return x;
        }
    }

    inline void round_rows(Dtype d, double *x, int64_t n)
    {
        if (d == Dtype::F64)
            return;
        if (d == Dtype::BF16)
        {
#pragma omp parallel for schedule(static)
            for (int64_t i = 0; i < n; ++i)
                x[i] = round_bf16(x[i]);
        }
        else
        {
#pragma omp parallel for schedule(static)
            for (int64_t i = 0; i < n; ++i)
                x[i] = round_f16(x[i]);
        }
    }

    // torch CPU gelu(approximate="tanh") on bf16/f16 evaluates the tanh formula
    // stepwise in f32 (verified association, py/probe_bf16.py): the value below
    // is bit-identical to torch's f32 computation except for torch's SLEEF-class
    // f32 tanh (ours is the correctly rounded one; ~0.4% of elements differ by
    // 1 f32 ulp, which survives the storage rounding only at ~2^-15 rate).
    // Caller rounds the returned f32-valued double to the storage dtype.
    template <typename TanhF64>
    inline double gelu_tanh_f32chain(double x, TanhF64 tanh_f64)
    {
        const float xf = static_cast<float>(x); // exact: x is bf16/f16-valued
        float i = 0.044715f * xf;
        i = i * xf;
        i = i * xf;
        i = xf + i;
        i = float(1.4142135623730951 * 1.1283791670955126 * 0.5) * i; // f32(kBeta)
        const float t = static_cast<float>(tanh_f64(double(i)));
        const float o = 0.5f * xf * (1.0f + t);
        return double(o);
    }

} // namespace prec
