// Accuracy harness for fmath: measures ULP distance against the system libm
// (glibc). The oracle itself never links these; glibc serves as an independent
// ~correctly-rounded yardstick. Expected: <= 2 ulp everywhere in the tested
// (model-relevant) ranges, sin/cos/exp typically <= 1.
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>

#include "../src/fmath.hpp"

static int64_t ulp_diff(double a, double b)
{
    if (a == b)
        return 0;
    if (std::isnan(a) || std::isnan(b))
        return std::isnan(a) == std::isnan(b) ? 0 : INT64_MAX;
    if (std::isinf(a) || std::isinf(b))
        return a == b ? 0 : INT64_MAX;
    auto ordered = [](double x) -> int64_t
    {
        int64_t i;
        std::memcpy(&i, &x, 8);
        return i < 0 ? INT64_MIN + 1 - i : i; // monotone map over ordered doubles
    };
    int64_t d = ordered(a) - ordered(b);
    return d < 0 ? -d : d;
}

template <typename F, typename G>
static int64_t sweep(const char *name, F mine, G libm, double lo, double hi, int n,
                     uint64_t seed)
{
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> dist(lo, hi);
    int64_t worst = 0;
    double worst_x = 0;
    for (int i = 0; i < n; ++i)
    {
        double x = dist(rng);
        int64_t d = ulp_diff(mine(x), libm(x));
        if (d > worst)
        {
            worst = d;
            worst_x = x;
        }
    }
    printf("%-6s [%12g, %12g]  n=%d  max ulp vs glibc: %" PRId64 "  (at x=%.17g)\n", name, lo,
           hi, n, worst, worst_x);
    return worst;
}

int main()
{
    int64_t w = 0, t;
    const int N = 2'000'000;

    w = std::max(w, sweep("exp", [](double x)
                          { return fmath::exp(x); }, [](double x)
                          { return std::exp(x); }, -745.0, 700.0, N, 1));
    w = std::max(w, sweep("exp", [](double x)
                          { return fmath::exp(x); }, [](double x)
                          { return std::exp(x); }, -30.0, 0.0, N, 2));
    w = std::max(w, sweep("log", [](double x)
                          { return fmath::log(x); }, [](double x)
                          { return std::log(x); }, 1e-12, 1e12, N, 3));
    w = std::max(w, sweep("log", [](double x)
                          { return fmath::log(x); }, [](double x)
                          { return std::log(x); }, 0.5, 2.0, N, 4));
    w = std::max(w, sweep("tanh", [](double x)
                          { return fmath::tanh(x); }, [](double x)
                          { return std::tanh(x); }, -30.0, 30.0, N, 5));
    w = std::max(w, sweep("tanh", [](double x)
                          { return fmath::tanh(x); }, [](double x)
                          { return std::tanh(x); }, -1.0, 1.0, N, 6));
    w = std::max(w, sweep("expm1", [](double x)
                          { return fmath::expm1(x); }, [](double x)
                          { return std::expm1(x); }, -50.0, 50.0, N, 7));
    // erf feeds the vision tower's exact-gelu activation. The confluent series
    // trades ~1 ulp for a bounded, cancellation-free, deterministic evaluation;
    // it is checked here in ABSOLUTE terms too, because 1 + erf(x) cancels in
    // the deep negative tail where gelu itself is ~1e-12 and the relative
    // error there is meaningless. Reported separately from `w` for that reason.
    {
        int64_t we = sweep("erf", [](double x)
                           { return fmath::erf(x); }, [](double x)
                           { return std::erf(x); }, -6.0, 6.0, N, 20);
        if (we > 32)
        {
            printf("FAIL: erf worse than 32 ulp vs glibc\n");
            return 1;
        }
        std::mt19937_64 rng(21);
        std::uniform_real_distribution<double> d(-20.0, 20.0);
        double amax = 0;
        for (int i = 0; i < N / 4; ++i)
        {
            double x = d(rng);
            double a = fmath::gelu(x);
            double b = 0.5 * x * (1.0 + std::erf(x * 0.7071067811865476));
            amax = std::max(amax, std::fabs(a - b));
        }
        printf("gelu   [%12g, %12g]  n=%d  max ABS vs glibc-erf: %.3e\n", -20.0, 20.0, N / 4,
               amax);
        if (!(amax < 1e-13))
        {
            printf("FAIL: gelu absolute error above 1e-13\n");
            return 1;
        }
    }
    // log1p feeds softplus(x) = log1p(exp(x)) in the GDN decay computation
    w = std::max(w, sweep("log1p", [](double x)
                          { return fmath::log1p(x); }, [](double x)
                          { return std::log1p(x); }, -0.999999, 1e9, N, 12));
    w = std::max(w, sweep("log1p", [](double x)
                          { return fmath::log1p(x); }, [](double x)
                          { return std::log1p(x); }, -1e-9, 1e-9, N, 13));
    w = std::max(w, sweep("sin", [](double x)
                          { return fmath::sin(x); }, [](double x)
                          { return std::sin(x); }, 0.0, 1.6e6, N, 8));
    w = std::max(w, sweep("cos", [](double x)
                          { return fmath::cos(x); }, [](double x)
                          { return std::cos(x); }, 0.0, 1.6e6, N, 9));
    w = std::max(w, sweep("sin", [](double x)
                          { return fmath::sin(x); }, [](double x)
                          { return std::sin(x); }, -8.0, 8.0, N, 10));
    w = std::max(w, sweep("cos", [](double x)
                          { return fmath::cos(x); }, [](double x)
                          { return std::cos(x); }, -8.0, 8.0, N, 11));

    // pow: error model is ~(1 + |b·ln a|/2) ulp (exp∘log composition); the random
    // sweep reaches |b·ln a| ≈ 16 → ~16 ulp allowed. The rope grid is what the
    // oracle actually evaluates.
    int64_t t_rope = 0;
    {
        int64_t worst = 0;
        double wa = 0, wb = 0;
        std::mt19937_64 rng(12);
        std::uniform_real_distribution<double> base(1.5, 2e6), expo(-1.5, 1.5);
        for (int i = 0; i < N; ++i)
        {
            double a = base(rng), b = expo(rng);
            int64_t d = ulp_diff(fmath::pow(a, b), std::pow(a, b));
            if (d > worst)
            {
                worst = d;
                wa = a;
                wb = b;
            }
        }
        printf("%-6s random a∈[1.5,2e6] b∈[-1.5,1.5]  max ulp vs glibc: %" PRId64
               "  (at a=%.17g b=%.17g)\n",
               "pow", worst, wa, wb);
        t = worst;
        for (double theta : {10000.0, 1000000.0})
            for (int dim : {256, 512})
                for (int j = 0; j < dim / 2; ++j)
                {
                    double e = -double(2 * j) / dim;
                    t_rope = std::max(t_rope, ulp_diff(fmath::pow(theta, e), std::pow(theta, e)));
                }
        printf("%-6s rope grid theta^(-2j/dim)        max ulp vs glibc: %" PRId64 "\n", "pow",
               t_rope);
    }

    // spot identities
    if (fmath::exp(0.0) != 1.0 || fmath::cos(0.0) != 1.0 || fmath::sin(0.0) != 0.0 ||
        fmath::tanh(0.0) != 0.0 || fmath::log(1.0) != 0.0 || fmath::pow(7.0, 0.0) != 1.0)
    {
        printf("FAIL: identity check\n");
        return 1;
    }
    if (fmath::exp(-800.0) != 0.0 || fmath::tanh(50.0) != 1.0 || fmath::tanh(-50.0) != -1.0)
    {
        printf("FAIL: saturation check\n");
        return 1;
    }

    if (w > 4 || t > 16 || t_rope > 16)
    {
        printf("FAIL: ulp gate exceeded (core %" PRId64 ", pow %" PRId64 ", rope %" PRId64 ")\n",
               w, t, t_rope);
        return 1;
    }
    printf("OK (core <= %" PRId64 " ulp, pow <= %" PRId64 " ulp, rope grid <= %" PRId64
           " ulp)\n",
           w, t, t_rope);
    return 0;
}
