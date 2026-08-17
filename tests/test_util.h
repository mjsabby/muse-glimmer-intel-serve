#pragma once
#include <cmath>
#include <cstdio>
#include <string>

inline int g_failures = 0;
inline int g_checks = 0;

#define CHECK(cond)                                                                                                    \
    do                                                                                                                 \
    {                                                                                                                  \
        ++g_checks;                                                                                                    \
        if (!(cond))                                                                                                   \
        {                                                                                                              \
            ++g_failures;                                                                                              \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);                                                \
        }                                                                                                              \
    } while (0)

#define CHECK_NEAR(a, b, tol)                                                                                          \
    do                                                                                                                 \
    {                                                                                                                  \
        ++g_checks;                                                                                                    \
        double _a = (a), _b = (b);                                                                                     \
        if (!(std::fabs(_a - _b) <= (tol)))                                                                            \
        {                                                                                                              \
            ++g_failures;                                                                                              \
            std::printf("FAIL %s:%d: |%.17g - %.17g| > %g\n", __FILE__, __LINE__, _a, _b, (double)(tol));              \
        }                                                                                                              \
    } while (0)

void test_json();
void test_npy();
void test_resolver();
void test_json_schema();
void test_sha256();
void test_muse_config();
