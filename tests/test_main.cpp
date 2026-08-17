#include <cstdio>

#include "test_util.h"

int main()
{
    test_json();
    test_npy();
    test_resolver();
    test_json_schema();
    test_sha256();
    test_muse_config();
    std::printf("%d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
