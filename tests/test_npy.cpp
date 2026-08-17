#include <cstdlib>

#include "tensor.h"
#include "test_util.h"

using namespace oracle;

void test_npy()
{
    const char *tmp = std::getenv("TMPDIR");
    std::string dir = tmp ? tmp : "/tmp";
    std::string p = dir + "/oracle_test.npy";
    Tensor t({2, 3});
    for (size_t i = 0; i < 6; ++i)
    {
        t.data[i] = 0.1 * static_cast<double>(i) - 0.25;
    }
    save_npy(p, t);
    Tensor r = load_npy(p);
    CHECK(r.shape == t.shape);
    for (size_t i = 0; i < 6; ++i)
    {
        CHECK(r.data[i] == t.data[i]);
    }

    save_npy_i64(p, {5, -1, 262143});
    Tensor ri = load_npy(p);
    CHECK(ri.shape.size() == 1 && ri.shape[0] == 3);
    CHECK(ri.data[2] == 262143.0);
}
