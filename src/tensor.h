// Tiny f64 tensor + .npy I/O (the interchange format with the JAX harness).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace oracle
{

struct Tensor
{
    std::vector<int64_t> shape;
    std::vector<double> data;

    Tensor() = default;
    explicit Tensor(std::vector<int64_t> s) : shape(std::move(s))
    {
        data.resize(static_cast<size_t>(numel()));
    }

    int64_t numel() const
    {
        int64_t n = 1;
        for (int64_t d : shape)
        {
            n *= d;
        }
        return n;
    }
    double *ptr()
    {
        return data.data();
    }
    const double *ptr() const
    {
        return data.data();
    }
};

// .npy v1.0, little-endian. save: '<f8' only. load: '<f8', '<f4', '<i8',
// '<i4', '|u1' (all widened to double; u1 values stay 0..255).
void save_npy(const std::string &path, const Tensor &t);
// descr_out (optional) receives the stored numpy descr (e.g. "|u1", "<f4").
Tensor load_npy(const std::string &path, std::string *descr_out = nullptr);

// Raw int64 vector as .npy (token ids for the harness).
void save_npy_i64(const std::string &path, const std::vector<int64_t> &v);

} // namespace oracle
