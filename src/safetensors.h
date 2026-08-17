// Memory-mapped safetensors reader (single file or sharded via index.json).
// Weights are exposed as raw typed views into the mapped file; the oracle
// converts bf16 -> f64 lazily inside kernels (conversion is exact).
#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace oracle
{

enum class DType
{
    BF16,
    F16,
    F32,
    F64,
    I64,
    I32,
    U8,
    BOOL
};

const char *dtype_name(DType d);
size_t dtype_size(DType d);

struct TensorView
{
    std::string name;
    DType dtype = DType::BF16;
    std::vector<int64_t> shape;
    const uint8_t *data = nullptr; // valid while the owning SafetensorsFile lives
    size_t nbytes = 0;

    int64_t numel() const
    {
        int64_t n = 1;
        for (int64_t d : shape)
        {
            n *= d;
        }
        return n;
    }
    const uint16_t *bf16() const; // throws unless dtype == BF16
    const float *f32() const;     // throws unless dtype == F32
};

class SafetensorsFile
{
  public:
    explicit SafetensorsFile(const std::string &path);
    ~SafetensorsFile();
    SafetensorsFile(const SafetensorsFile &) = delete;
    SafetensorsFile &operator=(const SafetensorsFile &) = delete;

    const std::vector<std::string> &names() const
    {
        return names_;
    }
    bool has(const std::string &name) const
    {
        return tensors_.count(name) > 0;
    }
    const TensorView &tensor(const std::string &name) const;
    const std::string &path() const
    {
        return path_;
    }

  private:
    std::string path_;
    int fd_ = -1;
    const uint8_t *map_ = nullptr;
    size_t map_size_ = 0;
    std::map<std::string, TensorView> tensors_;
    std::vector<std::string> names_;
};

// A model checkpoint: either one model.safetensors or an index + shards.
class Checkpoint
{
  public:
    // dir must contain model.safetensors.index.json or model.safetensors.
    explicit Checkpoint(const std::string &dir);

    bool has(const std::string &name) const;
    const TensorView &tensor(const std::string &name) const; // throws if absent
    std::vector<std::string> names() const;                  // sorted

  private:
    std::vector<std::unique_ptr<SafetensorsFile>> files_;
    std::map<std::string, const TensorView *> by_name_;
};

} // namespace oracle
