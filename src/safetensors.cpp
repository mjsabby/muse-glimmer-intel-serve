#include "safetensors.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstring>
#include <filesystem>
#include <set>
#include <stdexcept>

#include "json.h"

namespace fs = std::filesystem;

namespace oracle
{

const char *dtype_name(DType d)
{
    switch (d)
    {
    case DType::BF16:
        return "BF16";
    case DType::F16:
        return "F16";
    case DType::F32:
        return "F32";
    case DType::F64:
        return "F64";
    case DType::I64:
        return "I64";
    case DType::I32:
        return "I32";
    case DType::U8:
        return "U8";
    case DType::BOOL:
        return "BOOL";
    }
    return "?";
}

size_t dtype_size(DType d)
{
    switch (d)
    {
    case DType::BF16:
    case DType::F16:
        return 2;
    case DType::F32:
    case DType::I32:
        return 4;
    case DType::F64:
    case DType::I64:
        return 8;
    case DType::U8:
    case DType::BOOL:
        return 1;
    }
    return 0;
}

namespace
{
DType parse_dtype(const std::string &s)
{
    if (s == "BF16")
    {
        return DType::BF16;
    }
    if (s == "F16")
    {
        return DType::F16;
    }
    if (s == "F32")
    {
        return DType::F32;
    }
    if (s == "F64")
    {
        return DType::F64;
    }
    if (s == "I64")
    {
        return DType::I64;
    }
    if (s == "I32")
    {
        return DType::I32;
    }
    if (s == "U8")
    {
        return DType::U8;
    }
    if (s == "BOOL")
    {
        return DType::BOOL;
    }
    throw std::runtime_error("safetensors: unsupported dtype " + s);
}
} // namespace

const uint16_t *TensorView::bf16() const
{
    if (dtype != DType::BF16)
    {
        throw std::runtime_error("tensor " + name + " is not BF16");
    }
    return reinterpret_cast<const uint16_t *>(data);
}

const float *TensorView::f32() const
{
    if (dtype != DType::F32)
    {
        throw std::runtime_error("tensor " + name + " is not F32");
    }
    return reinterpret_cast<const float *>(data);
}

SafetensorsFile::SafetensorsFile(const std::string &path) : path_(path)
{
    fd_ = ::open(path.c_str(), O_RDONLY);
    if (fd_ < 0)
    {
        throw std::runtime_error("cannot open " + path);
    }
    struct stat st{};
    if (fstat(fd_, &st) != 0)
    {
        ::close(fd_);
        throw std::runtime_error("fstat failed: " + path);
    }
    map_size_ = static_cast<size_t>(st.st_size);
    void *p = ::mmap(nullptr, map_size_, PROT_READ, MAP_SHARED, fd_, 0);
    if (p == MAP_FAILED)
    {
        ::close(fd_);
        throw std::runtime_error("mmap failed: " + path);
    }
    map_ = static_cast<const uint8_t *>(p);

    if (map_size_ < 8)
    {
        throw std::runtime_error("file too small: " + path);
    }
    uint64_t header_len = 0;
    std::memcpy(&header_len, map_, 8); // little-endian host assumed (x86)
    if (8 + header_len > map_size_)
    {
        throw std::runtime_error("bad safetensors header length: " + path);
    }

    Json header = Json::parse(std::string_view(reinterpret_cast<const char *>(map_ + 8), header_len));
    const uint8_t *base = map_ + 8 + header_len;
    size_t data_size = map_size_ - 8 - header_len;

    for (const auto &[name, meta] : header.as_object())
    {
        if (name == "__metadata__")
        {
            continue;
        }
        TensorView tv;
        tv.name = name;
        tv.dtype = parse_dtype(meta.at("dtype").as_string());
        for (const auto &d : meta.at("shape").as_array())
        {
            tv.shape.push_back(d.as_int());
        }
        const auto &offs = meta.at("data_offsets").as_array();
        uint64_t begin = static_cast<uint64_t>(offs[0].as_int());
        uint64_t end = static_cast<uint64_t>(offs[1].as_int());
        if (end < begin || end > data_size)
        {
            throw std::runtime_error("bad data_offsets for " + name);
        }
        tv.data = base + begin;
        tv.nbytes = end - begin;
        if (tv.nbytes != static_cast<size_t>(tv.numel()) * dtype_size(tv.dtype))
        {
            throw std::runtime_error("size mismatch for tensor " + name);
        }
        names_.push_back(name);
        tensors_.emplace(name, std::move(tv));
    }
}

SafetensorsFile::~SafetensorsFile()
{
    if (map_)
    {
        ::munmap(const_cast<uint8_t *>(map_), map_size_);
    }
    if (fd_ >= 0)
    {
        ::close(fd_);
    }
}

const TensorView &SafetensorsFile::tensor(const std::string &name) const
{
    auto it = tensors_.find(name);
    if (it == tensors_.end())
    {
        throw std::runtime_error("tensor not found: " + name + " in " + path_);
    }
    return it->second;
}

Checkpoint::Checkpoint(const std::string &dir)
{
    fs::path index = fs::path(dir) / "model.safetensors.index.json";
    std::error_code ec;
    std::set<std::string> shards;
    if (fs::exists(index, ec))
    {
        Json idx = Json::parse_file(index.string());
        for (const auto &[tensor, shard] : idx.at("weight_map").as_object())
        {
            (void)tensor;
            shards.insert(shard.as_string());
        }
    }
    else if (fs::exists(fs::path(dir) / "model.safetensors", ec))
    {
        shards.insert("model.safetensors");
    }
    else
    {
        throw std::runtime_error("no model.safetensors[.index.json] in " + dir);
    }
    for (const auto &s : shards)
    {
        files_.push_back(std::make_unique<SafetensorsFile>((fs::path(dir) / s).string()));
        for (const auto &n : files_.back()->names())
        {
            by_name_[n] = &files_.back()->tensor(n);
        }
    }
}

bool Checkpoint::has(const std::string &name) const
{
    return by_name_.count(name) > 0;
}

const TensorView &Checkpoint::tensor(const std::string &name) const
{
    auto it = by_name_.find(name);
    if (it == by_name_.end())
    {
        throw std::runtime_error("tensor not found in checkpoint: " + name);
    }
    return *it->second;
}

std::vector<std::string> Checkpoint::names() const
{
    std::vector<std::string> out;
    out.reserve(by_name_.size());
    for (const auto &[k, v] : by_name_)
    {
        (void)v;
        out.push_back(k);
    }
    return out;
}

} // namespace oracle
