// Memory-mapped safetensors reader. Header = 8-byte LE length + JSON; tensor data
// follows, addressed by data_offsets relative to the end of the header.
#pragma once

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdint>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(__AVX512F__)
#include <immintrin.h>
#endif

#include "json.hpp"

namespace st
{

    enum class DType
    {
        BF16,
        F16,
        F32,
        F64,
        I64,
        U8,
        Other
    };

    inline DType dtype_from_str(const std::string &s)
    {
        if (s == "BF16")
            return DType::BF16;
        if (s == "F16")
            return DType::F16;
        if (s == "F32")
            return DType::F32;
        if (s == "F64")
            return DType::F64;
        if (s == "I64")
            return DType::I64;
        if (s == "U8")
            return DType::U8;
        return DType::Other;
    }

    inline size_t dtype_size(DType d)
    {
        switch (d)
        {
        case DType::BF16:
        case DType::F16:
            return 2;
        case DType::F32:
            return 4;
        case DType::F64:
        case DType::I64:
            return 8;
        case DType::U8:
            return 1;
        default:
            throw std::runtime_error("unsupported dtype");
        }
    }

    struct Tensor
    {
        std::string name;
        DType dtype = DType::Other;
        std::vector<int64_t> shape;
        const uint8_t *data = nullptr;
        int64_t nbytes = 0;

        int64_t numel() const
        {
            int64_t n = 1;
            for (int64_t s : shape)
                n *= s;
            return n;
        }
        int64_t dim(size_t i) const { return shape.at(i); }
    };

    // exact conversions into f64 (bf16/f16/f32 are all subsets of f64)
    inline double bf16_to_f64(uint16_t v)
    {
        uint32_t bits = uint32_t(v) << 16;
        float f;
        std::memcpy(&f, &bits, 4);
        return double(f);
    }

    inline double f16_to_f64(uint16_t h)
    {
        uint32_t sign = (h >> 15) & 1, exp = (h >> 10) & 0x1F, man = h & 0x3FF;
        uint32_t fbits;
        if (exp == 0)
        {
            if (man == 0)
                fbits = sign << 31;
            else
            {
                int e = -1;
                do
                {
                    man <<= 1;
                    ++e;
                } while (!(man & 0x400));
                fbits = (sign << 31) | uint32_t(127 - 15 - e) << 23 | ((man & 0x3FF) << 13);
            }
        }
        else if (exp == 0x1F)
        {
            fbits = (sign << 31) | 0x7F800000u | (man << 13);
        }
        else
        {
            fbits = (sign << 31) | ((exp - 15 + 127) << 23) | (man << 13);
        }
        float f;
        std::memcpy(&f, &fbits, 4);
        return double(f);
    }

    // copy one contiguous run of `count` elements starting at flat element index
    // `offset` into an f64 buffer — the only accessor the model needs.
    // All conversions are EXACT (bf16/f16/f32 ⊂ f64), so the AVX-512 fast paths
    // below are bitwise-identical to the scalar loops — no order, no rounding.
    inline void to_f64(const Tensor &t, int64_t offset, int64_t count, double *dst)
    {
        const uint8_t *base = t.data + size_t(offset) * dtype_size(t.dtype);
        int64_t i = 0;
        switch (t.dtype)
        {
        case DType::BF16:
#if defined(__AVX512F__)
            for (; i + 8 <= count; i += 8)
            {
                __m128i h = _mm_loadu_si128(reinterpret_cast<const __m128i *>(base + 2 * i));
                __m256i w = _mm256_slli_epi32(_mm256_cvtepu16_epi32(h), 16);
                _mm512_storeu_pd(dst + i, _mm512_cvtps_pd(_mm256_castsi256_ps(w)));
            }
#endif
            for (; i < count; ++i)
            {
                uint16_t v;
                std::memcpy(&v, base + 2 * i, 2);
                dst[i] = bf16_to_f64(v);
            }
            break;
        case DType::F16:
#if defined(__AVX512F__) && defined(__F16C__)
            for (; i + 8 <= count; i += 8)
            {
                __m128i h = _mm_loadu_si128(reinterpret_cast<const __m128i *>(base + 2 * i));
                _mm512_storeu_pd(dst + i, _mm512_cvtps_pd(_mm256_cvtph_ps(h)));
            }
#endif
            for (; i < count; ++i)
            {
                uint16_t v;
                std::memcpy(&v, base + 2 * i, 2);
                dst[i] = f16_to_f64(v);
            }
            break;
        case DType::F32:
#if defined(__AVX512F__)
            for (; i + 8 <= count; i += 8)
                _mm512_storeu_pd(
                    dst + i,
                    _mm512_cvtps_pd(_mm256_loadu_ps(reinterpret_cast<const float *>(base + 4 * i))));
#endif
            for (; i < count; ++i)
            {
                float f;
                std::memcpy(&f, base + 4 * i, 4);
                dst[i] = double(f);
            }
            break;
        case DType::F64:
            std::memcpy(dst, base, size_t(count) * 8);
            break;
        default:
            throw std::runtime_error(t.name + ": unsupported dtype for f64 conversion");
        }
    }

    // element accessor for integer tensors (e.g. masked_embedding.token_ordering)
    inline int64_t i64_at(const Tensor &t, int64_t idx)
    {
        if (t.dtype != DType::I64)
            throw std::runtime_error(t.name + ": expected I64");
        int64_t v;
        std::memcpy(&v, t.data + size_t(idx) * 8, 8);
        return v;
    }

    class SafeTensors
    {
    public:
        explicit SafeTensors(const std::string &path) : path_(path)
        {
            int fd = ::open(path.c_str(), O_RDONLY);
            if (fd < 0)
                throw std::runtime_error("cannot open " + path);
            struct stat sb{};
            if (fstat(fd, &sb) != 0)
            {
                ::close(fd);
                throw std::runtime_error("fstat failed: " + path);
            }
            size_ = size_t(sb.st_size);
            map_ = mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd, 0);
            ::close(fd);
            if (map_ == MAP_FAILED)
                throw std::runtime_error("mmap failed: " + path);
            // perf hints only — no effect on results, harmless where unsupported:
            // hugepages cut dTLB walks while streaming the weights every forward,
            // WILLNEED prefaults/readaheads the file on cold page cache
#ifdef MADV_HUGEPAGE
            madvise(map_, size_, MADV_HUGEPAGE);
#endif
#ifdef MADV_WILLNEED
            madvise(map_, size_, MADV_WILLNEED);
#endif

            if (size_ < 8)
                throw std::runtime_error("file too small: " + path);
            uint64_t hlen;
            std::memcpy(&hlen, map_, 8);
            if (8 + hlen > size_)
                throw std::runtime_error("bad safetensors header length: " + path);

            auto header = minijson::parse(static_cast<const char *>(map_) + 8, hlen);
            const uint8_t *data_base = static_cast<const uint8_t *>(map_) + 8 + hlen;
            for (auto &[name, val] : header->obj)
            {
                if (name == "__metadata__")
                    continue;
                Tensor t;
                t.name = name;
                t.dtype = dtype_from_str(val->at("dtype").as_str());
                for (size_t i = 0; i < val->at("shape").size(); ++i)
                    t.shape.push_back(val->at("shape")[i].as_int());
                int64_t begin = val->at("data_offsets")[0].as_int();
                int64_t end = val->at("data_offsets")[1].as_int();
                t.data = data_base + begin;
                t.nbytes = end - begin;
                if (8 + hlen + uint64_t(end) > size_)
                    throw std::runtime_error("tensor data out of range: " + name);
                if (t.nbytes != t.numel() * int64_t(dtype_size(t.dtype)))
                    throw std::runtime_error("tensor size mismatch: " + name);
                tensors_[name] = std::move(t);
            }
        }

        ~SafeTensors()
        {
            if (map_ && map_ != MAP_FAILED)
                munmap(map_, size_);
        }
        SafeTensors(const SafeTensors &) = delete;
        SafeTensors &operator=(const SafeTensors &) = delete;

        bool has(const std::string &name) const { return tensors_.count(name) > 0; }
        const Tensor &tensor(const std::string &name) const
        {
            auto it = tensors_.find(name);
            if (it == tensors_.end())
                throw std::runtime_error(path_ + ": no tensor " + name);
            return it->second;
        }
        const std::map<std::string, Tensor> &tensors() const { return tensors_; }

    private:
        std::string path_;
        void *map_ = nullptr;
        size_t size_ = 0;
        std::map<std::string, Tensor> tensors_;
    };

} // namespace st
