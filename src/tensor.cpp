#include "tensor.h"

#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace oracle
{

namespace
{

void write_npy_header(std::FILE *f, const std::string &descr, const std::vector<int64_t> &shape)
{
    std::string dict = "{'descr': '" + descr + "', 'fortran_order': False, 'shape': (";
    for (size_t i = 0; i < shape.size(); ++i)
    {
        dict += std::to_string(shape[i]);
        if (shape.size() == 1 || i + 1 < shape.size())
        {
            dict += ",";
        }
        if (i + 1 < shape.size())
        {
            dict += " ";
        }
    }
    dict += "), }";
    size_t base = 6 + 2 + 2; // magic + version + header-len field
    size_t total = base + dict.size() + 1;
    size_t pad = (64 - (total % 64)) % 64;
    dict += std::string(pad, ' ');
    dict += '\n';
    uint16_t hlen = static_cast<uint16_t>(dict.size());
    std::fwrite("\x93NUMPY\x01\x00", 1, 8, f);
    std::fwrite(&hlen, 2, 1, f);
    std::fwrite(dict.data(), 1, dict.size(), f);
}

std::string parse_header_field(const std::string &h, const std::string &key)
{
    auto k = h.find("'" + key + "'");
    if (k == std::string::npos)
    {
        throw std::runtime_error("npy: missing header key " + key);
    }
    auto colon = h.find(':', k);
    auto rest = h.substr(colon + 1);
    size_t i = 0;
    while (i < rest.size() && rest[i] == ' ')
    {
        ++i;
    }
    return rest.substr(i);
}

} // namespace

void save_npy(const std::string &path, const Tensor &t)
{
    std::FILE *f = std::fopen(path.c_str(), "wb");
    if (!f)
    {
        throw std::runtime_error("cannot write " + path);
    }
    write_npy_header(f, "<f8", t.shape);
    size_t n = std::fwrite(t.data.data(), 8, t.data.size(), f);
    std::fclose(f);
    if (n != t.data.size())
    {
        throw std::runtime_error("short write: " + path);
    }
}

void save_npy_i64(const std::string &path, const std::vector<int64_t> &v)
{
    std::FILE *f = std::fopen(path.c_str(), "wb");
    if (!f)
    {
        throw std::runtime_error("cannot write " + path);
    }
    write_npy_header(f, "<i8", {static_cast<int64_t>(v.size())});
    size_t n = std::fwrite(v.data(), 8, v.size(), f);
    std::fclose(f);
    if (n != v.size())
    {
        throw std::runtime_error("short write: " + path);
    }
}

Tensor load_npy(const std::string &path, std::string *descr_out)
{
    std::FILE *f = std::fopen(path.c_str(), "rb");
    if (!f)
    {
        throw std::runtime_error("cannot open " + path);
    }
    char magic[8];
    if (std::fread(magic, 1, 8, f) != 8 || std::memcmp(magic, "\x93NUMPY", 6) != 0)
    {
        std::fclose(f);
        throw std::runtime_error("not an npy file: " + path);
    }
    uint32_t hlen = 0;
    if (magic[6] == 1)
    {
        uint16_t h16 = 0;
        if (std::fread(&h16, 2, 1, f) != 1)
        {
            std::fclose(f);
            throw std::runtime_error("npy: bad header");
        }
        hlen = h16;
    }
    else
    {
        if (std::fread(&hlen, 4, 1, f) != 1)
        {
            std::fclose(f);
            throw std::runtime_error("npy: bad header");
        }
    }
    std::string header(hlen, '\0');
    if (std::fread(header.data(), 1, hlen, f) != hlen)
    {
        std::fclose(f);
        throw std::runtime_error("npy: bad header");
    }

    std::string descr = parse_header_field(header, "descr");
    descr = descr.substr(1, descr.find('\'', 1) - 1);
    if (descr_out)
    {
        *descr_out = descr;
    }
    std::string order = parse_header_field(header, "fortran_order");
    if (order.rfind("False", 0) != 0)
    {
        std::fclose(f);
        throw std::runtime_error("npy: fortran order unsupported");
    }
    std::string shape_s = parse_header_field(header, "shape");

    Tensor t;
    {
        auto open = shape_s.find('(');
        auto close = shape_s.find(')');
        std::string inner = shape_s.substr(open + 1, close - open - 1);
        int64_t cur = -1;
        for (char c : inner)
        {
            if (c >= '0' && c <= '9')
            {
                cur = (cur < 0 ? 0 : cur) * 10 + (c - '0');
            }
            else if (cur >= 0)
            {
                t.shape.push_back(cur);
                cur = -1;
            }
        }
        if (cur >= 0)
        {
            t.shape.push_back(cur);
        }
    }
    size_t n = static_cast<size_t>(t.numel());
    t.data.resize(n);

    bool ok = true;
    if (descr == "<f8")
    {
        ok = std::fread(t.data.data(), 8, n, f) == n;
    }
    else if (descr == "<i8")
    {
        std::vector<int64_t> tmp(n);
        ok = std::fread(tmp.data(), 8, n, f) == n;
        for (size_t i = 0; i < n; ++i)
        {
            t.data[i] = static_cast<double>(tmp[i]);
        }
    }
    else if (descr == "<i4")
    {
        std::vector<int32_t> tmp(n);
        ok = std::fread(tmp.data(), 4, n, f) == n;
        for (size_t i = 0; i < n; ++i)
        {
            t.data[i] = static_cast<double>(tmp[i]);
        }
    }
    else if (descr == "<f4")
    {
        std::vector<float> tmp(n);
        ok = std::fread(tmp.data(), 4, n, f) == n;
        for (size_t i = 0; i < n; ++i)
        {
            t.data[i] = static_cast<double>(tmp[i]);
        }
    }
    else if (descr == "|u1")
    {
        // Raw uint8 image rasters: values are carried verbatim (0..255); the
        // caller applies the reference's f32 /255 rescale.
        std::vector<uint8_t> tmp(n);
        ok = std::fread(tmp.data(), 1, n, f) == n;
        for (size_t i = 0; i < n; ++i)
        {
            t.data[i] = static_cast<double>(tmp[i]);
        }
    }
    else
    {
        std::fclose(f);
        throw std::runtime_error("npy: unsupported descr " + descr + " in " + path);
    }
    std::fclose(f);
    if (!ok)
    {
        throw std::runtime_error("npy: short read " + path);
    }
    return t;
}

} // namespace oracle
