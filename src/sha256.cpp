#include "sha256.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace oracle
{

namespace
{

const uint32_t *k_table()
{
    static uint32_t K[64];
    static bool init = [] {
        int primes[64];
        int count = 0;
        for (int n = 2; count < 64; ++n)
        {
            bool prime = true;
            for (int d = 2; d * d <= n; ++d)
            {
                if (n % d == 0)
                {
                    prime = false;
                    break;
                }
            }
            if (prime)
            {
                primes[count++] = n;
            }
        }
        for (int i = 0; i < 64; ++i)
        {
            long double r = cbrtl(static_cast<long double>(primes[i]));
            K[i] = static_cast<uint32_t>((r - floorl(r)) * 4294967296.0L);
        }
        return true;
    }();
    (void)init;
    return K;
}

void initial_state(uint32_t h[8])
{
    int primes[8];
    int count = 0;
    for (int n = 2; count < 8; ++n)
    {
        bool prime = true;
        for (int d = 2; d * d <= n; ++d)
        {
            if (n % d == 0)
            {
                prime = false;
                break;
            }
        }
        if (prime)
        {
            primes[count++] = n;
        }
    }
    for (int i = 0; i < 8; ++i)
    {
        long double r = sqrtl(static_cast<long double>(primes[i]));
        h[i] = static_cast<uint32_t>((r - floorl(r)) * 4294967296.0L);
    }
}

inline uint32_t rotr(uint32_t x, int n)
{
    return (x >> n) | (x << (32 - n));
}

} // namespace

Sha256::Sha256()
{
    initial_state(h_);
}

void Sha256::process_block(const uint8_t *p)
{
    const uint32_t *K = k_table();
    uint32_t w[64];
    for (int i = 0; i < 16; ++i)
    {
        w[i] = (static_cast<uint32_t>(p[4 * i]) << 24) | (static_cast<uint32_t>(p[4 * i + 1]) << 16) |
               (static_cast<uint32_t>(p[4 * i + 2]) << 8) | static_cast<uint32_t>(p[4 * i + 3]);
    }
    for (int i = 16; i < 64; ++i)
    {
        uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
        uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    uint32_t a = h_[0], b = h_[1], c = h_[2], d = h_[3], e = h_[4], f = h_[5], g = h_[6], h = h_[7];
    for (int i = 0; i < 64; ++i)
    {
        uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
        uint32_t ch = (e & f) ^ (~e & g);
        uint32_t t1 = h + S1 + ch + K[i] + w[i];
        uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
        uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t t2 = S0 + maj;
        h = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }
    h_[0] += a;
    h_[1] += b;
    h_[2] += c;
    h_[3] += d;
    h_[4] += e;
    h_[5] += f;
    h_[6] += g;
    h_[7] += h;
}

void Sha256::update(const void *data, size_t len)
{
    const uint8_t *p = static_cast<const uint8_t *>(data);
    total_ += len;
    if (buflen_ > 0)
    {
        size_t take = std::min(len, sizeof(buf_) - buflen_);
        std::memcpy(buf_ + buflen_, p, take);
        buflen_ += take;
        p += take;
        len -= take;
        if (buflen_ == 64)
        {
            process_block(buf_);
            buflen_ = 0;
        }
    }
    while (len >= 64)
    {
        process_block(p);
        p += 64;
        len -= 64;
    }
    if (len > 0)
    {
        std::memcpy(buf_, p, len);
        buflen_ = len;
    }
}

void Sha256::finish(uint8_t out[32])
{
    uint64_t bits = total_ * 8;
    uint8_t pad = 0x80;
    update(&pad, 1);
    uint8_t zero = 0;
    while (buflen_ != 56)
    {
        update(&zero, 1);
    }
    uint8_t lenbe[8];
    for (int i = 0; i < 8; ++i)
    {
        lenbe[i] = static_cast<uint8_t>(bits >> (56 - 8 * i));
    }
    // bypass total_ accounting for the length field by writing directly
    std::memcpy(buf_ + 56, lenbe, 8);
    process_block(buf_);
    buflen_ = 0;
    for (int i = 0; i < 8; ++i)
    {
        out[4 * i] = static_cast<uint8_t>(h_[i] >> 24);
        out[4 * i + 1] = static_cast<uint8_t>(h_[i] >> 16);
        out[4 * i + 2] = static_cast<uint8_t>(h_[i] >> 8);
        out[4 * i + 3] = static_cast<uint8_t>(h_[i]);
    }
}

uint64_t Sha256::digest64(const void *data, size_t len)
{
    Sha256 s;
    s.update(data, len);
    uint8_t d[32];
    s.finish(d);
    uint64_t v = 0;
    for (int i = 0; i < 8; ++i)
    {
        v = (v << 8) | d[i];
    }
    return v;
}

std::string sha256_hex(const void *data, size_t len)
{
    Sha256 s;
    s.update(data, len);
    uint8_t d[32];
    s.finish(d);
    static const char *hex = "0123456789abcdef";
    std::string out;
    for (int i = 0; i < 32; ++i)
    {
        out.push_back(hex[d[i] >> 4]);
        out.push_back(hex[d[i] & 15]);
    }
    return out;
}

} // namespace oracle
