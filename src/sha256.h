// Compact SHA-256 (FIPS 180-4). The round constants K and initial state H0
// are derived at startup from the fractional parts of cube/square roots of
// the first primes (long double has enough mantissa), removing any chance of
// a mistyped constant table; tests validate against the standard "abc" vector.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace oracle
{

class Sha256
{
  public:
    Sha256();
    void update(const void *data, size_t len);
    // Finalizes into 32 bytes. The object must not be updated afterwards.
    void finish(uint8_t out[32]);

    // First 8 bytes of the digest, big-endian, as uint64 (the slot digest).
    static uint64_t digest64(const void *data, size_t len);

  private:
    void process_block(const uint8_t *p);
    uint32_t h_[8];
    uint64_t total_ = 0;
    uint8_t buf_[64];
    size_t buflen_ = 0;
};

std::string sha256_hex(const void *data, size_t len);

} // namespace oracle
