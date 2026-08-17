#include <cstring>
#include <string>

#include "sha256.h"
#include "test_util.h"

using namespace oracle;

void test_sha256()
{
    // FIPS 180-4 test vectors.
    CHECK(sha256_hex("abc", 3) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    CHECK(sha256_hex("", 0) == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    std::string two_blocks = "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
    CHECK(sha256_hex(two_blocks.data(), two_blocks.size()) ==
          "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
    // 1M 'a' (streamed in odd chunk sizes to exercise buffering).
    {
        Sha256 s;
        std::string chunk(997, 'a');
        size_t total = 0;
        while (total + chunk.size() <= 1000000)
        {
            s.update(chunk.data(), chunk.size());
            total += chunk.size();
        }
        s.update(chunk.data(), 1000000 - total);
        uint8_t d[32];
        s.finish(d);
        static const uint8_t want[32] = {0xcd, 0xc7, 0x6e, 0x5c, 0x99, 0x14, 0xfb, 0x92, 0x81, 0xa1, 0xc7,
                                         0xe2, 0x84, 0xd7, 0x3e, 0x67, 0xf1, 0x80, 0x9a, 0x48, 0xa4, 0x97,
                                         0x20, 0x0e, 0x04, 0x6d, 0x39, 0xcc, 0xc7, 0x11, 0x2c, 0xd0};
        CHECK(std::memcmp(d, want, 32) == 0);
    }
    // digest64 = first 8 bytes big-endian.
    CHECK(Sha256::digest64("abc", 3) == 0xba7816bf8f01cfeaULL);
}
