// Muse Glimmer on Intel Arc — public interface to the SYCL/oneDNN engine.
//
// Deliberately free of SYCL and oneDNN types so the g++ side of the build (the
// f64 oracle, the tests, the serving library) can talk to this without the
// DPC++ toolchain and without inheriting its numerics defaults. Everything
// behind this header is compiled by icpx in build-gpu/.
//
// Relationship to the referee stack: this engine is a *candidate*, not an
// oracle. `src/muse_glimmer.hpp` (f64) defines the model function, the bf16
// twin defines the deviation band, and this file is gated against them —
// bitwise where the op is elementwise and order-free, envelope-gated where a
// GPU reduction order necessarily differs (GEMM, attention softmax).
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace muse
{
    struct Config;
    struct Weights;
}

namespace muse::gpu
{

    struct DeviceInfo
    {
        std::string name;
        int64_t total_mem = 0;
        int64_t free_mem = 0;
        int compute_units = 0;
    };

    // Wall-clock breakdown of a run, filled when `profile` is set.
    struct Timings
    {
        double upload_s = 0;
        double prefill_s = 0;
        double decode_s = 0;
        int64_t prefill_tokens = 0;
        int64_t decode_tokens = 0;
    };

    struct EngineOptions
    {
        int64_t max_seq = 4096;   // allocation ceiling for the KV caches
        int64_t block = 512;      // widest prefill chunk
        int gpus = 1;             // physical cards to use
        // Tensor-parallel shards. 0 means "one per card". Setting it larger
        // than `gpus` puts several shards on one card, which changes only
        // WHERE the arithmetic happens: the split is what fixes the reduction
        // order, so --shards 2 --gpus 1 and --shards 2 --gpus 2 are bitwise
        // identical and that equality is the dual-GPU gate.
        int shards = 0;
        bool profile = false;
        bool verbose = false;
        // Force the oneDNN matmul off and use the hand-written SYCL GEMV
        // everywhere. Diagnostic: the two must agree inside the envelope.
        bool no_dnnl = false;
    };

    // Enumerate the Level-Zero GPUs the process can see. Never throws; an
    // empty vector means "no usable device", which callers must report rather
    // than silently fall back to CPU.
    std::vector<DeviceInfo> enumerate_devices();

    class Engine
    {
    public:
        virtual ~Engine() = default;

        // Binds `w` (BF16 mmap views over the safetensors shards) onto the
        // GPUs and allocates the caches. Throws on allocation failure — memory
        // must fail at startup, not mid-request.
        static std::unique_ptr<Engine> create(const Config &c, const Weights &w,
                                              const EngineOptions &opt);

        // Prefill `ids` and return the last position's logits (post-softcap),
        // `vocab_size` floats.
        virtual void prefill(const std::vector<int64_t> &ids, float *logits_last) = 0;

        // Append one token against the existing cache; same logit contract.
        virtual void decode_step(int64_t id, float *logits_last) = 0;

        // Logical tokens currently in the KV cache.
        virtual int64_t cache_len() const = 0;
        virtual void reset_cache() = 0;

        virtual const Timings &timings() const = 0;
        virtual void report_profile(std::FILE *f) const = 0;
    };

} // namespace muse::gpu
