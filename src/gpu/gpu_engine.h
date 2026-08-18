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
    namespace dflash
    {
        struct Config;
        struct Weights;
    }
    namespace vision
    {
        struct Config;
        struct Weights;
        struct Grid;
    }
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
        // Opt-in prefill attention tier: q.k and p.v on the matrix engines,
        // with the softmax max/sum taken per TILE instead of per key. Much
        // faster and a DIFFERENT numerical contract from the twin — it is
        // gated on the logit envelope, never bitwise. Decode is unaffected.
        bool flash_prefill = false;
        // Split-K decode attention. Needed past ~2K context, where the exact
        // kernel's single-query parallelism (nqs sub-groups) leaves the card
        // idle. Looser contract, same as flash_prefill.
        bool flash_decode = false;
        // Q8_0 weight tier for the text model: quantized at load from the same
        // BF16 checkpoint with llama.cpp's quantize_row_q8_0_ref semantics.
        // Halves weight VRAM. A SEPARATE ACCURACY TIER, not the bf16 band --
        // gated against the f64 oracle, never against the twin bitwise.
        bool q8 = false;
        // Q8_0 for the DFlash drafter, independently of the target: the build
        // plan's recommended shape is a BF16 target with a quantized drafter.
        bool q8_assistant = false;
        // Target hidden-state layers the DFlash drafter reads (its
        // `target_layer_ids`). Must be set before create(): the taps are
        // captured on the way through the forward pass, not recomputed.
        std::vector<int64_t> tap_layers;
        // The drafter's block_size, needed before create(): it sizes the
        // all-row logits buffer that speculative verification writes into.
        int64_t spec_block = 1;
        // The drafter sliding_window, so the tap ring can be sized before
        // create(). 0 means no drafter.
        int64_t tap_window = 0;
    };

    // Result of a speculative run.
    struct SpecResult
    {
        std::vector<int64_t> tokens; // everything generated
        int64_t rounds = 0, drafted = 0, accepted = 0;
        double seconds = 0, draft_s = 0, verify_s = 0;
        std::vector<int> per_round;
    };

    // One DFlash drafting round: `block_size - 1` proposals anchored on the
    // target's bonus token.
    struct DraftResult
    {
        std::vector<int64_t> tokens;
        int64_t anchor = -1;
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

        // Binds the drafter onto the same shards as the target. The target's
        // embedding and lm_head are reused (the drafter embeds with the RAW
        // table and heads with the BARE lm_head), so only the drafter's own
        // tensors are uploaded.
        virtual void bind_drafter(const dflash::Config &dc, const dflash::Weights &dw) = 0;

        // One drafting round against the taps captured by the last forward
        // pass. `n` context positions, anchored on `anchor` at absolute
        // position `pos0`.
        virtual DraftResult draft(int64_t n, int64_t anchor, int64_t pos0) = 0;

        // Greedy speculative decoding: draft a block, verify it with ONE target
        // forward over the candidates, accept the matching prefix. `rounds`
        // rounds against a cache already holding `prompt`.
        virtual SpecResult spec_decode(const std::vector<int64_t> &prompt, int64_t anchor,
                                       int64_t rounds) = 0;

        // Vision tower. `max_patches` sizes the scratch once, so an image
        // larger than it is a startup-shaped error rather than a mid-request
        // allocation.
        virtual void bind_vision(const vision::Config &vc, const vision::Weights &vw,
                                 int64_t max_patches) = 0;
        // pixel_values [N, patch_dim] -> [M, text hidden], ready to scatter
        virtual std::vector<float> vision_features(const double *pixels,
                                                   const std::vector<vision::Grid> &grids) = 0;
        // Scattered into the residual stream AFTER the embedding norm, which is
        // where the reference puts them.
        virtual void set_vision_embeds(const std::vector<float> &feats,
                                       const std::vector<int64_t> &positions) = 0;

        // "Seal after load": every device allocation after this point should
        // have been reserved up front, because the engine serves one request at
        // a time and a mid-request allocation is a mid-request OOM. mode 1
        // logs and continues (enumerate them all in one run), mode 2 throws on
        // the first. Turns "I believe the footprint is static" into a checked
        // property.
        virtual void seal_allocs(int mode) = 0;

        virtual const Timings &timings() const = 0;
        virtual void report_profile(std::FILE *f) const = 0;
    };

} // namespace muse::gpu
