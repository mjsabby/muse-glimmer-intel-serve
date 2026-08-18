// Muse Glimmer 30B on Intel Arc — SYCL + oneDNN engine.
//
// This is a candidate under the referee stack, not an oracle. Its op order is
// `src/bf16exec.hpp`'s, op for op and rounding for rounding, because that is
// the thing it is gated against; where a GPU reduction cannot reproduce a CPU
// reduction order (the GEMMs, the attention softmax) the gate is the logit
// envelope rather than bitwise equality. Any change here that alters *which*
// values are materialized at BF16 — as opposed to the order a sum is taken in
// — is a contract change and has to be made in the twin first.
//
// ---------------------------------------------------------------------------
// Two layout decisions, both measured on this box (Arc Pro B70, oneDNN 3.11.2)
// rather than assumed:
//
// 1. Weights stay in checkpoint order, W[out, in] row-major, and are never
//    transposed. oneDNN is driven as Y[out, T] = W[out, in] * X[in, T], which
//    makes the *weight* the natural row-major operand. The obvious alternative
//    — Y[T, out] = X[T, in] * W^T, reading the weight through a transposed
//    stride — measured 133-153 TFLOP/s against this form's 173, and
//    pre-transposing the weights instead would cost a 55 GiB shuffle at load
//    for the same 173. So: no transpose, full speed.
//
// 2. Every activation is therefore dim-major, [dim, T], with the allocated
//    block as the leading dimension. This is also the better layout for the
//    elementwise kernels: a work-item per token means adjacent lanes touch
//    adjacent addresses. It costs nothing at decode, where T = 1 makes
//    dim-major and row-major the same bytes.
#include "gpu/gpu_engine.h"

#include "bf16exec.hpp"
#include "dflash.hpp"
#include "vision.hpp"
#include "muse_glimmer.hpp"

#include <sycl/sycl.hpp>

#if ORACLE_GPU_DNNL
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>
#endif

#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <unordered_map>
#include <stdexcept>

namespace muse::gpu
{
    namespace
    {

        // ------------------------------------------------------------- scalars
        //
        // Bit-identical to muse::bf16::f32_to_bf16 / bf16_to_f32. Written out
        // rather than reused because these have to be device code.

        inline float bf2f(uint16_t v)
        {
            uint32_t bits = uint32_t(v) << 16;
            float f;
            std::memcpy(&f, &bits, 4);
            return f;
        }
        inline uint16_t f2bf(float f)
        {
            uint32_t u;
            std::memcpy(&u, &f, 4);
            if (f != f)
                return 0x7fc0;
            return uint16_t((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
        }
        // one BF16 materialization, value back in f32 — the twin's
        // `bf16_to_f32(f32_to_bf16(x))` idiom, which appears once per
        // nn.Linear output and once per elementwise op
        inline float rb(float f) { return bf2f(f2bf(f)); }

        // host-side bf16 round-trip (device code uses rb())
        inline float bf16_rt(float f) { return bf2f(muse::bf16::f32_to_bf16(f)); }

        constexpr int SG = 32; // sub-group width used by every reduction kernel

        [[noreturn]] void die(const std::string &m) { throw std::runtime_error("gpu: " + m); }

        // --------------------------------------------------------- device side

        // A weight, in whichever tier it was loaded at. Q8_0 keeps the int8
        // quants and the per-32 f16 scales in SEPARATE arrays rather than
        // llama.cpp's interleaved 34-byte block: the GEMV reads a whole row of
        // quants contiguously and touches the scales 32x less often, and an
        // interleaved layout would break both accesses.
        struct QW
        {
            const uint16_t *w = nullptr;  // bf16 weights, or null in the Q8 tier
            const int8_t *qs = nullptr;   // int8 quants [out, in]
            const uint16_t *d = nullptr;  // f16 scales  [out, in/32]
            bool q8() const { return qs != nullptr; }
        };

        // llama.cpp quantize_row_q8_0_ref, so the quantized weights are the
        // same bits a Q8_0 GGUF would carry: per 32 elements, d = amax/127 in
        // f16, q = round(x/d).
        constexpr int64_t QK8 = 32;
        // f32 -> f16, via the compiler's IEEE conversion rather than a
        // hand-rolled one.
        //
        // The hand-rolled version flushed SUBNORMALS to zero, and that is not a
        // corner case here: up to 1.25% of Q8_0 blocks in this checkpoint have
        // a scale below f16's smallest normal (6.1e-5). Flushing them zeroed
        // ~1% of every weight matrix, which does not look like a bug from the
        // outside -- the model still runs and still puts the right token first
        // -- it just quietly loses most of its accuracy (logits differed from
        // bf16 by 9.98 on a scale of 11).
        inline uint16_t f32_to_f16(float f)
        {
            _Float16 h = static_cast<_Float16>(f);
            uint16_t b;
            std::memcpy(&b, &h, 2);
            return b;
        }

        // One tensor-parallel shard. Shards map onto physical cards by
        // `gpu`; two shards may share a card (that is how the 1-GPU vs 2-GPU
        // bitwise gate runs on the tiny model).
        struct Dev
        {
            sycl::device dev;
            int gpu = 0;
            sycl::queue q;
#if ORACLE_GPU_DNNL
            dnnl::engine eng;
            dnnl::stream strm;
            // Matmul primitives are keyed by shape: building one JITs, so it
            // must be cached. The memory OBJECTS are cached with it and their
            // handles swapped per call — constructing three dnnl::memory
            // wrappers per GEMM costs ~1100 constructions per decode token and
            // showed up as decode being ~6x off the card's bandwidth while no
            // individual kernel was slow.
            struct Prim
            {
                dnnl::matmul prim;
                dnnl::memory a, b, c;
                std::unordered_map<int, dnnl::memory> args;
            };
            std::map<std::array<int64_t, 5>, Prim> prims;
#endif
            // scratch, all dim-major [dim, block]
            float *h = nullptr, *xf = nullptr, *qf = nullptr, *kf = nullptr, *vf = nullptr;
            float *gf = nullptr, *of = nullptr, *g1 = nullptr, *u1 = nullptr;
            uint16_t *xb = nullptr, *qb = nullptr, *kb = nullptr, *ob = nullptr, *g1b = nullptr;
            float *logits = nullptr;
            float *rope_cos = nullptr, *rope_sin = nullptr;
            int32_t *ids = nullptr;
            uint16_t *pack = nullptr; // one packed activation column, for k_gemv1
            float *xfer = nullptr;    // contiguous [H, n] staging, for --trace-dir
            // All-reduce exchange buffer: `nshard` slots of [H, block], slot k
            // holding shard k's partial. Shard s packs straight into its own
            // slot, so the reduction reads a uniform array and needs no extra
            // local copy.
            float *peer = nullptr;
            uint16_t *deq = nullptr; // Q8 prefill: one weight expanded to bf16
            // --flash-prefill tier scratch (allocated only for that tier)
            float *fs = nullptr;    // S tile [heads, FBQ, FBK] f32
            uint16_t *fp = nullptr; // P tile [heads, FBQ, FBK] bf16
            float *foacc = nullptr; // O accumulator [heads, FBQ, D] f32
            float *fm = nullptr;    // running row max  [heads, FBQ]
            float *fl = nullptr;    // running row sumexp [heads, FBQ]
            // DFlash: target hidden states at the drafter's target_layer_ids,
            // one [H, block] buffer per tap. h is replicated across shards, so
            // every shard captures its own and no exchange is needed.
            std::vector<float *> taps;
        };

        // One layer's weights, split across the tensor-parallel shards. Every
        // array below is indexed by shard.
        //
        // The split follows the shapes: q/gate/mlp_gate/mlp_up are sharded on
        // their OUTPUT dim (contiguous row slices, no communication), o_proj
        // and mlp_down on their INPUT dim (so each shard produces a partial
        // [H, n] that must be all-reduced). Norms are replicated because they
        // are 13 KB and a broadcast is not free.
        //
        // k_proj/v_proj and the KV cache are replicated rather than sharded by
        // KV head. Muse Glimmer has only 2 KV heads and the tiny gate model has
        // 1, so sharding them would either fail or need a second code path; the
        // duplicated weight is 3.4 MB per layer per shard and the duplicated
        // work is ~0.7% of the layer.
        struct GpuLayer
        {
            std::vector<uint16_t *> input_ln, post_attn_ln, pre_ff_ln, post_ff_ln; // replicated
            std::vector<uint16_t *> kc, vc;                                        // KV cache
            std::vector<QW> k_, v_;                        // replicated, tier-agnostic
            std::vector<QW> q, gate, mlp_gate, mlp_up;     // row-sharded
            std::vector<QW> o, mlp_down;                   // col-sharded
            int64_t cap = 0;
        };

        // One drafter layer, sharded like the target's: outputs row-split,
        // inputs col-split (partial sums, all-reduced), norms and the KV
        // projections replicated.
        struct DLayer
        {
            std::vector<uint16_t *> input_ln, post_attn_ln, q_norm, k_norm; // replicated
            std::vector<uint16_t *> k, v;                                   // replicated
            std::vector<uint16_t *> q, mlp_gate, mlp_up;                    // row-sharded
            std::vector<uint16_t *> o, mlp_down;                            // col-sharded
        };

        // One vision-tower layer. LayerNorms carry a bias as well as a weight
        // (this is not the text stack's RMSNorm), and every projection has a
        // bias. Biases on the COLUMN-sharded projections are replicated and
        // added after the all-reduce -- adding them per shard would add them
        // `nshard` times.
        struct VLayer
        {
            std::vector<uint16_t *> n1w, n1b, n2w, n2b;   // replicated
            std::vector<uint16_t *> qw, kw, vw, fc1w;     // row-sharded
            std::vector<uint16_t *> qb, kb, vb, fc1b;     // row-sharded bias
            std::vector<uint16_t *> ow, fc2w;             // column-sharded
            std::vector<uint16_t *> ob, fc2b;             // replicated bias
        };

        // =====================================================================
        // Kernels
        //
        // Convention: activation element (token t, dim i) lives at i * ld + t,
        // where `ld` is the allocated block. Grouped variants (per attention
        // head) index (t, g, d) at (g * D + d) * ld + t.
        // =====================================================================

        // RMSNorm, all three flavours. The sum of squares accumulates in f64
        // exactly as the twin does — the B70 emulates fp64 and it is slow, but
        // this is O(rows*dim) against the GEMMs' O(rows*dim*out), and dropping
        // to f32 here moves the logits.
        enum class NK
        {
            Centered,
            Plain,
            Weightless
        };

        // out is BF16. `groups` = 1 normalizes the whole row (dim = hidden);
        // groups = n_heads normalizes each head's head_dim slice (QK-norm).
        void k_rmsnorm(sycl::queue &q, const float *X, const uint16_t *W, uint16_t *out,
                       int64_t n, int64_t groups, int64_t D, int64_t ld, double eps, NK kind)
        {
            const int64_t rows = n * groups;
            q.submit([&](sycl::handler &h) {
                h.parallel_for(
                    sycl::nd_range<1>(size_t(rows) * SG, SG),
                    [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
                        const int64_t row = int64_t(it.get_group(0));
                        const int64_t t = row % n, g = row / n;
                        auto sg = it.get_sub_group();
                        const int lane = int(sg.get_local_id()[0]);

                        double ss = 0;
                        for (int64_t i = lane; i < D; i += SG)
                        {
                            const double v = double(X[(g * D + i) * ld + t]);
                            ss += v * v;
                        }
                        ss = sycl::reduce_over_group(sg, ss, sycl::plus<double>());
                        const double rs = 1.0 / sycl::sqrt(ss / double(D) + eps);

                        for (int64_t i = lane; i < D; i += SG)
                        {
                            const int64_t off = (g * D + i) * ld + t;
                            double v = double(X[off]) * rs;
                            if (kind == NK::Centered)
                                v = v * (1.0 + double(bf2f(W[i])));
                            else if (kind == NK::Plain)
                                v = v * double(bf2f(W[i]));
                            out[off] = f2bf(float(v));
                        }
                    });
            });
        }

        // Wide RMSNorm: one work-group (not one sub-group) per row.
        //
        // The narrow kernel above gives a row to a single 32-lane sub-group.
        // That is right for prefill, where there are hundreds of rows to fill
        // the card with, and badly wrong for decode, where there is exactly
        // ONE row: 6656 elements walked 32 at a time by one sub-group measured
        // 68 us, and the model does 4 of these per layer, 208 per token.
        // Spreading the row over WG threads and reducing in two levels cuts the
        // dependent-load chain by WG/SG.
        //
        // It is a different summation order, so it is NOT interchangeable with
        // the narrow kernel — the caller picks by shape, and the choice must be
        // a pure function of (rows, D) so that a rerun cannot pick differently.
        constexpr int NORM_WG = 256;
        void k_rmsnorm_wide(sycl::queue &q, const float *X, const uint16_t *W, uint16_t *out,
                            int64_t n, int64_t groups, int64_t D, int64_t ld, double eps,
                            NK kind)
        {
            const int64_t rows = n * groups;
            q.submit([&](sycl::handler &h) {
                sycl::local_accessor<double, 1> part(sycl::range<1>(NORM_WG / SG), h);
                h.parallel_for(
                    sycl::nd_range<1>(size_t(rows) * NORM_WG, NORM_WG),
                    [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
                        const int64_t row = int64_t(it.get_group(0));
                        const int64_t t = row % n, g = row / n;
                        auto sg = it.get_sub_group();
                        const int lid = int(it.get_local_id(0));

                        double ss = 0;
                        for (int64_t i = lid; i < D; i += NORM_WG)
                        {
                            const double v = double(X[(g * D + i) * ld + t]);
                            ss += v * v;
                        }
                        ss = sycl::reduce_over_group(sg, ss, sycl::plus<double>());
                        if (sg.get_local_id()[0] == 0)
                            part[sg.get_group_id()[0]] = ss;
                        sycl::group_barrier(it.get_group());
                        double tot = 0;
                        for (int k = 0; k < NORM_WG / SG; ++k)
                            tot += part[k];
                        const double rs = 1.0 / sycl::sqrt(tot / double(D) + eps);

                        for (int64_t i = lid; i < D; i += NORM_WG)
                        {
                            const int64_t off = (g * D + i) * ld + t;
                            double v = double(X[off]) * rs;
                            if (kind == NK::Centered)
                                v = v * (1.0 + double(bf2f(W[i])));
                            else if (kind == NK::Plain)
                                v = v * double(bf2f(W[i]));
                            out[off] = f2bf(float(v));
                        }
                    });
            });
        }

        // Pick by shape only: never by a runtime measurement, so that two runs
        // of the same prompt cannot take different kernels. Wide only pays off
        // when the row is long and there are too few rows to fill the card.
        void k_rmsnorm_auto(sycl::queue &q, const float *X, const uint16_t *W, uint16_t *out,
                            int64_t n, int64_t groups, int64_t D, int64_t ld, double eps,
                            NK kind)
        {
            if (D >= 1024 && n * groups <= 64)
                k_rmsnorm_wide(q, X, W, out, n, groups, D, ld, eps, kind);
            else
                k_rmsnorm(q, X, W, out, n, groups, D, ld, eps, kind);
        }

        // f32-out variant, used for the embedding norm (the twin keeps `h` in
        // f32 holding BF16-representable values).
        void k_rmsnorm_f32(sycl::queue &q, const float *X, float *out, int64_t n, int64_t D,
                           int64_t ld, double eps)
        {
            q.submit([&](sycl::handler &h) {
                h.parallel_for(sycl::nd_range<1>(size_t(n) * SG, SG),
                               [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
                                   const int64_t t = int64_t(it.get_group(0));
                                   auto sg = it.get_sub_group();
                                   const int lane = int(sg.get_local_id()[0]);
                                   double ss = 0;
                                   for (int64_t i = lane; i < D; i += SG)
                                   {
                                       const double v = double(X[i * ld + t]);
                                       ss += v * v;
                                   }
                                   ss = sycl::reduce_over_group(sg, ss, sycl::plus<double>());
                                   const double rs = 1.0 / sycl::sqrt(ss / double(D) + eps);
                                   for (int64_t i = lane; i < D; i += SG)
                                       out[i * ld + t] = rb(float(double(X[i * ld + t]) * rs));
                               });
            });
        }

        // embedding gather: table is [V, H] row-major, out is dim-major f32
        void k_embed(sycl::queue &q, const uint16_t *table, const int32_t *ids, float *out,
                     int64_t n, int64_t H, int64_t ld)
        {
            q.submit([&](sycl::handler &h) {
                h.parallel_for(sycl::range<2>(size_t(H), size_t(n)), [=](sycl::id<2> id) {
                    const int64_t i = int64_t(id[0]), t = int64_t(id[1]);
                    out[i * ld + t] = bf2f(table[int64_t(ids[t]) * H + i]);
                });
            });
        }

        // Every elementwise kernel below takes (dim, n, ld) and touches only the
        // live columns. Running them over the whole allocated block instead is
        // correct but pointless work: at decode n is 1 while the block may be
        // 512, so it inflates these kernels by the block width.
        void k_scale_bf16(sycl::queue &q, uint16_t *x, int64_t dim, int64_t n, int64_t ld,
                          float s)
        {
            q.parallel_for(sycl::range<2>(size_t(dim), size_t(n)), [=](sycl::id<2> id) {
                const int64_t o = int64_t(id[0]) * ld + int64_t(id[1]);
                x[o] = f2bf(bf2f(x[o]) * s);
            });
        }

        // RoPE, rotate-half over the full head_dim, with the twin's rounding:
        // each product is materialized at BF16 before the add.
        void k_rope(sycl::queue &q, uint16_t *qb, uint16_t *kb, int64_t n, int64_t nq,
                    int64_t nkv, int64_t D, int64_t ld, const float *cos, const float *sin,
                    int64_t pos0)
        {
            const int64_t half = D / 2;
            q.submit([&](sycl::handler &h) {
                h.parallel_for(
                    sycl::range<3>(size_t(n), size_t(nq + nkv), size_t(half)),
                    [=](sycl::id<3> id) {
                        const int64_t t = int64_t(id[0]), hh = int64_t(id[1]),
                                      j = int64_t(id[2]);
                        uint16_t *p;
                        int64_t base;
                        if (hh < nq)
                        {
                            p = qb;
                            base = hh * D;
                        }
                        else
                        {
                            p = kb;
                            base = (hh - nq) * D;
                        }
                        const float cj = cos[(pos0 + t) * half + j];
                        const float sj = sin[(pos0 + t) * half + j];
                        const int64_t lo_off = (base + j) * ld + t;
                        const int64_t hi_off = (base + j + half) * ld + t;
                        const float lo = bf2f(p[lo_off]), hi = bf2f(p[hi_off]);
                        p[lo_off] = f2bf(rb(lo * cj) + rb(-hi * sj));
                        p[hi_off] = f2bf(rb(hi * cj) + rb(lo * sj));
                    });
            });
        }

        // Append this block's k/v to the ring. Activations are dim-major and
        // the cache is row-major [cap, KD] (a key vector must be contiguous for
        // the attention dot), so this is also the transpose.
        void k_kv_append(sycl::queue &q, const uint16_t *kb, const float *vf, uint16_t *kc,
                         uint16_t *vc, int64_t n, int64_t KD, int64_t ld, int64_t pos0,
                         int64_t cap)
        {
            q.submit([&](sycl::handler &h) {
                h.parallel_for(sycl::range<2>(size_t(n), size_t(KD)), [=](sycl::id<2> id) {
                    const int64_t t = int64_t(id[0]), i = int64_t(id[1]);
                    const int64_t slot = (pos0 + t) % cap;
                    kc[slot * KD + i] = kb[i * ld + t];
                    vc[slot * KD + i] = f2bf(vf[i * ld + t]);
                });
            });
        }

        // Flash-style attention: S and P never materialize at BF16, matching
        // `--attn flash` and the twin's online-softmax loop. One sub-group per
        // (token, query head); each lane carries head_dim/SG accumulators.
        void k_attention(sycl::queue &q, const uint16_t *qb, const uint16_t *kc,
                         const uint16_t *vc, float *out, int64_t n, int64_t nq, int64_t D,
                         int64_t KD, int64_t ld, int64_t pos0, int64_t groups, int64_t cap,
                         int64_t window, float scaling, int64_t head0)
        {
            const int64_t rows = n * nq;
            q.submit([&](sycl::handler &h) {
                h.parallel_for(
                    sycl::nd_range<1>(size_t(rows) * SG, SG),
                    [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
                        const int64_t row = int64_t(it.get_group(0));
                        const int64_t t = row % n, hh = row / n;
                        auto sg = it.get_sub_group();
                        const int lane = int(sg.get_local_id()[0]);

                        // `hh` is local to this shard; the GQA group index must
                        // come from the global head number because the KV cache
                        // holds all heads.
                        const int64_t qpos = pos0 + t, g = (head0 + hh) / groups;
                        const int64_t lo = window > 0 ? sycl::max<int64_t>(0, qpos - window + 1) : 0;

                        // at head_dim 128 and SG 32 this is 4 accumulators
                        constexpr int MAXACC = 16;
                        float acc[MAXACC];
                        float qv[MAXACC];
                        int na = 0;
                        for (int64_t i = lane; i < D; i += SG, ++na)
                        {
                            acc[na] = 0.f;
                            qv[na] = bf2f(qb[(hh * D + i) * ld + t]);
                        }

                        float mx = -INFINITY, sum = 0.f;
                        for (int64_t j = lo; j <= qpos; ++j)
                        {
                            const int64_t slot = j % cap;
                            const uint16_t *kp = kc + slot * KD + g * D;
                            float part = 0.f;
                            for (int a = 0, i = lane; a < na; ++a, i += SG)
                                part += qv[a] * bf2f(kp[i]);
                            const float s =
                                sycl::reduce_over_group(sg, part, sycl::plus<float>()) * scaling;

                            const float m2 = sycl::max(mx, s);
                            const float corr = sycl::exp(mx - m2);
                            const float e = sycl::exp(s - m2);
                            sum = sum * corr + e;
                            const uint16_t *vp = vc + slot * KD + g * D;
                            for (int a = 0, i = lane; a < na; ++a, i += SG)
                                acc[a] = acc[a] * corr + e * bf2f(vp[i]);
                            mx = m2;
                        }
                        const float inv = 1.0f / sum;
                        for (int a = 0, i = lane; a < na; ++a, i += SG)
                            out[(hh * D + i) * ld + t] = acc[a] * inv;
                    });
            });
        }

        // Tiled prefill attention.
        //
        // IDENTICAL ARITHMETIC to k_attention: the same keys in the same order,
        // the same per-key sub-group dot, the same online-softmax update. The
        // only thing that changes is where K and V are read FROM. A work-group
        // takes BQ consecutive queries of one head and stages each key tile in
        // local memory, so the tile is fetched from global memory once per BQ
        // queries instead of once per query.
        //
        // That is the whole cost at depth. The untiled kernel re-reads the
        // cache per (query, head): at T=2048 that is ~894 GB per forward
        // against a card that streams 599 GB/s, and the profile showed
        // attention at 49% of prefill, which is exactly what those numbers
        // predict. Dividing the traffic by BQ moves it off the critical path.
        //
        // Because the arithmetic is unchanged, this kernel is bitwise
        // interchangeable with the untiled one and the gates check that
        // (chunk-invariance and decode-vs-prefill both cross the boundary).
        void k_attention_tiled(sycl::queue &q, const uint16_t *qb, const uint16_t *kc,
                               const uint16_t *vc, float *out, int64_t n, int64_t nq, int64_t D,
                               int64_t KD, int64_t ld, int64_t pos0, int64_t groups, int64_t cap,
                               int64_t window, float scaling, int64_t head0, int64_t BQ,
                               int64_t BK)
        {
            const int64_t qtiles = (n + BQ - 1) / BQ;
            const size_t wg = size_t(BQ) * SG;
            q.submit([&](sycl::handler &h) {
                sycl::local_accessor<uint16_t, 1> ks(sycl::range<1>(size_t(BK * D)), h);
                sycl::local_accessor<uint16_t, 1> vs(sycl::range<1>(size_t(BK * D)), h);
                h.parallel_for(
                    sycl::nd_range<1>(size_t(qtiles * nq) * wg, wg),
                    [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
                        const int64_t gid = int64_t(it.get_group(0));
                        const int64_t qt = gid % qtiles, hh = gid / qtiles;
                        auto sg = it.get_sub_group();
                        const int lane = int(sg.get_local_id()[0]);
                        const int64_t sgi = int64_t(sg.get_group_id()[0]);
                        const int64_t lid = int64_t(it.get_local_id(0));

                        const int64_t t = qt * BQ + sgi;
                        const bool live = t < n;
                        const int64_t qpos = pos0 + t;
                        const int64_t g = (head0 + hh) / groups;
                        const int64_t my_lo =
                            window > 0 ? sycl::max<int64_t>(0, qpos - window + 1) : 0;

                        // Key range for the WHOLE group, so every sub-group runs
                        // the same tile loop and the barriers stay aligned.
                        const int64_t t_lo = qt * BQ;
                        const int64_t t_hi = sycl::min<int64_t>(t_lo + BQ - 1, n - 1);
                        const int64_t hi_g = pos0 + t_hi;
                        const int64_t lo_g =
                            window > 0 ? sycl::max<int64_t>(0, pos0 + t_lo - window + 1) : 0;

                        constexpr int MAXACC = 16;
                        float acc[MAXACC], qv[MAXACC];
                        int na = 0;
                        for (int64_t i = lane; i < D; i += SG, ++na)
                        {
                            acc[na] = 0.f;
                            qv[na] = live ? bf2f(qb[(hh * D + i) * ld + t]) : 0.f;
                        }
                        float mx = -INFINITY, sum = 0.f;

                        // Loader indices, computed ONCE. The obvious flat loop
                        // (`e / D`, `e % D`, `(jt+jj) % cap` per element) costs
                        // three integer divisions per element, and Xe has no
                        // hardware integer divide — that alone made the first
                        // version of this kernel slower than the untiled one it
                        // replaced. Everything below is add-and-compare.
                        const int64_t my_d = lid % D;
                        const int64_t my_row0 = lid / D;
                        const int64_t row_step = int64_t(wg) / D;

                        for (int64_t jt = lo_g; jt <= hi_g; jt += BK)
                        {
                            const int64_t jn = sycl::min<int64_t>(BK, hi_g - jt + 1);
                            int64_t slot = jt % cap; // one modulo per tile
                            slot += my_row0;
                            if (slot >= cap)
                                slot -= cap;
                            for (int64_t jj = my_row0; jj < jn; jj += row_step)
                            {
                                ks[size_t(jj * D + my_d)] = kc[slot * KD + g * D + my_d];
                                vs[size_t(jj * D + my_d)] = vc[slot * KD + g * D + my_d];
                                slot += row_step;
                                if (slot >= cap)
                                    slot -= cap;
                            }
                            sycl::group_barrier(it.get_group());

                            if (live)
                            {
                                for (int64_t jj = 0; jj < jn; ++jj)
                                {
                                    const int64_t j = jt + jj;
                                    // sub-group-uniform: my_lo and qpos depend
                                    // only on t
                                    if (j < my_lo || j > qpos)
                                        continue;
                                    float part = 0.f;
                                    for (int a = 0, i = lane; a < na; ++a, i += SG)
                                        part += qv[a] * bf2f(ks[size_t(jj * D + i)]);
                                    const float sc =
                                        sycl::reduce_over_group(sg, part, sycl::plus<float>()) *
                                        scaling;
                                    const float m2 = sycl::max(mx, sc);
                                    const float corr = sycl::exp(mx - m2);
                                    const float ex = sycl::exp(sc - m2);
                                    sum = sum * corr + ex;
                                    for (int a = 0, i = lane; a < na; ++a, i += SG)
                                        acc[a] = acc[a] * corr + ex * bf2f(vs[size_t(jj * D + i)]);
                                    mx = m2;
                                }
                            }
                            sycl::group_barrier(it.get_group());
                        }
                        if (live)
                        {
                            const float inv = 1.0f / sum;
                            for (int a = 0, i = lane; a < na; ++a, i += SG)
                                out[(hh * D + i) * ld + t] = acc[a] * inv;
                        }
                    });
            });
        }

        // Flat helpers. The drafter's buffers are packed row-major, so its
        // elementwise steps are plain 1-D sweeps rather than the target's
        // (dim, token, leading-dimension) form.
        void k_round(sycl::queue &q, const float *src, uint16_t *dst, int64_t n)
        {
            q.parallel_for(sycl::range<1>(size_t(n)),
                           [=](sycl::id<1> i) { dst[i] = f2bf(src[i]); });
        }
        void k_swiglu_flat(sycl::queue &q, const float *g1, const float *u1, uint16_t *out,
                           int64_t n)
        {
            q.parallel_for(sycl::range<1>(size_t(n)), [=](sycl::id<1> i) {
                const float g = rb(g1[i]);
                const float sl = rb(g / (1.0f + sycl::exp(-g)));
                out[i] = f2bf(sl * rb(u1[i]));
            });
        }
        void k_embed_rm(sycl::queue &q, const uint16_t *table, const int32_t *ids, float *out,
                        int64_t n, int64_t H)
        {
            q.parallel_for(sycl::range<2>(size_t(n), size_t(H)), [=](sycl::id<2> id) {
                const int64_t t = int64_t(id[0]), i = int64_t(id[1]);
                out[t * H + i] = bf2f(table[int64_t(ids[t]) * H + i]);
            });
        }
        void k_sum_flat(sycl::queue &q, const float *src, float *dst, int64_t nsh, int64_t slot,
                        int64_t n)
        {
            q.parallel_for(sycl::range<1>(size_t(n)), [=](sycl::id<1> i) {
                float acc = src[i];
                for (int64_t k = 1; k < nsh; ++k)
                    acc += src[k * slot + int64_t(i)];
                dst[i] = acc;
            });
        }
        // rope over row-major [rows, heads*D], row r sitting at position pos0+r
        void k_drope(sycl::queue &q, float *X, int64_t rows, int64_t nh, int64_t D,
                     const float *cosT, const float *sinT, int64_t pos0)
        {
            const int64_t half = D / 2;
            q.parallel_for(sycl::range<3>(size_t(rows), size_t(nh), size_t(half)),
                           [=](sycl::id<3> id) {
                               const int64_t r = int64_t(id[0]), hh = int64_t(id[1]),
                                             j = int64_t(id[2]);
                               float *p = X + (r * nh + hh) * D;
                               const int64_t pos = pos0 + r;
                               const float cj = cosT[pos * half + j], sj = sinT[pos * half + j];
                               const float lo = p[j], hi = p[j + half];
                               p[j] = rb(rb(lo * cj) + rb(-hi * sj));
                               p[j + half] = rb(rb(hi * cj) + rb(lo * sj));
                           });
        }

        // weight-less RMSNorm over row-major rows (the vision projection tail
        // uses the SAME norm the text embedding does)
        void k_dnorm_wl(sycl::queue &q, const float *X, float *out, int64_t rows, int64_t D,
                        double eps)
        {
            q.submit([&](sycl::handler &h) {
                h.parallel_for(sycl::nd_range<1>(size_t(rows) * SG, SG),
                               [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
                                   const int64_t r = int64_t(it.get_group(0));
                                   auto sg = it.get_sub_group();
                                   const int lane = int(sg.get_local_id()[0]);
                                   double ss = 0;
                                   for (int64_t i = lane; i < D; i += SG)
                                   {
                                       const double v = double(X[r * D + i]);
                                       ss += v * v;
                                   }
                                   ss = sycl::reduce_over_group(sg, ss, sycl::plus<double>());
                                   const double rs = 1.0 / sycl::sqrt(ss / double(D) + eps);
                                   for (int64_t i = lane; i < D; i += SG)
                                       out[r * D + i] = rb(float(double(X[r * D + i]) * rs));
                               });
            });
        }

        // ------------------------------------------------------------- Q8_0
        //
        // Decode GEMV over Q8_0 weights, "variant A": dequantize in-register
        // into the ordinary f32 fma structure, one scale per 32-element block.
        // Quants are read 32 bytes per lane as 8 dwords and unpacked with
        // shifts -- byte-at-a-time loads measured 166 GB/s against this form's
        // ~330, which was the difference between Q8 being a win and a wash.
        // BLOCKDOT is a template parameter, not a flag: as a runtime branch inside
        // the unrolled inner loop it cost 12x (16.2 -> 1.31 tok/s), because both
        // arms stay in the loop body and the accumulators spill.
        template <bool BLOCKDOT>
        void k_gemv_q8(sycl::queue &q, const int8_t *W, const uint16_t *D, const uint16_t *X,
                       float *Y, int64_t in, int64_t out, int64_t ldy)
        {
            constexpr int RPG = 8;
            const int64_t nblk = in / QK8;
            const int64_t groups = (out + RPG - 1) / RPG;
            q.submit([&](sycl::handler &h) {
                h.parallel_for(
                    sycl::nd_range<1>(size_t(groups) * RPG * SG, RPG * SG),
                    [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
                        auto sg = it.get_sub_group();
                        const int64_t o =
                            int64_t(it.get_group(0)) * RPG + int64_t(sg.get_group_id()[0]);
                        if (o >= out)
                            return;
                        const int lane = int(sg.get_local_id()[0]);
                        const uint32_t *w = reinterpret_cast<const uint32_t *>(W + o * in);
                        const sycl::half *dd =
                            reinterpret_cast<const sycl::half *>(D + o * nblk);
                        float part = 0.f;
                        for (int64_t b = lane; b < nblk; b += SG)
                        {
                            const uint32_t *wb = w + b * (QK8 / 4);
                            const uint16_t *xb = X + b * QK8;
                            const float sc = float(dd[b]);
                            // Two arithmetics, and the choice is a real one.
                            //
                            // `blockdot` sums the block first and scales once,
                            // which is what llama.cpp's Q8_0 dot does -- and
                            // it is 24% faster here. But prefill cannot do it:
                            // it dequantizes to a BF16 tile so the matrix
                            // engines can run, which materializes bf16(q*d) per
                            // element. Taking the fast form at decode makes
                            // decode disagree with prefill INSIDE ONE TIER, so
                            // a token this engine generates would not be
                            // reproduced by prefilling the same sequence. The
                            // default therefore matches prefill; the fast form
                            // is behind MUSE_GPU_Q8_BLOCKDOT for anyone who
                            // wants llama.cpp-exact dot semantics instead.
                            float acc = 0.f;
#pragma unroll
                            for (int u = 0; u < QK8 / 4; ++u)
                            {
                                const uint32_t v = wb[u];
#pragma unroll
                                for (int k = 0; k < 4; ++k)
                                {
                                    const float qv = float(int8_t((v >> (8 * k)) & 0xff));
                                    const float xv = bf2f(xb[u * 4 + k]);
                                    if constexpr (BLOCKDOT)
                                        acc += qv * xv;
                                    else
                                        part += rb(qv * sc) * xv;
                                }
                            }
                            if constexpr (BLOCKDOT)
                                part += acc * sc;
                            else
                                (void)acc;
                        }
                        const float s = sycl::reduce_over_group(sg, part, sycl::plus<float>());
                        if (lane == 0)
                            Y[o * ldy] = s;
                    });
            });
        }

        // Prefill stays on the matrix engines: oneDNN has no grouped-scale
        // path on this stack (measured -- see docs), so the weight is expanded
        // to bf16 in a scratch tile and the ordinary DPAS GEMM runs on it. That
        // costs a write and a re-read of the tile, which is why Q8 is a decode
        // and memory tier rather than a prefill one.
        void k_dequant_q8(sycl::queue &q, const int8_t *W, const uint16_t *D, uint16_t *out,
                          int64_t in, int64_t out_rows)
        {
            const int64_t nblk = in / QK8;
            q.parallel_for(sycl::range<2>(size_t(out_rows), size_t(nblk)), [=](sycl::id<2> id) {
                const int64_t o = int64_t(id[0]), b = int64_t(id[1]);
                const sycl::half *dd = reinterpret_cast<const sycl::half *>(D + o * nblk);
                const float sc = float(dd[b]);
                const int8_t *wb = W + o * in + b * QK8;
                uint16_t *ob = out + o * in + b * QK8;
                for (int k = 0; k < QK8; ++k)
                    ob[k] = f2bf(float(wb[k]) * sc);
            });
        }

        // -------------------------------------------------------------- vision
        //
        // LayerNorm: mean and variance, then weight and bias. Not RMSNorm --
        // the tower is a ViT and centres its activations.
        void k_layernorm(sycl::queue &q, const float *X, const uint16_t *W, const uint16_t *Bs,
                         float *out, int64_t rows, int64_t dim, double eps)
        {
            q.submit([&](sycl::handler &h) {
                h.parallel_for(sycl::nd_range<1>(size_t(rows) * SG, SG),
                               [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
                                   const int64_t r = int64_t(it.get_group(0));
                                   auto sg = it.get_sub_group();
                                   const int lane = int(sg.get_local_id()[0]);
                                   const float *x = X + r * dim;
                                   double sum = 0;
                                   for (int64_t i = lane; i < dim; i += SG)
                                       sum += double(x[i]);
                                   sum = sycl::reduce_over_group(sg, sum, sycl::plus<double>());
                                   const double mean = sum / double(dim);
                                   double var = 0;
                                   for (int64_t i = lane; i < dim; i += SG)
                                   {
                                       const double d = double(x[i]) - mean;
                                       var += d * d;
                                   }
                                   var = sycl::reduce_over_group(sg, var, sycl::plus<double>());
                                   const double rs = 1.0 / sycl::sqrt(var / double(dim) + eps);
                                   float *y = out + r * dim;
                                   for (int64_t i = lane; i < dim; i += SG)
                                       y[i] = rb(float((double(x[i]) - mean) * rs *
                                                           double(bf2f(W[i])) +
                                                       double(bf2f(Bs[i]))));
                               });
            });
        }

        // add a bias row and round, the tail of every vision nn.Linear
        void k_bias(sycl::queue &q, float *Y, const uint16_t *B, int64_t rows, int64_t out)
        {
            q.parallel_for(sycl::range<2>(size_t(rows), size_t(out)), [=](sycl::id<2> id) {
                const int64_t r = int64_t(id[0]), o = int64_t(id[1]);
                Y[r * out + o] = rb(Y[r * out + o] + bf2f(B[o]));
            });
        }

        void k_gelu(sycl::queue &q, float *X, int64_t n)
        {
            q.parallel_for(sycl::range<1>(size_t(n)), [=](sycl::id<1> i) {
                const float v = X[i];
                // erf-exact GELU, matching fmath::gelu rather than the tanh
                // approximation -- the reference uses the exact one
                X[i] = rb(0.5f * v * (1.0f + sycl::erf(v * 0.70710678118654752f)));
            });
        }

        // patch embedding + the bilinear position table, whose four corners and
        // weights are resolved on the host (pos_taps) and uploaded
        void k_pos_add(sycl::queue &q, float *X, const uint16_t *table, const int32_t *idx,
                       const float *wgt, int64_t N, int64_t H)
        {
            q.parallel_for(sycl::range<2>(size_t(N), size_t(H)), [=](sycl::id<2> id) {
                const int64_t t = int64_t(id[0]), d = int64_t(id[1]);
                float acc = 0.f;
                for (int k = 0; k < 4; ++k)
                    acc += bf2f(table[int64_t(idx[t * 4 + k]) * H + d]) * wgt[t * 4 + k];
                X[t * H + d] = rb(X[t * H + d] + acc);
            });
        }

        // gather rows: dst[t] = src[map[t]]
        void k_gather_rows(sycl::queue &q, const float *src, float *dst, const int32_t *map,
                           int64_t N, int64_t H)
        {
            q.parallel_for(sycl::range<2>(size_t(N), size_t(H)), [=](sycl::id<2> id) {
                const int64_t t = int64_t(id[0]), d = int64_t(id[1]);
                dst[t * H + d] = src[int64_t(map[t]) * H + d];
            });
        }
        // scatter rows: dst[map[t]] = src[t]
        void k_scatter_rows(sycl::queue &q, const float *src, float *dst, const int32_t *map,
                            int64_t N, int64_t H)
        {
            q.parallel_for(sycl::range<2>(size_t(N), size_t(H)), [=](sycl::id<2> id) {
                const int64_t t = int64_t(id[0]), d = int64_t(id[1]);
                dst[int64_t(map[t]) * H + d] = src[t * H + d];
            });
        }

        // 2-D rope: cos/sin are per TOKEN and shared by every head, and the
        // rotation uses cos[j+half]/sin[j+half] on the upper half rather than
        // reusing the lower half's -- the frequency layout is [fw|fh|fw|fh].
        void k_vrope(sycl::queue &q, float *Q, float *K, const float *cosv, const float *sinv,
                     int64_t N, int64_t nh, int64_t D, int64_t Hs)
        {
            const int64_t half = D / 2;
            q.submit([&](sycl::handler &h) {
                h.parallel_for(sycl::range<3>(size_t(N), size_t(nh), size_t(half)),
                               [=](sycl::id<3> id) {
                                   const int64_t t = int64_t(id[0]), hh = int64_t(id[1]),
                                                 j = int64_t(id[2]);
                                   const float *co = cosv + t * D, *si = sinv + t * D;
                                   float *pq = Q + t * Hs + hh * D;
                                   float *pk = K + t * Hs + hh * D;
                                   for (float *p : {pq, pk})
                                   {
                                       const float lo = p[j], hi = p[j + half];
                                       p[j] = rb(rb(lo * co[j]) + rb(-hi * si[j]));
                                       p[j + half] =
                                           rb(rb(hi * co[j + half]) + rb(lo * si[j + half]));
                                   }
                               });
            });
        }

        // Bidirectional attention inside each cu_seqlens segment (a window, or
        // a whole image for the full-attention layers). One sub-group per
        // (token, head); `seg` maps a token to its segment.
        void k_vattn(sycl::queue &q, const float *Q, const float *K, const float *V, float *O,
                     float *sc, const int32_t *seg, const int32_t *cu, int64_t N, int64_t nh,
                     int64_t D, int64_t Hs, int64_t maxseg, float scaling)
        {
            const int64_t rows = N * nh;
            q.submit([&](sycl::handler &h) {
                h.parallel_for(
                    sycl::nd_range<1>(size_t(rows) * SG, SG),
                    [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
                        const int64_t gid = int64_t(it.get_group(0));
                        const int64_t i = gid % N, hh = gid / N;
                        auto sg = it.get_sub_group();
                        const int lane = int(sg.get_local_id()[0]);
                        const int64_t s = seg[i];
                        const int64_t a = cu[s], b = cu[s + 1];
                        const float *qv = Q + i * Hs + hh * D;
                        float *row = sc + gid * maxseg;

                        float mx = -INFINITY;
                        for (int64_t j = a + lane; j < b; j += SG)
                        {
                            const float *kv = K + j * Hs + hh * D;
                            float acc = 0.f;
                            for (int64_t d = 0; d < D; ++d)
                                acc += qv[d] * kv[d];
                            const float v = rb(rb(acc) * scaling);
                            row[j - a] = v;
                            mx = sycl::max(mx, v);
                        }
                        mx = sycl::reduce_over_group(sg, mx, sycl::maximum<float>());
                        float sum = 0.f;
                        for (int64_t j = a + lane; j < b; j += SG)
                        {
                            const float e = sycl::exp(row[j - a] - mx);
                            row[j - a] = e;
                            sum += e;
                        }
                        sum = sycl::reduce_over_group(sg, sum, sycl::plus<float>());
                        sycl::group_barrier(sg);

                        float *o = O + i * Hs + hh * D;
                        for (int64_t d = lane; d < D; d += SG)
                        {
                            float acc = 0.f;
                            for (int64_t j = a; j < b; ++j)
                                acc += rb(row[j - a] / sum) * V[j * Hs + hh * D + d];
                            o[d] = rb(acc);
                        }
                    });
            });
        }

        // pixel shuffle: out[row][d*mu + k] = X[tok[row*mu + k]][d]
        void k_pixel_shuffle(sycl::queue &q, const float *X, float *out, const int32_t *tok,
                             int64_t M, int64_t H, int64_t mu)
        {
            q.parallel_for(sycl::range<3>(size_t(M), size_t(H), size_t(mu)),
                           [=](sycl::id<3> id) {
                               const int64_t r = int64_t(id[0]), d = int64_t(id[1]),
                                             k = int64_t(id[2]);
                               out[r * H * mu + d * mu + k] =
                                   X[int64_t(tok[r * mu + k]) * H + d];
                           });
        }

        // ------------------------------------------------------------- DFlash
        //
        // The drafter's RMSNorm is NOT the target's. It rounds BEFORE the
        // weight multiply -- `w[i] * bf16(x[i] * rs)` against the target's
        // `bf16(x[i] * rs * w[i])` -- and every one of its norms is PLAIN
        // (weight around 1), including the ones whose names match the target's
        // zero-centered sandwich norms. Same trap as in the target, one level
        // down, so it gets its own kernel rather than a flag on the other one.
        void k_dnorm(sycl::queue &q, const float *X, const uint16_t *W, float *out, int64_t rows,
                     int64_t D, int64_t ldx, int64_t ldo, double eps)
        {
            q.submit([&](sycl::handler &h) {
                h.parallel_for(sycl::nd_range<1>(size_t(rows) * SG, SG),
                               [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
                                   const int64_t r = int64_t(it.get_group(0));
                                   auto sg = it.get_sub_group();
                                   const int lane = int(sg.get_local_id()[0]);
                                   double ss = 0;
                                   for (int64_t i = lane; i < D; i += SG)
                                   {
                                       const double v = double(X[r * ldx + i]);
                                       ss += v * v;
                                   }
                                   ss = sycl::reduce_over_group(sg, ss, sycl::plus<double>());
                                   const double rs = 1.0 / sycl::sqrt(ss / double(D) + eps);
                                   for (int64_t i = lane; i < D; i += SG)
                                   {
                                       const float t = rb(float(double(X[r * ldx + i]) * rs));
                                       out[r * ldo + i] = rb(bf2f(W[i]) * t);
                                   }
                               });
            });
        }

        // Bidirectional sliding-window attention over [context ++ block].
        //
        // Not the target's kernel: there is no causal constraint at all, only
        // |q_pos - kv_pos| <= sliding_window, and the softmax is the ordinary
        // materialized max/exp/sum rather than an online update -- which is
        // what the reference does, so it is what the twin does.
        // One sub-group per (block row, head); the block is 16 rows.
        void k_dattn(sycl::queue &q, const float *Q, const float *K, const float *V, float *O,
                     float *sc, int64_t B, int64_t S, int64_t nqs, int64_t nkv, int64_t D,
                     int64_t pos0, int64_t window, float scaling, int64_t head0,
                     int64_t groups)
        {
            const int64_t NKV = nkv;
            const int64_t nq = nqs;
            const int64_t rows = B * nq;
            q.submit([&](sycl::handler &h) {
                h.parallel_for(
                    sycl::nd_range<1>(size_t(rows) * SG, SG),
                    [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
                        const int64_t gid = int64_t(it.get_group(0));
                        const int64_t i = gid % B, hh = gid / B;
                        auto sg = it.get_sub_group();
                        const int lane = int(sg.get_local_id()[0]);
                        const int64_t g = (head0 + hh) / groups, qpos = pos0 + i;
                        int64_t lo = 0, hi = S - 1;
                        if (window > 0)
                        {
                            lo = sycl::max<int64_t>(0, qpos - window);
                            hi = sycl::min<int64_t>(S - 1, qpos + window);
                        }
                        const float *qv = Q + (i * nq + hh) * D;
                        float *row = sc + gid * S;

                        float mx = -INFINITY;
                        for (int64_t j = lo + lane; j <= hi; j += SG)
                        {
                            const float *kv = K + (j * NKV + g) * D;
                            float acc = 0.f;
                            for (int64_t d = 0; d < D; ++d)
                                acc += qv[d] * kv[d];
                            const float v = rb(rb(acc) * scaling);
                            row[j] = v;
                            mx = sycl::max(mx, v);
                        }
                        mx = sycl::reduce_over_group(sg, mx, sycl::maximum<float>());
                        float sum = 0.f;
                        for (int64_t j = lo + lane; j <= hi; j += SG)
                        {
                            const float e = sycl::exp(row[j] - mx);
                            row[j] = e;
                            sum += e;
                        }
                        sum = sycl::reduce_over_group(sg, sum, sycl::plus<float>());

                        sycl::group_barrier(sg); // `row` is written strided, read by all
                        float *o = O + (i * nq + hh) * D;
                        for (int64_t d = lane; d < D; d += SG)
                        {
                            float a = 0.f;
                            for (int64_t j = lo; j <= hi; ++j)
                                a += rb(row[j] / sum) * V[(j * NKV + g) * D + d];
                            o[d] = rb(a);
                        }
                    });
            });
        }

        // h = round(h + mix), the drafter's plain residual (no sandwich norm)
        void k_dresid(sycl::queue &q, float *h, const float *mix, int64_t n)
        {
            q.parallel_for(sycl::range<1>(size_t(n)),
                           [=](sycl::id<1> i) { h[i] = rb(h[i] + mix[i]); });
        }

        // rows of `dst` gathered from the target's taps: dst[t][k*H+i] = tap_k[t][i]
        void k_tapcat(sycl::queue &q, const float *tap, float *dst, int64_t n, int64_t H,
                      int64_t k, int64_t taps)
        {
            q.parallel_for(sycl::range<2>(size_t(n), size_t(H)), [=](sycl::id<2> id) {
                const int64_t t = int64_t(id[0]), i = int64_t(id[1]);
                dst[t * taps * H + k * H + i] = tap[i * n + t];
            });
        }

        // ---------------------------------------------------------- flash tier
        //
        // The kernels below belong to the OPT-IN --flash-prefill tier, whose
        // numerics are deliberately NOT the twin's. See attention_flash().

        // One tile's softmax step. Reads the raw q.k tile the matrix engine
        // produced, applies the scale and the causal/window mask, folds the
        // tile into the running (max, sumexp), rescales the output accumulator,
        // and emits the bf16 probability tile the second GEMM consumes.
        //
        // One sub-group per (head, query row).
        void k_flash_softmax(sycl::queue &q, float *S, uint16_t *P, float *Oacc, float *mrow,
                             float *lrow, int64_t nh, int64_t rows, int64_t mk, int64_t FBQ,
                             int64_t FBK, int64_t D, int64_t t0, int64_t j0, int64_t pos0,
                             int64_t window, float scaling)
        {
            const int64_t total = nh * rows;
            q.submit([&](sycl::handler &h) {
                h.parallel_for(
                    sycl::nd_range<1>(size_t(total) * SG, SG),
                    [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
                        const int64_t gid = int64_t(it.get_group(0));
                        const int64_t r = gid % rows, hh = gid / rows;
                        auto sg = it.get_sub_group();
                        const int lane = int(sg.get_local_id()[0]);

                        const int64_t qpos = pos0 + t0 + r;
                        const int64_t lo = window > 0 ? sycl::max<int64_t>(0, qpos - window + 1) : 0;
                        float *srow = S + (hh * FBQ + r) * FBK;

                        // running max over the unmasked entries of this tile
                        float lm = -INFINITY;
                        for (int64_t k = lane; k < mk; k += SG)
                        {
                            const int64_t j = j0 + k;
                            const float v =
                                (j > qpos || j < lo) ? -INFINITY : srow[k] * scaling;
                            srow[k] = v;
                            lm = sycl::max(lm, v);
                        }
                        lm = sycl::reduce_over_group(sg, lm, sycl::maximum<float>());

                        const float m_old = mrow[hh * FBQ + r];
                        const float m2 = sycl::max(m_old, lm);
                        // both -inf means this row saw nothing yet and nothing
                        // now; leave the accumulator alone rather than make NaN
                        const float corr = (m2 == -INFINITY) ? 1.0f : sycl::exp(m_old - m2);

                        float lsum = 0.f;
                        for (int64_t k = lane; k < mk; k += SG)
                        {
                            const float v = srow[k];
                            const float e = (v == -INFINITY) ? 0.f : sycl::exp(v - m2);
                            P[(hh * FBQ + r) * FBK + k] = f2bf(e);
                            lsum += e;
                        }
                        lsum = sycl::reduce_over_group(sg, lsum, sycl::plus<float>());

                        // rescale the accumulator by the max correction before
                        // the next GEMM accumulates into it
                        float *o = Oacc + (hh * FBQ + r) * D;
                        for (int64_t i = lane; i < D; i += SG)
                            o[i] *= corr;

                        if (lane == 0)
                        {
                            mrow[hh * FBQ + r] = m2;
                            lrow[hh * FBQ + r] = lrow[hh * FBQ + r] * corr + lsum;
                        }
                    });
            });
        }

        // Normalize by the running sumexp and scatter into the engine's
        // dim-major activation layout.
        void k_flash_finish(sycl::queue &q, const float *Oacc, const float *lrow, float *out,
                            int64_t nh, int64_t rows, int64_t FBQ, int64_t D, int64_t ld,
                            int64_t t0)
        {
            q.parallel_for(sycl::range<3>(size_t(nh), size_t(rows), size_t(D)),
                           [=](sycl::id<3> id) {
                               const int64_t hh = int64_t(id[0]), r = int64_t(id[1]),
                                             i = int64_t(id[2]);
                               const float inv = 1.0f / lrow[hh * FBQ + r];
                               out[(hh * D + i) * ld + (t0 + r)] =
                                   Oacc[(hh * FBQ + r) * D + i] * inv;
                           });
        }

        void k_fill(sycl::queue &q, float *p, int64_t n, float v)
        {
            q.parallel_for(sycl::range<1>(size_t(n)), [=](sycl::id<1> i) { p[i] = v; });
        }

        // output gate from the PRE-attention normed input: one BF16
        // materialization per nn.Linear output, then one per elementwise op
        void k_out_gate(sycl::queue &q, const float *of, const float *gf, uint16_t *ob,
                        int64_t dim, int64_t n, int64_t ld)
        {
            q.parallel_for(sycl::range<2>(size_t(dim), size_t(n)), [=](sycl::id<2> id) {
                const int64_t i = int64_t(id[0]) * ld + int64_t(id[1]);
                const float pv = rb(of[i]);
                const float gt = rb(gf[i]);
                const float sg = rb(1.0f / (1.0f + sycl::exp(-gt)));
                ob[i] = f2bf(pv * sg);
            });
        }

        // h = bf16(h + bf16(x)); h stays f32 but holds BF16-representable values
        void k_residual(sycl::queue &q, float *h, const uint16_t *xb, int64_t dim, int64_t n,
                        int64_t ld)
        {
            q.parallel_for(sycl::range<2>(size_t(dim), size_t(n)), [=](sycl::id<2> id) {
                const int64_t i = int64_t(id[0]) * ld + int64_t(id[1]);
                h[i] = rb(h[i] + bf2f(xb[i]));
            });
        }

        void k_swiglu(sycl::queue &q, const float *g1, const float *u1, uint16_t *out,
                      int64_t dim, int64_t n, int64_t ld)
        {
            q.parallel_for(sycl::range<2>(size_t(dim), size_t(n)), [=](sycl::id<2> id) {
                const int64_t i = int64_t(id[0]) * ld + int64_t(id[1]);
                const float g = rb(g1[i]);
                const float s = rb(g / (1.0f + sycl::exp(-g)));
                out[i] = f2bf(s * rb(u1[i]));
            });
        }

        void k_softcap(sycl::queue &q, float *z, int64_t n, float mult, float cap)
        {
            q.parallel_for(sycl::range<1>(size_t(n)), [=](sycl::id<1> i) {
                float v = rb(z[i]);
                v = rb(v * mult);
                if (cap != 0.f)
                {
                    v = rb(v / cap);
                    v = rb(sycl::tanh(v));
                    v = rb(v * cap);
                }
                z[i] = v;
            });
        }

        // Hand-written GEMV for the latency-bound decode path, and the fallback
        // whenever oneDNN is off. Y[out] = W[out, in] * X[in]; one sub-group
        // per output row, so the weight row is read contiguously — which is the
        // whole game, decode being weight-bandwidth-bound.
        //
        // `ldx`/`ldy` are load-bearing and not always 1. Activation buffers are
        // allocated [dim, block] with the *allocated* block as leading
        // dimension, so a single token's column is strided by `block` even when
        // only one column is live. Assuming a lone column is contiguous is
        // wrong for every decode step and for the last-token head GEMM.
        // Gather one token's column out of a dim-major buffer into a packed
        // vector. Decode reads the activation with stride `ld` — 256 bytes at
        // block 128 — so every 2-byte element costs a cache line. Packing it
        // once per GEMV is O(in) and makes the inner loop contiguous.
        void k_compact(sycl::queue &q, const uint16_t *src, uint16_t *dst, int64_t n_elem,
                       int64_t ld)
        {
            q.parallel_for(sycl::range<1>(size_t(n_elem)),
                           [=](sycl::id<1> i) { dst[i] = src[int64_t(i) * ld]; });
        }

        // Pack / unpack the live [dim, n] sub-block of a dim-major buffer into a
        // contiguous [dim * n] staging buffer. Used for the cross-card handoff:
        // a 2D copy of the strided region degenerates at decode into `dim`
        // separate 4-byte transfers (n = 1), which measured slower than every
        // GEMM in the model put together.
        void k_pack2d(sycl::queue &q, const float *src, float *dst, int64_t dim, int64_t n,
                      int64_t ld)
        {
            q.parallel_for(sycl::range<2>(size_t(dim), size_t(n)), [=](sycl::id<2> id) {
                const int64_t i = int64_t(id[0]), t = int64_t(id[1]);
                dst[i * n + t] = src[i * ld + t];
            });
        }
        void k_unpack2d(sycl::queue &q, const float *src, float *dst, int64_t dim, int64_t n,
                        int64_t ld)
        {
            q.parallel_for(sycl::range<2>(size_t(dim), size_t(n)), [=](sycl::id<2> id) {
                const int64_t i = int64_t(id[0]), t = int64_t(id[1]);
                dst[i * ld + t] = src[i * n + t];
            });
        }

        // Sum the shards' packed partials and scatter the total back into a
        // dim-major [dim, n] buffer with leading dimension `ld`.
        //
        // The loop runs in SHARD ORDER on every shard, so all shards compute
        // bit-identical totals. That matters more than it looks: the norms
        // after an all-reduce are replicated, so if two shards disagreed by one
        // ulp here their residual streams would diverge and stay diverged.
        void k_sum_shards(sycl::queue &q, const float *src, float *dst, int64_t nsh,
                          int64_t slot, int64_t dim, int64_t n, int64_t ld)
        {
            q.parallel_for(sycl::range<2>(size_t(dim), size_t(n)), [=](sycl::id<2> id) {
                const int64_t i = int64_t(id[0]), t = int64_t(id[1]);
                float acc = src[i * n + t];
                for (int64_t k = 1; k < nsh; ++k)
                    acc += src[k * slot + i * n + t];
                dst[i * ld + t] = acc;
            });
        }

        // Decode GEMV: Y[out] = W[out, in] * X[in], X packed. One sub-group per
        // output row with 8 bf16 per lane per step, so a sub-group pulls 512
        // contiguous bytes at a time instead of 64. Decode is weight-bandwidth
        // bound, so this loop's job is to keep enough loads in flight to
        // saturate; the scalar version measured ~12x off the card's 1704 GB/s.
        void k_gemv1(sycl::queue &q, const uint16_t *W, const uint16_t *X, float *Y, int64_t in,
                     int64_t out, int64_t ldy)
        {
            constexpr int VEC = 8;
            constexpr int RPG = 4; // output rows per work-group
            const int64_t groups = (out + RPG - 1) / RPG;
            q.submit([&](sycl::handler &h) {
                h.parallel_for(
                    sycl::nd_range<1>(size_t(groups) * RPG * SG, RPG * SG),
                    [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
                        auto sg = it.get_sub_group();
                        const int64_t o =
                            int64_t(it.get_group(0)) * RPG + int64_t(sg.get_group_id()[0]);
                        if (o >= out)
                            return;
                        const int lane = int(sg.get_local_id()[0]);
                        const uint16_t *w = W + o * in;

                        const int64_t rounds = in / (int64_t(SG) * VEC);
                        float part = 0.f;
                        for (int64_t r = 0; r < rounds; ++r)
                        {
                            const int64_t i = r * int64_t(SG) * VEC + int64_t(lane) * VEC;
#pragma unroll
                            for (int k = 0; k < VEC; ++k)
                                part += bf2f(w[i + k]) * bf2f(X[i + k]);
                        }
                        for (int64_t i = rounds * int64_t(SG) * VEC + lane; i < in; i += SG)
                            part += bf2f(w[i]) * bf2f(X[i]);

                        const float s = sycl::reduce_over_group(sg, part, sycl::plus<float>());
                        if (lane == 0)
                            Y[o * ldy] = s;
                    });
            });
        }

        void k_gemv(sycl::queue &q, const uint16_t *W, const uint16_t *X, float *Y, int64_t n,
                    int64_t in, int64_t out, int64_t ldx, int64_t ldy)
        {
            const int64_t rows = out * n;
            q.submit([&](sycl::handler &h) {
                h.parallel_for(sycl::nd_range<1>(size_t(rows) * SG, SG),
                               [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
                                   const int64_t row = int64_t(it.get_group(0));
                                   const int64_t t = row % n, o = row / n;
                                   auto sg = it.get_sub_group();
                                   const int lane = int(sg.get_local_id()[0]);
                                   const uint16_t *w = W + o * in;
                                   float part = 0.f;
                                   for (int64_t i = lane; i < in; i += SG)
                                       part += bf2f(w[i]) * bf2f(X[i * ldx + t]);
                                   const float s =
                                       sycl::reduce_over_group(sg, part, sycl::plus<float>());
                                   if (lane == 0)
                                       Y[o * ldy + t] = s;
                               });
            });
        }

        // =====================================================================

        class SyclEngine final : public Engine
        {
        public:
            // Selects the cards AND enables peer access between them.
            //
            // The ordering is load-bearing and not obvious: the Level-Zero V2
            // adapter establishes its cross-device mappings when the spanning
            // context is CREATED, so peer access has to be enabled before that.
            // Enabling it afterwards is not an error — it just silently leaves
            // the mappings absent, and every cross-card copy then crawls
            // through a fallback (measured here: 0.4 GB/s against 28 for host
            // staging and 48 for a properly mapped peer copy). This function
            // runs before ctx_ because `used_` is declared before it.
            static std::vector<sycl::device> pick_devices(int want)
            {
                auto all = sycl::device::get_devices(sycl::info::device_type::gpu);
                const int n = want > 0 ? std::min<int>(want, int(all.size())) : 1;
                std::vector<sycl::device> devs(all.begin(), all.begin() + n);
                for (int i = 0; i < n; ++i)
                    for (int j = 0; j < n; ++j)
                        if (i != j &&
                            devs[size_t(i)].ext_oneapi_can_access_peer(
                                devs[size_t(j)], sycl::ext::oneapi::peer_access::access_supported))
                            devs[size_t(i)].ext_oneapi_enable_peer_access(devs[size_t(j)]);
                return devs;
            }

            SyclEngine(const Config &c, const Weights &w, const EngineOptions &opt)
                : cfg_(&c), opt_(opt), used_(pick_devices(opt.gpus)),
                  ctx_(used_.empty() ? sycl::context() : sycl::context(used_))
            {
                if (used_.empty())
                    die("no Level-Zero GPU visible (check ONEAPI_DEVICE_SELECTOR and /dev/dri)");
                ngpu_ = int(used_.size());
                nshard_ = opt.shards > 0 ? opt.shards : ngpu_;

                // The tensor-parallel split has to divide the tensors it cuts.
                // Checked here rather than discovered as a wrong logit later.
                const int64_t ns = nshard_;
                if (c.num_attention_heads % ns || c.intermediate_size % ns || c.vocab_size % ns)
                    die("cannot split " + std::to_string(ns) + " ways: needs " +
                        std::to_string(c.num_attention_heads) + " q heads, " +
                        std::to_string(c.intermediate_size) + " intermediate and " +
                        std::to_string(c.vocab_size) + " vocab all divisible");

                block_ = opt.block;
                max_seq_ = opt.max_seq;
                // Tile geometry for the prefill attention. BQ queries share one
                // staged key tile; BK is whatever keeps both tiles inside 32 KiB
                // of local memory (the card has 128 KiB, but smaller tiles leave
                // room for more concurrent work-groups).
                attn_bq_ = 8;
                attn_bk_ = std::clamp<int64_t>(8192 / c.head_dim, 8, 64);
                // the tile loader gives each thread one (row, dim) slot, so the
                // work-group must cover head_dim exactly
                tiled_attn_ = (attn_bq_ * SG) % c.head_dim == 0 &&
                              c.head_dim <= attn_bq_ * SG;
                if (const char *e = getenv("MUSE_GPU_ATTN"); e && std::string(e) == "plain")
                    tiled_attn_ = false;

                flash_prefill_ = opt.flash_prefill;
                q8_ = opt.q8;
                if (const char *e = getenv("MUSE_GPU_Q8_BLOCKDOT"); e && *e && *e != '0')
                    q8_blockdot_ = true;
                tap_layers_ = opt.tap_layers;
                fbq_ = std::min<int64_t>(512, std::max<int64_t>(1, block_));
                fbk_ = 512;
                if (flash_prefill_)
                {
                    const int64_t nqs = c.num_attention_heads / nshard_;
                    // the tier broadcasts one KV head across the shard's query
                    // heads; that is only valid if they all land in one GQA
                    // group
                    const int64_t g0 = 0, g1 = (nqs - 1) / c.kv_groups();
                    if (g0 != g1)
                        die("--flash-prefill needs every query head in a shard to share one "
                            "KV head (this split spans " + std::to_string(g1 + 1) + " groups)");
                }
                if (const char *e = getenv("MUSE_GPU_DECODE_GEMV"); e && *e && *e != '0')
                    decode_gemv_ = true;
                trace_dir_ = getenv("MUSE_GPU_TRACE");
                // direct card-to-card by default; MUSE_GPU_XFER=host selects
                // the pinned-host staging path (same speed, one hop longer)
                p2p_ = true;
                if (const char *e = getenv("MUSE_GPU_XFER"); e && std::string(e) == "host")
                    p2p_ = false;
                if (const char *e = getenv("MUSE_GPU_PROFILE"); e && *e && *e != '0')
                    prof_on_ = true;
                // k_attention carries head_dim/SG accumulators per lane in a
                // fixed-size array; over the bound it would silently drop the
                // tail of every head instead of failing.
                if (c.head_dim > int64_t(SG) * 16)
                    die("head_dim " + std::to_string(c.head_dim) + " exceeds the attention "
                        "kernel's per-lane accumulator bound (" + std::to_string(SG * 16) + ")");

                // One context spanning every card in use. Without it the
                // shards' USM pointers live in different contexts and the
                // cross-card memcpy the all-reduce depends on is invalid.
                shards_.resize(size_t(nshard_));
                for (int i = 0; i < nshard_; ++i)
                {
                    Dev &d = shards_[size_t(i)];
                    d.gpu = i % ngpu_;
                    d.dev = used_[size_t(d.gpu)];
                    d.q = sycl::queue(ctx_, d.dev, sycl::property::queue::in_order{});
#if ORACLE_GPU_DNNL
                    d.eng = dnnl::sycl_interop::make_engine(d.dev, ctx_);
                    d.strm = dnnl::sycl_interop::make_stream(d.eng, d.q);
#endif
                }

                peer_probe("before weight upload");

                auto t0 = std::chrono::steady_clock::now();
                upload_weights(w);
                alloc_scratch();
                // after alloc_scratch: the check drives the real gemm(), which
                // uses the packing scratch on its n == 1 path
                self_check();
                peer_probe("after weights + scratch");
                for (auto &d : shards_)
                    d.q.wait();
                tbytes_ = wbytes_;
                tim_.upload_s =
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

                if (opt.verbose)
                {
                    std::fprintf(stderr,
                                 "[gpu] %d shard(s) on %d card(s), %s, %.2f GiB weights "
                                 "(%.2f/shard), %.2f GiB KV, upload %.1f s\n",
                                 nshard_, ngpu_, q8_ ? "Q8_0" : "BF16",
                                 double(wbytes_) / 1073741824.0,
                                 double(wbytes_) / 1073741824.0 / nshard_,
                                 double(kvbytes_) / 1073741824.0, tim_.upload_s);
                }
            }

            ~SyclEngine() override
            {
                if (host_ring_)
                    sycl::free(host_ring_, ctx_);
                for (const auto &p : owned_)
                    if (p.second >= 0 && p.first)
                        sycl::free(p.first, shards_[size_t(p.second)].q);
            }

            void prefill(const std::vector<int64_t> &ids, float *logits_last) override
            {
                const int64_t T = int64_t(ids.size());
                auto t0 = std::chrono::steady_clock::now();
                for (int64_t p = 0; p < T; p += block_)
                {
                    const int64_t n = std::min(block_, T - p);
                    forward_block(ids.data() + p, p, n, (p + n >= T) ? logits_last : nullptr);
                }
                tim_.prefill_s +=
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                tim_.prefill_tokens += T;
            }

            void decode_step(int64_t id, float *logits_last) override
            {
                auto t0 = std::chrono::steady_clock::now();
                forward_block(&id, len_, 1, logits_last);
                tim_.decode_s +=
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                tim_.decode_tokens += 1;
            }

            int64_t cache_len() const override { return len_; }
            void reset_cache() override { len_ = 0; }
            const Timings &timings() const override { return tim_; }

            void report_profile(std::FILE *f) const override
            {
                std::fprintf(f, "  weights   %8.2f GiB, %d shard(s) on %d card(s)\n",
                             double(wbytes_) / 1073741824.0, nshard_, ngpu_);
                std::fprintf(f, "  kv cache  %8.2f GiB\n", double(kvbytes_) / 1073741824.0);
                std::fprintf(f, "  upload    %8.2f s\n", tim_.upload_s);
                if (tim_.prefill_tokens)
                    std::fprintf(f, "  prefill   %8.3f s  %8ld tok  %9.2f tok/s\n",
                                 tim_.prefill_s, long(tim_.prefill_tokens),
                                 double(tim_.prefill_tokens) / tim_.prefill_s);
                if (tim_.decode_tokens)
                    std::fprintf(f, "  decode    %8.3f s  %8ld tok  %9.2f tok/s\n", tim_.decode_s,
                                 long(tim_.decode_tokens),
                                 double(tim_.decode_tokens) / tim_.decode_s);
                if (prof_on_)
                {
                    static const char *nm[S_NSTAGE] = {"norm", "qkv+gate", "attn",
                                                       "o_proj+resid", "mlp", "head", "xfer"};
                    double tot = 0;
                    for (int i = 0; i < S_NSTAGE; ++i)
                        tot += stage_s_[i];
                    std::fprintf(f, "  --- stage attribution (queue-serialized) ---\n");
                    for (int i = 0; i < S_NSTAGE; ++i)
                        if (stage_s_[i] > 0)
                            std::fprintf(f, "  %-14s %8.3f s  %5.1f%%\n", nm[i], stage_s_[i],
                                         100.0 * stage_s_[i] / tot);
                    std::fprintf(f, "  %-14s %8.3f s\n", "total", tot);
                    if (ar_calls_)
                        std::fprintf(f,
                                     "  all-reduce: %ld calls  pack %.3f s  exchange %.3f s "
                                     "(%.2f GB, %.1f GB/s)  sum %.3f s\n",
                                     ar_calls_, ar_pack_s_, ar_xchg_s_, ar_bytes_ / 1e9,
                                     ar_bytes_ / ar_xchg_s_ / 1e9, ar_sum_s_);
                }
            }

        private:
            // ------------------------------------------------------------ alloc

            void seal_allocs(int mode) override
            {
                sealed_ = mode;
                if (opt_.verbose && mode)
                    std::fprintf(stderr, "[gpu] allocations sealed (mode %d) at %.2f GiB\n", mode,
                                 double(wbytes_ + kvbytes_) / 1073741824.0);
            }

            template <class T> T *dalloc(int dev, size_t count)
            {
                auto &d = shards_[size_t(dev)];
                if (sealed_)
                {
                    const std::string msg =
                        "allocation of " + std::to_string(count * sizeof(T) / 1048576) +
                        " MiB on shard " + std::to_string(dev) + " AFTER the load seal";
                    if (sealed_ >= 2)
                        die(msg + " (mode 2: refusing — this would be a mid-request OOM)");
                    std::fprintf(stderr, "[seal] %s\n", msg.c_str());
                }
                T *p = sycl::malloc_device<T>(count, d.q);
                if (!p)
                    die("device " + std::to_string(dev) + ": out of memory allocating " +
                        std::to_string(count * sizeof(T) / 1048576) + " MiB");
                owned_.emplace_back(static_cast<void *>(p), dev);
                return p;
            }

            uint16_t *up(int dev, const uint16_t *src, size_t count)
            {
                uint16_t *p = dalloc<uint16_t>(dev, count);
                shards_[size_t(dev)].q.memcpy(p, src, count * 2).wait();
                wbytes_ += int64_t(count * 2);
                return p;
            }

            // Gather columns [c0, c0+cn) out of a row-major [rows, cols] tensor
            // and upload. o_proj [H, QD] and mlp_down [H, I] are the
            // column-sharded pair, and a column range of a row-major tensor is
            // `rows` strided segments, not a slice.
            uint16_t *up_cols(int s, const uint16_t *src, int64_t rows, int64_t cols, int64_t c0,
                              int64_t cn)
            {
                stage_.resize(static_cast<size_t>(rows * cn));
                for (int64_t r = 0; r < rows; ++r)
                    std::memcpy(stage_.data() + r * cn, src + r * cols + c0, size_t(cn) * 2);
                return up(s, stage_.data(), static_cast<size_t>(rows * cn));
            }

            // Upload one [out, in] matrix in whatever tier is active. Q8_0 is
            // quantized here, from the same BF16 checkpoint the bf16 tier
            // binds, with llama.cpp's quantize_row_q8_0_ref semantics.
            QW up_w(int sh, const uint16_t *src, int64_t out, int64_t in)
            {
                if (!q8_)
                    return QW{up(sh, src, size_t(out * in)), nullptr, nullptr};
                if (in % QK8)
                    die("Q8_0 needs the input dim divisible by 32 (got " +
                        std::to_string(in) + ")");
                const int64_t nblk = in / QK8;
                qbuf_.resize(static_cast<size_t>(out * in));
                dbuf8_.resize(static_cast<size_t>(out * nblk));
                // 25 G elements for the 30B, so this is not a place for a
                // serial loop: rows are independent and write disjoint output.
#pragma omp parallel for schedule(static)
                for (int64_t o = 0; o < out; ++o)
                    for (int64_t b = 0; b < nblk; ++b)
                    {
                        const uint16_t *row = src + o * in + b * QK8;
                        float amax = 0.f;
                        for (int64_t i = 0; i < QK8; ++i)
                            amax = std::max(amax, std::fabs(muse::bf16::bf16_to_f32(row[i])));
                        const float d = amax / 127.f;
                        const float id = d != 0.f ? 1.f / d : 0.f;
                        dbuf8_[size_t(o * nblk + b)] = f32_to_f16(d);
                        for (int64_t i = 0; i < QK8; ++i)
                            qbuf_[size_t(o * in + b * QK8 + i)] = int8_t(
                                std::lround(muse::bf16::bf16_to_f32(row[i]) * id));
                    }
                int8_t *dq = dalloc<int8_t>(sh, size_t(out * in));
                uint16_t *dd = dalloc<uint16_t>(sh, size_t(out * nblk));
                shards_[size_t(sh)].q.memcpy(dq, qbuf_.data(), qbuf_.size()).wait();
                shards_[size_t(sh)].q.memcpy(dd, dbuf8_.data(), dbuf8_.size() * 2).wait();
                wbytes_ += int64_t(out * in) + int64_t(out * nblk) * 2;
                return QW{nullptr, dq, dd};
            }

            QW up_cols_w(int sh, const uint16_t *src, int64_t rows, int64_t cols, int64_t c0,
                         int64_t cn)
            {
                stage_.resize(static_cast<size_t>(rows * cn));
                for (int64_t r = 0; r < rows; ++r)
                    std::memcpy(stage_.data() + r * cn, src + r * cols + c0, size_t(cn) * 2);
                return up_w(sh, stage_.data(), rows, cn);
            }

            void upload_weights(const Weights &w)
            {
                const Config &c = *cfg_;
                auto v = muse::bf16::bind(w);
                const int64_t H = c.hidden_size, I = c.intermediate_size, V = c.vocab_size;
                const int64_t QD = c.q_dim(), KD = c.kv_dim();
                const int64_t L = c.num_hidden_layers, ns = nshard_;
                const int64_t QDs = QD / ns, Is = I / ns, Vs = V / ns;

                embed_.assign(size_t(ns), nullptr);
                lm_head_.assign(size_t(ns), QW{});
                final_norm_.assign(size_t(ns), nullptr);
                for (int64_t sh = 0; sh < ns; ++sh)
                {
                    // The embedding table is replicated so every shard can build
                    // its own copy of the residual stream with no broadcast.
                    embed_[size_t(sh)] = up(int(sh), v.embed, size_t(V * H));
                    final_norm_[size_t(sh)] = up(int(sh), v.final_norm, size_t(H));
                    // vocab-parallel head: shard sh owns rows [sh*Vs, (sh+1)*Vs).
                    // The multiplier and softcap are elementwise and commute
                    // with the split, so no reduction is needed here.
                    lm_head_[size_t(sh)] = up_w(int(sh), v.lm_head + sh * Vs * H, Vs, H);
                }

                layers_.resize(size_t(L));
                for (int64_t li = 0; li < L; ++li)
                {
                    const auto &t = v.layers[size_t(li)];
                    GpuLayer &g = layers_[size_t(li)];
                    g.cap = c.layer_is_sliding(li) ? std::min(c.sliding_window + block_, max_seq_)
                                                   : max_seq_;
                    for (auto *arr : {&g.input_ln, &g.post_attn_ln, &g.pre_ff_ln, &g.post_ff_ln,
                                      &g.kc, &g.vc})
                        arr->assign(size_t(ns), nullptr);
                    for (auto *arr : {&g.k_, &g.v_, &g.q, &g.gate, &g.mlp_gate, &g.mlp_up, &g.o,
                                      &g.mlp_down})
                        arr->assign(size_t(ns), QW{});

                    for (int64_t sh = 0; sh < ns; ++sh)
                    {
                        const int d = int(sh);
                        const size_t k = size_t(sh);
                        g.input_ln[k] = up(d, t.input_ln, size_t(H));
                        g.post_attn_ln[k] = up(d, t.post_attn_ln, size_t(H));
                        g.pre_ff_ln[k] = up(d, t.pre_ff_ln, size_t(H));
                        g.post_ff_ln[k] = up(d, t.post_ff_ln, size_t(H));
                        g.k_[k] = up_w(d, t.k, KD, H);
                        g.v_[k] = up_w(d, t.v, KD, H);
                        // row shards: contiguous slices of a row-major [out, in]
                        g.q[k] = up_w(d, t.q + sh * QDs * H, QDs, H);
                        g.gate[k] = up_w(d, t.gate + sh * QDs * H, QDs, H);
                        g.mlp_gate[k] = up_w(d, t.mlp_gate + sh * Is * H, Is, H);
                        g.mlp_up[k] = up_w(d, t.mlp_up + sh * Is * H, Is, H);
                        // column shards: partial sums, all-reduced after the GEMM
                        g.o[k] = up_cols_w(d, t.o, H, QD, sh * QDs, QDs);
                        g.mlp_down[k] = up_cols_w(d, t.mlp_down, H, I, sh * Is, Is);

                        g.kc[k] = dalloc<uint16_t>(d, size_t(g.cap * KD));
                        g.vc[k] = dalloc<uint16_t>(d, size_t(g.cap * KD));
                        shards_[k].q.memset(g.kc[k], 0, size_t(g.cap * KD) * 2);
                        shards_[k].q.memset(g.vc[k], 0, size_t(g.cap * KD) * 2);
                        kvbytes_ += int64_t(g.cap * KD) * 4;
                    }
                }
                stage_.clear();
                stage_.shrink_to_fit();
                qbuf_.clear();
                qbuf_.shrink_to_fit();
                dbuf8_.clear();
                dbuf8_.shrink_to_fit();
            }

            void alloc_scratch()
            {
                const Config &c = *cfg_;
                const int64_t H = c.hidden_size, I = c.intermediate_size;
                const int64_t QD = c.q_dim(), KD = c.kv_dim(), V = c.vocab_size;
                const int64_t B = block_, ns = nshard_;
                bprims_.resize(size_t(ns));
                const int64_t QDs = QD / ns, Is = I / ns, Vs = V / ns;

                // The twin builds cos/sin through the stock f32 chain; reuse
                // the oracle's builder so the two engines cannot drift on rope.
                rope_ = muse::build_rope_table(c, max_seq_, false, prec::Dtype::BF16);
                std::vector<float> cos32(rope_.cos.size()), sin32(rope_.sin.size());
                for (size_t i = 0; i < rope_.cos.size(); ++i)
                {
                    cos32[i] = float(rope_.cos[i]);
                    sin32[i] = float(rope_.sin[i]);
                }

                for (int i = 0; i < nshard_; ++i)
                {
                    Dev &d = shards_[size_t(i)];
                    // replicated full-width: the residual stream and the KV
                    // projections (k/v are not sharded, see GpuLayer)
                    d.h = dalloc<float>(i, size_t(B * H));
                    d.xf = dalloc<float>(i, size_t(B * H));
                    d.xb = dalloc<uint16_t>(i, size_t(B * H));
                    d.kf = dalloc<float>(i, size_t(B * KD));
                    d.kb = dalloc<uint16_t>(i, size_t(B * KD));
                    d.vf = dalloc<float>(i, size_t(B * KD));
                    // sharded widths
                    d.qf = dalloc<float>(i, size_t(B * QDs));
                    d.qb = dalloc<uint16_t>(i, size_t(B * QDs));
                    d.gf = dalloc<float>(i, size_t(B * QDs));
                    d.of = dalloc<float>(i, size_t(B * QDs));
                    d.ob = dalloc<uint16_t>(i, size_t(B * QDs));
                    d.g1 = dalloc<float>(i, size_t(B * Is));
                    d.u1 = dalloc<float>(i, size_t(B * Is));
                    d.g1b = dalloc<uint16_t>(i, size_t(B * Is));
                    d.logits = dalloc<float>(i, size_t(Vs));
                    d.ids = dalloc<int32_t>(i, size_t(B));
                    // widest GEMV input: H for q/k/v/gate/mlp/head, QDs for
                    // o_proj, Is for mlp_down
                    d.pack = dalloc<uint16_t>(i, size_t(std::max({H, QDs, Is, KD})));
                    if (q8_)
                    {
                        // widest sharded weight, expanded to bf16 for prefill
                        deq_cap_ = std::max({Is * H, H * Is, QDs * H, H * QDs, Vs * H});
                        d.deq = dalloc<uint16_t>(i, size_t(deq_cap_));
                    }
                    d.xfer = dalloc<float>(i, size_t(B * H));
                    d.taps.assign(tap_layers_.size(), nullptr);
                    for (size_t k = 0; k < tap_layers_.size(); ++k)
                        d.taps[k] = dalloc<float>(i, size_t(B * H));
                    d.peer = dalloc<float>(i, size_t(ns * B * H));
                    if (flash_prefill_)
                    {
                        const int64_t nqs = c.num_attention_heads / ns;
                        d.fs = dalloc<float>(i, size_t(nqs * fbq_ * fbk_));
                        d.fp = dalloc<uint16_t>(i, size_t(nqs * fbq_ * fbk_));
                        d.foacc = dalloc<float>(i, size_t(nqs * fbq_ * c.head_dim));
                        d.fm = dalloc<float>(i, size_t(nqs * fbq_));
                        d.fl = dalloc<float>(i, size_t(nqs * fbq_));
                    }
                    d.rope_cos = dalloc<float>(i, cos32.size());
                    d.rope_sin = dalloc<float>(i, sin32.size());
                    d.q.memcpy(d.rope_cos, cos32.data(), cos32.size() * 4);
                    d.q.memcpy(d.rope_sin, sin32.data(), sin32.size() * 4);
                    d.q.wait();
                }
                if (ns > 1 && !p2p_)
                {
                    // pinned: pageable host memory would halve the transfer
                    host_ring_ = sycl::malloc_host<float>(size_t(ns * B * H), ctx_);
                    if (!host_ring_)
                        die("could not pin the all-reduce staging ring");
                }
            }

            // Sum the shards' partial residual streams and leave the total on
            // every shard. Called after o_proj and after mlp_down, the two
            // column-sharded GEMMs — 2 per layer, 104 per token at 52 layers.
            //
            // Two rules, both load-bearing on this hardware:
            //  * no kernel ever dereferences peer USM. The exchange is a
            //    copy-engine memcpy between cards in one shared context, which
            //    is the only cross-card access the Arc driver tolerates.
            //  * the sum runs in shard order on every shard (k_sum_shards), so
            //    all shards hold bit-identical totals. The norms that follow
            //    are replicated, so any disagreement here would compound.
            void all_reduce(int64_t n, int64_t ld)
            {
                const int64_t ns = nshard_, H = cfg_->hidden_size;
                if (ns == 1)
                    return; // xf is already the whole sum
                const int64_t slot = block_ * H;         // per-shard slot stride
                const size_t bytes = size_t(n * H) * 4;  // live payload
                const auto t_ar_ = std::chrono::steady_clock::now();

                // Pack the live [H, n] block straight into this shard's own
                // slot: it is contiguous only when n == ld, and a strided
                // cross-card copy degenerates badly (see docs/gpu.md).
                for (int64_t sh = 0; sh < ns; ++sh)
                {
                    Dev &d = shards_[size_t(sh)];
                    k_pack2d(d.q, d.xf, d.peer + sh * slot, H, n, ld);
                }
                for (int64_t sh = 0; sh < ns; ++sh)
                    shards_[size_t(sh)].q.wait();
                const auto tp = std::chrono::steady_clock::now();

                // Transport: direct card-to-card, serialized (see below), with
                // pinned-host staging available via MUSE_GPU_XFER=host. Both
                // land at ~27 GB/s, which is the PCIe link, so the choice is
                // about robustness rather than speed.
                std::vector<sycl::event> ev;
                if (p2p_)
                {
                    // PULL, not push: issued on the DESTINATION's queue, which
                    // is how both sibling repos do it.
                    //
                    // SERIALIZED, one leg at a time. Letting both cards pull
                    // from each other concurrently is bimodal on this driver:
                    // the same 13.6 MB disjoint exchange measures 0.56 ms
                    // (48 GB/s) on one run and 639 ms on the next, with no
                    // change in size, alignment or aliasing. Serializing costs
                    // the concurrency but is stable at 28 GB/s — the same rate
                    // host staging achieves, and PCIe is the limit either way.
                    for (int64_t r = 0; r < ns; ++r)
                        for (int64_t sh = 0; sh < ns; ++sh)
                            if (r != sh)
                                shards_[size_t(r)]
                                    .q.memcpy(shards_[size_t(r)].peer + sh * slot,
                                              shards_[size_t(sh)].peer + sh * slot, bytes)
                                    .wait();
                }
                else
                {
                    for (int64_t sh = 0; sh < ns; ++sh)
                        ev.push_back(shards_[size_t(sh)].q.memcpy(
                            host_ring_ + sh * slot, shards_[size_t(sh)].peer + sh * slot, bytes));
                    for (auto &e : ev)
                        e.wait();
                    ev.clear();
                    for (int64_t sh = 0; sh < ns; ++sh)
                        for (int64_t r = 0; r < ns; ++r)
                            if (r != sh)
                                ev.push_back(shards_[size_t(r)].q.memcpy(
                                    shards_[size_t(r)].peer + sh * slot, host_ring_ + sh * slot,
                                    bytes));
                    for (auto &e : ev)
                        e.wait();
                }
                const auto tx = std::chrono::steady_clock::now();

                for (int64_t sh = 0; sh < ns; ++sh)
                {
                    Dev &d = shards_[size_t(sh)];
                    k_sum_shards(d.q, d.peer, d.xf, ns, slot, H, n, ld);
                }
                if (prof_on_)
                {
                    for (int64_t sh = 0; sh < ns; ++sh)
                        shards_[size_t(sh)].q.wait();
                    const auto te = std::chrono::steady_clock::now();
                    ar_pack_s_ += std::chrono::duration<double>(tp - t_ar_).count();
                    ar_xchg_s_ += std::chrono::duration<double>(tx - tp).count();
                    ar_sum_s_ += std::chrono::duration<double>(te - tx).count();
                    ar_bytes_ += double(bytes) * double(ns - 1) * double(ns);
                    ar_calls_++;
                }
            }

            // ----------------------------------------------------------- gemm
            //
            // Y[out, n] = W[out, in] * X[in, n], every operand dim-major with
            // leading dimension `block_`. See the header comment for why the
            // weight is the natural-layout operand rather than the transposed
            // one.
#if ORACLE_GPU_DNNL
            std::map<std::array<int64_t, 5>, Dev::Prim>::iterator
            make_prim(Dev &d, const std::array<int64_t, 5> &key, const dnnl::memory::desc &a_md,
                      const dnnl::memory::desc &b_md, const dnnl::memory::desc &c_md)
            {
                dnnl::matmul::primitive_desc pd(d.eng, a_md, b_md, c_md);
                Dev::Prim p{dnnl::matmul(pd), dnnl::memory(a_md, d.eng, nullptr),
                            dnnl::memory(b_md, d.eng, nullptr),
                            dnnl::memory(c_md, d.eng, nullptr), {}};
                p.args = {{DNNL_ARG_SRC, p.a}, {DNNL_ARG_WEIGHTS, p.b}, {DNNL_ARG_DST, p.c}};
                return d.prims.emplace(key, std::move(p)).first;
            }

            // dnnl::memory is a handle with shared semantics, so the copies
            // sitting in `args` see the swapped pointers.
            void execute_prim(Dev &d, Dev::Prim &p, const uint16_t *W, const uint16_t *X,
                              float *Y)
            {
                p.a.set_data_handle(const_cast<uint16_t *>(W));
                p.b.set_data_handle(const_cast<uint16_t *>(X));
                p.c.set_data_handle(Y);
                p.prim.execute(d.strm, p.args);
            }
#endif

            // `ldx`/`ldy` are REQUIRED, never defaulted. They used to default to
            // the allocated block, which silently desynchronized from the
            // per-block leading dimension the moment decode started collapsing
            // it to 1 — the GEMMs addressed a stride the rest of the kernels no
            // longer used, and decode produced a stuck token with no error.
            // Tier-aware. Q8_0 takes the hand-written GEMV at decode (where the
            // halved byte count is the whole point) and expands to bf16 for
            // prefill (where oneDNN's DPAS path is, and where oneDNN has no
            // grouped-scale entry to take instead).
            void gemm(int dev, const QW &w, const uint16_t *X, float *Y, int64_t n, int64_t in,
                      int64_t out, int64_t ldx, int64_t ldy)
            {
                Dev &d = shards_[size_t(dev)];
                if (w.q8())
                {
                    if (n == 1)
                    {
                        k_compact(d.q, X, d.pack, in, ldx);
                        if (q8_blockdot_)
                            k_gemv_q8<true>(d.q, w.qs, w.d, d.pack, Y, in, out, ldy);
                        else
                            k_gemv_q8<false>(d.q, w.qs, w.d, d.pack, Y, in, out, ldy);
                        return;
                    }
                    if (int64_t(out * in) > deq_cap_)
                        die("Q8 prefill scratch too small for a " + std::to_string(out) + "x" +
                            std::to_string(in) + " weight");
                    k_dequant_q8(d.q, w.qs, w.d, d.deq, in, out);
                    gemm_bf16(dev, d.deq, X, Y, n, in, out, ldx, ldy);
                    return;
                }
                gemm_bf16(dev, w.w, X, Y, n, in, out, ldx, ldy);
            }

            void gemm_bf16(int dev, const uint16_t *W, const uint16_t *X, float *Y, int64_t n,
                           int64_t in, int64_t out, int64_t ldx, int64_t ldy)
            {
                Dev &d = shards_[size_t(dev)];
#if ORACLE_GPU_DNNL
                // n == 1 NEVER goes to oneDNN. Measured on oneDNN 3.11.2 /
                // Arc B70: a matmul whose B has exactly one column silently
                // ignores the offset of the B handle, so `x + (n-1)` — the
                // last-token view the head GEMM and every decode step use —
                // reads the wrong column and the logits are quietly wrong
                // while every layer above is bitwise correct. Offset 0 is
                // fine, and n >= 2 with an offset is fine; it is specifically
                // the one-column case. `--no-dnnl` and the startup self-check
                // below exist because of this bug: it cost a full bisect to
                // find, and nothing about it is visible as an error.
                if (!opt_.no_dnnl && n > 1)
                {
                    using dt = dnnl::memory::data_type;
                    const dnnl::memory::desc a_md({out, in}, dt::bf16, dnnl::memory::dims{in, 1});
                    const dnnl::memory::desc b_md({in, n}, dt::bf16, dnnl::memory::dims{ldx, 1});
                    const dnnl::memory::desc c_md({out, n}, dt::f32, dnnl::memory::dims{ldy, 1});
                    const std::array<int64_t, 5> key{out, in, n, ldx, ldy};
                    auto it = d.prims.find(key);
                    if (it == d.prims.end())
                        it = make_prim(d, key, a_md, b_md, c_md);
                    execute_prim(d, it->second, W, X, Y);
                    return;
                }
#endif
                if (n == 1)
                {
                    // Pack the live column unconditionally. Two reasons, both
                    // load-bearing: it turns a stride-`ldx` read into a
                    // contiguous one, and it puts the activation at offset 0 of
                    // its own allocation — which is the one place oneDNN's
                    // one-column path is safe (see the comment above; the bug
                    // is that it drops a non-zero B-handle offset).
                    k_compact(d.q, X, d.pack, in, ldx);
#if ORACLE_GPU_DNNL
                    if (!opt_.no_dnnl)
                    {
                        // measured on the real decode shapes: 1344 GB/s here
                        // against 849 for the hand-written kernel below, on a
                        // card that streams 1704
                        using dt = dnnl::memory::data_type;
                        const dnnl::memory::desc a_md({out, in}, dt::bf16,
                                                      dnnl::memory::dims{in, 1});
                        const dnnl::memory::desc b_md({in, 1}, dt::bf16,
                                                      dnnl::memory::dims{1, 1});
                        const dnnl::memory::desc c_md({out, 1}, dt::f32,
                                                      dnnl::memory::dims{ldy, 1});
                        const std::array<int64_t, 5> key{out, in, 1, 1, ldy};
                        auto it = d.prims.find(key);
                        if (it == d.prims.end())
                            it = make_prim(d, key, a_md, b_md, c_md);
                        execute_prim(d, it->second, W, d.pack, Y);
                        return;
                    }
#endif
                    k_gemv1(d.q, W, d.pack, Y, in, out, ldy);
                    return;
                }
                k_gemv(d.q, W, X, Y, n, in, out, ldx, ldy);
            }

#if ORACLE_GPU_DNNL
            // Batched matmul with explicit strides, for the flash tier's two
            // GEMMs. `bdims`/`bstr` are {batch, rows, cols} triples; a batch
            // extent of 1 on B broadcasts across A's heads, which is what lets
            // one call cover every query head against a single shared KV head.
            void bmm(int sh, const void *A, dnnl::memory::data_type ta,
                     const std::array<int64_t, 3> &ad, const std::array<int64_t, 3> &as,
                     const void *B, dnnl::memory::data_type tb,
                     const std::array<int64_t, 3> &bd, const std::array<int64_t, 3> &bs, void *C,
                     const std::array<int64_t, 3> &cd, const std::array<int64_t, 3> &cs,
                     bool accumulate)
            {
                Dev &d = shards_[size_t(sh)];
                using dt = dnnl::memory::data_type;
                const dnnl::memory::desc a_md({ad[0], ad[1], ad[2]}, ta,
                                              dnnl::memory::dims{as[0], as[1], as[2]});
                const dnnl::memory::desc b_md({bd[0], bd[1], bd[2]}, tb,
                                              dnnl::memory::dims{bs[0], bs[1], bs[2]});
                const dnnl::memory::desc c_md({cd[0], cd[1], cd[2]}, dt::f32,
                                              dnnl::memory::dims{cs[0], cs[1], cs[2]});
                const std::array<int64_t, 8> key{ad[0],  ad[1], ad[2], bd[2],
                                                 cs[0],  cs[1], int64_t(accumulate),
                                                 int64_t(ta == dt::bf16 ? 1 : 0)};
                auto it = bprims_[size_t(sh)].find(key);
                if (it == bprims_[size_t(sh)].end())
                {
                    dnnl::primitive_attr attr;
                    if (accumulate)
                    {
                        // C = C + A*B, so the second GEMM folds each key tile
                        // into the running output without a separate add
                        dnnl::post_ops po;
                        po.append_sum(1.0f);
                        attr.set_post_ops(po);
                    }
                    dnnl::matmul::primitive_desc pd(d.eng, a_md, b_md, c_md, attr);
                    it = bprims_[size_t(sh)].emplace(key, dnnl::matmul(pd)).first;
                }
                dnnl::memory am(a_md, d.eng, const_cast<void *>(A));
                dnnl::memory bm(b_md, d.eng, const_cast<void *>(B));
                dnnl::memory cm(c_md, d.eng, C);
                it->second.execute(
                    d.strm, {{DNNL_ARG_SRC, am}, {DNNL_ARG_WEIGHTS, bm}, {DNNL_ARG_DST, cm}});
            }

            // ------------------------------------------------ --flash-prefill
            //
            // Attention with the q.k and p.v products on the matrix engines
            // instead of a sub-group reduction per (query, key).
            //
            // THIS IS A DIFFERENT NUMERICAL CONTRACT and that is the whole
            // point of it being opt-in. The exact kernel folds each key into
            // the running softmax one at a time, which is what reproduces the
            // twin. A matrix engine can only be used if the scores for a whole
            // tile are produced at once, so the max and the sum are taken over
            // the TILE and the accumulator is rescaled once per tile rather
            // than once per key. Same function, different summation schedule —
            // it cannot be bitwise against the eager twin or the flash twin,
            // and it is gated on the logit envelope instead.
            //
            // Measured floor for the GEMMs alone was 0.09 s per forward at
            // T=2048 against 1.18 s for the exact kernel; this is that 13x
            // being cashed in, minus the softmax passes.
            void attention_flash(int sh, const GpuLayer &l, int64_t n, int64_t pos0, int64_t nqs,
                                 int64_t D, int64_t KD, int64_t ld, int64_t groups, int64_t cap,
                                 int64_t window, float scaling, int64_t head0)
            {
                using dt = dnnl::memory::data_type;
                Dev &d = shards_[size_t(sh)];
                const int64_t FBQ = fbq_, FBK = fbk_;
                // Every query head in this shard reads the same KV head, so the
                // K/V operands broadcast across the batch. (Checked at startup;
                // shards that would span several GQA groups do not take this
                // path.)
                const int64_t g = head0 / groups;

                for (int64_t t0 = 0; t0 < n; t0 += FBQ)
                {
                    const int64_t rows = std::min(FBQ, n - t0);
                    const int64_t qhi = pos0 + t0 + rows - 1;
                    const int64_t klo =
                        window > 0 ? std::max<int64_t>(0, pos0 + t0 - window + 1) : 0;

                    k_fill(d.q, d.fm, nqs * FBQ, -INFINITY);
                    k_fill(d.q, d.fl, nqs * FBQ, 0.f);
                    k_fill(d.q, d.foacc, nqs * FBQ * D, 0.f);

                    int64_t j = klo;
                    while (j <= qhi)
                    {
                        // A run must not cross the ring's wrap point, or the
                        // keys stop being contiguous and the GEMM's strides lie.
                        const int64_t slot = j % cap;
                        const int64_t mk =
                            std::min({FBK, qhi - j + 1, cap - slot});

                        // S[h, rows, mk] = Q[h, rows, D] . K[mk, D]^T
                        bmm(sh, d.qb + t0, dt::bf16, {nqs, rows, D}, {D * ld, 1, ld},
                            l.kc[size_t(sh)] + slot * KD + g * D, dt::bf16, {1, D, mk},
                            {D * KD, 1, KD}, d.fs, {nqs, rows, mk}, {FBQ * FBK, FBK, 1}, false);

                        k_flash_softmax(d.q, d.fs, d.fp, d.foacc, d.fm, d.fl, nqs, rows, mk, FBQ,
                                        FBK, D, t0, j, pos0, window, scaling);

                        // O[h, rows, D] += P[h, rows, mk] . V[mk, D]
                        bmm(sh, d.fp, dt::bf16, {nqs, rows, mk}, {FBQ * FBK, FBK, 1},
                            l.vc[size_t(sh)] + slot * KD + g * D, dt::bf16, {1, mk, D},
                            {mk * KD, KD, 1}, d.foacc, {nqs, rows, D}, {FBQ * D, D, 1}, true);

                        j += mk;
                    }
                    k_flash_finish(d.q, d.foacc, d.fl, d.of, nqs, rows, FBQ, D, ld, t0);
                }
            }
#endif

#if ORACLE_GPU_DNNL
            // Y[rows, out] = X[rows, in] . W[out, in]^T, everything row-major.
            // The drafter's shapes are small (16 rows), so the layout that
            // matches the reference beats the one tuned for 2048-token GEMMs.
            // Tier-aware overload, for the one row-major call that reaches a
            // tier-carrying weight: the drafter heads through the TARGET's
            // lm_head. The drafter's and the tower's own weights stay bf16.
            void gemm_rm(int sh, const QW &w, const uint16_t *X, float *Y, int64_t rows,
                         int64_t in, int64_t out)
            {
                if (!w.q8())
                {
                    gemm_rm(sh, w.w, X, Y, rows, in, out);
                    return;
                }
                Dev &d = shards_[size_t(sh)];
                if (int64_t(out * in) > deq_cap_)
                    die("Q8 scratch too small for a " + std::to_string(out) + "x" +
                        std::to_string(in) + " weight");
                k_dequant_q8(d.q, w.qs, w.d, d.deq, in, out);
                gemm_rm(sh, d.deq, X, Y, rows, in, out);
            }

            void gemm_rm(int sh, const uint16_t *W, const uint16_t *X, float *Y, int64_t rows,
                         int64_t in, int64_t out)
            {
                using dt = dnnl::memory::data_type;
                bmm(sh, X, dt::bf16, {1, rows, in}, {rows * in, in, 1}, W, dt::bf16,
                    {1, in, out}, {in * out, 1, in}, Y, {1, rows, out}, {rows * out, out, 1},
                    false);
            }
#endif

            // All-reduce over a contiguous per-shard buffer of `count` floats,
            // chunked through the existing exchange slots so the context
            // projection (which is n*H and grows with the prompt) does not need
            // a staging buffer of its own.
            template <class F> void all_reduce_buf(int64_t count, F get)
            {
                const int64_t ns = nshard_;
                if (ns == 1)
                    return;
                const int64_t slot = block_ * cfg_->hidden_size;
                for (int64_t off = 0; off < count; off += slot)
                {
                    const int64_t m = std::min(slot, count - off);
                    for (int64_t sh = 0; sh < ns; ++sh)
                        shards_[size_t(sh)].q.memcpy(shards_[size_t(sh)].peer + sh * slot,
                                                     get(int(sh)) + off, size_t(m) * 4);
                    for (int64_t sh = 0; sh < ns; ++sh)
                        shards_[size_t(sh)].q.wait();
                    for (int64_t r = 0; r < ns; ++r)
                        for (int64_t sh = 0; sh < ns; ++sh)
                            if (r != sh)
                                shards_[size_t(r)]
                                    .q.memcpy(shards_[size_t(r)].peer + sh * slot,
                                              shards_[size_t(sh)].peer + sh * slot, size_t(m) * 4)
                                    .wait();
                    for (int64_t sh = 0; sh < ns; ++sh)
                        k_sum_flat(shards_[size_t(sh)].q, shards_[size_t(sh)].peer,
                                   get(int(sh)) + off, ns, slot, m);
                }
            }

            // ============================================================ vision
            //
            // The tower is 50 layers of ViT at hidden 1536 and runs once per
            // image; on CPU that is seconds, which is why it is worth moving.
            //
            // Everything that is an INDEX rather than an arithmetic result is
            // computed on the host by the already-gated code in vision.hpp --
            // the window permutation, the cu_seqlens segment bounds, the
            // bilinear position taps, the 2-D rope table, the pixel-shuffle
            // source map. Those are the parts with the fiddly conventions (the
            // w/h flip and +1, the [fw|fh|fw|fh] frequency layout, the
            // channel-major merge), and reimplementing them on the device would
            // be reimplementing the traps.
            void bind_vision(const vision::Config &vc, const vision::Weights &vw,
                             int64_t max_patches) override
            {
                vcfg_ = vc;
                vmaxn_ = max_patches;
                const int64_t ns = nshard_, H = vc.hidden_size, I = vc.intermediate_size;
                const int64_t Hs = H / ns, Is = I / ns;
                const int64_t P = vc.projector_hidden_size, OH = vc.out_hidden_size;
                const int64_t TH = cfg_->hidden_size;
                if (H % ns || I % ns || vc.num_attention_heads % ns)
                    die("vision tower does not split " + std::to_string(ns) + " ways");
                auto bf = [](const st::Tensor &t) { return muse::bf16::as_bf16(t); };

                vpatch_.assign(size_t(ns), nullptr);
                vpos_.assign(size_t(ns), nullptr);
                vlnpre_w_.assign(size_t(ns), nullptr);
                vlnpre_b_.assign(size_t(ns), nullptr);
                vlnpost_w_.assign(size_t(ns), nullptr);
                vlnpost_b_.assign(size_t(ns), nullptr);
                vad1_.assign(size_t(ns), nullptr);
                vad2_.assign(size_t(ns), nullptr);
                vproj_.assign(size_t(ns), nullptr);
                for (int64_t sh = 0; sh < ns; ++sh)
                {
                    const int d = int(sh);
                    vpatch_[size_t(sh)] = up(d, bf(*vw.patch_embed), size_t(H * vc.patch_dim()));
                    vpos_[size_t(sh)] = up(d, bf(*vw.pos_table),
                                           size_t(vc.pos_emb_height * vc.pos_emb_width * H));
                    vlnpre_w_[size_t(sh)] = up(d, bf(*vw.ln_pre_w), size_t(H));
                    vlnpre_b_[size_t(sh)] = up(d, bf(*vw.ln_pre_b), size_t(H));
                    vlnpost_w_[size_t(sh)] = up(d, bf(*vw.ln_post_w), size_t(H));
                    vlnpost_b_[size_t(sh)] = up(d, bf(*vw.ln_post_b), size_t(H));
                    vad1_[size_t(sh)] = up(d, bf(*vw.adapter_fc1), size_t(P * OH));
                    vad2_[size_t(sh)] = up(d, bf(*vw.adapter_fc2), size_t(P * P));
                    vproj_[size_t(sh)] = up(d, bf(*vw.projection), size_t(TH * P));
                }

                vlayers_.resize(size_t(vc.num_hidden_layers));
                for (int64_t li = 0; li < vc.num_hidden_layers; ++li)
                {
                    const auto &t = vw.layers[size_t(li)];
                    VLayer &g = vlayers_[size_t(li)];
                    for (auto *arr : {&g.n1w, &g.n1b, &g.n2w, &g.n2b, &g.qw, &g.kw, &g.vw,
                                      &g.fc1w, &g.qb, &g.kb, &g.vb, &g.fc1b, &g.ow, &g.fc2w,
                                      &g.ob, &g.fc2b})
                        arr->assign(size_t(ns), nullptr);
                    for (int64_t sh = 0; sh < ns; ++sh)
                    {
                        const int d = int(sh);
                        const size_t k = size_t(sh);
                        g.n1w[k] = up(d, bf(*t.norm1_w), size_t(H));
                        g.n1b[k] = up(d, bf(*t.norm1_b), size_t(H));
                        g.n2w[k] = up(d, bf(*t.norm2_w), size_t(H));
                        g.n2b[k] = up(d, bf(*t.norm2_b), size_t(H));
                        g.qw[k] = up(d, bf(*t.q_w) + sh * Hs * H, size_t(Hs * H));
                        g.kw[k] = up(d, bf(*t.k_w) + sh * Hs * H, size_t(Hs * H));
                        g.vw[k] = up(d, bf(*t.v_w) + sh * Hs * H, size_t(Hs * H));
                        g.qb[k] = up(d, bf(*t.q_b) + sh * Hs, size_t(Hs));
                        g.kb[k] = up(d, bf(*t.k_b) + sh * Hs, size_t(Hs));
                        g.vb[k] = up(d, bf(*t.v_b) + sh * Hs, size_t(Hs));
                        g.fc1w[k] = up(d, bf(*t.fc1_w) + sh * Is * H, size_t(Is * H));
                        g.fc1b[k] = up(d, bf(*t.fc1_b) + sh * Is, size_t(Is));
                        g.ow[k] = up_cols(d, bf(*t.o_w), H, H, sh * Hs, Hs);
                        g.fc2w[k] = up_cols(d, bf(*t.fc2_w), H, I, sh * Is, Is);
                        g.ob[k] = up(d, bf(*t.o_b), size_t(H));
                        g.fc2b[k] = up(d, bf(*t.fc2_b), size_t(H));
                    }
                }
                stage_.clear();
                stage_.shrink_to_fit();

                const int64_t N = vmaxn_, D = vc.head_dim(), mu = vc.merge_unit();
                vbuf_.assign(size_t(ns), {});
                for (int64_t sh = 0; sh < ns; ++sh)
                {
                    const int d = int(sh);
                    VBuf &b = vbuf_[size_t(sh)];
                    b.x = dalloc<float>(d, size_t(N * H));
                    b.h = dalloc<float>(d, size_t(N * H));
                    b.xn = dalloc<float>(d, size_t(N * H));
                    b.mix = dalloc<float>(d, size_t(N * H));
                    b.Q = dalloc<float>(d, size_t(N * Hs));
                    b.K = dalloc<float>(d, size_t(N * Hs));
                    b.V = dalloc<float>(d, size_t(N * Hs));
                    b.O = dalloc<float>(d, size_t(N * Hs));
                    b.F1 = dalloc<float>(d, size_t(N * Is));
                    b.bstage = dalloc<uint16_t>(d, size_t(N * std::max({H, Is, vc.patch_dim(),
                                                                       OH, P})));
                    b.cosv = dalloc<float>(d, size_t(N * D));
                    b.sinv = dalloc<float>(d, size_t(N * D));
                    b.widx = dalloc<int32_t>(d, size_t(N));
                    b.seg = dalloc<int32_t>(d, size_t(N));
                    b.cuw = dalloc<int32_t>(d, size_t(N + 2));
                    b.cuf = dalloc<int32_t>(d, size_t(N + 2));
                    b.segf = dalloc<int32_t>(d, size_t(N));
                    b.posidx = dalloc<int32_t>(d, size_t(N * 4));
                    b.poswgt = dalloc<float>(d, size_t(N * 4));
                    b.shuf = dalloc<int32_t>(d, size_t(N * mu));
                    b.pix = dalloc<uint16_t>(d, size_t(N * vc.patch_dim()));
                    b.merged = dalloc<float>(d, size_t((N / mu + 1) * OH));
                    b.a1 = dalloc<float>(d, size_t((N / mu + 1) * P));
                    b.a2 = dalloc<float>(d, size_t((N / mu + 1) * P));
                    b.out = dalloc<float>(d, size_t((N / mu + 1) * TH));
                    b.sc = dalloc<float>(d, size_t(N * (vc.num_attention_heads / ns) *
                                                   vseg_cap_));
                    shards_[size_t(sh)].q.wait();
                }
                if (opt_.verbose)
                    std::fprintf(stderr, "[gpu] vision tower: %lld layers, up to %lld patches\n",
                                 (long long)vc.num_hidden_layers, (long long)N);
            }

            std::vector<float> vision_features(const double *pixels,
                                               const std::vector<vision::Grid> &grids) override
            {
                const vision::Config &vc = vcfg_;
                const int64_t ns = nshard_, H = vc.hidden_size, I = vc.intermediate_size;
                const int64_t Hs = H / ns, Is = I / ns, D = vc.head_dim();
                const int64_t nh = vc.num_attention_heads, nhs = nh / ns, mu = vc.merge_unit();
                const int64_t P = vc.projector_hidden_size, OH = vc.out_hidden_size;
                const int64_t TH = cfg_->hidden_size;
                if (vlayers_.empty())
                    die("vision_features() before bind_vision()");
                int64_t N = 0;
                for (const auto &g : grids)
                    N += g.tokens();
                if (N > vmaxn_)
                    die("image needs " + std::to_string(N) + " patches, tower was bound for " +
                        std::to_string(vmaxn_));

                // ---- host-side index work, from the gated CPU implementation
                std::vector<int64_t> widx, wcu;
                vision::window_index(vc, grids, widx, wcu);
                const std::vector<int64_t> fcu = vision::full_cu_seqlens(grids);
                std::vector<std::array<int64_t, 4>> pidx;
                std::vector<std::array<double, 4>> pwgt;
                vision::pos_taps(vc, grids, pidx, pwgt);
                std::vector<int64_t> wid, hid;
                vision::position_ids(grids, wid, hid);

                auto seg_of = [&](const std::vector<int64_t> &cu) {
                    std::vector<int32_t> v(static_cast<size_t>(N), 0);
                    for (size_t si = 0; si + 1 < cu.size(); ++si)
                        for (int64_t i = cu[si]; i < cu[si + 1]; ++i)
                            v[size_t(i)] = int32_t(si);
                    return v;
                };
                const std::vector<int32_t> segw = seg_of(wcu), segf = seg_of(fcu);
                int64_t maxseg = 0;
                for (const std::vector<int64_t> *cu : {(const std::vector<int64_t> *)&wcu,
                                                       (const std::vector<int64_t> *)&fcu})
                    for (size_t si = 0; si + 1 < cu->size(); ++si)
                        maxseg = std::max(maxseg, (*cu)[si + 1] - (*cu)[si]);
                if (maxseg > vseg_cap_)
                    die("attention segment of " + std::to_string(maxseg) +
                        " exceeds the tower's scratch (" + std::to_string(vseg_cap_) + ")");

                // 2-D rope, built exactly as the twin builds it
                const int64_t sd = vc.spatial_dim();
                std::vector<float> cosv(static_cast<size_t>(N * D)),
                    sinv(static_cast<size_t>(N * D));
                {
                    std::vector<float> inv32(static_cast<size_t>(sd / 2));
                    for (int64_t j = 0; j < sd / 2; ++j)
                        inv32[size_t(j)] = 1.0f / float(fmath::pow(double(float(vc.rope_theta)),
                                                                   double(float(2 * j) /
                                                                          float(sd))));
                    for (int64_t t = 0; t < N; ++t)
                    {
                        const int64_t src = widx[size_t(t)];
                        const float pw = float(wid[size_t(src)]), ph = float(hid[size_t(src)]);
                        for (int64_t j = 0; j < D; ++j)
                        {
                            const int64_t qd = j / (sd / 2), r = j % (sd / 2);
                            const float pp = (qd == 0 || qd == 2) ? pw : ph;
                            const double ang = double(pp * inv32[size_t(r)]);
                            cosv[size_t(t * D + j)] = bf16_rt(float(fmath::cos(ang)));
                            sinv[size_t(t * D + j)] = bf16_rt(float(fmath::sin(ang)));
                        }
                    }
                }

                // pixel-shuffle source map and the window permutation
                std::vector<int32_t> shuf;
                {
                    int64_t in_off = 0;
                    for (const auto &g : grids)
                        for (int64_t f = 0; f < g.t; ++f)
                            for (int64_t by = 0; by < g.h / vc.merge_size; ++by)
                                for (int64_t bx = 0; bx < g.w / vc.merge_size; ++bx)
                                    for (int64_t k = 0; k < mu; ++k)
                                    {
                                        const int64_t iy = by * vc.merge_size + k / vc.merge_size;
                                        const int64_t ix = bx * vc.merge_size + k % vc.merge_size;
                                        shuf.push_back(int32_t(in_off + f * g.h * g.w +
                                                               iy * g.w + ix));
                                    }
                    for (const auto &g : grids)
                        in_off += g.tokens();
                }
                const int64_t M = int64_t(shuf.size()) / mu;

                std::vector<int32_t> w32(widx.begin(), widx.end());
                std::vector<int32_t> cuw32(wcu.begin(), wcu.end()),
                    cuf32(fcu.begin(), fcu.end());
                std::vector<int32_t> pi(static_cast<size_t>(N * 4));
                std::vector<float> pwf(static_cast<size_t>(N * 4));
                for (int64_t t = 0; t < N; ++t)
                    for (int k = 0; k < 4; ++k)
                    {
                        pi[size_t(t * 4 + k)] = int32_t(pidx[size_t(t)][size_t(k)]);
                        pwf[size_t(t * 4 + k)] = float(pwgt[size_t(t)][size_t(k)]);
                    }
                std::vector<uint16_t> pxb(static_cast<size_t>(N * vc.patch_dim()));
                for (size_t i = 0; i < pxb.size(); ++i)
                    pxb[i] = muse::bf16::f32_to_bf16(float(pixels[i]));

                for (int64_t sh = 0; sh < ns; ++sh)
                {
                    Dev &d = shards_[size_t(sh)];
                    VBuf &b = vbuf_[size_t(sh)];
                    d.q.memcpy(b.pix, pxb.data(), pxb.size() * 2);
                    d.q.memcpy(b.widx, w32.data(), w32.size() * 4);
                    d.q.memcpy(b.cuw, cuw32.data(), cuw32.size() * 4);
                    d.q.memcpy(b.cuf, cuf32.data(), cuf32.size() * 4);
                    d.q.memcpy(b.seg, segw.data(), segw.size() * 4);
                    d.q.memcpy(b.segf, segf.data(), segf.size() * 4);
                    d.q.memcpy(b.posidx, pi.data(), pi.size() * 4);
                    d.q.memcpy(b.poswgt, pwf.data(), pwf.size() * 4);
                    d.q.memcpy(b.cosv, cosv.data(), cosv.size() * 4);
                    d.q.memcpy(b.sinv, sinv.data(), sinv.size() * 4);
                    d.q.memcpy(b.shuf, shuf.data(), shuf.size() * 4);
                    d.q.wait();

                    // patch embed (no bias) + bilinear position table, ln_pre,
                    // then the window permutation
                    gemm_rm(int(sh), vpatch_[size_t(sh)], b.pix, b.x, N, vc.patch_dim(), H);
                    k_pos_add(d.q, b.x, vpos_[size_t(sh)], b.posidx, b.poswgt, N, H);
                    k_layernorm(d.q, b.x, vlnpre_w_[size_t(sh)], vlnpre_b_[size_t(sh)], b.x, N,
                                H, vc.layer_norm_eps);
                    k_gather_rows(d.q, b.x, b.h, b.widx, N, H);
                }

                const float scaling = float(1.0 / std::sqrt(double(D)));
                for (int64_t li = 0; li < vc.num_hidden_layers; ++li)
                {
                    const VLayer &l = vlayers_[size_t(li)];
                    const bool win = vc.layer_is_window(li);
                    for (int64_t sh = 0; sh < ns; ++sh)
                    {
                        Dev &d = shards_[size_t(sh)];
                        VBuf &b = vbuf_[size_t(sh)];
                        k_layernorm(d.q, b.h, l.n1w[size_t(sh)], l.n1b[size_t(sh)], b.xn, N, H,
                                    vc.layer_norm_eps);
                        k_round(d.q, b.xn, b.bstage, N * H);
                        gemm_rm(int(sh), l.qw[size_t(sh)], b.bstage, b.Q, N, H, Hs);
                        gemm_rm(int(sh), l.kw[size_t(sh)], b.bstage, b.K, N, H, Hs);
                        gemm_rm(int(sh), l.vw[size_t(sh)], b.bstage, b.V, N, H, Hs);
                        k_bias(d.q, b.Q, l.qb[size_t(sh)], N, Hs);
                        k_bias(d.q, b.K, l.kb[size_t(sh)], N, Hs);
                        k_bias(d.q, b.V, l.vb[size_t(sh)], N, Hs);
                        k_vrope(d.q, b.Q, b.K, b.cosv, b.sinv, N, nhs, D, Hs);
                        k_vattn(d.q, b.Q, b.K, b.V, b.O, b.sc, win ? b.seg : b.segf,
                                win ? b.cuw : b.cuf, N, nhs, D, Hs, vseg_cap_, scaling);
                        k_round(d.q, b.O, b.bstage, N * Hs);
                        gemm_rm(int(sh), l.ow[size_t(sh)], b.bstage, b.mix, N, Hs, H);
                    }
                    all_reduce_buf(N * H, [&](int sh) { return vbuf_[size_t(sh)].mix; });
                    for (int64_t sh = 0; sh < ns; ++sh)
                    {
                        Dev &d = shards_[size_t(sh)];
                        VBuf &b = vbuf_[size_t(sh)];
                        // bias AFTER the all-reduce: per shard it would land
                        // `nshard` times
                        k_bias(d.q, b.mix, l.ob[size_t(sh)], N, H);
                        k_dresid(d.q, b.h, b.mix, N * H);
                        k_layernorm(d.q, b.h, l.n2w[size_t(sh)], l.n2b[size_t(sh)], b.xn, N, H,
                                    vc.layer_norm_eps);
                        k_round(d.q, b.xn, b.bstage, N * H);
                        gemm_rm(int(sh), l.fc1w[size_t(sh)], b.bstage, b.F1, N, H, Is);
                        k_bias(d.q, b.F1, l.fc1b[size_t(sh)], N, Is);
                        k_gelu(d.q, b.F1, N * Is);
                        k_round(d.q, b.F1, b.bstage, N * Is);
                        gemm_rm(int(sh), l.fc2w[size_t(sh)], b.bstage, b.mix, N, Is, H);
                    }
                    all_reduce_buf(N * H, [&](int sh) { return vbuf_[size_t(sh)].mix; });
                    for (int64_t sh = 0; sh < ns; ++sh)
                    {
                        Dev &d = shards_[size_t(sh)];
                        VBuf &b = vbuf_[size_t(sh)];
                        k_bias(d.q, b.mix, l.fc2b[size_t(sh)], N, H);
                        k_dresid(d.q, b.h, b.mix, N * H);
                    }
                }

                // undo the window permutation, ln_post, pixel shuffle, adapter,
                // projection, then the SAME weight-less RMSNorm the text
                // embedding uses -- which is what puts both on one scale
                std::vector<float> feats(static_cast<size_t>(M * TH));
                for (int64_t sh = 0; sh < ns; ++sh)
                {
                    Dev &d = shards_[size_t(sh)];
                    VBuf &b = vbuf_[size_t(sh)];
                    k_scatter_rows(d.q, b.h, b.x, b.widx, N, H);
                    k_layernorm(d.q, b.x, vlnpost_w_[size_t(sh)], vlnpost_b_[size_t(sh)], b.x, N,
                                H, vc.layer_norm_eps);
                    k_pixel_shuffle(d.q, b.x, b.merged, b.shuf, M, H, mu);
                    k_round(d.q, b.merged, b.bstage, M * OH);
                    gemm_rm(int(sh), vad1_[size_t(sh)], b.bstage, b.a1, M, OH, P);
                    k_gelu(d.q, b.a1, M * P);
                    k_round(d.q, b.a1, b.bstage, M * P);
                    gemm_rm(int(sh), vad2_[size_t(sh)], b.bstage, b.a2, M, P, P);
                    k_gelu(d.q, b.a2, M * P);
                    k_round(d.q, b.a2, b.bstage, M * P);
                    gemm_rm(int(sh), vproj_[size_t(sh)], b.bstage, b.out, M, P, TH);
                    k_dnorm_wl(d.q, b.out, b.out, M, TH, cfg_->rms_norm_eps);
                }
                shards_[0].q.memcpy(feats.data(), vbuf_[0].out, feats.size() * 4).wait();
                return feats;
            }

            // features are scattered AFTER the embedding norm, which is where
            // the reference puts them
            void set_vision_embeds(const std::vector<float> &feats,
                                   const std::vector<int64_t> &positions) override
            {
                vfeat_ = feats;
                vpos_at_ = positions;
            }

            // ============================================================ DFlash
            //
            // The block-diffusion drafter. One forward per round over
            // `block_size` rows -- an anchor plus block_size-1 MASK tokens --
            // attending BIDIRECTIONALLY within a sliding window, against a
            // context projected from the target's hidden states at
            // `target_layer_ids`. A round proposes block_size-1 tokens (row 0
            // is the anchor's own position and is dropped) through the target's
            // BARE lm_head: no output multiplier, no softcap.
            //
            // Activations here are row-major [rows, dim], not the target's
            // dim-major: the block is 16 rows and the shapes are small, so the
            // layout that reads like the reference is worth more than the one
            // that suits a 2048-token GEMM.
            void bind_drafter(const dflash::Config &dc, const dflash::Weights &dw) override
            {
                dcfg_ = dc;
                const int64_t ns = nshard_, H = dc.hidden_size, I = dc.intermediate_size;
                const int64_t QD = dc.q_dim(), KD = dc.kv_dim(), D = dc.head_dim;
                const int64_t taps = int64_t(dc.target_layer_ids.size());
                if (H != cfg_->hidden_size)
                    die("drafter/target hidden_size mismatch");
                if (dc.num_attention_heads % ns || I % ns)
                    die("drafter does not split " + std::to_string(ns) + " ways");
                const int64_t QDs = QD / ns, Is = I / ns, Hs = H / ns;
                auto bf = [](const st::Tensor &t) { return muse::bf16::as_bf16(t); };

                // encoder.fc is [H, taps*H]: instead of concatenating the taps
                // into an [n, taps*H] buffer (which is n*taps*H floats and
                // blows up with context), keep it as `taps` separate [H, H]
                // blocks and accumulate. Each block is col-sharded on its input
                // so the shards produce partial ctx and all-reduce once.
                enc_.assign(size_t(ns), {});
                for (int64_t sh = 0; sh < ns; ++sh)
                {
                    enc_[size_t(sh)].resize(size_t(taps));
                    for (int64_t k = 0; k < taps; ++k)
                        enc_[size_t(sh)][size_t(k)] =
                            up_cols(int(sh), bf(*dw.enc_fc), H, taps * H, k * H + sh * Hs, Hs);
                }
                enc_norm_.assign(size_t(ns), nullptr);
                dfinal_norm_.assign(size_t(ns), nullptr);
                for (int64_t sh = 0; sh < ns; ++sh)
                {
                    enc_norm_[size_t(sh)] = up(int(sh), bf(*dw.enc_norm), size_t(H));
                    dfinal_norm_[size_t(sh)] = up(int(sh), bf(*dw.final_norm), size_t(H));
                }

                dlayers_.resize(size_t(dc.num_hidden_layers));
                for (int64_t li = 0; li < dc.num_hidden_layers; ++li)
                {
                    const auto &t = dw.layers[size_t(li)];
                    DLayer &g = dlayers_[size_t(li)];
                    for (auto *arr : {&g.input_ln, &g.post_attn_ln, &g.q_norm, &g.k_norm, &g.k,
                                      &g.v, &g.q, &g.mlp_gate, &g.mlp_up, &g.o, &g.mlp_down})
                        arr->assign(size_t(ns), nullptr);
                    for (int64_t sh = 0; sh < ns; ++sh)
                    {
                        const int d = int(sh);
                        const size_t k = size_t(sh);
                        g.input_ln[k] = up(d, bf(*t.input_ln), size_t(H));
                        g.post_attn_ln[k] = up(d, bf(*t.post_attn_ln), size_t(H));
                        g.q_norm[k] = up(d, bf(*t.q_norm), size_t(D));
                        g.k_norm[k] = up(d, bf(*t.k_norm), size_t(D));
                        g.k[k] = up(d, bf(*t.k_proj), size_t(KD * H));
                        g.v[k] = up(d, bf(*t.v_proj), size_t(KD * H));
                        g.q[k] = up(d, bf(*t.q_proj) + sh * QDs * H, size_t(QDs * H));
                        g.mlp_gate[k] = up(d, bf(*t.mlp_gate) + sh * Is * H, size_t(Is * H));
                        g.mlp_up[k] = up(d, bf(*t.mlp_up) + sh * Is * H, size_t(Is * H));
                        g.o[k] = up_cols(d, bf(*t.o_proj), H, QD, sh * QDs, QDs);
                        g.mlp_down[k] = up_cols(d, bf(*t.mlp_down), H, I, sh * Is, Is);
                    }
                }
                stage_.clear();
                stage_.shrink_to_fit();

                // rope over the whole [context ++ block] range, built by the
                // oracle's own table builder so drafter and target cannot drift
                muse::Config rc;
                rc.head_dim = dc.head_dim;
                rc.rope_theta = dc.rope_theta;
                rc.max_position_embeddings = dc.max_position_embeddings;
                muse::RopeTable rt = muse::build_rope_table(rc, max_seq_, false, prec::Dtype::BF16);
                std::vector<float> c32(rt.cos.size()), s32(rt.sin.size());
                for (size_t i = 0; i < rt.cos.size(); ++i)
                {
                    c32[i] = float(rt.cos[i]);
                    s32[i] = float(rt.sin[i]);
                }
                const int64_t B = dc.block_size, S = max_seq_ + B;
                drope_cos_.assign(size_t(ns), nullptr);
                drope_sin_.assign(size_t(ns), nullptr);
                dbuf_.assign(size_t(ns), {});
                for (int64_t sh = 0; sh < ns; ++sh)
                {
                    const int d = int(sh);
                    drope_cos_[size_t(sh)] = dalloc<float>(d, c32.size());
                    drope_sin_[size_t(sh)] = dalloc<float>(d, s32.size());
                    shards_[size_t(sh)].q.memcpy(drope_cos_[size_t(sh)], c32.data(),
                                                 c32.size() * 4);
                    shards_[size_t(sh)].q.memcpy(drope_sin_[size_t(sh)], s32.data(),
                                                 s32.size() * 4);
                    DBuf &b = dbuf_[size_t(sh)];
                    b.ctx = dalloc<float>(d, size_t(max_seq_ * H));
                    b.ctxb = dalloc<uint16_t>(d, size_t(max_seq_ * H));
                    b.h = dalloc<float>(d, size_t(B * H));
                    b.xn = dalloc<float>(d, size_t(B * H));
                    b.kvin = dalloc<uint16_t>(d, size_t(S * H));
                    b.Q = dalloc<float>(d, size_t(B * QDs));
                    b.Qb = dalloc<uint16_t>(d, size_t(B * QDs));
                    b.K = dalloc<float>(d, size_t(S * KD));
                    b.V = dalloc<float>(d, size_t(S * KD));
                    b.O = dalloc<float>(d, size_t(B * QDs));
                    b.Ob = dalloc<uint16_t>(d, size_t(B * QDs));
                    b.mix = dalloc<float>(d, size_t(B * H));
                    b.G = dalloc<float>(d, size_t(B * Is));
                    b.U = dalloc<float>(d, size_t(B * Is));
                    b.Gb = dalloc<uint16_t>(d, size_t(B * Is));
                    b.sc = dalloc<float>(d, size_t(B * (QD / ns / D) * S));
                    b.logits = dalloc<float>(d, size_t((B - 1) * (cfg_->vocab_size / ns)));
                    shards_[size_t(sh)].q.wait();
                }
                if (opt_.verbose)
                    std::fprintf(stderr,
                                 "[gpu] drafter: %lld layers, block %lld (=> %lld/round), "
                                 "%.2f GiB\n",
                                 (long long)dc.num_hidden_layers, (long long)B,
                                 (long long)(B - 1), double(wbytes_ - tbytes_) / 1073741824.0);
            }

            DraftResult draft(int64_t n, int64_t anchor, int64_t pos0) override
            {
                const dflash::Config &dc = dcfg_;
                const Config &tc = *cfg_;
                const int64_t ns = nshard_, H = dc.hidden_size, I = dc.intermediate_size;
                const int64_t D = dc.head_dim, nq = dc.num_attention_heads;
                const int64_t nkv = dc.num_key_value_heads, KD = dc.kv_dim();
                const int64_t B = dc.block_size, S = n + B, V = tc.vocab_size;
                const int64_t QDs = dc.q_dim() / ns, Is = I / ns, Hs = H / ns, Vs = V / ns;
                const int64_t nqs = nq / ns, taps = int64_t(dc.target_layer_ids.size());
                const float scaling = float(1.0 / std::sqrt(double(D)));
                if (dlayers_.empty())
                    die("draft() before bind_drafter()");

                // ---- context: ctx[n,H] = sum_k tap_k[n,H] . Wk^T, each shard
                //      owning a slice of the input, then one all-reduce.
                for (int64_t sh = 0; sh < ns; ++sh)
                {
                    Dev &d = shards_[size_t(sh)];
                    DBuf &b = dbuf_[size_t(sh)];
                    for (int64_t k = 0; k < taps; ++k)
                        bmm(int(sh), d.taps[size_t(k)] + sh * Hs * n, dnnl::memory::data_type::f32,
                            {1, n, Hs}, {n * Hs, 1, n}, enc_[size_t(sh)][size_t(k)],
                            dnnl::memory::data_type::bf16, {1, Hs, H}, {Hs * H, 1, Hs}, b.ctx,
                            {1, n, H}, {n * H, H, 1}, k > 0);
                }
                all_reduce_buf(n * H, [&](int sh) { return dbuf_[size_t(sh)].ctx; });
                for (int64_t sh = 0; sh < ns; ++sh)
                {
                    Dev &d = shards_[size_t(sh)];
                    DBuf &b = dbuf_[size_t(sh)];
                    k_dnorm(d.q, b.ctx, enc_norm_[size_t(sh)], b.ctx, n, H, H, H,
                            dc.rms_norm_eps);
                    // the block: [anchor, MASK x (B-1)], embedded with the
                    // target's RAW table (not the normed-embedding forward)
                    std::vector<int32_t> ids(static_cast<size_t>(B));
                    ids[0] = int32_t(anchor);
                    for (int64_t i = 1; i < B; ++i)
                        ids[size_t(i)] = int32_t(dc.mask_token_id);
                    d.q.memcpy(d.ids, ids.data(), size_t(B) * 4).wait();
                    k_embed_rm(d.q, embed_[size_t(sh)], d.ids, b.h, B, H);
                }

                for (int64_t li = 0; li < dc.num_hidden_layers; ++li)
                {
                    const DLayer &l = dlayers_[size_t(li)];
                    const int64_t window = dc.layer_is_sliding(li) ? dc.sliding_window : 0;
                    for (int64_t sh = 0; sh < ns; ++sh)
                    {
                        Dev &d = shards_[size_t(sh)];
                        DBuf &b = dbuf_[size_t(sh)];
                        k_dnorm(d.q, b.h, l.input_ln[size_t(sh)], b.xn, B, H, H, H,
                                dc.rms_norm_eps);
                        // k/v see [projected context ++ normed block]; q sees
                        // the block only, and the context does NOT go through
                        // input_layernorm
                        k_round(d.q, b.ctx, b.kvin, n * H);
                        k_round(d.q, b.xn, b.kvin + n * H, B * H);
                        k_round(d.q, b.xn, b.ctxb, B * H); // q input
                        gemm_rm(int(sh), l.q[size_t(sh)], b.ctxb, b.Q, B, H, QDs);
                        gemm_rm(int(sh), l.k[size_t(sh)], b.kvin, b.K, S, H, KD);
                        gemm_rm(int(sh), l.v[size_t(sh)], b.kvin, b.V, S, H, KD);
                        k_dnorm(d.q, b.Q, l.q_norm[size_t(sh)], b.Q, B * nqs, D, D, D,
                                dc.rms_norm_eps);
                        k_dnorm(d.q, b.K, l.k_norm[size_t(sh)], b.K, S * nkv, D, D, D,
                                dc.rms_norm_eps);
                        k_drope(d.q, b.Q, B, nqs, D, drope_cos_[size_t(sh)],
                                drope_sin_[size_t(sh)], pos0);
                        k_drope(d.q, b.K, S, nkv, D, drope_cos_[size_t(sh)],
                                drope_sin_[size_t(sh)], 0);
                        k_dattn(d.q, b.Q, b.K, b.V, b.O, b.sc, B, S, nqs, nkv, D, pos0, window,
                                scaling, sh * nqs, nq / nkv);
                        k_round(d.q, b.O, b.Ob, B * QDs);
                        gemm_rm(int(sh), l.o[size_t(sh)], b.Ob, b.mix, B, QDs, H);
                    }
                    all_reduce_buf(B * H, [&](int sh) { return dbuf_[size_t(sh)].mix; });
                    for (int64_t sh = 0; sh < ns; ++sh)
                    {
                        Dev &d = shards_[size_t(sh)];
                        DBuf &b = dbuf_[size_t(sh)];
                        k_dresid(d.q, b.h, b.mix, B * H);
                        k_dnorm(d.q, b.h, l.post_attn_ln[size_t(sh)], b.xn, B, H, H, H,
                                dc.rms_norm_eps);
                        k_round(d.q, b.xn, b.ctxb, B * H);
                        gemm_rm(int(sh), l.mlp_gate[size_t(sh)], b.ctxb, b.G, B, H, Is);
                        gemm_rm(int(sh), l.mlp_up[size_t(sh)], b.ctxb, b.U, B, H, Is);
                        k_swiglu_flat(d.q, b.G, b.U, b.Gb, B * Is);
                        gemm_rm(int(sh), l.mlp_down[size_t(sh)], b.Gb, b.mix, B, Is, H);
                    }
                    all_reduce_buf(B * H, [&](int sh) { return dbuf_[size_t(sh)].mix; });
                    for (int64_t sh = 0; sh < ns; ++sh)
                        k_dresid(shards_[size_t(sh)].q, dbuf_[size_t(sh)].h,
                                 dbuf_[size_t(sh)].mix, B * H);
                }

                // final norm, then the target's BARE lm_head over rows 1..B-1
                std::vector<float> lg(static_cast<size_t>((B - 1) * V));
                for (int64_t sh = 0; sh < ns; ++sh)
                {
                    Dev &d = shards_[size_t(sh)];
                    DBuf &b = dbuf_[size_t(sh)];
                    k_dnorm(d.q, b.h, dfinal_norm_[size_t(sh)], b.h, B, H, H, H,
                            dc.rms_norm_eps);
                    k_round(d.q, b.h + H, b.ctxb, (B - 1) * H);
                    gemm_rm(int(sh), lm_head_[size_t(sh)], b.ctxb, b.logits, B - 1, H, Vs);
                }
                for (int64_t sh = 0; sh < ns; ++sh)
                {
                    std::vector<float> part(static_cast<size_t>((B - 1) * Vs));
                    shards_[size_t(sh)]
                        .q.memcpy(part.data(), dbuf_[size_t(sh)].logits, part.size() * 4)
                        .wait();
                    for (int64_t r = 0; r < B - 1; ++r)
                        std::memcpy(&lg[size_t(r * V + sh * Vs)], &part[size_t(r * Vs)],
                                    size_t(Vs) * 4);
                }
                DraftResult out;
                out.anchor = anchor;
                out.tokens.resize(size_t(B - 1));
                for (int64_t r = 0; r < B - 1; ++r)
                {
                    const float *row = &lg[size_t(r * V)];
                    int64_t best = 0;
                    for (int64_t v = 1; v < V; ++v)
                        if (row[v] > row[best])
                            best = v;
                    out.tokens[size_t(r)] = best;
                }
                return out;
            }

            // -------------------------------------------------------- forward

            // `block_ids` points at this block's tokens, not at the whole
            // sequence: decode has only the one token it is appending, and
            // indexing a 1-element vector at `pos0` reads out of bounds and
            // feeds a garbage id into the embedding gather (which hangs the
            // card rather than failing).
            void forward_block(const int64_t *block_ids, int64_t pos0, int64_t n,
                               float *out_logits_last)
            {
                const Config &c = *cfg_;
                const int64_t H = c.hidden_size, I = c.intermediate_size, D = c.head_dim;
                const int64_t nq = c.num_attention_heads, nkv = c.num_key_value_heads;
                const int64_t QD = c.q_dim(), KD = c.kv_dim(), groups = c.kv_groups();
                const int64_t V = c.vocab_size, ns = nshard_;
                const int64_t QDs = QD / ns, Is = I / ns, Vs = V / ns, nqs = nq / ns;
                const float scaling = float(1.0 / std::sqrt(double(D)));
                const float qk_scale = float(c.qk_scale_factor);
                // Leading dimension of every activation buffer for THIS block.
                // A single live column needs no stride at all, and giving it
                // one is expensive: oneDNN then writes each of the ~20k outputs
                // 512 bytes apart, and every elementwise kernel and norm reads
                // scattered. Collapsing to 1 at decode makes the whole step
                // contiguous; the buffers are allocated [dim, block] so a
                // 1-column view always fits. Prefill (n > 1) is unchanged.
                const int64_t B = (n == 1) ? 1 : block_;

                std::vector<int32_t> h_ids(static_cast<size_t>(n));
                for (int64_t t = 0; t < n; ++t)
                    h_ids[size_t(t)] = int32_t(block_ids[t]);

                // Every shard builds the whole residual stream itself. That is
                // what the replicated embedding table buys: no broadcast, and
                // every replicated op downstream (the norms) is computed from
                // identical inputs and so stays identical by construction.
                for (int64_t sh = 0; sh < ns; ++sh)
                {
                    Dev &d = shards_[size_t(sh)];
                    d.q.memcpy(d.ids, h_ids.data(), size_t(n) * 4).wait();
                    k_embed(d.q, embed_[size_t(sh)], d.ids, d.h, n, H, B);
                    k_rmsnorm_f32(d.q, d.h, d.h, n, H, B, c.rms_norm_eps);
                }
                // vision features replace the embedding at the image/video
                // placeholder positions, AFTER the norm
                if (!vfeat_.empty())
                {
                    const int64_t TH = H;
                    std::vector<float> col(static_cast<size_t>(TH));
                    for (size_t k = 0; k < vpos_at_.size(); ++k)
                    {
                        const int64_t p = vpos_at_[k] - pos0;
                        if (p < 0 || p >= n)
                            continue;
                        for (int64_t i = 0; i < TH; ++i)
                            col[size_t(i)] = vfeat_[k * size_t(TH) + size_t(i)];
                        for (int64_t sh = 0; sh < ns; ++sh)
                            shards_[size_t(sh)]
                                .q.ext_oneapi_memcpy2d(shards_[size_t(sh)].h + p, size_t(B) * 4,
                                                       col.data(), 4, 4, size_t(TH))
                                .wait();
                    }
                }
                trace(0, "embed", n, B);

                for (int64_t li = 0; li < c.num_hidden_layers; ++li)
                {
                    const GpuLayer &l = layers_[size_t(li)];
                    const bool sliding = c.layer_is_sliding(li);
                    const bool use_rope = c.layer_has_rope(li);
                    const int64_t window = sliding ? c.sliding_window : 0;

                    // Shards run concurrently: the queues are independent and
                    // only the two all-reduces below synchronize them.
                    tic();
                    for (int64_t sh = 0; sh < ns; ++sh)
                    {
                        Dev &d = shards_[size_t(sh)];
                        k_rmsnorm_auto(d.q, d.h, l.input_ln[size_t(sh)], d.xb, n, 1, H, B,
                                       c.rms_norm_eps, NK::Centered);
                    }
                    toc(S_NORM);

                    tic();
                    for (int64_t sh = 0; sh < ns; ++sh)
                    {
                        Dev &d = shards_[size_t(sh)];
                        gemm(int(sh), l.q[size_t(sh)], d.xb, d.qf, n, H, QDs, B, B);
                        gemm(int(sh), l.k_[size_t(sh)], d.xb, d.kf, n, H, KD, B, B);
                        gemm(int(sh), l.v_[size_t(sh)], d.xb, d.vf, n, H, KD, B, B);
                        gemm(int(sh), l.gate[size_t(sh)], d.xb, d.gf, n, H, QDs, B, B);
                    }
                    toc(S_QKV);

                    tic();
                    for (int64_t sh = 0; sh < ns; ++sh)
                    {
                        Dev &d = shards_[size_t(sh)];
                        // weight-less QK-norm over head_dim, then qk_scale on q
                        k_rmsnorm_auto(d.q, d.qf, nullptr, d.qb, n, nqs, D, B, c.rms_norm_eps,
                                       NK::Weightless);
                        k_rmsnorm_auto(d.q, d.kf, nullptr, d.kb, n, nkv, D, B, c.rms_norm_eps,
                                       NK::Weightless);
                        k_scale_bf16(d.q, d.qb, QDs, n, B, qk_scale);
                        if (use_rope)
                            k_rope(d.q, d.qb, d.kb, n, nqs, nkv, D, B, d.rope_cos, d.rope_sin,
                                   pos0);
                        k_kv_append(d.q, d.kb, d.vf, l.kc[size_t(sh)], l.vc[size_t(sh)], n, KD, B,
                                    pos0, l.cap);
                    }
                    toc(S_NORM);

                    tic();
                    for (int64_t sh = 0; sh < ns; ++sh)
                    {
                        Dev &d = shards_[size_t(sh)];
                        // shard sh owns q heads [sh*nqs, (sh+1)*nqs); the KV
                        // cache is replicated, so the kernel needs the global
                        // head offset to pick the right GQA group
#if ORACLE_GPU_DNNL
                        if (n > 1 && flash_prefill_)
                            attention_flash(int(sh), l, n, pos0, nqs, D, KD, B, groups, l.cap,
                                            window, scaling, sh * nqs);
                        else
#endif
                            if (n > 1 && tiled_attn_)
                            k_attention_tiled(d.q, d.qb, l.kc[size_t(sh)], l.vc[size_t(sh)], d.of,
                                              n, nqs, D, KD, B, pos0, groups, l.cap, window,
                                              scaling, sh * nqs, attn_bq_, attn_bk_);
                        else
                            k_attention(d.q, d.qb, l.kc[size_t(sh)], l.vc[size_t(sh)], d.of, n,
                                        nqs, D, KD, B, pos0, groups, l.cap, window, scaling,
                                        sh * nqs);
                    }
                    toc(S_ATTN);

                    tic();
                    for (int64_t sh = 0; sh < ns; ++sh)
                    {
                        Dev &d = shards_[size_t(sh)];
                        k_out_gate(d.q, d.of, d.gf, d.ob, QDs, n, B);
                        // column-sharded: each shard sums over its own QDs
                        // slice, so xf is a PARTIAL until the all-reduce
                        gemm(int(sh), l.o[size_t(sh)], d.ob, d.xf, n, QDs, H, B, B);
                    }
                    toc(S_OPROJ);
                    tic();
                    all_reduce(n, B);
                    toc(S_XFER);
                    tic();
                    for (int64_t sh = 0; sh < ns; ++sh)
                    {
                        Dev &d = shards_[size_t(sh)];
                        k_rmsnorm_auto(d.q, d.xf, l.post_attn_ln[size_t(sh)], d.xb, n, 1, H, B,
                                       c.post_norm_eps, NK::Centered);
                        k_residual(d.q, d.h, d.xb, H, n, B);
                    }
                    toc(S_OPROJ);

                    tic();
                    for (int64_t sh = 0; sh < ns; ++sh)
                    {
                        Dev &d = shards_[size_t(sh)];
                        k_rmsnorm_auto(d.q, d.h, l.pre_ff_ln[size_t(sh)], d.xb, n, 1, H, B,
                                       c.rms_norm_eps, NK::Centered);
                        gemm(int(sh), l.mlp_gate[size_t(sh)], d.xb, d.g1, n, H, Is, B, B);
                        gemm(int(sh), l.mlp_up[size_t(sh)], d.xb, d.u1, n, H, Is, B, B);
                        k_swiglu(d.q, d.g1, d.u1, d.g1b, Is, n, B);
                        gemm(int(sh), l.mlp_down[size_t(sh)], d.g1b, d.xf, n, Is, H, B, B);
                    }
                    toc(S_MLP);
                    tic();
                    all_reduce(n, B);
                    toc(S_XFER);
                    tic();
                    for (int64_t sh = 0; sh < ns; ++sh)
                    {
                        Dev &d = shards_[size_t(sh)];
                        k_rmsnorm_auto(d.q, d.xf, l.post_ff_ln[size_t(sh)], d.xb, n, 1, H, B,
                                       c.post_norm_eps, NK::Centered);
                        k_residual(d.q, d.h, d.xb, H, n, B);
                    }
                    toc(S_MLP);
                    // capture AFTER the layer's final residual: the drafter
                    // reads layer outputs, not mid-layer state
                    for (size_t k = 0; k < tap_layers_.size(); ++k)
                        if (tap_layers_[k] == li)
                            for (int64_t sh = 0; sh < ns; ++sh)
                            {
                                Dev &d = shards_[size_t(sh)];
                                k_pack2d(d.q, d.h, d.taps[k], H, n, B);
                            }
                    if (trace_dir_)
                    {
                        char tg[64];
                        std::snprintf(tg, sizeof tg, "layer_%02d", int(li));
                        trace(0, tg, n, B);
                    }
                }
                len_ = pos0 + n;

                if (!out_logits_last)
                {
                    for (int64_t sh = 0; sh < ns; ++sh)
                        shards_[size_t(sh)].q.wait();
                    return;
                }

                // Vocab-parallel head, last row only (the serving shape). The
                // multiplier and the softcap are elementwise, so they commute
                // with the split and each shard finishes its own slice.
                tic();
                for (int64_t sh = 0; sh < ns; ++sh)
                {
                    Dev &d = shards_[size_t(sh)];
                    k_rmsnorm_auto(d.q, d.h + (n - 1), final_norm_[size_t(sh)], d.xb + (n - 1), 1,
                                   1, H, B, c.rms_norm_eps, NK::Plain);
                    gemm(int(sh), lm_head_[size_t(sh)], d.xb + (n - 1), d.logits, 1, H, Vs, B, 1);
                    k_softcap(d.q, d.logits, Vs, float(c.output_multiplier),
                              float(c.final_logit_softcapping));
                }
                for (int64_t sh = 0; sh < ns; ++sh)
                    shards_[size_t(sh)]
                        .q.memcpy(out_logits_last + sh * Vs, shards_[size_t(sh)].logits,
                                  size_t(Vs) * 4)
                        .wait();
                toc(S_HEAD);
            }

            // Time a cross-card copy at a known point in startup. Peer copies
            // measured 48 GB/s standalone and 0.3 GB/s inside this engine; this
            // says whether the collapse happens before or after the weights
            // land, which is the difference between an allocation-pattern
            // problem and a mapping problem.
            void peer_probe(const char *when)
            {
                if (nshard_ < 2 || !getenv("MUSE_GPU_PEER_PROBE"))
                    return;
                // size to the engine's own block so the peer buffers are valid
                const size_t n = size_t(block_) * size_t(cfg_->hidden_size), bytes = n * 4;
                // 2 slots each, so the two directions touch DISJOINT regions —
                // exactly what all_reduce does. An aliased pair is a data race
                // and measures nonsense (or kills the device).
                float *a = sycl::malloc_device<float>(2 * n, shards_[0].q);
                float *b = sycl::malloc_device<float>(2 * n, shards_[1].q);
                if (!a || !b)
                {
                    std::fprintf(stderr, "[peer] %s: alloc failed\n", when);
                    return;
                }
                shards_[0].q.memset(a, 1, 2 * bytes).wait();
                shards_[1].q.memset(b, 2, 2 * bytes).wait();
                const int64_t slot = block_ * cfg_->hidden_size;
                auto bench = [&](const char *what, auto &&fn, double payload) {
                    fn();
                    const auto t0 = std::chrono::steady_clock::now();
                    for (int i = 0; i < 10; ++i)
                        fn();
                    const double sec =
                        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
                            .count() /
                        10;
                    std::fprintf(stderr, "[peer] %-22s %-30s %7.3f ms  %6.1f GB/s\n", when, what,
                                 sec * 1e3, payload / sec / 1e9);
                };
                bench("1 pull, own buffers", [&] { shards_[0].q.memcpy(a, b, bytes).wait(); },
                      double(bytes));
                bench("2 pulls, disjoint regions",
                      [&] {
                          auto e0 = shards_[0].q.memcpy(a, b, bytes);           // dev1[0]  -> dev0[0]
                          auto e1 = shards_[1].q.memcpy(b + n, a + n, bytes);   // dev0[1] -> dev1[1]
                          e0.wait();
                          e1.wait();
                      },
                      2.0 * double(bytes));
                bench("2 pulls, serialized",
                      [&] {
                          shards_[0].q.memcpy(a, b, bytes).wait();
                          shards_[1].q.memcpy(b + n, a + n, bytes).wait();
                      },
                      2.0 * double(bytes));
                // the combination all_reduce actually performs: a kernel writes
                // the source region, then the peer PULLS it
                bench("kernel-write then 2 pulls",
                      [&] {
                          shards_[0].q.parallel_for(sycl::range<1>(n),
                                                    [=](sycl::id<1> i) { a[n + i] = float(i); });
                          shards_[1].q.parallel_for(sycl::range<1>(n),
                                                    [=](sycl::id<1> i) { b[i] = float(i); });
                          shards_[0].q.wait();
                          shards_[1].q.wait();
                          auto e0 = shards_[0].q.memcpy(a, b, bytes);
                          auto e1 = shards_[1].q.memcpy(b + n, a + n, bytes);
                          e0.wait();
                          e1.wait();
                      },
                      2.0 * double(bytes));
                if (shards_[0].peer && shards_[1].peer)
                {
                    bench("1 pull, peer buffers",
                          [&] {
                              shards_[0]
                                  .q.memcpy(shards_[0].peer + slot, shards_[1].peer + slot, bytes)
                                  .wait();
                          },
                          double(bytes));
                    bench("2 pulls, peer buffers (all_reduce)",
                          [&] {
                              auto e0 = shards_[0].q.memcpy(shards_[0].peer + slot,
                                                            shards_[1].peer + slot, bytes);
                              auto e1 = shards_[1].q.memcpy(shards_[1].peer, shards_[0].peer,
                                                            bytes);
                              e0.wait();
                              e1.wait();
                          },
                          2.0 * double(bytes));
                }
                sycl::free(a, shards_[0].q);
                sycl::free(b, shards_[1].q);
                std::fflush(stderr);
            }

            // Prove, on this machine and this oneDNN build, that the GEMM
            // route actually computes the product the engine thinks it does.
            //
            // This is not paranoia: oneDNN 3.11.2 silently returns wrong
            // results for a one-column B with an offset handle (see gemm()),
            // and the failure is invisible — no error, no NaN, just different
            // logits. A library that is wrong for one descriptor family can be
            // wrong for another after an upgrade, so both routes are checked
            // against each other at startup on the shapes this model uses.
            void self_check()
            {
                const Config &c = *cfg_;
                Dev &d = shards_[0];
                const int64_t in = 64, out = 96;
                const int64_t ld = std::max<int64_t>(block_, 8);

                std::vector<uint16_t> hw(size_t(out * in)), hx(size_t(in * ld));
                for (size_t i = 0; i < hw.size(); ++i)
                    hw[i] = f2bf(float(int(i * 37 % 101) - 50) / 50.f);
                for (size_t i = 0; i < hx.size(); ++i)
                    hx[i] = f2bf(float(int(i * 53 % 97) - 48) / 48.f);

                auto *W = sycl::malloc_device<uint16_t>(hw.size(), d.q);
                auto *X = sycl::malloc_device<uint16_t>(hx.size(), d.q);
                auto *Y = sycl::malloc_device<float>(size_t(out * ld), d.q);
                if (!W || !X || !Y)
                    die("self-check allocation failed");
                d.q.memcpy(W, hw.data(), hw.size() * 2).wait();
                d.q.memcpy(X, hx.data(), hx.size() * 2).wait();

                // (n, offset) pairs covering what the forward pass issues:
                // a full prefill block, a short trailing chunk, and the
                // last-token view used by the head and by every decode step.
                const std::pair<int64_t, int64_t> cases[] = {
                    {ld, 0}, {std::max<int64_t>(1, ld / 2), 0}, {1, ld - 1}, {1, 0}};
                for (auto [n, off] : cases)
                {
                    std::vector<float> viaA(size_t(out * ld)), viaB(size_t(out * ld));
                    d.q.memset(Y, 0, size_t(out * ld) * 4).wait();
                    gemm_bf16(0, W, X + off, Y, n, in, out, ld, ld);
                    d.q.wait();
                    d.q.memcpy(viaA.data(), Y, viaA.size() * 4).wait();

                    d.q.memset(Y, 0, size_t(out * ld) * 4).wait();
                    k_gemv(d.q, W, X + off, Y, n, in, out, ld, ld);
                    d.q.wait();
                    d.q.memcpy(viaB.data(), Y, viaB.size() * 4).wait();

                    double maxrel = 0;
                    for (int64_t o = 0; o < out; ++o)
                        for (int64_t t = 0; t < n; ++t)
                        {
                            const double a = viaA[size_t(o * ld + t)],
                                         b = viaB[size_t(o * ld + t)];
                            maxrel = std::max(maxrel, std::fabs(a - b) / (std::fabs(b) + 1e-6));
                        }
                    // both routes accumulate in f32 over `in` terms in
                    // different orders, so this is an envelope, not equality
                    if (!(maxrel < 1e-2))
                        die("GEMM self-check failed at n=" + std::to_string(n) + " offset=" +
                            std::to_string(off) + " (max rel " + std::to_string(maxrel) +
                            "): the oneDNN route disagrees with the SYCL reference. "
                            "Re-run with --no-dnnl to confirm, and do not trust any logits "
                            "from this build until it is resolved.");
                }
                sycl::free(W, d.q);
                sycl::free(X, d.q);
                sycl::free(Y, d.q);
                (void)c;
            }

            // Dump the residual stream after a named stage, row-major [n, H],
            // when MUSE_GPU_TRACE names a directory. This is the same idea as
            // the oracle's --trace-dir: without it, a wrong logit only tells
            // you that something upstream is wrong, not which stage.
            void trace(int dev, const char *tag, int64_t n, int64_t ld)
            {
                if (!trace_dir_ || !*trace_dir_)
                    return;
                const int64_t H = cfg_->hidden_size;
                // same trap as hand_off: the live region is strided, so pull
                // the H x n sub-block rather than the first n*H floats
                std::vector<float> col(static_cast<size_t>(n * H));
                k_pack2d(shards_[size_t(dev)].q, shards_[size_t(dev)].h, shards_[size_t(dev)].xfer, H,
                         n, ld);
                shards_[size_t(dev)].q.memcpy(col.data(), shards_[size_t(dev)].xfer,
                                            size_t(n * H) * 4)
                    .wait();
                std::vector<float> row(static_cast<size_t>(n * H));
                for (int64_t t = 0; t < n; ++t)
                    for (int64_t i = 0; i < H; ++i)
                        row[size_t(t * H + i)] = col[size_t(i * n + t)];
                char path[512];
                std::snprintf(path, sizeof path, "%s/%s.bin", trace_dir_, tag);
                if (FILE *f = std::fopen(path, "wb"))
                {
                    std::fwrite(row.data(), 4, row.size(), f);
                    std::fclose(f);
                }
            }

            // The residual stream crossing a layer-split boundary. Staged
            // through host memory: the Arc driver will silently hand back zeros
            // (or DEVICE_LOST) if a kernel dereferences peer-card USM, so
            // cross-card traffic is always an explicit copy.
            // `n` is the live column count, which is NOT the allocated block:
            // the residual stream is dim-major [H, block], so the live region
            // is H segments of n floats at stride `block`, not the first n*H
            // floats. Treating it as contiguous is right only when n == block
            // (a full prefill chunk) and silently wrong for every decode step —
            // which is exactly the shape a prefill-only gate does not cover.
            void hand_off(int from, int to, int64_t n, int64_t ld)
            {
                const int64_t H = cfg_->hidden_size;
                const size_t bytes = size_t(n * H) * 4;
                if (host_stage_.size() < size_t(n * H))
                    host_stage_.resize(size_t(n * H));
                Dev &src = shards_[size_t(from)];
                Dev &dst = shards_[size_t(to)];
                k_pack2d(src.q, src.h, src.xfer, H, n, ld);
                src.q.memcpy(host_stage_.data(), src.xfer, bytes).wait();
                dst.q.memcpy(dst.xfer, host_stage_.data(), bytes).wait();
                k_unpack2d(dst.q, dst.xfer, dst.h, H, n, ld);
            }

            const Config *cfg_;
            EngineOptions opt_;
            // used_ must be declared before ctx_: the context is built from it
            std::vector<sycl::device> used_;
            sycl::context ctx_;
            int ngpu_ = 1, nshard_ = 1;
            bool decode_gemv_ = false;
            bool p2p_ = true;
            int64_t attn_bq_ = 8, attn_bk_ = 64;
            bool tiled_attn_ = true;
            bool flash_prefill_ = false;
            // ---- DFlash drafter state (empty until bind_drafter) ----
            dflash::Config dcfg_;
            std::vector<DLayer> dlayers_;
            std::vector<std::vector<uint16_t *>> enc_; // [shard][tap] col-sharded fc block
            std::vector<uint16_t *> enc_norm_, dfinal_norm_;
            std::vector<float *> drope_cos_, drope_sin_;
            std::vector<int64_t> tap_layers_;
            bool q8_ = false, q8_blockdot_ = false;
            int64_t deq_cap_ = 0;
            std::vector<int8_t> qbuf_;
            std::vector<uint16_t> dbuf8_;
            int sealed_ = 0; // 0 off, 1 log, 2 refuse
            int64_t tbytes_ = 0; // target weight bytes, so the drafter's can be reported apart
            struct DBuf
            {
                float *ctx = nullptr, *h = nullptr, *xn = nullptr, *Q = nullptr, *K = nullptr;
                float *V = nullptr, *O = nullptr, *mix = nullptr, *G = nullptr, *U = nullptr;
                float *sc = nullptr, *logits = nullptr;
                uint16_t *ctxb = nullptr, *kvin = nullptr, *Qb = nullptr, *Ob = nullptr,
                         *Gb = nullptr;
            };
            std::vector<DBuf> dbuf_;
            // ---- vision tower state (empty until bind_vision) ----
            vision::Config vcfg_;
            std::vector<VLayer> vlayers_;
            std::vector<uint16_t *> vpatch_, vpos_, vlnpre_w_, vlnpre_b_, vlnpost_w_,
                vlnpost_b_, vad1_, vad2_, vproj_;
            int64_t vmaxn_ = 0;
            static constexpr int64_t vseg_cap_ = 4096; // widest attention segment
            struct VBuf
            {
                float *x = nullptr, *h = nullptr, *xn = nullptr, *mix = nullptr, *Q = nullptr;
                float *K = nullptr, *V = nullptr, *O = nullptr, *F1 = nullptr, *sc = nullptr;
                float *cosv = nullptr, *sinv = nullptr, *poswgt = nullptr;
                float *merged = nullptr, *a1 = nullptr, *a2 = nullptr, *out = nullptr;
                uint16_t *bstage = nullptr, *pix = nullptr;
                int32_t *widx = nullptr, *seg = nullptr, *segf = nullptr, *cuw = nullptr;
                int32_t *cuf = nullptr, *posidx = nullptr, *shuf = nullptr;
            };
            std::vector<VBuf> vbuf_;
            std::vector<float> vfeat_;
            std::vector<int64_t> vpos_at_;
            int64_t fbq_ = 512, fbk_ = 512;
#if ORACLE_GPU_DNNL
            std::vector<std::map<std::array<int64_t, 8>, dnnl::matmul>> bprims_;
#endif
            float *host_ring_ = nullptr; // pinned staging, ns * block * H floats
            const char *trace_dir_ = nullptr;
            // Stage attribution. Only armed by MUSE_GPU_PROFILE, because it
            // inserts a queue wait per stage and so changes the very thing it
            // measures; it is for finding where decode's time goes, not for
            // reporting throughput.
            bool prof_on_ = false;
            enum Stage { S_NORM, S_QKV, S_ATTN, S_OPROJ, S_MLP, S_HEAD, S_XFER, S_NSTAGE };
            mutable double stage_s_[S_NSTAGE] = {};
            mutable std::chrono::steady_clock::time_point tic_;
            mutable double ar_pack_s_ = 0, ar_xchg_s_ = 0, ar_sum_s_ = 0, ar_bytes_ = 0;
            mutable long ar_calls_ = 0;
            void tic()
            {
                if (!prof_on_) return;
                for (auto &d : shards_) d.q.wait();
                tic_ = std::chrono::steady_clock::now();
            }
            void toc(Stage st)
            {
                if (!prof_on_) return;
                for (auto &d : shards_) d.q.wait();
                stage_s_[st] +=
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - tic_).count();
            }
            int64_t block_ = 0, max_seq_ = 0, len_ = 0;
            int64_t wbytes_ = 0, kvbytes_ = 0;
            std::vector<Dev> shards_;
            std::vector<GpuLayer> layers_;
            std::vector<uint16_t *> embed_, final_norm_;
            std::vector<QW> lm_head_;
            std::vector<uint16_t> stage_; // host gather buffer for column shards
            muse::RopeTable rope_;
            std::vector<float> host_stage_;
            std::vector<std::pair<void *, int>> owned_;
            Timings tim_;
        };

    } // namespace

    std::vector<DeviceInfo> enumerate_devices()
    {
        std::vector<DeviceInfo> out;
        try
        {
            for (const auto &d : sycl::device::get_devices(sycl::info::device_type::gpu))
            {
                DeviceInfo i;
                i.name = d.get_info<sycl::info::device::name>();
                i.total_mem = int64_t(d.get_info<sycl::info::device::global_mem_size>());
                i.compute_units = int(d.get_info<sycl::info::device::max_compute_units>());
                i.free_mem = i.total_mem;
                out.push_back(i);
            }
        }
        catch (...)
        {
        }
        return out;
    }

    std::unique_ptr<Engine> Engine::create(const Config &c, const Weights &w,
                                           const EngineOptions &opt)
    {
        return std::make_unique<SyclEngine>(c, w, opt);
    }

} // namespace muse::gpu
