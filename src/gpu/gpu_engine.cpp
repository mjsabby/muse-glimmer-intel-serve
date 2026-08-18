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

        constexpr int SG = 32; // sub-group width used by every reduction kernel

        [[noreturn]] void die(const std::string &m) { throw std::runtime_error("gpu: " + m); }

        // --------------------------------------------------------- device side

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
            std::vector<uint16_t *> k, v, kc, vc;                                  // replicated
            std::vector<uint16_t *> q, gate, mlp_gate, mlp_up;                     // row-sharded
            std::vector<uint16_t *> o, mlp_down;                                   // col-sharded
            int64_t cap = 0;
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
            static std::vector<sycl::device> pick_devices(int want)
            {
                auto all = sycl::device::get_devices(sycl::info::device_type::gpu);
                const int n = want > 0 ? std::min<int>(want, int(all.size())) : 1;
                return std::vector<sycl::device>(all.begin(), all.begin() + n);
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
                if (const char *e = getenv("MUSE_GPU_DECODE_GEMV"); e && *e && *e != '0')
                    decode_gemv_ = true;
                trace_dir_ = getenv("MUSE_GPU_TRACE");
                if (const char *e = getenv("MUSE_GPU_P2P"); e && *e && *e != '0')
                    p2p_ = true;
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

                auto t0 = std::chrono::steady_clock::now();
                upload_weights(w);
                alloc_scratch();
                // after alloc_scratch: the check drives the real gemm(), which
                // uses the packing scratch on its n == 1 path
                self_check();
                for (auto &d : shards_)
                    d.q.wait();
                tim_.upload_s =
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

                if (opt.verbose)
                {
                    std::fprintf(stderr,
                                 "[gpu] %d shard(s) on %d card(s), %.2f GiB weights "
                                 "(%.2f/shard), %.2f GiB KV, upload %.1f s\n",
                                 nshard_, ngpu_, double(wbytes_) / 1073741824.0,
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

            template <class T> T *dalloc(int dev, size_t count)
            {
                auto &d = shards_[size_t(dev)];
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

            void upload_weights(const Weights &w)
            {
                const Config &c = *cfg_;
                auto v = muse::bf16::bind(w);
                const int64_t H = c.hidden_size, I = c.intermediate_size, V = c.vocab_size;
                const int64_t QD = c.q_dim(), KD = c.kv_dim();
                const int64_t L = c.num_hidden_layers, ns = nshard_;
                const int64_t QDs = QD / ns, Is = I / ns, Vs = V / ns;

                embed_.assign(size_t(ns), nullptr);
                lm_head_.assign(size_t(ns), nullptr);
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
                    lm_head_[size_t(sh)] = up(int(sh), v.lm_head + sh * Vs * H, size_t(Vs * H));
                }

                layers_.resize(size_t(L));
                for (int64_t li = 0; li < L; ++li)
                {
                    const auto &t = v.layers[size_t(li)];
                    GpuLayer &g = layers_[size_t(li)];
                    g.cap = c.layer_is_sliding(li) ? std::min(c.sliding_window + block_, max_seq_)
                                                   : max_seq_;
                    for (auto *arr : {&g.input_ln, &g.post_attn_ln, &g.pre_ff_ln, &g.post_ff_ln,
                                      &g.k, &g.v, &g.kc, &g.vc, &g.q, &g.gate, &g.mlp_gate,
                                      &g.mlp_up, &g.o, &g.mlp_down})
                        arr->assign(size_t(ns), nullptr);

                    for (int64_t sh = 0; sh < ns; ++sh)
                    {
                        const int d = int(sh);
                        const size_t k = size_t(sh);
                        g.input_ln[k] = up(d, t.input_ln, size_t(H));
                        g.post_attn_ln[k] = up(d, t.post_attn_ln, size_t(H));
                        g.pre_ff_ln[k] = up(d, t.pre_ff_ln, size_t(H));
                        g.post_ff_ln[k] = up(d, t.post_ff_ln, size_t(H));
                        g.k[k] = up(d, t.k, size_t(KD * H));
                        g.v[k] = up(d, t.v, size_t(KD * H));
                        // row shards: contiguous slices of a row-major [out, in]
                        g.q[k] = up(d, t.q + sh * QDs * H, size_t(QDs * H));
                        g.gate[k] = up(d, t.gate + sh * QDs * H, size_t(QDs * H));
                        g.mlp_gate[k] = up(d, t.mlp_gate + sh * Is * H, size_t(Is * H));
                        g.mlp_up[k] = up(d, t.mlp_up + sh * Is * H, size_t(Is * H));
                        // column shards: partial sums, all-reduced after the GEMM
                        g.o[k] = up_cols(d, t.o, H, QD, sh * QDs, QDs);
                        g.mlp_down[k] = up_cols(d, t.mlp_down, H, I, sh * Is, Is);

                        g.kc[k] = dalloc<uint16_t>(d, size_t(g.cap * KD));
                        g.vc[k] = dalloc<uint16_t>(d, size_t(g.cap * KD));
                        shards_[k].q.memset(g.kc[k], 0, size_t(g.cap * KD) * 2);
                        shards_[k].q.memset(g.vc[k], 0, size_t(g.cap * KD) * 2);
                        kvbytes_ += int64_t(g.cap * KD) * 4;
                    }
                }
                stage_.clear();
                stage_.shrink_to_fit();
            }

            void alloc_scratch()
            {
                const Config &c = *cfg_;
                const int64_t H = c.hidden_size, I = c.intermediate_size;
                const int64_t QD = c.q_dim(), KD = c.kv_dim(), V = c.vocab_size;
                const int64_t B = block_, ns = nshard_;
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
                    d.xfer = dalloc<float>(i, size_t(B * H));
                    d.peer = dalloc<float>(i, size_t(ns * B * H));
                    d.rope_cos = dalloc<float>(i, cos32.size());
                    d.rope_sin = dalloc<float>(i, sin32.size());
                    d.q.memcpy(d.rope_cos, cos32.data(), cos32.size() * 4);
                    d.q.memcpy(d.rope_sin, sin32.data(), sin32.size() * 4);
                    d.q.wait();
                }
                if (ns > 1)
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

                // Transport. Direct card-to-card memcpy is the obvious choice
                // and benchmarks at ~48 GB/s in isolation, but measured 0.4
                // GB/s inside this engine — 70x slower — for reasons that do
                // not reproduce standalone (not VRAM pressure, not a
                // kernel-written source; both were tested). Staging through
                // PINNED host memory is a hop longer and still ~26 GB/s, so it
                // is the default. MUSE_GPU_P2P=1 selects the direct path for
                // re-measuring when the driver changes.
                std::vector<sycl::event> ev;
                if (p2p_)
                {
                    for (int64_t sh = 0; sh < ns; ++sh)
                        for (int64_t r = 0; r < ns; ++r)
                            if (r != sh)
                                ev.push_back(shards_[size_t(sh)].q.memcpy(
                                    shards_[size_t(r)].peer + sh * slot,
                                    shards_[size_t(sh)].peer + sh * slot, bytes));
                    for (auto &e : ev)
                        e.wait();
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
            void gemm(int dev, const uint16_t *W, const uint16_t *X, float *Y, int64_t n,
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
                        gemm(int(sh), l.k[size_t(sh)], d.xb, d.kf, n, H, KD, B, B);
                        gemm(int(sh), l.v[size_t(sh)], d.xb, d.vf, n, H, KD, B, B);
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
                        k_attention(d.q, d.qb, l.kc[size_t(sh)], l.vc[size_t(sh)], d.of, n, nqs,
                                    D, KD, B, pos0, groups, l.cap, window, scaling, sh * nqs);
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
                    toc(S_OPROJ);
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
                    gemm(0, W, X + off, Y, n, in, out, ld, ld);
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
            bool p2p_ = false;
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
            std::vector<uint16_t *> embed_, lm_head_, final_norm_;
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
