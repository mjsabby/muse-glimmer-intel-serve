// A fast BF16 CPU engine for Muse Glimmer — the production-shaped counterpart
// to the f64 oracle, and a CANDIDATE implementation rather than a referee.
//
// It is refereed against `muse-oracle --dtype bf16 --attn flash`: weights are
// the checkpoint's own BF16 bytes used in place (no conversion, no copy),
// activations materialize at BF16 exactly where the twin says they do, and the
// dot products accumulate in f32 via AVX-512 `vdpbf16ps`. That last part is why
// the gate is an envelope gate and not a bitwise one — the twin specifies an
// *ideal* accumulator and this engine has an f32 one, which is the same choice
// llama.cpp makes.
//
// Two structural differences from `muse_glimmer.hpp`, both of which are the
// point of the file:
//
//   1. **Weights are streamed as BF16.** The oracle converts every weight row
//      to f64 on the way past, so it touches 8 bytes per parameter; this engine
//      touches 2. On the released 30B that is 55.5 GiB per forward instead of
//      222 GiB, and decode is entirely bound by it.
//   2. **There is a KV cache**, so decode is O(1) in context rather than
//      re-running prefill: a `sliding_window + chunk` ring for the 39 sliding
//      layers and a linear buffer for the 13 global ones. Without this there is
//      no tokens/sec number to report at all.
#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "muse_glimmer.hpp"

#if defined(__AVX512F__)
#include <immintrin.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

namespace muse
{
    namespace bf16
    {

        // ------------------------------------------------------------ scalars

        inline float bf16_to_f32(uint16_t v)
        {
            uint32_t bits = uint32_t(v) << 16;
            float f;
            std::memcpy(&f, &bits, 4);
            return f;
        }

        // RNE, matching prec::round_bf16's second step (the first step, f64->f32,
        // has already happened: this engine's arithmetic is f32 throughout).
        inline uint16_t f32_to_bf16(float f)
        {
            uint32_t u;
            std::memcpy(&u, &f, 4);
            if (f != f)
                return 0x7fc0;
            return uint16_t((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
        }

        inline void round_to_bf16(const float *src, uint16_t *dst, int64_t n)
        {
            for (int64_t i = 0; i < n; ++i)
                dst[i] = f32_to_bf16(src[i]);
        }

        // ------------------------------------------------------------- kernels
        //
        // `vdpbf16ps` consumes 32 bf16 per operand and produces 16 f32 lanes,
        // each holding a[2j]*b[2j] + a[2j+1]*b[2j+1] + acc[j]. The reduction
        // order is therefore fixed by the instruction, not by us — which is
        // exactly why this engine cannot be bitwise against the twin and is
        // gated on the logit envelope instead.

#if defined(__AVX512BF16__)
        inline float dot_bf16(const uint16_t *a, const uint16_t *b, int64_t n)
        {
            __m512 acc = _mm512_setzero_ps();
            int64_t i = 0;
            for (; i + 32 <= n; i += 32)
                acc = _mm512_dpbf16_ps(acc,
                                       (__m512bh)_mm512_loadu_si512(a + i),
                                       (__m512bh)_mm512_loadu_si512(b + i));
            float s = _mm512_reduce_add_ps(acc);
            for (; i < n; ++i) // tail: head_dim 16 on the tiny models
                s += bf16_to_f32(a[i]) * bf16_to_f32(b[i]);
            return s;
        }
#else
        inline float dot_bf16(const uint16_t *a, const uint16_t *b, int64_t n)
        {
            float s = 0;
            for (int64_t i = 0; i < n; ++i)
                s += bf16_to_f32(a[i]) * bf16_to_f32(b[i]);
            return s;
        }
#endif

        // Y[t*out + o] = dot(W[o,:], X[t,:]) with W [out,in] and X [T,in], both
        // BF16, Y f32. Parallel over output rows; each weight row is read once
        // and reused across the whole token block, which is what makes prefill
        // compute-bound and decode bandwidth-bound.
        inline void gemm(const uint16_t *W, const uint16_t *X, float *Y, int64_t T, int64_t in,
                         int64_t out)
        {
#if defined(__AVX512BF16__)
            // Register tile RM=4 weight rows x RN=6 tokens — the shape
            // llama.cpp's tinyBLAS uses (llamafile/sgemm.cpp mnpack<4,6,..>),
            // and the best of everything measured end-to-end on the real model
            // with a warm page cache, best of 3:
            //   4x4 54.43   4x5 54.47   4x6 56.25   6x4 53.97   8x4 47.63
            // Two-level (output x token) blocking after tinyBLAS's BM/BN was
            // also tried and measured 55.09 — no better, because at these
            // shapes the activation matrix is already L3-resident and the
            // re-read traffic the blocking removes was never the constraint.
            constexpr int64_t RB = 4;
            constexpr int64_t TB = 6;
            const int64_t n32 = in & ~int64_t(31);
#pragma omp parallel for schedule(static)
            for (int64_t o0 = 0; o0 < out; o0 += RB)
            {
                const int64_t nr = std::min<int64_t>(RB, out - o0);
                for (int64_t t0 = 0; t0 < T; t0 += TB)
                {
                    const int64_t nt = std::min<int64_t>(TB, T - t0);
                    __m512 acc[RB][TB];
                    for (int64_t r = 0; r < nr; ++r)
                        for (int64_t t = 0; t < nt; ++t)
                            acc[r][t] = _mm512_setzero_ps();
                    for (int64_t i = 0; i < n32; i += 32)
                    {
                        __m512bh wv[RB];
                        for (int64_t r = 0; r < nr; ++r)
                            wv[r] = (__m512bh)_mm512_loadu_si512(W + (o0 + r) * in + i);
                        for (int64_t t = 0; t < nt; ++t)
                        {
                            const __m512bh xv =
                                (__m512bh)_mm512_loadu_si512(X + (t0 + t) * in + i);
                            for (int64_t r = 0; r < nr; ++r)
                                acc[r][t] = _mm512_dpbf16_ps(acc[r][t], wv[r], xv);
                        }
                    }
                    for (int64_t r = 0; r < nr; ++r)
                        for (int64_t t = 0; t < nt; ++t)
                        {
                            float s = _mm512_reduce_add_ps(acc[r][t]);
                            for (int64_t i = n32; i < in; ++i)
                                s += bf16_to_f32(W[(o0 + r) * in + i]) *
                                     bf16_to_f32(X[(t0 + t) * in + i]);
                            Y[(t0 + t) * out + o0 + r] = s;
                        }
                }
            }
#else
#pragma omp parallel for schedule(static)
            for (int64_t o = 0; o < out; ++o)
                for (int64_t t = 0; t < T; ++t)
                    Y[t * out + o] = dot_bf16(W + o * in, X + t * in, in);
#endif
        }

        // ---------------------------------------------------------- profiling
        //
        // MUSE_BF16_PROFILE=1 accumulates wall time per phase. Phases are
        // sequential within a layer, so plain (non-atomic) accumulation outside
        // the parallel regions is exact. Off by default and free when off.
        struct Profile
        {
            double qkv = 0, attn = 0, oproj = 0, mlp = 0, norm = 0, head = 0;
            double mlp_gemm = 0, mlp_act = 0, mlp_norm = 0, resid = 0, gate_elem = 0;
            bool on = false;
            std::chrono::steady_clock::time_point t0;
            void tic()
            {
                if (on)
                    t0 = std::chrono::steady_clock::now();
            }
            void toc(double &slot)
            {
                if (on)
                    slot += std::chrono::duration<double>(
                                std::chrono::steady_clock::now() - t0)
                                .count();
            }
            void report(FILE *f) const
            {
                if (!on)
                    return;
                const double tot = qkv + attn + oproj + mlp + norm + head;
                fprintf(f, "profile (s): qkv %.2f  attn %.2f  o_proj %.2f  mlp %.2f  "
                           "norm %.2f  head %.2f  | total %.2f\n",
                        qkv, attn, oproj, mlp, norm, head, tot);
                fprintf(f, "         (%%): qkv %.1f  attn %.1f  o_proj %.1f  mlp %.1f  "
                           "norm %.1f  head %.1f\n",
                        100 * qkv / tot, 100 * attn / tot, 100 * oproj / tot,
                        100 * mlp / tot, 100 * norm / tot, 100 * head / tot);
                fprintf(f, "  breakdown: mlp_gemm %.2f  mlp_act %.2f  mlp_norm %.2f  "
                           "resid %.2f  gate_elem %.2f\n",
                        mlp_gemm, mlp_act, mlp_norm, resid, gate_elem);
            }
        };

        // --------------------------------------------------------- weight view
        //
        // Views into the mmap. Nothing is copied: `st::Tensor::data` already
        // points at the BF16 bytes of the safetensors shard.

        struct LayerView
        {
            const uint16_t *input_ln = nullptr, *post_attn_ln = nullptr, *pre_ff_ln = nullptr,
                           *post_ff_ln = nullptr;
            const uint16_t *q = nullptr, *k = nullptr, *v = nullptr, *gate = nullptr,
                           *o = nullptr;
            const uint16_t *mlp_gate = nullptr, *mlp_up = nullptr, *mlp_down = nullptr;
        };

        struct WeightView
        {
            const uint16_t *embed = nullptr, *lm_head = nullptr, *final_norm = nullptr;
            std::vector<LayerView> layers;
        };

        inline const uint16_t *as_bf16(const st::Tensor &t)
        {
            if (t.dtype != st::DType::BF16)
                throw std::runtime_error(
                    t.name + ": the bf16 engine needs a BF16 checkpoint (this tensor is not); "
                             "the f64 oracle accepts any float dtype, this one does not");
            return reinterpret_cast<const uint16_t *>(t.data);
        }

        inline WeightView bind(const muse::Weights &w)
        {
            WeightView v;
            v.embed = as_bf16(*w.embed);
            v.final_norm = as_bf16(*w.final_norm);
            v.lm_head = as_bf16(w.lm_head ? *w.lm_head : *w.embed);
            for (const auto &l : w.layers)
            {
                LayerView lv;
                lv.input_ln = as_bf16(*l.input_ln);
                lv.post_attn_ln = as_bf16(*l.post_attn_ln);
                lv.pre_ff_ln = as_bf16(*l.pre_ff_ln);
                lv.post_ff_ln = as_bf16(*l.post_ff_ln);
                lv.q = as_bf16(*l.q_proj);
                lv.k = as_bf16(*l.k_proj);
                lv.v = as_bf16(*l.v_proj);
                lv.gate = as_bf16(*l.gate_proj);
                lv.o = as_bf16(*l.o_proj);
                lv.mlp_gate = as_bf16(*l.mlp_gate);
                lv.mlp_up = as_bf16(*l.mlp_up);
                lv.mlp_down = as_bf16(*l.mlp_down);
                v.layers.push_back(lv);
            }
            return v;
        }

        // ------------------------------------------------------------ KV cache
        //
        // The 39 sliding layers only ever need the last `sliding_window` keys,
        // so they get a ring of that size and their memory is independent of
        // context length. The 13 global layers grow linearly. With only 2 KV
        // heads at head_dim 128 this is 1 KiB per layer per token in BF16:
        // 13 KiB/token for the global layers plus a fixed 78 MiB of rings.
        struct KVCache
        {
            int64_t max_seq = 0, window = 0, kv_dim = 0;
            std::vector<int64_t> cap;              // rows per layer
            std::vector<std::vector<uint16_t>> K, V;
            int64_t len = 0;                       // logical tokens stored

            // A sliding layer's ring must hold `sliding_window + block` rows,
            // not `sliding_window`: a prefill chunk writes `block` new keys
            // before any of its queries read, and the oldest query in the chunk
            // still needs `window` keys behind it. Sizing it at `window` alone
            // silently overwrites live keys as soon as a chunk exceeds the
            // window — the model keeps running and the logits are wrong.
            void init(const muse::Config &c, int64_t max_seq_, int64_t block)
            {
                max_seq = max_seq_;
                window = c.sliding_window;
                kv_dim = c.kv_dim();
                const int64_t L = c.num_hidden_layers;
                cap.assign(size_t(L), 0);
                K.assign(size_t(L), {});
                V.assign(size_t(L), {});
                for (int64_t i = 0; i < L; ++i)
                {
                    cap[size_t(i)] = c.layer_is_sliding(i)
                                         ? std::min(window + block, max_seq)
                                         : max_seq;
                    K[size_t(i)].assign(size_t(cap[size_t(i)] * kv_dim), 0);
                    V[size_t(i)].assign(size_t(cap[size_t(i)] * kv_dim), 0);
                }
                len = 0;
            }
            int64_t slot(int64_t layer, int64_t pos) const
            {
                const int64_t c = cap[size_t(layer)];
                return c ? pos % c : 0;
            }
            uint16_t *k(int64_t layer, int64_t pos)
            {
                return K[size_t(layer)].data() + slot(layer, pos) * kv_dim;
            }
            uint16_t *v(int64_t layer, int64_t pos)
            {
                return V[size_t(layer)].data() + slot(layer, pos) * kv_dim;
            }
            size_t bytes() const
            {
                size_t n = 0;
                for (size_t i = 0; i < K.size(); ++i)
                    n += (K[i].size() + V[i].size()) * 2;
                return n;
            }
        };

        // ---------------------------------------------------------- the engine

        struct Engine
        {
            const muse::Config *cfg = nullptr;
            WeightView w;
            KVCache kv;
            muse::RopeTable rope;
            Profile prof;

            // scratch, sized for the widest token block
            int64_t block = 0;
            std::vector<uint16_t> xb, qb, kbuf, vbuf, gb, ob, g1b, u1b;
            std::vector<float> h, xf, qf, kf, vf, gf, of, g1, u1, logits_f;

            void init(const muse::Config &c, const muse::Weights &weights, int64_t max_seq,
                      int64_t max_block)
            {
                cfg = &c;
                w = bind(weights);
                if (const char *e = getenv("MUSE_BF16_PROFILE"); e && *e && *e != '0')
                    prof.on = true;
                block = max_block;
                kv.init(c, max_seq, block);
                // The bf16 twin builds cos/sin through the stock f32 chain and
                // rounds them to bf16; reuse the oracle's table builder so the
                // two engines cannot drift apart on rope.
                rope = muse::build_rope_table(c, max_seq, false, prec::Dtype::BF16);

                const int64_t H = c.hidden_size, I = c.intermediate_size;
                const int64_t QD = c.q_dim(), KD = c.kv_dim();
                h.assign(size_t(block * H), 0.f);
                xf.assign(size_t(block * H), 0.f);
                xb.assign(size_t(block * H), 0);
                qf.assign(size_t(block * QD), 0.f);
                qb.assign(size_t(block * QD), 0);
                kf.assign(size_t(block * KD), 0.f);
                kbuf.assign(size_t(block * KD), 0);
                vf.assign(size_t(block * KD), 0.f);
                vbuf.assign(size_t(block * KD), 0);
                gf.assign(size_t(block * QD), 0.f);
                gb.assign(size_t(block * QD), 0);
                of.assign(size_t(block * QD), 0.f);
                ob.assign(size_t(block * QD), 0);
                g1.assign(size_t(block * I), 0.f);
                g1b.assign(size_t(block * I), 0);
                u1.assign(size_t(block * I), 0.f);
                u1b.assign(size_t(block * I), 0);
                logits_f.assign(size_t(c.vocab_size), 0.f);
            }

            // ---- elementwise pieces, in the twin's op order

            // MuseGlimmerTextCenteredRMSNorm / RMSNorm / weight-less, f32
            // internals with one BF16 materialization at the end — the twin
            // canonicalizes stock's f32 upcast and rounds once at `.type_as(x)`.
            void norm_rows(muse::NormKind kind, const float *X, const uint16_t *wt, double eps,
                           uint16_t *out, int64_t rows, int64_t dim) const
            {
#pragma omp parallel for schedule(static)
                for (int64_t r = 0; r < rows; ++r)
                {
                    const float *x = X + r * dim;
                    double ss = 0;
                    for (int64_t i = 0; i < dim; ++i)
                        ss += double(x[i]) * double(x[i]);
                    const double rs = 1.0 / std::sqrt(ss / double(dim) + eps);
                    uint16_t *o = out + r * dim;
                    for (int64_t i = 0; i < dim; ++i)
                    {
                        double v = double(x[i]) * rs;
                        if (kind == muse::NormKind::Centered)
                            v = v * (1.0 + double(bf16_to_f32(wt[i])));
                        else if (kind == muse::NormKind::Plain)
                            v = v * double(bf16_to_f32(wt[i]));
                        o[i] = f32_to_bf16(float(v));
                    }
                }
            }

            void norm_rows_f32(muse::NormKind kind, const float *X, const uint16_t *wt,
                               double eps, float *out, int64_t rows, int64_t dim) const
            {
#pragma omp parallel for schedule(static)
                for (int64_t r = 0; r < rows; ++r)
                {
                    const float *x = X + r * dim;
                    double ss = 0;
                    for (int64_t i = 0; i < dim; ++i)
                        ss += double(x[i]) * double(x[i]);
                    const double rs = 1.0 / std::sqrt(ss / double(dim) + eps);
                    float *o = out + r * dim;
                    for (int64_t i = 0; i < dim; ++i)
                    {
                        double v = double(x[i]) * rs;
                        if (kind == muse::NormKind::Plain)
                            v = v * double(bf16_to_f32(wt[i]));
                        o[i] = bf16_to_f32(f32_to_bf16(float(v)));
                    }
                }
            }

            static float silu_f(float x) { return x / (1.0f + std::exp(-x)); }
            static float sigmoid_f(float x) { return 1.0f / (1.0f + std::exp(-x)); }

            // ---- one block of `n` tokens starting at absolute position `pos0`
            //
            // Used for both prefill (n = chunk) and decode (n = 1); the only
            // difference is how many query rows there are, so there is exactly
            // one attention implementation and decode cannot drift from prefill.
            void forward_block(const std::vector<int64_t> &ids, int64_t pos0, int64_t n,
                               float *out_logits_last)
            {
                const muse::Config &c = *cfg;
                const int64_t H = c.hidden_size, I = c.intermediate_size, D = c.head_dim;
                const int64_t nq = c.num_attention_heads, nkv = c.num_key_value_heads;
                const int64_t QD = c.q_dim(), KD = c.kv_dim(), groups = c.kv_groups();
                const float scaling = float(1.0 / std::sqrt(double(D)));
                const float qk_scale = float(c.qk_scale_factor);

                // embedding + weight-less embed-norm
                for (int64_t t = 0; t < n; ++t)
                {
                    const uint16_t *row = w.embed + ids[size_t(pos0 + t)] * H;
                    for (int64_t i = 0; i < H; ++i)
                        h[size_t(t * H + i)] = bf16_to_f32(row[i]);
                }
                norm_rows_f32(muse::NormKind::Weightless, h.data(), nullptr, c.rms_norm_eps,
                              h.data(), n, H);

                for (int64_t li = 0; li < c.num_hidden_layers; ++li)
                {
                    const LayerView &l = w.layers[size_t(li)];
                    const bool sliding = c.layer_is_sliding(li);
                    const bool use_rope = c.layer_has_rope(li);

                    prof.tic();
                    norm_rows(muse::NormKind::Centered, h.data(), l.input_ln, c.rms_norm_eps,
                              xb.data(), n, H);
                    prof.toc(prof.norm);

                    prof.tic();
                    gemm(l.q, xb.data(), qf.data(), n, H, QD);
                    gemm(l.k, xb.data(), kf.data(), n, H, KD);
                    gemm(l.v, xb.data(), vf.data(), n, H, KD);
                    gemm(l.gate, xb.data(), gf.data(), n, H, QD);
                    prof.toc(prof.qkv);

                    // weight-less QK-norm over head_dim, then qk_scale_factor on q only
                    norm_rows(muse::NormKind::Weightless, qf.data(), nullptr, c.rms_norm_eps,
                              qb.data(), n * nq, D);
                    norm_rows(muse::NormKind::Weightless, kf.data(), nullptr, c.rms_norm_eps,
                              kbuf.data(), n * nkv, D);
#pragma omp parallel for schedule(static)
                    for (int64_t i = 0; i < n * QD; ++i)
                        qb[size_t(i)] = f32_to_bf16(bf16_to_f32(qb[size_t(i)]) * qk_scale);

                    if (use_rope)
                    {
                        const int64_t half = D / 2;
#pragma omp parallel for schedule(static) collapse(2)
                        for (int64_t t = 0; t < n; ++t)
                            for (int64_t hh = 0; hh < nq + nkv; ++hh)
                            {
                                uint16_t *p = hh < nq ? qb.data() + (t * nq + hh) * D
                                                      : kbuf.data() + (t * nkv + (hh - nq)) * D;
                                const double *co = &rope.cos[size_t((pos0 + t) * half)];
                                const double *si = &rope.sin[size_t((pos0 + t) * half)];
                                for (int64_t j = 0; j < half; ++j)
                                {
                                    const float lo = bf16_to_f32(p[j]),
                                                hi = bf16_to_f32(p[j + half]);
                                    const float cj = float(co[j]), sj = float(si[j]);
                                    p[j] = f32_to_bf16(bf16_to_f32(f32_to_bf16(lo * cj)) +
                                                       bf16_to_f32(f32_to_bf16(-hi * sj)));
                                    p[j + half] =
                                        f32_to_bf16(bf16_to_f32(f32_to_bf16(hi * cj)) +
                                                    bf16_to_f32(f32_to_bf16(lo * sj)));
                                }
                            }
                    }

                    // append this block's k/v to the cache
                    for (int64_t t = 0; t < n; ++t)
                    {
                        std::memcpy(kv.k(li, pos0 + t), kbuf.data() + t * KD, size_t(KD) * 2);
                        for (int64_t i = 0; i < KD; ++i)
                            kv.v(li, pos0 + t)[i] = f32_to_bf16(vf[size_t(t * KD + i)]);
                    }

                    // attention: flash semantics — S and P never materialize at
                    // bf16, matching `--attn flash`
                    prof.tic();
#pragma omp parallel for schedule(static) collapse(2)
                    for (int64_t t = 0; t < n; ++t)
                        for (int64_t hh = 0; hh < nq; ++hh)
                        {
                            const int64_t qpos = pos0 + t, g = hh / groups;
                            const int64_t lo = sliding ? std::max<int64_t>(0, qpos - c.sliding_window + 1) : 0;
                            const uint16_t *q = qb.data() + (t * nq + hh) * D;
                            // Accumulate straight into the output row: no
                            // per-(token, head) allocation, and one fewer pass
                            // over D. At T=2048 this loop runs 2048 times per
                            // (token, head) and was the reason prefill fell
                            // from 3.5 to 2.4 TFLOP/s with depth.
                            float *o = of.data() + (t * nq + hh) * D;
                            std::memset(o, 0, size_t(D) * sizeof(float));
                            float mx = -INFINITY, sum = 0.f;
                            for (int64_t j = lo; j <= qpos; ++j)
                            {
                                const float s =
                                    dot_bf16(q, kv.k(li, j) + g * D, D) * scaling;
                                const float m2 = std::max(mx, s);
                                const float corr = std::exp(mx - m2);
                                const float e = std::exp(s - m2);
                                sum = sum * corr + e;
                                const uint16_t *vv = kv.v(li, j) + g * D;
                                int64_t d = 0;
#if defined(__AVX512F__)
                                const __m512 vc = _mm512_set1_ps(corr);
                                const __m512 ve = _mm512_set1_ps(e);
                                for (; d + 16 <= D; d += 16)
                                {
                                    // bf16 -> f32 is a 16-bit left shift
                                    const __m512 vv32 = _mm512_castsi512_ps(_mm512_slli_epi32(
                                        _mm512_cvtepu16_epi32(
                                            _mm256_loadu_si256((const __m256i *)(vv + d))),
                                        16));
                                    // mul, mul, add — deliberately NOT fmadd.
                                    // Measured identical envelope either way,
                                    // so the tie-break is that the scalar tail
                                    // below cannot fuse: keeping both
                                    // unfused means a head with D % 16 != 0
                                    // rounds the same way across the split.
                                    _mm512_storeu_ps(
                                        o + d,
                                        _mm512_add_ps(_mm512_mul_ps(_mm512_loadu_ps(o + d), vc),
                                                      _mm512_mul_ps(ve, vv32)));
                                }
#endif
                                for (; d < D; ++d)
                                    o[d] = o[d] * corr + e * bf16_to_f32(vv[d]);
                                mx = m2;
                            }
                            const float inv = 1.0f / sum;
                            for (int64_t d = 0; d < D; ++d)
                                o[d] *= inv;
                        }

                    prof.toc(prof.attn);

                    // output gate from the PRE-attention normed input
                    prof.tic();
#pragma omp parallel for schedule(static)
                    for (int64_t i = 0; i < n * QD; ++i)
                    {
                        // one BF16 materialization per nn.Linear output, then
                        // one per elementwise op — the twin's op order exactly
                        const float pv = bf16_to_f32(f32_to_bf16(of[size_t(i)]));
                        const float gt = bf16_to_f32(f32_to_bf16(gf[size_t(i)]));
                        const float sg = bf16_to_f32(f32_to_bf16(sigmoid_f(gt)));
                        ob[size_t(i)] = f32_to_bf16(pv * sg);
                    }
                    gemm(l.o, ob.data(), xf.data(), n, QD, H);
                    norm_rows(muse::NormKind::Centered, xf.data(), l.post_attn_ln,
                              c.post_norm_eps, xb.data(), n, H);
                    for (int64_t i = 0; i < n * H; ++i)
                        h[size_t(i)] =
                            bf16_to_f32(f32_to_bf16(h[size_t(i)] + bf16_to_f32(xb[size_t(i)])));
                    prof.toc(prof.oproj);

                    // SwiGLU
                    prof.tic();
                    Profile sub;
                    sub.on = prof.on;
                    sub.tic();
                    norm_rows(muse::NormKind::Centered, h.data(), l.pre_ff_ln, c.rms_norm_eps,
                              xb.data(), n, H);
                    sub.toc(prof.mlp_norm);
                    sub.tic();
                    gemm(l.mlp_gate, xb.data(), g1.data(), n, H, I);
                    gemm(l.mlp_up, xb.data(), u1.data(), n, H, I);
                    sub.toc(prof.mlp_gemm);
                    sub.tic();
#pragma omp parallel for schedule(static)
                    for (int64_t i = 0; i < n * I; ++i)
                        g1b[size_t(i)] = f32_to_bf16(
                            bf16_to_f32(f32_to_bf16(
                                silu_f(bf16_to_f32(f32_to_bf16(g1[size_t(i)]))))) *
                            bf16_to_f32(f32_to_bf16(u1[size_t(i)])));
                    sub.toc(prof.mlp_act);
                    sub.tic();
                    gemm(l.mlp_down, g1b.data(), xf.data(), n, I, H);
                    sub.toc(prof.mlp_gemm);
                    sub.tic();
                    norm_rows(muse::NormKind::Centered, xf.data(), l.post_ff_ln,
                              c.post_norm_eps, xb.data(), n, H);
                    sub.toc(prof.mlp_norm);
                    sub.tic();
                    for (int64_t i = 0; i < n * H; ++i)
                        h[size_t(i)] =
                            bf16_to_f32(f32_to_bf16(h[size_t(i)] + bf16_to_f32(xb[size_t(i)])));
                    sub.toc(prof.resid);
                    prof.toc(prof.mlp);
                }
                kv.len = pos0 + n;

                if (!out_logits_last)
                    return;
                // final plain norm + lm_head, last row only (the serving shape)
                prof.tic();
                norm_rows(muse::NormKind::Plain, h.data() + (n - 1) * H, w.final_norm,
                          c.rms_norm_eps, xb.data(), 1, H);
                gemm(w.lm_head, xb.data(), out_logits_last, 1, H, c.vocab_size);
                const float mult = float(c.output_multiplier);
                const float cap = float(c.final_logit_softcapping);
#pragma omp parallel for schedule(static)
                for (int64_t i = 0; i < c.vocab_size; ++i)
                {
                    float z = bf16_to_f32(f32_to_bf16(out_logits_last[i]));  // lm_head output
                    z = bf16_to_f32(f32_to_bf16(z * mult));
                    if (cap != 0.f)
                    {
                        z = bf16_to_f32(f32_to_bf16(z / cap));
                        z = bf16_to_f32(f32_to_bf16(std::tanh(z)));
                        z = bf16_to_f32(f32_to_bf16(z * cap));
                    }
                    out_logits_last[i] = z;
                }
                prof.toc(prof.head);
            }

            // Prefill `ids` in chunks of `block`, returning the last position's
            // logits. Decode then appends one token at a time against the cache.
            void prefill(const std::vector<int64_t> &ids, float *logits_last)
            {
                const int64_t T = int64_t(ids.size());
                for (int64_t p = 0; p < T; p += block)
                {
                    const int64_t n = std::min(block, T - p);
                    forward_block(ids, p, n, (p + n == T) ? logits_last : nullptr);
                }
            }

            int64_t argmax(const float *v, int64_t n) const
            {
                int64_t best = 0;
                for (int64_t i = 1; i < n; ++i)
                    if (v[i] > v[best])
                        best = i;
                return best;
            }
        };

    } // namespace bf16
} // namespace muse
