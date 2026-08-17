// Muse Glimmer 30B (model_type muse_glimmer / muse_glimmer_text) text-path
// forward in pure IEEE f64, mirroring transformers 5.15.0
// models/muse_glimmer/modeling_muse_glimmer.py op for op. See ARCHITECTURE.md
// for the line-referenced spec and docs/oracle.md for how to run it.
//
// Numerics: BF16/F32 checkpoint weights are converted exactly; every
// intermediate is f64; every reduction has a fixed order independent of thread
// count and ISA (parallelism only across independent output elements), so
// results are bitwise reproducible across thread counts, --kernels modes and
// machines.
//
// The four traps this file exists to get right (all verified, see
// ARCHITECTURE.md):
//   1. Three RMSNorm flavours. The four per-layer norms are ZERO-CENTERED
//      ((1+w)); the final `norm` is PLAIN (w); embed/qk/perception norms are
//      WEIGHT-LESS. assert_norm_flavours() checks this at load time.
//   2. Two eps values assigned by POSITION, not by name: rms_norm_eps (1e-5)
//      on input_layernorm / pre_feedforward_layernorm, post_norm_eps (1e-8) on
//      post_attention_layernorm / post_feedforward_layernorm.
//   3. NoPE on the global layers: layer_rope_theta[i] == 0 for every
//      full_attention layer and the reference passes position_embeddings=None
//      there. Only the 39 sliding layers rotate.
//   4. QK-norm is weight-less and `qk_scale_factor` (3.87) multiplies q ONLY,
//      in addition to the usual head_dim^-0.5.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "fmath.hpp"
#include "hf.hpp"
#include "rounding.hpp"
#include "safetensors.hpp"
#include "simd.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace muse
{

    // ---------------------------------------------------------------- config

    struct Config
    {
        int64_t hidden_size = 0, num_hidden_layers = 0, vocab_size = 0;
        int64_t num_attention_heads = 0, num_key_value_heads = 0, head_dim = 0;
        int64_t intermediate_size = 0;
        int64_t max_position_embeddings = 0;
        int64_t sliding_window = 0;
        double rms_norm_eps = 1e-5;
        double post_norm_eps = 1e-8;
        double qk_scale_factor = 1.0;
        double output_multiplier = 1.0;
        double final_logit_softcapping = 0.0; // 0 => no cap
        double rope_theta = 0.0;              // rope_parameters.rope_theta
        bool tie_word_embeddings = false;
        // per-layer: "sliding_attention" | "full_attention"
        std::vector<std::string> layer_types;
        // per-layer theta; 0.0 marks a NoPE layer (the reference passes
        // position_embeddings=None when layer_rope_theta[i] is falsy)
        std::vector<double> layer_rope_theta;
        // multimodal placeholders (top-level config; text-only runs ignore them)
        int64_t image_token_id = -1, video_token_id = -1;
        int64_t bos_token_id = -1, pad_token_id = -1;
        std::vector<int64_t> eos_token_ids;

        bool layer_is_sliding(int64_t i) const
        {
            return layer_types.at(size_t(i)) == "sliding_attention";
        }
        bool layer_has_rope(int64_t i) const
        {
            return layer_rope_theta.at(size_t(i)) != 0.0;
        }
        int64_t kv_groups() const { return num_attention_heads / num_key_value_heads; }
        int64_t q_dim() const { return num_attention_heads * head_dim; }
        int64_t kv_dim() const { return num_key_value_heads * head_dim; }
    };

    inline Config parse_config(const minijson::Value &root)
    {
        const minijson::Value *tcp = root.opt("text_config");
        const minijson::Value &tc = tcp ? *tcp : root;

        auto geti = [&](const char *k, int64_t dflt, bool required = false) -> int64_t
        {
            const minijson::Value *v = tc.opt(k);
            if (!v || v->is_null())
            {
                if (required)
                    throw std::runtime_error(std::string("config missing ") + k);
                return dflt;
            }
            return v->as_int();
        };
        auto getd = [&](const char *k, double dflt) -> double
        {
            const minijson::Value *v = tc.opt(k);
            return (v && !v->is_null()) ? v->as_double() : dflt;
        };

        std::string model_type;
        if (const auto *mt = root.opt("model_type"))
            model_type = mt->as_str();
        else if (const auto *mt2 = tc.opt("model_type"))
            model_type = mt2->as_str();
        if (model_type != "muse_glimmer" && model_type != "muse_glimmer_text")
            throw std::runtime_error("unsupported model_type: " + model_type);

        Config c;
        c.hidden_size = geti("hidden_size", 0, true);
        c.num_hidden_layers = geti("num_hidden_layers", 0, true);
        c.vocab_size = geti("vocab_size", 0, true);
        c.num_attention_heads = geti("num_attention_heads", 0, true);
        c.num_key_value_heads = geti("num_key_value_heads", 0, true);
        c.head_dim = geti("head_dim", 0, true);
        c.intermediate_size = geti("intermediate_size", 0, true);
        c.max_position_embeddings = geti("max_position_embeddings", 131072);
        c.sliding_window = geti("sliding_window", 0);
        c.rms_norm_eps = getd("rms_norm_eps", 1e-5);
        // post_norm_eps is what makes the sandwich norms differ by three orders
        // of magnitude; it must be present, never silently defaulted.
        if (!tc.opt("post_norm_eps"))
            throw std::runtime_error("config missing post_norm_eps (the sandwich "
                                     "post-norms use a different eps to the pre-norms)");
        c.post_norm_eps = getd("post_norm_eps", 1e-8);
        c.qk_scale_factor = getd("qk_scale_factor", 1.0);
        c.output_multiplier = getd("output_multiplier", 1.0);
        c.final_logit_softcapping = getd("final_logit_softcapping", 0.0);

        if (const auto *v = tc.opt("attention_bias"))
            if (v->as_bool())
                throw std::runtime_error("unsupported config: attention_bias");
        if (const auto *v = tc.opt("hidden_activation"))
            if (v->as_str() != "silu")
                throw std::runtime_error("unsupported hidden_activation: " + v->as_str());

        const minijson::Value *tie = root.opt("tie_word_embeddings");
        if (!tie)
            tie = tc.opt("tie_word_embeddings");
        c.tie_word_embeddings = tie && tie->as_bool();

        if (const auto *rp = tc.opt("rope_parameters"))
        {
            if (const auto *v = rp->opt("rope_type"))
                if (v->as_str() != "default")
                    throw std::runtime_error("unsupported rope_type: " + v->as_str());
            if (const auto *v = rp->opt("rope_theta"))
                c.rope_theta = v->as_double();
        }
        else if (const auto *v = tc.opt("rope_theta"))
            c.rope_theta = v->as_double();

        const auto *lt = tc.opt("layer_types");
        if (!lt)
            throw std::runtime_error("config missing layer_types");
        for (size_t i = 0; i < lt->size(); ++i)
        {
            std::string t = (*lt)[i].as_str();
            if (t != "sliding_attention" && t != "full_attention")
                throw std::runtime_error("unsupported layer type: " + t);
            c.layer_types.push_back(t);
        }
        if (int64_t(c.layer_types.size()) != c.num_hidden_layers)
            throw std::runtime_error("layer_types length mismatch");

        const auto *lrt = tc.opt("layer_rope_theta");
        if (lrt)
        {
            for (size_t i = 0; i < lrt->size(); ++i)
                c.layer_rope_theta.push_back((*lrt)[i].as_double());
            if (int64_t(c.layer_rope_theta.size()) != c.num_hidden_layers)
                throw std::runtime_error("layer_rope_theta length mismatch");
        }
        else
        {
            // configuration_muse_glimmer.py __post_init__ default: rope on
            // sliding layers, NoPE on full layers.
            for (int64_t i = 0; i < c.num_hidden_layers; ++i)
                c.layer_rope_theta.push_back(c.layer_is_sliding(i) ? c.rope_theta : 0.0);
        }

        // The oracle builds ONE rope table (the reference does the same: a
        // single MuseGlimmerTextRotaryEmbedding built from rope_parameters,
        // handed to every non-NoPE layer). A checkpoint with genuinely
        // per-layer thetas would need a table per distinct value — fail loudly
        // rather than silently rotating with the wrong base.
        for (int64_t i = 0; i < c.num_hidden_layers; ++i)
        {
            const double th = c.layer_rope_theta[size_t(i)];
            if (th != 0.0 && th != c.rope_theta)
                throw std::runtime_error(
                    "layer " + std::to_string(i) + " has layer_rope_theta " +
                    std::to_string(th) + " != rope_parameters.rope_theta " +
                    std::to_string(c.rope_theta) +
                    "; the reference builds a single rope table from "
                    "rope_parameters, so this checkpoint needs new code");
            if (th == 0.0 && c.layer_is_sliding(i))
                throw std::runtime_error("sliding layer " + std::to_string(i) +
                                         " is NoPE — unexpected layout");
        }
        if (c.sliding_window <= 0)
        {
            for (int64_t i = 0; i < c.num_hidden_layers; ++i)
                if (c.layer_is_sliding(i))
                    throw std::runtime_error("sliding layers present but sliding_window unset");
        }

        if (c.num_attention_heads % c.num_key_value_heads)
            throw std::runtime_error("num_attention_heads must be a multiple of "
                                     "num_key_value_heads");
        if (c.head_dim % 2)
            throw std::runtime_error("head_dim must be even (rotate_half pairs)");

        auto opt_int = [&](const minijson::Value &o, const char *k, int64_t &dst)
        {
            if (const auto *v = o.opt(k); v && !v->is_null())
                dst = v->as_int();
        };
        opt_int(root, "image_token_id", c.image_token_id);
        opt_int(root, "video_token_id", c.video_token_id);
        opt_int(tc, "bos_token_id", c.bos_token_id);
        opt_int(tc, "pad_token_id", c.pad_token_id);
        if (const auto *v = tc.opt("eos_token_id"); v && !v->is_null())
        {
            if (v->type == minijson::Value::Array)
                for (size_t i = 0; i < v->size(); ++i)
                    c.eos_token_ids.push_back((*v)[i].as_int());
            else
                c.eos_token_ids.push_back(v->as_int());
        }
        return c;
    }

    // ---------------------------------------------------------------- weights

    struct LayerWeights
    {
        // the Gemma-style sandwich: four zero-centered norms, two eps values
        const st::Tensor *input_ln = nullptr;   // eps = rms_norm_eps
        const st::Tensor *post_attn_ln = nullptr; // eps = post_norm_eps
        const st::Tensor *pre_ff_ln = nullptr;  // eps = rms_norm_eps
        const st::Tensor *post_ff_ln = nullptr; // eps = post_norm_eps
        const st::Tensor *q_proj = nullptr, *k_proj = nullptr, *v_proj = nullptr,
                         *gate_proj = nullptr, *o_proj = nullptr;
        const st::Tensor *mlp_gate = nullptr, *mlp_up = nullptr, *mlp_down = nullptr;
    };

    struct Weights
    {
        const st::Tensor *embed = nullptr;
        const st::Tensor *final_norm = nullptr;
        const st::Tensor *lm_head = nullptr; // null => tied to embed
        std::vector<LayerWeights> layers;
        std::string prefix;
    };

    inline void check_shape(const st::Tensor &t, std::vector<int64_t> want)
    {
        if (t.shape != want)
        {
            std::string msg = t.name + ": shape [";
            for (auto s : t.shape)
                msg += std::to_string(s) + ",";
            msg += "] expected [";
            for (auto s : want)
                msg += std::to_string(s) + ",";
            msg += "]";
            throw std::runtime_error(msg);
        }
    }

    // Guard against a +1-shifted (GGUF-style) tensor set reaching the
    // safetensors path, and against subtracting the +1 twice.
    //
    // What does NOT work, measured on the released 30B (tools/probe, and see
    // VERIFICATION.md §"Norm flavours"): telling the two flavours apart from
    // the weight statistics of one tensor. Zero-centered per-layer norms reach
    // mean +2.09 (layer 51 post_feedforward) with no negative entries at all,
    // while the PLAIN final `norm` has mean +0.0169 with 49.9% of its entries
    // negative — i.e. the final norm looks more "centered" than many of the
    // centered ones. The flavour comes from the module type in the reference,
    // not from the values, and the only value-level evidence is the
    // cross-source one (GGUF blk.*.attn_norm == safetensors + 1 exactly, GGUF
    // output_norm == safetensors exactly), which the GGUF loader checks.
    //
    // What IS sound is the aggregate: the centered parameterisation keeps
    // 1 + w >= 0, so every per-layer norm has min(w) >= -1, and across the
    // model most of them do contain negative entries (200 of 208 on the
    // released checkpoint). A +1-shifted set has min(w) >= 0 EVERYWHERE; a
    // doubly-subtracted set has min(w) ~ -2 somewhere.
    inline void assert_norm_flavours(const Weights &w, const Config &c)
    {
        int64_t n_norms = 0, n_with_negative = 0;
        std::vector<double> v;
        for (int64_t i = 0; i < c.num_hidden_layers; ++i)
        {
            const LayerWeights &lw = w.layers[size_t(i)];
            for (const st::Tensor *t : {lw.input_ln, lw.post_attn_ln, lw.pre_ff_ln, lw.post_ff_ln})
            {
                v.resize(size_t(t->numel()));
                st::to_f64(*t, 0, t->numel(), v.data());
                double mn = v[0];
                for (double x : v)
                    mn = std::min(mn, x);
                ++n_norms;
                n_with_negative += (mn < 0.0);
                if (mn < -1.0)
                    throw std::runtime_error(
                        t->name + ": min " + std::to_string(mn) +
                        " < -1, so this zero-centered norm would scale a channel by a "
                        "NEGATIVE (1+w); the +1 has most likely been subtracted twice");
            }
        }
        if (n_norms && n_with_negative * 2 < n_norms)
            throw std::runtime_error(
                "only " + std::to_string(n_with_negative) + " of " + std::to_string(n_norms) +
                " per-layer norms contain a negative weight; a zero-centered set has "
                "them almost everywhere (200/208 on the released 30B) and a +1-shifted "
                "GGUF-style set has none — subtract the 1 before binding these");
    }

    inline Weights bind_weights(hf::ModelFiles &mf, const Config &c)
    {
        Weights w;
        bool found = false;
        for (const char *p : {"model.language_model.", "language_model.", "model.", ""})
        {
            if (mf.has(std::string(p) + "embed_tokens.weight"))
            {
                w.prefix = p;
                found = true;
                break;
            }
        }
        if (!found)
            throw std::runtime_error("cannot find embed_tokens.weight under any known prefix");

        const int64_t H = c.hidden_size, D = c.head_dim, I = c.intermediate_size;
        const int64_t nq = c.num_attention_heads, nkv = c.num_key_value_heads;

        w.embed = &mf.tensor(w.prefix + "embed_tokens.weight");
        check_shape(*w.embed, {c.vocab_size, H});
        w.final_norm = &mf.tensor(w.prefix + "norm.weight");
        check_shape(*w.final_norm, {H});
        if (!c.tie_word_embeddings)
        {
            // lm_head is a top-level key even when the backbone is nested
            w.lm_head = &mf.tensor(mf.has("lm_head.weight") ? "lm_head.weight"
                                                            : w.prefix + "lm_head.weight");
            check_shape(*w.lm_head, {c.vocab_size, H});
        }

        for (int64_t i = 0; i < c.num_hidden_layers; ++i)
        {
            std::string lp = w.prefix + "layers." + std::to_string(i) + ".";
            auto T = [&](const std::string &n) -> const st::Tensor *
            { return &mf.tensor(lp + n); };
            LayerWeights lw{};
            lw.input_ln = T("input_layernorm.weight");
            lw.post_attn_ln = T("post_attention_layernorm.weight");
            lw.pre_ff_ln = T("pre_feedforward_layernorm.weight");
            lw.post_ff_ln = T("post_feedforward_layernorm.weight");
            for (const st::Tensor *t : {lw.input_ln, lw.post_attn_ln, lw.pre_ff_ln, lw.post_ff_ln})
                check_shape(*t, {H});
            lw.q_proj = T("self_attn.q_proj.weight");
            lw.k_proj = T("self_attn.k_proj.weight");
            lw.v_proj = T("self_attn.v_proj.weight");
            lw.gate_proj = T("self_attn.gate_proj.weight");
            lw.o_proj = T("self_attn.o_proj.weight");
            check_shape(*lw.q_proj, {nq * D, H});
            check_shape(*lw.k_proj, {nkv * D, H});
            check_shape(*lw.v_proj, {nkv * D, H});
            check_shape(*lw.gate_proj, {nq * D, H});
            check_shape(*lw.o_proj, {H, nq * D});
            // no q_norm/k_norm tensors: QK-norm is weight-less on this model
            if (mf.has(lp + "self_attn.q_norm.weight"))
                throw std::runtime_error(
                    lp + "self_attn.q_norm.weight exists, but MuseGlimmerTextAttention "
                         "uses a WEIGHT-LESS qk_norm plus the scalar qk_scale_factor");
            lw.mlp_gate = T("mlp.gate_proj.weight");
            lw.mlp_up = T("mlp.up_proj.weight");
            lw.mlp_down = T("mlp.down_proj.weight");
            check_shape(*lw.mlp_gate, {I, H});
            check_shape(*lw.mlp_up, {I, H});
            check_shape(*lw.mlp_down, {H, I});
            w.layers.push_back(lw);
        }
        assert_norm_flavours(w, c);
        return w;
    }

    // ---------------------------------------------------------------- kernels

    // Y[t,o] = sum_i W[o,i] * X[t,i]. W is a checkpoint tensor [out,in]
    // (bf16/f32), converted to f64 row by row. Parallel over output rows only;
    // the per-element reduction is simd::dot8 — the fixed 8-lane fma-blocked
    // order, identical between the AVX-512 and scalar executions.
    //
    // dt != F64: weights round to the storage dtype after conversion and each
    // output element rounds once after the exact f64 accumulation — nn.Linear
    // semantics with an ideal accumulator.
    constexpr int64_t GEMM_TILE_RMAX = 8;

    template <int R, int C>
    inline void gemm_rows(const st::Tensor &W, int64_t base, const double *X, double *Y,
                          int64_t T, int64_t in, int64_t out, int64_t o0, int64_t nr,
                          double *wbuf, prec::Dtype dt)
    {
        const double *wr[R];
        for (int64_t r = 0; r < nr; ++r)
        {
            double *wp = wbuf + r * in;
            st::to_f64(W, base + (o0 + r) * in, in, wp);
            if (dt != prec::Dtype::F64)
                for (int64_t i = 0; i < in; ++i)
                    wp[i] = prec::round_act(dt, wp[i]);
            wr[r] = wp;
        }
        int64_t t = 0;
#if defined(__AVX512F__)
        if (nr == R && !simd::force_scalar())
        {
            static thread_local std::vector<double> park, res;
            const int64_t ngroups = (T + C - 1) / C;
            if (int64_t(park.size()) < ngroups * R * C * 8)
                park.resize(size_t(ngroups * R * C * 8));
            if (int64_t(res.size()) < R * T)
                res.resize(size_t(R * T));
            simd::dot8_block_panel_avx512<R, C>(wr, X, T, in, res.data(), park.data());
            for (int64_t r = 0; r < R; ++r)
                for (int64_t tt = 0; tt < T; ++tt)
                    Y[tt * out + o0 + r] = prec::round_act(dt, res[size_t(r * T + tt)]);
            t = T;
        }
#endif
        for (; t < T; ++t)
            for (int64_t r = 0; r < nr; ++r)
                Y[t * out + o0 + r] =
                    prec::round_act(dt, simd::dot8(wr[r], X + t * in, in));
    }

    inline void gemm_slice(const st::Tensor &W, int64_t base, const double *X, double *Y,
                           int64_t T, int64_t in, int64_t out,
                           prec::Dtype dt = prec::Dtype::F64)
    {
#pragma omp parallel
        {
            static thread_local std::vector<double> wbuf;
            if (int64_t(wbuf.size()) < GEMM_TILE_RMAX * in)
                wbuf.resize(size_t(GEMM_TILE_RMAX * in));
            if (T >= 5)
            {
#pragma omp for schedule(static)
                for (int64_t ob = 0; ob < (out + 4) / 5; ++ob)
                    gemm_rows<5, 5>(W, base, X, Y, T, in, out, ob * 5,
                                    std::min<int64_t>(5, out - ob * 5), wbuf.data(), dt);
            }
            else
            {
#pragma omp for schedule(static)
                for (int64_t ob = 0; ob < (out + 7) / 8; ++ob)
                    gemm_rows<8, 3>(W, base, X, Y, T, in, out, ob * 8,
                                    std::min<int64_t>(8, out - ob * 8), wbuf.data(), dt);
            }
        }
    }

    inline void gemm(const st::Tensor &W, const double *X, double *Y, int64_t T, int64_t in,
                     int64_t out, prec::Dtype dt = prec::Dtype::F64)
    {
        check_shape(W, {out, in});
        gemm_slice(W, 0, X, Y, T, in, out, dt);
    }

    inline void load_vec(const st::Tensor &t, int64_t base, int64_t n, double *dst,
                         prec::Dtype dt)
    {
        st::to_f64(t, base, n, dst);
        if (dt != prec::Dtype::F64)
            for (int64_t i = 0; i < n; ++i)
                dst[i] = prec::round_act(dt, dst[i]);
    }

    inline double silu(double x) { return x / (1.0 + fmath::exp(-x)); }
    inline double sigmoid(double x) { return 1.0 / (1.0 + fmath::exp(-x)); }

    // ------------------------------------------------------------------ norms
    //
    // Three flavours, all sharing one mean-of-squares reduction:
    //   ms(x) = (sum_i x_i^2) / n, summed with 8 interleaved accumulators and
    //           the fixed tree ((a0+a1)+(a2+a3)) + ((a4+a5)+(a6+a7)), then the
    //           n mod 8 tail — the same shape as simd::dot8's epilogue, and
    //           what torch's f64 reduction agrees with on a 1-D row.
    //
    // The reciprocal square root is written `1.0 / std::sqrt(m)`. That is
    // bit-for-bit torch's `rsqrt` and `pow(m, -0.5)` in f64 (measured, 0/20000
    // mismatches, py/probe_torch_ops.py); torch's *vectorized* f64 `sqrt` is a
    // 1-ulp approximation, so `1/torch.sqrt(x)` is NOT the same thing and must
    // not be used to referee this.
    inline double mean_sq(const double *x, int64_t n)
    {
        double a0 = 0, a1 = 0, a2 = 0, a3 = 0, a4 = 0, a5 = 0, a6 = 0, a7 = 0;
        int64_t i = 0;
        for (; i + 8 <= n; i += 8)
        {
            a0 += x[i + 0] * x[i + 0];
            a1 += x[i + 1] * x[i + 1];
            a2 += x[i + 2] * x[i + 2];
            a3 += x[i + 3] * x[i + 3];
            a4 += x[i + 4] * x[i + 4];
            a5 += x[i + 5] * x[i + 5];
            a6 += x[i + 6] * x[i + 6];
            a7 += x[i + 7] * x[i + 7];
        }
        double tail = 0;
        for (; i < n; ++i)
            tail += x[i] * x[i];
        return ((((a0 + a1) + (a2 + a3)) + ((a4 + a5) + (a6 + a7))) + tail) / double(n);
    }

    // stock HF's f32 chain for the same reduction (--hf-f32-compat)
    inline float mean_sq_f32(const double *x, int64_t n)
    {
        float a0 = 0, a1 = 0, a2 = 0, a3 = 0, a4 = 0, a5 = 0, a6 = 0, a7 = 0;
        int64_t i = 0;
        for (; i + 8 <= n; i += 8)
        {
            float x0 = float(x[i + 0]), x1 = float(x[i + 1]), x2 = float(x[i + 2]),
                  x3 = float(x[i + 3]), x4 = float(x[i + 4]), x5 = float(x[i + 5]),
                  x6 = float(x[i + 6]), x7 = float(x[i + 7]);
            a0 += x0 * x0;
            a1 += x1 * x1;
            a2 += x2 * x2;
            a3 += x3 * x3;
            a4 += x4 * x4;
            a5 += x5 * x5;
            a6 += x6 * x6;
            a7 += x7 * x7;
        }
        float tail = 0;
        for (; i < n; ++i)
        {
            float xi = float(x[i]);
            tail += xi * xi;
        }
        return ((((a0 + a1) + (a2 + a3)) + ((a4 + a5) + (a6 + a7))) + tail) / float(n);
    }

    enum class NormKind
    {
        Centered,  // MuseGlimmerTextCenteredRMSNorm: y = x*rsqrt(ms+eps) * (1+w)
        Plain,     // MuseGlimmerRMSNorm(with_scale=True): y = x*pow(ms+eps,-.5) * w
        Weightless // MuseGlimmerRMSNorm(with_scale=False): y = x*pow(ms+eps,-.5)
    };

    // Row-parallel; in-place safe (Y may alias X). `w` may be null for
    // Weightless and must already hold storage-rounded values otherwise.
    //
    // Stock HF upcasts the whole chain to f32 (`x.float()`, `weight.float()`,
    // `.type_as(x)`) even in an f64 run; f32_compat replicates that placement.
    // dt != F64: internals stay exact f64 (stock's f32 upcast canonicalized
    // upward, as with softmax) and the output rounds once at `.type_as(x)`.
    inline void rmsnorm_rows(NormKind kind, const double *X, const double *w, double eps,
                             double *Y, int64_t rows, int64_t dim, bool f32_compat = false,
                             prec::Dtype dt = prec::Dtype::F64)
    {
#pragma omp parallel for schedule(static)
        for (int64_t r = 0; r < rows; ++r)
        {
            const double *x = X + r * dim;
            double *y = Y + r * dim;
            if (f32_compat)
            {
                float ms = mean_sq_f32(x, dim) + float(eps);
                float rs = 1.0f / std::sqrt(ms);
                for (int64_t i = 0; i < dim; ++i)
                {
                    float v = float(x[i]) * rs;
                    if (kind == NormKind::Centered)
                        v = v * (1.0f + float(w[i]));
                    else if (kind == NormKind::Plain)
                        v = v * float(w[i]);
                    y[i] = double(v);
                }
                continue;
            }
            const double ms = mean_sq(x, dim) + eps;
            const double rs = 1.0 / std::sqrt(ms);
            for (int64_t i = 0; i < dim; ++i)
            {
                double v = x[i] * rs;
                if (kind == NormKind::Centered)
                    v = v * (1.0 + w[i]);
                else if (kind == NormKind::Plain)
                    v = v * w[i];
                y[i] = prec::round_act(dt, v);
            }
        }
    }

    // ------------------------------------------------------------------- rope

    struct RopeTable
    {
        int64_t dim = 0;              // full head_dim (Muse Glimmer rotates all of it)
        std::vector<double> cos, sin; // [T, dim/2]
    };

    // MuseGlimmerTextRotaryEmbedding.compute_default_rope_parameters:
    //   inv_freq[j] = theta^(-2j/head_dim),  j in [0, head_dim/2)
    // Stock computes inv_freq AND the trig in f32 (the arange is `torch.float`
    // and the forward runs under a forced-f32 autocast region); pure mode
    // computes them in f64, f32_compat replicates the f32 chain, and
    // dt != F64 replicates the stock materialization chain: f32 inv_freq,
    // f32 freqs, cos/sin rounded to storage at `.to(x.dtype)`.
    inline RopeTable build_rope_table(const Config &c, int64_t T, bool f32_compat = false,
                                      prec::Dtype dt = prec::Dtype::F64)
    {
        RopeTable rt;
        const int64_t dim = c.head_dim, half = dim / 2;
        rt.dim = dim;
        rt.cos.resize(size_t(T * half));
        rt.sin.resize(size_t(T * half));

        const bool f32_chain = f32_compat || dt != prec::Dtype::F64;
        if (f32_chain)
        {
            std::vector<float> inv32(static_cast<size_t>(half));
            for (int64_t j = 0; j < half; ++j)
            {
                float e = float(2 * j) / float(dim);
                float pw = float(fmath::pow(double(float(c.rope_theta)), double(e)));
                inv32[size_t(j)] = 1.0f / pw;
            }
            for (int64_t t = 0; t < T; ++t)
                for (int64_t j = 0; j < half; ++j)
                {
                    float ang = float(t) * inv32[size_t(j)];
                    double cv = fmath::cos(double(ang)), sv = fmath::sin(double(ang));
                    if (dt != prec::Dtype::F64)
                    {
                        rt.cos[size_t(t * half + j)] = prec::round_act(dt, cv);
                        rt.sin[size_t(t * half + j)] = prec::round_act(dt, sv);
                    }
                    else
                    {
                        rt.cos[size_t(t * half + j)] = double(float(cv));
                        rt.sin[size_t(t * half + j)] = double(float(sv));
                    }
                }
            return rt;
        }

        std::vector<double> inv_freq(static_cast<size_t>(half));
        for (int64_t j = 0; j < half; ++j)
            inv_freq[size_t(j)] = 1.0 / fmath::pow(c.rope_theta, double(2 * j) / double(dim));
        for (int64_t t = 0; t < T; ++t)
            for (int64_t j = 0; j < half; ++j)
            {
                double ang = double(t) * inv_freq[size_t(j)];
                rt.cos[size_t(t * half + j)] = fmath::cos(ang);
                rt.sin[size_t(t * half + j)] = fmath::sin(ang);
            }
        return rt;
    }

    // apply_rotary_pos_emb with rotate_half over the FULL head_dim:
    //   out[j]      = x[j]*cos[j] - x[j+half]*sin[j]
    //   out[j+half] = x[j+half]*cos[j] + x[j]*sin[j]
    // dt != F64: torch materializes (x*cos), (rotate_half(x)*sin) and their sum
    // as three separate tensor ops — three roundings per element.
    inline void apply_rope(double *x, const RopeTable &rt, int64_t pos,
                           prec::Dtype dt = prec::Dtype::F64)
    {
        const int64_t half = rt.dim / 2;
        const double *co = &rt.cos[size_t(pos * half)];
        const double *si = &rt.sin[size_t(pos * half)];
        if (dt != prec::Dtype::F64)
        {
            for (int64_t j = 0; j < half; ++j)
            {
                double lo = x[j], hi = x[j + half];
                x[j] = prec::round_act(dt, prec::round_act(dt, lo * co[j]) +
                                               prec::round_act(dt, -hi * si[j]));
                x[j + half] = prec::round_act(dt, prec::round_act(dt, hi * co[j]) +
                                                      prec::round_act(dt, lo * si[j]));
            }
            return;
        }
        for (int64_t j = 0; j < half; ++j)
        {
            double lo = x[j], hi = x[j + half];
            x[j] = lo * co[j] - hi * si[j];
            x[j + half] = hi * co[j] + lo * si[j];
        }
    }

    // ---------------------------------------------------------------- forward

    struct ForwardHooks
    {
        // index convention matches HF output_hidden_states: 0 = embedding,
        // i+1 = output of decoder layer i (i < L-1), L = final-norm output.
        virtual void on_hidden(int64_t /*index*/, const double * /*data*/, int64_t /*width*/) {}
        // op-level trace: every named intermediate of layer `layer`, for
        // localizing a disagreement to a single op rather than a whole layer.
        virtual void on_trace(int64_t /*layer*/, const char * /*name*/,
                              const double * /*data*/, int64_t /*n*/) {}
        virtual bool tracing() const { return false; }
        virtual ~ForwardHooks() = default;
    };

    struct ForwardOptions
    {
        ForwardHooks *hooks = nullptr;
        // Replicate stock HF's hard-coded f32 sites (both RMSNorm kinds, the
        // weight-less embed/qk norms, the rope tables, the eager softmax)
        // instead of pure f64. Default: pure f64.
        bool hf_f32_compat = false;
        // Storage dtype of the executed model (bf16/f16 execution semantics).
        prec::Dtype dtype = prec::Dtype::F64;
        // dtype != F64 only: skip the eager materialization of attention
        // scores/probs (fused-kernel semantics), so a fused GPU kernel has a
        // legal target.
        bool attn_flash = false;
        // capture the per-layer hidden states DFlash taps (0-based layer
        // outputs); filled with [n_taps][T*H] when non-null.
        const std::vector<int64_t> *tap_layers = nullptr;
        std::vector<std::vector<double>> *taps = nullptr;
        // capture the raw (un-normed) embedding rows — what the DFlash drafter
        // consumes; MuseGlimmerTextNormedEmbedding cannot be folded into the
        // table because of this.
        std::vector<double> *raw_embedding = nullptr;
    };

    struct AttnScratch
    {
        std::vector<double> Q, K, Vv, O, Gate;
        std::vector<std::vector<double>> score;
    };

    // MuseGlimmerTextAttention on pre-normed input xn [T,H]; writes the o_proj
    // output to out [T,H]. Causal, batch 1, eager semantics.
    inline void attention_forward(const Config &c, const LayerWeights &lw, const double *xn,
                                  int64_t T, const RopeTable &rt, bool use_rope,
                                  bool sliding, AttnScratch &s, double *out, bool compat,
                                  prec::Dtype dt, bool attn_flash,
                                  ForwardHooks *tr = nullptr, int64_t li = -1)
    {
        const int64_t H = c.hidden_size, D = c.head_dim;
        const int64_t nq = c.num_attention_heads, nkv = c.num_key_value_heads;
        const int64_t groups = c.kv_groups();
        const bool lp = dt != prec::Dtype::F64;
        // head_dim ** -0.5 and qk_scale_factor are python-float scalars; in lp
        // mode torch's opmath makes them f32 values.
        const double scaling = lp ? double(float(1.0 / std::sqrt(double(D))))
                                  : 1.0 / std::sqrt(double(D));
        const double qk_scale = lp ? double(float(c.qk_scale_factor)) : c.qk_scale_factor;
        const int64_t window = sliding ? c.sliding_window : 0;

        s.Q.resize(size_t(T * nq * D));
        s.Gate.resize(size_t(T * nq * D));
        s.K.resize(size_t(T * nkv * D));
        s.Vv.resize(size_t(T * nkv * D));
        s.O.resize(size_t(T * nq * D));
#ifdef _OPENMP
        int max_threads = omp_get_max_threads();
#else
        int max_threads = 1;
#endif
        if (int(s.score.size()) < max_threads)
            s.score.assign(size_t(max_threads), std::vector<double>(size_t(T)));
        for (auto &v : s.score)
            if (int64_t(v.size()) < T)
                v.resize(size_t(T));

        gemm(*lw.q_proj, xn, s.Q.data(), T, H, nq * D, dt);
        gemm(*lw.k_proj, xn, s.K.data(), T, H, nkv * D, dt);
        gemm(*lw.v_proj, xn, s.Vv.data(), T, H, nkv * D, dt);
        // the output gate takes the SAME pre-attention normed input the q/k/v
        // projections see — not the attention output
        gemm(*lw.gate_proj, xn, s.Gate.data(), T, H, nq * D, dt);

        // weight-less QK-norm over head_dim, then q (only) scaled by 3.87
        rmsnorm_rows(NormKind::Weightless, s.Q.data(), nullptr, c.rms_norm_eps, s.Q.data(),
                     T * nq, D, compat, dt);
        rmsnorm_rows(NormKind::Weightless, s.K.data(), nullptr, c.rms_norm_eps, s.K.data(),
                     T * nkv, D, compat, dt);
#pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < T * nq * D; ++i)
            s.Q[size_t(i)] = lp ? prec::round_act(dt, s.Q[size_t(i)] * qk_scale)
                                : s.Q[size_t(i)] * qk_scale;

        if (use_rope)
        {
#pragma omp parallel for schedule(static) collapse(2)
            for (int64_t t = 0; t < T; ++t)
                for (int64_t h = 0; h < nq; ++h)
                    apply_rope(&s.Q[size_t((t * nq + h) * D)], rt, t, dt);
#pragma omp parallel for schedule(static) collapse(2)
            for (int64_t t = 0; t < T; ++t)
                for (int64_t g = 0; g < nkv; ++g)
                    apply_rope(&s.K[size_t((t * nkv + g) * D)], rt, t, dt);
        }
        if (tr && tr->tracing())
        {
            tr->on_trace(li, "q", s.Q.data(), T * nq * D);
            tr->on_trace(li, "k", s.K.data(), T * nkv * D);
            tr->on_trace(li, "v", s.Vv.data(), T * nkv * D);
            tr->on_trace(li, "attn_gate", s.Gate.data(), T * nq * D);
        }

        // eager attention: stock forces the softmax to f32 (compat); pure runs
        // it in f64. Masked keys are excluded exactly — the additive
        // finfo.min mask makes their probability exactly 0.
#pragma omp parallel for schedule(static) collapse(2)
        for (int64_t h = 0; h < nq; ++h)
        {
            for (int64_t tq = 0; tq < T; ++tq)
            {
#ifdef _OPENMP
                double *scores = s.score[size_t(omp_get_thread_num())].data();
#else
                double *scores = s.score[0].data();
#endif
                const int64_t g = h / groups;
                // sliding: kv_idx > q_idx - window AND kv_idx <= q_idx
                const int64_t lo = window ? std::max<int64_t>(0, tq - window + 1) : 0;
                const double *q = &s.Q[size_t((tq * nq + h) * D)];
                double mx = -HUGE_VAL;
                for (int64_t tk = lo; tk <= tq; ++tk)
                {
                    double sc = simd::dot8(q, &s.K[size_t((tk * nkv + g) * D)], D);
                    if (lp && !attn_flash) // eager: matmul out and *scaling materialize
                        sc = prec::round_act(dt, prec::round_act(dt, sc) * scaling);
                    else
                        sc = sc * scaling;
                    scores[tk] = sc;
                    mx = std::max(mx, sc);
                }
                if (compat)
                {
                    float m32 = -HUGE_VALF;
                    for (int64_t tk = lo; tk <= tq; ++tk)
                        m32 = std::max(m32, float(scores[tk]));
                    float sum32 = 0;
                    for (int64_t tk = lo; tk <= tq; ++tk)
                    {
                        float e = float(fmath::exp(double(float(scores[tk]) - m32)));
                        scores[tk] = e;
                        sum32 += e;
                    }
                    for (int64_t tk = lo; tk <= tq; ++tk)
                        scores[tk] = double(float(scores[tk]) / sum32);
                }
                else
                {
                    double sum = 0;
                    for (int64_t tk = lo; tk <= tq; ++tk)
                    {
                        scores[tk] = fmath::exp(scores[tk] - mx);
                        sum += scores[tk];
                    }
                    if (lp && !attn_flash) // P materializes at `.to(query.dtype)`
                        for (int64_t tk = lo; tk <= tq; ++tk)
                            scores[tk] = prec::round_act(dt, scores[tk] / sum);
                    else
                        for (int64_t tk = lo; tk <= tq; ++tk)
                            scores[tk] = scores[tk] / sum;
                }
                double *o = &s.O[size_t((tq * nq + h) * D)];
                for (int64_t d = 0; d < D; ++d)
                    o[d] = 0.0;
                for (int64_t tk = lo; tk <= tq; ++tk)
                {
                    const double p = scores[tk];
                    const double *v = &s.Vv[size_t((tk * nkv + g) * D)];
                    for (int64_t d = 0; d < D; ++d)
                        o[d] += p * v[d];
                }
                const double *gt = &s.Gate[size_t((tq * nq + h) * D)];
                if (lp)
                    for (int64_t d = 0; d < D; ++d)
                    {
                        const double pv = prec::round_act(dt, o[d]); // P@V materializes
                        const double sg = prec::round_act(dt, sigmoid(gt[d]));
                        o[d] = prec::round_act(dt, pv * sg);
                    }
                else
                    for (int64_t d = 0; d < D; ++d)
                        o[d] = o[d] * sigmoid(gt[d]);
            }
        }
        if (tr && tr->tracing())
            tr->on_trace(li, "attn_gated", s.O.data(), T * nq * D);
        gemm(*lw.o_proj, s.O.data(), out, T, nq * D, H, dt);
        if (tr && tr->tracing())
            tr->on_trace(li, "attn_out", out, T * H);
    }

    // logits * output_multiplier, then cap * tanh(logits / cap) — written in
    // the reference's op order (multiply, divide, tanh, multiply) because each
    // step is a separate materialization in the low-precision twin. Both
    // constants are python floats and take their f32 opmath value there.
    inline void apply_output_tail(const Config &c, double *logits, int64_t n,
                                  prec::Dtype dt)
    {
        const bool lp = dt != prec::Dtype::F64;
        const double mult = lp ? double(float(c.output_multiplier)) : c.output_multiplier;
        const double cap = lp ? double(float(c.final_logit_softcapping))
                              : c.final_logit_softcapping;
        auto rnd = [dt](double x)
        { return prec::round_act(dt, x); };
#pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < n; ++i)
        {
            double z = logits[i];
            if (mult != 1.0)
                z = lp ? rnd(z * mult) : z * mult;
            if (cap != 0.0)
            {
                z = lp ? rnd(z / cap) : z / cap;
                z = lp ? rnd(fmath::tanh(z)) : fmath::tanh(z);
                z = lp ? rnd(z * cap) : z * cap;
            }
            logits[i] = z;
        }
    }

    // MuseGlimmerForConditionalGeneration: input_ids -> logits [T, V].
    inline std::vector<double> forward(const Config &c, Weights &w,
                                       const std::vector<int64_t> &ids,
                                       const ForwardOptions &opt = {})
    {
        ForwardHooks *hooks = opt.hooks;
        const bool compat = opt.hf_f32_compat;
        const prec::Dtype dt = opt.dtype;
        const bool lp = dt != prec::Dtype::F64;
        if (compat && lp)
            throw std::runtime_error("--hf-f32-compat is an f64-mode option; not valid with --dtype");
        auto rnd = [dt](double x)
        { return prec::round_act(dt, x); };
        const int64_t T = int64_t(ids.size()), H = c.hidden_size;
        const int64_t V = c.vocab_size, L = c.num_hidden_layers, I = c.intermediate_size;
        if (T < 1)
            throw std::runtime_error("empty prompt");
        if (T > c.max_position_embeddings)
            throw std::runtime_error("prompt longer than max_position_embeddings");
        for (int64_t id : ids)
            if (id < 0 || id >= V)
                throw std::runtime_error("token id out of range: " + std::to_string(id));

        RopeTable rope = build_rope_table(c, T, compat, dt);

        // MuseGlimmerTextNormedEmbedding: table lookup, then a WEIGHT-LESS
        // RMSNorm. No learned scale and no sqrt(H) Gemma-style multiplier.
        std::vector<double> h(size_t(T * H));
        for (int64_t t = 0; t < T; ++t)
            load_vec(*w.embed, ids[size_t(t)] * H, H, &h[size_t(t * H)], dt);
        if (opt.raw_embedding) // DFlash consumes the un-normed rows
            opt.raw_embedding->assign(h.begin(), h.end());
        rmsnorm_rows(NormKind::Weightless, h.data(), nullptr, c.rms_norm_eps, h.data(), T, H,
                     compat, dt);
        if (hooks)
            hooks->on_hidden(0, h.data(), H);
        if (opt.taps && opt.tap_layers)
            opt.taps->assign(opt.tap_layers->size(), {});

        std::vector<double> xn(size_t(T * H)), mix(size_t(T * H)), m(size_t(T * H));
        std::vector<double> lnw(static_cast<size_t>(H)), G(size_t(T * I)), U(size_t(T * I));
        AttnScratch attn_scratch;

        for (int64_t li = 0; li < L; ++li)
        {
            const LayerWeights &lw = w.layers[size_t(li)];

            // ---- attention half of the sandwich
            load_vec(*lw.input_ln, 0, H, lnw.data(), dt);
            rmsnorm_rows(NormKind::Centered, h.data(), lnw.data(), c.rms_norm_eps, xn.data(),
                         T, H, compat, dt);
            if (hooks && hooks->tracing())
                hooks->on_trace(li, "input_ln", xn.data(), T * H);
            attention_forward(c, lw, xn.data(), T, rope, c.layer_has_rope(li),
                              c.layer_is_sliding(li), attn_scratch, mix.data(), compat, dt,
                              opt.attn_flash, hooks, li);
            load_vec(*lw.post_attn_ln, 0, H, lnw.data(), dt);
            rmsnorm_rows(NormKind::Centered, mix.data(), lnw.data(), c.post_norm_eps,
                         mix.data(), T, H, compat, dt);
            for (size_t j = 0; j < size_t(T * H); ++j)
                h[j] = rnd(h[j] + mix[j]);
            if (hooks && hooks->tracing())
                hooks->on_trace(li, "post_attn_resid", h.data(), T * H);

            // ---- MLP half of the sandwich
            load_vec(*lw.pre_ff_ln, 0, H, lnw.data(), dt);
            rmsnorm_rows(NormKind::Centered, h.data(), lnw.data(), c.rms_norm_eps, xn.data(),
                         T, H, compat, dt);
            if (hooks && hooks->tracing())
                hooks->on_trace(li, "pre_ff_ln", xn.data(), T * H);
            gemm(*lw.mlp_gate, xn.data(), G.data(), T, H, I, dt);
            gemm(*lw.mlp_up, xn.data(), U.data(), T, H, I, dt);
#pragma omp parallel for schedule(static)
            for (int64_t j = 0; j < T * I; ++j)
                G[size_t(j)] = lp ? rnd(rnd(silu(G[size_t(j)])) * U[size_t(j)])
                                  : silu(G[size_t(j)]) * U[size_t(j)];
            if (hooks && hooks->tracing())
                hooks->on_trace(li, "swiglu", G.data(), T * I);
            gemm(*lw.mlp_down, G.data(), m.data(), T, I, H, dt);
            if (hooks && hooks->tracing())
                hooks->on_trace(li, "mlp_out", m.data(), T * H);
            load_vec(*lw.post_ff_ln, 0, H, lnw.data(), dt);
            rmsnorm_rows(NormKind::Centered, m.data(), lnw.data(), c.post_norm_eps, m.data(),
                         T, H, compat, dt);
            for (size_t j = 0; j < size_t(T * H); ++j)
                h[j] = rnd(h[j] + m[j]);

            if (hooks && li < L - 1)
                hooks->on_hidden(li + 1, h.data(), H);
            if (opt.taps && opt.tap_layers)
                for (size_t k = 0; k < opt.tap_layers->size(); ++k)
                    if ((*opt.tap_layers)[k] == li)
                        (*opt.taps)[k].assign(h.begin(), h.end());
        }

        // final PLAIN norm + lm_head + the pre-scaled tanh softcap
        std::vector<double> fnw(static_cast<size_t>(H));
        load_vec(*w.final_norm, 0, H, fnw.data(), dt);
        rmsnorm_rows(NormKind::Plain, h.data(), fnw.data(), c.rms_norm_eps, h.data(), T, H,
                     compat, dt);
        if (hooks)
            hooks->on_hidden(L, h.data(), H);

        std::vector<double> logits(size_t(T * V));
        gemm(w.lm_head ? *w.lm_head : *w.embed, h.data(), logits.data(), T, H, V, dt);
        apply_output_tail(c, logits.data(), int64_t(logits.size()), dt);
        return logits;
    }

} // namespace muse
