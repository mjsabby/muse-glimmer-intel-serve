// MuseGlimmerAssistantModel — the DFlash block-diffusion drafter — in pure
// IEEE f64, mirroring transformers 5.15.0
// models/muse_glimmer_assistant/modeling_muse_glimmer_assistant.py and the
// drafting loop in generation/candidate_generator.py
// (DFlashTokenCandidateGenerator). See ARCHITECTURE.md §"DFlash block drafter".
//
// This is NOT EAGLE and NOT an in-checkpoint MTP head. Per round the drafter
// runs ONE forward over a block of `block_size` rows — an anchor token plus
// `block_size - 1` mask tokens — attending bidirectionally within the block and
// to a projected context built from the target's hidden states at
// `target_layer_ids`. Four things the module alone does not tell you, all read
// out of the generation utility and all load-bearing:
//
//   1. ONE pass per block. There is no iterative denoising loop.
//   2. A round proposes `block_size - 1` tokens, not `block_size`: the
//      candidate logits are `lm_head(h)[:, 1:]` — the anchor's own row is
//      dropped.
//   3. The head is the target's BARE `lm_head`: no `output_multiplier`, no
//      softcap. Monotone, so greedy drafting is unaffected; the sampled
//      distribution is not.
//   4. Acceptance is HF's ordinary assisted-decoding rule, so nothing
//      DFlash-specific is needed there — only block-shaped rollback.
//
// The drafter's norms are all PLAIN RMSNorms (weight around 1), including the
// ones whose names match the target's zero-centered sandwich norms. Mixing the
// two up is the same trap as in the target, one level down.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "muse_glimmer.hpp"

namespace muse
{
    namespace dflash
    {

        struct Config
        {
            int64_t hidden_size = 0, intermediate_size = 0, num_hidden_layers = 0;
            int64_t num_attention_heads = 0, num_key_value_heads = 0, head_dim = 0;
            int64_t sliding_window = 0, max_position_embeddings = 0;
            double rms_norm_eps = 1e-5, rope_theta = 0.0;
            int64_t block_size = 0, mask_token_id = -1;
            std::vector<int64_t> target_layer_ids;
            std::vector<std::string> layer_types;

            int64_t kv_groups() const { return num_attention_heads / num_key_value_heads; }
            int64_t q_dim() const { return num_attention_heads * head_dim; }
            int64_t kv_dim() const { return num_key_value_heads * head_dim; }
            bool layer_is_sliding(int64_t i) const
            {
                return layer_types.at(size_t(i)) == "sliding_attention";
            }
        };

        inline Config parse_config(const minijson::Value &root)
        {
            if (const auto *mt = root.opt("model_type"))
                if (mt->as_str() != "muse_glimmer_assistant")
                    throw std::runtime_error("not a DFlash drafter config: " + mt->as_str());

            auto geti = [&](const char *k, int64_t dflt, bool required = false) -> int64_t
            {
                const minijson::Value *v = root.opt(k);
                if (!v || v->is_null())
                {
                    if (required)
                        throw std::runtime_error(std::string("assistant config missing ") + k);
                    return dflt;
                }
                return v->as_int();
            };

            Config c;
            c.hidden_size = geti("hidden_size", 0, true);
            c.intermediate_size = geti("intermediate_size", 0, true);
            c.num_hidden_layers = geti("num_hidden_layers", 0, true);
            c.num_attention_heads = geti("num_attention_heads", 0, true);
            c.num_key_value_heads = geti("num_key_value_heads", 0, true);
            c.head_dim = geti("head_dim", 0, true);
            c.sliding_window = geti("sliding_window", 0, true);
            c.max_position_embeddings = geti("max_position_embeddings", 131072);
            c.block_size = geti("block_size", 0, true);
            c.mask_token_id = geti("mask_token_id", -1, true);
            if (const auto *v = root.opt("rms_norm_eps"))
                c.rms_norm_eps = v->as_double();
            if (const auto *rp = root.opt("rope_parameters"))
            {
                if (const auto *v = rp->opt("rope_type"))
                    if (v->as_str() != "default")
                        throw std::runtime_error("unsupported drafter rope_type: " + v->as_str());
                if (const auto *v = rp->opt("rope_theta"))
                    c.rope_theta = v->as_double();
            }
            const auto *tl = root.opt("target_layer_ids");
            if (!tl)
                throw std::runtime_error("assistant config missing target_layer_ids");
            for (size_t i = 0; i < tl->size(); ++i)
                c.target_layer_ids.push_back((*tl)[i].as_int());
            if (const auto *lt = root.opt("layer_types"))
                for (size_t i = 0; i < lt->size(); ++i)
                    c.layer_types.push_back((*lt)[i].as_str());
            else
                c.layer_types.assign(size_t(c.num_hidden_layers), "sliding_attention");
            if (int64_t(c.layer_types.size()) != c.num_hidden_layers)
                throw std::runtime_error("drafter layer_types length mismatch");
            for (const auto &t : c.layer_types)
                if (t != "sliding_attention" && t != "full_attention")
                    throw std::runtime_error("unsupported drafter layer type: " + t);
            if (c.num_attention_heads % c.num_key_value_heads)
                throw std::runtime_error("drafter heads not a multiple of kv heads");
            if (c.block_size < 2)
                throw std::runtime_error("block_size must be at least 2 (anchor + 1 mask)");
            return c;
        }

        struct LayerWeights
        {
            const st::Tensor *input_ln = nullptr, *post_attn_ln = nullptr; // PLAIN
            const st::Tensor *q_proj = nullptr, *k_proj = nullptr, *v_proj = nullptr,
                             *o_proj = nullptr;
            const st::Tensor *q_norm = nullptr, *k_norm = nullptr; // PLAIN, over head_dim
            const st::Tensor *mlp_gate = nullptr, *mlp_up = nullptr, *mlp_down = nullptr;
        };

        struct Weights
        {
            const st::Tensor *enc_fc = nullptr;   // [H, len(target_layer_ids) * H]
            const st::Tensor *enc_norm = nullptr; // [H], PLAIN
            const st::Tensor *final_norm = nullptr;
            std::vector<LayerWeights> layers;
        };

        inline Weights bind_weights(hf::ModelFiles &mf, const Config &c)
        {
            const int64_t H = c.hidden_size, D = c.head_dim, I = c.intermediate_size;
            Weights w;
            w.enc_fc = &mf.tensor("encoder.fc.weight");
            check_shape(*w.enc_fc, {H, int64_t(c.target_layer_ids.size()) * H});
            w.enc_norm = &mf.tensor("encoder.output_norm_enc.weight");
            check_shape(*w.enc_norm, {H});
            w.final_norm = &mf.tensor("norm.weight");
            check_shape(*w.final_norm, {H});
            for (int64_t i = 0; i < c.num_hidden_layers; ++i)
            {
                std::string lp = "layers." + std::to_string(i) + ".";
                auto T = [&](const std::string &n) -> const st::Tensor *
                { return &mf.tensor(lp + n); };
                LayerWeights lw{};
                lw.input_ln = T("input_layernorm.weight");
                lw.post_attn_ln = T("post_attention_layernorm.weight");
                lw.q_proj = T("self_attn.q_proj.weight");
                lw.k_proj = T("self_attn.k_proj.weight");
                lw.v_proj = T("self_attn.v_proj.weight");
                lw.o_proj = T("self_attn.o_proj.weight");
                lw.q_norm = T("self_attn.q_norm.weight");
                lw.k_norm = T("self_attn.k_norm.weight");
                lw.mlp_gate = T("mlp.gate_proj.weight");
                lw.mlp_up = T("mlp.up_proj.weight");
                lw.mlp_down = T("mlp.down_proj.weight");
                check_shape(*lw.input_ln, {H});
                check_shape(*lw.post_attn_ln, {H});
                check_shape(*lw.q_proj, {c.q_dim(), H});
                check_shape(*lw.k_proj, {c.kv_dim(), H});
                check_shape(*lw.v_proj, {c.kv_dim(), H});
                check_shape(*lw.o_proj, {H, c.q_dim()});
                check_shape(*lw.q_norm, {D});
                check_shape(*lw.k_norm, {D});
                check_shape(*lw.mlp_gate, {I, H});
                check_shape(*lw.mlp_up, {I, H});
                check_shape(*lw.mlp_down, {H, I});
                w.layers.push_back(lw);
            }
            return w;
        }

        // MuseGlimmerAssistantRMSNorm:
        //   y = w * ((x * rsqrt(mean(x^2) + eps)).to(input_dtype))
        // Note the rounding structure differs from the target's
        // MuseGlimmerRMSNorm, which rounds ONCE after the weight multiply: here
        // the normalized value is cast back to the storage dtype BEFORE being
        // scaled, so a low-precision run rounds twice.
        inline void drafter_rmsnorm_rows(const double *X, const double *w, double eps,
                                         double *Y, int64_t rows, int64_t dim,
                                         bool f32_compat, prec::Dtype dt)
        {
#pragma omp parallel for schedule(static)
            for (int64_t r = 0; r < rows; ++r)
            {
                const double *x = X + r * dim;
                double *y = Y + r * dim;
                if (f32_compat)
                {
                    float rs = 1.0f / std::sqrt(mean_sq_f32(x, dim) + float(eps));
                    for (int64_t i = 0; i < dim; ++i)
                        y[i] = double(float(w[i]) * (float(x[i]) * rs));
                    continue;
                }
                const double rs = 1.0 / std::sqrt(mean_sq(x, dim) + eps);
                for (int64_t i = 0; i < dim; ++i)
                    y[i] = prec::round_act(dt, w[i] * prec::round_act(dt, x[i] * rs));
            }
        }

        struct DraftResult
        {
            std::vector<int64_t> tokens;  // block_size - 1 proposed ids
            std::vector<double> logits;   // [block_size - 1, V], BARE lm_head
            std::vector<double> hidden;   // [block_size, H] after the drafter's final norm
            int64_t anchor = -1;          // the bonus token the block is anchored on
        };

        // One drafting round.
        //
        //   ctx_taps  [n_taps][n * H]  target hidden states at target_layer_ids,
        //                              for the n accepted context positions
        //   anchor    the target's bonus token (block row 0)
        //   pos0      absolute position of the anchor (== n when the context is
        //             a fresh prefill of n tokens)
        inline DraftResult draft(const Config &c, const Weights &w,
                                 const muse::Config &tc, const muse::Weights &tw,
                                 const std::vector<std::vector<double>> &ctx_taps,
                                 int64_t n, int64_t anchor, int64_t pos0,
                                 bool compat = false,
                                 prec::Dtype dt = prec::Dtype::F64)
        {
            const int64_t H = c.hidden_size, D = c.head_dim, I = c.intermediate_size;
            const int64_t nq = c.num_attention_heads, nkv = c.num_key_value_heads;
            const int64_t groups = c.kv_groups(), B = c.block_size;
            const int64_t taps = int64_t(c.target_layer_ids.size());
            const bool lp = dt != prec::Dtype::F64;
            if (int64_t(ctx_taps.size()) != taps)
                throw std::runtime_error("expected one hidden-state tap per target_layer_id");
            for (const auto &t : ctx_taps)
                if (int64_t(t.size()) != n * H)
                    throw std::runtime_error("context tap has the wrong length");
            if (H != tc.hidden_size)
                throw std::runtime_error("drafter/target hidden_size mismatch");
            auto rnd = [dt](double x)
            { return prec::round_act(dt, x); };

            // ---- context projection: concat the taps on the last dim, then
            //      encoder.fc + output_norm_enc. NOT re-normalized per layer.
            std::vector<double> cat(size_t(n * taps * H));
#pragma omp parallel for schedule(static)
            for (int64_t t = 0; t < n; ++t)
                for (int64_t k = 0; k < taps; ++k)
                    std::memcpy(&cat[size_t(t * taps * H + k * H)],
                                &ctx_taps[size_t(k)][size_t(t * H)], size_t(H) * 8);
            std::vector<double> ctx(size_t(n * H));
            gemm(*w.enc_fc, cat.data(), ctx.data(), n, taps * H, H, dt);
            {
                std::vector<double> nw(static_cast<size_t>(H));
                load_vec(*w.enc_norm, 0, H, nw.data(), dt);
                drafter_rmsnorm_rows(ctx.data(), nw.data(), c.rms_norm_eps, ctx.data(), n, H,
                                     compat, dt);
            }

            // ---- the noise block: [anchor, MASK x (B-1)], embedded with the
            //      target's RAW table (F.embedding on embed_tokens.weight —
            //      deliberately NOT MuseGlimmerTextNormedEmbedding's forward,
            //      which is why the norm is kept outside the table).
            std::vector<double> h(size_t(B * H));
            for (int64_t i = 0; i < B; ++i)
            {
                const int64_t id = i == 0 ? anchor : c.mask_token_id;
                if (id < 0 || id >= tc.vocab_size)
                    throw std::runtime_error("drafter token id out of range: " +
                                             std::to_string(id));
                load_vec(*tw.embed, id * H, H, &h[size_t(i * H)], dt);
            }

            // ---- rope table over the whole [context ++ block] position range.
            //      q takes the LAST B rows of cos/sin (the block sits at
            //      pos0 .. pos0+B-1); k takes all n+B.
            const int64_t P = pos0 + B;
            muse::Config ropecfg;
            ropecfg.head_dim = c.head_dim;
            ropecfg.rope_theta = c.rope_theta;
            ropecfg.max_position_embeddings = c.max_position_embeddings;
            RopeTable rope = build_rope_table(ropecfg, P, compat, dt);

            std::vector<double> xn(size_t(B * H)), kvin(size_t((n + B) * H));
            std::vector<double> Q(size_t(B * nq * D)), K(size_t((n + B) * nkv * D)),
                Vv(size_t((n + B) * nkv * D)), O(size_t(B * nq * D)), mix(size_t(B * H));
            std::vector<double> G(size_t(B * I)), U(size_t(B * I)), lnw(static_cast<size_t>(H)),
                nw(static_cast<size_t>(D));
            std::vector<double> scores(size_t(n + B));
            const double scaling = lp ? double(float(1.0 / std::sqrt(double(D))))
                                      : 1.0 / std::sqrt(double(D));

            for (int64_t li = 0; li < c.num_hidden_layers; ++li)
            {
                const LayerWeights &lw = w.layers[size_t(li)];
                const int64_t window = c.layer_is_sliding(li) ? c.sliding_window : 0;

                load_vec(*lw.input_ln, 0, H, lnw.data(), dt);
                drafter_rmsnorm_rows(h.data(), lnw.data(), c.rms_norm_eps, xn.data(), B, H,
                                     compat, dt);

                // k/v see [projected context ++ normed block]; q sees the block
                // only. The context is NOT passed through input_layernorm.
                std::memcpy(kvin.data(), ctx.data(), size_t(n * H) * 8);
                std::memcpy(&kvin[size_t(n * H)], xn.data(), size_t(B * H) * 8);
                gemm(*lw.q_proj, xn.data(), Q.data(), B, H, nq * D, dt);
                gemm(*lw.k_proj, kvin.data(), K.data(), n + B, H, nkv * D, dt);
                gemm(*lw.v_proj, kvin.data(), Vv.data(), n + B, H, nkv * D, dt);

                load_vec(*lw.q_norm, 0, D, nw.data(), dt);
                drafter_rmsnorm_rows(Q.data(), nw.data(), c.rms_norm_eps, Q.data(), B * nq, D,
                                     compat, dt);
                load_vec(*lw.k_norm, 0, D, nw.data(), dt);
                drafter_rmsnorm_rows(K.data(), nw.data(), c.rms_norm_eps, K.data(),
                                     (n + B) * nkv, D, compat, dt);

                for (int64_t i = 0; i < B; ++i)
                    for (int64_t hh = 0; hh < nq; ++hh)
                        apply_rope(&Q[size_t((i * nq + hh) * D)], rope, pos0 + i, dt);
                for (int64_t j = 0; j < n + B; ++j)
                    for (int64_t g = 0; g < nkv; ++g)
                        apply_rope(&K[size_t((j * nkv + g) * D)], rope, j, dt);

                // bidirectional attention: no causal constraint at all, only the
                // |q_pos - kv_pos| <= sliding_window overlay
                // (create_bidirectional_sliding_window_mask)
#pragma omp parallel for schedule(static) collapse(2)
                for (int64_t hh = 0; hh < nq; ++hh)
                {
                    for (int64_t i = 0; i < B; ++i)
                    {
                        std::vector<double> sc(size_t(n + B));
                        const int64_t g = hh / groups, qpos = pos0 + i;
                        const double *q = &Q[size_t((i * nq + hh) * D)];
                        int64_t lo = 0, hi = n + B - 1;
                        if (window)
                        {
                            lo = std::max<int64_t>(0, qpos - window);
                            hi = std::min<int64_t>(n + B - 1, qpos + window);
                        }
                        double mx = -HUGE_VAL;
                        for (int64_t j = lo; j <= hi; ++j)
                        {
                            double v = simd::dot8(q, &K[size_t((j * nkv + g) * D)], D);
                            v = lp ? prec::round_act(dt, prec::round_act(dt, v) * scaling)
                                   : v * scaling;
                            sc[size_t(j)] = v;
                            mx = std::max(mx, v);
                        }
                        double sum = 0;
                        for (int64_t j = lo; j <= hi; ++j)
                        {
                            sc[size_t(j)] = fmath::exp(sc[size_t(j)] - mx);
                            sum += sc[size_t(j)];
                        }
                        double *o = &O[size_t((i * nq + hh) * D)];
                        for (int64_t d = 0; d < D; ++d)
                            o[d] = 0.0;
                        for (int64_t j = lo; j <= hi; ++j)
                        {
                            const double p = lp ? prec::round_act(dt, sc[size_t(j)] / sum)
                                                : sc[size_t(j)] / sum;
                            const double *v = &Vv[size_t((j * nkv + g) * D)];
                            for (int64_t d = 0; d < D; ++d)
                                o[d] += p * v[d];
                        }
                        if (lp)
                            for (int64_t d = 0; d < D; ++d)
                                o[d] = prec::round_act(dt, o[d]);
                    }
                }
                gemm(*lw.o_proj, O.data(), mix.data(), B, nq * D, H, dt);
                for (size_t j = 0; j < size_t(B * H); ++j)
                    h[j] = rnd(h[j] + mix[j]);

                // ---- pre-norm SwiGLU block (no sandwich post-norms here)
                load_vec(*lw.post_attn_ln, 0, H, lnw.data(), dt);
                drafter_rmsnorm_rows(h.data(), lnw.data(), c.rms_norm_eps, xn.data(), B, H,
                                     compat, dt);
                gemm(*lw.mlp_gate, xn.data(), G.data(), B, H, I, dt);
                gemm(*lw.mlp_up, xn.data(), U.data(), B, H, I, dt);
                for (int64_t j = 0; j < B * I; ++j)
                    G[size_t(j)] = lp ? rnd(rnd(silu(G[size_t(j)])) * U[size_t(j)])
                                      : silu(G[size_t(j)]) * U[size_t(j)];
                gemm(*lw.mlp_down, G.data(), mix.data(), B, I, H, dt);
                for (size_t j = 0; j < size_t(B * H); ++j)
                    h[j] = rnd(h[j] + mix[j]);
            }

            std::vector<double> fnw(static_cast<size_t>(H));
            load_vec(*w.final_norm, 0, H, fnw.data(), dt);
            drafter_rmsnorm_rows(h.data(), fnw.data(), c.rms_norm_eps, h.data(), B, H, compat, dt);

            DraftResult out;
            out.anchor = anchor;
            out.hidden = h;
            // The BARE target head over rows 1..B-1: row 0 is the anchor's own
            // position and is dropped, so a round yields B-1 candidates.
            const int64_t nprop = B - 1, V = tc.vocab_size;
            out.logits.resize(size_t(nprop * V));
            gemm(tw.lm_head ? *tw.lm_head : *tw.embed, &h[size_t(H)], out.logits.data(), nprop,
                 H, V, dt);
            out.tokens.resize(size_t(nprop));
            for (int64_t i = 0; i < nprop; ++i)
            {
                const double *row = &out.logits[size_t(i * V)];
                int64_t best = 0;
                for (int64_t v = 1; v < V; ++v)
                    if (row[v] > row[best])
                        best = v;
                out.tokens[size_t(i)] = best;
            }
            return out;
        }

    } // namespace dflash
} // namespace muse
