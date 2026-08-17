// MuseGlimmerVisionModel + MuseGlimmerVisionAdapter + vision_projection in pure
// IEEE f64, mirroring transformers 5.15.0 modeling_muse_glimmer.py. See
// ARCHITECTURE.md §"Vision ops".
//
// A PE-style ViT-G/14 with four things that are easy to get backwards, each of
// which changes every vision logit while still producing a plausible-looking
// caption:
//
//   1. The 2-D RoPE position ids are **flipped to (w, h) and offset by +1**,
//      and the frequency vector is laid out cat[freq_w, freq_h, freq_w, freq_h]
//      over spatial_dim = head_dim/2 = 48.
//   2. The learned 32x32 position table is resampled with an EXPLICIT
//      four-corner bilinear gather whose taps and weights are built in f32 —
//      not `F.grid_sample`. Out-of-range corners are zero-WEIGHTED, not
//      clamped. The reference says so in a comment and deliberately differs
//      from the original implementation's grid_sample numerics.
//   3. Window attention reorders the tokens by `window_index` before the
//      layers and undoes it afterwards; `layer_types` decides per layer
//      whether the segment boundaries are the window ones or the whole frame.
//   4. `apply_rotary_pos_emb_vision` casts q and k to **f32** even in an f64
//      run — a downcast, not an upcast. The oracle lifts it; --hf-f32-compat
//      replicates it.
//
// The tower's activation is the exact erf gelu (ACT2FN["gelu"] is
// GELUActivation), not the tanh approximation.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <string>
#include <vector>

#include "muse_glimmer.hpp"

namespace muse
{
    namespace vision
    {

        struct Grid
        {
            int64_t t = 1, h = 0, w = 0;
            int64_t tokens() const { return t * h * w; }
        };

        struct Config
        {
            int64_t hidden_size = 0, intermediate_size = 0, num_hidden_layers = 0;
            int64_t num_attention_heads = 0;
            int64_t patch_size = 14, patch_temporal = 2, merge_size = 2;
            int64_t pos_emb_height = 32, pos_emb_width = 32;
            int64_t max_position_embeddings = 1024;
            double layer_norm_eps = 1e-5, rope_theta = 10000.0;
            std::vector<std::string> layer_types;
            // projector, from the TOP-level config
            int64_t out_hidden_size = 0, projector_hidden_size = 0;
            std::string hidden_act = "gelu", projector_hidden_act = "gelu";

            int64_t head_dim() const { return hidden_size / num_attention_heads; }
            int64_t spatial_dim() const { return head_dim() / 2; }
            int64_t patch_dim() const { return patch_temporal * 3 * patch_size * patch_size; }
            // MuseGlimmerVisionModel.forward: window_size = pos_emb_height * patch_size
            int64_t window_px() const { return pos_emb_height * patch_size; }
            int64_t merge_unit() const { return merge_size * merge_size; }
            bool layer_is_window(int64_t i) const
            {
                return layer_types.at(size_t(i)) == "window_attention";
            }
        };

        inline Config parse_config(const minijson::Value &root)
        {
            const minijson::Value *vcp = root.opt("vision_config");
            if (!vcp)
                throw std::runtime_error("config has no vision_config");
            const minijson::Value &vc = *vcp;

            auto geti = [&](const minijson::Value &o, const char *k, int64_t dflt,
                            bool required = false) -> int64_t
            {
                const minijson::Value *v = o.opt(k);
                if (!v || v->is_null())
                {
                    if (required)
                        throw std::runtime_error(std::string("vision config missing ") + k);
                    return dflt;
                }
                return v->as_int();
            };

            Config c;
            c.hidden_size = geti(vc, "hidden_size", 0, true);
            c.intermediate_size = geti(vc, "intermediate_size", 0, true);
            c.num_hidden_layers = geti(vc, "num_hidden_layers", 0, true);
            c.num_attention_heads = geti(vc, "num_attention_heads", 0, true);
            c.patch_size = geti(vc, "patch_size", 14);
            c.patch_temporal = geti(vc, "patch_temporal", 2);
            c.merge_size = geti(vc, "merge_size", 2);
            c.pos_emb_height = geti(vc, "pos_emb_height", 32);
            c.pos_emb_width = geti(vc, "pos_emb_width", 32);
            c.max_position_embeddings = geti(vc, "max_position_embeddings", 1024);
            if (const auto *v = vc.opt("layer_norm_eps"))
                c.layer_norm_eps = v->as_double();
            if (const auto *rp = vc.opt("rope_parameters"))
            {
                if (const auto *v = rp->opt("rope_type"))
                    if (v->as_str() != "default")
                        throw std::runtime_error("unsupported vision rope_type: " + v->as_str());
                if (const auto *v = rp->opt("rope_theta"))
                    c.rope_theta = v->as_double();
            }
            if (const auto *v = vc.opt("hidden_act"))
                c.hidden_act = v->as_str();
            if (c.hidden_act != "gelu")
                throw std::runtime_error("unsupported vision hidden_act: " + c.hidden_act +
                                         " (the oracle implements the exact erf gelu)");
            const auto *lt = vc.opt("layer_types");
            if (!lt)
                throw std::runtime_error("vision config missing layer_types");
            for (size_t i = 0; i < lt->size(); ++i)
            {
                std::string t = (*lt)[i].as_str();
                if (t != "window_attention" && t != "full_attention")
                    throw std::runtime_error("unsupported vision layer type: " + t);
                c.layer_types.push_back(t);
            }
            if (int64_t(c.layer_types.size()) != c.num_hidden_layers)
                throw std::runtime_error("vision layer_types length mismatch");
            // the tower ends on a FULL attention layer
            // (MuseGlimmerVisionConfig.__post_init__: `or i == num_hidden_layers - 1`)
            if (c.layer_is_window(c.num_hidden_layers - 1))
                throw std::runtime_error("the vision tower's last layer should be full_attention");

            c.out_hidden_size = geti(root, "out_hidden_size", 0, true);
            c.projector_hidden_size = geti(root, "projector_hidden_size", 0, true);
            if (const auto *v = root.opt("projector_hidden_act"))
                c.projector_hidden_act = v->as_str();
            if (c.projector_hidden_act != "gelu")
                throw std::runtime_error("unsupported projector_hidden_act: " +
                                         c.projector_hidden_act);
            if (c.out_hidden_size != c.hidden_size * c.merge_unit())
                throw std::runtime_error("out_hidden_size != vision hidden * merge_size^2");
            if (c.pos_emb_height != c.pos_emb_width)
                throw std::runtime_error("the reference assumes a square position grid");
            if (c.hidden_size % c.num_attention_heads)
                throw std::runtime_error("vision hidden_size not divisible by heads");
            if (c.head_dim() % 4)
                throw std::runtime_error("vision head_dim must be a multiple of 4 "
                                         "(cat[fw, fh, fw, fh] over head_dim/2)");
            return c;
        }

        struct LayerWeights
        {
            const st::Tensor *norm1_w = nullptr, *norm1_b = nullptr;
            const st::Tensor *norm2_w = nullptr, *norm2_b = nullptr;
            const st::Tensor *q_w = nullptr, *q_b = nullptr, *k_w = nullptr, *k_b = nullptr,
                             *v_w = nullptr, *v_b = nullptr, *o_w = nullptr, *o_b = nullptr;
            const st::Tensor *fc1_w = nullptr, *fc1_b = nullptr, *fc2_w = nullptr,
                             *fc2_b = nullptr;
        };

        struct Weights
        {
            const st::Tensor *patch_embed = nullptr; // [H, patch_dim], NO bias
            const st::Tensor *pos_table = nullptr;   // [pos_h * pos_w, H]
            const st::Tensor *ln_pre_w = nullptr, *ln_pre_b = nullptr;
            const st::Tensor *ln_post_w = nullptr, *ln_post_b = nullptr;
            std::vector<LayerWeights> layers;
            // projector (top-level module names, not under the tower)
            const st::Tensor *adapter_fc1 = nullptr, *adapter_fc2 = nullptr,
                             *projection = nullptr; // all bias-free
            std::string prefix;
        };

        inline Weights bind_weights(hf::ModelFiles &mf, const Config &c, const muse::Config &tc)
        {
            Weights w;
            for (const char *p : {"model.vision_tower.", "vision_tower.", ""})
                if (mf.has(std::string(p) + "patch_embedder.patch_embedding.weight"))
                {
                    w.prefix = p;
                    break;
                }
            if (w.prefix.empty() &&
                !mf.has("patch_embedder.patch_embedding.weight"))
                throw std::runtime_error("cannot find the vision tower under any known prefix");

            const int64_t H = c.hidden_size, I = c.intermediate_size;
            w.patch_embed = &mf.tensor(w.prefix + "patch_embedder.patch_embedding.weight");
            check_shape(*w.patch_embed, {H, c.patch_dim()});
            w.pos_table = &mf.tensor(w.prefix + "patch_embedder.position_embedding_table.weight");
            check_shape(*w.pos_table, {c.pos_emb_height * c.pos_emb_width, H});
            w.ln_pre_w = &mf.tensor(w.prefix + "ln_pre.weight");
            w.ln_pre_b = &mf.tensor(w.prefix + "ln_pre.bias");
            w.ln_post_w = &mf.tensor(w.prefix + "ln_post.weight");
            w.ln_post_b = &mf.tensor(w.prefix + "ln_post.bias");
            for (const st::Tensor *t : {w.ln_pre_w, w.ln_pre_b, w.ln_post_w, w.ln_post_b})
                check_shape(*t, {H});

            for (int64_t i = 0; i < c.num_hidden_layers; ++i)
            {
                std::string lp = w.prefix + "layers." + std::to_string(i) + ".";
                auto T = [&](const std::string &n) -> const st::Tensor *
                { return &mf.tensor(lp + n); };
                LayerWeights lw{};
                lw.norm1_w = T("norm1.weight");
                lw.norm1_b = T("norm1.bias");
                lw.norm2_w = T("norm2.weight");
                lw.norm2_b = T("norm2.bias");
                lw.q_w = T("attn.q_proj.weight");
                lw.q_b = T("attn.q_proj.bias");
                lw.k_w = T("attn.k_proj.weight");
                lw.k_b = T("attn.k_proj.bias");
                lw.v_w = T("attn.v_proj.weight");
                lw.v_b = T("attn.v_proj.bias");
                lw.o_w = T("attn.proj.weight");
                lw.o_b = T("attn.proj.bias");
                lw.fc1_w = T("mlp.fc1.weight");
                lw.fc1_b = T("mlp.fc1.bias");
                lw.fc2_w = T("mlp.fc2.weight");
                lw.fc2_b = T("mlp.fc2.bias");
                for (const st::Tensor *t : {lw.norm1_w, lw.norm1_b, lw.norm2_w, lw.norm2_b,
                                            lw.q_b, lw.k_b, lw.v_b, lw.o_b, lw.fc2_b})
                    check_shape(*t, {H});
                for (const st::Tensor *t : {lw.q_w, lw.k_w, lw.v_w, lw.o_w})
                    check_shape(*t, {H, H});
                check_shape(*lw.fc1_w, {I, H});
                check_shape(*lw.fc1_b, {I});
                check_shape(*lw.fc2_w, {H, I});
                w.layers.push_back(lw);
            }

            for (const char *p : {"model.", ""})
                if (mf.has(std::string(p) + "vision_adapter.fc1.weight"))
                {
                    w.adapter_fc1 = &mf.tensor(std::string(p) + "vision_adapter.fc1.weight");
                    w.adapter_fc2 = &mf.tensor(std::string(p) + "vision_adapter.fc2.weight");
                    w.projection = &mf.tensor(std::string(p) + "vision_projection.weight");
                    break;
                }
            if (!w.adapter_fc1)
                throw std::runtime_error("cannot find vision_adapter / vision_projection");
            check_shape(*w.adapter_fc1, {c.projector_hidden_size, c.out_hidden_size});
            check_shape(*w.adapter_fc2, {c.projector_hidden_size, c.projector_hidden_size});
            check_shape(*w.projection, {tc.hidden_size, c.projector_hidden_size});
            return w;
        }

        // ------------------------------------------------------------ helpers

        // nn.LayerNorm with bias, over the last dim. torch's f64 LayerNorm has
        // no hard-coded f32 cast, so there is no "compat" variant here; the
        // mean and the mean of squares use the oracle's blocked-8 order.
        inline void layernorm_rows(const double *X, const double *w, const double *b, double eps,
                                   double *Y, int64_t rows, int64_t dim, prec::Dtype dt)
        {
#pragma omp parallel for schedule(static)
            for (int64_t r = 0; r < rows; ++r)
            {
                const double *x = X + r * dim;
                double *y = Y + r * dim;
                double a0 = 0, a1 = 0, a2 = 0, a3 = 0, a4 = 0, a5 = 0, a6 = 0, a7 = 0;
                int64_t i = 0;
                for (; i + 8 <= dim; i += 8)
                {
                    a0 += x[i + 0];
                    a1 += x[i + 1];
                    a2 += x[i + 2];
                    a3 += x[i + 3];
                    a4 += x[i + 4];
                    a5 += x[i + 5];
                    a6 += x[i + 6];
                    a7 += x[i + 7];
                }
                double tail = 0;
                for (; i < dim; ++i)
                    tail += x[i];
                const double mean =
                    ((((a0 + a1) + (a2 + a3)) + ((a4 + a5) + (a6 + a7))) + tail) / double(dim);
                double b0 = 0, b1 = 0, b2 = 0, b3 = 0, b4 = 0, b5 = 0, b6 = 0, b7 = 0;
                i = 0;
                for (; i + 8 <= dim; i += 8)
                {
                    b0 += (x[i + 0] - mean) * (x[i + 0] - mean);
                    b1 += (x[i + 1] - mean) * (x[i + 1] - mean);
                    b2 += (x[i + 2] - mean) * (x[i + 2] - mean);
                    b3 += (x[i + 3] - mean) * (x[i + 3] - mean);
                    b4 += (x[i + 4] - mean) * (x[i + 4] - mean);
                    b5 += (x[i + 5] - mean) * (x[i + 5] - mean);
                    b6 += (x[i + 6] - mean) * (x[i + 6] - mean);
                    b7 += (x[i + 7] - mean) * (x[i + 7] - mean);
                }
                double vtail = 0;
                for (; i < dim; ++i)
                    vtail += (x[i] - mean) * (x[i] - mean);
                const double var =
                    ((((b0 + b1) + (b2 + b3)) + ((b4 + b5) + (b6 + b7))) + vtail) / double(dim);
                const double inv = 1.0 / std::sqrt(var + eps);
                for (int64_t j = 0; j < dim; ++j)
                    y[j] = prec::round_act(dt, (x[j] - mean) * inv * w[j] + b[j]);
            }
        }

        // Y[t,o] = sum_i W[o,i] * X[t,i] + b[o] — the tower's Linears all carry
        // a bias except patch_embedding and the projector.
        inline void gemm_bias(const st::Tensor &W, const st::Tensor *B, const double *X,
                              double *Y, int64_t T, int64_t in, int64_t out, prec::Dtype dt)
        {
            gemm(W, X, Y, T, in, out, dt);
            if (!B)
                return;
            std::vector<double> bv(static_cast<size_t>(out));
            load_vec(*B, 0, out, bv.data(), dt);
#pragma omp parallel for schedule(static)
            for (int64_t t = 0; t < T; ++t)
                for (int64_t o = 0; o < out; ++o)
                    Y[t * out + o] = prec::round_act(dt, Y[t * out + o] + bv[size_t(o)]);
        }

        // get_vision_position_ids(grid, spatial_merge_size=1) is the row-major
        // (h, w) meshgrid; the model then FLIPS it to (w, h) and adds 1.
        inline void position_ids(const std::vector<Grid> &grids, std::vector<int64_t> &wid,
                                 std::vector<int64_t> &hid)
        {
            wid.clear();
            hid.clear();
            for (const Grid &g : grids)
                for (int64_t f = 0; f < g.t; ++f)
                    for (int64_t y = 0; y < g.h; ++y)
                        for (int64_t x = 0; x < g.w; ++x)
                        {
                            wid.push_back(x + 1); // flip(-1) puts w first, then +1
                            hid.push_back(y + 1);
                        }
        }

        // get_vision_cu_seqlens(grid, merge_temporal=False): one segment per
        // frame, h*w tokens each.
        inline std::vector<int64_t> full_cu_seqlens(const std::vector<Grid> &grids)
        {
            std::vector<int64_t> cu{0};
            for (const Grid &g : grids)
                for (int64_t f = 0; f < g.t; ++f)
                    cu.push_back(cu.back() + g.h * g.w);
            return cu;
        }

        // get_vision_window_index(grid, spatial_merge_size=1, window_size,
        // patch_size). With spatial_merge_size == 1 the "llm grid" is the patch
        // grid and the window is `window_size / patch_size` patches square.
        //
        // Note the padding quirk inherited from the Qwen2.5-VL implementation:
        // `pad = win - (n % win)` is `win` (a whole extra, entirely-padding
        // window) when n is already a multiple of win. The padded entries are
        // dropped, so it only produces zero-length windows, which
        // `unique_consecutive` then collapses — but an implementation that
        // "fixes" the padding to 0 will produce a different window_index.
        inline void window_index(const Config &c, const std::vector<Grid> &grids,
                                 std::vector<int64_t> &index, std::vector<int64_t> &cu)
        {
            const int64_t win = c.window_px() / c.patch_size;
            index.clear();
            cu.assign(1, 0);
            int64_t base = 0;
            for (const Grid &g : grids)
            {
                const int64_t pad_h = win - g.h % win, pad_w = win - g.w % win;
                const int64_t nwh = (g.h + pad_h) / win, nww = (g.w + pad_w) / win;
                for (int64_t f = 0; f < g.t; ++f)
                    for (int64_t wy = 0; wy < nwh; ++wy)
                        for (int64_t wx = 0; wx < nww; ++wx)
                        {
                            int64_t n = 0;
                            for (int64_t iy = 0; iy < win; ++iy)
                                for (int64_t ix = 0; ix < win; ++ix)
                                {
                                    const int64_t y = wy * win + iy, x = wx * win + ix;
                                    if (y >= g.h || x >= g.w)
                                        continue; // padded entry, dropped
                                    index.push_back(base + f * g.h * g.w + y * g.w + x);
                                    ++n;
                                }
                            if (n) // unique_consecutive drops empty windows
                                cu.push_back(cu.back() + n);
                        }
                base += g.tokens();
            }
            return;
        }

        // get_vision_bilinear_indices_and_weights, replicated in f32.
        //
        // Deliberately f32: torch builds the sampling grid from
        // `torch.arange(h).float()`, and the FLOOR of that grid decides which
        // table entries are tapped. Lifting the grid to f64 would be a
        // structural change (a tap could move at a boundary), not a precision
        // one, so this stays in f32 in the oracle too and only the gather runs
        // in f64. `side / h` is a python float division, then demoted to f32
        // opmath by the tensor multiply.
        inline void pos_taps(const Config &c, const std::vector<Grid> &grids,
                             std::vector<std::array<int64_t, 4>> &idx,
                             std::vector<std::array<double, 4>> &wgt)
        {
            const int64_t side = c.pos_emb_height;
            idx.clear();
            wgt.clear();
            for (const Grid &g : grids)
            {
                std::vector<float> hf(size_t(g.h)), wf(size_t(g.w));
                std::vector<int64_t> hfl(size_t(g.h)), wfl(size_t(g.w));
                std::vector<bool> hfv(size_t(g.h)), hcv(size_t(g.h)), wfv(size_t(g.w)),
                    wcv(size_t(g.w));
                const float hs = float(double(side) / double(g.h));
                const float ws = float(double(side) / double(g.w));
                for (int64_t i = 0; i < g.h; ++i)
                {
                    const float v = (float(i) + 0.5f) * hs - 0.5f;
                    const float fl = std::floor(v);
                    hfl[size_t(i)] = int64_t(fl);
                    hf[size_t(i)] = v - fl;
                    hfv[size_t(i)] = (hfl[size_t(i)] >= 0 && hfl[size_t(i)] <= side - 1);
                    hcv[size_t(i)] = (hfl[size_t(i)] + 1 >= 0 && hfl[size_t(i)] + 1 <= side - 1);
                }
                for (int64_t i = 0; i < g.w; ++i)
                {
                    const float v = (float(i) + 0.5f) * ws - 0.5f;
                    const float fl = std::floor(v);
                    wfl[size_t(i)] = int64_t(fl);
                    wf[size_t(i)] = v - fl;
                    wfv[size_t(i)] = (wfl[size_t(i)] >= 0 && wfl[size_t(i)] <= side - 1);
                    wcv[size_t(i)] = (wfl[size_t(i)] + 1 >= 0 && wfl[size_t(i)] + 1 <= side - 1);
                }
                auto clamp = [&](int64_t v)
                { return std::min<int64_t>(std::max<int64_t>(v, 0), side - 1); };
                // per FRAME: the reorder tiles the h*w gather t times
                for (int64_t f = 0; f < g.t; ++f)
                    for (int64_t y = 0; y < g.h; ++y)
                        for (int64_t x = 0; x < g.w; ++x)
                        {
                            const int64_t hfc = clamp(hfl[size_t(y)]),
                                          hcc = clamp(hfl[size_t(y)] + 1),
                                          wfc = clamp(wfl[size_t(x)]),
                                          wcc = clamp(wfl[size_t(x)] + 1);
                            const float fh = hf[size_t(y)], fw = wf[size_t(x)];
                            idx.push_back({hfc * side + wfc, hfc * side + wcc,
                                           hcc * side + wfc, hcc * side + wcc});
                            wgt.push_back(
                                {double((1 - fh) * (1 - fw) *
                                        float(hfv[size_t(y)] && wfv[size_t(x)])),
                                 double((1 - fh) * fw * float(hfv[size_t(y)] && wcv[size_t(x)])),
                                 double(fh * (1 - fw) * float(hcv[size_t(y)] && wfv[size_t(x)])),
                                 double(fh * fw * float(hcv[size_t(y)] && wcv[size_t(x)]))});
                        }
            }
        }

        // pixel_shuffle: per frame, gather 2x2 spatial blocks and concatenate
        // their channels — [N, H] -> [N / merge_unit, H * merge_unit], with the
        // channel-major layout `permute(0, 2, 1)` produces.
        inline std::vector<double> pixel_shuffle(const Config &c, const std::vector<Grid> &grids,
                                                 const double *X)
        {
            const int64_t H = c.hidden_size, m = c.merge_size, mu = c.merge_unit();
            int64_t total_out = 0;
            for (const Grid &g : grids)
                total_out += g.t * (g.h / m) * (g.w / m);
            std::vector<double> out(size_t(total_out * H * mu));
            int64_t in_off = 0, out_row = 0;
            for (const Grid &g : grids)
            {
                for (int64_t f = 0; f < g.t; ++f)
                    for (int64_t by = 0; by < g.h / m; ++by)
                        for (int64_t bx = 0; bx < g.w / m; ++bx)
                        {
                            double *o = &out[size_t(out_row * H * mu)];
                            for (int64_t k = 0; k < mu; ++k)
                            {
                                const int64_t iy = by * m + k / m, ix = bx * m + k % m;
                                const double *src =
                                    X + size_t(in_off + f * g.h * g.w + iy * g.w + ix) * size_t(H);
                                // permute(0, 2, 1): channel-major, so channel d
                                // of sub-token k lands at d * mu + k
                                for (int64_t d = 0; d < H; ++d)
                                    o[d * mu + k] = src[d];
                            }
                            ++out_row;
                        }
                in_off += g.tokens();
            }
            return out;
        }

        struct Options
        {
            bool hf_f32_compat = false;
            prec::Dtype dtype = prec::Dtype::F64;
        };

        // pixel_values [N, patch_dim] -> [N / merge_unit, text hidden], ready to
        // be scattered into inputs_embeds at the image/video token positions.
        inline std::vector<double> forward(const Config &c, const Weights &w,
                                           const muse::Config &tc, const double *pixels,
                                           const std::vector<Grid> &grids,
                                           const Options &opt = {})
        {
            const int64_t H = c.hidden_size, I = c.intermediate_size, D = c.head_dim();
            const int64_t nh = c.num_attention_heads, half = D / 2, sd = c.spatial_dim();
            const bool lp = opt.dtype != prec::Dtype::F64;
            const prec::Dtype dt = opt.dtype;
            int64_t N = 0;
            for (const Grid &g : grids)
            {
                if (g.h % c.merge_size || g.w % c.merge_size)
                    throw std::runtime_error("grid dims must be multiples of merge_size");
                N += g.tokens();
            }
            if (N == 0)
                throw std::runtime_error("empty vision grid");

            // ---- patch embedding (no bias) + the resampled position table
            std::vector<double> x(size_t(N * H));
            gemm(*w.patch_embed, pixels, x.data(), N, c.patch_dim(), H, dt);
            {
                std::vector<std::array<int64_t, 4>> idx;
                std::vector<std::array<double, 4>> wgt;
                pos_taps(c, grids, idx, wgt);
                if (int64_t(idx.size()) != N)
                    throw std::runtime_error("position tap count != token count");
                std::vector<double> row(static_cast<size_t>(H));
#pragma omp parallel for schedule(static) firstprivate(row)
                for (int64_t t = 0; t < N; ++t)
                {
                    double *dst = &x[size_t(t * H)];
                    std::vector<double> acc(size_t(H), 0.0);
                    for (int k = 0; k < 4; ++k)
                    {
                        const double wk = wgt[size_t(t)][size_t(k)];
                        st::to_f64(*w.pos_table, idx[size_t(t)][size_t(k)] * H, H, row.data());
                        for (int64_t d = 0; d < H; ++d)
                            acc[size_t(d)] += row[size_t(d)] * wk;
                    }
                    for (int64_t d = 0; d < H; ++d)
                        dst[d] = prec::round_act(dt, dst[d] + acc[size_t(d)]);
                }
            }

            // ---- ln_pre, then the window reorder
            std::vector<double> lw(static_cast<size_t>(H)), lb(static_cast<size_t>(H));
            load_vec(*w.ln_pre_w, 0, H, lw.data(), dt);
            load_vec(*w.ln_pre_b, 0, H, lb.data(), dt);
            layernorm_rows(x.data(), lw.data(), lb.data(), c.layer_norm_eps, x.data(), N, H, dt);

            std::vector<int64_t> widx, wcu;
            window_index(c, grids, widx, wcu);
            if (int64_t(widx.size()) != N)
                throw std::runtime_error("window_index length != token count");
            std::vector<double> h(size_t(N * H));
            for (int64_t t = 0; t < N; ++t)
                std::memcpy(&h[size_t(t * H)], &x[size_t(widx[size_t(t)] * H)], size_t(H) * 8);

            // ---- 2-D rope over the REORDERED tokens: freq = cat[fw, fh, fw, fh]
            std::vector<int64_t> wid, hid;
            position_ids(grids, wid, hid);
            std::vector<double> cosv(size_t(N * D)), sinv(size_t(N * D));
            {
                std::vector<double> inv(size_t(sd / 2));
                std::vector<float> inv32(size_t(sd / 2));
                for (int64_t j = 0; j < sd / 2; ++j)
                {
                    inv[size_t(j)] =
                        1.0 / fmath::pow(c.rope_theta, double(2 * j) / double(sd));
                    inv32[size_t(j)] =
                        1.0f / float(fmath::pow(double(float(c.rope_theta)),
                                                double(float(2 * j) / float(sd))));
                }
                const bool f32_chain = opt.hf_f32_compat || lp;
                for (int64_t t = 0; t < N; ++t)
                {
                    const int64_t src = widx[size_t(t)];
                    const double pw = double(wid[size_t(src)]), ph = double(hid[size_t(src)]);
                    for (int64_t j = 0; j < D; ++j)
                    {
                        // [fw | fh | fw | fh], each block sd/2 = D/4 wide
                        const int64_t q = j / (sd / 2), r = j % (sd / 2);
                        const double p = (q == 0 || q == 2) ? pw : ph;
                        double ang;
                        if (f32_chain)
                            ang = double(float(p) * inv32[size_t(r)]);
                        else
                            ang = p * inv[size_t(r)];
                        double cv = fmath::cos(ang), sv = fmath::sin(ang);
                        if (lp)
                        {
                            cv = prec::round_act(dt, cv);
                            sv = prec::round_act(dt, sv);
                        }
                        else if (opt.hf_f32_compat)
                        {
                            cv = double(float(cv));
                            sv = double(float(sv));
                        }
                        cosv[size_t(t * D + j)] = cv;
                        sinv[size_t(t * D + j)] = sv;
                    }
                }
            }

            const std::vector<int64_t> fcu = full_cu_seqlens(grids);
            const double scaling = lp ? double(float(1.0 / std::sqrt(double(D))))
                                      : 1.0 / std::sqrt(double(D));

            std::vector<double> xn(size_t(N * H)), Q(size_t(N * H)), K(size_t(N * H)),
                Vv(size_t(N * H)), O(size_t(N * H)), mix(size_t(N * H)),
                F1(size_t(N * I));
            std::vector<int64_t> seg_of(size_t(N), 0);

            for (int64_t li = 0; li < c.num_hidden_layers; ++li)
            {
                const LayerWeights &l = w.layers[size_t(li)];
                const std::vector<int64_t> &cu = c.layer_is_window(li) ? wcu : fcu;

                load_vec(*l.norm1_w, 0, H, lw.data(), dt);
                load_vec(*l.norm1_b, 0, H, lb.data(), dt);
                layernorm_rows(h.data(), lw.data(), lb.data(), c.layer_norm_eps, xn.data(), N, H,
                               dt);
                gemm_bias(*l.q_w, l.q_b, xn.data(), Q.data(), N, H, H, dt);
                gemm_bias(*l.k_w, l.k_b, xn.data(), K.data(), N, H, H, dt);
                gemm_bias(*l.v_w, l.v_b, xn.data(), Vv.data(), N, H, H, dt);

                // apply_rotary_pos_emb_vision: rotate_half over the full
                // head_dim, with cos/sin shared by every head at that token.
#pragma omp parallel for schedule(static) collapse(2)
                for (int64_t t = 0; t < N; ++t)
                    for (int64_t hh = 0; hh < nh; ++hh)
                    {
                        const double *co = &cosv[size_t(t * D)], *si = &sinv[size_t(t * D)];
                        for (double *p : {&Q[size_t((t * nh + hh) * D)],
                                          &K[size_t((t * nh + hh) * D)]})
                        {
                            for (int64_t j = 0; j < half; ++j)
                            {
                                double lo = p[j], hi = p[j + half];
                                if (opt.hf_f32_compat)
                                { // stock DOWNCASTS q, k and cos/sin to f32 here
                                    lo = double(float(lo));
                                    hi = double(float(hi));
                                    p[j] = double(float(float(lo) * float(co[j]) -
                                                        float(hi) * float(si[j])));
                                    p[j + half] = double(float(float(hi) * float(co[j + half]) +
                                                               float(lo) * float(si[j + half])));
                                }
                                else if (lp)
                                {
                                    p[j] = prec::round_act(
                                        dt, prec::round_act(dt, lo * co[j]) +
                                                prec::round_act(dt, -hi * si[j]));
                                    p[j + half] = prec::round_act(
                                        dt, prec::round_act(dt, hi * co[j + half]) +
                                                prec::round_act(dt, lo * si[j + half]));
                                }
                                else
                                {
                                    p[j] = lo * co[j] - hi * si[j];
                                    p[j + half] = hi * co[j + half] + lo * si[j + half];
                                }
                            }
                        }
                    }

                // bidirectional attention inside each cu_seqlens segment.
                // Parallelism is over (query token, head) so a single full-
                // attention segment still saturates the machine; every output
                // element keeps its own fixed reduction order regardless.
                seg_of.assign(size_t(N), 0);
                for (int64_t s = 0; s + 1 < int64_t(cu.size()); ++s)
                    for (int64_t i = cu[size_t(s)]; i < cu[size_t(s + 1)]; ++i)
                        seg_of[size_t(i)] = s;
#pragma omp parallel for schedule(static) collapse(2)
                for (int64_t i = 0; i < N; ++i)
                {
                    for (int64_t hh = 0; hh < nh; ++hh)
                    {
                        const int64_t s = seg_of[size_t(i)];
                        const int64_t a = cu[size_t(s)], b = cu[size_t(s + 1)], n = b - a;
                        std::vector<double> sc(static_cast<size_t>(n));
                        {
                            const double *q = &Q[size_t((i * nh + hh) * D)];
                            double mx = -HUGE_VAL;
                            for (int64_t j = a; j < b; ++j)
                            {
                                double v =
                                    simd::dot8(q, &K[size_t((j * nh + hh) * D)], D);
                                v = lp ? prec::round_act(dt, prec::round_act(dt, v) * scaling)
                                       : v * scaling;
                                sc[size_t(j - a)] = v;
                                mx = std::max(mx, v);
                            }
                            double *o = &O[size_t((i * nh + hh) * D)];
                            for (int64_t d = 0; d < D; ++d)
                                o[d] = 0.0;
                            if (opt.hf_f32_compat)
                            { // stock softmax(dtype=float32)
                                float m32 = -HUGE_VALF, sum32 = 0;
                                for (int64_t j = 0; j < n; ++j)
                                    m32 = std::max(m32, float(sc[size_t(j)]));
                                for (int64_t j = 0; j < n; ++j)
                                {
                                    float e = float(fmath::exp(double(float(sc[size_t(j)]) - m32)));
                                    sc[size_t(j)] = e;
                                    sum32 += e;
                                }
                                for (int64_t j = 0; j < n; ++j)
                                    sc[size_t(j)] = double(float(sc[size_t(j)]) / sum32);
                            }
                            else
                            {
                                double sum = 0;
                                for (int64_t j = 0; j < n; ++j)
                                {
                                    sc[size_t(j)] = fmath::exp(sc[size_t(j)] - mx);
                                    sum += sc[size_t(j)];
                                }
                                for (int64_t j = 0; j < n; ++j)
                                    sc[size_t(j)] = lp ? prec::round_act(dt, sc[size_t(j)] / sum)
                                                       : sc[size_t(j)] / sum;
                            }
                            for (int64_t j = 0; j < n; ++j)
                            {
                                const double p = sc[size_t(j)];
                                const double *v = &Vv[size_t(((a + j) * nh + hh) * D)];
                                for (int64_t d = 0; d < D; ++d)
                                    o[d] += p * v[d];
                            }
                            if (lp)
                                for (int64_t d = 0; d < D; ++d)
                                    o[d] = prec::round_act(dt, o[d]);
                        }
                    }
                }
                gemm_bias(*l.o_w, l.o_b, O.data(), mix.data(), N, H, H, dt);
                for (size_t j = 0; j < size_t(N * H); ++j)
                    h[j] = prec::round_act(dt, h[j] + mix[j]);

                load_vec(*l.norm2_w, 0, H, lw.data(), dt);
                load_vec(*l.norm2_b, 0, H, lb.data(), dt);
                layernorm_rows(h.data(), lw.data(), lb.data(), c.layer_norm_eps, xn.data(), N, H,
                               dt);
                gemm_bias(*l.fc1_w, l.fc1_b, xn.data(), F1.data(), N, H, I, dt);
#pragma omp parallel for schedule(static)
                for (int64_t j = 0; j < N * I; ++j)
                    F1[size_t(j)] = prec::round_act(dt, fmath::gelu(F1[size_t(j)]));
                gemm_bias(*l.fc2_w, l.fc2_b, F1.data(), mix.data(), N, I, H, dt);
                for (size_t j = 0; j < size_t(N * H); ++j)
                    h[j] = prec::round_act(dt, h[j] + mix[j]);
            }

            // ---- undo the window reorder, ln_post, pixel shuffle
            for (int64_t t = 0; t < N; ++t)
                std::memcpy(&x[size_t(widx[size_t(t)] * H)], &h[size_t(t * H)], size_t(H) * 8);
            load_vec(*w.ln_post_w, 0, H, lw.data(), dt);
            load_vec(*w.ln_post_b, 0, H, lb.data(), dt);
            layernorm_rows(x.data(), lw.data(), lb.data(), c.layer_norm_eps, x.data(), N, H, dt);
            std::vector<double> merged = pixel_shuffle(c, grids, x.data());
            const int64_t M = int64_t(merged.size()) / c.out_hidden_size;

            // ---- adapter: gelu(fc2(gelu(fc1(y)))), then the projection, then
            //      the SAME weight-less RMSNorm the text embedding uses — which
            //      is why text and vision embeddings arrive on one scale.
            std::vector<double> a1(size_t(M * c.projector_hidden_size)),
                a2(size_t(M * c.projector_hidden_size));
            gemm(*w.adapter_fc1, merged.data(), a1.data(), M, c.out_hidden_size,
                 c.projector_hidden_size, dt);
#pragma omp parallel for schedule(static)
            for (int64_t j = 0; j < M * c.projector_hidden_size; ++j)
                a1[size_t(j)] = prec::round_act(dt, fmath::gelu(a1[size_t(j)]));
            gemm(*w.adapter_fc2, a1.data(), a2.data(), M, c.projector_hidden_size,
                 c.projector_hidden_size, dt);
#pragma omp parallel for schedule(static)
            for (int64_t j = 0; j < M * c.projector_hidden_size; ++j)
                a2[size_t(j)] = prec::round_act(dt, fmath::gelu(a2[size_t(j)]));
            std::vector<double> out(size_t(M * tc.hidden_size));
            gemm(*w.projection, a2.data(), out.data(), M, c.projector_hidden_size,
                 tc.hidden_size, dt);
            rmsnorm_rows(NormKind::Weightless, out.data(), nullptr, tc.rms_norm_eps, out.data(),
                         M, tc.hidden_size, opt.hf_f32_compat, dt);
            return out;
        }

    } // namespace vision
} // namespace muse
