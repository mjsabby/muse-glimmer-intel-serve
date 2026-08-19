// C ABI over the SYCL engine, for the Python serving frontend.
//
// The division of labour is the siblings': the engine owns the weights, the
// caches, and the forward pass; Python owns tokenization, chat rendering,
// sampling, grammars, protocols, and the speculative accept rule. Logits come
// back as f32 over the wire (808 KiB per step at this vocab, which at decode
// rates is nothing) and everything above them is easier to change in Python.
//
// Two rules this file exists to keep:
//
//   1. NOTHING here allocates device memory after muse_open() returns. The
//      engine prewarms every reachable shape and then arms the seal, so a
//      configuration that cannot serve fails during startup. Adding a call
//      that grows a device buffer per request would quietly undo that.
//   2. No exception crosses the boundary. Every entry point catches, stashes
//      the message in a thread-local, and returns a failure code; ctypes has
//      no way to see a C++ exception and the process would just abort.
#include "gpu/gpu_engine.h"

#include "dflash.hpp"
#include "hf.hpp"
#include "json_grammar.h"
#include "json_schema.h"
#include "muse_glimmer.hpp"
#include "vision.hpp"

#include <algorithm>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace
{
    thread_local std::string g_err;

    struct Ctx
    {
        std::unique_ptr<hf::ModelFiles> mf, amf;
        muse::Config cfg;
        muse::Weights w;
        muse::dflash::Config acfg;
        muse::dflash::Weights aw;
        muse::vision::Config vcfg;
        muse::vision::Weights vw;
        bool has_vision = false, vision_cpu = false;
        std::unique_ptr<muse::gpu::Engine> eng;
        std::vector<float> logits;      // one row, vocab wide
        std::vector<float> logits_all;  // spec_block rows
        std::vector<int64_t> scratch_ids;
    };

    // The vision placement enum, mirrored in serve/engine.py.
    enum VisionPlace
    {
        VISION_GPU = 0, // tower resident on the cards (fast, ~1.86 GiB/card)
        VISION_CPU = 1, // tower on the host (slow, and the bitwise-gated path)
        VISION_OFF = 2  // text only
    };
} // namespace

extern "C"
{

    // Kept as a struct rather than 18 positional arguments: ctypes gets the
    // field names, and adding an option later does not silently shift every
    // caller's arguments by one.
    struct MuseOpenOpts
    {
        const char *model;     // repo id or directory
        const char *revision;  // "main" unless pinned
        const char *assistant; // DFlash drafter repo/dir, or null
        int gpus;
        int shards;
        int64_t max_seq;
        int64_t chunk;
        int q8;
        int q8_assistant;
        int flash_prefill;
        int flash_decode;
        int vision_place; // VisionPlace
        int64_t max_patches;
        int prewarm;
        int seal;
        int verbose;
    };

    const char *muse_last_error(void)
    {
        return g_err.c_str();
    }

    void *muse_open(const MuseOpenOpts *o)
    {
        g_err.clear();
        try
        {
            auto c = std::make_unique<Ctx>();
            const std::string rev = (o->revision && *o->revision) ? o->revision : "main";
            c->mf = std::make_unique<hf::ModelFiles>(hf::resolve_model(o->model, rev));
            c->cfg = muse::parse_config(*c->mf->config);
            c->w = muse::bind_weights(*c->mf, c->cfg);

            muse::gpu::EngineOptions e;
            e.gpus = o->gpus;
            e.shards = o->shards;
            e.max_seq = o->max_seq;
            e.block = o->chunk;
            e.q8 = o->q8 != 0;
            e.q8_assistant = o->q8_assistant != 0;
            e.flash_prefill = o->flash_prefill != 0;
            e.flash_decode = o->flash_decode != 0;
            e.verbose = o->verbose != 0;

            const bool want_draft = o->assistant && *o->assistant;
            if (want_draft)
            {
                // The drafter's taps are captured on the way through the
                // target's forward pass, so its config has to be known before
                // the engine is built, not after.
                c->amf = std::make_unique<hf::ModelFiles>(hf::resolve_model(o->assistant, rev));
                c->acfg = muse::dflash::parse_config(*c->amf->config);
                c->aw = muse::dflash::bind_weights(*c->amf, c->acfg);
                for (int64_t l : c->acfg.target_layer_ids)
                    if (l < 0 || l >= c->cfg.num_hidden_layers)
                        throw std::runtime_error("drafter target_layer_id " + std::to_string(l) +
                                                 " out of range for this target");
                e.tap_layers = c->acfg.target_layer_ids;
                e.spec_block = c->acfg.block_size;
                e.tap_window = c->acfg.sliding_window;
            }

            c->eng = muse::gpu::Engine::create(c->cfg, c->w, e);
            if (want_draft)
                c->eng->bind_drafter(c->acfg, c->aw);

            if (o->vision_place != VISION_OFF)
            {
                c->vcfg = muse::vision::parse_config(*c->mf->config);
                c->vw = muse::vision::bind_weights(*c->mf, c->vcfg, c->cfg);
                c->has_vision = true;
                c->vision_cpu = o->vision_place == VISION_CPU;
                if (!c->vision_cpu)
                    c->eng->bind_vision(c->vcfg, c->vw, o->max_patches);
            }

            if (o->prewarm)
                c->eng->prewarm();
            c->eng->seal_allocs(o->seal);

            c->logits.assign(static_cast<size_t>(c->cfg.vocab_size), 0.f);
            const int64_t rows = want_draft ? c->acfg.block_size : 1;
            c->logits_all.assign(static_cast<size_t>(c->cfg.vocab_size * rows), 0.f);
            return c.release();
        }
        catch (const std::exception &ex)
        {
            g_err = ex.what();
            return nullptr;
        }
    }

    void muse_close(void *h)
    {
        delete static_cast<Ctx *>(h);
    }

// Every entry point below this line follows the same shape, so the shape is
// written once. `BODY` may return; a throw becomes `FAIL`.
#define MUSE_GUARD(FAIL, BODY)                                                                     \
    do                                                                                             \
    {                                                                                              \
        g_err.clear();                                                                             \
        try                                                                                        \
        {                                                                                          \
            BODY                                                                                   \
        }                                                                                          \
        catch (const std::exception &ex)                                                           \
        {                                                                                          \
            g_err = ex.what();                                                                     \
            return FAIL;                                                                           \
        }                                                                                          \
        catch (...)                                                                                \
        {                                                                                          \
            g_err = "unknown error";                                                               \
            return FAIL;                                                                           \
        }                                                                                          \
    } while (0)

    int64_t muse_vocab(void *h) { return static_cast<Ctx *>(h)->cfg.vocab_size; }
    int64_t muse_sliding_window(void *h) { return static_cast<Ctx *>(h)->eng->sliding_window(); }
    int64_t muse_spec_block(void *h) { return static_cast<Ctx *>(h)->eng->spec_block(); }
    int64_t muse_image_token(void *h) { return static_cast<Ctx *>(h)->cfg.image_token_id; }
    int64_t muse_video_token(void *h) { return static_cast<Ctx *>(h)->cfg.video_token_id; }
    int64_t muse_hidden_size(void *h) { return static_cast<Ctx *>(h)->cfg.hidden_size; }
    int64_t muse_patch_dim(void *h)
    {
        Ctx *c = static_cast<Ctx *>(h);
        return c->has_vision ? c->vcfg.patch_dim() : 0;
    }
    int64_t muse_merge_unit(void *h)
    {
        Ctx *c = static_cast<Ctx *>(h);
        return c->has_vision ? c->vcfg.merge_unit() : 0;
    }
    int muse_has_vision(void *h) { return static_cast<Ctx *>(h)->has_vision ? 1 : 0; }
    int64_t muse_cache_len(void *h) { return static_cast<Ctx *>(h)->eng->cache_len(); }
    void muse_reset(void *h) { static_cast<Ctx *>(h)->eng->reset_cache(); }

    int muse_set_cache_len(void *h, int64_t n)
    {
        MUSE_GUARD(-1, {
            static_cast<Ctx *>(h)->eng->set_cache_len(n);
            return 0;
        });
    }

    // Free VRAM per CARD, or 0 when the driver will not say (no SYSMAN).
    int64_t muse_free_mem(void *h, int dev)
    {
        MUSE_GUARD(0, {
            const auto v = static_cast<Ctx *>(h)->eng->free_mem();
            return (dev >= 0 && size_t(dev) < v.size()) ? v[size_t(dev)] : 0;
        });
    }

    // Append `n` ids at the current cache length and return the LAST row's
    // logits. `logits` must hold vocab floats; it is filled only when non-null.
    // Returns the new cache length, or -1.
    int64_t muse_forward(void *h, const int32_t *ids, int64_t n, float *logits)
    {
        MUSE_GUARD(-1, {
            Ctx *c = static_cast<Ctx *>(h);
            c->scratch_ids.assign(ids, ids + n);
            c->eng->append(c->scratch_ids, logits ? logits : c->logits.data());
            return c->eng->cache_len();
        });
    }

    // Verification pass: logits for EVERY row, `n * vocab` floats, row-major.
    // Leaves the cache holding all n; the caller rolls back to the accepted
    // prefix with muse_set_cache_len.
    int64_t muse_forward_all(void *h, const int32_t *ids, int64_t n, float *logits_all)
    {
        MUSE_GUARD(-1, {
            Ctx *c = static_cast<Ctx *>(h);
            c->scratch_ids.assign(ids, ids + n);
            c->eng->verify(c->scratch_ids, logits_all);
            return c->eng->cache_len();
        });
    }

    // One DFlash round against the taps the last forward captured: writes up to
    // `cap` proposals and returns how many. `anchor` is the token the target
    // just committed, sitting at absolute position cache_len().
    int64_t muse_draft(void *h, int32_t anchor, int32_t *out, int64_t cap)
    {
        MUSE_GUARD(-1, {
            Ctx *c = static_cast<Ctx *>(h);
            const int64_t n = c->eng->cache_len();
            const auto r = c->eng->draft(n, anchor, n);
            const int64_t m = std::min<int64_t>(cap, int64_t(r.tokens.size()));
            for (int64_t i = 0; i < m; ++i)
                out[i] = int32_t(r.tokens[size_t(i)]);
            return m;
        });
    }

    // Vision tower. `pixels` is f64 [npatch, patch_dim] exactly as the
    // checkpoint processor produces it; `grids` is ngrid triples (t, h, w).
    // Writes merged features, [npatch / merge_unit, hidden], and returns the
    // row count (or -1). Runs on the cards or on the host per --vision.
    int64_t muse_vision_features(void *h, const double *pixels, const int64_t *grids,
                                 int64_t ngrid, float *out, int64_t out_cap)
    {
        MUSE_GUARD(-1, {
            Ctx *c = static_cast<Ctx *>(h);
            if (!c->has_vision)
                throw std::runtime_error("this engine was opened without the vision tower");
            std::vector<muse::vision::Grid> g;
            int64_t npatch = 0;
            for (int64_t i = 0; i < ngrid; ++i)
            {
                g.push_back({grids[i * 3], grids[i * 3 + 1], grids[i * 3 + 2]});
                npatch += g.back().tokens();
            }
            const int64_t rows = npatch / c->vcfg.merge_unit();
            if (rows * c->cfg.hidden_size > out_cap)
                throw std::runtime_error("vision output buffer holds " + std::to_string(out_cap) +
                                         " floats, needs " +
                                         std::to_string(rows * c->cfg.hidden_size));
            std::vector<float> f;
            if (c->vision_cpu)
            {
                muse::vision::Options vopt;
                vopt.dtype = prec::Dtype::BF16;
                const std::vector<double> d =
                    muse::vision::forward(c->vcfg, c->vw, c->cfg, pixels, g, vopt);
                f.assign(d.begin(), d.end());
            }
            else
            {
                f = c->eng->vision_features(pixels, g);
            }
            std::memcpy(out, f.data(), f.size() * sizeof(float));
            return rows;
        });
    }

    // Bind `m` feature rows to `m` absolute token positions. They replace the
    // embedding at those positions on every subsequent forward, which is where
    // the reference puts them (after the embedding norm). m == 0 clears.
    int muse_set_vision_embeds(void *h, const float *feats, int64_t m, const int64_t *positions)
    {
        MUSE_GUARD(-1, {
            Ctx *c = static_cast<Ctx *>(h);
            const int64_t H = c->cfg.hidden_size;
            std::vector<float> f(feats, feats + m * H);
            std::vector<int64_t> p(positions, positions + m);
            c->eng->set_vision_embeds(f, p);
            return 0;
        });
    }

    // Wall-clock counters since open, for /metrics. Order:
    // upload_s, prefill_s, decode_s, prefill_tokens, decode_tokens.
    void muse_timings(void *h, double *out)
    {
        const auto &t = static_cast<Ctx *>(h)->eng->timings();
        out[0] = t.upload_s;
        out[1] = t.prefill_s;
        out[2] = t.decode_s;
        out[3] = double(t.prefill_tokens);
        out[4] = double(t.decode_tokens);
    }

    // ---------------------------------------------------------- grammars
    //
    // Guided decoding lives in C++ because the mask is swept over the vocab at
    // every step: 202048 candidates, ~0.3 ms here, and a Python loop would be
    // two orders of magnitude worse. Python compiles the schema text and
    // builds the piece table once; the automaton and the mask are ours.
    //
    // The piece table is per TOKENIZER, not per request — building it walks
    // the whole vocab — so it is opened once by the server and shared.

    void *muse_pieces_new(const uint8_t *bytes, const int64_t *offsets, const uint8_t *eog,
                          int64_t vocab)
    {
        MUSE_GUARD(nullptr, {
            auto p = std::make_unique<oracle::TokenPieces>();
            p->V = vocab;
            p->off.assign(offsets, offsets + vocab + 1);
            p->bytes.assign(reinterpret_cast<const char *>(bytes),
                            static_cast<size_t>(offsets[vocab]));
            p->eog.assign(eog, eog + vocab);
            p->build_index();
            return static_cast<void *>(p.release());
        });
    }
    void muse_pieces_free(void *p) { delete static_cast<oracle::TokenPieces *>(p); }

    void *muse_jsong_new(void *pieces, int object_only)
    {
        MUSE_GUARD(nullptr, {
            return static_cast<void *>(new oracle::JsonConstraint(*static_cast<oracle::TokenPieces *>(pieces),
                                                          object_only != 0));
        });
    }
    void muse_jsong_free(void *g) { delete static_cast<oracle::JsonConstraint *>(g); }
    void muse_jsong_mask(void *g, uint8_t *allowed)
    {
        static_cast<oracle::JsonConstraint *>(g)->mask(allowed);
    }
    void muse_jsong_accept(void *g, int32_t id)
    {
        static_cast<oracle::JsonConstraint *>(g)->accept(id);
    }
    int muse_jsong_complete(void *g)
    {
        return static_cast<oracle::JsonConstraint *>(g)->complete() ? 1 : 0;
    }

    void *muse_jsonschema_new(void *pieces, const char *schema_json)
    {
        MUSE_GUARD(nullptr, {
            return static_cast<void *>(
                new oracle::SchemaConstraint(*static_cast<oracle::TokenPieces *>(pieces), schema_json));
        });
    }
    void muse_jsonschema_free(void *g) { delete static_cast<oracle::SchemaConstraint *>(g); }
    void muse_jsonschema_mask(void *g, uint8_t *allowed)
    {
        static_cast<oracle::SchemaConstraint *>(g)->mask(allowed);
    }
    void muse_jsonschema_accept(void *g, int32_t id)
    {
        static_cast<oracle::SchemaConstraint *>(g)->accept(id);
    }
    int muse_jsonschema_complete(void *g)
    {
        return static_cast<oracle::SchemaConstraint *>(g)->complete() ? 1 : 0;
    }

#undef MUSE_GUARD

} // extern "C"
