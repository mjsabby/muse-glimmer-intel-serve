// muse-gpu — run Muse Glimmer on the Arc cards.
//
// Writes the same artifacts as `muse-oracle --exec bf16` (logits.bin as f64,
// meta.json) so tools/bf16_parity.py, py/diff_logits.py and the gate scripts
// compare the GPU against the CPU twin without knowing which produced which.
#include "gpu/gpu_engine.h"

#include "hf.hpp"
#include "dflash.hpp"
#include "vision.hpp"
#include "muse_glimmer.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <memory>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace
{
    std::vector<int64_t> parse_ids(const std::string &spec)
    {
        std::vector<int64_t> ids;
        std::string cur;
        for (char ch : spec + ",")
        {
            if (ch == ',' || ch == ' ')
            {
                if (!cur.empty())
                    ids.push_back(std::stoll(cur));
                cur.clear();
            }
            else
                cur += ch;
        }
        return ids;
    }

    void write_bin(const std::string &path, const double *data, size_t n)
    {
        std::ofstream f(path, std::ios::binary);
        f.write(reinterpret_cast<const char *>(data), std::streamsize(n * sizeof(double)));
    }

    void usage()
    {
        std::fprintf(stderr,
                     "usage: muse-gpu --model DIR --ids A,B,C [options]\n"
                     "  --ids-file FILE    read the id list from a file (long prompts)\n"
                     "  --out DIR          artifact directory (default out/gpu)\n"
                     "  --gpus N           physical cards to use (default 2)\n"
                     "  --shards N         tensor-parallel shards (default: one per card)\n"
                     "  --chunk N          prefill block width (default 512)\n"
                     "  --max-seq N        KV allocation ceiling (default prompt+decode+64)\n"
                     "  --decode N         greedy-decode N tokens after prefill\n"
                     "  --top K            print the last position's top-K\n"
                     "  --no-dnnl          hand-written GEMV everywhere (diagnostic)\n"
                     "  --flash-decode     split-K decode attention: needed past ~2K context\n"
                     "  --flash-prefill    matrix-engine prefill attention (faster, and a\n"
                     "                     LOOSER numerical contract - envelope-gated)\n"
                     "  --assistant DIR    DFlash drafter; runs one drafting round after prefill\n"
                     "  --draft-rounds N   greedy speculative decoding for N rounds\n"
                     "  --pixels FILE      f64 [N, patch_dim] pixel values (with --grid)\n"
                     "  --grid t,h,w[;..]  patch grid per image\n"
                     "  --vision gpu|cpu   where to run the tower. cpu keeps ~1.9 GiB/card of\n"
                     "                     tower weights out of VRAM (it is bitwise-gated there)\n"
                     "  --seal N           0 off, 1 log, 2 refuse post-load allocations\n"
                     "  --no-prewarm       skip the startup prewarm (diagnosis only: it is\n"
                     "                     what makes --seal a guarantee rather than a hope)\n"
                     "  --seal-probe       after prewarm+seal, sweep prompt lengths and decode\n"
                     "                     depths in ONE process and report free-VRAM drift.\n"
                     "                     Exits non-zero if the footprint moved at all\n"
                     "  --q8-assistant     Q8_0 for the drafter only (BF16 target + Q8 drafter)\n"
                     "  --q8               Q8_0 weight tier: half the weight VRAM, a separate\n"
                     "                     accuracy tier (gated vs the oracle, not the twin)\n"
                     "  --list-devices     enumerate the visible GPUs and exit\n"
                     "  --revision REV     HF revision when --model is a repo id\n");
    }
} // namespace

int main(int argc, char **argv)
{
    std::string model, ids_spec, out_dir = "out/gpu", revision = "main", assistant, pixels_path, grid_spec;
    int64_t chunk = 512, max_seq = 0, decode_n = 0;
    int gpus = 2, shards = 0, topk = 5;
    int64_t draft_rounds = 0;
    bool no_dnnl = false, flash_prefill = false, flash_decode = false;
    std::string vision_on = "gpu";
    bool q8 = false, q8_assistant = false;
    int seal = 2;
    bool prewarm = true, seal_probe = false;

    for (int i = 1; i < argc; ++i)
    {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc)
            {
                std::fprintf(stderr, "%s needs a value\n", a.c_str());
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--model")
            model = next();
        else if (a == "--ids")
            ids_spec = next();
        else if (a == "--ids-file")
        {
            // long-context prompts exceed the argv limit
            std::ifstream f(next());
            if (!f)
                throw std::runtime_error("cannot read --ids-file");
            std::ostringstream ss;
            ss << f.rdbuf();
            ids_spec = ss.str();
        }
        else if (a == "--out")
            out_dir = next();
        else if (a == "--revision")
            revision = next();
        else if (a == "--gpus")
            gpus = std::stoi(next());
        else if (a == "--shards")
            shards = std::stoi(next());
        else if (a == "--chunk")
            chunk = std::stoll(next());
        else if (a == "--max-seq")
            max_seq = std::stoll(next());
        else if (a == "--decode")
            decode_n = std::stoll(next());
        else if (a == "--top")
            topk = std::stoi(next());
        else if (a == "--no-dnnl")
            no_dnnl = true;
        else if (a == "--flash-prefill")
            flash_prefill = true;
        else if (a == "--flash-decode")
            flash_decode = true;
        else if (a == "--assistant")
            assistant = next();
        else if (a == "--pixels")
            pixels_path = next();
        else if (a == "--grid")
            grid_spec = next();
        else if (a == "--vision")
            vision_on = next();
        else if (a == "--seal")
            seal = std::stoi(next());
        else if (a == "--no-prewarm")
            prewarm = false;
        else if (a == "--seal-probe")
            seal_probe = true;
        else if (a == "--q8")
            q8 = true;
        else if (a == "--q8-assistant")
            q8_assistant = true;
        else if (a == "--draft-rounds")
            draft_rounds = std::stoll(next());
        else if (a == "--list-devices")
        {
            auto ds = muse::gpu::enumerate_devices();
            std::printf("%zu GPU(s)\n", ds.size());
            for (size_t k = 0; k < ds.size(); ++k)
                std::printf("  [%zu] %s  %.2f GiB  %d EUs\n", k, ds[k].name.c_str(),
                            double(ds[k].total_mem) / 1073741824.0, ds[k].compute_units);
            return 0;
        }
        else if (a == "-h" || a == "--help")
        {
            usage();
            return 0;
        }
        else
        {
            std::fprintf(stderr, "unknown argument: %s\n", a.c_str());
            usage();
            return 2;
        }
    }
    if (model.empty() || ids_spec.empty())
    {
        usage();
        return 2;
    }

    try
    {
        const std::string snap = hf::resolve_model(model, revision);
        hf::ModelFiles mf(snap);
        muse::Config cfg = muse::parse_config(*mf.config);
        muse::Weights w = muse::bind_weights(mf, cfg);

        std::vector<int64_t> ids = parse_ids(ids_spec);
        const int64_t T = int64_t(ids.size());
        if (max_seq <= 0)
            max_seq = T + decode_n + 64;
        // NOTE: the chunk is NOT clamped to this prompt's length. It sizes
        // the activation scratch and it is what prewarm warms, so clamping it
        // to whatever the first request happens to be would make the startup
        // guarantee describe a different engine from the one that then serves.
        chunk = std::max<int64_t>(1, chunk);

        // --assistant taps the target's hidden states at the drafter's
        // target_layer_ids on the way through, so the drafter has to be parsed
        // BEFORE the engine is built.
        std::unique_ptr<hf::ModelFiles> amf;
        muse::dflash::Config acfg;
        muse::dflash::Weights aw;
        if (!assistant.empty())
        {
            amf = std::make_unique<hf::ModelFiles>(hf::resolve_model(assistant, revision));
            acfg = muse::dflash::parse_config(*amf->config);
            aw = muse::dflash::bind_weights(*amf, acfg);
            for (int64_t l : acfg.target_layer_ids)
                if (l < 0 || l >= cfg.num_hidden_layers)
                    throw std::runtime_error("target_layer_id " + std::to_string(l) +
                                             " out of range");
        }

        muse::gpu::EngineOptions opt;
        opt.gpus = gpus;
        if (!assistant.empty())
        {
            opt.tap_layers = acfg.target_layer_ids;
            opt.spec_block = acfg.block_size;
            opt.tap_window = acfg.sliding_window;
        }
        opt.shards = shards;
        opt.block = chunk;
        opt.max_seq = max_seq;
        opt.no_dnnl = no_dnnl;
        opt.flash_prefill = flash_prefill;
        opt.flash_decode = flash_decode;
        opt.q8 = q8;
        opt.q8_assistant = q8_assistant;
        opt.verbose = true;

        auto eng = muse::gpu::Engine::create(cfg, w, opt);
        if (!assistant.empty())
            eng->bind_drafter(acfg, aw);

        // ---- vision: run the tower, then scatter its output at the
        //      image/video placeholder tokens
        if (!pixels_path.empty() != !grid_spec.empty())
            throw std::runtime_error("--pixels and --grid must be given together");
        if (!pixels_path.empty())
        {
            muse::vision::Config vcfg = muse::vision::parse_config(*mf.config);
            muse::vision::Weights vw = muse::vision::bind_weights(mf, vcfg, cfg);
            std::vector<muse::vision::Grid> grids;
            {
                std::string spec = grid_spec;
                for (auto &ch : spec)
                    if (ch == ',' || ch == ';')
                        ch = ' ';
                std::istringstream gs(spec);
                int64_t gt, gh, gw;
                while (gs >> gt >> gh >> gw)
                    grids.push_back({gt, gh, gw});
                if (grids.empty())
                    throw std::runtime_error("--grid must be t,h,w[;t,h,w...]");
            }
            int64_t npatch = 0;
            for (const auto &g : grids)
                npatch += g.tokens();
            std::vector<double> px(size_t(npatch * vcfg.patch_dim()));
            {
                std::ifstream f(pixels_path, std::ios::binary);
                if (!f)
                    throw std::runtime_error("cannot read " + pixels_path);
                f.seekg(0, std::ios::end);
                const std::streamoff have = f.tellg();
                const std::streamoff want = std::streamoff(px.size() * 8);
                if (have != want)
                    throw std::runtime_error(pixels_path + ": " + std::to_string(int64_t(have)) +
                                             " bytes, expected " + std::to_string(int64_t(want)));
                f.seekg(0);
                f.read(reinterpret_cast<char *>(px.data()), want);
            }
            std::fprintf(stderr,
                         "vision: %zu grid(s), %lld patches, tower %lld layers x %lld, "
                         "%lld merged tokens\n",
                         grids.size(), (long long)npatch, (long long)vcfg.num_hidden_layers,
                         (long long)vcfg.hidden_size,
                         (long long)(npatch / vcfg.merge_unit()));
            // The tower is ~1.9 GiB/card once sharded. Running it on the CPU
            // keeps that out of VRAM entirely, and the CPU path is the
            // bitwise-gated one — for a single image per request the tower is
            // not the latency that matters, so this is often the better trade.
            std::vector<float> feats;
            if (vision_on == "gpu")
                eng->bind_vision(vcfg, vw, npatch); // upload, not forward
            const auto vt0 = std::chrono::steady_clock::now();
            if (vision_on == "cpu")
            {
                muse::vision::Options vopt;
                vopt.dtype = prec::Dtype::BF16;
                const std::vector<double> vd =
                    muse::vision::forward(vcfg, vw, cfg, px.data(), grids, vopt);
                feats.assign(vd.begin(), vd.end());
            }
            else if (vision_on == "gpu")
                feats = eng->vision_features(px.data(), grids);
            else
                throw std::runtime_error("--vision must be gpu or cpu");
            std::fprintf(stderr, "  tower forward (%s): %.3f s\n", vision_on.c_str(),
                         std::chrono::duration<double>(std::chrono::steady_clock::now() - vt0)
                             .count());
            {
                std::vector<double> fd(feats.begin(), feats.end());
                std::filesystem::create_directories(out_dir);
                write_bin(out_dir + "/vision.bin", fd.data(), fd.size());
            }
            std::vector<int64_t> at;
            for (size_t i = 0; i < ids.size(); ++i)
                if (ids[i] == cfg.image_token_id || ids[i] == cfg.video_token_id)
                    at.push_back(int64_t(i));
            const int64_t M = npatch / vcfg.merge_unit();
            if (int64_t(at.size()) != M)
                throw std::runtime_error("placeholder count " + std::to_string(at.size()) +
                                         " != merged vision tokens " + std::to_string(M));
            eng->set_vision_embeds(feats, at);
        }

        // Drive every shape a request can produce while we are still
        // starting up, so the libraries' lazy allocations happen HERE, and
        // then refuse the rest. After this line a memory failure is a startup
        // failure, which is the only kind that is anyone's to act on.
        if (prewarm)
            eng->prewarm();
        eng->seal_allocs(seal);

        std::vector<float> lg(size_t(cfg.vocab_size), 0.f);

        // ---- the seal probe: does the sealed engine's footprint actually
        //      hold across the shapes a served workload produces?
        //
        // prewarm() claims it warmed every shape; --seal 2 turns a violation
        // into an abort. Neither statement is worth much untested, and the
        // test has to happen INSIDE one process — a fresh process reloads 55
        // GiB and re-warms everything, which is precisely what it would need
        // to not do to prove anything.
        if (seal_probe)
        {
            // Lengths chosen to straddle every width bucket boundary, since
            // that is what forward_block rounds on: one below, one on, one
            // above, plus the degenerate single token.
            std::vector<int64_t> lens = {1, 2, 3, 7, 63, 64, 65, 127, 128, 129, 255, 257, 511, 512, 513};
            int64_t worst = 0;
            // Two passes. The first one absorbs a fixed, one-time lump the
            // Level-Zero runtime takes on its own account the first time this
            // mix of kernels runs (measured at 320 KiB here, and identical
            // whether the prompt is 3 tokens or 129, which is what says it is
            // not per-shape growth). The second pass is the claim being
            // gated: a served engine's footprint does not move.
            std::vector<int64_t> before, mid;
            int64_t first_touch = 0;
            for (int pass = 0; pass < 2; ++pass)
            {
            if (pass == 0)
                before = eng->free_mem();
            else
                mid = eng->free_mem();
            for (int64_t L : lens)
            {
                if (L > max_seq - 4)
                    continue;
                std::vector<int64_t> p(static_cast<size_t>(L));
                for (int64_t i = 0; i < L; ++i)
                    p[size_t(i)] = ids[size_t(i % ids.size())];
                eng->reset_cache();
                eng->prefill(p, lg.data());
                for (int k = 0; k < 3; ++k)
                    eng->decode_step(int64_t(std::max_element(lg.begin(), lg.end()) - lg.begin()),
                                     lg.data());
                if (!assistant.empty())
                    eng->draft(eng->cache_len(),
                               int64_t(std::max_element(lg.begin(), lg.end()) - lg.begin()),
                               eng->cache_len());
                if (getenv("MUSE_SEAL_PROBE_TRACE"))
                {
                    const auto f = eng->free_mem();
                    std::fprintf(stderr, "[seal-probe] pass %d len %5lld: %lld\n", pass,
                                 (long long)L, (long long)((pass ? mid[0] : before[0]) - f[0]));
                }
            }
            }
            const auto after = eng->free_mem();
            for (size_t i = 0; i < after.size() && i < mid.size(); ++i)
            {
                const int64_t d = mid[i] - after[i];
                std::fprintf(stderr, "[seal-probe] card %zu: %lld bytes drift (steady state), "
                                     "%lld first-touch\n",
                             i, (long long)d, (long long)(before[i] - mid[i]));
                worst = std::max(worst, d < 0 ? -d : d);
                first_touch = std::max(first_touch, before[i] - mid[i]);
            }
            eng->reset_cache();
            // A fixed one-time lump is the runtime's; anything at the scale of
            // a shape-dependent workspace is ours and is a bug.
            if (first_touch > 4 * 1024 * 1024)
            {
                std::fprintf(stderr, "[seal-probe] FAIL: %lld bytes of first-touch growth is "
                                     "too large to be the runtime's own\n",
                             (long long)first_touch);
                return 3;
            }
            if (before.empty() || before[0] == 0)
                std::fprintf(stderr,
                             "[seal-probe] free memory unavailable; set ZES_ENABLE_SYSMAN=1 "
                             "for the drift half of this check\n");
            else if (worst != 0)
            {
                std::fprintf(stderr, "[seal-probe] FAIL: footprint moved by %lld bytes\n",
                             (long long)worst);
                return 3;
            }
            std::fprintf(stderr, "[seal-probe] %zu prompt lengths, %s\n", lens.size(),
                         "no post-seal allocation and no VRAM drift");
        }

        eng->prefill(ids, lg.data());

        muse::gpu::SpecResult sp;
        if (!assistant.empty() && draft_rounds > 0)
        {
            const int64_t anchor =
                int64_t(std::max_element(lg.begin(), lg.end()) - lg.begin());
            sp = eng->spec_decode(ids, anchor, draft_rounds);
            const double rate = sp.drafted ? double(sp.accepted) / double(sp.drafted) : 0.0;
            std::fprintf(stderr,
                         "spec: %lld rounds, drafted %lld, accepted %lld, accept_rate %.4f\n"
                         "  %zu tokens in %.3f s = %.2f tok/s (%.3f tokens/round)\n"
                         "  per round: draft %.1f ms, verify %.1f ms\n",
                         (long long)sp.rounds, (long long)sp.drafted, (long long)sp.accepted,
                         rate, sp.tokens.size(), sp.seconds,
                         double(sp.tokens.size()) / sp.seconds,
                         double(sp.tokens.size()) / double(sp.rounds),
                         1e3 * sp.draft_s / double(sp.rounds),
                         1e3 * sp.verify_s / double(sp.rounds));
            std::printf("spec accepted per round:");
            for (int m : sp.per_round)
                std::printf(" %d", m);
            std::printf("\nspec sequence:");
            for (int64_t t : sp.tokens)
                std::printf(" %lld", (long long)t);
            std::printf("\n");
        }

        muse::gpu::DraftResult dr;
        if (!assistant.empty() && draft_rounds == 0)
        {
            // the anchor is the target's own next token; the block sits at
            // absolute position T
            const int64_t anchor =
                int64_t(std::max_element(lg.begin(), lg.end()) - lg.begin());
            dr = eng->draft(T, anchor, T);
            std::printf("anchor %lld, drafted %zu tokens:", (long long)anchor,
                        dr.tokens.size());
            for (int64_t t : dr.tokens)
                std::printf(" %lld", (long long)t);
            std::printf("\n");
        }

        std::vector<int64_t> seq = ids;
        for (int64_t i = 0; i < decode_n; ++i)
        {
            const int64_t best =
                int64_t(std::max_element(lg.begin(), lg.end()) - lg.begin());
            seq.push_back(best);
            eng->decode_step(best, lg.data());
        }

        std::fprintf(stderr, "exec gpu:\n");
        eng->report_profile(stderr);
        // Post-run free VRAM, so a run can be compared against the
        // post-prewarm line above: any drift is a library allocating behind
        // the seal, which is the one failure mode the seal cannot see.
        {
            const auto fm = eng->free_mem();
            for (size_t i = 0; i < fm.size(); ++i)
                if (fm[i] > 0)
                    std::fprintf(stderr, "  card %zu free %.3f GiB (after the run)\n", i,
                                 double(fm[i]) / 1073741824.0);
        }

        std::filesystem::create_directories(out_dir);
        std::vector<double> lgd(size_t(cfg.vocab_size));
        for (int64_t i = 0; i < cfg.vocab_size; ++i)
            lgd[size_t(i)] = double(lg[size_t(i)]);
        write_bin(out_dir + "/logits.bin", lgd.data(), lgd.size());

        const auto &tm = eng->timings();
        std::ostringstream bm;
        bm << "{\n \"kind\": \"gpu_sycl\",\n \"T\": 1,\n \"V\": " << cfg.vocab_size
           << ",\n \"prompt_len\": " << T << ",\n \"decode\": " << decode_n
           << ",\n \"gpus\": " << gpus << ",\n \"shards\": " << (shards > 0 ? shards : gpus)
           << ",\n \"q8\": " << (q8 ? "true" : "false")
           << ",\n \"flash_prefill\": " << (flash_prefill ? "true" : "false")
           << ",\n \"chunk\": " << chunk
           << ",\n \"max_seq\": " << max_seq << ",\n \"prefill_tok_s\": "
           << (tm.prefill_s > 0 ? double(tm.prefill_tokens) / tm.prefill_s : 0.0)
           << ",\n \"decode_tok_s\": "
           << (tm.decode_s > 0 ? double(tm.decode_tokens) / tm.decode_s : 0.0)
           << ",\n \"ids\": [";
        for (size_t i = 0; i < ids.size(); ++i)
            bm << (i ? "," : "") << ids[i];
        bm << "],\n \"generated\": [";
        for (size_t i = ids.size(); i < seq.size(); ++i)
            bm << (i > ids.size() ? "," : "") << seq[i];
        bm << "],\n \"draft_tokens\": [";
        for (size_t i = 0; i < dr.tokens.size(); ++i)
            bm << (i ? "," : "") << dr.tokens[i];
        bm << "],\n \"draft_anchor\": " << dr.anchor << ",\n \"n_hidden\": 0\n}\n";
        std::ofstream(out_dir + "/meta.json") << bm.str();

        if (decode_n > 0)
        {
            std::printf("generated:");
            for (size_t i = ids.size(); i < seq.size(); ++i)
                std::printf(" %lld", (long long)seq[i]);
            std::printf("\n");
        }
        std::vector<int64_t> idx(size_t(cfg.vocab_size));
        for (int64_t i = 0; i < cfg.vocab_size; ++i)
            idx[size_t(i)] = i;
        topk = int(std::min<int64_t>(topk, cfg.vocab_size));
        std::partial_sort(idx.begin(), idx.begin() + topk, idx.end(),
                          [&](int64_t a, int64_t b) { return lg[size_t(a)] > lg[size_t(b)]; });
        std::printf("last-position top%d:", topk);
        for (int k = 0; k < topk; ++k)
            std::printf(" (%lld, %.6f)", (long long)idx[size_t(k)],
                        lg[size_t(idx[size_t(k)])]);
        std::printf("\n");
        return 0;
    }
    catch (const std::exception &e)
    {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
