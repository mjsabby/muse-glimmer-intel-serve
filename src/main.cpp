// muse-oracle: float64 CPU reference ("oracle") forward for Muse Glimmer
// (model_type muse_glimmer / muse_glimmer_text). Input: token ids. Output:
// logits (f64 [T,V]) + optional per-layer hidden dumps, in the same dump format
// as py/ref_forward.py so py/diff_logits.py can compare.

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "bf16exec.hpp"
#include "dflash.hpp"
#include "hf.hpp"
#include "muse_glimmer.hpp"
#include "vision.hpp"

namespace
{

    std::vector<int64_t> parse_ids(const std::string &spec)
    {
        std::string text = spec;
        if (hf::is_file(spec))
            text = hf::read_file(spec);
        for (auto &ch : text)
            if (ch == ',' || ch == '\n' || ch == '\t' || ch == '\r')
                ch = ' ';
        std::vector<int64_t> ids;
        std::istringstream ss(text);
        int64_t v;
        while (ss >> v)
            ids.push_back(v);
        return ids;
    }

    void write_bin(const std::string &path, const double *data, size_t n)
    {
        std::ofstream f(path, std::ios::binary);
        if (!f)
            throw std::runtime_error("cannot write " + path);
        f.write(reinterpret_cast<const char *>(data), std::streamsize(n * 8));
        if (!f)
            throw std::runtime_error("short write to " + path);
    }

    struct DumpHooks final : muse::ForwardHooks
    {
        std::string dir, trace_dir;
        int64_t T = 0, count = 0;
        bool dump_hidden = false;
        void on_hidden(int64_t index, const double *data, int64_t width) override
        {
            if (!dump_hidden)
                return;
            char name[64];
            snprintf(name, sizeof name, "/hidden_%02lld.bin", (long long)index);
            write_bin(dir + name, data, size_t(T * width));
            count = std::max(count, index + 1);
        }
        bool tracing() const override { return !trace_dir.empty(); }
        void on_trace(int64_t layer, const char *name, const double *data, int64_t n) override
        {
            char fn[256];
            snprintf(fn, sizeof fn, "/L%02lld.%s.bin", (long long)layer, name);
            write_bin(trace_dir + fn, data, size_t(n));
        }
    };

    void usage()
    {
        fprintf(stderr,
                "usage: muse-oracle --model <dir|org/repo> --ids <list|file> --out <dir>\n"
                "                   [--revision main] [--dump-hidden] [--threads N] [--top K]\n"
                "                   [--trace-dir DIR]  (per-op intermediates, for localizing\n"
                "                                       a disagreement to a single op)\n"
                "                   [--hf-f32-compat]  (replicate the PLACEMENT of stock HF's\n"
                "                                       f32 casts; sizes their cost, but is not\n"
                "                                       a bitwise model of stock — VERIFICATION.md)\n"
                "                   [--dtype f64|bf16|f16]  (storage dtype; default f64)\n"
                "                   [--attn eager|flash]    (dtype!=f64: S/P materialization)\n"
                "                   [--kernels auto|scalar|avx512]\n"
                "                        force the scalar execution of the 8-lane fma\n"
                "                        reduction order; same bits, slower\n"
                "                   [--assistant <dir|org/repo>]  run one DFlash drafting\n"
                "                                       round after the target forward\n"
                "                   [--draft-rounds N]  greedy speculative loop, for the\n"
                "                                       acceptance-rate baseline\n"
                "                   [--exec f64|bf16]   f64 = the oracle; bf16 = the fast\n"
                "                        AVX-512 vdpbf16ps engine with a KV cache, refereed\n"
                "                        against --dtype bf16 --attn flash\n"
                "                   [--decode N] [--chunk N] [--max-seq N]\n"
                "                        --exec bf16 only: greedy decode after prefill, and\n"
                "                        the prefill/decode tok/s that go with it\n"
                "                   [--pixels FILE --grid t,h,w[;t,h,w...]]\n"
                "                        run the vision tower over f64 pixel_values\n"
                "                        [N, patch_dim] and scatter the result at the\n"
                "                        image/video placeholder tokens in --ids\n"
                "  --model     snapshot directory, or repo id resolved via the local HF cache\n"
                "  --ids       comma/space separated token ids, or a file containing them\n"
                "  --out       output dir: logits.bin (f64 [T,V]) + meta.json (+ hidden_XX.bin,\n"
                "              + draft/{logits.bin,meta.json} with --assistant)\n");
    }

} // namespace

int main(int argc, char **argv)
{
    std::string model, ids_spec, out_dir, revision = "main";
    std::string dtype_s = "f64", attn_s = "eager", kernels_s = "auto", trace_dir, assistant;
    std::string pixels_path, grid_spec, exec_s = "f64";
    int64_t decode_n = 0, chunk = 256, max_seq = 0;
    bool dump_hidden = false, hf_f32_compat = false;
    int threads = 0, topk = 5;
    int64_t draft_rounds = 0;
    for (int i = 1; i < argc; ++i)
    {
        std::string a = argv[i];
        auto next = [&]() -> std::string
        {
            if (i + 1 >= argc)
            {
                usage();
                exit(2);
            }
            return argv[++i];
        };
        if (a == "--model")
            model = next();
        else if (a == "--ids")
            ids_spec = next();
        else if (a == "--out")
            out_dir = next();
        else if (a == "--revision")
            revision = next();
        else if (a == "--dump-hidden")
            dump_hidden = true;
        else if (a == "--trace-dir")
            trace_dir = next();
        else if (a == "--assistant")
            assistant = next();
        else if (a == "--draft-rounds")
            draft_rounds = atoll(next().c_str());
        else if (a == "--exec")
            exec_s = next();
        else if (a == "--decode")
            decode_n = atoll(next().c_str());
        else if (a == "--chunk")
            chunk = atoll(next().c_str());
        else if (a == "--max-seq")
            max_seq = atoll(next().c_str());
        else if (a == "--pixels")
            pixels_path = next();
        else if (a == "--grid")
            grid_spec = next();
        else if (a == "--hf-f32-compat")
            hf_f32_compat = true;
        else if (a == "--threads")
            threads = atoi(next().c_str());
        else if (a == "--top")
            topk = atoi(next().c_str());
        else if (a == "--dtype")
            dtype_s = next();
        else if (a == "--attn")
            attn_s = next();
        else if (a == "--kernels")
            kernels_s = next();
        else
        {
            fprintf(stderr, "unknown arg: %s\n", a.c_str());
            usage();
            return 2;
        }
    }
    if (model.empty() || ids_spec.empty() || out_dir.empty())
    {
        usage();
        return 2;
    }
#ifdef _OPENMP
    if (threads > 0)
        omp_set_num_threads(threads);
#endif
    if (kernels_s == "scalar")
        simd::force_scalar() = true;
    else if (kernels_s == "avx512")
    {
        if (!simd::avx512_compiled())
        {
            fprintf(stderr, "--kernels avx512 requested but this binary was not built "
                            "with AVX-512 support\n");
            return 2;
        }
        simd::force_scalar() = false;
    }
    else if (kernels_s != "auto")
    {
        fprintf(stderr, "unknown --kernels: %s (want auto|scalar|avx512)\n", kernels_s.c_str());
        return 2;
    }

    try
    {
        auto t0 = std::chrono::steady_clock::now();
        std::string snap = hf::resolve_model(model, revision);
        hf::ModelFiles mf(snap);
        muse::Config cfg = muse::parse_config(*mf.config);
        muse::Weights w = muse::bind_weights(mf, cfg);
        auto t1 = std::chrono::steady_clock::now();

        std::vector<int64_t> ids = parse_ids(ids_spec);
        int64_t n_sliding = 0, n_rope = 0;
        for (int64_t i = 0; i < cfg.num_hidden_layers; ++i)
        {
            n_sliding += cfg.layer_is_sliding(i);
            n_rope += cfg.layer_has_rope(i);
        }
        fprintf(stderr,
                "model: %s\n  prefix '%s', %lld layers (%lld sliding + %lld global, "
                "%lld rotated), hidden %lld, heads %lld/%lld x %lld, vocab %lld, T=%zu\n",
                snap.c_str(), w.prefix.c_str(), (long long)cfg.num_hidden_layers,
                (long long)n_sliding, (long long)(cfg.num_hidden_layers - n_sliding),
                (long long)n_rope, (long long)cfg.hidden_size,
                (long long)cfg.num_attention_heads, (long long)cfg.num_key_value_heads,
                (long long)cfg.head_dim, (long long)cfg.vocab_size, ids.size());

        std::string mk = "mkdir -p '" + out_dir + "'";
        if (system(mk.c_str()) != 0)
            throw std::runtime_error("cannot create " + out_dir);

        DumpHooks hooks;
        hooks.dir = out_dir;
        hooks.T = int64_t(ids.size());
        hooks.dump_hidden = dump_hidden;
        hooks.trace_dir = trace_dir;
        if (!trace_dir.empty() && system(("mkdir -p '" + trace_dir + "'").c_str()) != 0)
            throw std::runtime_error("cannot create " + trace_dir);
        muse::ForwardOptions fopt;
        fopt.hooks = (dump_hidden || !trace_dir.empty()) ? &hooks : nullptr;
        fopt.hf_f32_compat = hf_f32_compat;
        fopt.dtype = prec::parse_dtype(dtype_s);
        if (attn_s == "flash")
            fopt.attn_flash = true;
        else if (attn_s != "eager")
            throw std::runtime_error("unknown --attn: " + attn_s + " (want eager|flash)");

        // --assistant taps the target's hidden states at the drafter's
        // target_layer_ids on the way through
        std::unique_ptr<hf::ModelFiles> amf;
        muse::dflash::Config acfg;
        muse::dflash::Weights aw;
        std::vector<std::vector<double>> taps;
        if (!assistant.empty())
        {
            std::string asnap = hf::resolve_model(assistant, revision);
            amf = std::make_unique<hf::ModelFiles>(asnap);
            acfg = muse::dflash::parse_config(*amf->config);
            aw = muse::dflash::bind_weights(*amf, acfg);
            fopt.tap_layers = &acfg.target_layer_ids;
            fopt.taps = &taps;
            fprintf(stderr,
                    "drafter: %s\n  %lld layers, heads %lld/%lld x %lld, block_size %lld "
                    "(=> %lld proposals/round), mask id %lld, taps at %s\n",
                    asnap.c_str(), (long long)acfg.num_hidden_layers,
                    (long long)acfg.num_attention_heads, (long long)acfg.num_key_value_heads,
                    (long long)acfg.head_dim, (long long)acfg.block_size,
                    (long long)(acfg.block_size - 1), (long long)acfg.mask_token_id,
                    [&]
                    {
                        static std::string s;
                        s.clear();
                        for (size_t i = 0; i < acfg.target_layer_ids.size(); ++i)
                            s += (i ? "," : "") + std::to_string(acfg.target_layer_ids[i]);
                        return s.c_str();
                    }());
            for (int64_t l : acfg.target_layer_ids)
                if (l < 0 || l >= cfg.num_hidden_layers)
                    throw std::runtime_error("target_layer_id " + std::to_string(l) +
                                             " out of range for a " +
                                             std::to_string(cfg.num_hidden_layers) +
                                             "-layer target");
        }

        // --pixels/--grid: run the vision tower and scatter its output at the
        // image/video placeholder tokens
        std::vector<double> vision_out;
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
                int64_t t, gh, gw;
                while (gs >> t >> gh >> gw)
                    grids.push_back({t, gh, gw});
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
                    throw std::runtime_error(
                        pixels_path + ": " + std::to_string(int64_t(have)) +
                        " bytes, expected " + std::to_string(int64_t(want)) + " (f64 [" +
                        std::to_string(npatch) + ", " + std::to_string(vcfg.patch_dim()) + "])");
                f.seekg(0);
                f.read(reinterpret_cast<char *>(px.data()), want);
                if (!f)
                    throw std::runtime_error("short read from " + pixels_path);
            }
            muse::vision::Options vopt;
            vopt.hf_f32_compat = hf_f32_compat;
            vopt.dtype = fopt.dtype;
            fprintf(stderr, "vision: %zu grid(s), %lld patches, tower %lld layers x %lld, "
                            "%lld merged tokens\n",
                    grids.size(), (long long)npatch, (long long)vcfg.num_hidden_layers,
                    (long long)vcfg.hidden_size,
                    (long long)(npatch / vcfg.merge_unit()));
            vision_out = muse::vision::forward(vcfg, vw, cfg, px.data(), grids, vopt);
            write_bin(out_dir + "/vision.bin", vision_out.data(), vision_out.size());
            int64_t placeholders = 0;
            for (int64_t id : ids)
                placeholders += (id == cfg.image_token_id || id == cfg.video_token_id);
            if (placeholders)
                fopt.vision_embeds = &vision_out;
            else
                fprintf(stderr, "vision: --ids has no image/video placeholder token "
                                "(%lld / %lld); wrote vision.bin only\n",
                        (long long)cfg.image_token_id, (long long)cfg.video_token_id);
        }

        if (exec_s == "bf16")
        {
            // The fast candidate engine. Not combinable with the oracle-only
            // options: it is a different implementation, refereed against the
            // twin rather than defining it.
            if (hf_f32_compat || fopt.dtype != prec::Dtype::F64 || !assistant.empty() ||
                !pixels_path.empty() || dump_hidden || !trace_dir.empty())
                throw std::runtime_error(
                    "--exec bf16 is a standalone engine; not combinable with --dtype / "
                    "--hf-f32-compat / --assistant / --pixels / --dump-hidden / --trace-dir");
            const int64_t T = int64_t(ids.size());
            const int64_t ms = max_seq > 0 ? max_seq : T + decode_n + 8;
            if (T + decode_n > ms)
                throw std::runtime_error("--max-seq smaller than prompt + --decode");

            muse::bf16::Engine eng;
            auto ti = std::chrono::steady_clock::now();
            eng.init(cfg, w, ms, std::max<int64_t>(1, chunk));
            auto ti2 = std::chrono::steady_clock::now();

            std::vector<float> lg(size_t(cfg.vocab_size));
            auto p0 = std::chrono::steady_clock::now();
            eng.prefill(ids, lg.data());
            auto p1 = std::chrono::steady_clock::now();

            std::vector<int64_t> seq = ids;
            auto d0 = p1, d1 = p1;
            if (decode_n > 0)
            {
                d0 = std::chrono::steady_clock::now();
                for (int64_t i = 0; i < decode_n; ++i)
                {
                    seq.push_back(eng.argmax(lg.data(), cfg.vocab_size));
                    eng.forward_block(seq, int64_t(seq.size()) - 1, 1, lg.data());
                }
                d1 = std::chrono::steady_clock::now();
            }

            eng.prof.report(stderr);
            auto secs = [](auto a, auto b)
            { return std::chrono::duration<double>(b - a).count(); };
            const double ps = secs(p0, p1), ds = secs(d0, d1);
            fprintf(stderr,
                    "exec bf16: cache %.2f MiB (max_seq %lld), alloc %.2f s\n"
                    "  prefill %lld tok in %.3f s = %.2f tok/s (chunk %lld)\n",
                    double(eng.kv.bytes()) / 1048576.0, (long long)ms, secs(ti, ti2),
                    (long long)T, ps, double(T) / ps, (long long)chunk);
            if (decode_n > 0)
                fprintf(stderr, "  decode  %lld tok in %.3f s = %.2f tok/s (%.1f ms/tok)\n",
                        (long long)decode_n, ds, double(decode_n) / ds,
                        1000.0 * ds / double(decode_n));

            std::vector<double> lgd(size_t(cfg.vocab_size));
            for (int64_t i = 0; i < cfg.vocab_size; ++i)
                lgd[size_t(i)] = double(lg[size_t(i)]);
            write_bin(out_dir + "/logits.bin", lgd.data(), lgd.size());
            std::ostringstream bm;
            bm << "{\n \"kind\": \"cpp_bf16exec\",\n \"T\": 1,\n \"V\": " << cfg.vocab_size
               << ",\n \"prompt_len\": " << T << ",\n \"decode\": " << decode_n
               << ",\n \"prefill_tok_s\": " << double(T) / ps
               << ",\n \"decode_tok_s\": " << (decode_n ? double(decode_n) / ds : 0.0)
               << ",\n \"chunk\": " << chunk << ",\n \"max_seq\": " << ms
               << ",\n \"kv_bytes\": " << eng.kv.bytes() << ",\n \"ids\": [";
            for (size_t i = 0; i < ids.size(); ++i)
                bm << (i ? "," : "") << ids[i];
            bm << "],\n \"generated\": [";
            for (size_t i = ids.size(); i < seq.size(); ++i)
                bm << (i > ids.size() ? "," : "") << seq[i];
            bm << "],\n \"n_hidden\": 0\n}\n";
            std::ofstream(out_dir + "/meta.json") << bm.str();
            if (decode_n > 0)
            {
                printf("generated:");
                for (size_t i = ids.size(); i < seq.size(); ++i)
                    printf(" %lld", (long long)seq[i]);
                printf("\n");
            }
            std::vector<int64_t> idx(static_cast<size_t>(cfg.vocab_size));
            for (int64_t i = 0; i < cfg.vocab_size; ++i)
                idx[size_t(i)] = i;
            std::partial_sort(idx.begin(), idx.begin() + topk, idx.end(),
                              [&](int64_t a, int64_t b)
                              { return lg[size_t(a)] > lg[size_t(b)]; });
            printf("last-position top%d:", topk);
            for (int k = 0; k < topk; ++k)
                printf(" (%lld, %.6f)", (long long)idx[size_t(k)], lg[size_t(idx[size_t(k)])]);
            printf("\n");
            return 0;
        }
        if (exec_s != "f64")
            throw std::runtime_error("unknown --exec: " + exec_s + " (want f64|bf16)");

        std::vector<double> logits = muse::forward(cfg, w, ids, fopt);
        auto t2 = std::chrono::steady_clock::now();

        const int64_t T = int64_t(ids.size()), V = cfg.vocab_size;
        write_bin(out_dir + "/logits.bin", logits.data(), size_t(T * V));

        std::ostringstream meta;
        meta << "{\n \"kind\": \"cpp_oracle\",\n \"model\": \"" << snap << "\",\n \"ids\": [";
        for (size_t i = 0; i < ids.size(); ++i)
            meta << (i ? "," : "") << ids[i];
        const char *mode = hf_f32_compat ? "hf_f32_compat"
                           : fopt.dtype != prec::Dtype::F64
                               ? (fopt.attn_flash ? "lp_flash" : "lp_eager")
                               : "pure_f64";
        meta << "],\n \"T\": " << T << ",\n \"V\": " << V << ",\n \"pure\": "
             << (hf_f32_compat ? "false" : "true") << ",\n \"dtype\": \""
             << prec::dtype_name(fopt.dtype) << "\",\n \"mode\": \"" << mode
             << "\",\n \"kernels\": \"" << (simd::force_scalar() ? "scalar" : "auto")
             << "\",\n \"n_hidden\": " << (dump_hidden ? hooks.count : 0) << "\n}\n";
        std::ofstream(out_dir + "/meta.json") << meta.str();

        if (!assistant.empty())
        {
            auto argmax_row = [&](const std::vector<double> &L, int64_t row)
            {
                const double *p = &L[size_t(row * V)];
                int64_t best = 0;
                for (int64_t v = 1; v < V; ++v)
                    if (p[v] > p[best])
                        best = v;
                return best;
            };

            // One drafting round, exactly as DFlashTokenCandidateGenerator does
            // it on its first call: the whole prompt is the accepted context,
            // the anchor is the target's own bonus token (greedy argmax at the
            // last position), and the block sits at positions T .. T+B-1.
            const int64_t anchor = argmax_row(logits, T - 1);

            muse::dflash::DraftResult dr = muse::dflash::draft(
                acfg, aw, cfg, w, taps, T, anchor, T, hf_f32_compat, fopt.dtype);

            std::string ddir = out_dir + "/draft";
            if (system(("mkdir -p '" + ddir + "'").c_str()) != 0)
                throw std::runtime_error("cannot create " + ddir);
            write_bin(ddir + "/logits.bin", dr.logits.data(), dr.logits.size());
            write_bin(ddir + "/hidden.bin", dr.hidden.data(), dr.hidden.size());
            std::ostringstream dm;
            dm << "{\n \"kind\": \"cpp_oracle_dflash\",\n \"ids\": [";
            for (size_t i = 0; i < ids.size(); ++i)
                dm << (i ? "," : "") << ids[i];
            dm << "],\n \"T\": " << (acfg.block_size - 1) << ",\n \"V\": " << V
               << ",\n \"block_size\": " << acfg.block_size << ",\n \"anchor\": " << anchor
               << ",\n \"context_len\": " << T << ",\n \"draft_tokens\": [";
            for (size_t i = 0; i < dr.tokens.size(); ++i)
                dm << (i ? "," : "") << dr.tokens[i];
            dm << "],\n \"n_hidden\": 0\n}\n";
            std::ofstream(ddir + "/meta.json") << dm.str();

            printf("anchor %lld, drafted %zu tokens:", (long long)anchor, dr.tokens.size());
            for (int64_t t : dr.tokens)
                printf(" %lld", (long long)t);
            printf("\n");

            // Greedy speculative loop, for the acceptance-rate baseline.
            //
            // The acceptance rule is HF's ordinary _assisted_decoding one: the
            // target's argmax at each verified position is compared with the
            // candidate, the first mismatch ends the run, and the target's own
            // argmax at that position is taken as a bonus token — so a round
            // always yields at least 1 token and at most block_size.
            //
            // The oracle has no KV cache (it is a prefill referee), so each
            // round re-runs the full target forward over the accepted prefix
            // plus the candidates. That is one target forward per round, the
            // same count the real engine does; only the cost per forward is
            // different.
            if (draft_rounds > 0)
            {
                std::vector<int64_t> seq = ids;
                std::vector<double> L = logits;
                std::vector<std::vector<double>> tp = taps;
                int64_t bonus = anchor, n_ctx = T;
                int64_t total_accepted = 0, total_drafted = 0, tokens_out = 0;
                std::vector<int64_t> per_round;

                for (int64_t r = 0; r < draft_rounds; ++r)
                {
                    muse::dflash::DraftResult d = muse::dflash::draft(
                        acfg, aw, cfg, w, tp, n_ctx, bonus, n_ctx, hf_f32_compat, fopt.dtype);
                    seq.push_back(bonus);
                    ++tokens_out;
                    const int64_t base = int64_t(seq.size()); // first candidate position
                    std::vector<int64_t> cand = seq;
                    cand.insert(cand.end(), d.tokens.begin(), d.tokens.end());

                    L = muse::forward(cfg, w, cand, fopt);
                    int64_t matches = 0;
                    while (matches < int64_t(d.tokens.size()) &&
                           argmax_row(L, base - 1 + matches) == cand[size_t(base + matches)])
                        ++matches;
                    total_drafted += int64_t(d.tokens.size());
                    total_accepted += matches;
                    per_round.push_back(matches);

                    seq.insert(seq.end(), d.tokens.begin(), d.tokens.begin() + matches);
                    tokens_out += matches;
                    bonus = argmax_row(L, base - 1 + matches);
                    n_ctx = int64_t(seq.size());
                    // The verification forward covered `cand`, which is at
                    // least as long as the accepted prefix. Attention is
                    // causal, so its hidden states at positions 0..n_ctx-1 are
                    // exactly the accepted prefix's — truncate rather than
                    // re-running the target.
                    tp.assign(taps.size(), {});
                    for (size_t k = 0; k < taps.size(); ++k)
                        tp[k].assign(taps[k].begin(),
                                     taps[k].begin() + size_t(n_ctx * cfg.hidden_size));
                }
                seq.push_back(bonus);

                const double rate = total_drafted ? double(total_accepted) / double(total_drafted)
                                                  : 0.0;
                printf("spec: rounds %lld, drafted %lld, accepted %lld, accept_rate %.4f, "
                       "tokens/round %.3f\n",
                       (long long)draft_rounds, (long long)total_drafted,
                       (long long)total_accepted, rate,
                       double(tokens_out) / double(draft_rounds));
                printf("spec: accepted per round:");
                for (int64_t m : per_round)
                    printf(" %lld", (long long)m);
                printf("\nspec: sequence:");
                for (size_t i = ids.size(); i < seq.size(); ++i)
                    printf(" %lld", (long long)seq[i]);
                printf("\n");

                std::ostringstream sm;
                sm << "{\n \"kind\": \"cpp_oracle_spec\",\n \"rounds\": " << draft_rounds
                   << ",\n \"block_size\": " << acfg.block_size << ",\n \"drafted\": "
                   << total_drafted << ",\n \"accepted\": " << total_accepted
                   << ",\n \"accept_rate\": " << rate << ",\n \"tokens_per_round\": "
                   << double(tokens_out) / double(draft_rounds) << ",\n \"per_round\": [";
                for (size_t i = 0; i < per_round.size(); ++i)
                    sm << (i ? "," : "") << per_round[i];
                sm << "],\n \"generated\": [";
                for (size_t i = ids.size(); i < seq.size(); ++i)
                    sm << (i > ids.size() ? "," : "") << seq[i];
                sm << "]\n}\n";
                std::ofstream(out_dir + "/draft/spec.json") << sm.str();
            }
        }

        // human sanity: top-k of the last position
        const double *last = &logits[size_t((T - 1) * V)];
        std::vector<int64_t> idx(static_cast<size_t>(V));
        for (int64_t i = 0; i < V; ++i)
            idx[size_t(i)] = i;
        std::partial_sort(idx.begin(), idx.begin() + topk, idx.end(),
                          [&](int64_t a, int64_t b)
                          { return last[a] > last[b]; });
        printf("last-position top%d:", topk);
        for (int k = 0; k < topk; ++k)
            printf(" (%lld, %.6f)", (long long)idx[size_t(k)], last[idx[size_t(k)]]);
        printf("\n");

        auto ms = [](auto a, auto b)
        {
            return std::chrono::duration_cast<std::chrono::milliseconds>(b - a).count();
        };
        fprintf(stderr, "timing: load %lld ms, forward %lld ms\n", (long long)ms(t0, t1),
                (long long)ms(t1, t2));
    }
    catch (const std::exception &e)
    {
        fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
