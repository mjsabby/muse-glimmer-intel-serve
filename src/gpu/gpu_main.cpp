// muse-gpu — run Muse Glimmer on the Arc cards.
//
// Writes the same artifacts as `muse-oracle --exec bf16` (logits.bin as f64,
// meta.json) so tools/bf16_parity.py, py/diff_logits.py and the gate scripts
// compare the GPU against the CPU twin without knowing which produced which.
#include "gpu/gpu_engine.h"

#include "hf.hpp"
#include "muse_glimmer.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
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
                     "  --out DIR          artifact directory (default out/gpu)\n"
                     "  --gpus N           physical cards to use (default 2)\n"
                     "  --shards N         tensor-parallel shards (default: one per card)\n"
                     "  --chunk N          prefill block width (default 512)\n"
                     "  --max-seq N        KV allocation ceiling (default prompt+decode+64)\n"
                     "  --decode N         greedy-decode N tokens after prefill\n"
                     "  --top K            print the last position's top-K\n"
                     "  --no-dnnl          hand-written GEMV everywhere (diagnostic)\n"
                     "  --list-devices     enumerate the visible GPUs and exit\n"
                     "  --revision REV     HF revision when --model is a repo id\n");
    }
} // namespace

int main(int argc, char **argv)
{
    std::string model, ids_spec, out_dir = "out/gpu", revision = "main";
    int64_t chunk = 512, max_seq = 0, decode_n = 0;
    int gpus = 2, shards = 0, topk = 5;
    bool no_dnnl = false;

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
        chunk = std::min(chunk, std::max<int64_t>(1, T));

        muse::gpu::EngineOptions opt;
        opt.gpus = gpus;
        opt.shards = shards;
        opt.block = chunk;
        opt.max_seq = max_seq;
        opt.no_dnnl = no_dnnl;
        opt.verbose = true;

        auto eng = muse::gpu::Engine::create(cfg, w, opt);

        std::vector<float> lg(size_t(cfg.vocab_size), 0.f);
        eng->prefill(ids, lg.data());

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
        bm << "],\n \"n_hidden\": 0\n}\n";
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
