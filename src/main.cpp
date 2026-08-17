// muse-oracle: float64 CPU reference ("oracle") forward for Muse Glimmer
// (model_type muse_glimmer / muse_glimmer_text). Input: token ids. Output:
// logits (f64 [T,V]) + optional per-layer hidden dumps, in the same dump format
// as py/ref_forward.py so py/diff_logits.py can compare.

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "hf.hpp"
#include "muse_glimmer.hpp"

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
                "  --model     snapshot directory, or repo id resolved via the local HF cache\n"
                "  --ids       comma/space separated token ids, or a file containing them\n"
                "  --out       output dir: logits.bin (f64 [T,V]) + meta.json (+ hidden_XX.bin)\n");
    }

} // namespace

int main(int argc, char **argv)
{
    std::string model, ids_spec, out_dir, revision = "main";
    std::string dtype_s = "f64", attn_s = "eager", kernels_s = "auto", trace_dir;
    bool dump_hidden = false, hf_f32_compat = false;
    int threads = 0, topk = 5;
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
