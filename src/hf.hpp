// Local HuggingFace hub resolver: repo id (+revision) → snapshot dir in the local
// cache, and sharded-safetensors tensor lookup. Never touches the network.
#pragma once

#include <cstdlib>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <vector>

#include "json.hpp"
#include "safetensors.hpp"

namespace hf
{

    inline bool is_dir(const std::string &p)
    {
        struct stat sb{};
        return stat(p.c_str(), &sb) == 0 && S_ISDIR(sb.st_mode);
    }
    inline bool is_file(const std::string &p)
    {
        struct stat sb{};
        return stat(p.c_str(), &sb) == 0 && S_ISREG(sb.st_mode);
    }

    inline std::string read_file(const std::string &p)
    {
        std::ifstream f(p, std::ios::binary);
        if (!f)
            throw std::runtime_error("cannot read " + p);
        std::ostringstream ss;
        ss << f.rdbuf();
        return ss.str();
    }

    inline std::string strip(const std::string &s)
    {
        size_t a = s.find_first_not_of(" \t\r\n");
        if (a == std::string::npos)
            return "";
        size_t b = s.find_last_not_of(" \t\r\n");
        return s.substr(a, b - a + 1);
    }

    // $HF_HUB_CACHE > $HF_HOME/hub > $XDG_CACHE_HOME/huggingface/hub > ~/.cache/huggingface/hub
    inline std::string cache_root()
    {
        if (const char *v = getenv("HF_HUB_CACHE"); v && *v)
            return v;
        if (const char *v = getenv("HF_HOME"); v && *v)
            return std::string(v) + "/hub";
        if (const char *v = getenv("XDG_CACHE_HOME"); v && *v)
            return std::string(v) + "/huggingface/hub";
        const char *home = getenv("HOME");
        if (!home)
            throw std::runtime_error("HOME not set; cannot locate HF cache");
        return std::string(home) + "/.cache/huggingface/hub";
    }

    inline bool looks_like_commit(const std::string &rev)
    {
        if (rev.size() != 40)
            return false;
        for (char c : rev)
            if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f')))
                return false;
        return true;
    }

    // Accepts a filesystem directory (containing config.json) or a "org/name" repo id
    // resolved through the local cache (refs/<revision> → snapshots/<commit>).
    inline std::string resolve_model(const std::string &ref, const std::string &revision = "main")
    {
        if (is_dir(ref))
        {
            if (!is_file(ref + "/config.json"))
                throw std::runtime_error(ref + " is a directory but has no config.json");
            return ref;
        }
        std::string repo_dir = cache_root() + "/models--";
        for (char c : ref)
            repo_dir += (c == '/') ? std::string("--") : std::string(1, c);
        if (!is_dir(repo_dir))
            throw std::runtime_error("'" + ref + "' is neither a local directory nor cached at " +
                                     repo_dir + " (fetch it first, e.g.: hf download " + ref + ")");
        std::string commit;
        if (looks_like_commit(revision))
        {
            commit = revision;
        }
        else
        {
            std::string ref_file = repo_dir + "/refs/" + revision;
            if (!is_file(ref_file))
                throw std::runtime_error("revision '" + revision + "' not found at " + ref_file);
            commit = strip(read_file(ref_file));
        }
        std::string snap = repo_dir + "/snapshots/" + commit;
        if (!is_dir(snap))
            throw std::runtime_error("snapshot dir missing: " + snap);
        return snap;
    }

    // Snapshot dir → parsed config.json + tensor lookup across single-file or
    // sharded (model.safetensors.index.json) checkpoints. Shards are mmapped once.
    class ModelFiles
    {
    public:
        explicit ModelFiles(const std::string &snapshot_dir) : dir_(snapshot_dir)
        {
            config = minijson::parse(read_file(dir_ + "/config.json"));

            std::string index_path = dir_ + "/model.safetensors.index.json";
            if (is_file(index_path))
            {
                auto index = minijson::parse(read_file(index_path));
                for (auto &[tname, fname] : index->at("weight_map").obj)
                    tensor_file_[tname] = fname->as_str();
            }
            else if (is_file(dir_ + "/model.safetensors"))
            {
                const auto &f = shard("model.safetensors");
                for (auto &[tname, _] : f.tensors())
                    tensor_file_[tname] = "model.safetensors";
            }
            else
            {
                throw std::runtime_error(dir_ + ": no model.safetensors[.index.json]");
            }
        }

        bool has(const std::string &name) const { return tensor_file_.count(name) > 0; }

        const st::Tensor &tensor(const std::string &name)
        {
            auto it = tensor_file_.find(name);
            if (it == tensor_file_.end())
                throw std::runtime_error("checkpoint has no tensor '" + name + "'");
            return shard(it->second).tensor(name);
        }

        std::vector<std::string> tensor_names() const
        {
            std::vector<std::string> out;
            out.reserve(tensor_file_.size());
            for (auto &[k, _] : tensor_file_)
                out.push_back(k);
            return out;
        }

        const std::string &dir() const { return dir_; }
        minijson::ValuePtr config;

    private:
        const st::SafeTensors &shard(const std::string &fname)
        {
            auto it = shards_.find(fname);
            if (it == shards_.end())
                it = shards_.emplace(fname, std::make_unique<st::SafeTensors>(dir_ + "/" + fname)).first;
            return *it->second;
        }

        std::string dir_;
        std::map<std::string, std::string> tensor_file_;
        std::map<std::string, std::unique_ptr<st::SafeTensors>> shards_;
    };

} // namespace hf
