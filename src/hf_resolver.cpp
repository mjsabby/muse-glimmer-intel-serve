#include "hf_resolver.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>

namespace fs = std::filesystem;

namespace oracle
{

namespace
{

std::string trim(std::string s)
{
    auto issp = [](unsigned char c) { return c == ' ' || c == '\n' || c == '\r' || c == '\t'; };
    while (!s.empty() && issp(static_cast<unsigned char>(s.back())))
    {
        s.pop_back();
    }
    size_t i = 0;
    while (i < s.size() && issp(static_cast<unsigned char>(s[i])))
    {
        ++i;
    }
    return s.substr(i);
}

bool is_hex_lower(const std::string &s)
{
    return !s.empty() &&
           std::all_of(s.begin(), s.end(), [](char c) { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'); });
}

// A git-LFS pointer is a tiny text file:
//   version https://git-lfs.github.com/spec/v1
//   oid sha256:<64 hex>
//   size <bytes>
// Return the sha256 oid if `p` looks like one, else empty. Reads only the head.
std::string lfs_pointer_oid(const fs::path &p)
{
    std::error_code ec;
    auto sz = fs::file_size(p, ec);
    if (ec || sz == 0 || sz > 1024)
    {
        return {}; // pointers are ~130 bytes
    }
    std::FILE *f = std::fopen(p.c_str(), "rb");
    if (!f)
    {
        return {};
    }
    char buf[1025];
    size_t n = std::fread(buf, 1, sizeof buf - 1, f);
    std::fclose(f);
    buf[n] = '\0';
    std::string head(buf, n);
    if (head.rfind("version https://git-lfs.github.com/spec/v1", 0) != 0)
    {
        return {};
    }
    auto pos = head.find("oid sha256:");
    if (pos == std::string::npos)
    {
        return {};
    }
    pos += std::string("oid sha256:").size();
    std::string oid;
    while (pos < head.size() && std::isxdigit(static_cast<unsigned char>(head[pos])))
    {
        oid.push_back(static_cast<char>(std::tolower(head[pos++])));
    }
    return (oid.size() == 64 && is_hex_lower(oid)) ? oid : std::string{};
}

// Walk up from `start` to find a git working tree (a `.git` dir, or a `.git`
// file pointing elsewhere for worktrees/submodules). Returns the directory
// that holds the LFS object store (`<gitdir>/lfs/objects`), or empty.
fs::path git_lfs_store(const fs::path &start)
{
    std::error_code ec;
    for (fs::path d = start; !d.empty(); d = d.parent_path())
    {
        fs::path g = d / ".git";
        if (fs::is_directory(g, ec))
        {
            return g / "lfs" / "objects";
        }
        if (fs::is_regular_file(g, ec))
        {
            // `.git` file: "gitdir: <path>" (absolute, or relative to d).
            std::FILE *f = std::fopen(g.c_str(), "rb");
            if (f)
            {
                char buf[512];
                size_t n = std::fread(buf, 1, sizeof buf - 1, f);
                std::fclose(f);
                std::string s = trim(std::string(buf, n));
                const std::string pfx = "gitdir:";
                if (s.rfind(pfx, 0) == 0)
                {
                    std::string gd = trim(s.substr(pfx.size()));
                    fs::path gp = fs::path(gd).is_absolute() ? fs::path(gd) : d / gd;
                    // For worktrees, LFS lives in the common dir (parent of worktrees/<name>).
                    fs::path common = gp;
                    if (gp.filename() != ".git")
                    {
                        // e.g. <repo>/.git/worktrees/<name> -> <repo>/.git
                        auto wt = gp;
                        while (!wt.empty() && wt.filename() != ".git")
                        {
                            wt = wt.parent_path();
                        }
                        if (!wt.empty())
                        {
                            common = wt;
                        }
                    }
                    return common / "lfs" / "objects";
                }
            }
        }
        if (d.parent_path() == d)
        {
            break;
        }
    }
    return {};
}

// Given an LFS oid, locate the object bytes on the same filesystem. Order:
//   $ORACLE_LFS_STORE/<oid[:2]>/<oid[2:4]>/<oid>   (explicit override)
//   <git worktree>/.git/lfs/objects/<oid[:2]>/<oid[2:4]>/<oid>
// Returns the resolved path, or empty if not materialized locally.
std::optional<std::string> find_lfs_object(const std::string &oid, const fs::path &near)
{
    std::error_code ec;
    auto sharded = [&](const fs::path &store) -> std::optional<std::string> {
        if (store.empty())
        {
            return std::nullopt;
        }
        fs::path o = store / oid.substr(0, 2) / oid.substr(2, 2) / oid;
        if (fs::exists(o, ec))
        {
            return o.string();
        }
        fs::path flat = store / oid; // some stores are flat
        if (fs::exists(flat, ec))
        {
            return flat.string();
        }
        return std::nullopt;
    };
    if (const char *s = std::getenv("ORACLE_LFS_STORE"); s && *s)
    {
        if (auto r = sharded(s))
        {
            return r;
        }
    }
    if (auto r = sharded(git_lfs_store(near)))
    {
        return r;
    }
    return std::nullopt;
}

bool is_hex40(const std::string &s)
{
    if (s.size() != 40)
    {
        return false;
    }
    return std::all_of(s.begin(), s.end(), [](char c) {
        return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
    });
}

std::string read_text_file(const fs::path &p)
{
    std::string data;
    std::FILE *f = std::fopen(p.c_str(), "rb");
    if (!f)
    {
        throw std::runtime_error("cannot read " + p.string());
    }
    char buf[256];
    size_t n;
    while ((n = std::fread(buf, 1, sizeof buf, f)) > 0)
    {
        data.append(buf, n);
    }
    std::fclose(f);
    return data;
}

} // namespace

std::string hf_cache_root()
{
    if (const char *c = std::getenv("HF_HUB_CACHE"); c && *c)
    {
        return c;
    }
    if (const char *h = std::getenv("HF_HOME"); h && *h)
    {
        return std::string(h) + "/hub";
    }
    const char *home = std::getenv("HOME");
    if (!home)
    {
        throw std::runtime_error("cannot determine HF cache root ($HOME unset)");
    }
    return std::string(home) + "/.cache/huggingface/hub";
}

std::string ResolvedModel::file(const std::string &name) const
{
    auto p = try_file(name);
    if (!p)
    {
        throw std::runtime_error("model file missing: " + name + " in " + snapshot_dir);
    }
    return *p;
}

std::string resolve_lfs_path(const std::string &path)
{
    std::string oid = lfs_pointer_oid(fs::path(path));
    if (oid.empty())
    {
        return path; // not an LFS pointer
    }
    if (auto obj = find_lfs_object(oid, fs::path(path).parent_path()))
    {
        return *obj;
    }
    throw std::runtime_error("LFS pointer not materialized locally: " + path + " (oid " + oid +
                             "); run `git lfs pull` or set $ORACLE_LFS_STORE");
}

std::optional<std::string> ResolvedModel::try_file(const std::string &name) const
{
    fs::path p = fs::path(snapshot_dir) / name;
    std::error_code ec;
    if (!fs::exists(p, ec))
    {
        return std::nullopt;
    }
    // A git checkout may hold an un-smudged LFS pointer instead of the blob;
    // redirect to the object store on the same filesystem when so.
    return resolve_lfs_path(p.string());
}

ResolvedModel resolve_model(const std::string &spec)
{
    ResolvedModel m;

    // Raw directory?
    {
        std::error_code ec;
        fs::path p(spec);
        if (fs::is_directory(p, ec) && fs::exists(p / "config.json", ec))
        {
            m.snapshot_dir = fs::absolute(p).string();
            return m;
        }
    }

    std::string repo = spec;
    std::string revision = "main";
    if (auto at = spec.find('@'); at != std::string::npos)
    {
        repo = spec.substr(0, at);
        revision = spec.substr(at + 1);
    }
    auto slash = repo.find('/');
    if (slash == std::string::npos || repo.find('/', slash + 1) != std::string::npos)
    {
        throw std::runtime_error(
            "model spec '" + spec +
            "' is neither an existing directory with config.json nor an 'org/name[@revision]' repo id");
    }

    fs::path root = hf_cache_root();
    fs::path repo_dir = root / ("models--" + repo.substr(0, slash) + "--" + repo.substr(slash + 1));
    std::error_code ec;
    if (!fs::is_directory(repo_dir, ec))
    {
        throw std::runtime_error("repo not in local HF cache: " + repo_dir.string() +
                                 " (offline resolver; download it first, e.g. with `hf download`)");
    }

    std::string commit;
    if (is_hex40(revision))
    {
        commit = revision;
    }
    else
    {
        fs::path ref = repo_dir / "refs" / revision;
        if (!fs::exists(ref, ec))
        {
            throw std::runtime_error("revision '" + revision + "' not found (no " + ref.string() + ")");
        }
        commit = trim(read_text_file(ref));
        if (!is_hex40(commit))
        {
            throw std::runtime_error("ref file " + ref.string() + " does not contain a commit hash");
        }
    }

    fs::path snap = repo_dir / "snapshots" / commit;
    if (!fs::is_directory(snap, ec))
    {
        throw std::runtime_error("snapshot missing: " + snap.string());
    }
    if (!fs::exists(snap / "config.json", ec))
    {
        throw std::runtime_error("snapshot has no config.json: " + snap.string());
    }

    m.repo_id = repo;
    m.revision = revision;
    m.commit = commit;
    m.snapshot_dir = snap.string();
    return m;
}

} // namespace oracle
