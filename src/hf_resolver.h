// HuggingFace local cache resolver — strictly offline.
//
// Resolution rules (mirrors huggingface_hub's on-disk layout):
//   cache root: $HF_HUB_CACHE, else $HF_HOME/hub, else ~/.cache/huggingface/hub
//   repo dir:   <root>/models--{org}--{name}
//   revision:   refs/<revision> file contains the commit hash (default "main");
//               a 40-char hex revision is used as a commit directly
//   snapshot:   <repo>/snapshots/<commit>/  (files are symlinks into blobs/)
//
// A model spec may also be a filesystem path to a snapshot-like directory
// (anything containing config.json).
#pragma once

#include <optional>
#include <string>
#include <vector>

namespace oracle
{

struct ResolvedModel
{
    std::string repo_id;      // e.g. "google/gemma-4-31B-it" (empty for raw dirs)
    std::string revision;     // requested revision (e.g. "main")
    std::string commit;       // resolved commit hash (empty for raw dirs)
    std::string snapshot_dir; // absolute path containing config.json etc.

    std::string file(const std::string &name) const; // throws if missing
    std::optional<std::string> try_file(const std::string &name) const;
};

// `spec` = "org/name", "org/name@revision", or a directory path.
ResolvedModel resolve_model(const std::string &spec);

std::string hf_cache_root();

// If `path` is a git-LFS pointer file, return the object path on the same
// filesystem ($ORACLE_LFS_STORE or the enclosing worktree's .git/lfs/objects);
// otherwise return `path` unchanged. Throws if it is a pointer whose object is
// not materialized locally. Lets a bare .gguf path be an un-smudged LFS pointer.
std::string resolve_lfs_path(const std::string &path);

} // namespace oracle
