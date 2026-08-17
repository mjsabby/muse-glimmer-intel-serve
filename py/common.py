"""Shared helpers: local HF cache resolution (mirrors the C++ resolver) and IO."""

import json
import os
import re


def hf_cache_root():
    if os.environ.get("HF_HUB_CACHE"):
        return os.environ["HF_HUB_CACHE"]
    if os.environ.get("HF_HOME"):
        return os.path.join(os.environ["HF_HOME"], "hub")
    if os.environ.get("XDG_CACHE_HOME"):
        return os.path.join(os.environ["XDG_CACHE_HOME"], "huggingface", "hub")
    return os.path.expanduser("~/.cache/huggingface/hub")


def resolve_model(ref, revision="main"):
    """Resolve a local dir path or a repo id (via the local HF cache) to a snapshot dir."""
    if os.path.isdir(ref):
        return ref
    repo_dir = os.path.join(hf_cache_root(), "models--" + ref.replace("/", "--"))
    if not os.path.isdir(repo_dir):
        raise FileNotFoundError(f"{ref}: not a directory and not in HF cache at {repo_dir}")
    if re.fullmatch(r"[0-9a-f]{40}", revision):
        commit = revision
    else:
        ref_file = os.path.join(repo_dir, "refs", revision)
        with open(ref_file) as f:
            commit = f.read().strip()
    snap = os.path.join(repo_dir, "snapshots", commit)
    if not os.path.isdir(snap):
        raise FileNotFoundError(f"snapshot dir missing: {snap}")
    return snap


def parse_ids(spec):
    """Token ids from a comma/space separated string or a file containing one."""
    if os.path.exists(spec):
        with open(spec) as f:
            spec = f.read()
    return [int(x) for x in spec.replace(",", " ").split()]


def write_f64(path, arr):
    import numpy as np

    a = np.ascontiguousarray(arr, dtype=np.float64)
    a.tofile(path)
    return list(a.shape)


def write_meta(outdir, meta):
    with open(os.path.join(outdir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
