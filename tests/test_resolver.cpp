#include <cstdlib>
#include <filesystem>
#include <fstream>

#include "hf_resolver.h"
#include "test_util.h"

namespace fs = std::filesystem;
using namespace oracle;

void test_resolver()
{
    // Build a fake cache layout under TMPDIR and resolve through it.
    std::string tmp = std::getenv("TMPDIR") ? std::getenv("TMPDIR") : "/tmp";
    fs::path root = fs::path(tmp) / "oracle_fake_hf";
    fs::remove_all(root);
    fs::path repo = root / "models--acme--tiny";
    fs::create_directories(repo / "refs");
    std::string commit = "0123456789abcdef0123456789abcdef01234567";
    fs::create_directories(repo / "snapshots" / commit);
    std::ofstream(repo / "refs" / "main") << commit << "\n";
    std::ofstream(repo / "snapshots" / commit / "config.json") << "{}";

    ::setenv("HF_HUB_CACHE", root.c_str(), 1);
    ResolvedModel m = resolve_model("acme/tiny");
    CHECK(m.commit == commit);
    CHECK(m.try_file("config.json").has_value());
    CHECK(!m.try_file("nope.bin").has_value());

    ResolvedModel m2 = resolve_model("acme/tiny@" + commit);
    CHECK(m2.snapshot_dir == m.snapshot_dir);

    // Raw dir path form.
    ResolvedModel m3 = resolve_model((repo / "snapshots" / commit).string());
    CHECK(m3.snapshot_dir == m.snapshot_dir);
    CHECK(m3.commit.empty());

    bool threw = false;
    try
    {
        resolve_model("acme/absent");
    }
    catch (const std::exception &)
    {
        threw = true;
    }
    CHECK(threw);
    ::unsetenv("HF_HUB_CACHE");

    // ---- git-LFS pointer resolution ----
    // A git checkout whose large file is an un-smudged LFS pointer resolves to
    // the object bytes in .git/lfs/objects/<xx>/<yy>/<oid> on the same FS.
    fs::path gitrepo = root / "gitco";
    fs::create_directories(gitrepo / ".git");
    std::ofstream(gitrepo / "config.json") << "{}";
    // sha256("hello\n") — the object content we materialize below.
    std::string oid = "5891b5b522d5df086d0ff0b110fbd9d21bb4fc7163af34d08286a2e846f6be03";
    std::ofstream(gitrepo / "model.bin") << "version https://git-lfs.github.com/spec/v1\n"
                                         << "oid sha256:" << oid << "\n"
                                         << "size 6\n";
    fs::path objdir = gitrepo / ".git" / "lfs" / "objects" / oid.substr(0, 2) / oid.substr(2, 2);
    fs::create_directories(objdir);
    std::ofstream(objdir / oid) << "hello\n";

    ResolvedModel g = resolve_model(gitrepo.string());
    auto rf = g.try_file("model.bin");
    CHECK(rf.has_value());
    CHECK(rf->find("/.git/lfs/objects/") != std::string::npos); // redirected, not the pointer
    { // resolved path holds the object bytes, not the pointer text
        std::ifstream in(*rf);
        std::string content((std::istreambuf_iterator<char>(in)), {});
        CHECK(content == "hello\n");
    }
    // A non-pointer file is returned verbatim.
    CHECK(*g.try_file("config.json") == (gitrepo / "config.json").string());

    // Pointer with no local object -> a clear throw.
    std::ofstream(gitrepo / "missing.bin") << "version https://git-lfs.github.com/spec/v1\n"
                                           << "oid sha256:" << std::string(64, 'a') << "\nsize 1\n";
    bool lfs_threw = false;
    try
    {
        g.try_file("missing.bin");
    }
    catch (const std::exception &)
    {
        lfs_threw = true;
    }
    CHECK(lfs_threw);

    // $ORACLE_LFS_STORE override wins.
    fs::path store = root / "shared_lfs";
    fs::path sdir = store / oid.substr(0, 2) / oid.substr(2, 2);
    fs::create_directories(sdir);
    std::ofstream(sdir / oid) << "hello\n";
    ::setenv("ORACLE_LFS_STORE", store.c_str(), 1);
    auto rf2 = g.try_file("model.bin");
    CHECK(rf2.has_value() && rf2->find("shared_lfs") != std::string::npos);
    ::unsetenv("ORACLE_LFS_STORE");
}
