// tokenizer.json (HF "tokenizers" format) implementation for Gemma 4:
//   normalizer: Replace(" " -> "▁")
//   pre_tokenizer: Split(" ", behavior=MergedWithPrevious)
//   model: BPE with byte_fallback, fuse_unk, ignore_merges=false
//   decoder: Replace("▁" -> " ") + ByteFallback + Fuse
// plus the added-token (special-token) trie that runs before normalization.
// Implemented in P5; the referee interface accepts raw token ids regardless.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace oracle
{

class Tokenizer
{
  public:
    explicit Tokenizer(const std::string &tokenizer_json_path);
    ~Tokenizer();

    std::vector<int32_t> encode(const std::string &text, bool add_bos) const;
    std::string decode(const std::vector<int32_t> &ids) const;
    int32_t bos_id() const;

    // Single-turn -it chat wrap (verified against chat_template.jinja +
    // reference dialog.Format.GEMMA4): <bos><|turn>user\n{...}<turn|>\n<|turn>model\n
    std::vector<int32_t> encode_chat_single_turn(const std::string &user_text) const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace oracle
