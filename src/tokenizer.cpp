#include "tokenizer.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <unordered_map>

#include "json.h"

namespace oracle
{

namespace
{
constexpr const char *kSpace = "\xe2\x96\x81"; // ▁ U+2581

struct AddedToken
{
    std::string content;
    int32_t id = 0;
    bool lstrip = false, rstrip = false, single_word = false, normalized = false;
};
} // namespace

struct Tokenizer::Impl
{
    std::unordered_map<std::string, int32_t> vocab;
    std::vector<std::string> id_to_token;
    std::unordered_map<std::string, int32_t> merge_rank; // "left\nright" -> rank
    std::vector<AddedToken> added;                       // sorted by content length desc
    int32_t unk_id = 3;
    int32_t byte_ids[256];
    bool byte_fallback = true;
    bool fuse_unk = true;
    bool ignore_merges = false;

    // ---- helpers ----
    int32_t lookup(const std::string &s) const
    {
        auto it = vocab.find(s);
        return it == vocab.end() ? -1 : it->second;
    }

    // Splits a normalized piece into initial unicode-character symbols.
    static std::vector<std::string> to_chars(const std::string &s)
    {
        std::vector<std::string> out;
        size_t i = 0;
        while (i < s.size())
        {
            unsigned char c = static_cast<unsigned char>(s[i]);
            size_t len = c < 0x80 ? 1 : (c < 0xE0 ? 2 : (c < 0xF0 ? 3 : 4));
            if (c >= 0x80 && c < 0xC0)
            {
                len = 1; // stray continuation byte; take alone
            }
            len = std::min(len, s.size() - i);
            out.push_back(s.substr(i, len));
            i += len;
        }
        return out;
    }

    // GPT2-style BPE loop: repeatedly merge the lowest-rank adjacent pair.
    std::vector<std::string> bpe(std::vector<std::string> word) const
    {
        if (word.size() <= 1)
        {
            return word;
        }
        while (true)
        {
            int best_rank = std::numeric_limits<int32_t>::max();
            size_t best_i = 0;
            for (size_t i = 0; i + 1 < word.size(); ++i)
            {
                auto it = merge_rank.find(word[i] + "\n" + word[i + 1]);
                if (it != merge_rank.end() && it->second < best_rank)
                {
                    best_rank = it->second;
                    best_i = i;
                }
            }
            if (best_rank == std::numeric_limits<int32_t>::max())
            {
                break;
            }
            // merge ALL occurrences of that exact pair, left to right
            const std::string a = word[best_i], b = word[best_i + 1];
            std::vector<std::string> next;
            next.reserve(word.size());
            size_t i = 0;
            while (i < word.size())
            {
                if (i + 1 < word.size() && word[i] == a && word[i + 1] == b)
                {
                    next.push_back(a + b);
                    i += 2;
                }
                else
                {
                    next.push_back(word[i]);
                    ++i;
                }
            }
            word.swap(next);
            if (word.size() <= 1)
            {
                break;
            }
        }
        return word;
    }

    void encode_piece(const std::string &piece, std::vector<int32_t> &out) const
    {
        if (piece.empty())
        {
            return;
        }
        if (ignore_merges)
        {
            if (int32_t id = lookup(piece); id >= 0)
            {
                out.push_back(id);
                return;
            }
        }
        std::vector<std::string> symbols = bpe(to_chars(piece));
        int pending_unk = 0;
        for (const auto &sym : symbols)
        {
            int32_t id = lookup(sym);
            if (id >= 0)
            {
                if (pending_unk)
                {
                    out.push_back(unk_id);
                    pending_unk = 0;
                }
                out.push_back(id);
            }
            else if (byte_fallback)
            {
                for (unsigned char b : sym)
                {
                    out.push_back(byte_ids[b]);
                }
            }
            else if (fuse_unk)
            {
                pending_unk = 1;
            }
            else
            {
                out.push_back(unk_id);
            }
        }
        if (pending_unk)
        {
            out.push_back(unk_id);
        }
    }

    // normalizer(Replace " "->▁) + Split(" ", MergedWithPrevious) + BPE.
    // Post-normalization no plain spaces remain, so the Split is a no-op and
    // the whole segment is a single BPE word (SP-style); verified vs HF
    // tokenizers by tools/tokenizer_parity.py.
    void encode_text_segment(const std::string &raw, std::vector<int32_t> &out) const
    {
        std::string norm;
        norm.reserve(raw.size() + 8);
        for (char c : raw)
        {
            if (c == ' ')
            {
                norm += kSpace;
            }
            else
            {
                norm.push_back(c);
            }
        }
        encode_piece(norm, out);
    }
};

Tokenizer::Tokenizer(const std::string &path) : impl_(std::make_unique<Impl>())
{
    Json root = Json::parse_file(path);

    const Json &model = root.at("model");
    if (model.at("type").as_string() != "BPE")
    {
        throw std::runtime_error("tokenizer: expected BPE model, got " + model.at("type").as_string());
    }
    impl_->byte_fallback = model.at("byte_fallback").as_bool();
    impl_->fuse_unk = model.at("fuse_unk").as_bool();
    if (const Json *im = model.get("ignore_merges"))
    {
        impl_->ignore_merges = !im->is_null() && im->as_bool();
    }

    const auto &vocab = model.at("vocab").as_object();
    impl_->vocab.reserve(vocab.size() * 2);
    int64_t max_id = 0;
    for (const auto &[tok, idj] : vocab)
    {
        max_id = std::max(max_id, idj.as_int());
    }
    impl_->id_to_token.resize(static_cast<size_t>(max_id) + 1);
    for (const auto &[tok, idj] : vocab)
    {
        int32_t id = static_cast<int32_t>(idj.as_int());
        impl_->vocab.emplace(tok, id);
        impl_->id_to_token[static_cast<size_t>(id)] = tok;
    }
    if (const Json *unk = model.get("unk_token"); unk && !unk->is_null())
    {
        impl_->unk_id = impl_->lookup(unk->as_string());
    }

    const auto &merges = model.at("merges").as_array();
    impl_->merge_rank.reserve(merges.size() * 2);
    for (size_t r = 0; r < merges.size(); ++r)
    {
        std::string left, right;
        if (merges[r].is_array())
        {
            const auto &pair = merges[r].as_array();
            left = pair.at(0).as_string();
            right = pair.at(1).as_string();
        }
        else
        {
            const std::string &s = merges[r].as_string();
            auto sp = s.find(' ');
            if (sp == std::string::npos)
            {
                throw std::runtime_error("tokenizer: bad merge entry");
            }
            left = s.substr(0, sp);
            right = s.substr(sp + 1);
        }
        impl_->merge_rank.emplace(left + "\n" + right, static_cast<int32_t>(r));
    }

    for (int b = 0; b < 256; ++b)
    {
        char buf[8];
        std::snprintf(buf, sizeof buf, "<0x%02X>", b);
        impl_->byte_ids[b] = impl_->lookup(buf);
        if (impl_->byte_fallback && impl_->byte_ids[b] < 0)
        {
            throw std::runtime_error("tokenizer: byte_fallback set but byte token missing: " + std::string(buf));
        }
    }

    if (const Json *added = root.get("added_tokens"))
    {
        for (const auto &a : added->as_array())
        {
            AddedToken t;
            t.content = a.at("content").as_string();
            t.id = static_cast<int32_t>(a.at("id").as_int());
            auto flag = [&](const char *k) {
                const Json *v = a.get(k);
                return v && !v->is_null() && v->as_bool();
            };
            t.lstrip = flag("lstrip");
            t.rstrip = flag("rstrip");
            t.single_word = flag("single_word");
            t.normalized = flag("normalized");
            impl_->added.push_back(std::move(t));
        }
        std::sort(impl_->added.begin(), impl_->added.end(),
                  [](const AddedToken &x, const AddedToken &y) { return x.content.size() > y.content.size(); });
    }
}

Tokenizer::~Tokenizer() = default;

int32_t Tokenizer::bos_id() const
{
    return 2;
}

std::vector<int32_t> Tokenizer::encode(const std::string &text, bool add_bos) const
{
    std::vector<int32_t> out;
    if (add_bos)
    {
        out.push_back(bos_id());
    }

    // Added-token scan (leftmost, longest-first). Gemma's added tokens all have
    // lstrip/rstrip/single_word=false, so plain substring matching applies.
    size_t start = 0, i = 0;
    while (i < text.size())
    {
        const AddedToken *hit = nullptr;
        for (const auto &t : impl_->added)
        {
            if (text.compare(i, t.content.size(), t.content) == 0)
            {
                hit = &t;
                break;
            }
        }
        if (hit)
        {
            if (i > start)
            {
                impl_->encode_text_segment(text.substr(start, i - start), out);
            }
            out.push_back(hit->id);
            i += hit->content.size();
            start = i;
        }
        else
        {
            ++i;
        }
    }
    if (start < text.size())
    {
        impl_->encode_text_segment(text.substr(start), out);
    }
    return out;
}

std::string Tokenizer::decode(const std::vector<int32_t> &ids) const
{
    // decoder: Replace(▁->' ') + ByteFallback + Fuse.
    std::string out;
    std::string byte_buf;
    auto flush_bytes = [&]() {
        if (!byte_buf.empty())
        {
            out += byte_buf; // raw bytes; invalid UTF-8 remains as-is
            byte_buf.clear();
        }
    };
    for (int32_t id : ids)
    {
        if (id < 0 || static_cast<size_t>(id) >= impl_->id_to_token.size())
        {
            continue;
        }
        const std::string &tok = impl_->id_to_token[static_cast<size_t>(id)];
        // byte token?
        if (tok.size() == 6 && tok.rfind("<0x", 0) == 0 && tok[5] == '>')
        {
            int hi = std::isdigit(tok[3]) ? tok[3] - '0' : std::toupper(tok[3]) - 'A' + 10;
            int lo = std::isdigit(tok[4]) ? tok[4] - '0' : std::toupper(tok[4]) - 'A' + 10;
            byte_buf.push_back(static_cast<char>(hi * 16 + lo));
            continue;
        }
        flush_bytes();
        // Replace ▁ with space
        size_t p = 0;
        while (p < tok.size())
        {
            if (tok.compare(p, 3, kSpace) == 0)
            {
                out.push_back(' ');
                p += 3;
            }
            else
            {
                out.push_back(tok[p]);
                ++p;
            }
        }
    }
    flush_bytes();
    return out;
}

std::vector<int32_t> Tokenizer::encode_chat_single_turn(const std::string &user_text) const
{
    // Single-turn -it wrap matching chat_template.jinja with
    // add_generation_prompt=True and thinking disabled (the template's
    // default): the generation prompt ends with an EMPTY thought channel.
    // Validated token-for-token by tools/tokenizer_parity.py.
    std::string s = "<|turn>user\n" + user_text + "<turn|>\n<|turn>model\n<|channel>thought\n<channel|>";
    return encode(s, /*add_bos=*/true);
}

} // namespace oracle
