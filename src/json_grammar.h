// Guided decoding: constrain the sampler to emit only VALID JSON. A byte-level
// pushdown automaton advances through the output; at each decode step it masks
// any token whose bytes would break the JSON, so the completion is guaranteed
// parseable. Terminal (a complete top-level value) also re-allows EOG so
// generation can stop. This is the `response_format: json_object` engine; a
// full JSON-schema / GBNF grammar is a later extension over the same hook.
//
// Ported from the author's irun runtime (common/json_grammar.hpp) with the
// tokenizer dependency inverted: the per-vocab piece bytes + EOG flags arrive
// through the C ABI (built by the Python frontend from the HF tokenizer), and
// mask() writes an allowed[V] byte array instead of mutating logits.
#pragma once
#include <cctype>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace oracle
{
namespace jsong
{

// what the top-of-stack frame expects next
enum ctx : uint8_t
{
    VAL,
    OBJ_KEY,
    OBJ_KEY_OR_END,
    OBJ_COLON,
    OBJ_COMMA_END,
    ARR_VAL_END,
    ARR_COMMA_END
};

// Inline (heap-free) stack with the vector subset step() uses, so `state` is a
// trivially-copyable POD — the guided-decoding mask copies it per candidate
// token, so making the copy a memcpy (no allocation) is the hot lever. Depth 32
// is far beyond any real JSON nesting; deeper input over-constrains (safe).
template <class T, int N> struct small_stack
{
    T a[N];
    int n = 0;
    T &back()
    {
        return a[n - 1];
    }
    const T &back() const
    {
        return a[n - 1];
    }
    bool empty() const
    {
        return n == 0;
    }
    int size() const
    {
        return n;
    }
    void pop_back()
    {
        if (n > 0)
        {
            --n;
        }
    }
    void push_back(const T &x)
    {
        if (n < N)
        {
            a[n++] = x;
        }
    }
};

struct state
{
    small_stack<ctx, 32> stk; // frame stack; empty + started => complete top-level value
    int instr = 0;            // 0 none, 1 in string, 2 after backslash, 3 in \uXXXX
    int hex = 0;
    int innum = 0;             // 1 while consuming a number
    const char *lit = nullptr; // non-null while matching true/false/null (remaining chars)
    bool started = false;
    bool obj_only = false; // json_object mode: the top-level value must be an object
    state()
    {
        stk.push_back(VAL);
    }
    bool complete() const
    {
        return started && stk.empty() && instr == 0 && innum == 0 && !lit;
    }
};

inline bool is_ws(char c)
{
    return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}

// Consume one byte; return false if it would break the JSON. `s` mutates on success.
inline bool step(state &s, char c)
{
    if (s.instr == 3)
    {
        if (!std::isxdigit(static_cast<unsigned char>(c)))
        {
            return false;
        }
        if (--s.hex == 0)
        {
            s.instr = 1;
        }
        return true;
    }
    if (s.instr == 2)
    {
        if (c == 'u')
        {
            s.instr = 3;
            s.hex = 4;
            return true;
        }
        if (std::string("\"\\/bfnrt").find(c) == std::string::npos)
        {
            return false;
        }
        s.instr = 1;
        return true;
    }
    if (s.instr == 1)
    {
        if (c == '\\')
        {
            s.instr = 2;
            return true;
        }
        if (c == '"')
        {
            s.instr = 0; // string closes
            if (s.stk.back() == OBJ_KEY || s.stk.back() == OBJ_KEY_OR_END)
            {
                s.stk.back() = OBJ_COLON;
            }
            else
            {
                s.stk.pop_back();
            }
            return true;
        }
        return static_cast<unsigned char>(c) >= 0x20; // raw control chars illegal in strings
    }
    if (s.lit)
    {
        if (c != *s.lit)
        {
            return false;
        }
        if (*++s.lit == '\0')
        {
            s.lit = nullptr;
            s.stk.pop_back();
        }
        return true;
    }
    if (s.innum)
    {
        if (c == '-' || c == '+' || c == '.' || c == 'e' || c == 'E' || std::isdigit(static_cast<unsigned char>(c)))
        {
            return true;
        }
        s.innum = 0;
        s.stk.pop_back(); // number ended; reprocess c below
    }
    if (is_ws(c))
    {
        return true;
    }
    if (s.stk.empty())
    {
        return false; // complete top-level value; nothing more but ws
    }
    const ctx top = s.stk.back();
    auto open = [&]() -> bool { // open a value (top frame is a VAL slot)
        if (s.obj_only && !s.started && c != '{')
        {
            return false; // json_object: top must be {
        }
        s.started = true;
        switch (c)
        {
        case '"':
            s.instr = 1;
            return true;
        case '{':
            s.stk.back() = OBJ_KEY_OR_END;
            return true;
        case '[':
            s.stk.back() = ARR_VAL_END;
            return true;
        case 't':
            s.lit = "rue";
            return true;
        case 'f':
            s.lit = "alse";
            return true;
        case 'n':
            s.lit = "ull";
            return true;
        default:
            if (c == '-' || std::isdigit(static_cast<unsigned char>(c)))
            {
                s.innum = 1;
                return true;
            }
            return false;
        }
    };
    switch (top)
    {
    case VAL:
        return open();
    case OBJ_KEY_OR_END: // right after { : key or }
        if (c == '}')
        {
            s.stk.pop_back();
            return true;
        }
        [[fallthrough]];
    case OBJ_KEY: // after , : key required
        if (c == '"')
        {
            s.started = true;
            s.instr = 1;
            return true;
        }
        return false;
    case OBJ_COLON:
        if (c == ':')
        {
            s.stk.back() = OBJ_COMMA_END;
            s.stk.push_back(VAL);
            return true;
        }
        return false;
    case OBJ_COMMA_END:
        if (c == ',')
        {
            s.stk.back() = OBJ_KEY;
            return true;
        }
        if (c == '}')
        {
            s.stk.pop_back();
            return true;
        }
        return false;
    case ARR_VAL_END:
        if (c == ']')
        {
            s.stk.pop_back();
            return true;
        } // empty array
        s.stk.back() = ARR_COMMA_END;
        s.stk.push_back(VAL);
        return step(s, c); // first element
    case ARR_COMMA_END:
        if (c == ',')
        {
            s.stk.push_back(VAL);
            return true;
        }
        if (c == ']')
        {
            s.stk.pop_back();
            return true;
        }
        return false;
    }
    return false;
}

inline bool allows(state s, const char *p, size_t n)
{ // by value: test on a copy
    for (size_t i = 0; i < n; ++i)
    {
        if (!step(s, p[i]))
        {
            return false;
        }
    }
    return true;
}
inline void advance(state &s, const char *p, size_t n)
{
    for (size_t i = 0; i < n; ++i)
    {
        step(s, p[i]);
    }
}

} // namespace jsong

// Per-tokenizer piece cache (built once; the mask sweeps the vocab each step).
// Pieces arrive as one concatenated byte blob + offsets from the frontend;
// special/control tokens get EMPTY pieces (always masked, except EOG at
// completion).
struct TokenPieces
{
    std::string bytes;
    std::vector<int64_t> off; // [V+1]
    std::vector<uint8_t> eog; // [V]
    int64_t V = 0;
    const char *piece(int64_t i) const
    {
        return bytes.data() + off[static_cast<size_t>(i)];
    }
    size_t len(int64_t i) const
    {
        return static_cast<size_t>(off[static_cast<size_t>(i) + 1] - off[static_cast<size_t>(i)]);
    }

    // First-byte bucket index (a depth-1 token trie): tokens grouped by their
    // first byte so a mask only walks the ~6% of the vocab whose first byte the
    // grammar permits, instead of iterating all V. Built once per tokenizer.
    std::vector<int32_t> by_first; // token ids, grouped by first byte
    int32_t first_off[257] = {0};  // ids with first byte b: [first_off[b], first_off[b+1])
    std::vector<int32_t> eog_ids;  // the (few) end-of-generation tokens
    bool indexed = false;
    void build_index()
    {
        int64_t cnt[256] = {0};
        eog_ids.clear();
        for (int64_t i = 0; i < V; ++i)
        {
            if (eog[static_cast<size_t>(i)])
            {
                eog_ids.push_back(static_cast<int32_t>(i));
                continue;
            }
            if (len(i) == 0)
            {
                continue;
            }
            cnt[static_cast<unsigned char>(piece(i)[0])]++;
        }
        for (int b = 0; b < 256; ++b)
        {
            first_off[b + 1] = first_off[b] + static_cast<int32_t>(cnt[b]);
        }
        by_first.resize(static_cast<size_t>(first_off[256]));
        int32_t pos[256];
        for (int b = 0; b < 256; ++b)
        {
            pos[b] = first_off[b];
        }
        for (int64_t i = 0; i < V; ++i)
        {
            if (eog[static_cast<size_t>(i)] || len(i) == 0)
            {
                continue;
            }
            by_first[static_cast<size_t>(pos[static_cast<unsigned char>(piece(i)[0])]++)] = static_cast<int32_t>(i);
        }
        indexed = true;
    }
};

// State-keyed mask cache. Guided decoding revisits byte-identical automaton
// states constantly — every char inside a string/number leaves the (POD) state
// unchanged — so caching the last few (state -> allowed[V]) masks turns those
// steps into a memcpy instead of a full vocab sweep. Buffers are pre-allocated:
// zero heap allocation after init().
struct MaskCache
{
    int V = 0;
    struct Slot
    {
        bool valid = false;
        uint64_t h = 0;
        std::vector<uint8_t> state, mask;
    };
    std::vector<Slot> slots;
    int next = 0;
    void init(int v, size_t state_bytes, int nslots = 4)
    {
        V = v;
        slots.resize(nslots);
        for (auto &s : slots)
        {
            s.state.resize(state_bytes);
            s.mask.resize(v);
            s.valid = false;
        }
    }
    static uint64_t hash(const void *p, size_t n)
    {
        const uint8_t *b = static_cast<const uint8_t *>(p);
        uint64_t h = 1469598103934665603ull;
        for (size_t i = 0; i < n; ++i)
        {
            h ^= b[i];
            h *= 1099511628211ull;
        }
        return h;
    }
    const uint8_t *find(const void *stp, size_t n, uint64_t h) const
    {
        for (const auto &s : slots)
        {
            if (s.valid && s.h == h && std::memcmp(s.state.data(), stp, n) == 0)
            {
                return s.mask.data();
            }
        }
        return nullptr;
    }
    void store(const void *stp, size_t n, uint64_t h, const uint8_t *mask)
    {
        Slot &s = slots[next];
        next = (next + 1) % static_cast<int>(slots.size());
        s.valid = true;
        s.h = h;
        std::memcpy(s.state.data(), stp, n);
        std::memcpy(s.mask.data(), mask, static_cast<size_t>(V));
    }
};

// A guided-decoding constraint: JSON PDA + shared piece cache. Cheap per
// request. mask() writes allowed[V] (1 = legal); once the JSON is a complete
// value only EOG stays legal so generation stops instead of trailing ws.
struct JsonConstraint
{
    const TokenPieces &tp;
    jsong::state st;
    mutable MaskCache cache_;
    explicit JsonConstraint(const TokenPieces &p, bool object_only) : tp(p)
    {
        st.obj_only = object_only;
        cache_.init(static_cast<int>(p.V), sizeof(st));
    }

    void mask(uint8_t *allowed) const
    {
        const uint64_t sh = MaskCache::hash(&st, sizeof(st));
        if (const uint8_t *c = cache_.find(&st, sizeof(st), sh))
        {
            std::memcpy(allowed, c, static_cast<size_t>(tp.V));
            return;
        }
        const bool done = st.complete();
        std::memset(allowed, 0, static_cast<size_t>(tp.V)); // default: deny
        for (int32_t id : tp.eog_ids)
        {
            allowed[id] = done ? 1 : 0; // stop only when complete
        }
        if (!done)
        { // a complete json value is self-delimiting -> only EOG remains
            bool first_ok[256];
            for (int b = 0; b < 256; ++b)
            {
                jsong::state s = st;
                first_ok[b] = jsong::step(s, static_cast<char>(b));
            }
            auto walk = [&](int64_t id) {
                const size_t n = tp.len(id);
                const char *p = tp.piece(id);
                allowed[id] = (n == 1) || jsong::allows(st, p, n) ? 1 : 0;
            };
            if (tp.indexed)
            { // only the ~6% of tokens whose first byte is permitted
                for (int b = 0; b < 256; ++b)
                {
                    if (first_ok[b])
                    {
                        for (int32_t k = tp.first_off[b]; k < tp.first_off[b + 1]; ++k)
                        {
                            walk(tp.by_first[k]);
                        }
                    }
                }
            }
            else
            { // fallback: full sweep (index not built)
                for (int64_t i = 0; i < tp.V; ++i)
                {
                    if (!tp.eog[static_cast<size_t>(i)] && tp.len(i) &&
                        first_ok[static_cast<unsigned char>(tp.piece(i)[0])])
                    {
                        walk(i);
                    }
                }
            }
        }
        cache_.store(&st, sizeof(st), sh, allowed);
    }
    void accept(int32_t id)
    {
        if (id >= 0 && id < tp.V)
        {
            jsong::advance(st, tp.piece(id), tp.len(id));
        }
    }
    bool complete() const
    {
        return st.complete();
    }
};

} // namespace oracle
