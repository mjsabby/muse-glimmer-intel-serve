// Grammar-level JSON-Schema enforcement: a byte pushdown automaton that only
// accepts JSON conforming to a schema (response_format json_schema, strict).
// Same hook/cost profile as the generic JSON PDA (json_grammar.h) — the vocab
// sweep + piece bytes live in C++; the Python frontend only passes the schema
// JSON string, which is parsed + flat-compiled here.
//
// The schema is compiled ONCE into a flat program (index-addressed nodes +
// packed property/enum tables) so the hot path does no std::map lookups, and
// the automaton state is a trivially-copyable POD (bitmasks + a fixed frame
// stack). The mask therefore trial-steps by snapshotting only the live frames
// instead of deep-copying std::vector/std::string per candidate token.
//
// Supported (OpenAI strict subset): object (properties, required,
// additionalProperties=false default), array (items, minItems), string,
// integer, number, boolean, null, enum, const, arbitrary nesting. Unknown
// constructs (anyOf/oneOf/$ref/patternProperties/pattern/format/min-max...)
// relax to "any valid JSON" for that subtree. Caps: 64 properties/enum values
// per node, 24 levels of schema nesting (deeper over-constrains, never crashes).
#pragma once

#include <cctype>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "json.h"
#include "json_grammar.h" // TokenPieces + jsong POD value matcher

namespace oracle
{
namespace jsons
{

enum SType : uint8_t
{
    T_ANY,
    T_STR,
    T_INT,
    T_NUM,
    T_BOOL,
    T_NULL,
    T_OBJ,
    T_ARR,
    T_ENUM
};

// ---- flat compiled schema ----------------------------------------------
struct Prog
{
    enum K : uint8_t
    {
        K_OBJ,
        K_ARR,
        K_STR,
        K_INT,
        K_NUM,
        K_BOOL,
        K_NULL,
        K_ENUM,
        K_ANY
    };
    struct Node
    {
        K kind = K_ANY;
        int prop0 = 0, nprop = 0; // properties: [prop0, prop0+nprop) into names/childs
        bool additional = false;
        uint64_t required = 0; // bit i => property i is required
        int item = -1;         // array items: node index (-1 = any)
        int64_t min_items = 0;
        int enum0 = 0, nenum = 0; // enum/const literals: [enum0, enum0+nenum) into enums
    };
    std::vector<Node> nodes;
    std::vector<std::string> names; // packed property names
    std::vector<int> childs;        // packed property child node indices
    std::vector<std::string> enums; // packed enum literal serializations
    int root = -1;

    int classify_compile(const Json *n)
    { // returns a node index (never -1)
        Node nd;
        if (!n || !n->is_object())
        {
            nd.kind = K_ANY;
            nodes.push_back(nd);
            return (int)nodes.size() - 1;
        }
        if (n->contains("enum") || n->contains("const"))
        {
            nd.kind = K_ENUM;
            nd.enum0 = (int)enums.size();
            if (const Json *e = n->get("enum"); e && e->is_array())
            {
                for (const Json &v : e->as_array())
                {
                    if (nd.nenum < 64)
                    {
                        enums.push_back(v.dump());
                        nd.nenum++;
                    }
                }
            }
            else if (const Json *c = n->get("const"))
            {
                enums.push_back(c->dump());
                nd.nenum = 1;
            }
            nodes.push_back(nd);
            return (int)nodes.size() - 1;
        }
        std::string ty;
        if (const Json *t = n->get("type"); t && t->is_string())
        {
            ty = t->as_string();
        }
        if (ty == "object" || (ty.empty() && n->contains("properties")))
        {
            nd.kind = K_OBJ;
            if (const Json *ap = n->get("additionalProperties"); ap && ap->is_bool())
            {
                nd.additional = ap->as_bool();
            }
            const Json *props = n->get("properties");
            std::vector<std::string> keys;
            std::vector<const Json *> subs;
            if (props && props->is_object())
            {
                for (const auto &kv : props->as_object())
                {
                    if ((int)keys.size() < 64)
                    {
                        keys.push_back(kv.first);
                        subs.push_back(&kv.second);
                    }
                }
            }
            // required set
            uint64_t req = 0;
            if (const Json *rq = n->get("required"); rq && rq->is_array())
            {
                for (const Json &r : rq->as_array())
                {
                    if (r.is_string())
                    {
                        for (size_t i = 0; i < keys.size(); ++i)
                        {
                            if (keys[i] == r.as_string())
                            {
                                req |= (1ull << i);
                            }
                        }
                    }
                }
            }
            // reserve this node BEFORE compiling children (children append after)
            int self = (int)nodes.size();
            nd.prop0 = (int)names.size();
            nd.nprop = (int)keys.size();
            nd.required = req;
            nodes.push_back(nd);
            for (size_t i = 0; i < keys.size(); ++i)
            {
                names.push_back(keys[i]);
                childs.push_back(0);
            }
            for (size_t i = 0; i < keys.size(); ++i)
            {
                childs[nd.prop0 + (int)i] = classify_compile(subs[i]);
            }
            return self;
        }
        if (ty == "array" || (ty.empty() && n->contains("items")))
        {
            nd.kind = K_ARR;
            if (const Json *m = n->get("minItems"); m && m->is_number())
            {
                nd.min_items = m->as_int();
            }
            int self = (int)nodes.size();
            nodes.push_back(nd);
            const Json *it = n->get("items");
            nodes[self].item = (it && it->is_object()) ? classify_compile(it) : -1;
            return self;
        }
        if (ty == "string")
        {
            nd.kind = K_STR;
        }
        else if (ty == "integer")
        {
            nd.kind = K_INT;
        }
        else if (ty == "number")
        {
            nd.kind = K_NUM;
        }
        else if (ty == "boolean")
        {
            nd.kind = K_BOOL;
        }
        else if (ty == "null")
        {
            nd.kind = K_NULL;
        }
        else
        {
            nd.kind = K_ANY;
        }
        nodes.push_back(nd);
        return (int)nodes.size() - 1;
    }
    void compile(const Json &schema)
    {
        root = classify_compile(&schema);
    }
};

inline bool is_ws(char c)
{
    return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}
inline bool is_term(char c)
{
    return is_ws(c) || c == ',' || c == '}' || c == ']';
}

enum FKind : uint8_t
{
    F_SCALAR,
    F_ENUM,
    F_OBJECT,
    F_ARRAY
};

// POD frame — no heap members, so the whole machine is trivially copyable.
struct Frame
{
    int node = -1;
    FKind kind = F_SCALAR;
    jsong::state js; // scalar/any value mechanics (POD)
    SType stype = T_ANY;
    bool opened = false, num_ok = false;
    int ostate = 0;       // object/array sub-state
    uint64_t used = 0;    // object: property i emitted
    uint64_t keycand = 0; // object: property i still prefix-matches the current key
    int keylen = 0;
    int selkey = -1;     // object: property selected by the just-closed key (-1 = arbitrary)
    bool addkey = false; // object: current key diverged from all props (additionalProperties)
    int64_t items = 0;   // array element count
    uint64_t ecand = 0;  // enum: literal j still matching
    int mpos = 0;        // enum: matched length
};

struct machine
{
    static constexpr int MAXD = 24;
    const Prog *prog = nullptr;
    Frame st[MAXD];
    int n = 0;
    bool started = false;

    machine() = default;
    explicit machine(const Prog *p) : prog(p)
    {
        push(p ? p->root : -1);
    }

    Prog::K nkind(int node) const
    {
        return node < 0 ? Prog::K_ANY : prog->nodes[node].kind;
    }
    const Prog::Node &nd(int node) const
    {
        return prog->nodes[node];
    }

    void push(int node)
    {
        if (n >= MAXD)
        {
            return; // over-deep: leave the top; step() then rejects
        }
        Frame f;
        f.node = node;
        Prog::K k = nkind(node);
        if (k == Prog::K_OBJ)
        {
            f.kind = F_OBJECT;
        }
        else if (k == Prog::K_ARR)
        {
            f.kind = F_ARRAY;
        }
        else if (k == Prog::K_ENUM)
        {
            f.kind = F_ENUM;
            int ne = nd(node).nenum;
            f.ecand = ne >= 64 ? ~0ull : ((1ull << ne) - 1);
        }
        else
        {
            f.kind = F_SCALAR;
            f.stype = (k == Prog::K_STR)    ? T_STR
                      : (k == Prog::K_INT)  ? T_INT
                      : (k == Prog::K_NUM)  ? T_NUM
                      : (k == Prog::K_BOOL) ? T_BOOL
                      : (k == Prog::K_NULL) ? T_NULL
                                            : T_ANY;
        }
        st[n++] = f;
    }
    void value_done()
    {
        --n;
        if (n > 0)
        {
            Frame &p = st[n - 1];
            if (p.kind == F_OBJECT)
            {
                p.ostate = 4;
            }
            else if (p.kind == F_ARRAY)
            {
                p.ostate = 3;
            }
        }
    }

    bool complete() const
    {
        if (!started)
        {
            return false;
        }
        if (n == 0)
        {
            return true;
        }
        if (n != 1)
        {
            return false;
        }
        const Frame &f = st[0];
        if (f.kind == F_SCALAR)
        {
            return f.js.innum && f.num_ok && f.js.stk.size() == 1;
        }
        if (f.kind == F_ENUM)
        {
            for (int j = 0; j < nd(f.node).nenum; ++j)
            {
                if ((f.ecand >> j) & 1 && (int)prog->enums[nd(f.node).enum0 + j].size() == f.mpos)
                {
                    return true;
                }
            }
        }
        return false;
    }

    static uint64_t fullmask(int nprop)
    {
        return nprop >= 64 ? ~0ull : ((1ull << nprop) - 1);
    }

    static bool type_opens(SType t, char c)
    {
        switch (t)
        {
        case T_STR:
            return c == '"';
        case T_INT:
        case T_NUM:
            return c == '-' || std::isdigit(static_cast<unsigned char>(c));
        case T_BOOL:
            return c == 't' || c == 'f';
        case T_NULL:
            return c == 'n';
        case T_ANY:
            return c == '"' || c == '{' || c == '[' || c == 't' || c == 'f' || c == 'n' || c == '-' ||
                   std::isdigit(static_cast<unsigned char>(c));
        default:
            return false;
        }
    }

    bool step(char c)
    {
        if (n == 0)
        {
            return is_ws(c);
        }
        if (n >= MAXD)
        {
            return false; // over-deep
        }
        Frame &f = st[n - 1];

        if (f.kind == F_SCALAR)
        {
            if (!f.opened)
            {
                if (is_ws(c))
                {
                    return true;
                }
                if (!type_opens(f.stype, c))
                {
                    return false;
                }
                f.opened = true;
                started = true;
            }
            else
            {
                if (f.js.innum && f.js.stk.size() == 1 && is_term(c))
                { // top-of-value number ends
                    if (!f.num_ok)
                    {
                        return false;
                    }
                    value_done();
                    return step(c);
                }
                if (f.stype == T_INT && f.js.innum && (c == '.' || c == 'e' || c == 'E'))
                {
                    return false;
                }
            }
            if (!jsong::step(f.js, c))
            {
                return false;
            }
            f.num_ok = f.js.innum && std::isdigit(static_cast<unsigned char>(c));
            if (f.js.complete())
            {
                value_done();
            }
            return true;
        }

        if (f.kind == F_ENUM)
        {
            const Prog::Node &node = nd(f.node);
            if (!f.opened)
            {
                if (is_ws(c))
                {
                    return true;
                }
                f.opened = true;
                started = true;
            }
            else if (is_term(c))
            { // a matched non-self-delimiting literal (number) ends
                for (int j = 0; j < node.nenum; ++j)
                {
                    if ((f.ecand >> j) & 1 && (int)prog->enums[node.enum0 + j].size() == f.mpos)
                    {
                        value_done();
                        return step(c);
                    }
                }
            }
            uint64_t keep = 0;
            for (int j = 0; j < node.nenum; ++j)
            {
                if (!((f.ecand >> j) & 1))
                {
                    continue;
                }
                const std::string &s = prog->enums[node.enum0 + j];
                if (f.mpos < s.size() && s[f.mpos] == c)
                {
                    keep |= (1ull << j);
                }
            }
            if (!keep)
            {
                return false;
            }
            f.ecand = keep;
            f.mpos++;
            int only = -1, cnt = 0;
            for (int j = 0; j < node.nenum; ++j)
            {
                if ((keep >> j) & 1)
                {
                    only = j;
                    cnt++;
                }
            }
            if (cnt == 1 && (int)prog->enums[node.enum0 + only].size() == f.mpos)
            {
                value_done();
            }
            return true;
        }

        if (f.kind == F_OBJECT)
        {
            const Prog::Node &node = nd(f.node);
            auto start_key = [&]() {
                f.keycand = fullmask(node.nprop) & ~f.used;
                f.keylen = 0;
                f.addkey = false;
                f.selkey = -1;
            };
            switch (f.ostate)
            {
            case 0:
                if (is_ws(c))
                {
                    return true;
                }
                if (c != '{')
                {
                    return false;
                }
                started = true;
                f.ostate = 1;
                return true;
            case 1: // key or '}'
                if (is_ws(c))
                {
                    return true;
                }
                if (c == '}')
                {
                    if ((node.required & ~f.used) != 0)
                    {
                        return false;
                    }
                    value_done();
                    return true;
                }
                if (c == '"')
                {
                    f.ostate = 2;
                    start_key();
                    return true;
                }
                return false;
            case 2: // inside key string
                if (c == '"')
                {
                    if (!f.addkey)
                    { // must select an exact property (unless additional)
                        int sel = -1;
                        for (int i = 0; i < node.nprop; ++i)
                        {
                            if ((f.keycand >> i) & 1 && (int)prog->names[node.prop0 + i].size() == f.keylen)
                            {
                                sel = i;
                                break;
                            }
                        }
                        if (sel < 0)
                        {
                            if (!node.additional)
                            {
                                return false;
                            }
                            f.selkey = -1;
                        }
                        else
                        {
                            f.selkey = sel;
                        }
                    }
                    else
                    {
                        f.selkey = -1;
                    }
                    f.ostate = 3;
                    return true;
                }
                if (static_cast<unsigned char>(c) < 0x20)
                {
                    return false;
                }
                if (f.addkey)
                {
                    return true; // arbitrary key: accept any string byte
                }
                {
                    uint64_t nk = 0;
                    for (int i = 0; i < node.nprop; ++i)
                    {
                        if ((f.keycand >> i) & 1)
                        {
                            const std::string &nm = prog->names[node.prop0 + i];
                            if ((int)nm.size() > f.keylen && nm[f.keylen] == c)
                            {
                                nk |= (1ull << i);
                            }
                        }
                    }
                    f.keycand = nk;
                    f.keylen++;
                    if (!nk)
                    {
                        if (!node.additional)
                        {
                            return false;
                        }
                        f.addkey = true;
                    }
                    return true;
                }
            case 3: // ':'
                if (is_ws(c))
                {
                    return true;
                }
                if (c != ':')
                {
                    return false;
                }
                {
                    int sel = f.selkey;
                    if (sel >= 0)
                    {
                        f.used |= (1ull << sel);
                    }
                    int child = sel >= 0 ? prog->childs[node.prop0 + sel] : -1;
                    f.ostate = 5;
                    push(child);
                }
                return true;
            case 4: // ',' or '}'
                if (is_ws(c))
                {
                    return true;
                }
                if (c == ',')
                {
                    bool avail = node.additional || ((fullmask(node.nprop) & ~f.used) != 0);
                    if (!avail)
                    {
                        return false;
                    }
                    f.ostate = 6;
                    return true;
                }
                if (c == '}')
                {
                    if ((node.required & ~f.used) != 0)
                    {
                        return false;
                    }
                    value_done();
                    return true;
                }
                return false;
            case 6: // key required
                if (is_ws(c))
                {
                    return true;
                }
                if (c == '"')
                {
                    f.ostate = 2;
                    start_key();
                    return true;
                }
                return false;
            default:
                return false;
            }
        }

        if (f.kind == F_ARRAY)
        {
            const Prog::Node &node = nd(f.node);
            switch (f.ostate)
            {
            case 0:
                if (is_ws(c))
                {
                    return true;
                }
                if (c != '[')
                {
                    return false;
                }
                started = true;
                f.ostate = 1;
                return true;
            case 1: // element or ']'
                if (is_ws(c))
                {
                    return true;
                }
                if (c == ']')
                {
                    if (f.items < node.min_items)
                    {
                        return false;
                    }
                    value_done();
                    return true;
                }
                f.ostate = 2;
                f.items++;
                push(node.item);
                return step(c);
            case 3: // ',' or ']'
                if (is_ws(c))
                {
                    return true;
                }
                if (c == ',')
                {
                    f.ostate = 2;
                    f.items++;
                    push(node.item);
                    return true;
                }
                if (c == ']')
                {
                    if (f.items < node.min_items)
                    {
                        return false;
                    }
                    value_done();
                    return true;
                }
                return false;
            default:
                return false;
            }
        }
        return false;
    }
};

} // namespace jsons

// Schema-constrained guided-decoding constraint (parallels JsonConstraint).
struct SchemaConstraint
{
    const TokenPieces &tp;
    Json schema; // owns the parsed schema (Prog copies out of it at build)
    jsons::Prog prog;
    jsons::machine m;
    mutable MaskCache cache_;

    SchemaConstraint(const TokenPieces &p, const std::string &schema_json) : tp(p), schema(Json::parse(schema_json))
    {
        prog.compile(schema);
        m = jsons::machine(&prog);
        cache_.init(static_cast<int>(p.V), sizeof(m));
    }

    void mask(uint8_t *allowed) const
    {
        const uint64_t sh = MaskCache::hash(&m, sizeof(m));
        if (const uint8_t *c = cache_.find(&m, sizeof(m), sh))
        {
            std::memcpy(allowed, c, static_cast<size_t>(tp.V));
            return;
        }
        const bool done = m.complete();
        const bool force = done && m.n == 0; // self-delimiting value closed -> stop only

        // Copy-free trial: snapshot the LIVE frames, step the bytes, restore. No
        // heap (the state is POD), and only `m.n` (usually 2-4) frames move.
        jsons::machine s = m; // one cheap POD copy; s.step mutates, then re-snapshot
        const int n0 = m.n;
        const bool started0 = m.started;
        auto try_bytes = [&](const char *pc, size_t ln) -> bool {
            for (int i = 0; i < n0; ++i)
            {
                s.st[i] = m.st[i];
            }
            s.n = n0;
            s.started = started0;
            for (size_t i = 0; i < ln; ++i)
            {
                if (!s.step(pc[i]))
                {
                    return false;
                }
            }
            return true;
        };

        std::memset(allowed, 0, static_cast<size_t>(tp.V)); // default: deny
        for (int32_t id : tp.eog_ids)
        {
            allowed[id] = done ? 1 : 0; // stop only when complete
        }
        if (!force)
        { // force => complete self-delimiting value: only EOG remains
            bool first_ok[256];
            for (int b = 0; b < 256; ++b)
            {
                for (int i = 0; i < n0; ++i)
                {
                    s.st[i] = m.st[i];
                }
                s.n = n0;
                s.started = started0;
                first_ok[b] = s.step(static_cast<char>(b));
            }
            auto walk = [&](int64_t id) {
                const size_t ln = tp.len(id);
                const char *pc = tp.piece(id);
                allowed[id] = (ln == 1) || try_bytes(pc, ln) ? 1 : 0;
            };
            if (tp.indexed)
            { // only tokens whose first byte the schema permits
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
        cache_.store(&m, sizeof(m), sh, allowed);
    }
    void accept(int32_t id)
    {
        if (id >= 0 && id < tp.V)
        {
            const char *p = tp.piece(id);
            size_t nb = tp.len(id);
            for (size_t i = 0; i < nb; ++i)
            {
                m.step(p[i]);
            }
        }
    }
    bool complete() const
    {
        return m.complete();
    }
};

} // namespace oracle
