#include "json.h"

#include <cerrno>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace oracle
{

namespace
{

class Parser
{
  public:
    explicit Parser(std::string_view s) : s_(s)
    {
    }

    Json parse()
    {
        Json v = value();
        skip_ws();
        if (pos_ != s_.size())
        {
            fail("trailing data after JSON value");
        }
        return v;
    }

  private:
    [[noreturn]] void fail(const std::string &msg) const
    {
        throw std::runtime_error("json parse error at byte " + std::to_string(pos_) + ": " + msg);
    }

    void skip_ws()
    {
        while (pos_ < s_.size())
        {
            char c = s_[pos_];
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r')
            {
                ++pos_;
            }
            else
            {
                break;
            }
        }
    }

    char peek()
    {
        if (pos_ >= s_.size())
        {
            fail("unexpected end of input");
        }
        return s_[pos_];
    }

    void expect(char c)
    {
        if (pos_ >= s_.size() || s_[pos_] != c)
        {
            fail(std::string("expected '") + c + "'");
        }
        ++pos_;
    }

    bool consume_lit(std::string_view lit)
    {
        if (s_.substr(pos_, lit.size()) == lit)
        {
            pos_ += lit.size();
            return true;
        }
        return false;
    }

    Json value()
    {
        skip_ws();
        char c = peek();
        switch (c)
        {
        case '{':
            return object();
        case '[':
            return array();
        case '"':
            return Json(string());
        case 't':
            if (consume_lit("true"))
            {
                return Json(true);
            }
            fail("bad literal");
        case 'f':
            if (consume_lit("false"))
            {
                return Json(false);
            }
            fail("bad literal");
        case 'n':
            if (consume_lit("null"))
            {
                return Json();
            }
            fail("bad literal");
        default:
            return number();
        }
    }

    Json object()
    {
        expect('{');
        JsonObject o;
        skip_ws();
        if (peek() == '}')
        {
            ++pos_;
            return Json(std::move(o));
        }
        while (true)
        {
            skip_ws();
            std::string key = string();
            skip_ws();
            expect(':');
            o.emplace(std::move(key), value());
            skip_ws();
            char c = peek();
            if (c == ',')
            {
                ++pos_;
                continue;
            }
            if (c == '}')
            {
                ++pos_;
                break;
            }
            fail("expected ',' or '}' in object");
        }
        return Json(std::move(o));
    }

    Json array()
    {
        expect('[');
        JsonArray a;
        skip_ws();
        if (peek() == ']')
        {
            ++pos_;
            return Json(std::move(a));
        }
        while (true)
        {
            a.push_back(value());
            skip_ws();
            char c = peek();
            if (c == ',')
            {
                ++pos_;
                continue;
            }
            if (c == ']')
            {
                ++pos_;
                break;
            }
            fail("expected ',' or ']' in array");
        }
        return Json(std::move(a));
    }

    // Appends UTF-8 encoding of a code point.
    static void append_utf8(std::string &out, uint32_t cp)
    {
        if (cp <= 0x7F)
        {
            out.push_back(static_cast<char>(cp));
        }
        else if (cp <= 0x7FF)
        {
            out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        }
        else if (cp <= 0xFFFF)
        {
            out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        }
        else
        {
            out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        }
    }

    uint32_t hex4()
    {
        if (pos_ + 4 > s_.size())
        {
            fail("truncated \\u escape");
        }
        uint32_t v = 0;
        for (int i = 0; i < 4; ++i)
        {
            char c = s_[pos_++];
            v <<= 4;
            if (c >= '0' && c <= '9')
            {
                v |= static_cast<uint32_t>(c - '0');
            }
            else if (c >= 'a' && c <= 'f')
            {
                v |= static_cast<uint32_t>(c - 'a' + 10);
            }
            else if (c >= 'A' && c <= 'F')
            {
                v |= static_cast<uint32_t>(c - 'A' + 10);
            }
            else
            {
                fail("bad hex digit in \\u escape");
            }
        }
        return v;
    }

    std::string string()
    {
        expect('"');
        std::string out;
        // Fast path: scan for a segment without escapes/quotes.
        while (true)
        {
            size_t start = pos_;
            while (pos_ < s_.size())
            {
                unsigned char c = static_cast<unsigned char>(s_[pos_]);
                if (c == '"' || c == '\\')
                {
                    break;
                }
                if (c < 0x20)
                {
                    fail("unescaped control character in string");
                }
                ++pos_;
            }
            out.append(s_.data() + start, pos_ - start);
            if (pos_ >= s_.size())
            {
                fail("unterminated string");
            }
            if (s_[pos_] == '"')
            {
                ++pos_;
                return out;
            }
            // escape
            ++pos_; // backslash
            if (pos_ >= s_.size())
            {
                fail("truncated escape");
            }
            char e = s_[pos_++];
            switch (e)
            {
            case '"':
                out.push_back('"');
                break;
            case '\\':
                out.push_back('\\');
                break;
            case '/':
                out.push_back('/');
                break;
            case 'b':
                out.push_back('\b');
                break;
            case 'f':
                out.push_back('\f');
                break;
            case 'n':
                out.push_back('\n');
                break;
            case 'r':
                out.push_back('\r');
                break;
            case 't':
                out.push_back('\t');
                break;
            case 'u': {
                uint32_t cp = hex4();
                if (cp >= 0xD800 && cp <= 0xDBFF)
                { // high surrogate
                    if (pos_ + 1 < s_.size() && s_[pos_] == '\\' && s_[pos_ + 1] == 'u')
                    {
                        pos_ += 2;
                        uint32_t lo = hex4();
                        if (lo < 0xDC00 || lo > 0xDFFF)
                        {
                            fail("invalid low surrogate");
                        }
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                    }
                    else
                    {
                        fail("lone high surrogate");
                    }
                }
                else if (cp >= 0xDC00 && cp <= 0xDFFF)
                {
                    fail("lone low surrogate");
                }
                append_utf8(out, cp);
                break;
            }
            default:
                fail("bad escape character");
            }
        }
    }

    Json number()
    {
        size_t start = pos_;
        if (peek() == '-')
        {
            ++pos_;
        }
        while (pos_ < s_.size() && s_[pos_] >= '0' && s_[pos_] <= '9')
        {
            ++pos_;
        }
        bool integral = true;
        if (pos_ < s_.size() && s_[pos_] == '.')
        {
            integral = false;
            ++pos_;
            while (pos_ < s_.size() && s_[pos_] >= '0' && s_[pos_] <= '9')
            {
                ++pos_;
            }
        }
        if (pos_ < s_.size() && (s_[pos_] == 'e' || s_[pos_] == 'E'))
        {
            integral = false;
            ++pos_;
            if (pos_ < s_.size() && (s_[pos_] == '+' || s_[pos_] == '-'))
            {
                ++pos_;
            }
            while (pos_ < s_.size() && s_[pos_] >= '0' && s_[pos_] <= '9')
            {
                ++pos_;
            }
        }
        if (pos_ == start || (s_[start] == '-' && pos_ == start + 1))
        {
            fail("bad number");
        }
        std::string_view tok = s_.substr(start, pos_ - start);
        if (integral)
        {
            int64_t iv = 0;
            auto [p, ec] = std::from_chars(tok.data(), tok.data() + tok.size(), iv);
            if (ec == std::errc() && p == tok.data() + tok.size())
            {
                return Json(iv);
            }
            // fall through to double on overflow
        }
        double dv = 0.0;
        auto [p, ec] = std::from_chars(tok.data(), tok.data() + tok.size(), dv);
        if (ec != std::errc() || p != tok.data() + tok.size())
        {
            fail("bad number");
        }
        return Json(dv);
    }

    std::string_view s_;
    size_t pos_ = 0;
};

void dump_string(const std::string &s, std::string &out)
{
    out.push_back('"');
    for (unsigned char c : s)
    {
        switch (c)
        {
        case '"':
            out += "\\\"";
            break;
        case '\\':
            out += "\\\\";
            break;
        case '\b':
            out += "\\b";
            break;
        case '\f':
            out += "\\f";
            break;
        case '\n':
            out += "\\n";
            break;
        case '\r':
            out += "\\r";
            break;
        case '\t':
            out += "\\t";
            break;
        default:
            if (c < 0x20)
            {
                char buf[8];
                std::snprintf(buf, sizeof buf, "\\u%04x", c);
                out += buf;
            }
            else
            {
                out.push_back(static_cast<char>(c));
            }
        }
    }
    out.push_back('"');
}

void dump_value(const Json &v, std::string &out)
{
    switch (v.type())
    {
    case Json::Type::Null:
        out += "null";
        break;
    case Json::Type::Bool:
        out += v.as_bool() ? "true" : "false";
        break;
    case Json::Type::Int:
        out += std::to_string(v.as_int());
        break;
    case Json::Type::Double: {
        char buf[32];
        std::snprintf(buf, sizeof buf, "%.17g", v.as_double());
        out += buf;
        break;
    }
    case Json::Type::String:
        dump_string(v.as_string(), out);
        break;
    case Json::Type::Array: {
        out.push_back('[');
        const auto &a = v.as_array();
        for (size_t i = 0; i < a.size(); ++i)
        {
            if (i)
            {
                out.push_back(',');
            }
            dump_value(a[i], out);
        }
        out.push_back(']');
        break;
    }
    case Json::Type::Object: {
        out.push_back('{');
        bool first = true;
        for (const auto &[k, val] : v.as_object())
        {
            if (!first)
            {
                out.push_back(',');
            }
            first = false;
            dump_string(k, out);
            out.push_back(':');
            dump_value(val, out);
        }
        out.push_back('}');
        break;
    }
    }
}

} // namespace

Json Json::parse(std::string_view text)
{
    return Parser(text).parse();
}

Json Json::parse_file(const std::string &path)
{
    return parse(read_file(path));
}

std::string Json::dump() const
{
    std::string out;
    dump_value(*this, out);
    return out;
}

std::string read_file(const std::string &path)
{
    std::FILE *f = std::fopen(path.c_str(), "rb");
    if (!f)
    {
        throw std::runtime_error("cannot open file: " + path + " (" + std::strerror(errno) + ")");
    }
    std::fseek(f, 0, SEEK_END);
    long size = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    std::string buf(static_cast<size_t>(size), '\0');
    size_t rd = std::fread(buf.data(), 1, buf.size(), f);
    std::fclose(f);
    if (rd != buf.size())
    {
        throw std::runtime_error("short read: " + path);
    }
    return buf;
}

} // namespace oracle
