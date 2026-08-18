// Minimal JSON parser — enough for config.json, model.safetensors.index.json and
// safetensors headers. Integers are kept as int64 (safetensors data_offsets exceed
// float precision needs), everything else standard. No external dependencies.
#pragma once

#include <cstdint>
#include <cstdlib>
#include <cstring> // strlen/memcmp: libstdc++ leaks these transitively, icpx does not
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace minijson
{

    struct Value;
    using ValuePtr = std::shared_ptr<Value>;

    struct Value
    {
        enum Type
        {
            Null,
            Bool,
            Int,
            Double,
            String,
            Array,
            Object
        };
        Type type = Null;
        bool b = false;
        int64_t i = 0;
        double d = 0.0;
        std::string s;
        std::vector<ValuePtr> arr;
        std::map<std::string, ValuePtr> obj;

        bool is_null() const { return type == Null; }
        bool is_number() const { return type == Int || type == Double; }

        bool as_bool() const
        {
            if (type != Bool)
                throw std::runtime_error("json: not a bool");
            return b;
        }
        int64_t as_int() const
        {
            if (type == Int)
                return i;
            throw std::runtime_error("json: not an int");
        }
        double as_double() const
        {
            if (type == Int)
                return double(i);
            if (type == Double)
                return d;
            throw std::runtime_error("json: not a number");
        }
        const std::string &as_str() const
        {
            if (type != String)
                throw std::runtime_error("json: not a string");
            return s;
        }
        const Value &at(const std::string &key) const
        {
            if (type != Object)
                throw std::runtime_error("json: not an object (want key '" + key + "')");
            auto it = obj.find(key);
            if (it == obj.end())
                throw std::runtime_error("json: missing key '" + key + "'");
            return *it->second;
        }
        // returns nullptr if absent or json-null
        const Value *opt(const std::string &key) const
        {
            if (type != Object)
                return nullptr;
            auto it = obj.find(key);
            if (it == obj.end() || it->second->type == Null)
                return nullptr;
            return it->second.get();
        }
        size_t size() const
        {
            if (type == Array)
                return arr.size();
            if (type == Object)
                return obj.size();
            throw std::runtime_error("json: no size");
        }
        const Value &operator[](size_t idx) const
        {
            if (type != Array)
                throw std::runtime_error("json: not an array");
            return *arr.at(idx);
        }
    };

    class Parser
    {
    public:
        Parser(const char *data, size_t len) : p_(data), end_(data + len) {}

        ValuePtr parse()
        {
            ValuePtr v = parse_value();
            skip_ws();
            return v;
        }

    private:
        const char *p_;
        const char *end_;

        [[noreturn]] void fail(const std::string &msg)
        {
            throw std::runtime_error("json parse error: " + msg);
        }
        void skip_ws()
        {
            while (p_ < end_ && (*p_ == ' ' || *p_ == '\t' || *p_ == '\n' || *p_ == '\r'))
                ++p_;
        }
        char peek()
        {
            skip_ws();
            if (p_ >= end_)
                fail("unexpected end");
            return *p_;
        }
        void expect(char c)
        {
            if (peek() != c)
                fail(std::string("expected '") + c + "' got '" + *p_ + "'");
            ++p_;
        }
        bool consume(char c)
        {
            if (p_ < end_ && peek() == c)
            {
                ++p_;
                return true;
            }
            return false;
        }
        bool literal(const char *lit)
        {
            size_t n = strlen(lit);
            if (size_t(end_ - p_) >= n && memcmp(p_, lit, n) == 0)
            {
                p_ += n;
                return true;
            }
            return false;
        }

        ValuePtr parse_value()
        {
            char c = peek();
            auto v = std::make_shared<Value>();
            switch (c)
            {
            case '{':
                parse_object(*v);
                break;
            case '[':
                parse_array(*v);
                break;
            case '"':
                v->type = Value::String;
                v->s = parse_string();
                break;
            case 't':
                if (!literal("true"))
                    fail("bad literal");
                v->type = Value::Bool;
                v->b = true;
                break;
            case 'f':
                if (!literal("false"))
                    fail("bad literal");
                v->type = Value::Bool;
                v->b = false;
                break;
            case 'n':
                if (!literal("null"))
                    fail("bad literal");
                v->type = Value::Null;
                break;
            default:
                parse_number(*v);
                break;
            }
            return v;
        }

        void parse_object(Value &v)
        {
            v.type = Value::Object;
            expect('{');
            if (consume('}'))
                return;
            while (true)
            {
                skip_ws();
                std::string key = parse_string();
                expect(':');
                v.obj[key] = parse_value();
                if (consume(','))
                    continue;
                expect('}');
                break;
            }
        }

        void parse_array(Value &v)
        {
            v.type = Value::Array;
            expect('[');
            if (consume(']'))
                return;
            while (true)
            {
                v.arr.push_back(parse_value());
                if (consume(','))
                    continue;
                expect(']');
                break;
            }
        }

        std::string parse_string()
        {
            if (peek() != '"')
                fail("expected string");
            ++p_;
            std::string out;
            while (p_ < end_ && *p_ != '"')
            {
                char c = *p_++;
                if (c != '\\')
                {
                    out += c;
                    continue;
                }
                if (p_ >= end_)
                    fail("bad escape");
                char e = *p_++;
                switch (e)
                {
                case '"':
                    out += '"';
                    break;
                case '\\':
                    out += '\\';
                    break;
                case '/':
                    out += '/';
                    break;
                case 'b':
                    out += '\b';
                    break;
                case 'f':
                    out += '\f';
                    break;
                case 'n':
                    out += '\n';
                    break;
                case 'r':
                    out += '\r';
                    break;
                case 't':
                    out += '\t';
                    break;
                case 'u':
                {
                    if (end_ - p_ < 4)
                        fail("bad \\u");
                    unsigned cp = 0;
                    for (int k = 0; k < 4; ++k)
                    {
                        char h = *p_++;
                        cp <<= 4;
                        if (h >= '0' && h <= '9')
                            cp |= unsigned(h - '0');
                        else if (h >= 'a' && h <= 'f')
                            cp |= unsigned(h - 'a' + 10);
                        else if (h >= 'A' && h <= 'F')
                            cp |= unsigned(h - 'A' + 10);
                        else
                            fail("bad \\u hex");
                    }
                    // surrogate pair
                    if (cp >= 0xD800 && cp <= 0xDBFF && end_ - p_ >= 6 && p_[0] == '\\' && p_[1] == 'u')
                    {
                        unsigned lo = 0;
                        const char *q = p_ + 2;
                        for (int k = 0; k < 4; ++k)
                        {
                            char h = *q++;
                            lo <<= 4;
                            if (h >= '0' && h <= '9')
                                lo |= unsigned(h - '0');
                            else if (h >= 'a' && h <= 'f')
                                lo |= unsigned(h - 'a' + 10);
                            else if (h >= 'A' && h <= 'F')
                                lo |= unsigned(h - 'A' + 10);
                            else
                            {
                                lo = 0xFFFFFFFF;
                                break;
                            }
                        }
                        if (lo >= 0xDC00 && lo <= 0xDFFF)
                        {
                            cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                            p_ = q;
                        }
                    }
                    // encode UTF-8
                    if (cp < 0x80)
                        out += char(cp);
                    else if (cp < 0x800)
                    {
                        out += char(0xC0 | (cp >> 6));
                        out += char(0x80 | (cp & 0x3F));
                    }
                    else if (cp < 0x10000)
                    {
                        out += char(0xE0 | (cp >> 12));
                        out += char(0x80 | ((cp >> 6) & 0x3F));
                        out += char(0x80 | (cp & 0x3F));
                    }
                    else
                    {
                        out += char(0xF0 | (cp >> 18));
                        out += char(0x80 | ((cp >> 12) & 0x3F));
                        out += char(0x80 | ((cp >> 6) & 0x3F));
                        out += char(0x80 | (cp & 0x3F));
                    }
                    break;
                }
                default:
                    fail("bad escape char");
                }
            }
            if (p_ >= end_)
                fail("unterminated string");
            ++p_; // closing quote
            return out;
        }

        void parse_number(Value &v)
        {
            const char *start = p_;
            bool is_float = false;
            if (p_ < end_ && (*p_ == '-' || *p_ == '+'))
                ++p_;
            while (p_ < end_)
            {
                char c = *p_;
                if (c >= '0' && c <= '9')
                    ++p_;
                else if (c == '.' || c == 'e' || c == 'E' || c == '+' || c == '-')
                {
                    is_float = true;
                    ++p_;
                }
                else
                    break;
            }
            if (p_ == start)
                fail("bad number");
            std::string num(start, size_t(p_ - start));
            if (!is_float)
            {
                v.type = Value::Int;
                v.i = strtoll(num.c_str(), nullptr, 10);
                v.d = double(v.i);
            }
            else
            {
                v.type = Value::Double;
                v.d = strtod(num.c_str(), nullptr);
            }
        }
    };

    inline ValuePtr parse(const std::string &text)
    {
        Parser p(text.data(), text.size());
        return p.parse();
    }
    inline ValuePtr parse(const char *data, size_t len)
    {
        Parser p(data, len);
        return p.parse();
    }

} // namespace minijson
