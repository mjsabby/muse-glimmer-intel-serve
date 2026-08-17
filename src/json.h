// Minimal strict JSON parser (RFC 8259) — no external deps.
// Used for: config.json, model.safetensors.index.json, safetensors headers,
// tokenizer.json (~32 MB), and manifest output. Integers that fit int64 are
// preserved exactly (safetensors byte offsets exceed float53 comfort zone).
#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace oracle
{

class Json;
using JsonArray = std::vector<Json>;
using JsonObject = std::map<std::string, Json>;

class Json
{
  public:
    enum class Type
    {
        Null,
        Bool,
        Int,
        Double,
        String,
        Array,
        Object
    };

    Json() : type_(Type::Null)
    {
    }
    explicit Json(bool b) : type_(Type::Bool), bool_(b)
    {
    }
    explicit Json(int64_t i) : type_(Type::Int), int_(i)
    {
    }
    explicit Json(double d) : type_(Type::Double), dbl_(d)
    {
    }
    explicit Json(std::string s) : type_(Type::String), str_(std::make_shared<std::string>(std::move(s)))
    {
    }
    explicit Json(JsonArray a) : type_(Type::Array), arr_(std::make_shared<JsonArray>(std::move(a)))
    {
    }
    explicit Json(JsonObject o) : type_(Type::Object), obj_(std::make_shared<JsonObject>(std::move(o)))
    {
    }

    Type type() const
    {
        return type_;
    }
    bool is_null() const
    {
        return type_ == Type::Null;
    }
    bool is_bool() const
    {
        return type_ == Type::Bool;
    }
    bool is_int() const
    {
        return type_ == Type::Int;
    }
    bool is_number() const
    {
        return type_ == Type::Int || type_ == Type::Double;
    }
    bool is_string() const
    {
        return type_ == Type::String;
    }
    bool is_array() const
    {
        return type_ == Type::Array;
    }
    bool is_object() const
    {
        return type_ == Type::Object;
    }

    bool as_bool() const
    {
        require(Type::Bool, "bool");
        return bool_;
    }
    int64_t as_int() const
    {
        if (type_ == Type::Int)
        {
            return int_;
        }
        if (type_ == Type::Double && dbl_ == static_cast<double>(static_cast<int64_t>(dbl_)))
        {
            return static_cast<int64_t>(dbl_);
        }
        throw std::runtime_error("json: expected integer");
    }
    double as_double() const
    {
        if (type_ == Type::Int)
        {
            return static_cast<double>(int_);
        }
        require(Type::Double, "number");
        return dbl_;
    }
    const std::string &as_string() const
    {
        require(Type::String, "string");
        return *str_;
    }
    const JsonArray &as_array() const
    {
        require(Type::Array, "array");
        return *arr_;
    }
    const JsonObject &as_object() const
    {
        require(Type::Object, "object");
        return *obj_;
    }

    // Object access. at() throws with the key name; get() returns nullptr when absent.
    const Json &at(const std::string &key) const
    {
        const auto &o = as_object();
        auto it = o.find(key);
        if (it == o.end())
        {
            throw std::runtime_error("json: missing key '" + key + "'");
        }
        return it->second;
    }
    const Json *get(const std::string &key) const
    {
        if (type_ != Type::Object)
        {
            return nullptr;
        }
        auto it = obj_->find(key);
        return it == obj_->end() ? nullptr : &it->second;
    }
    bool contains(const std::string &key) const
    {
        return get(key) != nullptr;
    }

    static Json parse(std::string_view text);
    static Json parse_file(const std::string &path);

    // Serialize (used for provenance manifests). Not pretty-printed.
    std::string dump() const;

  private:
    void require(Type t, const char *name) const
    {
        if (type_ != t)
        {
            throw std::runtime_error(std::string("json: expected ") + name);
        }
    }

    Type type_;
    bool bool_ = false;
    int64_t int_ = 0;
    double dbl_ = 0.0;
    std::shared_ptr<std::string> str_;
    std::shared_ptr<JsonArray> arr_;
    std::shared_ptr<JsonObject> obj_;
};

std::string read_file(const std::string &path);

} // namespace oracle
