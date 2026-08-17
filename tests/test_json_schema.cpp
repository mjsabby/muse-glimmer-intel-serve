#include <string>

#include "json_schema.h"
#include "test_util.h"

using namespace oracle;

namespace
{
// Feed `s` byte-by-byte; true if every byte is accepted. *complete = terminal.
bool accepts(const std::string &schema, const std::string &s, bool *complete = nullptr)
{
    Json sch = Json::parse(schema);
    jsons::Prog prog;
    prog.compile(sch);
    jsons::machine m(&prog);
    for (char c : s)
    {
        if (!m.step(c))
        {
            return false;
        }
    }
    if (complete)
    {
        *complete = m.complete();
    }
    return true;
}
int reject_at(const std::string &schema, const std::string &s)
{
    Json sch = Json::parse(schema);
    jsons::Prog prog;
    prog.compile(sch);
    jsons::machine m(&prog);
    for (size_t i = 0; i < s.size(); ++i)
    {
        if (!m.step(s[i]))
        {
            return static_cast<int>(i);
        }
    }
    return -1;
}
} // namespace

void test_json_schema()
{
    bool comp = false;

    // ---- object: properties, types, required, no additional ----
    const std::string obj =
        R"({"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"}},"required":["name"]})";
    CHECK(accepts(obj, R"({"name":"bob","age":3})", &comp) && comp);
    CHECK(accepts(obj, R"({"age":3,"name":"bob"})", &comp) && comp); // any key order
    CHECK(accepts(obj, R"({"name":"x"})", &comp) && comp);           // age optional
    CHECK(!accepts(obj, R"({"age":3})"));                            // required 'name' missing
    CHECK(reject_at(obj, R"({"age":3})") == 8);                      // rejected at the closing '}'
    CHECK(!accepts(obj, R"({"name":3})"));                           // 'name' must be a string
    CHECK(!accepts(obj, R"({"x":1})"));                              // unknown key (no additional)
    CHECK(reject_at(obj, R"({"x":1})") == 2);                        // rejected at the 'x'
    CHECK(!accepts(obj, R"({"name":"a","name":"b"})"));              // duplicate key
    CHECK(!accepts(obj, "[]"));                                      // wrong top-level type

    // additionalProperties:true allows arbitrary keys
    const std::string open_obj = R"({"type":"object","additionalProperties":true})";
    CHECK(accepts(open_obj, R"({"anything":1,"x":[1,2]})", &comp) && comp);

    // ---- array of typed items + minItems ----
    const std::string arr = R"({"type":"array","items":{"type":"integer"},"minItems":1})";
    CHECK(accepts(arr, "[1,2,3]", &comp) && comp);
    CHECK(accepts(arr, "[ 1 , 2 ]", &comp) && comp); // whitespace ok
    CHECK(!accepts(arr, R"([1,"a"])"));              // item must be integer
    CHECK(!accepts(arr, "[]"));                      // minItems 1

    // ---- enum / const ----
    const std::string en = R"({"enum":["red","green","blue"]})";
    CHECK(accepts(en, R"("green")", &comp) && comp);
    CHECK(!accepts(en, R"("gray")"));
    CHECK(accepts(R"({"const":"yes"})", R"("yes")", &comp) && comp);
    CHECK(!accepts(R"({"const":"yes"})", R"("no")"));

    // numeric enum with a prefix relationship (1 is a prefix of 12)
    const std::string ne = R"({"enum":[1,12,2]})";
    CHECK(accepts(ne, "1", &comp) && comp); // top-level number, completable as-is
    CHECK(accepts(ne, "12", &comp) && comp);
    CHECK(!accepts(ne, "13"));
    CHECK(accepts("{\"type\":\"array\",\"items\":" + ne + "}", "[1,12]", &comp) && comp); // nested terminate

    // ---- scalars: integer vs number, boolean, null, string ----
    CHECK(accepts(R"({"type":"integer"})", "42", &comp) && comp);
    CHECK(accepts(R"({"type":"integer"})", "-7", &comp) && comp);
    CHECK(!accepts(R"({"type":"integer"})", "4.2")); // integer forbids fraction
    CHECK(accepts(R"({"type":"number"})", "4.2", &comp) && comp);
    CHECK(accepts(R"({"type":"number"})", "1e5", &comp) && comp);
    CHECK(accepts(R"({"type":"boolean"})", "true", &comp) && comp);
    CHECK(!accepts(R"({"type":"boolean"})", "tru3"));
    CHECK(accepts(R"({"type":"null"})", "null", &comp) && comp);
    CHECK(accepts(R"({"type":"string"})", R"("hi \"q\" tab\t")", &comp) && comp);
    CHECK(!accepts(R"({"type":"string"})", "5"));

    // ---- nesting ----
    const std::string nest =
        R"({"type":"object","properties":{"pt":{"type":"object","properties":{"x":{"type":"integer"},"y":{"type":"integer"}},"required":["x","y"]},"tags":{"type":"array","items":{"type":"string"}}}})";
    CHECK(accepts(nest, R"({"pt":{"x":1,"y":2},"tags":["a","b"]})", &comp) && comp);
    CHECK(!accepts(nest, R"({"pt":{"x":1}})"));       // nested required 'y' missing
    CHECK(!accepts(nest, R"({"pt":{"x":1,"z":2}})")); // nested unknown key

    // ---- unknown constructs relax to any-valid-JSON (parseable, not checked) ----
    const std::string any_sub = R"({"type":"object","properties":{"data":{}}})";
    CHECK(accepts(any_sub, R"({"data":{"whatever":[1,true,null]}})", &comp) && comp);
    CHECK(accepts(any_sub, R"({"data":"a string"})", &comp) && comp);
}
