#include "json.h"
#include "test_util.h"

using oracle::Json;

void test_json()
{
    Json v = Json::parse(R"({"a": 1, "b": [true, null, -2.5e3], "s": "xé😀\n", "big": 62546177752})");
    CHECK(v.at("a").as_int() == 1);
    CHECK(v.at("b").as_array().size() == 3);
    CHECK(v.at("b").as_array()[0].as_bool());
    CHECK(v.at("b").as_array()[1].is_null());
    CHECK_NEAR(v.at("b").as_array()[2].as_double(), -2500.0, 0);
    CHECK(v.at("s").as_string() == std::string("x\xc3\xa9\xf0\x9f\x98\x80\n"));
    CHECK(v.at("big").as_int() == 62546177752LL);
    CHECK(v.get("zzz") == nullptr);

    bool threw = false;
    try
    {
        Json::parse("{\"a\": }");
    }
    catch (const std::exception &)
    {
        threw = true;
    }
    CHECK(threw);

    // dump round-trip
    Json r = Json::parse(v.dump());
    CHECK(r.at("s").as_string() == v.at("s").as_string());
    CHECK(r.at("big").as_int() == 62546177752LL);
}
