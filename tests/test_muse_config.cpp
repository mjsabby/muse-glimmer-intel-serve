// muse::parse_config — the traps that make Muse Glimmer different from every
// other Gemma-shaped checkpoint, asserted against synthetic configs.
#include <string>

#include "dflash.hpp"
#include "muse_glimmer.hpp"
#include "test_util.h"

namespace
{

    // A 8-layer miniature of the real text_config: [sliding x3, full] x2,
    // NoPE on the full layers, both eps values, all three scalars.
    std::string tiny_config(const std::string &extra = "", int layers = 8)
    {
        std::string lt, lrt;
        for (int i = 0; i < layers; ++i)
        {
            const bool full = ((i + 1) % 4) == 0;
            lt += std::string(i ? "," : "") + (full ? "\"full_attention\"" : "\"sliding_attention\"");
            lrt += std::string(i ? "," : "") + (full ? "0" : "500000.0");
        }
        return "{\n"
               "  \"model_type\": \"muse_glimmer\",\n"
               "  \"image_token_id\": 200092,\n"
               "  \"video_token_id\": 200091,\n"
               "  \"out_hidden_size\": 6144,\n"
               "  \"text_config\": {\n"
               "    \"model_type\": \"muse_glimmer_text\",\n"
               "    \"hidden_size\": 64,\n"
               "    \"num_hidden_layers\": " + std::to_string(layers) + ",\n"
               "    \"intermediate_size\": 128,\n"
               "    \"num_attention_heads\": 4,\n"
               "    \"num_key_value_heads\": 1,\n"
               "    \"head_dim\": 16,\n"
               "    \"vocab_size\": 512,\n"
               "    \"sliding_window\": 8,\n"
               "    \"rms_norm_eps\": 1e-05,\n"
               "    \"post_norm_eps\": 1e-08,\n"
               "    \"qk_scale_factor\": 3.87,\n"
               "    \"output_multiplier\": 0.19611613513818404,\n"
               "    \"final_logit_softcapping\": 20.0,\n"
               "    \"hidden_activation\": \"silu\",\n"
               "    \"attention_bias\": false,\n"
               "    \"tie_word_embeddings\": false,\n"
               "    \"max_position_embeddings\": 4096,\n"
               "    \"eos_token_id\": [200001, 200008],\n"
               "    \"rope_parameters\": {\"rope_type\": \"default\", \"rope_theta\": 500000.0},\n"
               "    \"layer_types\": [" + lt + "],\n"
               "    \"layer_rope_theta\": [" + lrt + "]" + extra + "\n"
               "  }\n"
               "}\n";
    }

    bool parse_throws(const std::string &json)
    {
        try
        {
            auto v = minijson::parse(json);
            muse::parse_config(*v);
            return false;
        }
        catch (const std::exception &)
        {
            return true;
        }
    }

} // namespace

void test_muse_config()
{
    auto root = minijson::parse(tiny_config());
    muse::Config c = muse::parse_config(*root);

    CHECK(c.hidden_size == 64);
    CHECK(c.num_hidden_layers == 8);
    CHECK(c.intermediate_size == 128);
    CHECK(c.num_attention_heads == 4);
    CHECK(c.num_key_value_heads == 1);
    CHECK(c.head_dim == 16);
    CHECK(c.kv_groups() == 4);
    CHECK(c.q_dim() == 64);
    CHECK(c.kv_dim() == 16);
    CHECK(c.vocab_size == 512);
    CHECK(c.sliding_window == 8);
    CHECK(!c.tie_word_embeddings);

    // the two eps values differ by three orders of magnitude and are assigned
    // by position, not by name
    CHECK_NEAR(c.rms_norm_eps, 1e-5, 0.0);
    CHECK_NEAR(c.post_norm_eps, 1e-8, 0.0);
    CHECK(c.rms_norm_eps != c.post_norm_eps);

    // the three scalars nothing else in the Gemma family has
    CHECK_NEAR(c.qk_scale_factor, 3.87, 0.0);
    CHECK_NEAR(c.output_multiplier, 0.19611613513818404, 0.0);
    CHECK_NEAR(c.final_logit_softcapping, 20.0, 0.0);
    CHECK_NEAR(c.rope_theta, 500000.0, 0.0);

    // [sliding, sliding, sliding, full] with NoPE on every full layer
    for (int64_t i = 0; i < c.num_hidden_layers; ++i)
    {
        const bool full = ((i + 1) % 4) == 0;
        CHECK(c.layer_is_sliding(i) == !full);
        CHECK(c.layer_has_rope(i) == !full); // global layers carry NO position signal
    }

    CHECK(c.image_token_id == 200092);
    CHECK(c.video_token_id == 200091);
    CHECK(c.eos_token_ids.size() == 2);
    CHECK(c.eos_token_ids[0] == 200001 && c.eos_token_ids[1] == 200008);

    // ---- things that must fail loudly rather than default silently

    // post_norm_eps missing: defaulting it to rms_norm_eps would drift every
    // logit while passing a smoke test
    {
        std::string s = tiny_config();
        size_t p = s.find("    \"post_norm_eps\": 1e-08,\n");
        CHECK(p != std::string::npos);
        s.erase(p, std::string("    \"post_norm_eps\": 1e-08,\n").size());
        CHECK(parse_throws(s));
    }
    // attention_bias true — the text stack has no bias tensors anywhere
    {
        std::string s = tiny_config();
        size_t p = s.find("\"attention_bias\": false");
        s.replace(p, std::string("\"attention_bias\": false").size(), "\"attention_bias\": true");
        CHECK(parse_throws(s));
    }
    // a per-layer theta that disagrees with rope_parameters would need a
    // second rope table; the reference builds exactly one
    CHECK(parse_throws(tiny_config(", \"layer_rope_theta\": [1,2,3,4,5,6,7,8]")));
    // wrong model type
    {
        std::string s = tiny_config();
        size_t p = s.find("\"muse_glimmer\"");
        s.replace(p, std::string("\"muse_glimmer\"").size(), "\"gemma3\"");
        CHECK(parse_throws(s));
    }
    // non-silu activation
    {
        std::string s = tiny_config();
        size_t p = s.find("\"hidden_activation\": \"silu\"");
        s.replace(p, std::string("\"hidden_activation\": \"silu\"").size(),
                  "\"hidden_activation\": \"gelu\"");
        CHECK(parse_throws(s));
    }

    // ---- the real 30B shape, so a checkpoint revision that changes it trips here
    {
        std::string lt, lrt;
        for (int i = 0; i < 52; ++i)
        {
            const bool full = ((i + 1) % 4) == 0;
            lt += std::string(i ? "," : "") + (full ? "\"full_attention\"" : "\"sliding_attention\"");
            lrt += std::string(i ? "," : "") + (full ? "0" : "500000.0");
        }
        std::string s = tiny_config();
        // swap in the real dimensions
        auto sub = [&](const std::string &from, const std::string &to)
        {
            size_t p = s.find(from);
            CHECK(p != std::string::npos);
            s.replace(p, from.size(), to);
        };
        sub("\"hidden_size\": 64", "\"hidden_size\": 6656");
        sub("\"num_hidden_layers\": 8", "\"num_hidden_layers\": 52");
        sub("\"intermediate_size\": 128", "\"intermediate_size\": 19968");
        sub("\"num_attention_heads\": 4", "\"num_attention_heads\": 32");
        sub("\"num_key_value_heads\": 1", "\"num_key_value_heads\": 2");
        sub("\"head_dim\": 16", "\"head_dim\": 128");
        sub("\"vocab_size\": 512", "\"vocab_size\": 202048");
        sub("\"sliding_window\": 8", "\"sliding_window\": 2048");
        sub("\"max_position_embeddings\": 4096", "\"max_position_embeddings\": 131072");
        size_t p = s.find("\"layer_types\": [");
        size_t e = s.find("]", p);
        s = s.substr(0, p) + "\"layer_types\": [" + lt + s.substr(e);
        p = s.find("\"layer_rope_theta\": [");
        e = s.find("]", p);
        s = s.substr(0, p) + "\"layer_rope_theta\": [" + lrt + s.substr(e);

        auto r2 = minijson::parse(s);
        muse::Config big = muse::parse_config(*r2);
        CHECK(big.num_hidden_layers == 52);
        CHECK(big.kv_groups() == 16);
        CHECK(big.q_dim() == 4096); // narrower than the 6656 residual
        CHECK(big.kv_dim() == 256);
        int64_t sliding = 0, rotated = 0;
        for (int64_t i = 0; i < 52; ++i)
        {
            sliding += big.layer_is_sliding(i);
            rotated += big.layer_has_rope(i);
        }
        CHECK(sliding == 39); // 39 sliding + 13 global
        CHECK(rotated == 39); // only the sliding layers rotate
    }

    // ---- the DFlash drafter config (the released -assistant repo's values)
    {
        const std::string js =
            "{\n"
            "  \"model_type\": \"muse_glimmer_assistant\",\n"
            "  \"hidden_size\": 6656,\n"
            "  \"intermediate_size\": 19968,\n"
            "  \"num_hidden_layers\": 5,\n"
            "  \"num_attention_heads\": 32,\n"
            "  \"num_key_value_heads\": 8,\n"
            "  \"head_dim\": 128,\n"
            "  \"rms_norm_eps\": 1e-05,\n"
            "  \"sliding_window\": 2048,\n"
            "  \"max_position_embeddings\": 131072,\n"
            "  \"block_size\": 16,\n"
            "  \"mask_token_id\": 201818,\n"
            "  \"target_layer_ids\": [1, 13, 25, 37, 49],\n"
            "  \"rope_parameters\": {\"rope_type\": \"default\", \"rope_theta\": 500000.0},\n"
            "  \"layer_types\": [\"sliding_attention\", \"sliding_attention\", "
            "\"sliding_attention\", \"sliding_attention\", \"sliding_attention\"]\n"
            "}\n";
        auto r = minijson::parse(js);
        muse::dflash::Config d = muse::dflash::parse_config(*r);
        CHECK(d.num_hidden_layers == 5);
        CHECK(d.num_attention_heads == 32);
        CHECK(d.num_key_value_heads == 8); // GQA 4:1, unlike the target's 16:1
        CHECK(d.kv_groups() == 4);
        CHECK(d.block_size == 16);
        // a round proposes block_size - 1 tokens: the anchor row is dropped
        CHECK(d.block_size - 1 == 15);
        CHECK(d.mask_token_id == 201818);
        CHECK(d.target_layer_ids.size() == 5);
        CHECK(d.target_layer_ids[0] == 1 && d.target_layer_ids[4] == 49);
        for (int64_t i = 0; i < d.num_hidden_layers; ++i)
            CHECK(d.layer_is_sliding(i)); // every drafter layer is sliding
        CHECK_NEAR(d.rope_theta, 500000.0, 0.0);

        // block_size 1 would leave nothing to propose
        {
            std::string s = js;
            size_t p = s.find("\"block_size\": 16");
            s.replace(p, std::string("\"block_size\": 16").size(), "\"block_size\": 1");
            bool threw = false;
            try
            {
                auto v = minijson::parse(s);
                muse::dflash::parse_config(*v);
            }
            catch (const std::exception &)
            {
                threw = true;
            }
            CHECK(threw);
        }
    }
}
