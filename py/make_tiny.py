#!/usr/bin/env python
"""Create tiny randomly-initialized Muse Glimmer checkpoints with the REAL
architecture at toy dimensions, saved through transformers itself so the
on-disk layout (nested config, model.language_model.* names, untied lm_head,
bf16 storage) matches the released checkpoint.

Three models, matching docs/plan.md Phase 1:

  tiny_text    4 layers [sliding, sliding, sliding, full], H=64, D=16,
               2 q heads / 1 kv head, I=128, V=512, window 8. Exercises the
               sandwich norms, the two eps values, weight-less QK-norm +
               qk_scale_factor, the attention output gate, NoPE-on-global and
               the softcapped scaled head.
  tiny_vision  tiny_text plus a 4-layer tower, hidden 32, 2 heads,
               [window, window, window, full], pos grid 4x4, patch 14, merge 2,
               plus the adapter and projection.
  tiny_dflash  a 2-layer DFlash drafter over tiny_text, block_size 4,
               target_layer_ids [0, 2].

Do not debug the 30B checkpoint: the oracle reaches agreement here first.
"""

import argparse
import json
import os

import torch
from transformers import (
    MuseGlimmerAssistantConfig,
    MuseGlimmerAssistantModel,
    MuseGlimmerConfig,
    MuseGlimmerForConditionalGeneration,
)

# ---------------------------------------------------------------------------
# The four per-layer text norms are stored ZERO-CENTERED ((1+w) at runtime); the
# final `norm`, the drafter's norms and every vision LayerNorm are PLAIN. A
# random init that ignores the difference makes both flavours look alike and
# hides the single most likely bug in the whole stack, so the initializer is
# explicit about which is which — and src/muse_glimmer.hpp asserts the same
# signature at load time.
CENTERED_SUFFIXES = (
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "pre_feedforward_layernorm.weight",
    "post_feedforward_layernorm.weight",
)


def is_centered(name: str) -> bool:
    # ... but the DFlash drafter's same-named norms are PLAIN RMSNorms
    return name.endswith(CENTERED_SUFFIXES)


def randomize(model, centered_ok=True, seed_note=""):
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.dim() >= 2:
                param.normal_(0.0, 0.05)
            elif name.endswith(".bias"):
                param.uniform_(-0.05, 0.05)
            elif centered_ok and is_centered(name):
                # zero-centered: stays around 0, away from the degenerate zeros
                param.uniform_(-0.5, 0.5)
            elif "norm" in name.lower() or "layernorm" in name.lower():
                # plain: stays around 1, so a centered/plain mix-up is visible
                param.uniform_(0.5, 1.5)
            else:
                param.uniform_(-0.5, 0.5)
    return model


def text_config(layers=4):
    return dict(
        vocab_size=512,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=layers,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        hidden_activation="silu",
        max_position_embeddings=4096,
        rms_norm_eps=1e-5,
        post_norm_eps=1e-8,
        qk_scale_factor=3.87,
        output_multiplier=0.19611613513818404,
        final_logit_softcapping=20.0,
        sliding_window=8,
        tie_word_embeddings=False,
        attention_bias=False,
        attention_dropout=0.0,
        bos_token_id=1,
        eos_token_id=[2, 3],
        pad_token_id=0,
        rope_parameters={"rope_type": "default", "rope_theta": 500000.0},
        # [sliding, sliding, sliding, full] with NoPE on every full layer —
        # the released pattern, counted backward from the last layer
        layer_types=[
            "full_attention" if (layers - 1 - i) % 4 == 0 else "sliding_attention"
            for i in range(layers)
        ],
        layer_rope_theta=[
            0 if (layers - 1 - i) % 4 == 0 else 500000.0 for i in range(layers)
        ],
    )


def vision_config(layers=4):
    return dict(
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=layers,
        num_attention_heads=2,
        hidden_act="gelu",
        patch_size=14,
        patch_temporal=2,
        merge_size=2,
        pos_emb_height=4,
        pos_emb_width=4,
        max_position_embeddings=16,
        layer_norm_eps=1e-5,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
        layer_types=[
            "full_attention" if (i + 1) % 4 == 0 or i == layers - 1 else "window_attention"
            for i in range(layers)
        ],
    )


def build_full(with_vision: bool, layers: int):
    tc = text_config(layers)
    vc = vision_config()
    cfg = MuseGlimmerConfig(
        text_config=tc,
        vision_config=vc,
        image_token_id=500,
        video_token_id=501,
        # out_hidden_size = vision hidden * merge_size**2
        out_hidden_size=vc["hidden_size"] * vc["merge_size"] ** 2,
        projector_hidden_size=64,
        projector_hidden_act="gelu",
        tie_word_embeddings=False,
    )
    model = MuseGlimmerForConditionalGeneration(cfg)
    randomize(model)
    if not with_vision:
        # keep the tower in the file anyway (the released checkpoint has one);
        # the text gate simply never executes it
        pass
    return model.to(torch.bfloat16)


def build_dflash(layers=2, block_size=4, target_layer_ids=(0, 2)):
    tc = text_config()
    cfg = MuseGlimmerAssistantConfig(
        hidden_size=tc["hidden_size"],
        intermediate_size=tc["intermediate_size"],
        num_hidden_layers=layers,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=tc["head_dim"],
        rms_norm_eps=tc["rms_norm_eps"],
        rope_parameters={"rope_type": "default", "rope_theta": 500000.0},
        max_position_embeddings=tc["max_position_embeddings"],
        sliding_window=tc["sliding_window"],
        hidden_act="silu",
        block_size=block_size,
        mask_token_id=500,
        target_layer_ids=list(target_layer_ids),
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    model = MuseGlimmerAssistantModel(cfg)
    # every norm in the drafter is a plain RMSNorm, including the ones whose
    # names match the target's centered ones
    randomize(model, centered_ok=False)
    return model.to(torch.bfloat16)


def save(model, out):
    os.makedirs(out, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    n = sum(p.numel() for p in model.parameters())
    with open(os.path.join(out, "config.json")) as f:
        saved = json.load(f)
    print(f"saved {saved.get('model_type', '?'):24s} {n / 1e6:7.3f}M params -> {out}")
    return saved


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="tiny", help="parent directory for the three models")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--which", default="all",
                   choices=["all", "text", "vision", "dflash"])
    a = p.parse_args()

    if a.which in ("all", "text"):
        torch.manual_seed(a.seed)
        save(build_full(with_vision=False, layers=4), os.path.join(a.out, "tiny_text"))
    if a.which in ("all", "vision"):
        torch.manual_seed(a.seed + 1)
        save(build_full(with_vision=True, layers=4), os.path.join(a.out, "tiny_vision"))
    if a.which in ("all", "dflash"):
        torch.manual_seed(a.seed + 2)
        save(build_dflash(), os.path.join(a.out, "tiny_dflash"))


if __name__ == "__main__":
    main()
