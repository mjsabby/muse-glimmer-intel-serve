#!/usr/bin/env python
"""HF-transformers float64 reference for one DFlash drafting round.

Runs the target once to get the hidden-state taps, then executes
`MuseGlimmerAssistantModel` exactly as
`generation/candidate_generator.py::DFlashTokenCandidateGenerator.get_candidates`
does on its first call, and dumps the proposed tokens *and* the per-position
logits — the draft tokens are discrete argmax decisions, so near-ties diverge
between implementations and a token-only gate would hide it.

The four facts this script encodes, all read out of the generation utility
rather than the model file (see ARCHITECTURE.md §"The drafting loop"):

  * one forward per block, no iterative denoising;
  * the block is `[anchor] + [MASK] * (block_size - 1)` and the candidate
    logits are `lm_head(h)[:, 1:]` — `block_size - 1` proposals, not
    `block_size`;
  * the head is the target's BARE `lm_head`: no `output_multiplier`, no softcap;
  * the anchor is embedded with the target's RAW embedding table, without
    `MuseGlimmerTextNormedEmbedding`'s weight-less norm.

Modes mirror py/ref_forward.py: --pure lifts HF's f32 casts, --fixed-reduce
additionally routes the reductions and transcendentals through the oracle's
kernels for a bitwise gate.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ref_forward as RF  # noqa: E402
from common import parse_ids, resolve_model, write_f64  # noqa: E402

import transformers  # noqa: E402
from transformers import AutoConfig, MuseGlimmerAssistantModel  # noqa: E402
from transformers.models.muse_glimmer_assistant import (  # noqa: E402
    modeling_muse_glimmer_assistant as MA,
)


def apply_assistant_pure_patches():
    """Lift the drafter's own f32 casts. Note the rounding STRUCTURE differs
    from the target's plain RMSNorm: `MuseGlimmerAssistantRMSNorm` casts the
    normalized value back to the input dtype BEFORE the weight multiply
    (`self.weight * hidden_states.to(input_dtype)`), where
    `MuseGlimmerRMSNorm` multiplies first and casts once at the end."""

    def rms_forward(self, x):
        if RF.OK is not None:
            var = RF.n2t(RF.OK.meansq(RF.t2n(x).reshape(-1, x.shape[-1])), x).reshape(
                *x.shape[:-1], 1)
        else:
            var = x.pow(2).mean(-1, keepdim=True)
        return self.weight * (x * torch.rsqrt(var + self.variance_epsilon))

    def rope_forward(self, x, position_ids):
        theta = float(self.config.rope_parameters["rope_theta"])
        dim = getattr(self.config, "head_dim", None) or (
            self.config.hidden_size // self.config.num_attention_heads)
        e = torch.arange(0, dim, 2, dtype=torch.float64) / dim
        inv = (torch.from_numpy(1.0 / RF.OK.pow(theta, RF.t2n(e)))
               if RF.OK is not None else 1.0 / (theta ** e))
        inv_expanded = inv[None, :, None].expand(position_ids.shape[0], -1, 1)
        pos = position_ids[:, None, :].to(torch.float64)
        freqs = (inv_expanded @ pos).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        if RF.OK is not None:
            cos = RF.n2t(RF.OK.cos(RF.t2n(emb)), emb).reshape(emb.shape)
            sin = RF.n2t(RF.OK.sin(RF.t2n(emb)), emb).reshape(emb.shape)
        else:
            cos, sin = emb.cos(), emb.sin()
        return (cos * self.attention_scaling).to(x.dtype), (sin * self.attention_scaling).to(x.dtype)

    def eager_pure(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
        key_states = MA.repeat_kv(key, module.num_key_value_groups)
        value_states = MA.repeat_kv(value, module.num_key_value_groups)
        if RF.OK is not None:
            return RF.oracle_order_attention(query, key_states, value_states,
                                             attention_mask, scaling)
        aw = torch.matmul(query, key_states.transpose(2, 3)) * scaling
        if attention_mask is not None:
            aw = aw + attention_mask
        aw = torch.nn.functional.softmax(aw, dim=-1)
        out = torch.matmul(aw, value_states)
        return out.transpose(1, 2).contiguous(), aw

    MA.MuseGlimmerAssistantRMSNorm.forward = rms_forward
    MA.MuseGlimmerAssistantRotaryEmbedding.forward = rope_forward
    MA.eager_attention_forward = eager_pure


def build_drafter(asnap, acfg, load):
    """The released drafter is 2.556 B parameters — 20.4 GiB in f64, which does
    not fit alongside the target. `--load streamed` materializes its five
    decoder layers one at a time (3.7 GiB each) and keeps only the context
    projection and the final norm resident (1.8 GiB)."""
    if load == "full":
        m = MuseGlimmerAssistantModel.from_pretrained(
            asnap, dtype=torch.float64, attn_implementation="eager", local_files_only=True)
        m.eval()
        return m

    weights = RF.StreamedWeights(asnap)
    torch.set_default_dtype(torch.float64)
    with torch.device("meta"):
        m = MuseGlimmerAssistantModel(acfg)
    m = m.to(torch.float64)
    m.eval()
    m.rotary_emb = m.rotary_emb.__class__(acfg)
    RF.materialize(m.encoder, "encoder.", weights)
    RF.materialize(m.norm, "norm.", weights)
    for i, layer in enumerate(m.layers):
        prefix = f"layers.{i}."

        def pre_hook(module, args, kwargs, prefix=prefix):
            RF.materialize(module, prefix, weights)
            return None

        def post_hook(module, args, kwargs, output):
            module.to("meta")
            return None

        layer.register_forward_pre_hook(pre_hook, with_kwargs=True)
        layer.register_forward_hook(post_hook, with_kwargs=True)
    return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="the TARGET snapshot dir or repo id")
    p.add_argument("--assistant", required=True, help="the DFlash drafter snapshot dir or repo id")
    p.add_argument("--ids", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--load", choices=["full", "streamed"], default="full")
    p.add_argument("--pure", action="store_true")
    p.add_argument("--fixed-reduce", action="store_true")
    p.add_argument("--kernels", default=None)
    p.add_argument("--threads", type=int, default=0)
    a = p.parse_args()

    if a.threads:
        torch.set_num_threads(a.threads)
    snap = resolve_model(a.model)
    asnap = resolve_model(a.assistant)
    ids = parse_ids(a.ids)
    os.makedirs(a.out, exist_ok=True)

    if a.fixed_reduce:
        assert a.pure, "--fixed-reduce only makes sense with --pure"
        RF.OK = RF.OracleKernels(a.kernels)

    cfg = AutoConfig.from_pretrained(snap, local_files_only=True)
    tc = cfg.get_text_config()
    cfg._attn_implementation = "eager"
    tc._attn_implementation = "eager"
    acfg = AutoConfig.from_pretrained(asnap, local_files_only=True)
    acfg._attn_implementation = "eager"

    if a.pure:
        RF.apply_pure_f64_patches(cfg)
        apply_assistant_pure_patches()
        if a.fixed_reduce:
            RF.apply_fixed_reduce_patches()

    if a.load == "full":
        model = RF.model_class(cfg).from_pretrained(
            snap, dtype=torch.float64, attn_implementation="eager", local_files_only=True)
        model.eval()
    else:
        model = RF.build_streamed(snap, cfg)
    drafter = build_drafter(asnap, acfg, a.load)

    B = acfg.block_size
    T = len(ids)
    input_ids = torch.tensor([ids], dtype=torch.long)

    with torch.inference_mode():
        out = model.model(input_ids=input_ids, use_cache=False, output_hidden_states=True)
        logits = model.lm_head(out.last_hidden_state)
        logits = RF.output_tail(logits, tc)
        anchor = int(logits[0, -1].argmax())

        # hidden_states[i + 1] is the OUTPUT of decoder layer i — the same
        # indexing DFlashTokenCandidateGenerator uses. The whole prompt is the
        # accepted context on the first round.
        ctx = torch.cat([out.hidden_states[i + 1][:, :T] for i in acfg.target_layer_ids], dim=-1)

        # the drafter embeds with the RAW table (no embed_norm)
        noise_ids = torch.tensor([[anchor] + [acfg.mask_token_id] * (B - 1)], dtype=torch.long)
        emb_w = model.model.language_model.embed_tokens.weight
        noise_embeds = F.embedding(noise_ids, emb_w).to(torch.float64)

        # positions: context 0..T-1, block T..T+B-1
        position_ids = torch.arange(T + B, dtype=torch.long)[None, :]
        attention_mask = torch.ones(1, T + B, dtype=torch.long)

        # The cache is not an optimization here, it is part of the semantics:
        # DFlashCache.get_query_offset() adds `previous_accepted_tokens`, which
        # is what places the block's queries at absolute positions T..T+B-1 for
        # the |q_idx - kv_idx| <= sliding_window overlay. Without it the block
        # would be masked as if it sat at positions 0..B-1 and a long context
        # would be windowed against the wrong centre.
        from transformers.cache_utils import DFlashCache

        cache = DFlashCache(config=acfg)
        cache.set_previous_accepted_tokens(T)
        dout = drafter(noise_embeds=noise_embeds, context_hidden_states=ctx,
                       position_ids=position_ids, attention_mask=attention_mask,
                       past_key_values=cache)
        # BARE head, anchor row dropped
        draft_logits = model.lm_head(dout.last_hidden_state)[:, 1:]
        tokens = draft_logits.argmax(-1)[0].tolist()

    write_f64(os.path.join(a.out, "logits.bin"), draft_logits[0].to(torch.float64).numpy())
    write_f64(os.path.join(a.out, "hidden.bin"),
              dout.last_hidden_state[0].to(torch.float64).numpy())
    with open(os.path.join(a.out, "meta.json"), "w") as f:
        json.dump(dict(kind="hf_reference_dflash", ids=ids, T=B - 1,
                       V=int(draft_logits.shape[-1]), block_size=B, anchor=anchor,
                       context_len=T, draft_tokens=[int(t) for t in tokens], n_hidden=0,
                       pure=a.pure, fixed_reduce=a.fixed_reduce,
                       torch=torch.__version__, transformers=transformers.__version__), f,
                  indent=1)
    print(f"anchor {anchor}, drafted {len(tokens)} tokens:", tokens)
    top = np.argsort(draft_logits[0, 0].to(torch.float64).numpy())[::-1][:5]
    print("row-0 (first proposal) top5:",
          [(int(t), round(float(draft_logits[0, 0, t]), 6)) for t in top])


if __name__ == "__main__":
    main()
