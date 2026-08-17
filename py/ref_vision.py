#!/usr/bin/env python
"""HF-transformers float64 reference for the vision path.

Runs `MuseGlimmerImageProcessor` on real images, dumps the resulting
`pixel_values` as f64 so the C++ oracle can consume exactly the same pixels,
then runs `MuseGlimmerVisionModel` + the adapter + `vision_projection` +
`perception_emb_norm` and dumps the projected features — the tensor that gets
scattered into `inputs_embeds` at the image/video placeholder positions.

Splitting it there is deliberate. Pixel ingestion (LANCZOS resize, rescale,
normalize, patchify) and the tower are two separate correctness problems: the
first is a bit-exactness question about torchvision's resampling kernel, the
second is about the model's architecture. Feeding the oracle the reference's own
pixel_values gates the second without waiting on the first.

Modes mirror py/ref_forward.py: --pure lifts HF's f32 casts, --fixed-reduce
additionally routes reductions and transcendentals through the oracle's kernels.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ref_forward as RF  # noqa: E402
from common import resolve_model, write_f64  # noqa: E402

import transformers  # noqa: E402
from transformers import AutoConfig, AutoImageProcessor  # noqa: E402
from transformers.models.muse_glimmer import modeling_muse_glimmer as MG  # noqa: E402


def apply_vision_pure_patches():
    """Lift the vision tower's own f32 casts.

    `apply_rotary_pos_emb_vision` is the notable one: it calls `.float()` on q,
    k, cos and sin, which in an f64 run is a DOWNCAST — stock loses ~7 decimal
    digits there even when everything else is f64."""

    def apply_rope_vision_pure(q, k, cos, sin):
        cos = cos.unsqueeze(-2)
        sin = sin.unsqueeze(-2)
        q_embed = (q * cos) + (MG.rotate_half(q) * sin)
        k_embed = (k * cos) + (MG.rotate_half(k) * sin)
        return q_embed, k_embed

    def vision_rope_forward(self, x, position_ids):
        theta = float(self.config.rope_parameters["rope_theta"])
        dim = getattr(self.config, "head_dim", None) or (
            self.config.hidden_size // self.config.num_attention_heads)
        spatial_dim = dim // 2
        e = torch.arange(0, spatial_dim, 2, dtype=torch.float64) / spatial_dim
        inv = (torch.from_numpy(1.0 / RF.OK.pow(theta, RF.t2n(e)))
               if RF.OK is not None else 1.0 / (theta ** e))
        inv_expanded = inv[None, :, None].expand(position_ids.shape[0], -1, 1)
        w_ids = position_ids[:, :, 0][:, None, :].to(torch.float64)
        h_ids = position_ids[:, :, 1][:, None, :].to(torch.float64)
        freq_h = (inv_expanded @ h_ids).transpose(1, 2)
        freq_w = (inv_expanded @ w_ids).transpose(1, 2)
        freq = torch.cat([freq_w, freq_h, freq_w, freq_h], dim=-1)
        if RF.OK is not None:
            cos = RF.n2t(RF.OK.cos(RF.t2n(freq)), freq).reshape(freq.shape)
            sin = RF.n2t(RF.OK.sin(RF.t2n(freq)), freq).reshape(freq.shape)
        else:
            cos, sin = freq.cos(), freq.sin()
        return (cos * self.attention_scaling).to(x.dtype), (sin * self.attention_scaling).to(x.dtype)

    MG.apply_rotary_pos_emb_vision = apply_rope_vision_pure
    MG.MuseGlimmerVisionRotaryEmbedding.forward = vision_rope_forward
    # MuseGlimmerVisionAttention resolved `apply_rotary_pos_emb_vision` at
    # module scope, so the rebind above is picked up at call time.


def build_vision_streamed(snap, cfg, with_text=False):
    """The released tower is ~1.8 B parameters (14.4 GiB in f64). Its 50 encoder
    layers are only ~37 M each, so materializing them one at a time costs ~300
    MiB peak — the same trick py/ref_forward.py uses for the text stack."""
    weights = RF.StreamedWeights(snap)
    torch.set_default_dtype(torch.float64)
    with torch.device("meta"):
        model = RF.model_class(cfg)(cfg)
    model = model.to(torch.float64)
    model.eval()

    vt = model.model.vision_tower
    vt.rotary_emb = vt.rotary_emb.__class__(cfg.vision_config)
    RF.materialize(vt.patch_embedder, "model.vision_tower.patch_embedder.", weights)
    RF.materialize(vt.ln_pre, "model.vision_tower.ln_pre.", weights)
    RF.materialize(vt.ln_post, "model.vision_tower.ln_post.", weights)
    RF.materialize(model.model.vision_adapter, "model.vision_adapter.", weights)
    RF.materialize(model.model.vision_projection, "model.vision_projection.", weights)
    for i, layer in enumerate(vt.layers):
        prefix = f"model.vision_tower.layers.{i}."

        def pre_hook(module, args, kwargs, prefix=prefix):
            RF.materialize(module, prefix, weights)
            return None

        def post_hook(module, args, kwargs, output):
            module.to("meta")
            return None

        layer.register_forward_pre_hook(pre_hook, with_kwargs=True)
        layer.register_forward_hook(post_hook, with_kwargs=True)

    if with_text:
        lm = model.model.language_model
        lm.rotary_emb = lm.rotary_emb.__class__(cfg.get_text_config())

        def materialize_raw(module, name):
            module.to_empty(device="cpu")
            module.weight = torch.nn.Parameter(weights.get(name), requires_grad=False)

        materialize_raw(lm.embed_tokens, "model.language_model.embed_tokens.weight")
        type(lm.embed_tokens).forward = lambda self, ids: self.embed_norm(
            torch.nn.functional.embedding(ids, self.weight).to(torch.float64))
        RF.materialize(lm.norm, "model.language_model.norm.", weights)
        materialize_raw(model.lm_head, "lm_head.weight")
        head = model.lm_head
        chunk = 8192

        def head_forward(self, x):
            out = torch.empty(*x.shape[:-1], self.weight.shape[0], dtype=torch.float64)
            for o0 in range(0, self.weight.shape[0], chunk):
                wq = self.weight[o0:o0 + chunk].to(torch.float64)
                if RF.OK is not None:
                    out[..., o0:o0 + chunk] = torch.from_numpy(
                        RF.OK.gemm(RF.t2n(wq), RF.t2n(x).reshape(-1, x.shape[-1]))).reshape(
                            *x.shape[:-1], -1)
                else:
                    out[..., o0:o0 + chunk] = torch.nn.functional.linear(x, wq)
            return out

        head.forward = head_forward.__get__(head)
        for i, layer in enumerate(lm.layers):
            prefix = f"model.language_model.layers.{i}."

            def tpre(module, args, kwargs, prefix=prefix):
                RF.materialize(module, prefix, weights)
                return None

            def tpost(module, args, kwargs, output):
                module.to("meta")
                return None

            layer.register_forward_pre_hook(tpre, with_kwargs=True)
            layer.register_forward_hook(tpost, with_kwargs=True)
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--load", choices=["full", "streamed"], default="full")
    p.add_argument("--images", nargs="+", required=True,
                   help="image files, or SYNTH:H,W[:seed] for a deterministic test image")
    p.add_argument("--out", required=True)
    p.add_argument("--pure", action="store_true")
    p.add_argument("--fixed-reduce", action="store_true")
    p.add_argument("--kernels", default=None)
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--ids", default=None,
                   help="token ids containing the image placeholders; runs the "
                        "full multimodal forward and dumps logits.bin too")
    a = p.parse_args()

    if a.threads:
        torch.set_num_threads(a.threads)
    snap = resolve_model(a.model)
    os.makedirs(a.out, exist_ok=True)

    if a.fixed_reduce:
        assert a.pure, "--fixed-reduce only makes sense with --pure"
        RF.OK = RF.OracleKernels(a.kernels)

    cfg = AutoConfig.from_pretrained(snap, local_files_only=True)
    cfg._attn_implementation = "eager"
    cfg.get_text_config()._attn_implementation = "eager"
    cfg.vision_config._attn_implementation = "eager"

    if a.pure:
        RF.apply_pure_f64_patches(cfg)
        apply_vision_pure_patches()
        if a.fixed_reduce:
            RF.apply_fixed_reduce_patches()

    from PIL import Image

    images = []
    for spec in a.images:
        if spec.startswith("SYNTH:"):
            parts = spec.split(":")[1].split(",")
            h, wdt = int(parts[0]), int(parts[1])
            seed = int(parts[2]) if len(parts) > 2 else 0
            rng = np.random.default_rng(seed)
            images.append(Image.fromarray(rng.integers(0, 256, (h, wdt, 3), dtype=np.uint8)))
        else:
            images.append(Image.open(spec).convert("RGB"))

    ip = AutoImageProcessor.from_pretrained(snap, local_files_only=True)
    batch = ip(images=images, return_tensors="pt")
    pixel_values = batch["pixel_values"].to(torch.float64)
    grid_thw = batch["image_grid_thw"]
    print("pixel_values", tuple(pixel_values.shape), "grid", grid_thw.tolist())

    # the oracle consumes exactly these pixels
    write_f64(os.path.join(a.out, "pixel_values.bin"), pixel_values.numpy())

    if a.load == "full":
        model = RF.model_class(cfg).from_pretrained(
            snap, dtype=torch.float64, attn_implementation="eager", local_files_only=True)
        model.eval()
    else:
        model = build_vision_streamed(snap, cfg, with_text=bool(a.ids))

    with torch.inference_mode():
        tower = model.model.vision_tower(pixel_values=pixel_values, grid_thw=grid_thw)
        feats = model.model.vision_adapter(tower.last_hidden_state)
        feats = model.model.vision_projection(feats)
        feats = model.model.perception_emb_norm(feats)

    write_f64(os.path.join(a.out, "tower.bin"), tower.last_hidden_state.to(torch.float64).numpy())
    write_f64(os.path.join(a.out, "vision.bin"), feats.to(torch.float64).numpy())

    # optional end-to-end: the same features scattered into inputs_embeds at the
    # image placeholders, then the text stack
    ids = None
    if a.ids:
        from common import parse_ids

        ids = parse_ids(a.ids)
        n_ph = sum(1 for i in ids if i in (cfg.image_token_id, cfg.video_token_id))
        assert n_ph == feats.shape[0], (
            f"{n_ph} placeholder tokens in --ids but {feats.shape[0]} vision features")
        with torch.inference_mode():
            out = model.model(input_ids=torch.tensor([ids], dtype=torch.long),
                              pixel_values=pixel_values, image_grid_thw=grid_thw,
                              use_cache=False)
            logits = model.lm_head(out.last_hidden_state)
            logits = RF.output_tail(logits, cfg.get_text_config())
        lg = logits[0].to(torch.float64).numpy()
        write_f64(os.path.join(a.out, "logits.bin"), lg)
        top = np.argsort(lg[-1])[::-1][:5]
        print("last-position top5:", [(int(t), round(float(lg[-1, t]), 6)) for t in top])
    with open(os.path.join(a.out, "meta.json"), "w") as f:
        json.dump(dict(kind="hf_reference_vision", grid=grid_thw.tolist(),
                       ids=ids, T=(len(ids) if ids else 0),
                       V=int(cfg.get_text_config().vocab_size), n_hidden=0,
                       n_patches=int(pixel_values.shape[0]),
                       patch_dim=int(pixel_values.shape[1]),
                       n_merged=int(feats.shape[0]), hidden=int(feats.shape[1]),
                       pure=a.pure, fixed_reduce=a.fixed_reduce,
                       grid_arg=";".join(",".join(str(int(v)) for v in g)
                                         for g in grid_thw.tolist()),
                       torch=torch.__version__, transformers=transformers.__version__), f,
                  indent=1)
    print("tower", tuple(tower.last_hidden_state.shape), "-> features", tuple(feats.shape))
    print("feature[0][:6]", [round(float(v), 6) for v in feats[0, :6]])


if __name__ == "__main__":
    main()
