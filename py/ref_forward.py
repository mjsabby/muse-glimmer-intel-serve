#!/usr/bin/env python
"""HF-transformers float64 reference forward for Muse Glimmer (text path;
model_type muse_glimmer / muse_glimmer_text).

Modes:
  --load full      : from_pretrained in f64 (tiny models).
  --load streamed  : model skeleton on meta, HF's own modules executed with
                     weights materialized layer-by-layer in f64 (peak RAM ~
                     embed + lm_head + 1 layer) — for the 30B.

  --pure           : lift HF's hard-coded f32 casts to f64. Sites for this
                     architecture:
                       * MuseGlimmerRMSNorm (plain AND weight-less: the embed
                         norm, both qk-norms, the final norm, the perception
                         norm) — `.float()` / `.weight.float()` / `.type_as()`
                       * MuseGlimmerTextCenteredRMSNorm (the four sandwich
                         norms) — same three casts
                       * the rope inverse frequencies (`torch.arange(...,
                         dtype=torch.float)`) and the whole rope forward, which
                         runs under a forced-f32 autocast region
                       * the eager attention softmax (`dtype=torch.float32`)
                     Precision-only: formulas and op order stay HF's own code.

  --fixed-reduce   : additionally execute HF's modules with the ORACLE's
                     reduction orders and transcendentals (src/simd.hpp's
                     8-lane fma dot, the blocked-8 mean of squares, and
                     src/fmath.hpp's FDLIBM exp/tanh/sin/cos/pow), loaded from
                     libmuse_refkernels.so. This is what makes a BITWISE gate
                     possible: without it the reference uses BLAS' unspecified
                     dot order and libm's transcendentals, and agreement is
                     only at the f64 noise floor. Tiny models only (it is a
                     scalar C loop, not BLAS).

  --dtype bf16|f16 : with --pure, the rounding-instrumented f64 twin (round to
                     the storage dtype at exactly the points a stock
                     low-precision run materializes a tensor); without --pure,
                     stock torch execution in that dtype.

  --attn eager|flash : dtype != f64 + --pure only — materialize the attention
                     S/P tensors (eager) or not (flash, fused-kernel
                     semantics).

Dumps logits.bin (f64 [T,V]) + meta.json (+ hidden_XX.bin with --dump-hidden)
into --out DIR, in the same format as the C++ oracle so py/diff_logits.py and
py/diff_lp.py can compare the two directly."""

import argparse
import ctypes
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import parse_ids, resolve_model, write_f64, write_meta  # noqa: E402

import transformers  # noqa: E402
from transformers import AutoConfig  # noqa: E402
from transformers.models.muse_glimmer import modeling_muse_glimmer as MG  # noqa: E402
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS  # noqa: E402


def model_class(cfg):
    arch = getattr(cfg, "architectures", None)
    if arch:
        return getattr(transformers, arch[0])
    return transformers.MuseGlimmerForConditionalGeneration


# ------------------------------------------------------- oracle kernel bridge
#
# The oracle DEFINES the reduction order (ARCHITECTURE.md §"Numerics policy"),
# so a bitwise gate has to run the reference through the same kernels. They are
# compiled from the oracle's own headers — see tools/oracle_kernels.cpp — and
# separately gated by tests/fmath_test.cpp (transcendental bounds) and by the
# oracle's own --kernels scalar / thread-invariance checks (the order itself).

class OracleKernels:
    def __init__(self, path=None):
        if path is None:
            here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            path = os.path.join(here, "build", "libmuse_refkernels.so")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} not found — build it first:\n"
                f"    cmake -B build && cmake --build build --target muse_refkernels")
        self.lib = ctypes.CDLL(path)
        d = ctypes.POINTER(ctypes.c_double)
        i64 = ctypes.c_int64
        self.lib.ok_gemm.argtypes = [d, d, d, i64, i64, i64]
        self.lib.ok_meansq.argtypes = [d, d, i64, i64]
        for fn in ("ok_expv", "ok_tanhv", "ok_sinv", "ok_cosv", "ok_siluv", "ok_sigmoidv",
                   "ok_erfv", "ok_geluv"):
            getattr(self.lib, fn).argtypes = [d, d, i64]
        self.lib.ok_powv.argtypes = [ctypes.c_double, d, d, i64]
        self.lib.ok_layernorm.argtypes = [d, d, d, ctypes.c_double, d, i64, i64]

    @staticmethod
    def _c(a):
        a = np.ascontiguousarray(a, dtype=np.float64)
        return a, a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    def gemm(self, W, X):
        """Y[t,o] = dot8(W[o,:], X[t,:]) — the oracle's 8-lane fma order."""
        W, wp = self._c(W)
        X, xp = self._c(X)
        out, inn = W.shape
        T = X.shape[0]
        Y = np.empty((T, out), dtype=np.float64)
        _, yp = self._c(Y)
        self.lib.ok_gemm(wp, xp, yp, T, inn, out)
        return Y

    def meansq(self, X):
        """mean(x^2) per row in the oracle's blocked-8 order."""
        X, xp = self._c(X)
        rows, dim = X.shape
        out = np.empty(rows, dtype=np.float64)
        _, op = self._c(out)
        self.lib.ok_meansq(xp, op, rows, dim)
        return out

    def _map(self, fn, x):
        x, xp = self._c(x)
        out = np.empty_like(x)
        _, op = self._c(out)
        getattr(self.lib, fn)(xp, op, x.size)
        return out

    def exp(self, x):
        return self._map("ok_expv", x)

    def tanh(self, x):
        return self._map("ok_tanhv", x)

    def sin(self, x):
        return self._map("ok_sinv", x)

    def cos(self, x):
        return self._map("ok_cosv", x)

    def silu(self, x):
        return self._map("ok_siluv", x)

    def sigmoid(self, x):
        return self._map("ok_sigmoidv", x)

    def erf(self, x):
        return self._map("ok_erfv", x)

    def gelu(self, x):
        return self._map("ok_geluv", x)

    def layernorm(self, X, w, b, eps):
        """nn.LayerNorm in the oracle's blocked-8 mean/variance order."""
        X, xp = self._c(X)
        w, wp = self._c(w)
        b, bp = self._c(b)
        rows, dim = X.shape
        Y = np.empty_like(X)
        _, yp = self._c(Y)
        self.lib.ok_layernorm(xp, wp, bp, ctypes.c_double(eps), yp, rows, dim)
        return Y

    def pow(self, base, e):
        e, ep = self._c(e)
        out = np.empty_like(e)
        _, op = self._c(out)
        self.lib.ok_powv(ctypes.c_double(base), ep, op, e.size)
        return out


OK = None  # set by --fixed-reduce


def t2n(t):
    return t.detach().to(torch.float64).numpy()


def n2t(a, like):
    return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float64)).to(like.dtype)


# ---------------------------------------------------------------- pure f64

def pure_inv_freq(cfg, f64=True):
    """MuseGlimmerTextRotaryEmbedding.compute_default_rope_parameters with the
    f32 arange lifted to f64:  inv_freq[j] = theta^(-2j/head_dim)."""
    theta = float(cfg.rope_parameters["rope_theta"])
    dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    e = torch.arange(0, dim, 2, dtype=torch.float64) / dim
    if OK is not None:
        return torch.from_numpy(1.0 / OK.pow(theta, t2n(e)))
    return 1.0 / (theta ** e)


def apply_pure_f64_patches(cfg):
    tc = cfg.get_text_config()

    # (1) MuseGlimmerRMSNorm — the plain (final `norm`) and weight-less
    #     (embed_norm, qk_norm, perception_emb_norm) flavour. Note the
    #     reference uses torch.pow(ms, -0.5) rather than rsqrt to match JAX;
    #     the two are bit-identical in f64 (py/probe_torch_ops.py) and both
    #     equal the oracle's `1.0 / std::sqrt(ms)`.
    def rms_forward(self, x):
        if OK is not None:
            ms = n2t(OK.meansq(t2n(x).reshape(-1, x.shape[-1])), x).reshape(*x.shape[:-1], 1)
        else:
            ms = x.pow(2).mean(-1, keepdim=True)
        out = x * torch.pow(ms + self.eps, -0.5)
        if self.with_scale:
            out = out * self.weight
        return out

    # (2) MuseGlimmerTextCenteredRMSNorm — the four sandwich norms, (1 + w)
    def centered_forward(self, x):
        if OK is not None:
            ms = n2t(OK.meansq(t2n(x).reshape(-1, x.shape[-1])), x).reshape(*x.shape[:-1], 1)
        else:
            ms = x.pow(2).mean(-1, keepdim=True)
        out = x * torch.rsqrt(ms + self.eps)
        return out * (1.0 + self.weight)

    # (3) rope tables in f64 (stock builds inv_freq from an f32 arange and runs
    #     the whole forward under a forced-f32 autocast region)
    f64_inv = pure_inv_freq(tc)

    def rope_forward(self, x, position_ids):
        inv = f64_inv.to(x.device)
        inv_expanded = inv[None, :, None].expand(position_ids.shape[0], -1, 1)
        pos = position_ids[:, None, :].to(torch.float64)
        freqs = (inv_expanded @ pos).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        if OK is not None:
            cos = n2t(OK.cos(t2n(emb)), emb).reshape(emb.shape) * self.attention_scaling
            sin = n2t(OK.sin(t2n(emb)), emb).reshape(emb.shape) * self.attention_scaling
        else:
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

    # (4) eager attention with a dtype-preserving softmax (stock forces f32)
    def eager_pure(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
        key_states = MG.repeat_kv(key, module.num_key_value_groups)
        value_states = MG.repeat_kv(value, module.num_key_value_groups)
        if OK is not None:
            return oracle_order_attention(query, key_states, value_states,
                                          attention_mask, scaling)
        attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        return attn_output, attn_weights

    MG.MuseGlimmerRMSNorm.forward = rms_forward
    MG.MuseGlimmerTextCenteredRMSNorm.forward = centered_forward
    MG.MuseGlimmerTextRotaryEmbedding.forward = rope_forward
    MG.eager_attention_forward = eager_pure
    try:
        ALL_ATTENTION_FUNCTIONS.register("eager", eager_pure)
    except Exception:
        ALL_ATTENTION_FUNCTIONS["eager"] = eager_pure
    resolved = ALL_ATTENTION_FUNCTIONS.get_interface("eager", MG.eager_attention_forward)
    assert resolved is eager_pure, "failed to route eager attention to the pure-f64 version"


def oracle_order_attention(query, key_states, value_states, attention_mask, scaling):
    """Eager attention executed in the oracle's exact order: an 8-lane fma dot
    per score, a running max, a sequential softmax sum with FDLIBM exp, and a
    sequential probability-weighted accumulation of V (see
    src/muse_glimmer.hpp::attention_forward). Batch 1."""
    assert query.shape[0] == 1, "oracle-order attention is batch-1 (the oracle's contract)"
    q = t2n(query)[0]   # [nq, Tq, D]
    k = t2n(key_states)[0]
    v = t2n(value_states)[0]
    nq, Tq, D = q.shape
    Tk = k.shape[1]
    mask = None if attention_mask is None else t2n(attention_mask)
    out = np.zeros((nq, Tq, D), dtype=np.float64)
    probs = np.zeros((nq, Tq, Tk), dtype=np.float64)
    for h in range(nq):
        for tq in range(Tq):
            sc = OK.gemm(k[h], q[h, tq:tq + 1])[0] * scaling
            if mask is not None:
                m = mask[0, h % mask.shape[1], tq] if mask.shape[1] > 1 else mask[0, 0, tq]
                allowed = np.nonzero(m > -1e30)[0]
            else:
                allowed = np.arange(Tk)
            s = sc[allowed]
            e = OK.exp(s - s.max())
            tot = 0.0
            for val in e:  # sequential sum, ascending — the oracle's order
                tot += val
            p = e / tot
            probs[h, tq, allowed] = p
            acc = np.zeros(D, dtype=np.float64)
            for j, tk in enumerate(allowed):  # sequential over keys
                acc += p[j] * v[h, tk]
            out[h, tq] = acc
    o = torch.from_numpy(out)[None].transpose(1, 2).contiguous()
    return o.to(query.dtype), torch.from_numpy(probs)[None].to(query.dtype)


def apply_fixed_reduce_patches():
    """Route every remaining reduction/transcendental through the oracle's
    kernels. Must be applied AFTER apply_pure_f64_patches (which already
    consults OK for the norms, rope tables and attention)."""

    def linear_forward(self, x):
        shp = x.shape
        y = OK.gemm(t2n(self.weight), t2n(x).reshape(-1, shp[-1]))
        y = torch.from_numpy(y).reshape(*shp[:-1], -1).to(x.dtype)
        if self.bias is not None:
            y = y + self.bias
        return y

    torch.nn.Linear.forward = linear_forward

    def silu(x):
        return n2t(OK.silu(t2n(x)), x).reshape(x.shape)

    def sigmoid(x):
        return n2t(OK.sigmoid(t2n(x)), x).reshape(x.shape)

    # ACT2FN["silu"] is transformers.activations.SiLUActivation, NOT nn.SiLU —
    # patch whatever class the registry actually hands the MLP, and assert it.
    from transformers.activations import ACT2FN

    silu_cls = type(ACT2FN["silu"])
    silu_cls.forward = lambda self, x: silu(x)
    probe = torch.tensor([0.7], dtype=torch.float64)
    assert bool(ACT2FN["silu"](probe) == n2t(OK.silu(t2n(probe)), probe)), \
        "failed to route the SwiGLU activation through the oracle's silu"

    # the vision tower and the projector use the EXACT erf gelu
    def gelu(x):
        return n2t(OK.gelu(t2n(x)), x).reshape(x.shape)

    gelu_cls = type(ACT2FN["gelu"])
    gelu_cls.forward = lambda self, x: gelu(x)
    assert bool(ACT2FN["gelu"](probe) == n2t(OK.gelu(t2n(probe)), probe)), \
        "failed to route the gelu activation through the oracle's gelu"

    # nn.LayerNorm is the vision tower's normalizer; torch's f64 reduction
    # order for it matches no simple blocked form, same as for the text norms
    def layernorm_forward(self, x):
        shp = x.shape
        y = OK.layernorm(t2n(x).reshape(-1, shp[-1]), t2n(self.weight), t2n(self.bias),
                         float(self.eps))
        return torch.from_numpy(y).reshape(shp).to(x.dtype)

    torch.nn.LayerNorm.forward = layernorm_forward

    MG.torch.sigmoid = sigmoid  # the attention output gate


# ------------------------------------------- low-precision (bf16/f16) twin
#
# The model stays f64 (weights hold exact bf16 values); every forward below
# inserts a round-to-storage-dtype at the points a stock bf16/f16 run
# materializes a tensor, with exact f64 arithmetic in between — mirroring
# src/muse_glimmer.hpp's --dtype mode op for op, which is the bitwise gate
# for it.

def make_rounder(rdt):
    def rnd(x):
        return x.to(rdt).to(torch.float64)
    return rnd


def rnd32(x):
    return x.to(torch.float32).to(torch.float64)


def f32val(c):
    """A python float holding f32(c) — scalar constants as torch's opmath sees
    them in a low-precision run."""
    return float(torch.tensor(c, dtype=torch.float32).item())


def apply_lp_patches(cfg, rdt, attn_flash):
    tc = cfg.get_text_config()
    rnd = make_rounder(rdt)

    # every nn.Linear output materializes at the storage dtype (the f64
    # accumulation stands in for an ideal accumulator)
    def linear_forward(self, x):
        return rnd(F.linear(x, self.weight, self.bias))

    torch.nn.Linear.forward = linear_forward

    # both RMSNorm kinds: stock's f32 internals canonicalized upward to f64,
    # one rounding at `.type_as(x)`
    def rms_lp(self, x):
        out = x * torch.pow(x.pow(2).mean(-1, keepdim=True) + self.eps, -0.5)
        if self.with_scale:
            out = out * self.weight
        return rnd(out)

    def centered_lp(self, x):
        out = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return rnd(out * (1.0 + self.weight))

    # rope: f32 inv_freq buffer chain, f32 freqs, cos/sin round to storage at
    # `.to(x.dtype)`
    def lp_inv_freq():
        theta = float(tc.rope_parameters["rope_theta"])
        dim = getattr(tc, "head_dim", None) or tc.hidden_size // tc.num_attention_heads
        e = (torch.arange(0, dim, 2, dtype=torch.int64).to(torch.float32) / float(dim))
        pw = torch.pow(torch.tensor(f32val(theta), dtype=torch.float64),
                       e.to(torch.float64)).to(torch.float32)
        return torch.tensor(1.0, dtype=torch.float32) / pw

    lp_inv = lp_inv_freq()

    def rope_lp(self, x, position_ids):
        inv = lp_inv.to(torch.float64)
        inv_expanded = inv[None, :, None].expand(position_ids.shape[0], -1, 1)
        pos = position_ids[:, None, :].to(torch.float64)
        freqs = rnd32((inv_expanded @ pos).transpose(1, 2))
        emb = torch.cat((freqs, freqs), dim=-1)
        return rnd(emb.cos()), rnd(emb.sin())

    # (x*cos), (rotate_half(x)*sin) and their sum are three tensor ops
    def apply_rope_lp(q, k, cos, sin, unsqueeze_dim=1):
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)

        def rot(x):
            return rnd(rnd(x * cos) + rnd(MG.rotate_half(x) * sin))

        return rot(q), rot(k)

    # eager attention: matmul out and *scaling materialize (unless flash), the
    # f32-forced softmax canonicalized upward with P rounded at
    # `.to(query.dtype)`, one rounding for P@V
    def eager_lp(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
        key_states = MG.repeat_kv(key, module.num_key_value_groups)
        value_states = MG.repeat_kv(value, module.num_key_value_groups)
        aw = torch.matmul(query, key_states.transpose(2, 3))
        if not attn_flash:
            aw = rnd(aw)
        aw = aw * f32val(scaling)
        if not attn_flash:
            aw = rnd(aw)
        if attention_mask is not None:
            aw = aw + attention_mask
        aw = torch.nn.functional.softmax(aw, dim=-1)
        if not attn_flash:
            aw = rnd(aw)
        out = rnd(torch.matmul(aw, value_states))
        return out.transpose(1, 2).contiguous(), aw

    # attention forward: the qk_scale_factor multiply and both output-gate ops
    # materialize
    def attn_forward_lp(self, hidden_states, position_embeddings=None, attention_mask=None,
                        past_key_values=None, **kwargs):
        assert past_key_values is None, "the reference runs prefill only"
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        query_states = rnd(self.qk_norm(query_states) * f32val(self.qk_scale_factor))
        key_states = self.qk_norm(key_states)
        if position_embeddings is not None:  # NoPE layers get None
            cos, sin = position_embeddings
            query_states, key_states = apply_rope_lp(query_states, key_states, cos, sin)
        attn_output, _ = eager_lp(self, query_states, key_states, value_states,
                                  attention_mask, scaling=self.scaling)
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = rnd(attn_output * rnd(torch.sigmoid(self.gate_proj(hidden_states))))
        return self.o_proj(attn_output), None

    # decoder layer: the two residual adds materialize
    def layer_forward_lp(self, hidden_states, position_embeddings=None, attention_mask=None,
                         position_ids=None, past_key_values=None, **kwargs):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states, position_embeddings=position_embeddings,
            attention_mask=attention_mask, past_key_values=past_key_values, **kwargs)
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = rnd(residual + hidden_states)

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        return rnd(residual + hidden_states)

    # SwiGLU: silu(gate) and the product each materialize
    def mlp_lp(self, x):
        return self.down_proj(rnd(rnd(F.silu(self.gate_proj(x))) * self.up_proj(x)))

    MG.MuseGlimmerRMSNorm.forward = rms_lp
    MG.MuseGlimmerTextCenteredRMSNorm.forward = centered_lp
    MG.MuseGlimmerTextRotaryEmbedding.forward = rope_lp
    MG.MuseGlimmerTextAttention.forward = attn_forward_lp
    MG.MuseGlimmerTextDecoderLayer.forward = layer_forward_lp
    MG.MuseGlimmerTextMLP.forward = mlp_lp


def apply_lp_weight_fixups(model, rdt):
    """A stock from_pretrained(dtype=rdt) load rounds every parameter to the
    storage dtype. Weights that are already bf16 in the checkpoint are exact in
    bf16 mode; f16 mode (and any f32 tensor) needs the cast applied."""
    with torch.no_grad():
        for _, prm in model.named_parameters():
            if prm.is_meta:
                continue
            prm.copy_(prm.to(rdt).to(torch.float64))


# ---------------------------------------------------------- streamed loading

class StreamedWeights:
    def __init__(self, snap):
        from safetensors import safe_open

        index_path = os.path.join(snap, "model.safetensors.index.json")
        self.handles = {}
        self.where = {}
        if os.path.exists(index_path):
            wm = json.load(open(index_path))["weight_map"]
            self.where.update(wm)
            files = sorted(set(wm.values()))
        else:
            files = ["model.safetensors"]
        for fname in files:
            self.handles[fname] = safe_open(os.path.join(snap, fname), framework="pt",
                                            device="cpu")
        if not os.path.exists(index_path):
            for fname, h in self.handles.items():
                for name in h.keys():
                    self.where[name] = fname

    def has(self, name):
        return name in self.where

    def get(self, name):
        return self.handles[self.where[name]].get_tensor(name)


def materialize(module, prefix, weights, dtype=torch.float64):
    module.to_empty(device="cpu")
    sd = {}
    for name, _ in module.named_parameters():
        full = prefix + name
        assert weights.has(full), f"checkpoint missing parameter {full}"
        sd[name] = weights.get(full).to(dtype)
    for name, _ in module.named_buffers():
        full = prefix + name
        if weights.has(full):
            sd[name] = weights.get(full).to(dtype)
    _, unexpected = module.load_state_dict(sd, strict=False)
    assert not unexpected, f"unexpected keys {unexpected}"


def build_streamed(snap, cfg):
    """The 30B in f64 does not fit in host RAM if every module is materialized:
    one decoder layer is 484M parameters (3.9 GiB in f64) and the embedding and
    lm_head are 1.345G each (10.8 GiB in f64, twice over). Decoder layers are
    materialized and released one at a time; the two vocab-sized tables stay
    BF16 and are converted at their use site, which is exact (bf16 subset f64)
    and costs 2.7 GiB each instead of 10.8."""
    weights = StreamedWeights(snap)

    torch.set_default_dtype(torch.float64)
    with torch.device("meta"):
        model = model_class(cfg)(cfg)
    model = model.to(torch.float64)
    model.eval()

    lm = model.model.language_model
    # non-persistent buffers were created on meta: rebuild on cpu
    lm.rotary_emb = lm.rotary_emb.__class__(cfg.get_text_config())

    def materialize_raw(module, name):
        module.to_empty(device="cpu")
        module.weight = torch.nn.Parameter(weights.get(name), requires_grad=False)

    # embedding: exact gather in bf16, then the f64 cast BEFORE embed_norm, so
    # MuseGlimmerTextNormedEmbedding's weight-less norm still runs in f64
    materialize_raw(lm.embed_tokens, "model.language_model.embed_tokens.weight")
    emb_cls = type(lm.embed_tokens)
    emb_cls.forward = lambda self, ids: self.embed_norm(
        F.embedding(ids, self.weight).to(torch.float64))

    materialize(lm.norm, "model.language_model.norm.", weights)

    if getattr(cfg, "tie_word_embeddings", False):
        model.lm_head.weight = lm.embed_tokens.weight
    else:
        materialize_raw(model.lm_head, "lm_head.weight")
    head = model.lm_head
    chunk = 8192  # output rows converted at a time: 8192 x 6656 f64 = 436 MiB

    def head_forward(self, x):
        out = torch.empty(*x.shape[:-1], self.weight.shape[0], dtype=torch.float64)
        for o0 in range(0, self.weight.shape[0], chunk):
            w = self.weight[o0:o0 + chunk].to(torch.float64)
            # chunking is over OUTPUT rows, so every output element still
            # reduces over the full hidden dim and the values are unchanged
            if OK is not None:  # --fixed-reduce must not be bypassed here
                out[..., o0:o0 + chunk] = torch.from_numpy(
                    OK.gemm(t2n(w), t2n(x).reshape(-1, x.shape[-1]))).reshape(
                        *x.shape[:-1], -1)
            else:
                out[..., o0:o0 + chunk] = F.linear(x, w)
        return out

    head.forward = head_forward.__get__(head)
    # model.model.vision_tower stays on meta — never executed on the text path

    for i, layer in enumerate(lm.layers):
        prefix = f"model.language_model.layers.{i}."

        def pre_hook(module, args, kwargs, prefix=prefix):
            materialize(module, prefix, weights)
            return None

        def post_hook(module, args, kwargs, output):
            module.to("meta")
            return None

        layer.register_forward_pre_hook(pre_hook, with_kwargs=True)
        layer.register_forward_hook(post_hook, with_kwargs=True)

    return model


# ----------------------------------------------------------------- the tail
#
# modeling_muse_glimmer.py MuseGlimmerForConditionalGeneration.forward, verbatim:
#     logits = lm_head(h)
#     logits = logits * output_multiplier
#     logits = logits / final_logit_softcapping
#     logits = tanh(logits)
#     logits = logits * final_logit_softcapping
# Four separate tensor ops, so four materializations in a low-precision run.

def install_trace(model, trace_dir):
    """Dump the same per-op intermediates the oracle writes with --trace-dir, so
    a disagreement can be localized to one op instead of a whole layer. The
    names match src/muse_glimmer.hpp's trace points."""
    os.makedirs(trace_dir, exist_ok=True)

    def dump(layer, name, t):
        write_f64(os.path.join(trace_dir, f"L{layer:02d}.{name}.bin"),
                  t.detach().to(torch.float64).numpy())

    for i, layer in enumerate(model.model.language_model.layers):
        def out_hook(name, i=i):
            return lambda m, a, o: dump(i, name, o if not isinstance(o, tuple) else o[0])

        def in_hook(name, i=i):
            return lambda m, a: dump(i, name, a[0])

        layer.input_layernorm.register_forward_hook(out_hook("input_ln"))
        layer.self_attn.o_proj.register_forward_pre_hook(in_hook("attn_gated"))
        layer.self_attn.o_proj.register_forward_hook(out_hook("attn_out"))
        layer.pre_feedforward_layernorm.register_forward_pre_hook(in_hook("post_attn_resid"))
        layer.pre_feedforward_layernorm.register_forward_hook(out_hook("pre_ff_ln"))
        layer.mlp.down_proj.register_forward_pre_hook(in_hook("swiglu"))
        layer.mlp.down_proj.register_forward_hook(out_hook("mlp_out"))


def output_tail(logits, tc, rdt=None):
    rnd = make_rounder(rdt) if rdt is not None else (lambda x: x)
    mult = f32val(tc.output_multiplier) if rdt is not None else tc.output_multiplier
    cap = f32val(tc.final_logit_softcapping) if rdt is not None else tc.final_logit_softcapping
    logits = rnd(logits * mult)
    logits = rnd(logits / cap)
    if OK is not None:
        logits = torch.from_numpy(OK.tanh(t2n(logits)).reshape(logits.shape)).to(logits.dtype)
    else:
        logits = torch.tanh(logits)
    logits = rnd(logits)
    return rnd(logits * cap)


# ---------------------------------------------------------------------- main

def main():
    global OK
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="snapshot dir or repo id (local HF cache)")
    p.add_argument("--ids", required=True, help="comma/space separated token ids, or a file")
    p.add_argument("--out", required=True)
    p.add_argument("--load", choices=["full", "streamed"], default="full")
    p.add_argument("--pure", action="store_true", help="lift HF's f32 casts to f64")
    p.add_argument("--fixed-reduce", action="store_true",
                   help="also use the oracle's reduction orders + FDLIBM math "
                        "(tiny models only; makes the gate bitwise)")
    p.add_argument("--kernels", default=None, help="path to libmuse_refkernels.so")
    p.add_argument("--dtype", choices=["f64", "bf16", "f16"], default="f64",
                   help="storage dtype; with --pure = rounding-instrumented f64 "
                        "reference, without = stock torch execution in that dtype")
    p.add_argument("--attn", choices=["eager", "flash"], default="eager",
                   help="dtype!=f64 + --pure: materialize S/P (eager) or not (flash)")
    p.add_argument("--dump-hidden", action="store_true")
    p.add_argument("--trace-dir", default=None,
                   help="dump per-op intermediates (same names as the oracle's --trace-dir)")
    p.add_argument("--threads", type=int, default=0)
    a = p.parse_args()

    if a.threads:
        torch.set_num_threads(a.threads)

    snap = resolve_model(a.model)
    ids = parse_ids(a.ids)
    os.makedirs(a.out, exist_ok=True)

    cfg = AutoConfig.from_pretrained(snap, local_files_only=True)
    tc = cfg.get_text_config()
    cfg._attn_implementation = "eager"
    tc._attn_implementation = "eager"

    lp = a.dtype != "f64"
    rdt = {"bf16": torch.bfloat16, "f16": torch.float16}.get(a.dtype)
    if a.fixed_reduce:
        assert a.pure, "--fixed-reduce only makes sense with --pure"
        assert not lp, "--fixed-reduce is an f64 mode (the twin rounds away the difference)"
        OK = OracleKernels(a.kernels)

    if a.pure and lp:
        apply_lp_patches(cfg, rdt, a.attn == "flash")
    elif a.pure:
        apply_pure_f64_patches(cfg)
        if a.fixed_reduce:
            apply_fixed_reduce_patches()

    if lp and not a.pure:
        assert a.load == "full", "stock low-precision runs load the checkpoint directly"
        model = model_class(cfg).from_pretrained(
            snap, dtype=rdt, attn_implementation="eager", local_files_only=True)
        model.eval()
    elif a.load == "full":
        model = model_class(cfg).from_pretrained(
            snap, dtype=torch.float64, attn_implementation="eager", local_files_only=True)
        model.eval()
    else:
        model = build_streamed(snap, cfg)

    if lp and a.pure:
        apply_lp_weight_fixups(model, rdt)

    lm = model.model.language_model
    assert lm.layers[0].self_attn.config._attn_implementation == "eager"
    want = rdt if (lp and not a.pure) else torch.float64
    for name, prm in lm.named_parameters():
        if a.load == "streamed" and name.startswith("embed_tokens."):
            # deliberately BF16: converted exactly at the use site (see
            # build_streamed) so the f64 run fits in host RAM
            assert prm.dtype == torch.bfloat16, prm.dtype
            continue
        assert prm.dtype == want or prm.is_meta, prm.dtype

    if a.trace_dir:
        install_trace(model, a.trace_dir)

    input_ids = torch.tensor([ids], dtype=torch.long)
    with torch.inference_mode():
        out = model.model(input_ids=input_ids, use_cache=False,
                          output_hidden_states=a.dump_hidden)
        h = out.last_hidden_state
        logits = model.lm_head(h)
        logits = output_tail(logits, tc, rdt if (lp and a.pure) else None)

    logits = logits[0].to(torch.float64).numpy()
    shape = write_f64(os.path.join(a.out, "logits.bin"), logits)
    print(f"logits shape {shape}", flush=True)

    n_hidden = 0
    if a.dump_hidden:
        hs = out.hidden_states
        n_hidden = len(hs)
        print(f"captured {n_hidden} hidden-state tensors, each {list(hs[0].shape)}",
              flush=True)
        for i, hh in enumerate(hs):
            write_f64(os.path.join(a.out, f"hidden_{i:02d}.bin"),
                      hh[0].to(torch.float64).numpy())

    top = np.argsort(logits[-1])[::-1][:5]
    print("last-position top5:", [(int(t), round(float(logits[-1, t]), 6)) for t in top])

    write_meta(a.out, dict(
        kind="hf_reference", model=os.path.abspath(snap), ids=ids, T=len(ids),
        V=int(logits.shape[-1]), load=a.load, pure=a.pure,
        fixed_reduce=a.fixed_reduce, dtype=a.dtype, attn=a.attn,
        n_hidden=n_hidden, torch=torch.__version__,
        transformers=transformers.__version__,
    ))


if __name__ == "__main__":
    main()
