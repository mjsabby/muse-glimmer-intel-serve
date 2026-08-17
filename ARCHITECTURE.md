# Muse Glimmer 30B — exact model specification (oracle contract)

Source of truth: `transformers` `models/muse_glimmer/modeling_muse_glimmer.py`
(`MuseGlimmerForConditionalGeneration`, `transformers_version` in the checkpoint
is `5.15.0.dev0`) and `models/muse_glimmer_assistant/modeling_muse_glimmer_assistant.py`
(`MuseGlimmerAssistantModel`, the DFlash block drafter). Checkpoints:

| repo | what |
|---|---|
| `meta-models/Muse-Glimmer-30B` | BF16 safetensors, 2 shards, 1436 tensors, 29.78 B params |
| `meta-models/Muse-Glimmer-30B-assistant` | DFlash drafter, 1 file, 58 tensors |
| `meta-models/Muse-Glimmer-30B-GGUF` | official `muse-glimmer` / `clip` mmproj / `dflash` GGUFs (Q4_K tiers) |
| `bartowski/Muse-Glimmer-30B-GGUF`, `unsloth/…` | community GGUF spread (IQ2…Q8_0) |

Everything below was read out of the reference implementation and verified
against the shipped tensors (safetensors headers + byte-level value probes of
both safetensors and GGUF). Where the two disagree, the disagreement is
documented — those are the traps.

The oracle implements the **full** batch-1 path: `input_ids (+ pixel_values) →
logits`, no padding, `attn_implementation="eager"` semantics, `model.eval()`.

## Config

From the checkpoint `config.json`.

### Top level (`MuseGlimmerConfig`)

| field | value |
|---|---|
| `model_type` | `muse_glimmer` |
| `image_token_id` / `video_token_id` | 200092 / 200091 |
| `out_hidden_size` | 6144 (= vision hidden 1536 × merge_size²) |
| `projector_hidden_size` / `projector_hidden_act` | 4096 / `gelu` |
| `dtype` | `bfloat16` |

### Text (`text_config`, `muse_glimmer_text`)

| field | value |
|---|---|
| hidden_size H | 6656 |
| num_hidden_layers L | 52 |
| intermediate_size I | 19968 |
| hidden_activation | `silu` (SwiGLU) |
| num_attention_heads nq / num_key_value_heads nkv | 32 / 2 (GQA ratio 16) |
| head_dim D | 128 (nq·D = 4096 ≠ H — the attention block is *narrower* than the residual) |
| vocab_size V | 202048 |
| layer_types | `[sliding, sliding, sliding, full] × 13`, full at `(i+1)%4 == 0` |
| sliding_window | 2048 |
| layer_rope_theta | `500000.0` on sliding layers, **`0` on every full layer** |
| rope_parameters | `{rope_theta: 500000.0, rope_type: default}` |
| rms_norm_eps | 1e-5 |
| **post_norm_eps** | **1e-8** |
| **qk_scale_factor** | **3.87** |
| **output_multiplier** | **0.19611613513818404** (= 26^-1/2 = (H/256)^-1/2) |
| **final_logit_softcapping** | **20.0** |
| attention_bias | false (no bias anywhere in the text stack) |
| tie_word_embeddings | false |
| max_position_embeddings | 131072 |
| bos / eos / pad | 200000 / 200001 (+200008 `<\|eot\|>`) / 200018 |

### Vision (`vision_config`, `muse_glimmer_vision`) — PE-style ViT-G/14, ~1.8 B

| field | value |
|---|---|
| hidden_size | 1536 |
| num_hidden_layers | 50 |
| intermediate_size | 8960 |
| hidden_act | `gelu` |
| num_attention_heads | 16 (head_dim 96) |
| layer_types | `[window, window, window, full] × 12` + `[window, full]` — the tower **ends on a full-attention layer**; `MuseGlimmerVisionConfig.__post_init__` is `full if (i+1) % 4 == 0 or i == L-1` |
| patch_size / patch_temporal / merge_size | 14 / 2 / 2 |
| pos_emb_height × pos_emb_width | 32 × 32 (1024-entry learned table) |
| layer_norm_eps | 1e-5 (LayerNorm **with bias**, not RMSNorm) |
| rope_parameters | `{rope_theta: 10000.0, rope_type: default}` |
| max_position_embeddings | 1024 |

### Assistant / DFlash (`MuseGlimmerAssistantConfig`)

| field | value |
|---|---|
| hidden_size / intermediate_size | 6656 / 19968 (same as target) |
| num_hidden_layers | 5, all `sliding_attention` (window 2048) |
| heads q / kv, head_dim | 32 / 8, 128 |
| **block_size** | **16** |
| **mask_token_id** | **201818** |
| **target_layer_ids** | **[1, 13, 25, 37, 49]** (0-based target layer outputs) |
| rms_norm_eps / rope_theta | 1e-5 / 500000.0 |
| has q_norm / k_norm | **yes** (unlike the target) |

## Tensor inventory

Text backbone, prefix `model.language_model.`:

```
embed_tokens.weight                          [202048, 6656]
layers.{i}.input_layernorm.weight            [6656]      centered (1+w)
layers.{i}.post_attention_layernorm.weight   [6656]      centered (1+w)
layers.{i}.pre_feedforward_layernorm.weight  [6656]      centered (1+w)
layers.{i}.post_feedforward_layernorm.weight [6656]      centered (1+w)
layers.{i}.self_attn.q_proj.weight           [4096, 6656]
layers.{i}.self_attn.k_proj.weight           [ 256, 6656]
layers.{i}.self_attn.v_proj.weight           [ 256, 6656]
layers.{i}.self_attn.gate_proj.weight        [4096, 6656]   <- attention output gate
layers.{i}.self_attn.o_proj.weight           [6656, 4096]
layers.{i}.mlp.{gate,up}_proj.weight         [19968, 6656]
layers.{i}.mlp.down_proj.weight              [6656, 19968]
norm.weight                                  [6656]      PLAIN (w), not centered
lm_head.weight                               [202048, 6656]   (untied, top-level key)
```

There are **no `q_norm` / `k_norm` tensors** on the target: QK-norm is
weight-less (see §4). All tensors are BF16 — exactly convertible to f64.

Vision, prefix `model.vision_tower.` plus the projector:

```
patch_embedder.patch_embedding.weight         [1536, 1176]   1176 = 2·3·14²
patch_embedder.position_embedding_table.weight[1024, 1536]
ln_pre.{weight,bias}                          [1536]
layers.{i}.norm1.{weight,bias}                [1536]
layers.{i}.attn.{q,k,v}_proj.{weight,bias}    [1536,1536] / [1536]
layers.{i}.attn.proj.{weight,bias}            [1536,1536] / [1536]
layers.{i}.norm2.{weight,bias}                [1536]
layers.{i}.mlp.fc1.{weight,bias}              [8960,1536] / [8960]
layers.{i}.mlp.fc2.{weight,bias}              [1536,8960] / [1536]
ln_post.{weight,bias}                         [1536]
model.vision_adapter.fc1.weight               [4096, 6144]
model.vision_adapter.fc2.weight               [4096, 4096]
model.vision_projection.weight                [6656, 4096]
```

DFlash drafter (no prefix, own repo):

```
encoder.fc.weight            [6656, 33280]    33280 = 5 · 6656
encoder.output_norm_enc.weight [6656]         plain
layers.{0..4}.input_layernorm.weight          [6656]  plain
layers.{0..4}.post_attention_layernorm.weight [6656]  plain
layers.{0..4}.self_attn.q_proj.weight         [4096, 6656]
layers.{0..4}.self_attn.k_proj.weight         [1024, 6656]
layers.{0..4}.self_attn.v_proj.weight         [1024, 6656]
layers.{0..4}.self_attn.o_proj.weight         [6656, 4096]
layers.{0..4}.self_attn.{q,k}_norm.weight     [128]   plain
layers.{0..4}.mlp.{gate,up,down}_proj.weight  as target
norm.weight                  [6656]           plain
```

No `embed_tokens`, no `lm_head` — the drafter borrows both from the target.

## Text ops, in execution order

### 1. Embedding — `MuseGlimmerTextNormedEmbedding`

```
e[t,:] = embed_tokens[ids[t],:]
h[t,:] = e[t,:] · (mean(e[t,:]²) + rms_norm_eps)^(-1/2)      # weight-LESS RMSNorm
```

There is no learned scale and **no `sqrt(H)` Gemma-style multiplier**. Note the
reference comment: the raw (un-normed) embedding is what DFlash consumes, so
the loader must keep both forms reachable.

### 2. RMSNorm — two distinct kinds, do not mix them up

`MuseGlimmerRMSNorm` (plain, `with_scale=True`):
```
y = x · (mean(x²) + eps)^(-1/2) · w
```
`MuseGlimmerTextCenteredRMSNorm` (zero-centered):
```
y = x · rsqrt(mean(x²) + eps) · (1 + w)
```

- The four **per-layer** norms are **centered** → `(1 + w)`.
- The **final** `model.language_model.norm` is **plain** → `w`.
- Verified by byte-probe: safetensors `layers.0.input_layernorm[0..3] =
  [0.096191, 0.112793, 0.357422, 0.449219]` while the GGUF's
  `blk.0.attn_norm[0..3] = [1.096191, 1.112793, 1.357422, 1.449219]` — the
  converter adds exactly +1. For `norm.weight` / `output_norm.weight` the two
  files are **identical** (`[-3.34375, 3.625, -2.796875, …]`), so no +1 there.
  Any GGUF path must reproduce that asymmetry.
- The plain kind computes `pow(mean+eps, -0.5)`, the centered kind
  `rsqrt(mean+eps)` — same value, different torch op (the reference comments
  that `pow` was chosen to match JAX). Immaterial in f64; noted so the bf16
  twin does not gratuitously differ.
- Stock HF upcasts both to f32 internally (`x.float()`, `weight.float()`,
  `.type_as(x)`). The oracle lifts to f64.

### 3. Decoder layer — Gemma-style sandwich, four norms

```
r = h
x = input_layernorm(h)                       # centered, eps = rms_norm_eps  = 1e-5
a = self_attn(x)
a = post_attention_layernorm(a)              # centered, eps = post_norm_eps = 1e-8
h = r + a

r = h
x = pre_feedforward_layernorm(h)             # centered, eps = rms_norm_eps  = 1e-5
m = down_proj( silu(gate_proj(x)) * up_proj(x) )
m = post_feedforward_layernorm(m)            # centered, eps = post_norm_eps = 1e-8
h = r + m
```

The two eps values differ by three orders of magnitude and are assigned by
position, not by name. `post_norm_eps` is **not** in the GGUF metadata.

### 4. Attention — `MuseGlimmerTextAttention`

```
q = q_proj(x) → [T, 32, 128]
k = k_proj(x) → [T,  2, 128]
v = v_proj(x) → [T,  2, 128]

q = rmsnorm_weightless(q, eps=rms_norm_eps) · qk_scale_factor   # 3.87
k = rmsnorm_weightless(k, eps=rms_norm_eps)                     # no scale

if layer_rope_theta[i] != 0:            # sliding layers only
    q, k = rope(q, k, cos, sin)         # full 128 dims, rotate_half, θ=500000

scores[hq,tq,tk] = (q·k) · D^-0.5       # scaling is STILL head_dim^-0.5 = 128^-0.5
                                        # kv head = hq // 16
softmax over allowed tk (f32-forced in stock)
out = Σ p·v → [T, 4096]

out = out ⊙ sigmoid( gate_proj(x) )     # gate from the PRE-attention normed x
attn = o_proj(out)                      # [T, 4096] → [T, 6656]
```

Three things that catch implementations out:

1. **QK-norm is weight-less and per-head over D.** The `qk_scale_factor = 3.87`
   multiplies **q only**, and is *in addition to* the usual `D^-0.5`. The GGUF
   encodes this as ordinary q/k norm tensors: `blk.*.attn_q_norm.weight` is the
   constant `3.87` in all 128 slots and `blk.*.attn_k_norm.weight` is the
   constant `1.0` — verified by byte-probe on
   `Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf`, blocks 0 and 51. Safetensors has
   no such tensors, so the safetensors and GGUF loaders converge on the same
   math from different inputs.
2. **NoPE on global layers.** `layer_rope_theta[i] == 0` for every
   `full_attention` layer, and the reference passes `position_embeddings=None`
   there. So the *global* layers carry no positional signal at all; only the
   2048-window sliding layers are rotated. This is the opposite of the usual
   "global layers get a bigger theta" pattern, and the GGUF metadata does not
   encode it (llama.cpp's `muse-glimmer` graph hardcodes it from
   `sliding_window_pattern`).
3. **The output gate takes `x`, not the attention output.** `gate_proj` is a
   separate `[4096, 6656]` projection applied to the same normed input the
   q/k/v projections see, and its sigmoid multiplies the concatenated heads
   *before* `o_proj`.

Masks: `create_causal_mask` on full layers, `create_sliding_window_causal_mask`
on sliding layers (`kv_idx > q_idx - 2048` **and** `kv_idx <= q_idx`). Eager
materialization is additive `0 / finfo.min`; masked keys have exactly zero
probability after softmax, so the oracle simply excludes them.

### 5. Final head

```
h      = norm(h)                                  # PLAIN RMSNorm, eps = 1e-5
logits = h @ lm_head^T                            # [T, 202048]
logits = logits * output_multiplier                # 0.19611613513818404
logits = logits / final_logit_softcapping          # 20.0
logits = tanh(logits)
logits = logits * final_logit_softcapping
```

The multiplier is applied *before* the cap, so the effective function is
`20 · tanh(0.196116… · z / 20)`. The four lines above are four separate tensor
ops in the reference and are written out in that order because each one is a
separate materialization in the low-precision twin. Both constants are Python
floats — the bf16 twin must take their f32 opmath value, not a rounded one.
`logit_scale` and `final_logit_softcapping` are both present in the GGUF
metadata.

## Vision ops

Preprocessing (`MuseGlimmerImageProcessor`): RGB, rescale `1/255`, normalize
`mean = std = 0.5`, bilinear-ish resize (`resample: 1` = PIL `LANCZOS`), patch
14, temporal patch 2, merge 2, `max_image_tokens: 4096`. Video
(`MuseGlimmerVideoProcessor`): 2 fps sampling, `num_frames: 96`,
`max_video_frame_tokens: 144`, `return_metadata: true`.

Tower (`MuseGlimmerVisionModel`), per image with grid `(t, h, w)` in patches:

```
x = patch_embedding(pixels)                      # Linear [1176 → 1536], NO bias
pos = bilinear_resample(position_embedding_table[32×32], grid)   # see below
x = x + pos
x = ln_pre(x)                                    # LayerNorm, with bias
x = x[window_index]                              # window reorder, window = 32·14 = 448 px
pos_ids = flip(get_vision_position_ids(grid)) + 1   # (w, h), 1-BASED
cos, sin = vision_rope(pos_ids)                  # freq = cat[fw, fh, fw, fh]
for i in 0..49:
    cu = window_cu_seqlens if layer_types[i] == "window_attention" else full_cu_seqlens
    x = x + attn(ln1(x), cu, cos, sin)           # bidirectional, scaling = 96^-0.5
    x = x + fc2(gelu(fc1(ln2(x))))
x = x[argsort(window_index)]                     # undo the reorder
x = ln_post(x)
x = pixel_shuffle(x, grid, merge=2)              # [N, 1536] → [N/4, 6144]
```

Projector:
```
y = gelu( fc2( gelu( fc1(x) ) ) )                # 6144 → 4096 → 4096, no bias
y = vision_projection(y)                         # 4096 → 6656, no bias
y = rmsnorm_weightless(y, eps=rms_norm_eps)      # perception_emb_norm
```

`y` is then scattered into `inputs_embeds` at the positions where
`input_ids == image_token_id` (200092) / `video_token_id` (200091). Because
both the text embedding and `perception_emb_norm` are the *same* weight-less
RMSNorm, text and vision embeddings arrive on the same scale.

Position-embedding resample detail: the HF reference does **not** call
`F.grid_sample`. `get_vision_bilinear_indices_and_weights` (defined in the model
file, not in `vision_utils`, precisely because of this) builds the four corner
indices and their weights explicitly in **f32** and gathers
`position_embedding_table(idx) * weight` — documented as "equivalent with
`F.grid_sample(inputs, align_corners=False, padding='zeros')`", with a code
comment stating that it deliberately differs from the original reference's
`grid_sample` numerics ("we compute manually in fp32"). The sampling grid is
`(arange(h) + 0.5) * (32 / h) - 0.5` per axis, with `floor` (not truncation),
out-of-range corners **zero-weighted rather than clamped**, and the gathered
result reordered by the merge permutation. The oracle must reproduce the
explicit gather, not `grid_sample`.

Vision RoPE detail: `inv_freq` is computed over `spatial_dim = head_dim // 2 =
48`, i.e. `inv_freq[j] = θ^(-2j/48)`, `j ∈ [0, 24)`, and the per-token
frequency vector is `cat[freq_w, freq_h, freq_w, freq_h]` (96 = head_dim). The
position ids are **flipped to (w, h) and offset by +1**. Both are easy to get
backwards and both change every logit.

Video reuses `get_image_features` verbatim — the only difference is `grid_thw`
having `t > 1` and the pixel-shuffle index carrying per-frame offsets.

## DFlash block drafter

Not EAGLE, and not an in-checkpoint MTP layer: a **block-diffusion** drafter
that proposes `block_size = 16` tokens per forward pass. Meta's own numbers
(RTX 5090, K-Quant-17GB + quantized drafter) are 74.9 → 233.4 tok/s, 3.1×.

Per drafting step:

```
ctx  = concat over l ∈ {1,13,25,37,49} of target_hidden[l][accepted positions]   # [n, 33280]
ctx  = encoder.output_norm_enc( encoder.fc(ctx) )                                # [n, 6656]

noise = embed_tokens_RAW([anchor_token, MASK×15])    # [16, 6656], NO embed_norm
for layer in 0..4:
    k/v = concat(proj(ctx), proj(block))             # context k/v then block k/v
    q   = proj(block) only
    q, k = q_norm(q), k_norm(k)   → rope(θ=500000)   # sliding window 2048
    block attends BI-DIRECTIONALLY within itself, causally to ctx
    ... standard pre-norm SwiGLU block (plain RMSNorm, 2 norms per layer)
hn      = norm(block)
logits  = lm_head(hn)[1:]                            # BARE head: no output_multiplier,
                                                     # no softcap, and row 0 dropped
tokens  = argmax(logits)                             # 15 candidates per round
```

Note the rope slice: `q` covers only the 16 block rows while `k` covers
`ctx ++ block`, so the drafter's `apply_rotary_pos_emb` takes
`cos[..., -q_len:, :]` for the queries and the full table for the keys — the
block sits at positions `n .. n+15`, immediately after the context.

Two more details the module makes easy to miss:

* the **projected context is not re-normalized per layer**. Each layer computes
  `k_proj/v_proj` over `concat(ctx, input_layernorm(block))` — the block rows
  are normed, the context rows are the encoder's output unchanged, all five
  layers over.
* every norm in the drafter is a **plain** RMSNorm (weight around 1), including
  `input_layernorm` and `post_attention_layernorm`, whose names match the
  target's zero-centered ones. And the drafter's `MuseGlimmerAssistantRMSNorm`
  casts back to the storage dtype *before* the weight multiply
  (`self.weight * hidden_states.to(input_dtype)`), whereas the target's
  `MuseGlimmerRMSNorm` multiplies first and casts once — same value in f64, one
  extra rounding in a low-precision run.

The cache holds only the context k/v — the 16 block rows are appended and then
evicted each step (`DFlashCache`). `target_layer_ids` are 0-based layer
*outputs*; the GGUF spells the same thing 1-based as
`dflash.target_layers = [2, 14, 26, 38, 50]`.

### The drafting loop — resolved

Read out of `generation/candidate_generator.py::DFlashTokenCandidateGenerator`
and `generation/utils.py::_assisted_decoding` (transformers 5.15.0). Four facts,
all of which change the engine and none of which are visible in
`MuseGlimmerAssistantModel.forward`:

1. **One denoising pass per block. There is no iterative refine loop.**
   `get_candidates` calls the assistant exactly once per round and reads the
   tokens straight off that pass.
2. **A round proposes `block_size - 1 = 15` tokens, not 16.** The noise block is
   `[anchor] + [MASK] × (block_size - 1)` — 16 rows — but the candidate logits
   are `lm_head(last_hidden_state)[:, 1:]`: row 0 is the anchor's own position
   and is **dropped**. Sizing a verification pass for 16 new tokens per round is
   wrong by one.
3. **The drafter's head is the BARE `lm_head`** — `main_model_output_embeddings`
   is `target.get_output_embeddings()`, applied with no `output_multiplier` and
   no softcap. For greedy drafting this is invisible (both are monotone, so the
   argmax is unchanged), but under sampling, `logit_bias`, or temperature the
   drafter's distribution is *not* the target's, and a serving implementation
   that "helpfully" adds the target's tail math will change acceptance rates.
4. **Acceptance is HF's ordinary assisted-decoding rule**, not something
   DFlash-specific: greedy compares the target's argmax to the candidates and
   rolls back to the first mismatch; sampling uses `_speculative_sampling`
   (algorithm 1 of the speculative-decoding paper). Distribution-preserving
   serving therefore needs no new acceptance machinery — only block-shaped
   rollback.

5. **The block's queries sit at absolute positions `n … n+B-1` for masking
   purposes, and that offset comes from the cache, not from `position_ids`.**
   `MuseGlimmerAssistantModel.forward` builds its mask with
   `create_bidirectional_sliding_window_mask(config, inputs_embeds=noise_embeds,
   attention_mask=…, past_key_values=…)` — no `position_ids` — so the query
   index used by the `abs(q_idx - kv_idx) <= sliding_window` overlay comes from
   `DFlashCache.get_query_offset()`, which is `super().get_query_offset() +
   previous_accepted_tokens`. Run the drafter without a `DFlashCache` (or
   without `set_previous_accepted_tokens(n)`) and the block is masked as if it
   sat at positions `0 … B-1`: for a short context every key is inside the
   window either way and nothing looks wrong, but the drafted tokens are
   different. Measured on the tiny harness: `[113, 94, 94]` without the offset
   versus `[356, 356, 356]` with it.

   There is no causal constraint anywhere in the drafter — the bidirectional
   mask allows every `(q, kv)` pair inside the window, block-to-block included.

Context bookkeeping: `context_hidden_states` are the target's hidden states at
`target_layer_ids` for the **accepted positions only**
(`n_last_matches + 1` rows after the first round, `T - 1` on the prefill round),
concatenated on the last dim; `cache.crop(-block_size)` evicts the previous
round's block rows before the next call, so the cache holds context k/v only.
The anchor token is `input_ids[:, -1:]`, and the drafter embeds it with the
target's **raw** table (`F.embedding(ids, embed_tokens.weight)`) — no
`embed_norm`, which is why `MuseGlimmerTextNormedEmbedding` keeps the norm
outside the table.

## Serving contract (`chat_template.jinja`, "ATEM")

A Harmony-style channel protocol. Header: `<|start|>{role}[ to={recipient}]<|message|>`,
terminator `<|eom|>` (turn continues) or `<|eot|>` (turn ends).

- **System block** is always synthesized when absent, and always carries
  `Reasoning strength: {low|medium|high|xhigh}` (default `high`), the tool
  definitions, and a `# Valid recipients: "self", "<ns>.*", "user".` line.
- **Reasoning** is `<|start|>assistant to=self<|message|>…<|eom|>`.
- **Answer** is `<|start|>assistant to=user<|message|>…<|eot|>`.
- **Tool call** is `<|start|>assistant to={tool name}<|message|>` followed by
  `<atem:function_calls><atem:invoke name="…"><atem:parameter name="k">v</atem:parameter>…</atem:invoke></atem:function_calls>`.
- **Tool result** is `<|start|>tool {name}<|message|><tool_output name="{name}">\n…\n</tool_output><|eot|>`.
- Media placeholders in the template are `<|patch|>` (image) and `<|video|>`;
  the processor expands them into runs of token 200092 / 200091.
- The generation prompt is bare `<|start|>assistant` — **the model itself emits
  the ` to=…` recipient**. That is the grammar hook for `tool_choice`.
- `tokenizer_config.response_template` gives the machine-readable parse
  contract (open/close patterns per field) and should be the parser's source of
  truth rather than hand-written regexes.
- The template `raise_exception`s if `tool_calls[].function.arguments` is a
  string rather than a mapping — the same OpenAI-transport mismatch the gemma4
  server fixes by deserializing at the wire boundary.

Sampling defaults from `generation_config.json`: `temperature 1.0`,
`top_p 0.95`, `top_k 64`; `eos ∈ {200001, 200008}`.

## Numerics policy (oracle)

Identical in spirit to the qwen35 and gemma4 oracles, and deliberately so — the
three are meant to be refereed the same way.

Pure IEEE f64 end to end: BF16 weights are converted exactly, every intermediate
is f64, and every reduction has a **fixed, thread-count- and ISA-independent
order** (parallelism only across output rows / heads / positions). Built with
`-ffp-contract=off`, no fast-math. GEMM-class reductions use the 8-lane
fma-blocked order of `simd.hpp` (lane `l` accumulates `i ≡ l mod 8` via IEEE
fma; lanes combine as `((l0+l1)+(l2+l3)) + ((l4+l5)+(l6+l7))`, plus a sequential
fma tail), so the AVX-512 and scalar paths are **bitwise identical**.

Stock HF, even when run in f64, rounds through f32 at hard-coded casts: both
RMSNorm kinds, the weight-less QK-norm and embed-norm, the RoPE tables (text and
vision), the eager softmax, and the vision RoPE application. The oracle lifts
all of them. Equivalence to HF is proven with a precision-only patch of the
reference; the gap to *stock* unpatched HF-f64 is measured and published.

One torch-CPU quirk is load-bearing for the norms and is easy to get backwards
(measured on torch 2.13.0+cpu, `py/probe_torch_ops.py`, and recorded in
VERIFICATION.md): **`torch.rsqrt(x)` and `torch.pow(x, -0.5)` are the correctly
rounded values in f64 and agree bit-for-bit with C's `1.0 / std::sqrt(x)`,
while `torch.sqrt` is a ~1-ulp approximation** — so `1.0 / torch.sqrt(x)`
differs from `torch.rsqrt(x)` on ~1% of elements by up to 2 ulp. The oracle's
norms therefore compute `1.0 / std::sqrt(ms)`, and a reference written with
`1 / torch.sqrt(...)` would not referee them. The reference's use of
`torch.pow(ms, -0.5)` in `MuseGlimmerRMSNorm` and `torch.rsqrt` in
`MuseGlimmerTextCenteredRMSNorm` is therefore *not* a numerical difference
between the two norm kinds, as the reference's own comment about matching JAX
might suggest — the two are bit-identical.

Low-precision twins (`--dtype bf16|f16`) follow the same rules the qwen35 oracle
documents: `rnd(x)` = RNE(f64→f32) then RNE(f32→bf16/f16); one rounding per
`nn.Linear` output, per residual add, per activation, per elementwise product;
f32-materialized regions round per-op to f32; Python-scalar constants
(`D^-0.5`, `qk_scale_factor`, `output_multiplier`, `final_logit_softcapping`)
take their f32 opmath value.
