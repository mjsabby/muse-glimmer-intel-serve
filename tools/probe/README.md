# Checkpoint probes

Four small scripts that read checkpoint metadata and individual tensor values
over HTTP range requests, without downloading the weights. They are what
established the verified facts in [`../../ARCHITECTURE.md`](../../ARCHITECTURE.md),
and they are the cheapest way to re-check those facts when a checkpoint is
revised.

All four take a URL and print to stdout. Python 3 and `curl` only.

```bash
# safetensors: every tensor name, dtype and shape (reads the JSON header only)
python3 st_head.py https://huggingface.co/meta-models/Muse-Glimmer-30B/resolve/main/model-00001-of-00002.safetensors

# safetensors: the first 16 values of named tensors, decoded from BF16
python3 st_probe.py <url> model.language_model.layers.0.input_layernorm.weight,model.language_model.norm.weight

# GGUF: all KV metadata plus deduplicated tensor shapes/types
#       (arg 2 = bytes to fetch; the tokenizer arrays push the header past 12 MB)
python3 gguf_head.py https://huggingface.co/meta-models/Muse-Glimmer-30B-GGUF/resolve/main/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf 40000000

# GGUF: min/max/mean/first-8 of named F32 tensors
python3 gguf_probe.py <url> blk.0.attn_q_norm.weight,blk.0.attn_norm.weight,output_norm.weight 40000000
```

## The three checks worth re-running

1. **Norm centering.** `st_probe` on `layers.0.input_layernorm.weight` and
   `gguf_probe` on `blk.0.attn_norm.weight` must differ by exactly `1.0`
   elementwise; `norm.weight` and `output_norm.weight` must be identical. That
   asymmetry is what the GGUF loader has to reproduce.
2. **QK-norm constants.** `blk.*.attn_q_norm.weight` must be the constant
   `3.87` (= `qk_scale_factor`) and `blk.*.attn_k_norm.weight` the constant
   `1.0` in every block. Safetensors has no such tensors — this is purely the
   GGUF spelling of weight-less QK-norm.
3. **DFlash wiring.** `gguf_head` on the `dflash-*` GGUF must show
   `dflash.block_size = 16` and `dflash.target_layers = [2, 14, 26, 38, 50]`
   (1-based), matching the assistant config's 0-based
   `target_layer_ids = [1, 13, 25, 37, 49]`.
