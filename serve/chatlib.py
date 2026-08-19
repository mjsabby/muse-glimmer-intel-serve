"""Tokenizer, chat rendering, and the checkpoint-derived serving constants.

Rendering goes through the checkpoint's own `chat_template.jinja` via HF
`apply_chat_template`, so the prompt is byte-exact by construction rather than
by a hand-written imitation that drifts on the next revision.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

# ---------------------------------------------------------------- the protocol
#
# Muse Glimmer's assistant turn is a sequence of CHANNELS. The generation
# prompt is a bare `<|start|>assistant`, and the model itself emits the
# recipient:
#
#   <|start|>assistant to=self<|message|>   ... reasoning ...     <|eom|>
#   <|start|>assistant to=user<|message|>   ... the answer ...    <|eot|>
#   <|start|>assistant to=<tool><|message|> ... an ATEM call ...  <|eot|>
#
# <|eom|> ends a channel and the turn continues; <|eot|> ends the turn. That
# distinction is the whole reason `reasoning_content` never leaks into
# `content` here: they are different channels, not different regions of one
# string.
BOS = "<|begin_of_text|>"
START = "<|start|>"
MESSAGE = "<|message|>"
EOM = "<|eom|>"
EOT = "<|eot|>"
END_OF_TEXT = "<|end_of_text|>"

REASONING_LEVELS = ("low", "medium", "high", "xhigh")
DEFAULT_REASONING = "high"


def resolve_dir(model: str) -> str:
    """Repo id or directory -> a snapshot directory holding config.json."""
    p = Path(model)
    if (p / "config.json").exists():
        return str(p)
    root = (os.environ.get("HF_HUB_CACHE")
            or (os.environ.get("HF_HOME", "") + "/hub" if os.environ.get("HF_HOME") else "")
            or str(Path.home() / ".cache/huggingface/hub"))
    repo = Path(root) / ("models--" + model.replace("/", "--"))
    ref = repo / "refs" / "main"
    if ref.exists():
        snap = repo / "snapshots" / ref.read_text().strip()
        if (snap / "config.json").exists():
            return str(snap)
    snaps = sorted((repo / "snapshots").glob("*")) if (repo / "snapshots").exists() else []
    for s in snaps:
        if (s / "config.json").exists():
            return str(s)
    raise FileNotFoundError(f"{model} is neither a directory with config.json nor cached at {repo}")


def load_tokenizer(model: str):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(resolve_dir(model))


def load_json(model: str, name: str) -> dict:
    p = Path(resolve_dir(model)) / name
    return json.loads(p.read_text()) if p.exists() else {}


def eos_ids(tokenizer, model: str) -> set[int]:
    """Every id that ends a TURN.

    <|eom|> is deliberately not here: it ends a channel, and stopping on it
    would truncate the answer to its reasoning."""
    gen = load_json(model, "generation_config.json")
    ids = gen.get("eos_token_id", [])
    if isinstance(ids, int):
        ids = [ids]
    out = {int(i) for i in ids}
    for tok in (END_OF_TEXT, EOT):
        i = tokenizer.convert_tokens_to_ids(tok)
        if isinstance(i, int) and i >= 0:
            out.add(i)
    return out


def special_ids(tokenizer) -> dict[str, int]:
    return {t: int(tokenizer.convert_tokens_to_ids(t))
            for t in (BOS, START, MESSAGE, EOM, EOT, END_OF_TEXT)}


def default_sampling(model: str) -> dict:
    """generation_config.json's own defaults, so `temperature` unset means what
    the checkpoint says it means (1.0 / 0.95 / 64) rather than what some other
    server's default happened to be."""
    g = load_json(model, "generation_config.json")
    return {"temperature": float(g.get("temperature", 1.0)),
            "top_p": float(g.get("top_p", 0.95)),
            "top_k": int(g.get("top_k", 64))}


# ------------------------------------------------------------------ rendering

class ToolShapeError(ValueError):
    """A tools[] entry the template cannot render. Surfaced as a 400."""


def _infer_type(schema: dict) -> str:
    """A JSON-Schema fragment with no `type`. Agent clients send these; the
    template renders them verbatim and the model then guesses. Infer instead."""
    if "properties" in schema or "required" in schema:
        return "object"
    if "items" in schema:
        return "array"
    if "enum" in schema and schema["enum"]:
        v = schema["enum"][0]
        return {str: "string", bool: "boolean", int: "integer", float: "number"}.get(type(v),
                                                                                    "string")
    return "string"


def _fix_properties(props: dict) -> dict:
    out = {}
    for k, v in (props or {}).items():
        if not isinstance(v, dict):
            out[k] = v
            continue
        v = dict(v)
        if "type" not in v and not ({"anyOf", "oneOf", "allOf", "$ref"} & set(v)):
            v["type"] = _infer_type(v)
        if v.get("type") == "object" and "properties" in v:
            v["properties"] = _fix_properties(v["properties"])
        if v.get("type") == "array" and isinstance(v.get("items"), dict):
            v["items"] = _fix_properties({"i": v["items"]})["i"]
        out[k] = v
    return out


def sanitize_tools(tools):
    """Normalize an OpenAI tools[] array into what the template expects."""
    if not tools:
        return None
    out = []
    for t in tools:
        if not isinstance(t, dict):
            raise ToolShapeError(f"tools[] entries must be objects, got {type(t).__name__}")
        fn = t.get("function", t)
        name = fn.get("name")
        if not name:
            raise ToolShapeError("every tool needs a function.name")
        params = fn.get("parameters") or {"type": "object", "properties": {}}
        if isinstance(params, dict):
            params = dict(params)
            params.setdefault("type", "object")
            if "properties" in params:
                params["properties"] = _fix_properties(params["properties"])
        out.append({"type": "function",
                    "function": {"name": name,
                                 "description": fn.get("description", ""),
                                 "parameters": params}})
    return out


def normalize_tool_calls(messages: list) -> list:
    """Deserialize `tool_calls[].function.arguments`.

    The wire format is a JSON *string*; the template calls `args.items()` and
    `raise_exception`s on anything that is not a mapping. Every agentic second
    turn would 400 without this, which is exactly the kind of failure that
    looks like a model problem and is not."""
    out = []
    for m in messages:
        if not isinstance(m, dict) or not m.get("tool_calls"):
            out.append(m)
            continue
        m = dict(m)
        calls = []
        for tc in m["tool_calls"]:
            tc = dict(tc)
            fn = dict(tc.get("function") or {})
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    fn["arguments"] = json.loads(args) if args.strip() else {}
                except json.JSONDecodeError:
                    # A model that emitted un-parseable arguments still has to
                    # be renderable on the next turn, or the conversation is
                    # stuck forever. Keep the text where the model can see it.
                    fn["arguments"] = {"_raw": args}
            elif args is None:
                fn["arguments"] = {}
            tc["function"] = fn
            calls.append(tc)
        m["tool_calls"] = calls
        out.append(m)
    return out


def render(tokenizer, messages: list[dict], *, tools=None, add_generation_prompt: bool = True,
           reasoning: str = DEFAULT_REASONING, **kwargs) -> str:
    """Rendered prompt text. Tokenize with add_special_tokens=False: the
    template emits the BOS itself."""
    if reasoning not in REASONING_LEVELS:
        raise ValueError(f"reasoning strength must be one of {REASONING_LEVELS}")
    return tokenizer.apply_chat_template(
        normalize_tool_calls(messages), tools=tools, add_generation_prompt=add_generation_prompt,
        tokenize=False, reasoning_strength=reasoning, **kwargs)


def encode(tokenizer, text: str) -> list[int]:
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def render_ids(tokenizer, messages, **kw) -> list[int]:
    return encode(tokenizer, render(tokenizer, messages, **kw))


# ------------------------------------------------------------ capacity planning

def kv_bytes_per_token(model: str) -> float:
    """Cache slope from the checkpoint, so --max-seq guidance is derived rather
    than remembered. Only the GLOBAL layers grow: the sliding ones hold a fixed
    `window + chunk` ring."""
    cfg = load_json(model, "config.json")
    txt = cfg.get("text_config", cfg)
    types = txt.get("layer_types") or []
    heads = int(txt.get("num_key_value_heads", 1))
    dim = int(txt.get("head_dim") or (int(txt["hidden_size"]) // int(txt["num_attention_heads"])))
    n_global = sum(1 for t in types if "sliding" not in str(t)) or int(
        txt.get("num_hidden_layers", 0))
    return n_global * heads * dim * 2 * 2  # K and V, bf16


def weight_bytes(model: str) -> int:
    d = Path(resolve_dir(model))
    idx = d / "model.safetensors.index.json"
    if idx.exists():
        meta = json.loads(idx.read_text()).get("metadata", {})
        if "total_size" in meta:
            return int(meta["total_size"])
    return sum(f.stat().st_size for f in d.glob("*.safetensors"))
