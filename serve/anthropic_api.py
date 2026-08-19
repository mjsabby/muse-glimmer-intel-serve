"""Anthropic Messages (`/v1/messages`) over the same generation loop.

The translation is mostly mechanical, with one thing worth naming: Anthropic's
`thinking` blocks and OpenAI's `reasoning_content` are the SAME channel here —
`to=self` — so neither protocol needs a heuristic to separate reasoning from
the answer. The model already did.
"""
from __future__ import annotations

import json
import uuid

from fastapi import Header, Request
from fastapi.responses import JSONResponse, StreamingResponse

from serve import chatlib


def _rid() -> str:
    return "msg_" + uuid.uuid4().hex[:24]


def _blocks(content) -> list:
    return content if isinstance(content, list) else [{"type": "text", "text": content or ""}]


def to_openai_messages(body: dict) -> list:
    """Anthropic request -> the message list the chat template renders."""
    out: list = []
    system = body.get("system")
    if system:
        text = system if isinstance(system, str) else "".join(
            b.get("text", "") for b in system if b.get("type") == "text")
        out.append({"role": "system", "content": text})

    for m in body.get("messages") or []:
        role = m.get("role")
        blocks = _blocks(m.get("content"))
        if role == "user":
            parts, tool_results = [], []
            for b in blocks:
                t = b.get("type")
                if t == "text":
                    parts.append({"type": "text", "text": b.get("text", "")})
                elif t == "image":
                    src = b.get("source") or {}
                    if src.get("type") == "base64":
                        url = f"data:{src.get('media_type', 'image/png')};base64,{src.get('data', '')}"
                    else:
                        url = src.get("url", "")
                    parts.append({"type": "image_url", "image_url": {"url": url}})
                elif t == "tool_result":
                    body_text = b.get("content")
                    if isinstance(body_text, list):
                        body_text = "".join(x.get("text", "") for x in body_text
                                            if x.get("type") == "text")
                    tool_results.append({"role": "tool",
                                         "tool_call_id": b.get("tool_use_id", ""),
                                         "content": body_text or ""})
            # Tool results are their own turn in the template's world, and they
            # have to precede whatever the user said next.
            out.extend(tool_results)
            if parts:
                out.append({"role": "user", "content": parts})
        elif role == "assistant":
            msg = {"role": "assistant", "content": ""}
            calls = []
            for b in blocks:
                t = b.get("type")
                if t == "text":
                    msg["content"] += b.get("text", "")
                elif t == "thinking":
                    msg["reasoning_content"] = msg.get("reasoning_content", "") + \
                        b.get("thinking", "")
                elif t == "tool_use":
                    calls.append({"id": b.get("id", ""), "type": "function",
                                  "function": {"name": b.get("name", ""),
                                               "arguments": b.get("input") or {}}})
            if calls:
                msg["tool_calls"] = calls
            out.append(msg)
    return out


def to_openai_tools(body: dict):
    tools = []
    for t in body.get("tools") or []:
        if t.get("type", "custom") not in ("custom", "function", None):
            continue  # server-side tool types this deployment cannot run
        tools.append({"type": "function",
                      "function": {"name": t.get("name"),
                                   "description": t.get("description", ""),
                                   "parameters": t.get("input_schema")
                                   or t.get("parameters") or {}}})
    return chatlib.sanitize_tools(tools) if tools else None


def to_openai_tool_choice(body: dict):
    tc = body.get("tool_choice")
    if not tc:
        return None
    kind = tc.get("type") if isinstance(tc, dict) else tc
    if kind == "auto":
        return None
    if kind == "any":
        return "required"
    if kind == "none":
        return "none"
    if kind == "tool":
        return {"function": {"name": tc.get("name")}}
    return None


def to_openai_body(body: dict) -> dict:
    """Sampling and format knobs, renamed."""
    out = {"max_tokens": body.get("max_tokens", 1024),
           "temperature": body.get("temperature"),
           "top_p": body.get("top_p"),
           "stop": body.get("stop_sequences") or [],
           "tools": None, "tool_choice": to_openai_tool_choice(body)}
    if body.get("top_k") is not None:
        out["top_k"] = body["top_k"]
    thinking = body.get("thinking")
    if isinstance(thinking, dict):
        # Anthropic sends a token budget; this model takes a strength. Map the
        # budget onto the four levels rather than inventing a fifth.
        if thinking.get("type") == "disabled":
            out["reasoning_effort"] = "low"
        else:
            b = int(thinking.get("budget_tokens") or 0)
            out["reasoning_effort"] = ("low" if b and b < 2048 else
                                       "medium" if b and b < 8192 else
                                       "high" if b and b < 32768 else "xhigh")
    return {k: v for k, v in out.items() if v is not None}


STOP_REASON = {"stop": "end_turn", "length": "max_tokens", "tool_calls": "tool_use"}


def to_anthropic_response(collected: dict, model: str) -> dict:
    content = []
    if collected["reasoning"]:
        content.append({"type": "thinking", "thinking": collected["reasoning"],
                        "signature": ""})
    if collected["content"]:
        content.append({"type": "text", "text": collected["content"]})
    for tc in collected["tool_calls"]:
        fn = tc["function"]
        content.append({"type": "tool_use", "id": "toolu_" + uuid.uuid4().hex[:20],
                        "name": fn["name"], "input": json.loads(fn["arguments"] or "{}")})
    u = collected["usage"]
    return {"id": _rid(), "type": "message", "role": "assistant", "model": model,
            "content": content,
            "stop_reason": STOP_REASON.get(collected["finish_reason"], "end_turn"),
            "stop_sequence": None,
            "usage": {"input_tokens": u.get("prompt_tokens", 0),
                      "output_tokens": u.get("completion_tokens", 0)}}


def _ev(kind: str, payload: dict) -> str:
    return f"event: {kind}\ndata: {json.dumps(payload)}\n\n"


def anthropic_sse(runner, messages, obody, model: str, anth: dict):
    """Anthropic's block-structured stream. Blocks open and close as the model
    switches channels, which is exactly what the parser reports."""
    ident = _rid()
    yield _ev("message_start", {"type": "message_start", "message": {
        "id": ident, "type": "message", "role": "assistant", "model": model,
        "content": [], "stop_reason": None, "stop_sequence": None,
        "usage": {"input_tokens": 0, "output_tokens": 0}}})
    idx = -1
    open_kind = None
    finish, usage = "stop", {}
    ntools = 0

    def close():
        nonlocal open_kind
        if open_kind is not None:
            out = _ev("content_block_stop", {"type": "content_block_stop", "index": idx})
            open_kind = None
            return out
        return ""

    for ev in runner.stream(messages, obody):
        t = ev["type"]
        if t in ("content", "reasoning"):
            kind = "text" if t == "content" else "thinking"
            if open_kind != kind:
                pre = close()
                idx += 1
                open_kind = kind
                block = ({"type": "text", "text": ""} if kind == "text"
                         else {"type": "thinking", "thinking": ""})
                yield pre + _ev("content_block_start",
                                {"type": "content_block_start", "index": idx,
                                 "content_block": block})
            delta = ({"type": "text_delta", "text": ev["text"]} if kind == "text"
                     else {"type": "thinking_delta", "thinking": ev["text"]})
            yield _ev("content_block_delta",
                      {"type": "content_block_delta", "index": idx, "delta": delta})
        elif t == "tool_calls":
            for tc in ev["calls"]:
                pre = close()
                idx += 1
                ntools += 1
                fn = tc["function"]
                yield pre + _ev("content_block_start", {
                    "type": "content_block_start", "index": idx,
                    "content_block": {"type": "tool_use",
                                      "id": "toolu_" + uuid.uuid4().hex[:20],
                                      "name": fn["name"], "input": {}}})
                yield _ev("content_block_delta", {
                    "type": "content_block_delta", "index": idx,
                    "delta": {"type": "input_json_delta",
                              "partial_json": fn["arguments"] or "{}"}})
                yield _ev("content_block_stop", {"type": "content_block_stop", "index": idx})
        elif t == "done":
            finish, usage = ev["finish_reason"], ev["usage"]
    yield close()
    if ntools and finish == "stop":
        finish = "tool_calls"
    yield _ev("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": STOP_REASON.get(finish, "end_turn"), "stop_sequence": None},
        "usage": {"output_tokens": usage.get("completion_tokens", 0)}})
    yield _ev("message_stop", {"type": "message_stop"})


def mount_anthropic(app, runner, auth, run_blocking) -> None:
    from serve.server import ApiError, _collect

    @app.post("/v1/messages")
    async def messages(request: Request, authorization: str | None = Header(None),
                       x_api_key: str | None = Header(None)):
        auth(authorization, x_api_key)
        anth = await request.json()
        if not anth.get("messages"):
            raise ApiError("messages must be a non-empty array")
        msgs = to_openai_messages(anth)
        obody = to_openai_body(anth)
        obody["tools"] = to_openai_tools(anth)
        model = anth.get("model") or runner.model_id
        if anth.get("stream"):
            async def sse():
                async for piece in run_blocking(
                        lambda: anthropic_sse(runner, msgs, obody, model, anth), request):
                    yield piece
            return StreamingResponse(sse(), media_type="text/event-stream")
        out = _collect(runner.stream(msgs, obody))
        return JSONResponse(to_anthropic_response(out, model))

    @app.post("/v1/messages/count_tokens")
    async def count_tokens(request: Request, authorization: str | None = Header(None),
                           x_api_key: str | None = Header(None)):
        auth(authorization, x_api_key)
        anth = await request.json()
        msgs = to_openai_messages(anth)
        tools = to_openai_tools(anth)
        from serve.media import split_parts
        plain, media = split_parts(msgs, runner.pre)
        text = chatlib.render(runner.tok, plain, tools=tools,
                              reasoning=runner.defaults.reasoning)
        if media:
            from serve.media import expand_placeholders
            text = expand_placeholders(text, media, runner.eng.merge_unit)
        return JSONResponse({"input_tokens": len(chatlib.encode(runner.tok, text))})
