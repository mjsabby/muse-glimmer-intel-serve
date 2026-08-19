"""OpenAI Responses (`/v1/responses`) over the same generation loop.

Responses models a turn as a list of typed OUTPUT ITEMS — reasoning, message,
function_call — which is a closer fit to this model than Chat Completions is:
its channels already are typed items, so nothing has to be inferred from
markers in a string.

`store` is accepted and ignored: this server keeps no request history, and
`previous_response_id` is refused rather than silently answered without the
context it names.
"""
from __future__ import annotations

import json
import time
import uuid

from fastapi import Header, Request
from fastapi.responses import JSONResponse, StreamingResponse

from serve import chatlib


def _rid() -> str:
    return "resp_" + uuid.uuid4().hex[:24]


def _now() -> int:
    return int(time.time())


def to_openai_messages(body: dict) -> list:
    """Responses `input` (+ `instructions`) -> chat messages."""
    out = []
    if body.get("instructions"):
        out.append({"role": "system", "content": body["instructions"]})
    inp = body.get("input")
    if isinstance(inp, str):
        out.append({"role": "user", "content": inp})
        return out
    for item in inp or []:
        t = item.get("type", "message")
        if t == "function_call_output":
            out.append({"role": "tool", "tool_call_id": item.get("call_id", ""),
                        "content": item.get("output", "")})
            continue
        if t == "function_call":
            args = item.get("arguments")
            out.append({"role": "assistant", "content": "", "tool_calls": [
                {"id": item.get("call_id", ""), "type": "function",
                 "function": {"name": item.get("name", ""), "arguments": args}}]})
            continue
        if t == "reasoning":
            text = "".join(c.get("text", "") for c in (item.get("content") or []))
            if text:
                out.append({"role": "assistant", "content": "", "reasoning_content": text})
            continue
        role = item.get("role", "user")
        content = item.get("content")
        if isinstance(content, str):
            out.append({"role": role, "content": content})
            continue
        parts = []
        for c in content or []:
            ct = c.get("type")
            if ct in ("input_text", "output_text", "text"):
                parts.append({"type": "text", "text": c.get("text", "")})
            elif ct in ("input_image", "image"):
                url = c.get("image_url") or c.get("url")
                if isinstance(url, dict):
                    url = url.get("url")
                parts.append({"type": "image_url", "image_url": {"url": url}})
            elif ct in ("input_video", "video"):
                url = c.get("video_url") or c.get("url")
                if isinstance(url, dict):
                    url = url.get("url")
                parts.append({"type": "video_url", "video_url": {"url": url}})
        out.append({"role": role, "content": parts})
    return out


def to_openai_tools(body: dict):
    tools = []
    for t in body.get("tools") or []:
        if t.get("type") != "function":
            continue                       # hosted tool types this server has no way to run
        fn = t.get("function", t)
        tools.append({"type": "function",
                      "function": {"name": fn.get("name"),
                                   "description": fn.get("description", ""),
                                   "parameters": fn.get("parameters") or {}}})
    return chatlib.sanitize_tools(tools) if tools else None


def to_openai_body(body: dict) -> dict:
    out = {"max_tokens": body.get("max_output_tokens") or body.get("max_tokens"),
           "temperature": body.get("temperature"),
           "top_p": body.get("top_p"),
           "tool_choice": body.get("tool_choice"),
           "stop": body.get("stop") or []}
    if isinstance(body.get("reasoning"), dict):
        out["reasoning_effort"] = body["reasoning"].get("effort")
    fmt = ((body.get("text") or {}).get("format") or {})
    if fmt.get("type") == "json_schema":
        out["response_format"] = {"type": "json_schema",
                                  "json_schema": {"schema": fmt.get("schema") or {}}}
    elif fmt.get("type") == "json_object":
        out["response_format"] = {"type": "json_object"}
    return {k: v for k, v in out.items() if v is not None}


STATUS = {"stop": "completed", "length": "incomplete", "tool_calls": "completed"}


def to_response(collected: dict, model: str, ident: str) -> dict:
    output = []
    if collected["reasoning"]:
        output.append({"id": "rs_" + uuid.uuid4().hex[:20], "type": "reasoning",
                       "summary": [],
                       "content": [{"type": "reasoning_text",
                                    "text": collected["reasoning"]}]})
    if collected["content"]:
        output.append({"id": "msg_" + uuid.uuid4().hex[:20], "type": "message",
                       "role": "assistant", "status": "completed",
                       "content": [{"type": "output_text", "text": collected["content"],
                                    "annotations": []}]})
    for tc in collected["tool_calls"]:
        fn = tc["function"]
        output.append({"id": "fc_" + uuid.uuid4().hex[:20], "type": "function_call",
                       "call_id": "call_" + uuid.uuid4().hex[:20],
                       "name": fn["name"], "arguments": fn["arguments"],
                       "status": "completed"})
    u = collected["usage"]
    return {"id": ident, "object": "response", "created_at": _now(), "model": model,
            "status": STATUS.get(collected["finish_reason"], "completed"),
            "incomplete_details": ({"reason": "max_output_tokens"}
                                   if collected["finish_reason"] == "length" else None),
            "output": output,
            "output_text": collected["content"],
            "usage": {"input_tokens": u.get("prompt_tokens", 0),
                      "output_tokens": u.get("completion_tokens", 0),
                      "total_tokens": u.get("total_tokens", 0)}}


def _ev(kind: str, payload: dict, seq: list) -> str:
    payload = {"type": kind, "sequence_number": seq[0], **payload}
    seq[0] += 1
    return f"event: {kind}\ndata: {json.dumps(payload)}\n\n"


def responses_sse(runner, messages, obody, model: str):
    ident = _rid()
    seq = [0]
    yield _ev("response.created", {"response": {"id": ident, "object": "response",
                                                "status": "in_progress", "model": model,
                                                "created_at": _now(), "output": []}}, seq)
    out_index = -1
    open_kind = None
    item_id = None
    collected = {"content": "", "reasoning": "", "tool_calls": [], "finish_reason": "stop",
                 "usage": {}, "logprobs": [], "spec": None}

    def close():
        nonlocal open_kind
        if open_kind is None:
            return ""
        pieces = ""
        if open_kind == "content":
            pieces += _ev("response.output_text.done",
                          {"item_id": item_id, "output_index": out_index,
                           "content_index": 0, "text": collected["content"]}, seq)
            pieces += _ev("response.content_part.done",
                          {"item_id": item_id, "output_index": out_index,
                           "content_index": 0,
                           "part": {"type": "output_text", "text": collected["content"]}}, seq)
        pieces += _ev("response.output_item.done",
                      {"output_index": out_index,
                       "item": {"id": item_id, "type": (
                           "message" if open_kind == "content" else "reasoning")}}, seq)
        open_kind = None
        return pieces

    for ev in runner.stream(messages, obody):
        t = ev["type"]
        if t in ("content", "reasoning"):
            if open_kind != t:
                pre = close()
                out_index += 1
                open_kind = t
                item_id = ("msg_" if t == "content" else "rs_") + uuid.uuid4().hex[:20]
                item = ({"id": item_id, "type": "message", "role": "assistant",
                         "status": "in_progress", "content": []} if t == "content"
                        else {"id": item_id, "type": "reasoning", "summary": []})
                pre += _ev("response.output_item.added",
                           {"output_index": out_index, "item": item}, seq)
                if t == "content":
                    pre += _ev("response.content_part.added",
                               {"item_id": item_id, "output_index": out_index,
                                "content_index": 0,
                                "part": {"type": "output_text", "text": ""}}, seq)
                yield pre
            collected[t] += ev["text"]
            kind = ("response.output_text.delta" if t == "content"
                    else "response.reasoning_text.delta")
            yield _ev(kind, {"item_id": item_id, "output_index": out_index,
                             "content_index": 0, "delta": ev["text"]}, seq)
        elif t == "tool_calls":
            for tc in ev["calls"]:
                pre = close()
                out_index += 1
                fid = "fc_" + uuid.uuid4().hex[:20]
                call_id = "call_" + uuid.uuid4().hex[:20]
                fn = tc["function"]
                collected["tool_calls"].append(tc)
                pre += _ev("response.output_item.added", {
                    "output_index": out_index,
                    "item": {"id": fid, "type": "function_call", "call_id": call_id,
                             "name": fn["name"], "arguments": "", "status": "in_progress"}}, seq)
                pre += _ev("response.function_call_arguments.delta",
                           {"item_id": fid, "output_index": out_index,
                            "delta": fn["arguments"]}, seq)
                pre += _ev("response.function_call_arguments.done",
                           {"item_id": fid, "output_index": out_index,
                            "arguments": fn["arguments"]}, seq)
                pre += _ev("response.output_item.done", {
                    "output_index": out_index,
                    "item": {"id": fid, "type": "function_call", "call_id": call_id,
                             "name": fn["name"], "arguments": fn["arguments"],
                             "status": "completed"}}, seq)
                yield pre
        elif t == "done":
            collected.update({k: ev[k] for k in ("finish_reason", "usage", "logprobs", "spec")})
    yield close()
    yield _ev("response.completed", {"response": to_response(collected, model, ident)}, seq)


def mount_responses(app, runner, auth, run_blocking) -> None:
    from serve.server import ApiError, _collect

    @app.post("/v1/responses")
    async def responses(request: Request, authorization: str | None = Header(None),
                        x_api_key: str | None = Header(None)):
        auth(authorization, x_api_key)
        body = await request.json()
        if body.get("previous_response_id"):
            raise ApiError("previous_response_id is not supported: this server stores no "
                           "conversation state, so it cannot answer as if it had")
        if not body.get("input") and not body.get("instructions"):
            raise ApiError("input is required")
        msgs = to_openai_messages(body)
        obody = to_openai_body(body)
        obody["tools"] = to_openai_tools(body)
        model = body.get("model") or runner.model_id
        if body.get("stream"):
            async def sse():
                async for piece in run_blocking(
                        lambda: responses_sse(runner, msgs, obody, model), request):
                    yield piece
            return StreamingResponse(sse(), media_type="text/event-stream")
        out = _collect(runner.stream(msgs, obody))
        return JSONResponse(to_response(out, model, _rid()))
