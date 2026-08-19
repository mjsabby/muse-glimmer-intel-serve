"""HTTP frontend: OpenAI Chat Completions, OpenAI Responses, Anthropic Messages.

One engine, one KV cache, one request at a time — the lock is not a placeholder
for a batching scheduler, it is the honest shape of a two-card deployment whose
whole design is a static allocation. Concurrency here would mean either a
second cache or interleaved requests through one, and both undo the guarantee
that made the memory footprint checkable in the first place.

    python -m serve.server --model meta-models/Muse-Glimmer-30B --gpus 2 \\
        --assistant meta-models/Muse-Glimmer-30B-assistant --q8-assistant
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import threading
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Imported at MODULE scope on purpose: `from __future__ import annotations`
# makes every annotation a string, and FastAPI resolves them against the
# module globals. Importing Request inside create_app() left it unresolvable,
# and FastAPI then treated `request: Request` as a query parameter — every
# endpoint 422'd with "field required: query.request".
from fastapi import FastAPI, Header, Request                     # noqa: E402
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse  # noqa: E402

from serve import chatlib                                        # noqa: E402
from serve.atem import ResponseTemplate                          # noqa: E402
from serve.engine import Engine, VISION_CPU, VISION_GPU, VISION_OFF  # noqa: E402
from serve.generate import GenParams, Generation                 # noqa: E402
from serve.media import MediaError, Preprocessor, expand_placeholders, split_parts  # noqa: E402
from serve.recipient import recipients_for                       # noqa: E402
from serve.sampler import SampleParams                           # noqa: E402


def now() -> int:
    return int(time.time())


def rid(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:24]}"


class ApiError(Exception):
    def __init__(self, message: str, status: int = 400, kind: str = "invalid_request_error"):
        super().__init__(message)
        self.message, self.status, self.kind = message, status, kind


# --------------------------------------------------------------------- runner

class Runner:
    """Everything a request needs, behind one lock."""

    def __init__(self, eng: Engine, tok, rt: ResponseTemplate, model_id: str, defaults,
                 pieces=None, pre: Preprocessor | None = None):
        self.eng, self.tok, self.rt, self.model_id = eng, tok, rt, model_id
        self.defaults = defaults
        self.pieces = pieces
        self.pre = pre
        self.eos = chatlib.eos_ids(tok, model_id)
        self.lock = threading.Lock()
        self.sampling_defaults = chatlib.default_sampling(model_id)
        self.image_id = eng.image_token
        self.video_id = eng.video_token
        self.stats = {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0,
                      "spec_drafted": 0, "spec_accepted": 0, "spec_rounds": 0,
                      "prefill_reused": 0, "errors": 0}
        # Prefix reuse across turns is only sound while the media bound to
        # those positions is the same: two different images tokenize to the
        # same placeholders, so a cache hit on the text would silently serve
        # the previous picture.
        self._media_key = None

    # ------------------------------------------------------------- prompting
    @staticmethod
    def _append_user_instruction(messages: list, instr: str) -> list:
        out = [dict(m) for m in messages]
        for m in reversed(out):
            if m.get("role") == "user":
                c = m.get("content")
                if isinstance(c, list):
                    m["content"] = c + [{"type": "text", "text": "\n\n" + instr}]
                else:
                    m["content"] = (str(c) if c else "") + "\n\n" + instr
                return out
        out.append({"role": "user", "content": instr})
        return out

    def inject_format(self, messages: list, params: GenParams) -> list:
        """Tell the model what the grammar is going to make it produce.

        The mask alone is not enough, and the failure is instructive: asked
        "Tell me about Lisbon" under a json_schema constraint, the model wants
        to write prose, every prose token is masked, and the highest-scoring
        LEGAL token is a space — so it emits spaces until max_tokens. A
        constraint can only remove options; it cannot tell the model what the
        caller wanted. So the schema goes in the prompt too, and the grammar
        stays as the guarantee rather than the instruction."""
        if params.json_schema is not None:
            return self._append_user_instruction(
                messages,
                "Respond with a single JSON object that strictly conforms to this JSON "
                "Schema:\n" + json.dumps(params.json_schema) +
                "\nOutput ONLY the JSON object — no prose, no markdown fences.")
        if params.json_mode:
            return self._append_user_instruction(
                messages, "Respond with a single JSON object and nothing else — no prose, "
                          "no markdown fences.")
        return messages

    def build(self, messages: list, tools, reasoning: str):
        msgs, media = split_parts(messages, self.pre)
        text = chatlib.render(self.tok, msgs, tools=tools, reasoning=reasoning)
        if media:
            text = expand_placeholders(text, media, self.eng.merge_unit)
        ids = chatlib.encode(self.tok, text)
        return ids, media, text

    def bind_media(self, ids: list[int], media: list):
        """Run the tower and pin its rows to the placeholder positions."""
        if not media:
            if self._media_key is not None:
                self.eng.set_vision_embeds(None, None)
                self._media_key = None
            return
        import numpy as np
        key = tuple((m.kind, m.grid, float(m.pixels.sum())) for m in media)
        positions = [i for i, t in enumerate(ids) if t in (self.image_id, self.video_id)]
        want = sum(m.tokens(self.eng.merge_unit) for m in media)
        if len(positions) != want:
            raise ApiError(f"prompt has {len(positions)} media placeholders but the tower "
                           f"produces {want} rows")
        if key == self._media_key:
            return                      # same pictures, already bound
        pixels = np.concatenate([m.pixels for m in media], axis=0)
        grids = [m.grid for m in media]
        feats = self.eng.vision_features(pixels, grids)
        self.eng.set_vision_embeds(feats, positions)
        # Text-only reuse of a cache that has images pinned into it is fine
        # (the positions are part of the prompt), but a DIFFERENT image at the
        # same positions is not, so the key gates the reuse.
        self._media_key = key

    def params(self, body: dict, tools) -> GenParams:
        d, sd = self.defaults, self.sampling_defaults
        temp = body.get("temperature")
        sample = SampleParams(
            temperature=float(sd["temperature"] if temp is None else temp),
            top_k=int(body.get("top_k", d.top_k if d.top_k is not None else sd["top_k"])),
            top_p=float(body.get("top_p", sd["top_p"])),
            min_p=float(body.get("min_p", 0.0)),
            repeat_penalty=float(body.get("repetition_penalty", 1.0)),
            frequency_penalty=float(body.get("frequency_penalty", 0.0) or 0.0),
            presence_penalty=float(body.get("presence_penalty", 0.0) or 0.0),
            seed=body.get("seed", d.seed),
            logit_bias={int(k): float(v) for k, v in (body.get("logit_bias") or {}).items()},
            top_logprobs=int(body.get("top_logprobs") or 0) if body.get("logprobs") else 0)
        if d.greedy:
            sample.temperature = 0.0
        stop = body.get("stop") or []
        if isinstance(stop, str):
            stop = [stop]
        rf = body.get("response_format") or {}
        schema = None
        if rf.get("type") == "json_schema":
            schema = ((rf.get("json_schema") or {}).get("schema")
                      or rf.get("schema") or {})
        try:
            recipients = recipients_for(body.get("tool_choice"), tools)
        except ValueError as e:
            raise ApiError(str(e))
        return GenParams(
            max_tokens=int(body.get("max_tokens") or body.get("max_completion_tokens")
                           or d.max_tokens),
            stop=list(stop), sample=sample,
            json_mode=rf.get("type") == "json_object",
            json_schema=schema,
            recipients=recipients,
            # Per-request escape hatch, not an OpenAI field: the parity gate
            # needs the same request answered with and without the drafter,
            # and "restart the server" is not a check anyone runs.
            speculative=bool(body.get("speculative", not d.no_spec)))

    # ------------------------------------------------------------------ run
    def stream(self, messages, body):
        """Yield generation events. Holds the lock for the whole request."""
        tools = chatlib.sanitize_tools(body.get("tools"))
        reasoning = reasoning_of(body, self.defaults)
        params = self.params(body, tools)
        messages = self.inject_format(messages, params)
        with self.lock:
            if getattr(self.defaults, "no_prefix_reuse", False):
                # Strict-reproducibility mode. Reuse is not WRONG (a reused
                # cache and a cold one produce the same tokens, gated), but it
                # changes the prefill chunking, and the fast attention tiers
                # are envelope-level rather than bitwise — so at a near-tie the
                # argmax can land differently depending on how much of the
                # prompt was already resident. Anyone who needs "the same
                # request always returns the same bytes" wants this off.
                self.eng.reset()
            ids, media, text = self.build(messages, tools, reasoning)
            if len(ids) + params.max_tokens > self.eng.max_seq:
                raise ApiError(
                    f"prompt is {len(ids)} tokens and max_tokens is {params.max_tokens}, "
                    f"over this server's --max-seq {self.eng.max_seq}", status=400,
                    kind="context_length_exceeded")
            if media:
                # A cache built for other pictures cannot be reused even when
                # the tokens match; drop it rather than serve the wrong image.
                # This has to happen BEFORE the tower runs: reset() clears the
                # bound features as well as the cache, so doing it after was a
                # request whose images had been computed and then thrown away —
                # the model saw random embeddings and reported, reasonably,
                # that no image had been provided.
                self.eng.reset()
                self._media_key = None
            self.bind_media(ids, media)
            self.stats["requests"] += 1
            trace = getattr(self.defaults, "trace_dir", None)
            if trace:
                _trace(trace, {"prompt": text, "ids": ids, "body": body})
            gen = Generation(self.eng, self.tok, self.rt, params,
                             pieces=self.pieces, eos=self.eos)
            for ev in gen.run(ids):
                if ev["type"] == "done":
                    self._account(ev)
                yield ev

    def _account(self, ev) -> None:
        u = ev.get("usage") or {}
        self.stats["prompt_tokens"] += u.get("prompt_tokens", 0)
        self.stats["completion_tokens"] += u.get("completion_tokens", 0)
        self.stats["prefill_reused"] += u.get("prefill_reused", 0)
        s = ev.get("spec") or {}
        for k in ("spec_drafted", "spec_accepted", "spec_rounds"):
            self.stats[k] += s.get(k, 0)


def reasoning_of(body: dict, defaults) -> str:
    """`Reasoning strength` for this request.

    --no-thinking sets `low` rather than removing the line: the template always
    emits one, so "off" is a strength, not an absence."""
    r = body.get("reasoning_effort") or body.get("reasoning_strength")
    if isinstance(body.get("reasoning"), dict):
        r = body["reasoning"].get("effort") or r
    if r in chatlib.REASONING_LEVELS:
        return r
    if r in ("minimal", "none"):
        return "low"
    return "low" if getattr(defaults, "no_thinking", False) else defaults.reasoning


def _trace(dirname: str, payload: dict) -> None:
    p = Path(dirname)
    p.mkdir(parents=True, exist_ok=True)
    (p / f"{now()}-{uuid.uuid4().hex[:8]}.json").write_text(json.dumps(payload, indent=1))


# ------------------------------------------------------------ OpenAI wire form

def _collect(events):
    out = {"content": "", "reasoning": "", "tool_calls": [], "finish_reason": "stop",
           "usage": {}, "logprobs": [], "spec": None}
    for ev in events:
        t = ev["type"]
        if t == "content":
            out["content"] += ev["text"]
        elif t == "reasoning":
            out["reasoning"] += ev["text"]
        elif t == "tool_calls":
            out["tool_calls"].extend(ev["calls"])
        elif t == "done":
            out.update({k: ev[k] for k in ("finish_reason", "usage", "logprobs", "spec")})
    if out["tool_calls"] and out["finish_reason"] == "stop":
        out["finish_reason"] = "tool_calls"
    return out


def _usage(u: dict, spec) -> dict:
    d = {"prompt_tokens": u.get("prompt_tokens", 0),
         "completion_tokens": u.get("completion_tokens", 0),
         "total_tokens": u.get("total_tokens", 0),
         "prefill_s": round(u.get("prefill_s", 0.0), 4),
         "decode_s": round(u.get("decode_s", 0.0), 4),
         "tokens_per_second": round(u.get("tokens_per_second", 0.0), 2),
         "prefill_reused": u.get("prefill_reused", 0)}
    if spec:
        d.update(spec)
    return d


def _message(c: dict, model: str) -> dict:
    msg = {"role": "assistant", "content": c["content"] or None}
    if c["reasoning"]:
        msg["reasoning_content"] = c["reasoning"]
    if c["tool_calls"]:
        msg["tool_calls"] = [{"id": rid("call"), **tc} for tc in c["tool_calls"]]
    return {"id": rid("chatcmpl"), "object": "chat.completion", "created": now(),
            "model": model,
            "choices": [{"index": 0, "message": msg, "finish_reason": c["finish_reason"],
                         "logprobs": _logprobs(c["logprobs"])}],
            "usage": _usage(c["usage"], c["spec"])}


def _logprobs(lps):
    if not lps:
        return None
    return {"content": [{"token": str(x.token_id), "logprob": x.logprob,
                         "top_logprobs": [{"token": str(i), "logprob": v} for i, v in x.top]}
                        for x in lps]}


def _chunk(model: str, ident: str, delta: dict, finish=None, usage=None) -> str:
    body = {"id": ident, "object": "chat.completion.chunk", "created": now(), "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
    if usage is not None:
        body["usage"] = usage
    return "data: " + json.dumps(body) + "\n\n"


def chat_sse(runner: Runner, messages, body):
    model, ident = runner.model_id, rid("chatcmpl")
    yield _chunk(model, ident, {"role": "assistant", "content": ""})
    idx = 0
    finish, usage, spec = "stop", {}, None
    calls = []
    for ev in runner.stream(messages, body):
        t = ev["type"]
        if t == "content":
            yield _chunk(model, ident, {"content": ev["text"]})
        elif t == "reasoning":
            yield _chunk(model, ident, {"reasoning_content": ev["text"]})
        elif t == "tool_calls":
            for tc in ev["calls"]:
                calls.append(tc)
                yield _chunk(model, ident, {"tool_calls": [
                    {"index": idx, "id": rid("call"), "type": "function",
                     "function": tc["function"]}]})
                idx += 1
        elif t == "done":
            finish, usage, spec = ev["finish_reason"], ev["usage"], ev["spec"]
    if calls and finish == "stop":
        finish = "tool_calls"
    yield _chunk(model, ident, {}, finish=finish, usage=_usage(usage, spec))
    yield "data: [DONE]\n\n"


# ----------------------------------------------------------------------- app

def create_app(runner: Runner, defaults):
    app = FastAPI(title="muse-glimmer-intel-serve")

    if defaults.cors_origins:
        from fastapi.middleware.cors import CORSMiddleware
        app.add_middleware(CORSMiddleware, allow_origins=defaults.cors_origins.split(","),
                           allow_methods=["*"], allow_headers=["*"])

    def auth(authorization: str | None, x_api_key: str | None) -> None:
        if not defaults.api_key:
            return
        given = None
        if authorization and authorization.lower().startswith("bearer "):
            given = authorization[7:]
        given = given or x_api_key
        if given != defaults.api_key:
            raise ApiError("invalid API key", status=401, kind="authentication_error")

    @app.exception_handler(ApiError)
    async def _api_error(_req, exc: ApiError):
        runner.stats["errors"] += 1
        return JSONResponse(status_code=exc.status,
                            content={"error": {"message": exc.message, "type": exc.kind}})

    @app.exception_handler(MediaError)
    async def _media_error(_req, exc: MediaError):
        runner.stats["errors"] += 1
        return JSONResponse(status_code=400,
                            content={"error": {"message": str(exc), "type": "invalid_request_error"}})

    @app.exception_handler(chatlib.ToolShapeError)
    async def _tool_error(_req, exc):
        runner.stats["errors"] += 1
        return JSONResponse(status_code=400,
                            content={"error": {"message": str(exc), "type": "invalid_request_error"}})

    async def _run_blocking(gen_factory, request: Request):
        """Run the (blocking) generator on a worker thread and forward its
        events, dropping the request if the client goes away — a cancelled
        agent turn should stop costing GPU time immediately."""
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue(maxsize=64)
        stop = threading.Event()

        def pump():
            try:
                for item in gen_factory():
                    if stop.is_set():
                        break
                    asyncio.run_coroutine_threadsafe(queue.put(item), loop).result()
            except Exception as e:                                # noqa: BLE001
                asyncio.run_coroutine_threadsafe(queue.put(e), loop).result()
            finally:
                asyncio.run_coroutine_threadsafe(queue.put(None), loop).result()

        threading.Thread(target=pump, daemon=True).start()
        try:
            while True:
                if await request.is_disconnected():
                    stop.set()
                    break
                item = await queue.get()
                if item is None:
                    break
                if isinstance(item, Exception):
                    raise item
                yield item
        finally:
            stop.set()

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request, authorization: str | None = Header(None),
                               x_api_key: str | None = Header(None)):
        auth(authorization, x_api_key)
        body = await request.json()
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ApiError("messages must be a non-empty array")
        if body.get("n") not in (None, 1):
            raise ApiError("n > 1 is not supported: one engine, one cache, one completion")
        if body.get("stream"):
            async def sse():
                async for piece in _run_blocking(lambda: chat_sse(runner, messages, body),
                                                 request):
                    yield piece
            return StreamingResponse(sse(), media_type="text/event-stream")
        out = _collect(runner.stream(messages, body))
        return JSONResponse(_message(out, runner.model_id))

    @app.post("/v1/completions")
    async def completions(request: Request, authorization: str | None = Header(None),
                          x_api_key: str | None = Header(None)):
        """Raw text completion: the prompt is used VERBATIM, with no chat
        template and no generation prompt. Useful for evaluation harnesses and
        for reproducing a rendered prompt exactly."""
        auth(authorization, x_api_key)
        body = await request.json()
        prompt = body.get("prompt")
        if not isinstance(prompt, str):
            raise ApiError("prompt must be a string")
        tools = None
        params = runner.params(body, tools)
        ids = chatlib.encode(runner.tok, prompt)
        with runner.lock:
            if getattr(runner.defaults, "no_prefix_reuse", False):
                runner.eng.reset()
            runner.stats["requests"] += 1
            gen = Generation(runner.eng, runner.tok, runner.rt, params,
                             pieces=runner.pieces, eos=runner.eos, raw=True)
            text, done = "", None
            for ev in gen.run(ids):
                if ev["type"] in ("content", "reasoning"):
                    text += ev["text"]
                elif ev["type"] == "done":
                    done = ev
                    runner._account(ev)
        return JSONResponse({
            "id": rid("cmpl"), "object": "text_completion", "created": now(),
            "model": runner.model_id,
            "choices": [{"index": 0, "text": text, "finish_reason": done["finish_reason"]}],
            "usage": _usage(done["usage"], done["spec"])})

    @app.get("/v1/models")
    async def models():
        return {"object": "list",
                "data": [{"id": runner.model_id, "object": "model", "created": now(),
                          "owned_by": "meta-models"}]}

    @app.get("/health")
    async def health():
        free = runner.eng.free_mem()
        return {"status": "ok", "model": runner.model_id,
                "max_seq": runner.eng.max_seq, "chunk": runner.eng.chunk,
                "cache_len": runner.eng.cache_len,
                "speculative": runner.eng.spec_block > 1,
                "prefix_reuse": not getattr(defaults, "no_prefix_reuse", False),
                "vision": runner.eng.has_vision,
                "free_vram_gib": [round(f / 2**30, 3) for f in free]}

    @app.get("/metrics")
    async def metrics():
        s = dict(runner.stats)
        t = runner.eng.timings()
        lines = [f"muse_{k} {v}" for k, v in s.items()]
        lines += [f"muse_engine_{k} {v}" for k, v in t.items()]
        if s["spec_drafted"]:
            lines.append(f"muse_spec_accept_rate {s['spec_accepted'] / s['spec_drafted']:.4f}")
        for i, f in enumerate(runner.eng.free_mem()):
            lines.append(f'muse_free_vram_bytes{{card="{i}"}} {f}')
        return PlainTextResponse("\n".join(lines) + "\n")

    from serve.anthropic_api import mount_anthropic
    from serve.responses_api import mount_responses
    mount_anthropic(app, runner, auth, _run_blocking)
    mount_responses(app, runner, auth, _run_blocking)
    return app


# ---------------------------------------------------------------------- main

def build_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--model", default="meta-models/Muse-Glimmer-30B")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--assistant", default=None,
                    help="DFlash drafter repo/dir; enables speculative decoding")
    ap.add_argument("--gpus", type=int, default=2)
    ap.add_argument("--pin-gpu", default=None, metavar="IDX[,IDX]",
                    help="pin to specific Level-Zero cards, e.g. 0 or 1 or 0,1. Sets "
                         "ONEAPI_DEVICE_SELECTOR and overrides --gpus with the pin count; "
                         "must happen before the runtime enumerates anything")
    ap.add_argument("--shards", type=int, default=0)
    ap.add_argument("--max-seq", type=int, default=None,
                    help="KV allocation ceiling (default: derived from the checkpoint "
                         "size, the card count and the tier)")
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--q8", action="store_true", help="Q8_0 weights for the target")
    ap.add_argument("--q8-assistant", action="store_true", help="Q8_0 weights for the drafter")
    ap.add_argument("--no-flash-prefill", action="store_true")
    ap.add_argument("--no-flash-decode", action="store_true")
    ap.add_argument("--vision", choices=["gpu", "cpu", "off"], default="cpu",
                    help="where the tower runs. cpu keeps ~1.86 GiB/card out of VRAM "
                         "and is the bitwise-gated path; gpu is 20x faster per image")
    ap.add_argument("--max-patches", type=int, default=4096)
    ap.add_argument("--no-prewarm", action="store_true",
                    help="skip the startup prewarm (diagnosis only)")
    ap.add_argument("--seal", type=int, default=2, choices=[0, 1, 2])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--api-key", default=os.environ.get("MUSE_API_KEY"))
    ap.add_argument("--cors-origins", default=None)
    ap.add_argument("--trace-dir", default=None)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--no-spec", action="store_true", help="drafter resident but unused")
    ap.add_argument("--no-prefix-reuse", action="store_true",
                    help="drop the KV cache between requests: slower on follow-up turns, "
                         "but the answer then depends only on the prompt")
    ap.add_argument("--reasoning", choices=list(chatlib.REASONING_LEVELS), default="high")
    ap.add_argument("--no-thinking", action="store_true",
                    help="set Reasoning strength: low (the template always emits a line)")
    ap.add_argument("--verbose", action="store_true")


def apply_gpu_pin(a) -> None:
    """Restrict SYCL to the chosen cards. Must run before any Engine exists:
    the selector is read when the runtime first enumerates devices, and setting
    it afterwards silently does nothing."""
    if not a.pin_gpu:
        return
    idx = [x.strip() for x in str(a.pin_gpu).split(",") if x.strip() != ""]
    os.environ["ONEAPI_DEVICE_SELECTOR"] = "level_zero:" + ",".join(idx)
    a.gpus = len(idx)


def resolve_max_seq(a) -> None:
    if a.max_seq:
        return
    try:
        a.max_seq = chatlib.auto_max_seq(
            a.model, gpus=a.gpus, q8=a.q8, assistant=a.assistant,
            q8_assistant=a.q8_assistant, vision_on_card=(a.vision == "gpu"),
            chunk=a.chunk)
    except ValueError as e:
        raise SystemExit(f"[serve] {e}")
    print(f"[serve] --max-seq {a.max_seq} (derived; pass --max-seq to override)",
          file=sys.stderr)


def make_runner(a) -> Runner:
    tok = chatlib.load_tokenizer(a.model)
    rt = ResponseTemplate.load(a.model)
    place = {"gpu": VISION_GPU, "cpu": VISION_CPU, "off": VISION_OFF}[a.vision]
    eng = Engine(a.model, revision=a.revision, assistant=a.assistant, gpus=a.gpus,
                 shards=a.shards, max_seq=a.max_seq, chunk=a.chunk, q8=a.q8,
                 q8_assistant=a.q8_assistant, flash_prefill=not a.no_flash_prefill,
                 flash_decode=not a.no_flash_decode, vision=place,
                 max_patches=a.max_patches, prewarm=not a.no_prewarm, seal=a.seal,
                 verbose=True)
    pieces = None
    try:
        from serve.grammar import PieceTable
        pieces = PieceTable(eng.L, tok, eng.vocab, chatlib.eos_ids(tok, a.model))
    except Exception as e:                                        # noqa: BLE001
        print(f"[warn] guided decoding unavailable: {e}", file=sys.stderr)
    pre = Preprocessor(chatlib.resolve_dir(a.model)) if place != VISION_OFF else None
    runner = Runner(eng, tok, rt, a.model, a, pieces=pieces, pre=pre)
    free = eng.free_mem()
    if any(free):
        print("[serve] free VRAM after startup: "
              + ", ".join(f"card {i} {f / 2**30:.2f} GiB" for i, f in enumerate(free)),
              file=sys.stderr)
    print(f"[serve] {a.model} | max_seq {a.max_seq} | chunk {a.chunk} | "
          f"{'Q8' if a.q8 else 'BF16'} | vision {a.vision} | "
          f"spec {'on' if eng.spec_block > 1 else 'off'}", file=sys.stderr)
    return runner


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    build_args(ap)
    a = ap.parse_args()
    apply_gpu_pin(a)
    resolve_max_seq(a)
    runner = make_runner(a)
    app = create_app(runner, a)
    import uvicorn
    uvicorn.run(app, host=a.host, port=a.port, log_level="info" if a.verbose else "warning")


if __name__ == "__main__":
    main()
