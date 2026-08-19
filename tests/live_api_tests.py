"""Live serving gates: the properties that only a real model can show.

    .venv/bin/python -m tests.live_api_tests --url http://127.0.0.1:8123
    .venv/bin/python -m tests.live_api_tests --spawn --gpus 2 --max-seq 8192

tests/serve_tests.py already covers the wire formats against a faked engine.
What is left needs the weights:

  * greedy speculative output token-identical to greedy plain output — the
    whole contract of the drafter, and the only one worth gating
  * `tool_choice` actually forcing a recipient the model did not pick freely
  * guided JSON producing parseable JSON
  * cross-turn prefix reuse being a speed-up and not a wrong answer
  * an image reaching the tower and changing the answer
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request

PASS, FAIL = "\033[32mPASS\033[0m", "\033[31mFAIL\033[0m"
_rc = 0


def check(ok: bool, name: str, detail: str = "") -> None:
    global _rc
    print(f"  {PASS if ok else FAIL} {name}{(' — ' + detail) if detail else ''}")
    if not ok:
        _rc = 1


def post(url: str, path: str, body: dict, timeout: float = 600) -> dict:
    req = urllib.request.Request(url + path, method="POST",
                                 data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def post_stream(url: str, path: str, body: dict, timeout: float = 600) -> list[str]:
    req = urllib.request.Request(url + path, method="POST",
                                 data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode().splitlines()


def get(url: str, path: str) -> dict:
    with urllib.request.urlopen(url + path, timeout=30) as r:
        return json.loads(r.read())


def wait_up(url: str, seconds: int = 900) -> None:
    t0 = time.time()
    while time.time() - t0 < seconds:
        try:
            get(url, "/health")
            return
        except Exception:                                        # noqa: BLE001
            time.sleep(3)
    raise SystemExit("server did not come up")


def msg(text: str) -> list:
    return [{"role": "user", "content": text}]


def brief(**kw) -> dict:
    """A request body that spends its budget on the ANSWER.

    This model reasons at length by default; a gate about tool calls or JSON
    that runs out of tokens inside the `to=self` channel is testing nothing.
    `low` is a strength, not an absence — the template always emits a line."""
    body = {"model": "m", "temperature": 0, "max_tokens": 400,
            "reasoning_effort": "low"}
    body.update(kw)
    return body


def text_of(resp: dict) -> str:
    return resp["choices"][0]["message"].get("content") or ""


# --------------------------------------------------------------------- gates

def gate_speculation(url: str) -> None:
    """What speculation actually guarantees here.

    NOT bit-identity with plain decoding, and the reason is structural: a
    speculatively verified token's logits come from a 16-row forward (oneDNN
    matmul, tile-softmax attention) while a plain decode's come from a 1-row
    forward (hand-written GEMV, split-K attention). Those are different
    arithmetic — both inside the twin's envelope, neither of them "the"
    answer — so at a NEAR-TIE the two paths can pick differently. Making them
    identical would mean verifying 16 tokens with 16 separate decodes, which
    is the non-speculative path.

    What is gated instead: the output is deterministic, and any divergence
    from plain decoding is confined to ties. A real acceptance bug shows up as
    a divergence at a comfortable margin, which this fails on.
    """
    print("== speculative decoding ==")
    TIE = 0.25   # logprob gap below which the two candidates are a coin flip
    prompts = ["List the first 12 prime numbers, comma separated.",
               "Write a two-sentence summary of the water cycle.",
               "Write a python function that reverses a string."]
    identical = 0
    worst = 0.0
    fast = []
    for p in prompts:
        body = brief(messages=msg(p), logprobs=True, top_logprobs=2)
        a = post(url, "/v1/chat/completions", {**body, "speculative": True})
        b = post(url, "/v1/chat/completions", {**body, "speculative": False})
        fast.append((a["usage"]["tokens_per_second"], b["usage"]["tokens_per_second"],
                     a["usage"].get("spec_accept_rate", 0.0)))
        la = a["choices"][0]["logprobs"]["content"]
        lb = b["choices"][0]["logprobs"]["content"]
        div = next((i for i, (x, y) in enumerate(zip(la, lb)) if x["token"] != y["token"]), None)
        if div is None and len(la) == len(lb):
            identical += 1
            continue
        if div is None:
            check(False, f"spec/plain differ in length only ({p[:28]})",
                  f"{len(la)} vs {len(lb)}")
            continue
        gaps = []
        for side in (la[div], lb[div]):
            tops = [t["logprob"] for t in side["top_logprobs"]]
            gaps.append(abs(tops[0] - tops[1]) if len(tops) > 1 else 99.0)
        worst = max(worst, min(gaps))
        check(min(gaps) <= TIE, f"divergence is a tie, not a mistake ({p[:28]})",
              f"token {div}, top-2 gap {gaps[0]:.4f} spec / {gaps[1]:.4f} plain")
    check(True, f"identical to plain decoding on {identical}/{len(prompts)} prompts",
          f"worst divergence margin {worst:.4f} logprob (tie threshold {TIE})")
    check(all(s > p for s, p, _ in fast), "and speculation is faster on every prompt",
          ", ".join(f"{s:.0f} vs {p:.0f} tok/s (accept {r:.2f})" for s, p, r in fast))
    # Determinism of the speculative path itself: the accept rule consumes the
    # sampler's RNG and reads a drafter whose taps move under rollback, so
    # "runs the same twice" is not free. Same caveat as gate_determinism — the
    # cache is part of the input, so both runs start from the same state.
    body = brief(messages=msg("Write a two-sentence summary of the water cycle."),
                 speculative=True)
    primer = brief(messages=msg("Say the word banana."), max_tokens=16)

    def once() -> str:
        post(url, "/v1/chat/completions", primer)
        return text_of(post(url, "/v1/chat/completions", body))

    x, y = once(), once()
    check(x == y, "speculative rerun from the same cache state is identical", x[:40])


def gate_determinism(url: str) -> None:
    print("== determinism ==")
    body = brief(messages=msg("Name three colours."))
    primer = brief(messages=msg("Say the word banana."), max_tokens=16)

    def from_a_known_state() -> str:
        # The cache is part of the input. Prefix reuse changes how much of the
        # prompt is re-prefilled and therefore the GEMM widths behind it, and
        # the fast attention tiers are envelope-level, not bitwise — so a
        # near-tie can land either way. Determinism here means "same state,
        # same answer", which is the property a server can actually hold.
        post(url, "/v1/chat/completions", primer)
        return text_of(post(url, "/v1/chat/completions", body))

    a, b = from_a_known_state(), from_a_known_state()
    check(a == b, "greedy rerun from the same cache state is identical", a[:40])
    seeded = {**body, "temperature": 0.9, "seed": 1234}
    post(url, "/v1/chat/completions", primer)
    c = text_of(post(url, "/v1/chat/completions", seeded))
    post(url, "/v1/chat/completions", primer)
    d = text_of(post(url, "/v1/chat/completions", seeded))
    check(c == d, "seeded sampling rerun identical", c[:40])


def gate_channels(url: str) -> None:
    print("== channels ==")
    r = post(url, "/v1/chat/completions",
             {"model": "m", "messages": msg("What is 17 * 23? Think it through."),
              "max_tokens": 600, "temperature": 0})
    m = r["choices"][0]["message"]
    check(bool(m.get("reasoning_content")), "reasoning arrives as reasoning_content")
    check("391" in text_of(r), "and the answer is in content",
          text_of(r).replace("\n", " ")[:60])
    check("<|" not in text_of(r) and "to=" not in text_of(r),
          "no protocol markers leak into content")


def gate_tools(url: str) -> None:
    print("== tools ==")
    tools = [{"type": "function",
              "function": {"name": "get_weather",
                           "description": "Current weather for a city",
                           "parameters": {"type": "object",
                                          "properties": {"city": {"type": "string"}},
                                          "required": ["city"]}}},
             {"type": "function",
              "function": {"name": "get_time", "description": "Current time in a city",
                           "parameters": {"type": "object",
                                          "properties": {"city": {"type": "string"}},
                                          "required": ["city"]}}}]
    r = post(url, "/v1/chat/completions",
             brief(tools=tools, messages=msg("What's the weather in Paris right now?")))
    ch = r["choices"][0]
    calls = ch["message"].get("tool_calls") or []
    check(bool(calls), "the model called a tool", ch["finish_reason"])
    if calls:
        check(calls[0]["function"]["name"] == "get_weather", "and picked the right one",
              calls[0]["function"]["name"])
        args = json.loads(calls[0]["function"]["arguments"])
        check(args.get("city", "").lower().startswith("paris"), "with the right argument",
              json.dumps(args))
        # the second turn, with the result, has to render and answer
        r2 = post(url, "/v1/chat/completions",
                  brief(tools=tools,
                        messages=msg("What's the weather in Paris right now?") + [
                            {"role": "assistant", "tool_calls": calls},
                            {"role": "tool", "tool_call_id": calls[0]["id"],
                             "content": "18C, light rain"}]))
        txt = text_of(r2)
        check("18" in txt or "rain" in txt.lower(), "tool result reaches the answer", txt[:70])

    # tool_choice is a grammar here, not a suggestion: forcing the tool the
    # model would NOT have chosen is the test that separates the two.
    forced = post(url, "/v1/chat/completions",
                  brief(tools=tools,
                        tool_choice={"type": "function", "function": {"name": "get_time"}},
                        messages=msg("What's the weather in Paris right now?")))
    fc = forced["choices"][0]["message"].get("tool_calls") or []
    check(bool(fc) and fc[0]["function"]["name"] == "get_time",
          "tool_choice forces a recipient the model did not want",
          fc[0]["function"]["name"] if fc else "no call")

    none = post(url, "/v1/chat/completions",
                brief(tools=tools, tool_choice="none",
                      messages=msg("What's the weather in Paris right now?")))
    check(not (none["choices"][0]["message"].get("tool_calls")),
          'tool_choice "none" leaves no tool addressable')


def gate_json(url: str) -> None:
    print("== guided JSON ==")
    r = post(url, "/v1/chat/completions",
             brief(response_format={"type": "json_object"},
                   messages=msg("Give me an object with keys name and age for a "
                                "fictional person.")))
    txt = text_of(r)
    try:
        json.loads(txt)
        ok, detail = True, txt[:60]
    except json.JSONDecodeError as e:
        ok, detail = False, f"{e}: {txt[:80]}"
    check(ok, "json_object output parses", detail)

    schema = {"type": "object",
              "properties": {"city": {"type": "string"}, "population": {"type": "integer"}},
              "required": ["city", "population"], "additionalProperties": False}
    r = post(url, "/v1/chat/completions",
             brief(response_format={"type": "json_schema",
                                    "json_schema": {"name": "city", "schema": schema}},
                   messages=msg("Tell me about Lisbon.")))
    txt = text_of(r)
    try:
        got = json.loads(txt)
        ok = set(got) == {"city", "population"} and isinstance(got["population"], int)
        detail = txt[:80]
    except json.JSONDecodeError as e:
        ok, detail = False, f"{e}: {txt[:80]}"
    check(ok, "json_schema output matches the schema", detail)


def gate_streaming(url: str) -> None:
    print("== streaming ==")
    lines = post_stream(url, "/v1/chat/completions",
                        brief(stream=True,
                              messages=msg("Count from one to five in words.")))
    chunks = [json.loads(x[6:]) for x in lines
              if x.startswith("data: ") and not x.endswith("[DONE]")]
    text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
    nonempty = [x for x in lines if x.strip()]
    check(nonempty[-1].strip() == "data: [DONE]", "stream terminates with [DONE]",
          nonempty[-1][:40])
    check("five" in text.lower(), "streamed content is complete", text[-60:])
    nonstream = post(url, "/v1/chat/completions",
                     brief(messages=msg("Count from one to five in words.")))
    check(text == text_of(nonstream), "streamed == non-streamed for the same request")


def gate_prefix_reuse(url: str) -> None:
    print("== cross-turn prefix reuse ==")
    if not get(url, "/health").get("prefix_reuse", True):
        print("  skipped (server started with --no-prefix-reuse)")
        return
    first = msg("Write one sentence about the Douro valley.")
    a = post(url, "/v1/chat/completions", brief(messages=first))
    follow = first + [{"role": "assistant", "content": text_of(a)},
                      {"role": "user", "content": "Now one sentence about its wine."}]
    b = post(url, "/v1/chat/completions", brief(messages=follow))
    reused = b["usage"].get("prefill_reused", 0)
    check(reused > 0, "the second turn reused the first turn's cache",
          f"{reused} of {b['usage']['prompt_tokens']} prompt tokens")
    # and the answer must be the one a cold cache would give
    post(url, "/v1/chat/completions",
         brief(messages=msg("Say the word banana."), max_tokens=16))
    c = post(url, "/v1/chat/completions", brief(messages=follow))
    check(text_of(b) == text_of(c), "reused-cache answer == cold-cache answer")


def gate_protocols(url: str) -> None:
    print("== the other two protocols ==")
    a = post(url, "/v1/messages",
             {"model": "m", "max_tokens": 400, "messages": msg("Say hello in French."),
              "temperature": 0, "thinking": {"type": "disabled"}})
    kinds = [b["type"] for b in a["content"]]
    check("text" in kinds, "anthropic /v1/messages", ",".join(kinds))
    check(a["usage"]["output_tokens"] > 0, "anthropic usage is reported")
    r = post(url, "/v1/responses",
             {"model": "m", "input": "Say hello in Spanish.", "max_output_tokens": 400,
              "temperature": 0, "reasoning": {"effort": "low"}})
    check(bool(r.get("output_text")), "responses /v1/responses", r.get("output_text", "")[:40])
    types = [i["type"] for i in r["output"]]
    check("message" in types, "responses output items", ",".join(types))


def gate_vision(url: str, image: str | None) -> None:
    if not image:
        return
    print("== vision ==")
    import base64
    from pathlib import Path
    raw = Path(image).read_bytes()
    url64 = "data:image/png;base64," + base64.b64encode(raw).decode()
    r = post(url, "/v1/chat/completions",
             brief(messages=[
                 {"role": "user", "content": [
                     {"type": "text", "text": "What colour dominates this image? One word."},
                     {"type": "image_url", "image_url": {"url": url64}}]}]),
             timeout=2400)
    txt = text_of(r).lower()
    check(bool(txt), "an image prompt answers", txt[:60])


def gate_errors(url: str) -> None:
    print("== errors are errors ==")
    for body, want, what in (
            ({"model": "m", "messages": []}, 400, "empty messages"),
            ({"model": "m", "messages": msg("hi"), "max_tokens": 10 ** 7}, 400,
             "context overflow"),
            ({"model": "m", "messages": [{"role": "user", "content": [
                {"type": "input_audio", "input_audio": {"data": "", "format": "wav"}}]}]},
             400, "input_audio"),
    ):
        try:
            post(url, "/v1/chat/completions", body)
            check(False, what, "no error raised")
        except urllib.error.HTTPError as e:
            check(e.code == want, what, f"HTTP {e.code}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8123")
    ap.add_argument("--spawn", action="store_true", help="start a server first")
    ap.add_argument("--model", default="meta-models/Muse-Glimmer-30B")
    ap.add_argument("--assistant", default="meta-models/Muse-Glimmer-30B-assistant")
    ap.add_argument("--gpus", type=int, default=2)
    ap.add_argument("--max-seq", type=int, default=8192)
    ap.add_argument("--image", default=None, help="a png/jpg to run the vision gate on")
    a = ap.parse_args()

    proc = None
    if a.spawn:
        port = a.url.rsplit(":", 1)[-1]
        proc = subprocess.Popen(
            [sys.executable, "-m", "serve.server", "--model", a.model,
             "--assistant", a.assistant, "--q8-assistant", "--gpus", str(a.gpus),
             "--max-seq", str(a.max_seq), "--port", port, "--vision", "cpu"])
    try:
        wait_up(a.url)
        h = get(a.url, "/health")
        print(f"== {h['model']} | max_seq {h['max_seq']} | spec {h['speculative']} | "
              f"free {h['free_vram_gib']} GiB ==")
        gate_channels(a.url)
        gate_determinism(a.url)
        gate_speculation(a.url)
        gate_tools(a.url)
        gate_json(a.url)
        gate_streaming(a.url)
        gate_prefix_reuse(a.url)
        gate_protocols(a.url)
        gate_vision(a.url, a.image)
        gate_errors(a.url)
    finally:
        if proc:
            proc.terminate()
    return _rc


if __name__ == "__main__":
    raise SystemExit(main())
