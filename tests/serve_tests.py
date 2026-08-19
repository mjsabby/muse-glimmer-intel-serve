"""Serving tests that need no GPU: rendering, channel parsing, grammars, the
generation loop, and the three protocol translations.

    .venv/bin/python -m tests.serve_tests            (from the repo root)

The engine is faked. Everything above it — the ATEM channel machine, the
recipient grammar, speculative acceptance, the wire formats — is exercised
against a scripted token stream, so a break in any of it is a test failure
rather than a puzzling response on the box.
"""
from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from serve import chatlib                                       # noqa: E402
from serve.atem import ChannelParser, ResponseTemplate, parse_atem_body  # noqa: E402
from serve.generate import GenParams, Generation, generate      # noqa: E402
from serve.recipient import RecipientGrammar, recipients_for    # noqa: E402
from serve.sampler import SampleParams                          # noqa: E402

MODEL = os.environ.get("MUSE_MODEL", "meta-models/Muse-Glimmer-30B")


def _tokenizer():
    try:
        return chatlib.load_tokenizer(MODEL)
    except Exception as e:                                       # pragma: no cover
        raise unittest.SkipTest(f"tokenizer unavailable: {e}")


class FakeEngine:
    """A deterministic target that "wants" to emit `script`.

    Addressed by CACHE POSITION, not by call count: the distribution after a
    cache of length L is a point mass on script[L - prompt_len]. That is what
    makes it a legitimate stand-in under speculation — a real target re-derives
    the same distribution for the same prefix, so a rejected block that is
    rolled back and re-decoded must produce the same tokens. A fake that
    advanced a cursor per call would pass the accept path and quietly diverge
    on the reject path, which is the one the test is for.
    """

    def __init__(self, vocab: int, script: list[int], *, spec_block: int = 0,
                 proposals: list[list[int]] | None = None):
        self.vocab = vocab
        self.script = list(script)
        self.cache_len = 0
        self.prompt_len = None
        self.spec_block = spec_block
        self.proposals = list(proposals or [])
        self.window = 2048
        self.chunk = 512
        self.cached_ids: list[int] = []
        self.forwards = 0

    def _onehot(self, tid: int) -> np.ndarray:
        v = np.full(self.vocab, -30.0, dtype=np.float32)
        v[tid] = 30.0
        return v

    def _at(self, cache_len: int) -> np.ndarray:
        i = cache_len - self.prompt_len
        return self._onehot(self.script[min(max(i, 0), len(self.script) - 1)])

    def reuse_prefix(self, ids):
        self.cache_len = 0
        self.cached_ids = []
        return 0

    def reset(self):
        self.cache_len = 0

    def forward(self, ids, logits=None):
        self.forwards += 1
        if self.prompt_len is None:
            self.prompt_len = len(ids)
        self.cache_len += len(ids)
        return self._at(self.cache_len)

    def forward_all(self, ids):
        self.forwards += 1
        base = self.cache_len
        self.cache_len += len(ids)
        # Row i is the distribution at the position after block[i].
        return np.stack([self._at(base + i + 1) for i in range(len(ids))])

    def draft(self, anchor):
        return self.proposals.pop(0) if self.proposals else []

    def rollback(self, n):
        self.cache_len = n

    # -- the rest of the surface serve.server.Runner touches
    max_seq = 8192
    has_vision = False
    image_token = 200092
    video_token = 200091
    merge_unit = 4
    L = None

    def free_mem(self):
        return [0, 0]

    def timings(self):
        return {"upload_s": 0.0, "prefill_s": 0.0, "decode_s": 0.0,
                "prefill_tokens": 0, "decode_tokens": 0}

    def set_vision_embeds(self, feats, positions):
        pass


class TestRendering(unittest.TestCase):
    def setUp(self):
        self.tok = _tokenizer()

    def test_generation_prompt_is_a_bare_assistant_start(self):
        text = chatlib.render(self.tok, [{"role": "user", "content": "Hi"}])
        self.assertTrue(text.endswith("<|start|>assistant"), text[-40:])
        # ...and the model, not the template, writes the recipient. If this
        # ever changes, serve/recipient.py's whole premise goes with it.
        self.assertNotIn("<|start|>assistant to=", text)

    def test_reasoning_strength_is_always_emitted(self):
        for level in ("low", "high", "xhigh"):
            text = chatlib.render(self.tok, [{"role": "user", "content": "Hi"}],
                                  reasoning=level)
            self.assertIn(f"Reasoning strength: {level}.", text)

    def test_system_message_keeps_its_own_reasoning_line(self):
        text = chatlib.render(
            self.tok, [{"role": "system", "content": "Be terse.\n\nReasoning effort: low."},
                       {"role": "user", "content": "Hi"}], reasoning="high")
        # "effort" is normalized to "strength" and the kwarg line is suppressed
        self.assertIn("Reasoning strength: low.", text)
        self.assertNotIn("Reasoning strength: high.", text)

    def test_tool_arguments_must_be_deserialized(self):
        """The template raise_exception()s on a JSON *string*, which is what
        every OpenAI client sends back on the second turn."""
        msgs = [{"role": "user", "content": "weather?"},
                {"role": "assistant", "tool_calls": [
                    {"id": "call_1", "type": "function",
                     "function": {"name": "get_weather",
                                  "arguments": '{"city": "Paris"}'}}]},
                {"role": "tool", "tool_call_id": "call_1", "content": "18C"}]
        with self.assertRaises(Exception):
            self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        text = chatlib.render(self.tok, msgs)              # normalizes first
        self.assertIn('<atem:parameter name="city">Paris</atem:parameter>', text)

    def test_tools_render_into_the_recipient_list(self):
        tools = chatlib.sanitize_tools([
            {"type": "function", "function": {"name": "web.search",
                                              "description": "search",
                                              "parameters": {"type": "object",
                                                             "properties": {"q": {}}}}}])
        text = chatlib.render(self.tok, [{"role": "user", "content": "hi"}], tools=tools)
        self.assertIn('# Valid recipients: "self", "web.*", "user".', text)
        # a property with no `type` is repaired rather than passed through
        self.assertIn('"q": {"type": "string"}', text)

    def test_eos_ids_exclude_eom(self):
        eos = chatlib.eos_ids(self.tok, MODEL)
        self.assertIn(self.tok.convert_tokens_to_ids("<|eot|>"), eos)
        self.assertIn(self.tok.convert_tokens_to_ids("<|end_of_text|>"), eos)
        self.assertNotIn(self.tok.convert_tokens_to_ids("<|eom|>"), eos,
                         "stopping on <|eom|> truncates the turn to its reasoning")


class TestChannels(unittest.TestCase):
    def setUp(self):
        self.tok = _tokenizer()
        self.rt = ResponseTemplate.load(MODEL)

    def _run(self, text: str):
        ids = self.tok(text, add_special_tokens=False)["input_ids"]
        p = ChannelParser(self.tok, self.rt)
        out = {"reasoning": "", "content": "", "calls": []}
        for i in ids:
            for ev in p.push(i):
                if ev.kind == "reasoning":
                    out["reasoning"] += ev.text
                elif ev.kind == "content":
                    out["content"] += ev.text
                elif ev.kind == "tool_calls":
                    out["calls"].extend(ev.calls)
        for ev in p.flush():
            if ev.kind == "reasoning":
                out["reasoning"] += ev.text
            elif ev.kind == "content":
                out["content"] += ev.text
            elif ev.kind == "tool_calls":
                out["calls"].extend(ev.calls)
        return out

    def test_reasoning_never_leaks_into_content(self):
        got = self._run(" to=self<|message|>hmm, Paris.<|eom|>"
                        "<|start|>assistant to=user<|message|>It is 18C.<|eot|>")
        self.assertEqual(got["reasoning"], "hmm, Paris.")
        self.assertEqual(got["content"], "It is 18C.")

    def test_answer_only(self):
        got = self._run(" to=user<|message|>Hello.<|eot|>")
        self.assertEqual(got["content"], "Hello.")
        self.assertEqual(got["reasoning"], "")

    def test_tool_call(self):
        got = self._run(' to=web.search<|message|><atem:function_calls>\n'
                        '<atem:invoke name="web.search">\n'
                        '<atem:parameter name="q">rain in Paris</atem:parameter>\n'
                        '<atem:parameter name="n">3</atem:parameter>\n'
                        '<atem:parameter name="filters">{"lang": "en"}</atem:parameter>\n'
                        '</atem:invoke>\n</atem:function_calls><|eot|>')
        self.assertEqual(len(got["calls"]), 1)
        fn = got["calls"][0]["function"]
        self.assertEqual(fn["name"], "web.search")
        args = json.loads(fn["arguments"])
        # allow_non_json: a bare scalar stays text, a JSON container parses
        self.assertEqual(args["q"], "rain in Paris")
        self.assertEqual(args["n"], 3)
        self.assertEqual(args["filters"], {"lang": "en"})
        self.assertEqual(got["content"], "")

    def test_two_tool_calls_in_one_channel(self):
        got = self._run(' to=t<|message|><atem:function_calls>\n'
                        '<atem:invoke name="a">\n<atem:parameter name="x">1</atem:parameter>\n'
                        '</atem:invoke>\n'
                        '<atem:invoke name="b">\n<atem:parameter name="y">2</atem:parameter>\n'
                        '</atem:invoke>\n</atem:function_calls><|eot|>')
        self.assertEqual([c["function"]["name"] for c in got["calls"]], ["a", "b"])

    def test_truncated_turn_still_emits(self):
        got = self._run(" to=user<|message|>half a sen")
        self.assertEqual(got["content"], "half a sen")

    def test_template_shape_is_validated(self):
        rt = ResponseTemplate.load(MODEL)
        with self.assertRaises(ValueError):
            rt.validate({"content": {"open_pattern": "to=somebody", "close": ["<|eot|>"]},
                         "reasoning_content": {"open_pattern": "to=self", "close": "<|eom|>"}})


class TestRecipientGrammar(unittest.TestCase):
    def setUp(self):
        self.tok = _tokenizer()
        self.vocab = 202048

    def _spell(self, g: RecipientGrammar, text: str) -> bool:
        """Feed a recipient header token by token; True if every step was
        legal under the mask."""
        ids = self.tok(text, add_special_tokens=False)["input_ids"]
        for i in ids:
            m = g.mask(self.vocab)
            if m is not None and not m[i]:
                return False
            g.accept(i)
        return True

    def test_forced_tool_name_is_the_only_legal_recipient(self):
        g = RecipientGrammar(self.tok, self.vocab, ["self", "web.search"])
        self.assertTrue(self._spell(g, " to=web.search<|message|>"))
        g2 = RecipientGrammar(self.tok, self.vocab, ["self", "web.search"])
        self.assertFalse(self._spell(g2, " to=user<|message|>"))

    def test_none_allows_the_answer_and_no_tool(self):
        g = RecipientGrammar(self.tok, self.vocab, ["self", "user"], allow_self_once=False)
        self.assertTrue(self._spell(g, " to=user<|message|>"))
        g2 = RecipientGrammar(self.tok, self.vocab, ["self", "user"], allow_self_once=False)
        self.assertFalse(self._spell(g2, " to=web.search<|message|>"))

    def test_self_is_available_once_then_withdrawn(self):
        """`required` must not let the turn reason forever instead of calling."""
        g = RecipientGrammar(self.tok, self.vocab, ["self", "t"], allow_self_once=True)
        self.assertTrue(self._spell(g, " to=self<|message|>"))
        g.accept(self.tok.convert_tokens_to_ids("<|eom|>"))
        g.accept(self.tok.convert_tokens_to_ids("<|start|>"))
        self.assertFalse(self._spell(g, " to=self<|message|>"))

    def test_mask_is_none_inside_the_body(self):
        g = RecipientGrammar(self.tok, self.vocab, ["self", "user"])
        self._spell(g, " to=user<|message|>")
        self.assertIsNone(g.mask(self.vocab), "the body must not pay for the header")

    def test_tool_choice_mapping(self):
        tools = chatlib.sanitize_tools([{"type": "function",
                                         "function": {"name": "a", "parameters": {}}}])
        self.assertIsNone(recipients_for("auto", tools))
        self.assertEqual(recipients_for("none", tools), (["self", "user"], False))
        self.assertEqual(recipients_for("required", tools), (["self", "a"], True))
        self.assertEqual(recipients_for({"function": {"name": "a"}}, tools), (["self", "a"], True))
        with self.assertRaises(ValueError):
            recipients_for({"function": {"name": "nope"}}, tools)
        with self.assertRaises(ValueError):
            recipients_for("required", [])


class TestGenerationLoop(unittest.TestCase):
    def setUp(self):
        self.tok = _tokenizer()
        self.rt = ResponseTemplate.load(MODEL)
        self.eos = chatlib.eos_ids(self.tok, MODEL)

    def _ids(self, text):
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def _greedy(self):
        return GenParams(max_tokens=64, sample=SampleParams(temperature=0.0),
                         speculative=False)

    def test_plain_turn(self):
        script = self._ids(" to=user<|message|>Hello there.<|eot|>")
        eng = FakeEngine(202048, script)
        out = generate(eng, self.tok, self.rt, [1, 2, 3], self._greedy(), eos=self.eos)
        self.assertEqual(out["content"], "Hello there.")
        self.assertEqual(out["finish_reason"], "stop")
        self.assertEqual(out["usage"]["completion_tokens"], len(script))

    def test_max_tokens_is_length(self):
        script = self._ids(" to=user<|message|>" + "word " * 50 + "<|eot|>")
        eng = FakeEngine(202048, script)
        p = self._greedy()
        p.max_tokens = 12
        out = generate(eng, self.tok, self.rt, [1], p, eos=self.eos)
        self.assertEqual(out["finish_reason"], "length")
        self.assertEqual(out["usage"]["completion_tokens"], 12)

    def test_stop_string_truncates_content(self):
        script = self._ids(" to=user<|message|>alpha STOP beta<|eot|>")
        eng = FakeEngine(202048, script)
        p = self._greedy()
        p.stop = ["STOP"]
        out = generate(eng, self.tok, self.rt, [1], p, eos=self.eos)
        self.assertTrue(out["content"].startswith("alpha"))
        self.assertNotIn("beta", out["content"])
        self.assertEqual(out["finish_reason"], "stop")

    def test_tool_call_sets_the_finish_reason(self):
        script = self._ids(' to=x<|message|><atem:function_calls>\n<atem:invoke name="x">\n'
                           '<atem:parameter name="k">v</atem:parameter>\n</atem:invoke>\n'
                           '</atem:function_calls><|eot|>')
        eng = FakeEngine(202048, script)
        out = generate(eng, self.tok, self.rt, [1], self._greedy(), eos=self.eos)
        self.assertEqual(out["finish_reason"], "tool_calls")
        self.assertEqual(out["tool_calls"][0]["function"]["name"], "x")


class TestSpeculativeAcceptance(unittest.TestCase):
    """Greedy speculative output must be token-identical to greedy plain
    output. That is the whole contract of speculation, and the one thing a
    faster path is not allowed to trade away."""

    def setUp(self):
        self.tok = _tokenizer()
        self.rt = ResponseTemplate.load(MODEL)
        self.eos = chatlib.eos_ids(self.tok, MODEL)
        self.script = self.tok(" to=user<|message|>The capital of France is Paris.<|eot|>",
                               add_special_tokens=False)["input_ids"]

    def _run(self, *, proposals):
        eng = FakeEngine(202048, self.script, spec_block=16, proposals=proposals)
        p = GenParams(max_tokens=64, sample=SampleParams(temperature=0.0), speculative=True)
        return generate(eng, self.tok, self.rt, [1], p, eos=self.eos)

    def test_perfect_draft_is_identical(self):
        # The drafter proposes exactly what the target would have produced.
        rest = self.script[1:]
        prop = [rest[:4]] if len(rest) >= 4 else []
        got = self._run(proposals=prop)
        plain = generate(FakeEngine(202048, self.script), self.tok, self.rt, [1],
                         GenParams(max_tokens=64, sample=SampleParams(temperature=0.0),
                                   speculative=False), eos=self.eos)
        self.assertEqual(got["content"], plain["content"])
        self.assertEqual(got["spec"]["spec_accepted"], 4)

    def test_wrong_draft_is_still_identical(self):
        got = self._run(proposals=[[999, 998, 997]])
        plain = generate(FakeEngine(202048, self.script), self.tok, self.rt, [1],
                         GenParams(max_tokens=64, sample=SampleParams(temperature=0.0),
                                   speculative=False), eos=self.eos)
        self.assertEqual(got["content"], plain["content"])
        self.assertEqual(got["spec"]["spec_accepted"], 0)


class TestAtemParsing(unittest.TestCase):
    def setUp(self):
        self.rt = ResponseTemplate.load(MODEL)

    def test_multiline_values_and_quotes(self):
        calls = parse_atem_body(
            '<atem:invoke name="edit">\n'
            '<atem:parameter name="text">line one\nline "two"\n</atem:parameter>\n'
            '</atem:invoke>', self.rt)
        args = json.loads(calls[0]["function"]["arguments"])
        self.assertEqual(args["text"], 'line one\nline "two"')

    def test_no_call_is_no_call(self):
        self.assertEqual(parse_atem_body("just prose", self.rt), [])


class TestProtocols(unittest.TestCase):
    """The three wire formats, end to end over a faked engine.

    A protocol bug looks exactly like a model bug from the client side, and
    these are the requests real agent harnesses actually send: a tool result
    coming back on the second turn, `tool_choice` forcing a name, an image
    part, a stream that has to be parsed incrementally."""

    @classmethod
    def setUpClass(cls):
        try:
            from fastapi.testclient import TestClient
        except Exception as e:                                   # pragma: no cover
            raise unittest.SkipTest(f"fastapi unavailable: {e}")
        cls.TestClient = TestClient
        cls.tok = _tokenizer()

    def _client(self, script_text: str, **over):
        from types import SimpleNamespace

        from serve.atem import ResponseTemplate
        from serve.server import Runner, create_app

        script = self.tok(script_text, add_special_tokens=False)["input_ids"]
        eng = FakeEngine(202048, script)
        opts = dict(max_tokens=64, top_k=None, seed=None, greedy=True, no_spec=True,
                    reasoning="high", no_thinking=False, api_key=None, cors_origins=None,
                    trace_dir=None)
        opts.update(over)
        defaults = SimpleNamespace(**opts)
        runner = Runner(eng, self.tok, ResponseTemplate.load(MODEL), MODEL, defaults)
        runner.pre = None
        return self.TestClient(create_app(runner, defaults)), runner

    # ---------------------------------------------------------- chat completions
    def test_chat_completion_splits_the_channels(self):
        c, _ = self._client(" to=self<|message|>thinking<|eom|>"
                            "<|start|>assistant to=user<|message|>Answer.<|eot|>")
        r = c.post("/v1/chat/completions",
                   json={"model": MODEL, "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(r.status_code, 200, r.text)
        m = r.json()["choices"][0]["message"]
        self.assertEqual(m["content"], "Answer.")
        self.assertEqual(m["reasoning_content"], "thinking")
        self.assertEqual(r.json()["choices"][0]["finish_reason"], "stop")

    def test_chat_streaming(self):
        c, _ = self._client(" to=user<|message|>Hello world.<|eot|>")
        with c.stream("POST", "/v1/chat/completions",
                      json={"model": MODEL, "stream": True,
                            "messages": [{"role": "user", "content": "hi"}]}) as r:
            self.assertEqual(r.status_code, 200)
            body = "".join(r.iter_text())
        chunks = [json.loads(l[6:]) for l in body.splitlines()
                  if l.startswith("data: ") and not l.endswith("[DONE]")]
        text = "".join(ch["choices"][0]["delta"].get("content", "") for ch in chunks)
        self.assertEqual(text, "Hello world.")
        self.assertTrue(body.rstrip().endswith("data: [DONE]"))
        self.assertIn("usage", chunks[-1])

    def test_chat_tool_call_round_trip(self):
        c, _ = self._client(' to=get_weather<|message|><atem:function_calls>\n'
                            '<atem:invoke name="get_weather">\n'
                            '<atem:parameter name="city">Paris</atem:parameter>\n'
                            '</atem:invoke>\n</atem:function_calls><|eot|>')
        tools = [{"type": "function",
                  "function": {"name": "get_weather", "description": "w",
                               "parameters": {"type": "object",
                                              "properties": {"city": {"type": "string"}}}}}]
        r = c.post("/v1/chat/completions",
                   json={"model": MODEL, "tools": tools,
                         "messages": [{"role": "user", "content": "weather in Paris?"}]})
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertEqual(body["choices"][0]["finish_reason"], "tool_calls")
        call = body["choices"][0]["message"]["tool_calls"][0]
        self.assertEqual(call["function"]["name"], "get_weather")
        self.assertEqual(json.loads(call["function"]["arguments"]), {"city": "Paris"})
        # ...and the client's next turn, with the result, must render
        r2 = c.post("/v1/chat/completions", json={
            "model": MODEL, "tools": tools, "messages": [
                {"role": "user", "content": "weather in Paris?"},
                {"role": "assistant", "tool_calls": [call]},
                {"role": "tool", "tool_call_id": call["id"], "content": "18C"}]})
        self.assertEqual(r2.status_code, 200, r2.text)

    def test_context_overflow_is_a_400_that_names_the_flag(self):
        c, _ = self._client(" to=user<|message|>x<|eot|>")
        r = c.post("/v1/chat/completions",
                   json={"model": MODEL, "max_tokens": 99999,
                         "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(r.status_code, 400)
        self.assertIn("--max-seq", r.json()["error"]["message"])

    def test_api_key_is_enforced(self):
        c, _ = self._client(" to=user<|message|>x<|eot|>", api_key="secret")
        body = {"model": MODEL, "messages": [{"role": "user", "content": "hi"}]}
        self.assertEqual(c.post("/v1/chat/completions", json=body).status_code, 401)
        self.assertEqual(c.post("/v1/chat/completions", json=body,
                                headers={"Authorization": "Bearer secret"}).status_code, 200)

    def test_audio_is_a_capability_error(self):
        c, _ = self._client(" to=user<|message|>x<|eot|>")
        r = c.post("/v1/chat/completions", json={"model": MODEL, "messages": [
            {"role": "user", "content": [{"type": "input_audio",
                                          "input_audio": {"data": "", "format": "wav"}}]}]})
        self.assertEqual(r.status_code, 400)
        self.assertIn("audio", r.json()["error"]["message"])

    def test_raw_completions_use_the_prompt_verbatim(self):
        """No chat template, no generation prompt — and therefore no channel
        header, which the parser has to be told about or it waits forever for
        a `<|message|>` that never comes."""
        c, _ = self._client("Paris is the capital of France.<|end_of_text|>")
        r = c.post("/v1/completions", json={"model": MODEL, "prompt": "Complete: ",
                                            "max_tokens": 32})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertTrue(r.json()["choices"][0]["text"].startswith("Paris"),
                        r.json()["choices"][0]["text"][:40])

    def test_health_and_models(self):
        c, _ = self._client(" to=user<|message|>x<|eot|>")
        self.assertEqual(c.get("/health").json()["status"], "ok")
        self.assertEqual(c.get("/v1/models").json()["data"][0]["id"], MODEL)
        self.assertIn("muse_requests", c.get("/metrics").text)

    # ----------------------------------------------------------------- anthropic
    def test_anthropic_messages(self):
        c, _ = self._client(" to=self<|message|>hmm<|eom|>"
                            "<|start|>assistant to=user<|message|>Bonjour.<|eot|>")
        r = c.post("/v1/messages", json={
            "model": MODEL, "max_tokens": 64,
            "system": "Be brief.",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]})
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        kinds = [b["type"] for b in body["content"]]
        self.assertEqual(kinds, ["thinking", "text"])
        self.assertEqual(body["content"][1]["text"], "Bonjour.")
        self.assertEqual(body["stop_reason"], "end_turn")

    def test_anthropic_streaming_blocks(self):
        c, _ = self._client(" to=self<|message|>hmm<|eom|>"
                            "<|start|>assistant to=user<|message|>Bonjour.<|eot|>")
        with c.stream("POST", "/v1/messages", json={
                "model": MODEL, "max_tokens": 64, "stream": True,
                "messages": [{"role": "user", "content": "hi"}]}) as r:
            body = "".join(r.iter_text())
        events = [l[7:] for l in body.splitlines() if l.startswith("event: ")]
        self.assertEqual(events[0], "message_start")
        self.assertEqual(events[-1], "message_stop")
        self.assertIn("thinking_delta", body)
        self.assertIn("text_delta", body)

    def test_anthropic_count_tokens(self):
        c, _ = self._client(" to=user<|message|>x<|eot|>")
        r = c.post("/v1/messages/count_tokens", json={
            "model": MODEL, "messages": [{"role": "user", "content": "hello"}]})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertGreater(r.json()["input_tokens"], 10)

    def test_anthropic_tool_result_round_trip(self):
        c, _ = self._client(" to=user<|message|>18C.<|eot|>")
        r = c.post("/v1/messages", json={
            "model": MODEL, "max_tokens": 64,
            "tools": [{"name": "get_weather", "description": "w",
                       "input_schema": {"type": "object",
                                        "properties": {"city": {"type": "string"}}}}],
            "messages": [
                {"role": "user", "content": "weather?"},
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "toolu_1", "name": "get_weather",
                     "input": {"city": "Paris"}}]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_1", "content": "18C"}]}]})
        self.assertEqual(r.status_code, 200, r.text)

    # ----------------------------------------------------------------- responses
    def test_responses_output_items(self):
        c, _ = self._client(" to=self<|message|>hmm<|eom|>"
                            "<|start|>assistant to=user<|message|>Done.<|eot|>")
        r = c.post("/v1/responses", json={"model": MODEL, "input": "hi",
                                          "reasoning": {"effort": "low"}})
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertEqual([i["type"] for i in body["output"]], ["reasoning", "message"])
        self.assertEqual(body["output_text"], "Done.")
        self.assertEqual(body["status"], "completed")

    def test_responses_streaming(self):
        c, _ = self._client(" to=user<|message|>Streamed.<|eot|>")
        with c.stream("POST", "/v1/responses",
                      json={"model": MODEL, "input": "hi", "stream": True}) as r:
            body = "".join(r.iter_text())
        events = [l[7:] for l in body.splitlines() if l.startswith("event: ")]
        self.assertEqual(events[0], "response.created")
        self.assertEqual(events[-1], "response.completed")
        deltas = [json.loads(l[6:])["delta"] for l in body.splitlines()
                  if l.startswith("data: ") and '"response.output_text.delta"' in l]
        self.assertEqual("".join(deltas), "Streamed.")

    def test_responses_refuses_state_it_does_not_have(self):
        c, _ = self._client(" to=user<|message|>x<|eot|>")
        r = c.post("/v1/responses", json={"model": MODEL, "input": "hi",
                                          "previous_response_id": "resp_123"})
        self.assertEqual(r.status_code, 400)
        self.assertIn("previous_response_id", r.json()["error"]["message"])

    def test_responses_function_call_output_round_trip(self):
        c, _ = self._client(" to=user<|message|>ok<|eot|>")
        r = c.post("/v1/responses", json={
            "model": MODEL,
            "tools": [{"type": "function", "name": "f", "description": "",
                       "parameters": {"type": "object", "properties": {}}}],
            "input": [{"role": "user", "content": "go"},
                      {"type": "function_call", "call_id": "call_1", "name": "f",
                       "arguments": "{}"},
                      {"type": "function_call_output", "call_id": "call_1",
                       "output": "done"}]})
        self.assertEqual(r.status_code, 200, r.text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
