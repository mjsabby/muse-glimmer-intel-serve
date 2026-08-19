"""The generation loop: prefill, sample, parse channels, stop.

One loop serves every protocol. It yields dicts because three different wire
formats consume it and none of them wants the others' objects:

    {"type": "reasoning",  "text": str}
    {"type": "content",    "text": str}
    {"type": "tool_calls", "calls": [...]}
    {"type": "done",       "finish_reason": str, "usage": {...},
                           "logprobs": [...], "spec": {...}}

Speculation is in here rather than beside it, because the accept rule has to
see the same sampler, the same grammar masks and the same stop conditions as
ordinary decoding — a second loop is how those drift apart.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .atem import ChannelParser, ResponseTemplate
from .detok import StopStrings
from .sampler import SampleParams, Sampler


@dataclass
class GenParams:
    max_tokens: int = 1024
    stop: list[str] = field(default_factory=list)
    sample: SampleParams = field(default_factory=SampleParams)
    # Guided decoding, applied to the CONTENT channel only: constraining the
    # reasoning channel to JSON would be constraining the wrong string.
    json_mode: bool = False
    json_schema: dict | None = None
    # (recipients, allow_self_once) from recipient.recipients_for(), or None.
    recipients: tuple[list[str], bool] | None = None
    speculative: bool = True
    echo_prompt_logprobs: bool = False


def _finish(reason, usage, logprobs, spec):
    return {"type": "done", "finish_reason": reason, "usage": usage,
            "logprobs": logprobs, "spec": spec}


class Generation:
    """One request against one engine. Not reusable and not concurrent."""

    def __init__(self, eng, tokenizer, rt: ResponseTemplate, params: GenParams,
                 pieces=None, eos: set[int] | None = None, raw: bool = False):
        self.eng, self.tok, self.rt, self.p = eng, tokenizer, rt, params
        self.sampler = Sampler(params.sample)
        self.parser = ChannelParser(tokenizer, rt, raw=raw)
        self.stops = StopStrings(list(params.stop or []))
        self.eos = set(eos or ())
        self.pieces = pieces
        self.json_grammar = None
        self.recipient_grammar = None
        if params.recipients:
            from .recipient import RecipientGrammar
            names, once = params.recipients
            self.recipient_grammar = RecipientGrammar(tokenizer, eng.vocab, names,
                                                      allow_self_once=once)
        if (params.json_mode or params.json_schema) and pieces is not None:
            from .grammar import JsonGrammar, SchemaGrammar
            self.json_grammar = (SchemaGrammar(pieces, params.json_schema)
                                 if params.json_schema else JsonGrammar(pieces))
        self.generated: list[int] = []
        self.logprobs: list = []
        self.emitted = 0
        self._stop_hit = False

    # ------------------------------------------------------------------ masks
    def _mask(self) -> np.ndarray | None:
        """Intersection of the live constraints, or None when there are none.

        The JSON constraint applies only inside the answer channel: the model
        must still be able to spell ` to=user<|message|>` to get there, and its
        reasoning is not the JSON the caller asked for."""
        m = None
        if self.recipient_grammar is not None:
            m = self.recipient_grammar.mask(self.eng.vocab)
        if self.json_grammar is not None and self._in_answer():
            j = self.json_grammar.mask(self.eng.vocab)
            m = j if m is None else (m & j)
        return m

    def _in_answer(self) -> bool:
        from .atem import USER
        return self.parser.state == ChannelParser.BODY and self.parser.recipient == USER

    def _accept_token(self, tid: int) -> None:
        if self.recipient_grammar is not None:
            self.recipient_grammar.accept(tid)
        if self.json_grammar is not None and self._in_answer():
            self.json_grammar.accept(tid)

    # ------------------------------------------------------------------- loop
    def run(self, ids: list[int]):
        """Drive a whole request. `ids` is the rendered prompt."""
        eng, p = self.eng, self.p
        t0 = time.perf_counter()
        reuse = eng.reuse_prefix(ids)
        logits = eng.forward(ids[reuse:])
        t_prefill = time.perf_counter() - t0

        n_prompt = len(ids)
        spec_rounds = spec_drafted = spec_accepted = 0
        finish = None
        t1 = time.perf_counter()
        use_spec = bool(p.speculative and eng.spec_block > 1)

        # The loop's state is a COMMITTED TOKEN that is not yet in the KV
        # cache — `tid`, the anchor. That is the shape the drafter wants: it
        # proposes against the taps of the last forward with the anchor sitting
        # at cache_len, and the verification pass then puts the anchor and the
        # proposals into the cache together.
        #
        # Getting this wrong is expensive rather than incorrect: an earlier
        # version forwarded the anchor on its own before drafting, which is a
        # whole extra 52-layer decode step per round — measured at ~60 ms
        # against a ~95 ms round.
        tid, lp = self.sampler.sample(logits, self.generated, self._mask())
        if lp is not None:
            self.logprobs.append(lp)
        for out in self._commit(tid):
            yield out
        finish = self._finish_reason(tid)

        while not finish:
            if use_spec:
                proposals = eng.draft(tid)
                if proposals:
                    spec_rounds += 1
                    spec_drafted += len(proposals)
                    block = [tid] + proposals
                    rows = eng.forward_all(block)
                    base = eng.cache_len - len(block)   # absolute position of tid
                    acc, tid, finish, events = self._verify(rows, proposals)
                    spec_accepted += acc
                    # Keep exactly the tokens whose prefix survived: the anchor
                    # plus the accepted proposals. Every later cache row was
                    # computed against a prefix that did not.
                    eng.rollback(base + 1 + acc)
                    for out in events:
                        yield out
                    continue
            logits = eng.forward([tid])
            tid, lp = self.sampler.sample(logits, self.generated, self._mask())
            if lp is not None:
                self.logprobs.append(lp)
            for out in self._commit(tid):
                yield out
            finish = self._finish_reason(tid)

        for ev in self.parser.flush():
            for out in self._emit(ev):
                yield out
        tail = self.stops.flush()
        if tail:
            yield {"type": "content", "text": tail}

        dt = time.perf_counter() - t1
        usage = {"prompt_tokens": n_prompt, "completion_tokens": self.emitted,
                 "total_tokens": n_prompt + self.emitted,
                 "prefill_s": t_prefill, "decode_s": dt,
                 "prefill_reused": reuse,
                 "tokens_per_second": (self.emitted / dt) if dt > 0 else 0.0}
        spec = None
        if spec_rounds:
            spec = {"spec_rounds": spec_rounds, "spec_drafted": spec_drafted,
                    "spec_accepted": spec_accepted,
                    "spec_accept_rate": spec_accepted / spec_drafted if spec_drafted else 0.0}
        yield _finish(finish or "stop", usage, self.logprobs, spec)

    def _finish_reason(self, tid: int) -> str | None:
        if tid in self.eos:
            return "stop"
        if self._stop_hit:
            return "stop"
        if self.emitted >= self.p.max_tokens:
            return "length"
        return None

    # ------------------------------------------------------------ speculation
    def _verify(self, rows: np.ndarray, proposals: list[int]):
        """Walk the verification rows under the same sampler and grammars.

        The rule is the standard one for a DETERMINISTIC draft (the draft
        distribution q is a point mass on the proposal): accept with
        probability p_target(proposal), and on rejection draw from the target
        with that token's mass removed. That is distribution-preserving, and at
        temperature <= 0 it collapses to "accept iff the proposal is the
        target's argmax" — which is why greedy speculative output is
        bit-identical to greedy plain output, and why that is the gate.

        Returns (accepted, next_anchor, finish, events). The anchor is a
        committed token that is NOT in the cache, exactly like the one the
        caller came in with.
        """
        events: list[dict] = []
        accepted = 0
        for i, cand in enumerate(proposals):
            mask = self._mask()
            ids_, probs = self.sampler.dist(rows[i], self.generated, mask)
            hit = np.nonzero(ids_ == cand)[0]
            pc = float(probs[hit[0]]) if hit.size else 0.0
            if pc >= 1.0 or (pc > 0.0 and self.sampler.rng.random() < pc):
                tok, ok = cand, True
            else:
                if hit.size:
                    residual = probs.copy()
                    residual[hit[0]] = 0.0
                    total = residual.sum()
                    residual = residual / total if total > 0 else None
                else:
                    residual = probs
                tok = (int(ids_[int(self.sampler.rng.choice(len(ids_), p=residual))])
                       if residual is not None else int(ids_[0]))
                ok = False
            lp = self.sampler.logprobs_for(rows[i], self.generated, mask, tok)
            if lp is not None:
                self.logprobs.append(lp)
            events.extend(self._commit(tok))
            if ok:
                accepted += 1
            fin = self._finish_reason(tok)
            if fin:
                return accepted, tok, fin, events
            if not ok:
                # Every later row was conditioned on a token that did not
                # survive, so the block ends here and `tok` is the next anchor.
                return accepted, tok, None, events
        # Every proposal survived and is in the cache. The last row is the
        # distribution for the next position, and nothing has drawn from it —
        # so the bonus token comes from there.
        tok, lp = self.sampler.sample(rows[len(proposals)], self.generated, self._mask())
        if lp is not None:
            self.logprobs.append(lp)
        events.extend(self._commit(tok))
        return accepted, tok, self._finish_reason(tok), events

    # -------------------------------------------------------------- emission
    def _commit(self, tid: int) -> list[dict]:
        """Record a token and turn it into wire events."""
        self._accept_token(tid)
        self.generated.append(tid)
        self.emitted += 1
        out: list[dict] = []
        for ev in self.parser.push(tid):
            out.extend(self._emit(ev))
        return out

    def _emit(self, ev) -> list[dict]:
        if ev.kind == "channel":
            return []
        if ev.kind == "tool_calls":
            return [{"type": "tool_calls", "calls": ev.calls}]
        if ev.kind == "reasoning":
            return [{"type": "reasoning", "text": ev.text}] if ev.text else []
        if ev.kind == "content":
            text, hit = self.stops.feed(ev.text)
            if hit:
                self._stop_hit = True
            return [{"type": "content", "text": text}] if text else []
        return []


def generate(eng, tokenizer, rt, ids, params, **kw) -> dict:
    """Non-streaming convenience: run to completion and collect."""
    out = {"content": "", "reasoning": "", "tool_calls": [], "finish_reason": "stop",
           "usage": {}, "logprobs": [], "spec": None}
    for ev in Generation(eng, tokenizer, rt, params, **kw).run(ids):
        t = ev["type"]
        if t == "content":
            out["content"] += ev["text"]
        elif t == "reasoning":
            out["reasoning"] += ev["text"]
        elif t == "tool_calls":
            out["tool_calls"].extend(ev["calls"])
        elif t == "done":
            out["finish_reason"] = ev["finish_reason"]
            out["usage"] = ev["usage"]
            out["logprobs"] = ev["logprobs"]
            out["spec"] = ev["spec"]
    if out["tool_calls"] and out["finish_reason"] == "stop":
        out["finish_reason"] = "tool_calls"
    return out
