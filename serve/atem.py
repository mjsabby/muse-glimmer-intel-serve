"""The assistant-turn parser, driven by the checkpoint's `response_template`.

Muse Glimmer does not emit one string with markers in it. It emits a sequence
of CHANNELS, each addressed to a recipient:

    <|start|>assistant to=self<|message|>   thinking out loud            <|eom|>
    <|start|>assistant to=user<|message|>   the answer                   <|eot|>
    <|start|>assistant to=web.search<|message|> <atem:function_calls>…   <|eot|>

`<|eom|>` closes a channel and the turn continues; `<|eot|>` ends the turn.
That is why `reasoning_content` cannot leak into `content` here: they are
different channels, not different regions of one string, and the split is made
of single vocab tokens rather than text that might arrive half-decoded.

So the parser runs on TOKEN IDS, not on text. The four structural markers
(`<|start|>`, `<|message|>`, `<|eom|>`, `<|eot|>`) are single ids, which
removes every partial-marker holdback problem a text-level parser has; only the
recipient and the body are ordinary text.

The regexes for the ATEM tool-call body are read out of
`tokenizer_config.json`'s `response_template` rather than written here, and
`validate_template` fails loudly if a future revision changes the shape this
module assumes. A silently mis-parsed tool call is worse than a startup error.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .chatlib import EOM, EOT, MESSAGE, START, load_json

SELF = "self"
USER = "user"


# --------------------------------------------------------------- the template

@dataclass
class ResponseTemplate:
    """The subset of `response_template` this parser implements."""
    start_anchor: str = f"{START}assistant"
    tool_open: re.Pattern = None       # <atem:invoke name="...">
    tool_close: str = "</atem:invoke>"
    param: re.Pattern = None           # <atem:parameter name="k">v</atem:parameter>
    allow_non_json: bool = True

    @staticmethod
    def load(model: str) -> "ResponseTemplate":
        rt = load_json(model, "tokenizer_config.json").get("response_template") or {}
        fields = rt.get("fields") or {}
        tc = fields.get("tool_calls") or {}
        args = (tc.get("content_args") or {})
        vp = (args.get("value_parser") or {})
        out = ResponseTemplate(
            start_anchor=rt.get("start_anchor", f"{START}assistant"),
            tool_open=re.compile(tc.get("open_pattern", r'<atem:invoke\b[^>]*?\bname="(?P<name>[^"]+)">')),
            tool_close=tc.get("close", "</atem:invoke>"),
            param=re.compile(args.get("tag_pattern",
                                      r'<atem:parameter\b[^>]*?\bname="(?P<key>[^"]+)"[^>]*?>'
                                      r'(?P<value>.*?)</atem:parameter>'), re.S),
            allow_non_json=bool((vp.get("args") or {}).get("allow_non_json", True)))
        out.validate(fields)
        return out

    def validate(self, fields: dict) -> None:
        """The assumptions this module is built on, checked against the file.

        Checked rather than trusted because every one of these failing produces
        plausible output rather than an error: reasoning shown as the answer,
        a tool call rendered as prose, a turn that never ends."""
        problems = []
        c, r = fields.get("content") or {}, fields.get("reasoning_content") or {}
        if f"to={USER}" not in (c.get("open_pattern") or ""):
            problems.append("content channel is no longer addressed to=user")
        if f"to={SELF}" not in (r.get("open_pattern") or ""):
            problems.append("reasoning channel is no longer addressed to=self")
        closes = c.get("close") or []
        closes = [closes] if isinstance(closes, str) else closes
        if EOT not in closes:
            problems.append(f"content channel no longer closes on {EOT}")
        if (r.get("close") or "") != EOM:
            problems.append(f"reasoning channel no longer closes on {EOM}")
        if self.start_anchor != f"{START}assistant":
            problems.append(f"start anchor is {self.start_anchor!r}")
        if problems:
            raise ValueError("this checkpoint's response_template does not match the ATEM "
                             "channel protocol serve/atem.py implements:\n  - "
                             + "\n  - ".join(problems))


def parse_atem_body(text: str, rt: ResponseTemplate) -> list[dict]:
    """The ATEM XML in a tool channel -> OpenAI tool_calls entries."""
    out = []
    for m in rt.tool_open.finditer(text):
        name = m.group("name")
        end = text.find(rt.tool_close, m.end())
        body = text[m.end():end if end >= 0 else len(text)]
        args = {}
        for p in rt.param.finditer(body):
            args[p.group("key")] = _value(p.group("value"), rt.allow_non_json)
        out.append({"type": "function",
                    "function": {"name": name, "arguments": json.dumps(args)}})
    return out


def _value(raw: str, allow_non_json: bool):
    """`value_parser: json` with `allow_non_json`: try JSON, else keep the text.

    The template writes scalars bare (`<atem:parameter name="q">weather</…>`)
    and containers as JSON, so both have to round-trip. Trailing whitespace is
    the template's own newline before the close tag."""
    s = raw.strip("\n")
    try:
        return json.loads(s)
    except (json.JSONDecodeError, ValueError):
        if allow_non_json:
            return s
        raise


# ----------------------------------------------------------------- the parser

@dataclass
class Event:
    kind: str            # "reasoning" | "content" | "tool_calls" | "channel"
    text: str = ""
    calls: list = field(default_factory=list)
    recipient: str = ""


class ChannelParser:
    """Token-level assistant-turn parser.

    push(id) -> list[Event]. Deltas for the text channels arrive as they
    decode; a tool channel is buffered and emitted whole at its close, because
    an ATEM call is not meaningful in fragments.
    """

    HEADER, BODY = 0, 1

    def __init__(self, tokenizer, rt: ResponseTemplate, raw: bool = False):
        self.tok = tokenizer
        self.rt = rt
        # `raw` is /v1/completions: the prompt is used verbatim, so there is no
        # `<|start|>assistant` and no recipient to wait for. Without this the
        # parser sits in HEADER forever looking for a `<|message|>` that a raw
        # completion never sends, and the endpoint returns an empty string.
        self.state = self.BODY if raw else self.HEADER
        self.header: list[int] = []
        self.body: list[int] = []
        self.printed = 0
        self.recipient = USER
        self.specials = {tokenizer.convert_tokens_to_ids(t): t
                         for t in (START, MESSAGE, EOM, EOT)}
        # Everything the model has produced, for the trace and for the next
        # turn's prefix reuse.
        self.raw: list[int] = []

    # -- helpers
    def _decode(self, ids: list[int]) -> str:
        return self.tok.decode(ids, skip_special_tokens=False)

    def _body_delta(self) -> str:
        """Emit only what is safely decodable: a piece that ends mid-UTF-8
        would print a replacement character the next token completes."""
        text = self._decode(self.body)
        if text.endswith("�"):
            return ""
        piece = text[self.printed:]
        self.printed = len(text)
        return piece

    def _close_channel(self) -> list[Event]:
        ev: list[Event] = []
        if self.recipient == SELF:
            ev.append(Event("reasoning", self._body_delta()))
        elif self.recipient == USER:
            ev.append(Event("content", self._body_delta()))
        else:
            calls = parse_atem_body(self._decode(self.body), self.rt)
            if not calls:
                # The channel was addressed to a tool but the body is not an
                # ATEM block. Report it as a call with the raw text so the
                # client sees something actionable rather than silence.
                calls = [{"type": "function",
                          "function": {"name": self.recipient,
                                       "arguments": json.dumps(
                                           {"_raw": self._decode(self.body)})}}]
            ev.append(Event("tool_calls", calls=calls, recipient=self.recipient))
        self.state, self.header, self.body, self.printed = self.HEADER, [], [], 0
        self.recipient = USER
        # A channel whose text was fully emitted during push has nothing left
        # to say at close; do not send an empty delta for it.
        return [e for e in ev if e.calls or e.text]

    # -- the machine
    def push(self, token_id: int) -> list[Event]:
        self.raw.append(token_id)
        special = self.specials.get(token_id)

        if self.state == self.HEADER:
            if special == MESSAGE:
                # ` to=<recipient>` (the template always emits one; `assistant`
                # with no recipient means the user channel).
                head = self._decode(self.header)
                m = re.search(r"to=([^\s<|]+)", head)
                self.recipient = m.group(1) if m else USER
                self.state, self.header = self.BODY, []
                return [Event("channel", recipient=self.recipient)]
            if special in (EOM, EOT):
                # An empty channel: nothing to report, just re-arm.
                self.header = []
                return []
            if special != START:
                self.header.append(token_id)
            return []

        if special in (EOM, EOT):
            return self._close_channel()
        if special == START:
            # A new channel opened without closing the last one. Close it
            # ourselves rather than merging two channels into one string.
            return self._close_channel()

        self.body.append(token_id)
        if self.recipient == SELF:
            piece = self._body_delta()
            return [Event("reasoning", piece)] if piece else []
        if self.recipient == USER:
            piece = self._body_delta()
            return [Event("content", piece)] if piece else []
        return []  # tool bodies are emitted whole at close

    def flush(self) -> list[Event]:
        """End of generation without a closing marker (max_tokens, a stop
        string, a cancelled request). Emit what is in hand."""
        if self.state == self.BODY and (self.body or self.printed):
            return self._close_channel()
        return []
