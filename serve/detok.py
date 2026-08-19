"""Incremental UTF-8-safe detokenization + stop-string holdback.

IncrementalDetok mirrors transformers.TextStreamer: decode a growing token
cache, emit only when the text doesn't end in an incomplete UTF-8 sequence,
and flush the cache at newlines to bound re-decode cost.

StopStrings buffers just enough text that a stop string split across piece
boundaries is never emitted to the client.
"""
from __future__ import annotations


class IncrementalDetok:
    def __init__(self, tokenizer):
        self.tok = tokenizer
        self.cache: list[int] = []
        self.printed = 0  # chars of decode(cache) already emitted

    def put(self, token_id: int) -> str:
        self.cache.append(token_id)
        text = self.tok.decode(self.cache, skip_special_tokens=False)
        if text.endswith("�"):  # incomplete UTF-8 from a partial byte piece
            return ""
        piece = text[self.printed:]
        if text.endswith("\n"):
            self.cache, self.printed = [], 0
        else:
            self.printed = len(text)
        return piece

    def flush(self) -> str:
        if not self.cache:
            return ""
        text = self.tok.decode(self.cache, skip_special_tokens=False)
        piece = text[self.printed:]
        self.cache, self.printed = [], 0
        return piece


class StopStrings:
    """feed() returns (emit_now, stop_hit). Holds back len(longest stop)-1
    chars so a stop spanning piece boundaries is caught before emission."""

    def __init__(self, stops: list[str]):
        self.stops = [s for s in stops if s]
        self.hold = max((len(s) for s in self.stops), default=1) - 1
        self.buf = ""

    def feed(self, piece: str) -> tuple[str, bool]:
        if not self.stops:
            return piece, False
        self.buf += piece
        for s in self.stops:
            i = self.buf.find(s)
            if i >= 0:
                out, self.buf = self.buf[:i], ""
                return out, True
        if self.hold and len(self.buf) > self.hold:
            out, self.buf = self.buf[:-self.hold], self.buf[-self.hold:]
            return out, False
        return ("", False) if self.hold else (self._take(), False)

    def _take(self) -> str:
        out, self.buf = self.buf, ""
        return out

    def flush(self) -> str:
        return self._take()
