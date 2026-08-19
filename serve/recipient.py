"""`tool_choice` as a grammar over the recipient, not a prompt nudge.

Muse Glimmer's generation prompt is a bare `<|start|>assistant`, and the model
writes the recipient itself:

    <|start|>assistant to=self<|message|>            reasoning
    <|start|>assistant to=user<|message|>            the answer
    <|start|>assistant to=web.search<|message|>      a tool call

which means `tool_choice` is a constraint on a handful of tokens at a known
position, not a hope. That is a better deal than the sibling engines get: with
a `<|tool_call>`-opener FSM you can force the model to START a call and then
watch it invent a name that was never declared. Here the name IS the channel
address, so `required` can be a hard guarantee.

    none      recipients clamped to {self, user} — the model may reason and
              must answer; no tool can be addressed
    auto      no constraint at all (mask() returns None, so the sampler pays
              nothing)
    required  {self} + every declared tool, and `self` only for the FIRST
              channel — so the turn cannot reason forever and then answer the
              user instead of calling something
    {"function": {"name": X}}   {self} + {X}, same rule

Matching is byte-level against the vocabulary's piece strings, so every BPE
split of a legal recipient is admissible; assuming one canonical tokenization
is exactly the trap this avoids. And the forced region can never dead-end,
because a legal recipient is always reachable through single-byte pieces.
"""
from __future__ import annotations

import numpy as np

from .chatlib import MESSAGE, START
from .grammar import piece_list

PREFIX = b" to="


class RecipientGrammar:
    """A latent constraint: costs nothing outside the header region.

    Hooks match the JSON grammars': mask(vocab) -> bool[V] or None, accept(id).
    """

    def __init__(self, tokenizer, vocab: int, recipients: list[str], *,
                 allow_self_once: bool = True):
        self.vocab = vocab
        self.msg_id = int(tokenizer.convert_tokens_to_ids(MESSAGE))
        self.start_id = int(tokenizer.convert_tokens_to_ids(START))
        self.pieces = pieces = piece_list(tokenizer, vocab)
        # bytes -> ids. Several ids can share a piece in a vocabulary with
        # duplicates; all of them are equally legal.
        self.by_piece: dict[bytes, list[int]] = {}
        for i, pc in enumerate(pieces):
            if pc:
                self.by_piece.setdefault(pc, []).append(i)
        self.literals = [PREFIX + r.encode() for r in recipients]
        self.allow_self_once = allow_self_once
        self.self_literal = PREFIX + b"self"
        # State: we start inside a header, because the prompt ends at
        # `<|start|>assistant` and the recipient is the next thing generated.
        self.in_header = True
        self.matched = b""
        self.used_self = False
        self._cache: dict[tuple, np.ndarray] = {}

    # -------------------------------------------------------------- internals
    def _live(self) -> list[bytes]:
        """Literals still consistent with what has been matched so far."""
        out = [L for L in self.literals if L.startswith(self.matched)]
        if self.used_self and self.allow_self_once:
            out = [L for L in out if L != self.self_literal]
        return out

    def _allowed_ids(self) -> list[int]:
        ids: list[int] = []
        i = len(self.matched)
        for L in self._live():
            if i == len(L):
                ids.append(self.msg_id)  # this recipient is complete
                continue
            for l in range(1, len(L) - i + 1):
                ids.extend(self.by_piece.get(L[i:i + l], ()))
        return ids

    # ------------------------------------------------------------------ hooks
    def mask(self, vocab: int) -> np.ndarray | None:
        if not self.in_header:
            return None
        key = (self.matched, self.used_self)
        hit = self._cache.get(key)
        if hit is None:
            ids = self._allowed_ids()
            hit = np.zeros(vocab, dtype=bool)
            if ids:
                hit[np.asarray(ids, dtype=np.int64)] = True
            else:
                # Should be unreachable (a live literal always has a
                # single-byte continuation). Fail OPEN rather than hand the
                # sampler an all-masked vector.
                hit[:] = True
            self._cache[key] = hit
        return hit

    def accept(self, token_id: int) -> None:
        if token_id == self.start_id:
            # A new channel opened: constrain its recipient too.
            self.in_header, self.matched = True, b""
            return
        if not self.in_header:
            return
        if token_id == self.msg_id:
            if self.matched == self.self_literal:
                self.used_self = True
            self.in_header, self.matched = False, b""
            return
        # Only ever called with a token the mask allowed, so this always
        # extends the match.
        self.matched += self.pieces[token_id]

    @property
    def complete(self) -> bool:
        return not self.in_header


def recipients_for(tool_choice, tools) -> tuple[list[str], bool] | None:
    """(recipients, allow_self_once) for a request, or None for unconstrained.

    `tools` is the sanitized list; `tool_choice` follows OpenAI's shape.
    """
    names = [t["function"]["name"] for t in (tools or [])]
    if tool_choice in (None, "auto", "", {}):
        return None
    if tool_choice == "none":
        # Reasoning and the answer stay available; no tool is addressable.
        return (["self", "user"], False)
    if tool_choice == "required":
        if not names:
            raise ValueError('tool_choice "required" with no tools declared')
        return (["self"] + names, True)
    if isinstance(tool_choice, dict):
        fn = (tool_choice.get("function") or {})
        want = fn.get("name") or tool_choice.get("name")
        if not want:
            raise ValueError(f"unsupported tool_choice: {tool_choice!r}")
        if names and want not in names:
            raise ValueError(f'tool_choice names "{want}", which is not in tools[]')
        return (["self", want], True)
    raise ValueError(f"unsupported tool_choice: {tool_choice!r}")
