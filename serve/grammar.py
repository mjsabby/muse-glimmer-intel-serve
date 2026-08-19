"""Guided-decoding binding: the C++ JSON PDA (src/json_grammar.h via
libmuse-intel-serve.so) + the per-tokenizer piece table built from the HF
tokenizer.

JsonGrammar implements the generate-loop hooks: mask(vocab) -> bool[V],
accept(id). Special/added tokens get empty pieces (always masked; EOG ids
re-allowed once the JSON value is complete).
"""
from __future__ import annotations

import ctypes

import numpy as np

from .token_bytes import piece_bytes, uses_bytelevel


_PIECE_CACHE: dict[int, list[bytes]] = {}


def piece_list(tokenizer, vocab: int) -> list[bytes]:
    """The exact bytes each token contributes to the output, or b"" for a
    special/added token (which no byte-level grammar may ever sample).

    Walking a 202048-token vocabulary is not free, so it is done once per
    tokenizer and shared with the recipient grammar."""
    key = id(tokenizer)
    hit = _PIECE_CACHE.get(key)
    if hit is not None and len(hit) == vocab:
        return hit
    toks = tokenizer.convert_ids_to_tokens(list(range(vocab)))
    special = set(tokenizer.all_special_tokens) | set(tokenizer.get_added_vocab().keys())
    bytelevel = uses_bytelevel(toks)
    out = [piece_bytes(t, bytelevel) if (t and t not in special) else b"" for t in toks]
    _PIECE_CACHE[key] = out
    return out


def build_piece_table(tokenizer, vocab: int, eog_ids: set[int]):
    """(bytes_blob, offsets int64[V+1], eog uint8[V]) for the C ABI."""
    pieces = piece_list(tokenizer, vocab)
    blob = bytearray()
    off = np.zeros(vocab + 1, dtype=np.int64)
    eog = np.zeros(vocab, dtype=np.uint8)
    for i, pc in enumerate(pieces):
        if i in eog_ids:
            eog[i] = 1
        else:
            blob.extend(pc)
        off[i + 1] = len(blob)
    return bytes(blob), off, eog


class PieceTable:
    """Shared per-tokenizer piece cache (build once per process)."""

    def __init__(self, lib: ctypes.CDLL, tokenizer, vocab: int, eog_ids: set[int]):
        self._L = lib
        lib.muse_pieces_new.restype = ctypes.c_void_p
        lib.muse_pieces_new.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_int64),
                                          ctypes.c_char_p, ctypes.c_int64]
        lib.muse_pieces_free.argtypes = [ctypes.c_void_p]
        lib.muse_jsong_new.restype = ctypes.c_void_p
        lib.muse_jsong_new.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.muse_jsong_free.argtypes = [ctypes.c_void_p]
        lib.muse_jsong_mask.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        lib.muse_jsong_accept.argtypes = [ctypes.c_void_p, ctypes.c_int32]
        lib.muse_jsong_complete.restype = ctypes.c_int
        lib.muse_jsong_complete.argtypes = [ctypes.c_void_p]
        blob, off, eog = build_piece_table(tokenizer, vocab, eog_ids)
        self.vocab = vocab
        self._h = lib.muse_pieces_new(
            blob, off.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            eog.tobytes(), vocab)

    def __del__(self):
        if getattr(self, "_h", None):
            self._L.muse_pieces_free(self._h)
            self._h = None


class JsonGrammar:
    """Per-request constraint (response_format json_object)."""

    def __init__(self, pieces: PieceTable, object_only: bool = True):
        self._p = pieces
        self._L = pieces._L
        self._buf = np.empty(pieces.vocab, dtype=np.uint8)
        self._h = self._L.muse_jsong_new(pieces._h, int(object_only))

    def mask(self, vocab: int) -> np.ndarray:
        self._L.muse_jsong_mask(self._h, self._buf.ctypes.data_as(ctypes.c_char_p))
        return self._buf.view(bool)

    def accept(self, token_id: int) -> None:
        self._L.muse_jsong_accept(self._h, int(token_id))

    @property
    def complete(self) -> bool:
        return bool(self._L.muse_jsong_complete(self._h))

    def __del__(self):
        if getattr(self, "_h", None):
            self._L.muse_jsong_free(self._h)
            self._h = None


class SchemaGrammar:
    """Per-request constraint for response_format json_schema (strict). The
    schema is compiled + enforced in C++ (src/json_schema.h); Python only
    json.dumps() it once. Same mask/accept/complete hooks as JsonGrammar."""

    def __init__(self, pieces: PieceTable, schema: dict):
        import json as _json
        self._p = pieces
        self._L = L = pieces._L
        self._buf = np.empty(pieces.vocab, dtype=np.uint8)
        L.muse_jsonschema_new.restype = ctypes.c_void_p
        L.muse_jsonschema_new.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        L.muse_jsonschema_free.argtypes = [ctypes.c_void_p]
        L.muse_jsonschema_mask.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        L.muse_jsonschema_accept.argtypes = [ctypes.c_void_p, ctypes.c_int32]
        L.muse_jsonschema_complete.restype = ctypes.c_int
        L.muse_jsonschema_complete.argtypes = [ctypes.c_void_p]
        self._h = L.muse_jsonschema_new(pieces._h, _json.dumps(schema or {}).encode())
        if not self._h:
            L.muse_last_error.restype = ctypes.c_char_p
            raise ValueError(f"schema grammar: {L.muse_last_error().decode()}")

    def mask(self, vocab: int) -> np.ndarray:
        self._L.muse_jsonschema_mask(self._h, self._buf.ctypes.data_as(ctypes.c_char_p))
        return self._buf.view(bool)

    def accept(self, token_id: int) -> None:
        self._L.muse_jsonschema_accept(self._h, int(token_id))

    @property
    def complete(self) -> bool:
        return bool(self._L.muse_jsonschema_complete(self._h))

    def __del__(self):
        if getattr(self, "_h", None):
            self._L.muse_jsonschema_free(self._h)
            self._h = None
