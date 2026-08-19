"""ctypes binding for libmuse-intel-serve.so, plus KV prefix reuse.

One Engine = one loaded model on the cards. NOT thread-safe: it holds a single
KV cache and serves one generation at a time; the server takes a lock around
every request.

The logits buffer is REUSED across steps. Sample from it before the next
forward, or copy it.
"""
from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent

VISION_GPU, VISION_CPU, VISION_OFF = 0, 1, 2


def _find_lib(explicit: str | None) -> str:
    for cand in ([explicit] if explicit else []) + [
        os.environ.get("MUSE_LIB"),
        str(_REPO / "build-gpu" / "libmuse-intel-serve.so"),
    ]:
        if cand and Path(cand).exists():
            return cand
    raise FileNotFoundError(
        "libmuse-intel-serve.so not found — build it with\n"
        "  source /opt/intel/oneapi/setvars.sh\n"
        "  cmake --build build-gpu --target muse-intel-serve\n"
        "or point MUSE_LIB at it")


class _OpenOpts(ctypes.Structure):
    # Must match MuseOpenOpts in src/gpu/c_api.cpp field for field.
    _fields_ = [
        ("model", ctypes.c_char_p),
        ("revision", ctypes.c_char_p),
        ("assistant", ctypes.c_char_p),
        ("gpus", ctypes.c_int),
        ("shards", ctypes.c_int),
        ("max_seq", ctypes.c_int64),
        ("chunk", ctypes.c_int64),
        ("q8", ctypes.c_int),
        ("q8_assistant", ctypes.c_int),
        ("flash_prefill", ctypes.c_int),
        ("flash_decode", ctypes.c_int),
        ("vision_place", ctypes.c_int),
        ("max_patches", ctypes.c_int64),
        ("prewarm", ctypes.c_int),
        ("seal", ctypes.c_int),
        ("verbose", ctypes.c_int),
    ]


def open_error(err: str, max_seq: int) -> str:
    """Engine-open failure, with the knob that controls it named.

    The raw message is a byte count, and the first thing anyone needs to know
    is which flag makes it smaller."""
    msg = f"muse_open failed: {err}"
    low = err.lower()
    if "memory" in low or "alloc" in low:
        msg += (f"\n  The KV cache is sized by --max-seq (currently {max_seq}). "
                f"Retry with a smaller --max-seq, add --q8 to halve the weights, "
                f"use --vision cpu to keep the tower off the cards, or spread "
                f"across more of them with --gpus.")
    return msg


def _bind(lib: ctypes.CDLL) -> None:
    P, I64, I32, F32, F64 = (ctypes.c_void_p, ctypes.c_int64, ctypes.c_int32,
                             ctypes.c_float, ctypes.c_double)
    lib.muse_open.restype, lib.muse_open.argtypes = P, [ctypes.POINTER(_OpenOpts)]
    lib.muse_last_error.restype = ctypes.c_char_p
    lib.muse_close.argtypes = [P]
    for name in ("muse_vocab", "muse_sliding_window", "muse_spec_block", "muse_image_token",
                 "muse_video_token", "muse_hidden_size", "muse_patch_dim", "muse_merge_unit",
                 "muse_cache_len"):
        getattr(lib, name).restype, getattr(lib, name).argtypes = I64, [P]
    lib.muse_has_vision.restype, lib.muse_has_vision.argtypes = ctypes.c_int, [P]
    lib.muse_reset.argtypes = [P]
    lib.muse_set_cache_len.restype, lib.muse_set_cache_len.argtypes = ctypes.c_int, [P, I64]
    lib.muse_free_mem.restype, lib.muse_free_mem.argtypes = I64, [P, ctypes.c_int]
    lib.muse_forward.restype = I64
    lib.muse_forward.argtypes = [P, ctypes.POINTER(I32), I64, ctypes.POINTER(F32)]
    lib.muse_forward_all.restype = I64
    lib.muse_forward_all.argtypes = [P, ctypes.POINTER(I32), I64, ctypes.POINTER(F32)]
    lib.muse_draft.restype = I64
    lib.muse_draft.argtypes = [P, I32, ctypes.POINTER(I32), I64]
    lib.muse_vision_features.restype = I64
    lib.muse_vision_features.argtypes = [P, ctypes.POINTER(F64), ctypes.POINTER(I64), I64,
                                         ctypes.POINTER(F32), I64]
    lib.muse_set_vision_embeds.restype = ctypes.c_int
    lib.muse_set_vision_embeds.argtypes = [P, ctypes.POINTER(F32), I64, ctypes.POINTER(I64)]
    lib.muse_timings.argtypes = [P, ctypes.POINTER(F64)]


class Engine:
    def __init__(self, model: str, *, revision: str = "main", assistant: str | None = None,
                 gpus: int = 2, shards: int = 0, max_seq: int = 32768, chunk: int = 512,
                 q8: bool = False, q8_assistant: bool = False, flash_prefill: bool = True,
                 flash_decode: bool = True, vision: int = VISION_CPU,
                 max_patches: int = 4096, prewarm: bool = True, seal: int = 2,
                 verbose: bool = False, lib: str | None = None):
        self.L = L = ctypes.CDLL(_find_lib(lib))
        _bind(L)
        o = _OpenOpts(
            model=model.encode(), revision=revision.encode(),
            assistant=assistant.encode() if assistant else None,
            gpus=gpus, shards=shards, max_seq=max_seq, chunk=chunk,
            q8=int(q8), q8_assistant=int(q8_assistant), flash_prefill=int(flash_prefill),
            flash_decode=int(flash_decode), vision_place=vision, max_patches=max_patches,
            prewarm=int(prewarm), seal=seal, verbose=int(verbose))
        self.h = L.muse_open(ctypes.byref(o))
        if not self.h:
            raise RuntimeError(open_error(L.muse_last_error().decode(), max_seq))
        self.max_seq = max_seq
        self.chunk = chunk
        self.gpus = gpus
        self.vocab = int(L.muse_vocab(self.h))
        self.window = int(L.muse_sliding_window(self.h))
        self.spec_block = int(L.muse_spec_block(self.h))
        self.hidden = int(L.muse_hidden_size(self.h))
        self.image_token = int(L.muse_image_token(self.h))
        self.video_token = int(L.muse_video_token(self.h))
        self.has_vision = bool(L.muse_has_vision(self.h))
        self.patch_dim = int(L.muse_patch_dim(self.h))
        self.merge_unit = int(L.muse_merge_unit(self.h)) or 4
        self._logits = np.empty(self.vocab, dtype=np.float32)
        self._all = np.empty(self.vocab * max(1, self.spec_block), dtype=np.float32)
        # The token ids currently in the cache, so the next request can find
        # its longest common prefix without re-deriving one from the text.
        self.cached_ids: list[int] = []

    # ------------------------------------------------------------------ state
    def close(self) -> None:
        if getattr(self, "h", None):
            self.L.muse_close(self.h)
            self.h = None

    def __del__(self):
        self.close()

    @property
    def cache_len(self) -> int:
        return int(self.L.muse_cache_len(self.h))

    def reset(self) -> None:
        """Drop the KV cache. Deliberately does NOT touch the bound vision
        features: reset() is called from inside reuse_prefix(), so clearing
        them here silently undid the tower run that had just happened and the
        model saw random embeddings at the image positions. Media binding is
        the caller's to manage, and serve.server.Runner does it per request."""
        self.L.muse_reset(self.h)
        self.cached_ids = []

    def free_mem(self) -> list[int]:
        return [int(self.L.muse_free_mem(self.h, d)) for d in range(self.gpus)]

    def timings(self) -> dict:
        buf = (ctypes.c_double * 5)()
        self.L.muse_timings(self.h, buf)
        return {"upload_s": buf[0], "prefill_s": buf[1], "decode_s": buf[2],
                "prefill_tokens": int(buf[3]), "decode_tokens": int(buf[4])}

    # ---------------------------------------------------------------- forward
    def _ids_arg(self, ids):
        a = np.ascontiguousarray(ids, dtype=np.int32)
        return a, a.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))

    def forward(self, ids, logits: np.ndarray | None = None) -> np.ndarray:
        """Append ids at the current cache length; returns the LAST row's
        logits (a view of a reused buffer)."""
        out = self._logits if logits is None else logits
        arr, ptr = self._ids_arg(ids)
        n = self.L.muse_forward(self.h, ptr, arr.size,
                                out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        if n < 0:
            raise RuntimeError(self.L.muse_last_error().decode())
        self.cached_ids.extend(int(x) for x in arr)
        return out

    def forward_all(self, ids) -> np.ndarray:
        """Verification pass: [n, vocab] logits, and the cache holds all n."""
        arr, ptr = self._ids_arg(ids)
        if arr.size > max(1, self.spec_block):
            raise ValueError(f"forward_all over {arr.size} rows, engine holds {self.spec_block}")
        n = self.L.muse_forward_all(self.h, ptr, arr.size,
                                    self._all.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        if n < 0:
            raise RuntimeError(self.L.muse_last_error().decode())
        self.cached_ids.extend(int(x) for x in arr)
        return self._all[:arr.size * self.vocab].reshape(arr.size, self.vocab)

    def draft(self, anchor: int) -> list[int]:
        """One DFlash round against the taps the last forward captured."""
        cap = max(1, self.spec_block)
        out = np.empty(cap, dtype=np.int32)
        m = self.L.muse_draft(self.h, int(anchor),
                              out.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), cap)
        if m < 0:
            raise RuntimeError(self.L.muse_last_error().decode())
        return [int(x) for x in out[:m]]

    def rollback(self, n: int) -> None:
        """Drop the cache back to n tokens (speculative rejection, or prefix
        reuse). See reuse_prefix for the sliding-window bound."""
        if self.L.muse_set_cache_len(self.h, int(n)) != 0:
            raise RuntimeError(self.L.muse_last_error().decode())
        del self.cached_ids[n:]

    # ----------------------------------------------------------- prefix reuse
    def safe_rollback_depth(self) -> int:
        """How far back the cache can be rewound and still be *usable*.

        The 39 sliding layers hold a ring of `window + chunk` rows, so after
        writing up to L the rows for positions below L - (window + chunk) have
        been overwritten by newer ones. Appending at position P then needs
        [P - window, P) intact, which holds only while P >= L - chunk.

        So: rewinding by more than one chunk is not a slower path, it is a
        WRONG one, and the caller re-prefills instead. (The 13 global layers
        are linear and would have been fine; the sliding ones are the bound.)
        """
        return self.chunk if self.window > 0 else self.cache_len

    def reuse_prefix(self, ids: list[int]) -> int:
        """Set the cache up to serve `ids`, reusing what is already there.

        Returns the number of leading tokens now in the cache; the caller
        forwards ids[that:]. Never reuses the whole prompt — the last token
        has to go through a forward to produce logits.
        """
        keep = 0
        limit = min(len(self.cached_ids), len(ids) - 1)
        while keep < limit and self.cached_ids[keep] == ids[keep]:
            keep += 1
        if keep == 0:
            self.reset()
            return 0
        # Rewinding further than the ring can serve would attend over rows
        # that newer positions have already overwritten: silently wrong, not
        # slow. Re-prefill instead.
        if self.cache_len - keep > self.safe_rollback_depth():
            self.reset()
            return 0
        self.rollback(keep)
        return keep

    # ---------------------------------------------------------------- vision
    def vision_features(self, pixels: np.ndarray, grids: list[tuple[int, int, int]]) -> np.ndarray:
        """pixels f64 [npatch, patch_dim] -> features f32 [npatch/merge, hidden]."""
        px = np.ascontiguousarray(pixels, dtype=np.float64)
        g = np.ascontiguousarray(np.array(grids, dtype=np.int64).reshape(-1, 3))
        npatch = int(sum(int(t) * int(h) * int(w) for t, h, w in g))
        rows = npatch // self.merge_unit
        out = np.empty(rows * self.hidden, dtype=np.float32)
        m = self.L.muse_vision_features(
            self.h, px.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            g.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)), g.shape[0],
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), out.size)
        if m < 0:
            raise RuntimeError(self.L.muse_last_error().decode())
        return out.reshape(rows, self.hidden)

    def set_vision_embeds(self, feats: np.ndarray | None, positions: list[int] | None) -> None:
        """Bind features to absolute token positions for subsequent forwards.
        None clears — which every request must do when it finishes, or the next
        one inherits an image at whatever positions this one used."""
        if feats is None or positions is None or len(positions) == 0:
            self.L.muse_set_vision_embeds(self.h, None, 0, None)
            return
        f = np.ascontiguousarray(feats, dtype=np.float32).reshape(-1)
        p = np.ascontiguousarray(positions, dtype=np.int64)
        if f.size != p.size * self.hidden:
            raise ValueError(f"{p.size} positions but {f.size} feature floats")
        if self.L.muse_set_vision_embeds(
                self.h, f.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), p.size,
                p.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))) != 0:
            raise RuntimeError(self.L.muse_last_error().decode())
