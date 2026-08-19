"""Host sampler over the [vocab] f32 logit vector the engine returns.

Filter order matches llama.cpp/irun:
  logit_bias -> penalties -> (grammar mask) -> top_k -> top_p -> min_p
  -> temperature -> sample
temperature <= 0 means greedy (first-max ties, like the engine's top2).
Pure numpy, seeded, unit-tested (tests/serve_tests.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SampleParams:
    temperature: float = 0.7
    top_k: int = 64          # <= 0 keeps all
    top_p: float = 0.95      # >= 1 keeps all
    min_p: float = 0.0       # 0 keeps all
    repeat_penalty: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    repeat_last_n: int = 64
    seed: int | None = None
    logit_bias: dict[int, float] = field(default_factory=dict)
    top_logprobs: int = 0    # >0: report that many alternatives per token


@dataclass
class TokenLogprobs:
    token_id: int
    logprob: float
    top: list[tuple[int, float]]  # (id, logprob) alternatives, desc


class Sampler:
    def __init__(self, params: SampleParams):
        self.p = params
        self.rng = np.random.default_rng(params.seed)

    def _transformed(self, logits: np.ndarray, recent_ids: list[int],
                     mask: np.ndarray | None = None):
        """Shared filter pipeline. Returns (lg, cand, probs, kind) where lg is
        the biased/penalized (pre-truncation) logit vector, cand/probs the
        final candidate distribution (probs sums to 1), and kind one of
        "greedy" (temperature <= 0: point mass, no RNG), "masked" (fully
        masked fallback: point mass on raw argmax), "dist" (sample from it)."""
        p = self.p
        lg = logits.astype(np.float32, copy=True)

        for tid, bias in p.logit_bias.items():
            if 0 <= tid < lg.size:
                lg[tid] += np.float32(bias)

        if recent_ids and (p.repeat_penalty != 1.0 or p.frequency_penalty != 0.0
                           or p.presence_penalty != 0.0):
            window = recent_ids[-p.repeat_last_n:] if p.repeat_last_n > 0 else recent_ids
            ids, counts = np.unique(np.asarray(window, dtype=np.int64), return_counts=True)
            if p.repeat_penalty != 1.0:
                v = lg[ids]
                lg[ids] = np.where(v > 0, v / p.repeat_penalty, v * p.repeat_penalty)
            lg[ids] -= p.frequency_penalty * counts.astype(np.float32)
            lg[ids] -= p.presence_penalty

        if mask is not None:
            lg[~mask] = -np.inf

        if not np.isfinite(lg).any():
            # A malformed/finished grammar should normally never expose an
            # empty mask, but fail open instead of crashing logsumexp.
            tid = int(np.argmax(logits))
            return lg, np.array([tid]), np.array([1.0], dtype=np.float32), "masked"

        if p.temperature <= 0:  # greedy, first-max tie semantics
            tid = int(np.argmax(lg))
            return lg, np.array([tid]), np.array([1.0], dtype=np.float32), "greedy"

        # Candidate set. top_k<=0 means ALL tokens, exactly. The common
        # top_k=64 path only sorts 64 values; the all-token/no-top-p path needs
        # no sort at all. We pay a full sort only when exact nucleus sampling
        # over the full vocabulary was explicitly requested.
        k = min(p.top_k, lg.size) if p.top_k > 0 else 0
        if k:
            cand = np.argpartition(-lg, k - 1)[:k]
            cl = lg[cand]
            order = np.argsort(-cl, kind="stable")
            cand, cl = cand[order], cl[order]
        else:
            cand = np.arange(lg.size)
            cl = lg
            if p.top_p < 1.0:
                order = np.argsort(-cl, kind="stable")
                cand, cl = cand[order], cl[order]
        # drop -inf (masked) candidates
        keep = cl > -np.inf
        cand, cl = cand[keep], cl[keep]
        if cand.size == 0:  # fully masked: fall back to global argmax of bias'd logits
            tid = int(np.argmax(logits))
            return lg, np.array([tid]), np.array([1.0], dtype=np.float32), "masked"

        probs = _softmax(cl)
        if p.top_p < 1.0:
            n = int(np.searchsorted(np.cumsum(probs), p.top_p) + 1)
            cand, cl, probs = cand[:n], cl[:n], probs[:n]
        if p.min_p > 0.0:
            keep = probs >= p.min_p * probs.max()
            cand, cl = cand[keep], cl[keep]

        final = _softmax(cl / np.float32(p.temperature))
        return lg, cand, final / final.sum(), "dist"

    def sample(self, logits: np.ndarray, recent_ids: list[int],
               mask: np.ndarray | None = None) -> tuple[int, TokenLogprobs | None]:
        want_lp = self.p.top_logprobs > 0
        lg, cand, probs, kind = self._transformed(logits, recent_ids, mask)
        if kind == "greedy":
            tid = int(cand[0])
            return tid, self._logprobs(lg, tid) if want_lp else None
        if kind == "masked":
            return int(cand[0]), None
        idx = int(self.rng.choice(cand.size, p=probs))
        tid = int(cand[idx])
        return tid, self._logprobs(lg, tid) if want_lp else None

    def dist(self, logits: np.ndarray, recent_ids: list[int],
             mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """The final candidate distribution sample() draws from: (ids, probs),
        probs summing to 1 (a point mass when greedy/fully-masked). The spec
        Leviathan accept rule evaluates target and draft rows through this so
        both sides see the identical filter pipeline."""
        _lg, cand, probs, _kind = self._transformed(logits, recent_ids, mask)
        return cand, probs

    def logprobs_for(self, logits: np.ndarray, recent_ids: list[int],
                     mask: np.ndarray | None, chosen: int) -> TokenLogprobs | None:
        """Logprobs for a token this sampler did not itself draw.

        Speculative decoding commits most of its tokens through the accept
        rule, not through sample(); without this, `logprobs: true` reports one
        entry per ROUND instead of one per token, which looks like the model
        emitted 8x fewer tokens than it did."""
        if self.p.top_logprobs <= 0:
            return None
        lg, _cand, _probs, _kind = self._transformed(logits, recent_ids, mask)
        return self._logprobs(lg, chosen)

    def pick(self, cand: np.ndarray, probs: np.ndarray) -> int:
        """Draw from a dist() result using the sampler's RNG (the same stream
        sample() consumes, so seeded runs stay reproducible)."""
        return int(cand[int(self.rng.choice(cand.size, p=probs))])

    def _logprobs(self, lg: np.ndarray, chosen: int) -> TokenLogprobs:
        k = min(max(self.p.top_logprobs, 1), lg.size)
        lp = lg - _logsumexp(lg)
        top = np.argpartition(-lp, k - 1)[:k]
        top = top[np.argsort(-lp[top], kind="stable")]
        return TokenLogprobs(chosen, float(lp[chosen]),
                             [(int(i), float(lp[i])) for i in top])


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def _logsumexp(x: np.ndarray) -> np.float32:
    finite = x[np.isfinite(x)]
    m = finite.max()
    return m + np.log(np.exp(finite - m).sum())
