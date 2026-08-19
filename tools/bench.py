"""Throughput benchmark, shaped so it can be put next to `llama-bench`.

    .venv/bin/python tools/bench.py --gpus 2 -p 512,2048 -n 128 -r 3
    .venv/bin/python tools/bench.py --q8 -d 0,4096,16384 -n 64 --json out/bench.json

Test names follow llama-bench's: `ppN` is prefilling N tokens into an empty
cache, `tgN` is generating N tokens, and `-d D` runs either against a cache
already holding D tokens. Reported as mean ± stddev over `-r` repetitions, of
tokens per second, exactly as llama-bench reports them.

Two deliberate differences from llama-bench, both stated rather than hidden:

  * `tg` here includes the logits coming back to the host, because this
    engine's forward always delivers them (llama.cpp's does too) — but NOT the
    host-side sampler, which is timed separately and reported as `sample_ms`.
    The `serve` column adds it back, and is what a served token actually costs.
  * A `spec` row measures the DFlash drafter on REAL text, because acceptance
    depends on what is being written. llama.cpp has no equivalent for this
    model, so that row has no counterpart — it is reported, not compared.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from serve import chatlib                                        # noqa: E402
from serve.atem import ResponseTemplate                          # noqa: E402
from serve.engine import Engine, VISION_OFF                      # noqa: E402
from serve.generate import GenParams, Generation                 # noqa: E402
from serve.sampler import SampleParams                           # noqa: E402


def ints(spec: str) -> list[int]:
    return [int(x) for x in str(spec).split(",") if x.strip()]


def stats(xs: list[float]) -> tuple[float, float]:
    return (statistics.mean(xs), statistics.stdev(xs) if len(xs) > 1 else 0.0)


class Bench:
    def __init__(self, eng: Engine, tok, filler: list[int]):
        self.eng = eng
        self.tok = tok
        # A real token stream rather than a repeated id: repeated ids are not
        # slower or faster here (nothing is content-dependent below the
        # sampler) but they make a wrong result harder to notice.
        self.filler = filler

    def tokens(self, n: int) -> list[int]:
        out = []
        while len(out) < n:
            out.extend(self.filler)
        return out[:n]

    def prime(self, depth: int) -> None:
        """Put `depth` tokens in the cache, untimed."""
        self.eng.reset()
        if depth:
            self.eng.forward(self.tokens(depth))

    def pp(self, n: int, depth: int) -> float:
        self.prime(depth)
        ids = self.tokens(n)
        t0 = time.perf_counter()
        self.eng.forward(ids)
        return n / (time.perf_counter() - t0)

    def tg(self, n: int, depth: int) -> tuple[float, float, float]:
        """(tok/s excluding host sampling, tok/s including it, sample ms)."""
        self.prime(max(depth, 1))
        logits = self.eng.forward(self.tokens(1)) if depth == 0 else self.eng._logits
        fwd = 0.0
        smp = 0.0
        tid = int(np.argmax(logits))
        for _ in range(n):
            t0 = time.perf_counter()
            logits = self.eng.forward([tid])
            t1 = time.perf_counter()
            tid = int(np.argmax(logits))
            t2 = time.perf_counter()
            fwd += t1 - t0
            smp += t2 - t1
        return n / fwd, n / (fwd + smp), 1e3 * smp / n


def spec_row(eng, tok, rt, prompt: str, max_tokens: int) -> dict:
    """The speculative path on real text, through the serving loop."""
    ids = chatlib.render_ids(tok, [{"role": "user", "content": prompt}], reasoning="low")
    eng.reset()
    gen = Generation(eng, tok, rt,
                     GenParams(max_tokens=max_tokens, speculative=True,
                               sample=SampleParams(temperature=0.0)),
                     eos=chatlib.eos_ids(tok, MODEL))
    out = {}
    for ev in gen.run(ids):
        if ev["type"] == "done":
            out = {"tok_s": ev["usage"]["tokens_per_second"],
                   "tokens": ev["usage"]["completion_tokens"],
                   **(ev["spec"] or {})}
    return out


MODEL = "meta-models/Muse-Glimmer-30B"


def main() -> int:
    global MODEL
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--assistant", default=None)
    ap.add_argument("--gpus", type=int, default=2)
    ap.add_argument("--shards", type=int, default=0,
                    help="tensor-parallel shards (default: one per card). --gpus 1 --shards 2 "
                         "puts both shards on one card, which is what --flash-prefill needs "
                         "when there is only one: with 32 q heads over 2 KV heads, a single "
                         "shard spans both GQA groups and the tier refuses")
    ap.add_argument("--q8", action="store_true")
    ap.add_argument("--q8-assistant", action="store_true")
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--max-seq", type=int, default=None)
    ap.add_argument("--no-flash-prefill", action="store_true")
    ap.add_argument("--no-flash-decode", action="store_true")
    ap.add_argument("-p", "--prompt-sizes", default="512,2048")
    ap.add_argument("-n", "--gen-sizes", default="128")
    ap.add_argument("-d", "--depths", default="0")
    ap.add_argument("-r", "--repeats", type=int, default=3)
    ap.add_argument("--spec-prompt", default=None,
                    help="also measure the speculative path on this prompt")
    ap.add_argument("--spec-tokens", type=int, default=256)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    MODEL = a.model

    tok = chatlib.load_tokenizer(a.model)
    rt = ResponseTemplate.load(a.model)
    pp_sizes, tg_sizes, depths = ints(a.prompt_sizes), ints(a.gen_sizes), ints(a.depths)
    need = max([0] + [d + max(pp_sizes + tg_sizes + [0]) + 64 for d in depths])
    max_seq = a.max_seq or max(2048, 1 << (need - 1).bit_length())

    eng = Engine(a.model, assistant=a.assistant, gpus=a.gpus, shards=a.shards,
                 max_seq=max_seq,
                 chunk=a.chunk, q8=a.q8, q8_assistant=a.q8_assistant,
                 flash_prefill=not a.no_flash_prefill,
                 flash_decode=not a.no_flash_decode, vision=VISION_OFF, verbose=True)
    filler = chatlib.encode(tok, "The quick brown fox jumps over the lazy dog. " * 8)
    b = Bench(eng, tok, filler)

    tier = "Q8_0" if a.q8 else "BF16"
    rows = []
    label = f"{a.gpus}" + (f"x{a.shards}sh" if a.shards and a.shards != a.gpus else "")
    print(f"\n| model | tier | cards | test | t/s |")
    print(f"|---|---|---:|---|---:|")
    for d in depths:
        for n in pp_sizes:
            xs = [b.pp(n, d) for _ in range(a.repeats)]
            m, s = stats(xs)
            name = f"pp{n}" + (f" @ d{d}" if d else "")
            rows.append({"test": name, "tier": tier, "gpus": a.gpus, "mean": m, "stdev": s})
            print(f"| Muse Glimmer 30B | {tier} | {label} | {name} | {m:.2f} ± {s:.2f} |")
        for n in tg_sizes:
            fwd, srv, ms = [], [], []
            for _ in range(a.repeats):
                f, sv, msec = b.tg(n, d)
                fwd.append(f)
                srv.append(sv)
                ms.append(msec)
            m, s = stats(fwd)
            sm, _ = stats(srv)
            name = f"tg{n}" + (f" @ d{d}" if d else "")
            rows.append({"test": name, "tier": tier, "gpus": a.gpus, "mean": m, "stdev": s,
                         "served_mean": sm, "sample_ms": stats(ms)[0]})
            print(f"| Muse Glimmer 30B | {tier} | {label} | {name} | {m:.2f} ± {s:.2f} "
                  f"|  _(served {sm:.2f}, sampler {stats(ms)[0]:.2f} ms/tok)_")

    if a.spec_prompt and eng.spec_block > 1:
        r = spec_row(eng, tok, rt, a.spec_prompt, a.spec_tokens)
        rows.append({"test": "spec", "tier": tier, "gpus": a.gpus, **r})
        print(f"| Muse Glimmer 30B | {tier} | {label} | spec ({r['tokens']} tok) | "
              f"{r['tok_s']:.2f} | _accept {r.get('spec_accept_rate', 0):.2f}_")

    free = eng.free_mem()
    print(f"\nfree VRAM after the sweep: "
          + ", ".join(f"card {i} {f / 2**30:.2f} GiB" for i, f in enumerate(free)))
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(
            {"model": a.model, "tier": tier, "gpus": a.gpus, "chunk": a.chunk,
             "max_seq": max_seq, "flash_prefill": not a.no_flash_prefill,
             "flash_decode": not a.no_flash_decode, "rows": rows,
             "free_vram": free}, indent=1))
        print(f"wrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
