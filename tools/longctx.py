"""Long-context retrieval: does the model still find one sentence at depth?

    .venv/bin/python tools/longctx.py --url http://127.0.0.1:8123 \
        --lengths 4096,16384,65536 --depths 0.05,0.5,0.95

Everything else in this repo measures long context as *speed* — it prefills, it
decodes, the footprint holds. None of that says the answer is still in there.
This asks: a distinctive sentence is placed at a known fraction of the way
through a haystack, and the model is asked for it back.

The needle is randomized per run (a word and a number drawn at call time), so
a pass cannot come from the checkpoint having memorized the prompt, and the
question names the needle's subject without quoting its content.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
import urllib.request

PASS, FAIL = "\033[32mPASS\033[0m", "\033[31mFAIL\033[0m"

WORDS = ["cinnabar", "harbourmaster", "quicklime", "windlass", "saltpetre",
         "pergola", "kingfisher", "millrace", "thimble", "cartwright"]

# Deliberately dull, deliberately repetitive, and numbered so that no two
# paragraphs are identical: identical filler would let a model answer from the
# nearest copy rather than from the position the needle is actually at.
FILLER = ("Paragraph {i}. The survey party continued along the eastern bank, recording the "
          "depth of the channel at regular intervals and noting the condition of the "
          "embankment where the older stonework had been repointed. The weather remained "
          "settled. Provisions were adequate. The clerk copied the day's readings into the "
          "second ledger before the light went, as he had done on every previous day of the "
          "expedition, and the boats were made fast for the night.\n\n")


def post(url: str, body: dict, timeout: float = 3600) -> dict:
    req = urllib.request.Request(url + "/v1/chat/completions", method="POST",
                                 data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def count_tokens(url: str, text: str, timeout: float = 600) -> int:
    req = urllib.request.Request(url + "/v1/messages/count_tokens", method="POST",
                                 data=json.dumps({"model": "m", "messages": [
                                     {"role": "user", "content": text}]}).encode(),
                                 headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return int(json.loads(r.read())["input_tokens"])


def haystack(n_para: int, needle: str, depth: float) -> str:
    at = max(0, min(n_para - 1, int(round(depth * (n_para - 1)))))
    out = []
    for i in range(n_para):
        if i == at:
            out.append(needle + "\n\n")
        out.append(FILLER.format(i=i + 1))
    return "".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8123")
    ap.add_argument("--lengths", default="4096,16384",
                    help="target prompt lengths in tokens")
    ap.add_argument("--depths", default="0.05,0.5,0.95",
                    help="where in the haystack the needle goes, as a fraction")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    lengths = [int(x) for x in a.lengths.split(",") if x.strip()]
    depths = [float(x) for x in a.depths.split(",") if x.strip()]

    # Calibrate paragraphs -> tokens by DIFFERENCE: counting one paragraph
    # includes the chat template's own ~50 tokens, which would shrink every
    # haystack by a third.
    one = count_tokens(a.url, FILLER.format(i=1))
    ten = count_tokens(a.url, "".join(FILLER.format(i=i) for i in range(1, 11)))
    per_para = max(1, (ten - one) // 9)
    print(f"filler paragraph = {per_para} tokens (template overhead {one - per_para})")

    rc, rows = 0, []
    for n_tok in lengths:
        n_para = max(4, n_tok // per_para)
        for depth in depths:
            word = rng.choice(WORDS)
            number = rng.randrange(1000, 9999)
            needle = (f"NOTE FROM THE HARBOUR OFFICE: the clearance code for the Douro "
                      f"archive is {word}-{number}. Do not record it in the ledger.")
            body = haystack(n_para, needle, depth)
            prompt = (body + "\n\nUsing only the text above, what is the clearance code "
                      "for the Douro archive? Answer with the code alone.")
            t0 = time.perf_counter()
            r = post(a.url, {"model": "m", "temperature": 0, "max_tokens": 256,
                             "reasoning_effort": "low",
                             "messages": [{"role": "user", "content": prompt}]})
            dt = time.perf_counter() - t0
            msg = r["choices"][0]["message"]
            # The answer counts wherever the model put it: a code recalled in
            # the reasoning channel and then not repeated is still retrieval,
            # and scoring only `content` would call that a miss.
            got = ((msg.get("content") or "") + " " + (msg.get("reasoning_content") or "")).lower()
            u = r["usage"]
            ok = word in got and str(number) in got
            rc |= 0 if ok else 1
            rows.append({"tokens": u["prompt_tokens"], "depth": depth, "found": ok,
                         "seconds": round(dt, 2), "answer": got.strip()[:60]})
            print(f"  {PASS if ok else FAIL} {u['prompt_tokens']:>6} tokens, needle at "
                  f"{depth:>4.0%}  ({dt:5.1f} s)  -> {got.strip()[:48]!r}")
    if a.json:
        from pathlib import Path
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(rows, indent=1))
    found = sum(1 for r in rows if r["found"])
    print(f"retrieved {found}/{len(rows)}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
