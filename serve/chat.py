"""A streaming chat REPL against the engine, with no HTTP in the way.

    python -m serve.chat --gpus 2 --assistant meta-models/Muse-Glimmer-30B-assistant
    python -m serve.chat --prompt "one-shot question"

Needs the oneAPI environment (source setvars.sh) for libmuse-intel-serve.so.
Reasoning is shown dimmed and separately, because it IS separate: it arrives on
the `to=self` channel, not as a marked-up region of the answer.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from serve import chatlib                                        # noqa: E402
from serve.atem import ResponseTemplate                          # noqa: E402
from serve.engine import Engine, VISION_CPU, VISION_GPU, VISION_OFF  # noqa: E402
from serve.generate import GenParams, Generation                 # noqa: E402
from serve.sampler import SampleParams                           # noqa: E402

DIM, RESET, BOLD = "\033[2m", "\033[0m", "\033[1m"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="meta-models/Muse-Glimmer-30B")
    ap.add_argument("--assistant", default=None)
    ap.add_argument("--gpus", type=int, default=2)
    ap.add_argument("--max-seq", type=int, default=16384)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--q8", action="store_true")
    ap.add_argument("--q8-assistant", action="store_true")
    ap.add_argument("--vision", choices=["gpu", "cpu", "off"], default="off")
    ap.add_argument("--system", default=None)
    ap.add_argument("--reasoning", choices=list(chatlib.REASONING_LEVELS), default="high")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-k", type=int, default=64)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--no-spec", action="store_true")
    ap.add_argument("--hide-reasoning", action="store_true")
    ap.add_argument("--prompt", default=None, help="one-shot: ask this and exit")
    a = ap.parse_args()

    tok = chatlib.load_tokenizer(a.model)
    rt = ResponseTemplate.load(a.model)
    eos = chatlib.eos_ids(tok, a.model)
    place = {"gpu": VISION_GPU, "cpu": VISION_CPU, "off": VISION_OFF}[a.vision]
    eng = Engine(a.model, assistant=a.assistant, gpus=a.gpus, max_seq=a.max_seq,
                 chunk=a.chunk, q8=a.q8, q8_assistant=a.q8_assistant, vision=place,
                 verbose=True)
    print(f"{BOLD}{a.model}{RESET} | max_seq {a.max_seq} | "
          f"{'Q8' if a.q8 else 'BF16'} | spec {'on' if eng.spec_block > 1 else 'off'} "
          f"| reasoning {a.reasoning}   (/exit, /reset, /reasoning LEVEL)")

    messages: list[dict] = []
    if a.system:
        messages.append({"role": "system", "content": a.system})
    reasoning = a.reasoning

    def turn(text: str) -> None:
        nonlocal messages
        messages.append({"role": "user", "content": text})
        params = GenParams(max_tokens=a.max_tokens, speculative=not a.no_spec,
                           sample=SampleParams(temperature=a.temperature, top_k=a.top_k,
                                               top_p=a.top_p, seed=a.seed))
        ids = chatlib.render_ids(tok, messages, reasoning=reasoning)
        gen = Generation(eng, tok, rt, params, eos=eos)
        answer, thinking, calls, in_reason = "", "", [], False
        for ev in gen.run(ids):
            if ev["type"] == "reasoning":
                thinking += ev["text"]
                if not a.hide_reasoning:
                    if not in_reason:
                        print(DIM, end="")
                        in_reason = True
                    print(ev["text"], end="", flush=True)
            elif ev["type"] == "content":
                if in_reason:
                    print(RESET, end="")
                    in_reason = False
                answer += ev["text"]
                print(ev["text"], end="", flush=True)
            elif ev["type"] == "tool_calls":
                calls.extend(ev["calls"])
                for c in ev["calls"]:
                    print(f"\n{BOLD}[tool]{RESET} {c['function']['name']}"
                          f"({c['function']['arguments']})")
            elif ev["type"] == "done":
                if in_reason:
                    print(RESET, end="")
                u, sp = ev["usage"], ev["spec"]
                extra = (f", accept {sp['spec_accept_rate']:.2f}" if sp else "")
                print(f"\n{DIM}[{u['completion_tokens']} tok, "
                      f"{u['tokens_per_second']:.1f} tok/s, prefill "
                      f"{u['prefill_s']:.2f}s ({u['prefill_reused']} reused){extra}]{RESET}")
        msg = {"role": "assistant", "content": answer}
        if thinking:
            msg["reasoning_content"] = thinking
        if calls:
            msg["tool_calls"] = [{"id": f"call_{i}", **c} for i, c in enumerate(calls)]
        messages.append(msg)

    if a.prompt:
        turn(a.prompt)
        return
    while True:
        try:
            line = input(f"\n{BOLD}>{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not line:
            continue
        if line in ("/exit", "/quit"):
            return
        if line == "/reset":
            messages = [m for m in messages if m["role"] == "system"]
            eng.reset()
            print("(context cleared)")
            continue
        if line.startswith("/reasoning"):
            want = line.split(maxsplit=1)[-1].strip()
            if want in chatlib.REASONING_LEVELS:
                reasoning = want
                print(f"(reasoning strength: {reasoning})")
            else:
                print(f"(levels: {', '.join(chatlib.REASONING_LEVELS)})")
            continue
        if line == "/messages":
            print(json.dumps(messages, indent=1)[:4000])
            continue
        turn(line)


if __name__ == "__main__":
    main()
