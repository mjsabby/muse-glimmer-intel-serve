#!/usr/bin/env python
"""Tokenize a prompt with the model's own tokenizer and print token ids
(comma-separated) for feeding both the C++ oracle and ref_forward.py."""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import resolve_model  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--prompt", default=None)
    p.add_argument("--prompt-file", default=None)
    p.add_argument("--chat", action="store_true", help="wrap in chat template + generation prompt")
    p.add_argument("--show", action="store_true", help="also print the decoded pieces")
    p.add_argument("--out", default=None, help="write ids to file instead of stdout")
    a = p.parse_args()

    from transformers import AutoTokenizer

    text = a.prompt if a.prompt is not None else open(a.prompt_file).read()
    tok = AutoTokenizer.from_pretrained(resolve_model(a.model), local_files_only=True)

    if a.chat:
        templated = tok.apply_chat_template(
            [{"role": "user", "content": text}], add_generation_prompt=True, tokenize=False
        )
        # template text already contains <bos> etc.
        ids = tok(templated, add_special_tokens=False)["input_ids"]
    else:
        ids = tok(text)["input_ids"]
    if ids and isinstance(ids[0], list):
        ids = ids[0]

    s = ",".join(str(i) for i in ids)
    if a.out:
        with open(a.out, "w") as f:
            f.write(s + "\n")
        print(f"wrote {len(ids)} ids to {a.out}")
    else:
        print(s)
    if a.show:
        print([tok.decode([i]) for i in ids], file=sys.stderr)


if __name__ == "__main__":
    main()
