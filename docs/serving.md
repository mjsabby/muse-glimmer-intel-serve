# The serving stack

```bash
source /opt/intel/oneapi/setvars.sh
cmake --build build-gpu --target muse-intel-serve      # libmuse-intel-serve.so

ZES_ENABLE_SYSMAN=1 .venv/bin/python -m serve.server \
    --model meta-models/Muse-Glimmer-30B \
    --assistant meta-models/Muse-Glimmer-30B-assistant --q8-assistant \
    --gpus 2 --max-seq 8192 --vision cpu --port 8123
```

Three protocols over one engine: OpenAI Chat Completions, OpenAI Responses,
Anthropic Messages, plus raw `/v1/completions`, `/v1/models`, `/health` and
`/metrics`. `ZES_ENABLE_SYSMAN=1` is what lets the driver report free VRAM;
without it the startup line and `/health` say so rather than print a zero.

The split is the siblings': **the engine owns weights, caches and the forward
pass; Python owns tokenization, chat rendering, sampling, grammars, protocols
and the speculative accept rule.** Logits come back as f32 over the C ABI —
808 KiB per step at this vocabulary, which at decode rates is nothing — and
everything above them is easier to change in Python.

## One request at a time

`serve.server.Runner` holds a lock for the whole of every request. That is not
a placeholder for a batching scheduler; it is the honest shape of a deployment
whose entire memory design is a **static allocation**. Concurrency would mean
either a second KV cache or two requests interleaved through one, and both give
up the guarantee that makes the footprint checkable — see
[gpu.md's prewarm section](gpu.md#prewarm-and-why-the-seal-alone-was-not-enough).

## The channel protocol, and why the parser reads token ids

Muse Glimmer does not emit one string with markers in it. It emits a sequence
of **channels**, each addressed to a recipient:

```
<|start|>assistant to=self<|message|>      thinking out loud             <|eom|>
<|start|>assistant to=user<|message|>      the answer                    <|eot|>
<|start|>assistant to=web.search<|message|> <atem:function_calls>…       <|eot|>
```

`<|eom|>` closes a channel and the turn continues; `<|eot|>` ends the turn.
Two consequences, both load-bearing:

* **`reasoning_content` cannot leak into `content`.** They are different
  channels, not different regions of one string, so there is no heuristic to
  get wrong and no `<think>`-style splitter to write. `<|eom|>` is deliberately
  **not** a stop token — stopping on it would truncate the turn to its
  reasoning.
* **The parser runs on token ids, not text** (`serve/atem.py`). All four
  structural markers are single vocabulary tokens, which removes every
  partial-marker holdback problem a text-level parser has. Only the recipient
  and the body are ordinary text.

The ATEM tool-call regexes are read out of `tokenizer_config.json`'s
`response_template` rather than written into the source, and `validate()` fails
at startup if a future revision changes the shape this module assumes. Every
one of those assumptions failing produces *plausible output* rather than an
error — reasoning shown as the answer, a tool call rendered as prose — which is
exactly the kind of thing that must not be discovered in production.

## `tool_choice` is a grammar, not a nudge

The generation prompt is a bare `<|start|>assistant` and the model writes the
recipient itself. So `tool_choice` is a constraint on a handful of tokens at a
known position (`serve/recipient.py`):

| `tool_choice` | recipients the grammar admits |
|---|---|
| `auto` (default) | unconstrained — `mask()` returns `None` and the sampler pays nothing |
| `none` | `self`, `user` — the model may reason and must answer; no tool is addressable |
| `required` | `self` + every declared tool, with `self` available only for the FIRST channel, so the turn cannot reason forever instead of calling |
| `{"function": {"name": X}}` | `self` + `X`, same rule |

This is a better deal than the sibling engines get. With a `<|tool_call>`-opener
FSM you can force the model to *start* a call and then watch it invent a name
that was never declared (gemma4 documents exactly that failure). Here the name
**is** the channel address, so `required` is a hard guarantee rather than a
best effort. Matching is byte-level against the vocabulary's piece strings, so
every BPE split of a legal recipient is admissible — assuming one canonical
tokenization is the trap this avoids — and a legal recipient is always
reachable through single-byte pieces, so the forced region cannot dead-end.

Live gate: force `get_time` on a prompt that asks about the weather, and check
the model calls `get_time`.

## Rendering

Through the checkpoint's own `chat_template.jinja` via `apply_chat_template`,
so the prompt is byte-exact by construction. Three things the wrapper has to
do around it:

* **Deserialize `tool_calls[].function.arguments`.** The wire format is a JSON
  *string*; the template calls `args.items()` and `raise_exception`s on
  anything that is not a mapping. Without this every agentic second turn 400s —
  a failure that looks like a model problem and is not.
* **Repair tool schemas.** A property with no `type` is inferred rather than
  passed through for the model to guess at.
* **Reasoning strength.** `low | medium | high | xhigh`, default `high`.
  `--no-thinking` sets `low` rather than removing the line, because the
  template always emits one. A system prompt that already carries its own
  "Reasoning effort:" line wins — the template normalizes it to "strength" and
  suppresses the kwarg line.

## Guided JSON

`json_object` and `json_schema` run the byte-level PDA already in
`src/json_grammar.h` / `src/json_schema.h`, reached through the same shared
library. The mask is swept over 202048 candidates per step in C++ (~0.3 ms);
Python only compiles the schema and builds the per-tokenizer piece table once.

The constraint applies **only inside the answer channel**: the model still has
to be able to spell ` to=user<|message|>` to get there, and its reasoning is
not the JSON the caller asked for.

One thing worth knowing, because the failure is silent and instructive: a mask
can only *remove* options. Asked "Tell me about Lisbon" under a `json_schema`
constraint, the model wants to write prose, every prose token is masked, and
the highest-scoring **legal** token is a space — so it emits spaces until
`max_tokens`. The schema therefore also goes into the prompt, and the grammar
stays as the guarantee rather than as the instruction.

## Speculative decoding, and what it actually guarantees

One drafter round proposes 15 tokens; one target forward over
`[anchor] + proposals` verifies them all. The accept rule is the standard one
for a **deterministic** draft (the draft distribution is a point mass on the
proposal): accept with probability `p_target(proposal)`, and on rejection draw
from the target with that token's mass removed. It is distribution-preserving,
and at temperature ≤ 0 it collapses to "accept iff the proposal is the target's
argmax".

Measured, BF16 target + Q8 drafter, two B70s, 8192 context:

| prompt | plain | speculative | accept rate |
|---|---:|---:|---:|
| a list of primes | 15 tok/s | **98 tok/s** | 0.56 |
| a python function | 14 tok/s | **55 tok/s** | 0.29 |
| two sentences of prose | 15 tok/s | **31 tok/s** | 0.13 |

**It is not bit-identical to plain decoding, and it cannot be.** A verified
token's logits come from a 16-row forward — oneDNN matmul, tile-softmax
attention — while a plain decode's come from a 1-row forward — hand-written
GEMV, split-K attention. Those are different arithmetic, both inside the twin's
envelope, neither of them "the" answer. Making them identical would mean
verifying 16 tokens with 16 separate decode steps, which is the non-speculative
path.

So the gate checks the property that can hold: every divergence must be a
**tie**. Measured across the gate's prompts, each first divergence had one of
the two paths at an *exact* tie (0.0000 logprob gap between the top two
candidates) while the other saw 0.06–0.25. A real acceptance bug diverges at a
comfortable margin, and that is what the gate fails on.

## Cross-turn prefix reuse

The cache is part of the input. A follow-up turn rolls back to the longest
common prefix and appends only the tail (measured: 66 of 110 prompt tokens
reused on a second turn).

The rollback bound is not a performance choice. The 39 sliding layers hold a
ring of `window + chunk` rows, so appending at position P needs
`[P - window, P)` intact, which holds only while `P ≥ L - chunk`. Rewinding
further is not slower, it is **wrong**, so the engine re-prefills instead.

Two consequences to state plainly:

* Reuse is gated as *correct*: the answer from a reused cache equals the answer
  from a cold one.
* But reuse changes the prefill chunking, and the fast attention tiers are
  envelope-level rather than bitwise, so at a near-tie the same request can
  land differently depending on how much of it was already resident.
  `--no-prefix-reuse` makes the answer a function of the prompt alone, at the
  cost of re-prefilling every turn. The gates ask for "same state, same
  answer", which is the property a server can actually hold.

## Media

`image_url` and `video_url` content parts, `data:` / `http(s):` / `file:`, a
64 MiB cap, and a content-addressed decode cache so an agent re-sending the
same screenshot every turn pays for it once. Preprocessing runs the
checkpoint's own `MuseGlimmerImageProcessor` — a `TorchvisionBackend` whose
`resample: 1` is torchvision's antialiased LANCZOS, and only on
torchvision ≥ 0.27 (below that it silently substitutes BICUBIC).

`--vision gpu` keeps the tower resident (~1.86 GiB/card, 0.5 s per image);
`--vision cpu` keeps it out of VRAM entirely and pays ~3–13 s per image on the
host, and is the bitwise-gated path. `--vision off` refuses media.

`input_audio` returns a capability error. This model has no audio tower, no
audio config and no audio tensors; answering anyway would be worse than saying
so.

A media request always drops the cache first: two different images tokenize to
the same placeholders, so a text-level cache hit would serve the previous
picture.

## Endpoints

| endpoint | notes |
|---|---|
| `POST /v1/chat/completions` | SSE with `stream: true`; `reasoning_content` in the delta; `tool_calls` emitted whole at channel close |
| `POST /v1/completions` | the prompt is used **verbatim** — no chat template, no generation prompt |
| `POST /v1/responses` | typed output items (`reasoning`, `message`, `function_call`), which fit this model better than a marked-up string; `previous_response_id` is refused rather than silently answered without the context it names |
| `POST /v1/messages` | Anthropic blocks; `thinking` and `reasoning_content` are the same `to=self` channel |
| `POST /v1/messages/count_tokens` | renders the prompt and counts it |
| `GET /v1/models`, `/health`, `/metrics` | `/health` reports free VRAM, cache length, and whether speculation and prefix reuse are on |

Sampling: temperature / top-k / top-p / min-p / repetition, frequency and
presence penalties / seed / logit bias / logprobs / stop strings, with defaults
taken from the checkpoint's `generation_config.json` (1.0 / 0.95 / 64) rather
than from some other server's habits. Stops on both eos ids — 200001
`<|end_of_text|>` and 200008 `<|eot|>`.

`--api-key`, `--cors-origins`, `--trace-dir`, and cancel-on-disconnect: a
cancelled agent turn stops costing GPU time as soon as the client goes away.

## Gates

```bash
.venv/bin/python -m tests.serve_tests                       # 41 checks, no GPU
.venv/bin/python -m tests.live_api_tests --url http://127.0.0.1:8123 \
    --image some.png                                        # 30 checks, on the box
```

`serve_tests.py` fakes the engine with a **deterministic target addressed by
cache position** — a point mass on `script[L - prompt_len]` — rather than a
cursor advanced per call. That distinction is the test: a call-counting fake
passes the speculative accept path and quietly diverges on the reject path,
which is the one that needs testing.

`live_api_tests.py` covers what only the weights can show: the channel split,
determinism from a known cache state, the speculative tie property above,
`tool_choice` forcing a recipient the model did not want, guided JSON parsing,
streamed output equalling non-streamed, prefix reuse being both a speed-up and
the same answer, both other protocols, an image changing the answer, and the
errors that must stay errors.
