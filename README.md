# llm-cascade

**English** · [Русский](README.ru.md)

[![CI](https://github.com/dmitrykarpushkin/llm-cascade/actions/workflows/ci.yml/badge.svg)](https://github.com/dmitrykarpushkin/llm-cascade/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

**Turn a noisy feed into a short digest for a third of the price of sending everything to the best model.**

For anyone who summarises a stream where most items do not matter: news channels, support tickets, alerts,
reviews. A cheap classifier drops obvious noise, a small model scores the rest in big batches, and the large model
only writes the final text from the best candidates.

```text
$ cascade-digest examples/posts.jsonl --usage
{"classifier_in": 13, "classifier_kept": 10, ..., "filter_calls": 1, "items": 14, "unique": 13, "candidates": 10, "kept": 9}
Top
- Bitcoin ETF inflows hit a record $1.2 billion in a single day as BTC tests $118k — Market Wire https://example.com/mw/101
- Exchange hack: an invented exchange, Nimbus, halts withdrawals after a $40M hot wallet exploit — Chain Daily https://example.com/cd/56
- ...
```

*Offline demo on a made-up 15-post feed: the blank post and the repost are gone before any model runs, the spam
and the giveaway never reach the small model.*

## Why use it

- **3.4x cheaper than all-Opus, same digest quality.** On one day of a real feed (300 posts) the cascade cost
  $0.32 against $1.11 and kept as many important stories (9 of 14 vs 8 of 14).
- **An outage makes a run more expensive, never wrong.** If the classifier is down or unsure, items go on to the
  model instead of being dropped.
- **Survives messy model output.** The JSON from the model is parsed tolerantly; broken items are skipped, the run
  goes on.
- **Every run reports its own cost.** Items in and out of each stage, model calls, characters sent, classifier spend.
- **Runs without keys.** Keyword fakes stand in for every stage, so the demo and all 28 tests work offline.

## How it compares

Measured on the same 300 posts, 5 Oct 2026, against a reference made by the large model scoring every post
([details](docs/evaluation.md)):

| | All small model (Haiku) | All large model (Opus) | **llm-cascade** |
|---|---|---|---|
| Important stories in the digest | 6 of 14 | 8 of 14 | **9 of 14** |
| Cost per digest | $0.29 | $1.11 | **$0.32** |
| Time | 253 s | 137 s | 163 s |

Pick all-large if you need the most accurate filter and cost does not matter (its filter F1 is 0.74 vs 0.59).
Pick a general router such as RouteLLM or a gateway with fallbacks if your requests are independent questions;
this repo is for the batch case, where many items compete for a few places in one text.

## Quick start

```bash
git clone https://github.com/dmitrykarpushkin/llm-cascade && cd llm-cascade
pip install -e .
cascade-digest examples/posts.jsonl --usage        # offline demo, no keys, no network
```

With real models: copy `.env.example` to `.env`, load it, and add `--live`. Stages 1 and 2 call the `claude` CLI
with its own login; stage 0 is skipped without a classifier key. Any object with `complete(prompt, model) -> str`
can replace the CLI backend (an SDK client, a local model).

## How it works

```text
feed (JSONL) -> drop blanks, dedupe reposts
             -> stage 0: classifier API drops confident noise      (about a cent per thousand decisions)
             -> stage 1: small model scores items 1-5 in ~60k-char batches
             -> stage 2: large model writes the digest from items scored 3+
```

Stages, design choices, the library API and the `cascade-decide` CLI are in [docs/design.md](docs/design.md).

## Status and limits

Extracted from a daily digest that reads hundreds of posts a day from public news channels; this repo is the
reusable core with offline fakes and tests. The numbers above are one day, one run per writer, and a reference
made by a model rather than people: read them as a direction, not a benchmark. The weak spot is the small-model
filter: two identical runs agreed on only 42% of selected posts ([stability study](docs/evaluation.md)).

## Development

```bash
pip install -e ".[dev]"
ruff check . && ruff format --check .
pytest -q
```

---

License: [MIT](LICENSE) · Author: [Dmitry Karpushkin](https://github.com/dmitrykarpushkin)
