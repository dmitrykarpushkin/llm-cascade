# llm-cascade

A **cost-aware LLM cascade** for turning a high-volume text feed into a short, useful summary: a classifier
that costs about a cent per thousand decisions throws away obvious noise, a small model scores what is left in
large batches, and a large model writes only from the best candidates.

It is extracted from a daily digest that reads hundreds of posts a day from public news channels; the same
shape works for support tickets, alerts, reviews, or any inbox where most items do not matter.

## Why

Sending everything to the best model is slow and expensive; sending everything to a small model gives a
mediocre summary. A cascade spends money where it matters:

| Stage | Model | Sees | Job | Typical share of tokens |
|---|---|---|---|---|
| 0. Prefilter | Classification API (~$0.04 per 1M input tokens, output free) | every unique item | drop confident noise | all |
| 1. Filter | small LLM (Haiku) | what survived, in ~60k-char batches | topic, score 1-5, upcoming date, one-line gist, as JSON | most |
| 2. Writer | large LLM (Opus) | only gists with score ≥ 3 | the summary | small |

Design choices worth noting:

- **Fail open, never fail wrong.** If the classifier is down, has no key, or returns errors for some rows, those
  items simply go on to the small model, so an outage only makes a run more expensive. Low-confidence "noise"
  is kept too (`drop_confidence`).
- **Batching where it is free.** Several questions about the same text go in one classifier request (about the
  same price as one); the filter packs posts into large chunks by a character budget.
- **Tolerant parsing.** Model output is parsed for the first JSON array; malformed items, out-of-range indexes
  and non-integer scores are dropped instead of crashing the run.
- **Measured.** Every run returns usage: items in and out of each stage, number of filter calls, characters
  sent, classifier cost.
- **Exit code contract.** `cascade-decide` prints `unavailable` and exits with code 2 when the classifier
  cannot answer, so shell scripts can fall back to a model or a rule.

## Architecture

```
 feed (JSONL: source, text, url)
   │ drop blanks, dedupe reposts
   ▼
 Classifier.decide_batch ── same text → one request ── Unavailable → keep everything
   │ drop "noise" with confidence ≥ 0.5
   ▼
 Cascade.filter ── chunk_lines(60k chars) ── LLM.complete(FILTER_PROMPT, "haiku") ── parse_json_array
   │ keep score ≥ min_score, best first
   ▼
 Cascade.write ── LLM.complete(WRITER_PROMPT, "opus")   (skipped when nothing qualifies)
   ▼
 Result(text, candidates, usage)
```

| Module | Role |
|---|---|
| `classifier.py` | HTTP client for the classification API: yes/no and pick-one, batch grouping, one retry for retryable errors, cost accounting |
| `llm.py` | `LLM` protocol, `ClaudeCLI` backend (`claude -p` with all tools disabled), `parse_json_array` |
| `cascade.py` | The pipeline and its config (`CascadeConfig`: interests, topics, thresholds, models, prompts) |
| `fakes.py` | Keyword-rule stand-ins for every stage, so the demo and tests run offline |
| `cli.py` | `cascade-decide` and `cascade-digest` |

## Run

```bash
pip install -e .
cascade-digest examples/posts.jsonl --usage        # offline demo, no keys, no network
```

With real models (stages 1-2 use the `claude` CLI and its own login; stage 0 is skipped without a key):

```bash
cp .env.example .env    # CASCADE_CLASSIFIER_API_KEY, models, thresholds
set -a; . ./.env; set +a
cascade-digest my-feed.jsonl --live --usage
```

As a library:

```python
from llm_cascade import Cascade, CascadeConfig, Classifier
from llm_cascade.llm import ClaudeCLI

cfg = CascadeConfig(interests="1. our product's outages  2. competitor launches", min_score=4)
result = Cascade(ClaudeCLI(), Classifier.from_env(), cfg).run(items)
print(result.text, result.usage)
```

Any object with `complete(prompt, model) -> str` can replace `ClaudeCLI` (an SDK client, a local model).

## Measured

One day of a real feed: 300 posts sampled from 108 public Telegram channels (5 Oct 2026), production prompts,
every model through `claude -p` (cost is the CLI's API-equivalent USD). The reference is the large model scoring
every post 0-5 in batches of 25, the careful and expensive way: 32 posts at 3+, 14 important at 4+. It is a model
reference, not human labels. Script: [`eval/run_eval.py`](eval/run_eval.py), numbers:
[`eval/report-2026-10-05.json`](eval/report-2026-10-05.json).

| Variant | Filter vs reference (P / R / F1) | Important links in the digest | Cost | Time |
|---|---|---|---|---|
| **cascade** (Haiku filter → Opus writer) | 0.58 / 0.59 / 0.59 | 9 of 14 | **$0.32** | 163 s |
| all Haiku | same filter | 6 of 14 | $0.29 | 253 s |
| all Opus | 0.73 / 0.75 / 0.74 | 8 of 14 | $1.11 | 137 s |

- The cascade costs 3.4x less than all-Opus and its digest carries as many important stories (9 vs 8 of 14);
  the writer is where the large model pays off (Haiku as writer: 6 of 14).
- The weak spot is the small-model filter: two identical Haiku runs agreed on only 42% of selected posts
  (33 vs 52 selected). The union of two runs raises recall to 0.81 at precision 0.43.
- Stage 0 (the classifier API) was not run: an estimated $0.002 for these 300 posts (about 55k input tokens at $0.042 per 1M).
- One day, one run per writer, a model reference: read as a direction, not a benchmark.

**Can the filter be made stable?** ([`eval/stability.py`](eval/stability.py),
[`eval/stability-2026-10-05.json`](eval/stability-2026-10-05.json); two runs per variant, Opus writer on top)

| Haiku filter variant | Run-to-run overlap | Filter F1 | Important links in the digest | Cost per digest |
|---|---|---|---|---|
| 60k chars per call, list worth-knowing posts (current) | 0.42 | 0.59 / 0.55 | 9, 8, 9, 8 of 14 (4 writer runs) | $0.26-0.39 |
| 15k chars per call, list | 0.71 | 0.57 / 0.51 | 6, 6 of 14 | $0.69-0.74 |
| 15k chars per call, score every post | 0.64 | 0.52 / 0.53 | 6, 4 of 14 | $0.50-0.62 |
| 6k chars per call, score every post | 0.67 | 0.48 / 0.47 | 6, 7 of 14 | $0.66-0.92 |

Smaller batches make the filter agree with itself more often and catch more of the reference (recall up to 0.97),
but it then passes 60-100 candidates instead of ~30, the writer drowns, and the digest loses important stories,
at 2-3x the cost (each `claude -p` call also carries a fixed overhead). The current setting stays: its selection
wobbles on borderline posts, yet the digest holds at 8-9 of 14 across four writer runs.


## Example

`examples/posts.jsonl` is a made-up feed of 15 posts: news, a repost, pump spam, a giveaway, small talk and a
blank post. The offline demo (keyword rules standing in for the models) gives:

```text
$ cascade-digest examples/posts.jsonl --usage
{"classifier_in": 13, "classifier_kept": 10, "classifier_errors": 0, "classifier_cost_usd": 0.000169,
 "filter_calls": 1, "filter_chars": 1114, "items": 14, "unique": 13, "sources": 8, "candidates": 10, "kept": 9}
Top
- Bitcoin ETF inflows hit a record $1.2 billion in a single day as BTC tests $118k — Market Wire https://example.com/mw/101
- Exchange hack: an invented exchange, Nimbus, halts withdrawals after a $40M hot wallet exploit — Chain Daily https://example.com/cd/56
- Central bank signals a rate cut on Thursday after CPI came in at 2.1%, the lowest in four years. (soon: on Thursday) — Macro Notes https://example.com/mn/12
- …

Also
- Ceasefire talks resume next week in Geneva; both delegations confirmed attendance at the summit. (soon: next week) — World Desk https://example.com/wd/77
- …
```

The blank post and the repost are gone before any model runs; the pump post, the giveaway and "good morning"
never reach the small model; the weekend-market post scores 2 and never reaches the writer.

Single decisions against the live classifier (values illustrative):

```bash
$ cascade-decide "Is this about crypto?" --text "BTC broke 100k"
{"choice": "yes", "confidence": 0.94, "p_yes": 0.97, "model": "…", "cost_usd": 4e-07}
$ echo "buy milk tomorrow" | cascade-decide "What kind of note?" -o idea,task,junk
{"choice": "task", "confidence": 0.9, "probabilities": {"idea": 0.04, "task": 0.95, "junk": 0.01}, …}
$ CASCADE_CLASSIFIER_API_KEY= cascade-decide "q" --text "t"; echo $?
unavailable
2
```

## Development

```bash
pip install -e ".[dev]"
ruff check . && ruff format --check .
pytest -q
```

The classifier tests replace the HTTP call with a scripted fake (retries, 4xx vs 5xx, malformed payloads, batch
grouping and cost split); the pipeline tests run the whole cascade on the example feed with the offline fakes,
plus a down classifier, malformed model output, chunking and empty feeds.
