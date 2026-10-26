# Evaluation: cascade vs all-small vs all-large

Back to the [README](../README.md).


One day of a real feed: 300 posts sampled from 108 public Telegram channels (5 Oct 2026), production prompts,
every model through `claude -p` (cost is the CLI's API-equivalent USD). The reference is the large model scoring
every post 0-5 in batches of 25, the careful and expensive way: 32 posts at 3+, 14 important at 4+. It is a model
reference, not human labels. Script: [`eval/run_eval.py`](../eval/run_eval.py), numbers:
[`eval/report-2026-10-05.json`](../eval/report-2026-10-05.json).

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

**Can the filter be made stable?** ([`eval/stability.py`](../eval/stability.py),
[`eval/stability-2026-10-05.json`](../eval/stability-2026-10-05.json); two runs per variant, Opus writer on top)

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


