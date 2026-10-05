"""Small-model filter stability: run each filter variant twice on the same posts and compare.

  python eval/stability.py posts.jsonl --prompts my_prompts.py --gold results/gold.json --out results/stab

Variants: batch size (characters per call) x prompt ("list posts worth knowing" vs "score every post 0-5").
Per variant: run-to-run Jaccard overlap of the selected set, precision / recall / F1 against the reference
labels from run_eval.py, recall of important posts, cost and time. The best variant then feeds the large-model
writer, to check that important links in the digest hold up.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_eval import GOLD_PROMPT, add, claude, lines, load_prompts, prf, scores_of, write

from llm_cascade.cascade import chunk_lines
from llm_cascade.llm import parse_json_array

SCORE_ALL = """{rules}

For EVERY post below return one object, in a JSON array and nothing else (posts not worth knowing get score 0):
{{"n": N, "topic": "{topics}", "score": 0-5, "soon": "date and event or null", "gist": "the point in one sentence"}}

=== POSTS ===
{material}
"""


def run_filter(posts, prompts, model, chunk_chars, item_chars, mode):
    acc, raw = {}, []
    rules = getattr(prompts, "FILTER_RULES", "")
    for c in chunk_lines(lines(posts, item_chars), chunk_chars):
        if mode == "list":
            prompt = prompts.FILTER_PROMPT.format(
                interests=prompts.INTERESTS, topics=prompts.TOPICS, material="".join(c)
            )
        else:
            prompt = SCORE_ALL.format(rules=rules, topics=prompts.TOPICS or "topic", material="".join(c))
        text, u = claude(prompt, model)
        add(acc, u)
        raw += parse_json_array(text)
    return raw, acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("posts")
    ap.add_argument("--prompts")
    ap.add_argument("--gold", required=True)
    ap.add_argument("--out", default="eval/results/stab")
    ap.add_argument("--model", default="haiku")
    ap.add_argument("--writer", default="opus")
    ap.add_argument("--variants", default="60000:list,15000:list,15000:score,6000:score")
    ap.add_argument("--min-score", type=int, default=3)
    ap.add_argument("--item-chars", type=int, default=1200)
    a = ap.parse_args()
    _ = GOLD_PROMPT  # same reference as run_eval.py
    prompts = load_prompts(a.prompts)
    posts = [json.loads(x) for x in Path(a.posts).read_text().splitlines() if x.strip()]
    labels = {int(k): v for k, v in json.loads(Path(a.gold).read_text())["labels"].items()}
    ref = {i for i, s in labels.items() if s >= a.min_score}
    hi = {i for i, s in labels.items() if s >= 4}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    def sel(raw):
        return {i for i, s in scores_of(raw, len(posts)).items() if s >= a.min_score}

    report, raws = {}, {}
    for v in a.variants.split(","):
        chunk, mode = v.split(":")
        runs = []
        for k in range(2):
            f = out / f"{chunk}-{mode}-{k}.json"
            if f.exists():
                raw, cost = json.loads(f.read_text())
            else:
                raw, cost = run_filter(posts, prompts, a.model, int(chunk), a.item_chars, mode)
                f.write_text(json.dumps([raw, cost], ensure_ascii=False))
            runs.append((raw, cost))
        s0, s1 = sel(runs[0][0]), sel(runs[1][0])
        report[v] = {
            "overlap": round(len(s0 & s1) / len(s0 | s1), 3) if s0 | s1 else 1.0,
            "runs": [
                {
                    **prf(sel(r), ref),
                    "important_recall": prf(sel(r), hi)["recall"],
                    "usd": c["usd"],
                    "sec": c["sec"],
                    "calls": c["calls"],
                }
                for r, c in runs
            ],
        }
        raws[v] = runs
        print(v, json.dumps(report[v]), flush=True)

    best = max(report, key=lambda v: report[v]["overlap"] + sum(r["f1"] for r in report[v]["runs"]) / 2)
    sources = len({p.get("source") for p in posts})
    report["best"] = best
    report["digests"] = []
    for k, (raw, cost) in enumerate(raws[best]):
        text, wcost = write(posts, raw, prompts, a.writer, a.min_score, sources)
        (out / f"digest-best-{k}.txt").write_text(text)
        urls = [posts[i].get("url") for i in hi if posts[i].get("url")]
        report["digests"].append(
            {
                "important_links_in_text": f"{sum(u in text for u in urls)}/{len(urls)}",
                "total_usd": round(cost["usd"] + wcost["usd"], 4),
                "chars": len(text),
            }
        )
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
