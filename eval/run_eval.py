"""Measure the cascade against "everything in the large model" and "everything in the small model".

  python eval/run_eval.py posts.jsonl --prompts my_prompts.py --out results/

Reference labels: the large model scores EVERY post 0-5 in small batches (GOLD_BATCH posts per call), so it
reads each post with full attention. That is the expensive way to do the job, and the thing the cascade has to
approximate. It is a model reference, not human labels: agreement with it says "as good as the large model",
not "right".

Variants, all through `claude -p` (subscription login, cost reported as the CLI's API-equivalent USD):
  cascade    small-model filter (big batches) -> large-model writer
  all-small  small-model filter -> small-model writer
  all-large  large-model filter (same big batches) -> large-model writer

Per variant: precision / recall / F1 of the selected set (score >= min) against the reference, recall of the
reference's important posts (score >= 4), share of important posts whose link reached the final text, cost and
wall time of each stage. The classifier prefilter (stage 0) is a paid API and is not run here.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import time
from pathlib import Path

from llm_cascade.cascade import chunk_lines
from llm_cascade.llm import parse_json_array
from llm_cascade.prompts import FILTER_PROMPT, WRITER_PROMPT

DISALLOWED = "Bash,Edit,Write,Read,Glob,Grep,WebFetch,WebSearch,Agent"
GOLD_BATCH = 25

GOLD_PROMPT = """You label posts for a news digest. Reader's interests and rules are the same as for the filter:

{filter_rules}

For EVERY post below return one object, in a JSON array and nothing else:
{{"n": N, "score": 0-5}}
0 = leave out (ads, giveaways, bots, memes, personal, repeats, or simply not worth knowing); 2 = minor;
3 = notable; 4 = significant; 5 = moves markets or a major event. Read each post carefully.

=== POSTS ===
{material}
"""


def claude(prompt: str, model: str) -> tuple[str, dict]:
    t = time.time()
    r = subprocess.run(
        [
            "claude",
            "-p",
            "--model",
            model,
            "--output-format",
            "json",
            "--disallowedTools",
            DISALLOWED,
            "--max-turns",
            "2",
        ],
        input=prompt,
        text=True,
        capture_output=True,
        timeout=1800,
    )
    if r.returncode != 0:
        raise RuntimeError(f"claude -p {model}: {(r.stderr or r.stdout)[-300:]}")
    d = json.loads(r.stdout)
    u = d.get("usage") or {}
    return d.get("result") or "", {
        "usd": d.get("total_cost_usd") or 0,
        "sec": round(time.time() - t, 1),
        "in": (u.get("input_tokens") or 0)
        + (u.get("cache_read_input_tokens") or 0)
        + (u.get("cache_creation_input_tokens") or 0),
        "out": u.get("output_tokens") or 0,
        "model": ",".join(d.get("modelUsage") or {}),
    }


def add(acc: dict, u: dict) -> dict:
    for k in ("usd", "sec", "in", "out"):
        acc[k] = round(acc.get(k, 0) + u[k], 4)
    acc["calls"] = acc.get("calls", 0) + 1
    acc["model"] = u["model"]
    return acc


def lines(posts, item_chars):
    return [f"#{i} [{p.get('source', '')}] {(p['text'] or '')[:item_chars]}\n" for i, p in enumerate(posts)]


def scores_of(raw: list[dict], n: int) -> dict[int, int]:
    out = {}
    for x in raw:
        try:
            i, s = int(x["n"]), int(x.get("score") or 0)
        except (TypeError, ValueError, KeyError):
            continue
        if 0 <= i < n:
            out[i] = max(out.get(i, 0), s)
    return out


def gold(posts, prompts, model, item_chars):
    rules = getattr(prompts, "FILTER_RULES", None) or prompts.FILTER_PROMPT.split("Below are posts")[0].format(
        interests=prompts.INTERESTS, topics=prompts.TOPICS
    )
    ls, acc, labels = lines(posts, item_chars), {}, {}
    for start in range(0, len(ls), GOLD_BATCH):
        part = ls[start : start + GOLD_BATCH]
        text, u = claude(GOLD_PROMPT.format(filter_rules=rules, material="".join(part)), model)
        add(acc, u)
        labels.update(scores_of(parse_json_array(text), len(posts)))
    return labels, acc


def filt(posts, prompts, model, chunk_chars, item_chars):
    acc, raw = {}, []
    for c in chunk_lines(lines(posts, item_chars), chunk_chars):
        text, u = claude(
            prompts.FILTER_PROMPT.format(interests=prompts.INTERESTS, topics=prompts.TOPICS, material="".join(c)),
            model,
        )
        add(acc, u)
        raw += parse_json_array(text)
    return raw, acc


def write(posts, raw, prompts, model, min_score, sources):
    good = sorted(
        [
            x
            for x in raw
            if isinstance(x.get("n"), int) and 0 <= x["n"] < len(posts) and _int(x.get("score")) >= min_score
        ],
        key=lambda x: -_int(x.get("score")),
    )
    material = "\n".join(
        f"- [{x.get('topic')}|{x.get('score')}] {x.get('gist')}"
        + (f" (soon: {x['soon']})" if x.get("soon") else "")
        + f" — {posts[x['n']].get('source', '')} {posts[x['n']].get('url') or ''}".rstrip()
        for x in good
    )
    text, u = claude(
        prompts.WRITER_PROMPT.format(n=len(good), total=len(posts), sources=sources, material=material), model
    )
    return text, add({}, u)


def _int(v):
    try:
        return int(v or 0)
    except (TypeError, ValueError):
        return 0


def prf(selected: set, ref: set) -> dict:
    tp = len(selected & ref)
    p = tp / len(selected) if selected else 0.0
    r = tp / len(ref) if ref else 0.0
    return {
        "selected": len(selected),
        "precision": round(p, 3),
        "recall": round(r, 3),
        "f1": round(2 * p * r / (p + r), 3) if p + r else 0.0,
    }


def load_prompts(path: str | None):
    if not path:

        class Default:
            pass

        d = Default()
        d.FILTER_PROMPT, d.WRITER_PROMPT = FILTER_PROMPT, WRITER_PROMPT
        d.INTERESTS = "1. crypto and markets  2. geopolitics and macro  3. AI and tech"
        d.TOPICS = "markets|geopolitics|macro|ai_tech|other"
        return d
    spec = importlib.util.spec_from_file_location("prompts", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for k, v in (("INTERESTS", ""), ("TOPICS", "")):
        if not hasattr(mod, k):
            setattr(mod, k, v)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("posts")
    ap.add_argument("--prompts", help="module with FILTER_PROMPT, WRITER_PROMPT (optional INTERESTS, TOPICS)")
    ap.add_argument("--out", default="eval/results")
    ap.add_argument("--small", default="haiku")
    ap.add_argument("--large", default="opus")
    ap.add_argument("--min-score", type=int, default=3)
    ap.add_argument("--chunk-chars", type=int, default=60_000)
    ap.add_argument("--item-chars", type=int, default=1200)
    ap.add_argument("--repeats", type=int, default=2, help="small-model filter runs, for stability")
    a = ap.parse_args()
    prompts = load_prompts(a.prompts)
    posts = [json.loads(x) for x in Path(a.posts).read_text().splitlines() if x.strip()]
    sources = len({p.get("source") for p in posts})
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "gold.json"
    if cache.exists():
        g = json.loads(cache.read_text())
        labels, gold_cost = {int(k): v for k, v in g["labels"].items()}, g["cost"]
    else:
        labels, gold_cost = gold(posts, prompts, a.large, a.item_chars)
        cache.write_text(json.dumps({"labels": labels, "cost": gold_cost}))
    ref = {i for i, s in labels.items() if s >= a.min_score}
    ref_hi = {i for i, s in labels.items() if s >= 4}
    report = {
        "posts": len(posts),
        "labelled": len(labels),
        "reference": {"selected": len(ref), "important": len(ref_hi), "cost": gold_cost},
        "variants": {},
    }

    small_runs = []
    for k in range(a.repeats):
        raw, cost = filt(posts, prompts, a.small, a.chunk_chars, a.item_chars)
        small_runs.append((raw, cost))
        (out / f"filter-small-{k}.json").write_text(json.dumps(raw, ensure_ascii=False))
    large_raw, large_cost = filt(posts, prompts, a.large, a.chunk_chars, a.item_chars)
    (out / "filter-large.json").write_text(json.dumps(large_raw, ensure_ascii=False))

    def sel(raw):
        return {i for i, s in scores_of(raw, len(posts)).items() if s >= a.min_score}

    for k, (raw, cost) in enumerate(small_runs):
        report["variants"][f"small-filter-run{k}"] = {
            **prf(sel(raw), ref),
            "important_recall": prf(sel(raw), ref_hi)["recall"],
            "cost": cost,
        }
    if len(small_runs) > 1:
        s0, s1 = sel(small_runs[0][0]), sel(small_runs[1][0])
        report["small_filter_run_overlap"] = round(len(s0 & s1) / len(s0 | s1), 3) if s0 | s1 else 1.0
    report["variants"]["large-filter"] = {
        **prf(sel(large_raw), ref),
        "important_recall": prf(sel(large_raw), ref_hi)["recall"],
        "cost": large_cost,
    }

    small_raw = small_runs[0][0]
    for name, raw, wmodel, fcost in (
        ("cascade", small_raw, a.large, small_runs[0][1]),
        ("all-small", small_raw, a.small, small_runs[0][1]),
        ("all-large", large_raw, a.large, large_cost),
    ):
        text, wcost = write(posts, raw, prompts, wmodel, a.min_score, sources)
        (out / f"digest-{name}.txt").write_text(text)
        urls_hi = [posts[i].get("url") for i in ref_hi if posts[i].get("url")]
        hit = sum(1 for u in urls_hi if u in text)
        report["variants"][name] = {
            "filter_cost": fcost,
            "writer_cost": wcost,
            "total_usd": round(fcost["usd"] + wcost["usd"], 4),
            "total_sec": round(fcost["sec"] + wcost["sec"], 1),
            "important_links_in_text": f"{hit}/{len(urls_hi)}",
            "chars": len(text),
        }
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
