"""The cascade: cheap classifier -> small model filter -> large model writer.

  items --dedupe--> classifier prefilter --> small-model filter (chunked) --> score >= min --> writer
         (repost)   drop confident "noise"     JSON: topic, score, gist      top N         one summary
                    unavailable: keep all

Each stage sees less text than the one before and costs more per token, so most of the volume is handled by
the cheapest stage, and any stage being down only makes the run more expensive, never wrong.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field

from .classifier import Classifier, Unavailable
from .llm import LLM, parse_json_array
from .prompts import FILTER_PROMPT, WRITER_PROMPT


@dataclass
class CascadeConfig:
    interests: str = "1. crypto and markets  2. geopolitics and macro  3. AI and tech"
    topics: list[str] = field(default_factory=lambda: ["markets", "geopolitics", "macro", "ai_tech", "other"])
    sections: str = "Markets\nup to 6 items.\n\nGeopolitics and macro\nup to 4 items.\n\nOther\n1-2 lines."
    prefilter_question: str = (
        "Topic of this post for a news feed? noise = ad, giveaway, referral, 'easy money', automated "
        "pump or listing signal, meme, motivation, personal, empty."
    )
    noise_label: str = "noise"
    drop_confidence: float = 0.5  # classifier confidence in "noise" needed to drop a post
    filter_model: str = "haiku"
    writer_model: str = "opus"
    chunk_chars: int = 60_000  # post text per filter call
    item_chars: int = 1200  # per post, for the classifier and the filter
    min_score: int = 3
    max_candidates: int = 300
    filter_prompt: str = FILTER_PROMPT
    writer_prompt: str = WRITER_PROMPT
    empty_text: str = "Nothing important today."

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None, **kw) -> CascadeConfig:
        env = os.environ if env is None else env
        d = cls()
        return cls(
            filter_model=env.get("CASCADE_FILTER_MODEL", d.filter_model),
            writer_model=env.get("CASCADE_WRITER_MODEL", d.writer_model),
            min_score=int(env.get("CASCADE_MIN_SCORE", d.min_score)),
            drop_confidence=float(env.get("CASCADE_DROP_CONFIDENCE", d.drop_confidence)),
            **kw,
        )


@dataclass
class Result:
    text: str
    candidates: list[dict]
    usage: dict


def dedupe(items: list[dict], key_chars: int = 300) -> list[dict]:
    """Drop items whose text (first `key_chars`) was already seen: reposts across sources."""
    seen, out = set(), []
    for it in items:
        k = (it.get("text") or "").strip()[:key_chars]
        if k and k not in seen:
            seen.add(k)
            out.append(it)
    return out


def chunk_lines(lines: list[str], budget: int) -> list[list[str]]:
    chunks, cur, size = [], [], 0
    for line in lines:
        if size + len(line) > budget and cur:
            chunks.append(cur)
            cur, size = [], 0
        cur.append(line)
        size += len(line)
    if cur:
        chunks.append(cur)
    return chunks


class Cascade:
    def __init__(self, llm: LLM, classifier: Classifier | None = None, config: CascadeConfig | None = None):
        self.llm, self.classifier, self.cfg = llm, classifier, config or CascadeConfig()

    def prefilter(self, items: list[dict]) -> tuple[list[dict], dict]:
        """Drop what the classifier is confident is noise. Anything it cannot judge stays for the next stage."""
        if self.classifier is None or not items:
            return items, {"classifier": "off"}
        options = [*self.cfg.topics, self.cfg.noise_label]
        rows = [
            {
                "question": self.cfg.prefilter_question,
                "options": options,
                "text": (it["text"] or "")[: self.cfg.item_chars],
            }
            for it in items
        ]
        try:
            results = self.classifier.decide_batch(rows)
        except Unavailable as e:  # whole stage down: the small model sees everything
            return items, {"classifier": f"unavailable: {e}"}
        kept, cost, errors = [], 0.0, 0
        for it, x in zip(items, results, strict=True):
            errors += "error" in x
            cost += x.get("cost_usd") or 0
            if x.get("choice") == self.cfg.noise_label and (x.get("confidence") or 0) >= self.cfg.drop_confidence:
                continue
            kept.append({**it, "prefilter_topic": x.get("choice")})
        return kept, {
            "classifier_in": len(items),
            "classifier_kept": len(kept),
            "classifier_errors": errors,
            "classifier_cost_usd": round(cost, 6),
        }

    def filter(self, items: list[dict]) -> tuple[list[dict], dict]:
        lines = [
            f"#{i} [{it.get('source', '')}] {(it['text'] or '')[: self.cfg.item_chars]}\n" for i, it in enumerate(items)
        ]
        chunks = chunk_lines(lines, self.cfg.chunk_chars)
        out = []
        for c in chunks:
            prompt = self.cfg.filter_prompt.format(
                interests=self.cfg.interests, topics="|".join(self.cfg.topics), material="".join(c)
            )
            for x in parse_json_array(self.llm.complete(prompt, self.cfg.filter_model)):
                n = x.get("n")
                if isinstance(n, int) and 0 <= n < len(items):
                    out.append({**x, "item": items[n]})
        return out, {"filter_calls": len(chunks), "filter_chars": sum(map(len, lines))}

    def select(self, candidates: list[dict]) -> list[dict]:
        def score(c):
            try:
                return int(c.get("score") or 0)
            except (TypeError, ValueError):
                return 0

        good = [c for c in candidates if score(c) >= self.cfg.min_score]
        return sorted(good, key=lambda c: -score(c))[: self.cfg.max_candidates]

    def write(self, good: list[dict], total: int, sources: int) -> str:
        if not good:
            return self.cfg.empty_text
        material = "\n".join(
            f"- [{c.get('topic')}|{c.get('score')}] {c.get('gist')}"
            + (f" (soon: {c['soon']})" if c.get("soon") else "")
            + f" — {c['item'].get('source', '')} {c['item'].get('url') or ''}".rstrip()
            for c in good
        )
        prompt = self.cfg.writer_prompt.format(
            sections=self.cfg.sections, n=len(good), total=total, sources=sources, material=material
        )
        return self.llm.complete(prompt, self.cfg.writer_model)

    def run(self, items: list[dict]) -> Result:
        items = [it for it in items if (it.get("text") or "").strip()]
        sources = len({it.get("source") for it in items})
        uniq = dedupe(items)
        pre, usage = self.prefilter(uniq)
        if pre:
            candidates, fusage = self.filter(pre)
        else:
            candidates, fusage = [], {"filter_calls": 0, "filter_chars": 0}
        usage.update(fusage)
        good = self.select(candidates)
        text = self.write(good, len(items), sources)
        usage.update(items=len(items), unique=len(uniq), sources=sources, candidates=len(candidates), kept=len(good))
        return Result(text=text, candidates=good, usage=usage)
