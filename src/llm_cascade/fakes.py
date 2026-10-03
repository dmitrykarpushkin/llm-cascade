"""Deterministic offline stand-ins for every stage, used by the demo and the tests.

They follow the same contracts as the real backends (prompt in, text out; batch rows in, answers out), driven
by keyword rules instead of models, so the whole cascade runs without network or keys.
"""

from __future__ import annotations

import json
import re

NOISE = ("giveaway", "referral", "promo code", "guaranteed", "100x", "dm me", "good morning", "subscribe")
TOPICS = {
    "markets": ("btc", "bitcoin", "eth", "etf", "liquidat", "stablecoin", "exchange", "token"),
    "geopolitics": ("sanction", "election", "ceasefire", "summit", "tariff", "border"),
    "macro": ("rate", "inflation", "cpi", "fed ", "central bank", "jobs report"),
    "ai_tech": ("model", "gpu", "open-source", "chip", "agent"),
}
STRONG = ("hack", "record", "cut", "approved", "ban", "halt", "billion")


def topic_of(text: str) -> str:
    t = text.lower()
    if any(w in t for w in NOISE):
        return "noise"
    for topic, words in TOPICS.items():
        if any(w in t for w in words):
            return topic
    return "other"


class KeywordClassifier:
    """Stands in for `Classifier`: same `decide_batch` contract."""

    def __init__(self, confidence: float = 0.9, cost_per_row: float = 0.000013):
        self.confidence, self.cost_per_row = confidence, cost_per_row
        self.calls = 0

    def decide_batch(self, rows: list[dict]) -> list[dict]:
        self.calls += 1
        out = []
        for r in rows:
            choice = topic_of(r["text"])
            options = r.get("options") or ["yes", "no"]
            out.append(
                {
                    "choice": choice if choice in options else options[-1],
                    "confidence": self.confidence,
                    "cost_usd": self.cost_per_row,
                }
            )
        return out


class RuleLLM:
    """Stands in for the filter and writer models; records the prompts it was given."""

    POST = re.compile(r"^#(\d+) \[([^\]]*)\] (.*)$", re.M)

    def __init__(self):
        self.prompts: list[tuple[str, str]] = []

    def complete(self, prompt: str, model: str) -> str:
        self.prompts.append((model, prompt))
        if "=== POSTS ===" in prompt:
            return self._filter(prompt)
        return self._write(prompt)

    def _filter(self, prompt: str) -> str:
        out = []
        for n, _source, text in self.POST.findall(prompt):
            topic = topic_of(text)
            if topic == "noise":
                continue
            score = 2 + (topic != "other") + sum(w in text.lower() for w in STRONG)
            soon = re.search(r"\b(on|next) (monday|tuesday|wednesday|thursday|friday|week)\b", text, re.I)
            out.append(
                {
                    "n": int(n),
                    "topic": topic,
                    "score": min(score, 5),
                    "soon": soon.group(0) if soon else None,
                    "gist": text.split(". ")[0][:140],
                }
            )
        return "Here you go:\n" + json.dumps(out)

    def _write(self, prompt: str) -> str:
        items = re.findall(r"^- \[(\w+)\|(\d)\] (.*)$", prompt, re.M)
        top = [f"- {gist}" for _t, score, gist in items if int(score) >= 4][:5]
        rest = [f"- {gist}" for _t, score, gist in items if int(score) < 4]
        return "\n".join(["Top", *top, "", "Also", *rest]).strip()
