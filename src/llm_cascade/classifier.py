"""Stage 0: a very cheap classifier for yes/no and pick-one decisions.

The client speaks the TypeSafe "System One" API (a classification model billed per input token, output free):
POST {"state": <text>, "model": ..., "questions": {id: question}} -> {"answers": {id: answer}, "usage": ...}.
Several questions about the same text go in one request, which costs about the same as one.

Any failure (no key, network, HTTP error, bad payload) raises `Unavailable`; callers fall back to the next
stage of the cascade instead of failing.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass

NO_RETRY = {400, 401, 403, 404, 422}


class Unavailable(Exception):
    pass


def question(q: str, options: list[str] | None) -> dict:
    if options:
        return {"type": "choice", "instructions": q, "criteria": {o: None for o in options}}
    return {"type": "noul", "instructions": q}


def answer(a: dict) -> dict:
    """Normalize one answer: yes/no from a probability, or a choice with its confidence and probabilities."""
    if a["type"] == "noul":
        p = a["noul"]
        return {"choice": "yes" if p >= 0.5 else "no", "confidence": round(abs(2 * p - 1), 3), "p_yes": round(p, 3)}
    return {
        "choice": a["choice"],
        "confidence": round(a["confidence"], 3),
        "probabilities": {k: round(v, 3) for k, v in a["probabilities"].items()},
    }


Post = Callable[[urllib.request.Request, float], dict]


def _urlopen(req: urllib.request.Request, timeout: float) -> dict:
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


@dataclass
class Classifier:
    api_key: str | None
    url: str = "https://api.typesafe.ai/v1/systemone"
    model: str = "jev-latest"
    timeout: float = 15
    price_per_m_input: float = 0.042  # USD per 1M input tokens
    retry_delay: float = 1.0
    post: Post = _urlopen
    sleep: Callable[[float], None] = time.sleep

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None, **kw) -> Classifier:
        env = os.environ if env is None else env
        return cls(
            api_key=env.get("CASCADE_CLASSIFIER_API_KEY") or None,
            url=env.get("CASCADE_CLASSIFIER_URL", cls.url),
            model=env.get("CASCADE_CLASSIFIER_MODEL", cls.model),
            **kw,
        )

    def call(self, text: str, questions: dict) -> dict:
        if not self.api_key:
            raise Unavailable("no API key")
        body = json.dumps({"state": text, "model": self.model, "questions": questions}).encode()
        req = urllib.request.Request(
            self.url,
            data=body,
            method="POST",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
        )
        last = None
        for attempt in range(2):  # one retry, only for errors a retry can fix
            try:
                return self.post(req, self.timeout)
            except urllib.error.HTTPError as e:
                last = f"HTTP {e.code}"
                if e.code in NO_RETRY:
                    break
            except (urllib.error.URLError, TimeoutError, OSError, ValueError) as e:
                last = type(e).__name__
            if attempt == 0:
                self.sleep(self.retry_delay)
        raise Unavailable(last)

    def cost(self, usage: dict | None, n: int = 1) -> float:
        return round((usage or {}).get("input_tokens", 0) * self.price_per_m_input / 1e6 / n, 7)

    def decide(self, q: str, text: str, options: list[str] | None = None) -> dict:
        r = self.call(text, {"q": question(q, options)})
        try:
            out = answer(r["answers"]["q"])
        except (KeyError, TypeError) as e:
            raise Unavailable(f"bad response: {e!r}") from e
        out.update(model=r.get("model"), cost_usd=self.cost(r.get("usage")))
        return out

    def decide_batch(self, rows: list[dict]) -> list[dict]:
        """rows: {"question", "options"?, "text"}. Rows with the same text share one request.

        Returns one result per row, in order; a failed group gives {"error": "unavailable", "reason": ...}
        for its rows and the rest still succeed.
        """
        groups: dict[str, list[int]] = {}
        for i, row in enumerate(rows):
            groups.setdefault(row.get("text", ""), []).append(i)
        results: list[dict] = [{} for _ in rows]
        for text, idx in groups.items():
            qs = {f"q{i}": question(rows[i]["question"], rows[i].get("options")) for i in idx}
            try:
                r = self.call(text, qs)
                share = self.cost(r.get("usage"), len(idx))
                for i in idx:
                    results[i] = {**answer(r["answers"][f"q{i}"]), "cost_usd": share}
            except (Unavailable, KeyError, TypeError) as e:
                for i in idx:
                    results[i] = {"error": "unavailable", "reason": str(e)}
        return results
