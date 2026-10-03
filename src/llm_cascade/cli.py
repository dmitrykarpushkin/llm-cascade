"""Command line entry points.

  cascade-decide "Is this about crypto?" --text "BTC broke 100k"         yes/no
  echo "text" | cascade-decide "What kind of note?" -o idea,task,junk     pick one
  cascade-decide --batch rows.jsonl                                       {"id"?, "question", "options"?, "text"}

  cascade-digest examples/posts.jsonl             offline: keyword classifier + rule-based "models"
  cascade-digest posts.jsonl --live               real stages: classifier API (if keyed) + claude -p

cascade-decide prints JSON; when the classifier is unavailable it prints "unavailable" and exits with code 2,
so a calling script can fall back to a model or a rule.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .cascade import Cascade, CascadeConfig
from .classifier import Classifier, Unavailable
from .fakes import KeywordClassifier, RuleLLM
from .llm import ClaudeCLI

EXIT_UNAVAILABLE = 2


def decide_main(argv: list[str] | None = None, classifier: Classifier | None = None) -> int:
    ap = argparse.ArgumentParser(prog="cascade-decide", description="Cheap yes/no or pick-one decisions")
    ap.add_argument("question", nargs="?")
    ap.add_argument("-o", "--options", help="comma-separated options; without them the answer is yes/no")
    ap.add_argument("-t", "--text", help="the text to judge (default: stdin)")
    ap.add_argument("--batch", metavar="FILE.jsonl")
    a = ap.parse_args(argv)
    clf = classifier or Classifier.from_env()
    try:
        if a.batch:
            rows = [json.loads(line) for line in Path(a.batch).read_text().splitlines() if line.strip()]
            failed = False
            for row, res in zip(rows, clf.decide_batch(rows), strict=True):
                failed |= "error" in res
                print(json.dumps({"id": row["id"], **res} if "id" in row else res, ensure_ascii=False))
            return EXIT_UNAVAILABLE if failed else 0
        if not a.question:
            ap.error("a question or --batch is required")
        text = a.text if a.text is not None else sys.stdin.read()
        opts = [o.strip() for o in a.options.split(",") if o.strip()] if a.options else None
        print(json.dumps(clf.decide(a.question, text, opts), ensure_ascii=False))
        return 0
    except Unavailable as e:
        print("unavailable")
        print(f"cascade-decide: {e}", file=sys.stderr)
        return EXIT_UNAVAILABLE


def digest_main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="cascade-digest", description="Daily digest from a JSONL feed")
    ap.add_argument("feed", help='JSONL rows {"source", "text", "url"?}')
    ap.add_argument("--live", action="store_true", help="use the real classifier API and claude -p")
    ap.add_argument("--usage", action="store_true", help="print stage statistics to stderr")
    a = ap.parse_args(argv)
    items = [json.loads(line) for line in Path(a.feed).read_text().splitlines() if line.strip()]
    if a.live:
        clf = Classifier.from_env()
        cascade = Cascade(ClaudeCLI(), clf if clf.api_key else None, CascadeConfig.from_env())
    else:
        cascade = Cascade(RuleLLM(), KeywordClassifier(), CascadeConfig.from_env())
    result = cascade.run(items)
    print(result.text)
    if a.usage:
        print(json.dumps(result.usage, ensure_ascii=False), file=sys.stderr)
    return 0
