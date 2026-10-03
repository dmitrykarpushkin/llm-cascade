import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from llm_cascade.classifier import Classifier, Unavailable
from llm_cascade.cli import EXIT_UNAVAILABLE, decide_main, digest_main
from llm_cascade.llm import ClaudeCLI, LLMError, parse_json_array

FEED = Path(__file__).resolve().parents[1] / "examples" / "posts.jsonl"


class StubClassifier(Classifier):
    def __init__(self, fail=False):
        super().__init__(api_key="k")
        self.fail = fail

    def call(self, text, questions):
        if self.fail:
            raise Unavailable("HTTP 503")
        return {
            "answers": {q: {"type": "noul", "noul": 0.95} for q in questions},
            "usage": {"input_tokens": 100},
            "model": "m",
        }


def test_decide_cli_single(capsys):
    assert decide_main(["Is it about markets?", "--text", "BTC up"], classifier=StubClassifier()) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["choice"] == "yes" and out["confidence"] == 0.9


def test_decide_cli_unavailable_exit_code(capsys):
    assert decide_main(["q", "--text", "t"], classifier=StubClassifier(fail=True)) == EXIT_UNAVAILABLE
    assert capsys.readouterr().out.strip() == "unavailable"


def test_decide_cli_batch(tmp_path, capsys):
    rows = tmp_path / "rows.jsonl"
    rows.write_text('{"id": "a", "question": "q", "text": "x"}\n{"question": "q", "text": "x"}\n')
    assert decide_main(["--batch", str(rows)], classifier=StubClassifier()) == 0
    lines = [json.loads(x) for x in capsys.readouterr().out.splitlines()]
    assert lines[0]["id"] == "a" and "id" not in lines[1]

    assert decide_main(["--batch", str(rows)], classifier=StubClassifier(fail=True)) == EXIT_UNAVAILABLE


def test_digest_cli_offline_demo(capsys):
    assert digest_main([str(FEED), "--usage"]) == 0
    captured = capsys.readouterr()
    assert captured.out.startswith("Top")
    assert json.loads(captured.err)["kept"] > 0


def test_parse_json_array():
    assert parse_json_array('noise [{"n": 1}, {"m": 2}, 3] tail') == [{"n": 1}]
    assert parse_json_array("no json here") == []
    assert parse_json_array("[{broken") == []
    assert parse_json_array('[{"n": 1}') == []


def test_claude_cli_success_and_failure():
    seen = {}

    def ok(argv, **kw):
        seen.update(argv=argv, input=kw["input"])
        return SimpleNamespace(returncode=0, stdout=" answer \n", stderr="")

    assert ClaudeCLI(run=ok).complete("prompt", "haiku") == "answer"
    assert seen["argv"][:4] == ["claude", "-p", "--model", "haiku"]
    assert "Bash" in seen["argv"][seen["argv"].index("--disallowedTools") + 1]
    assert seen["input"] == "prompt"

    def bad(argv, **kw):
        return SimpleNamespace(returncode=1, stdout="", stderr="not logged in")

    with pytest.raises(LLMError, match="not logged in"):
        ClaudeCLI(run=bad).complete("p", "opus")
