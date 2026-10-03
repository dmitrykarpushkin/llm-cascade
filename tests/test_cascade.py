import json
from pathlib import Path

from llm_cascade.cascade import Cascade, CascadeConfig, chunk_lines, dedupe
from llm_cascade.classifier import Unavailable
from llm_cascade.fakes import KeywordClassifier, RuleLLM

FEED = Path(__file__).resolve().parents[1] / "examples" / "posts.jsonl"


def feed():
    return [json.loads(line) for line in FEED.read_text().splitlines() if line.strip()]


class DownClassifier:
    def decide_batch(self, rows):
        raise Unavailable("down")


class ErrorClassifier:
    def decide_batch(self, rows):
        return [{"error": "unavailable", "reason": "HTTP 500"} for _ in rows]


class ScriptedLLM:
    def __init__(self, filter_reply, writer_reply="digest"):
        self.filter_reply, self.writer_reply = filter_reply, writer_reply
        self.calls = []

    def complete(self, prompt, model):
        self.calls.append(model)
        return self.filter_reply if "=== POSTS ===" in prompt else self.writer_reply


def test_dedupe_and_chunking():
    items = [{"text": "a"}, {"text": " a "}, {"text": ""}, {"text": "b"}]
    assert [i["text"] for i in dedupe(items)] == ["a", "b"]
    assert chunk_lines(["aaaa", "bbbb", "cc", "dddddddd"], 6) == [["aaaa"], ["bbbb", "cc"], ["dddddddd"]]


def test_full_cascade_on_example_feed():
    clf, llm = KeywordClassifier(), RuleLLM()
    r = Cascade(llm, clf).run(feed())
    u = r.usage
    assert u["items"] == 14  # the blank post is ignored
    assert u["unique"] == 13  # one repost removed
    assert u["classifier_in"] == 13 and u["classifier_kept"] == 10  # three noise posts dropped at stage 0
    assert clf.calls == 1
    assert [m for m, _ in llm.prompts] == ["haiku", "opus"]
    filter_prompt = llm.prompts[0][1]
    assert "GUARANTEED" not in filter_prompt and "Giveaway" not in filter_prompt
    assert r.candidates[0]["score"] >= r.candidates[-1]["score"] >= 3
    assert "record $1.2 billion" in r.text and r.text.startswith("Top")
    assert "quiet weekend market" not in r.text  # score 2: never reaches the writer


def test_classifier_down_falls_back_to_small_model():
    llm = RuleLLM()
    r = Cascade(llm, DownClassifier()).run(feed())
    assert r.usage["classifier"] == "unavailable: down"
    assert "GUARANTEED" in llm.prompts[0][1]  # the small model now sees the noise and drops it itself
    assert "record $1.2 billion" in r.text


def test_classifier_row_errors_keep_items():
    r = Cascade(RuleLLM(), ErrorClassifier()).run(feed())
    assert r.usage["classifier_errors"] == 13 and r.usage["classifier_kept"] == 13


def test_no_classifier():
    r = Cascade(RuleLLM()).run(feed())
    assert r.usage["classifier"] == "off"


def test_low_confidence_noise_is_kept():
    r = Cascade(RuleLLM(), KeywordClassifier(confidence=0.3)).run(feed())
    assert r.usage["classifier_kept"] == 13


def test_bad_filter_output_and_out_of_range_indexes():
    reply = 'Sure! [{"n": 0, "score": 5, "gist": "ok"}, {"n": 99, "score": 5}, {"n": "1", "score": 5}, {"x": 1}]'
    llm = ScriptedLLM(reply)
    r = Cascade(llm).run([{"source": "s", "text": "one"}, {"source": "s", "text": "two"}])
    assert [c["gist"] for c in r.candidates] == ["ok"]

    r = Cascade(ScriptedLLM("I could not do it")).run([{"source": "s", "text": "one"}])
    assert r.text == CascadeConfig().empty_text and r.usage["kept"] == 0


def test_nothing_important_skips_writer():
    llm = ScriptedLLM('[{"n": 0, "score": 1, "gist": "meh"}]')
    r = Cascade(llm).run([{"source": "s", "text": "boring"}])
    assert r.text == "Nothing important today."
    assert llm.calls == ["haiku"]


def test_large_feed_is_chunked():
    items = [{"source": "s", "text": f"post number {i} " + "x" * 500} for i in range(10)]
    llm = ScriptedLLM("[]")
    r = Cascade(llm, config=CascadeConfig(chunk_chars=1200)).run(items)
    assert r.usage["filter_calls"] == 5


def test_empty_feed():
    llm = ScriptedLLM("[]")
    r = Cascade(llm, KeywordClassifier()).run([])
    assert r.usage["filter_calls"] == 0 and llm.calls == []


def test_config_from_env():
    c = CascadeConfig.from_env({"CASCADE_MIN_SCORE": "4", "CASCADE_WRITER_MODEL": "sonnet"})
    assert c.min_score == 4 and c.writer_model == "sonnet"
