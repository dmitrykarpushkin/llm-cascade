import io
import json
import urllib.error

import pytest

from llm_cascade.classifier import Classifier, Unavailable, answer, question


class FakeAPI:
    """Replaces the HTTP call: records requests, returns or raises scripted responses."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []

    def __call__(self, req, timeout):
        self.requests.append(json.loads(req.data))
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r(json.loads(req.data)) if callable(r) else r


def http_error(code):
    return urllib.error.HTTPError("u", code, "err", {}, io.BytesIO())


def auto_answer(choice="markets", conf=0.8, tokens=1000):
    def respond(body):
        answers = {}
        for qid, q in body["questions"].items():
            if q["type"] == "noul":
                answers[qid] = {"type": "noul", "noul": 0.9}
            else:
                answers[qid] = {"type": "choice", "choice": choice, "confidence": conf, "probabilities": {choice: conf}}
        return {"answers": answers, "usage": {"input_tokens": tokens}, "model": "m1"}

    return respond


def clf(api, key="k"):
    return Classifier(api_key=key, post=api, sleep=lambda s: None)


def test_question_shapes():
    assert question("Is it?", None) == {"type": "noul", "instructions": "Is it?"}
    assert question("Which?", ["a", "b"])["criteria"] == {"a": None, "b": None}


def test_answer_yes_no_confidence():
    assert answer({"type": "noul", "noul": 0.9}) == {"choice": "yes", "confidence": 0.8, "p_yes": 0.9}
    assert answer({"type": "noul", "noul": 0.25})["choice"] == "no"
    assert answer({"type": "noul", "noul": 0.5})["confidence"] == 0


def test_decide_and_cost():
    api = FakeAPI(auto_answer(tokens=1_000_000))
    out = clf(api).decide("Topic?", "BTC up", ["markets", "noise"])
    assert out["choice"] == "markets" and out["model"] == "m1"
    assert out["cost_usd"] == pytest.approx(0.042)
    assert api.requests[0]["state"] == "BTC up"


def test_no_key_is_unavailable_without_network():
    api = FakeAPI()
    with pytest.raises(Unavailable, match="no API key"):
        clf(api, key=None).decide("q", "t")
    assert api.requests == []


def test_retries_once_on_server_error():
    api = FakeAPI(http_error(503), auto_answer())
    assert clf(api).decide("q", "t")["choice"] == "yes"
    assert len(api.requests) == 2


def test_no_retry_on_auth_error():
    api = FakeAPI(http_error(401), auto_answer())
    with pytest.raises(Unavailable, match="HTTP 401"):
        clf(api).decide("q", "t")
    assert len(api.requests) == 1


def test_gives_up_after_two_network_errors():
    api = FakeAPI(TimeoutError(), urllib.error.URLError("down"))
    with pytest.raises(Unavailable, match="URLError"):
        clf(api).decide("q", "t")


def test_malformed_response_is_unavailable():
    with pytest.raises(Unavailable):
        clf(FakeAPI({"answers": {}})).decide("q", "t")


def test_batch_groups_same_text_into_one_request_and_splits_cost():
    api = FakeAPI(auto_answer(tokens=2000), auto_answer(tokens=1000))
    rows = [
        {"question": "Topic?", "options": ["markets", "noise"], "text": "same"},
        {"question": "Urgent?", "text": "same"},
        {"question": "Topic?", "options": ["markets", "noise"], "text": "other"},
    ]
    out = clf(api).decide_batch(rows)
    assert len(api.requests) == 2
    assert set(api.requests[0]["questions"]) == {"q0", "q1"}
    assert [o["choice"] for o in out] == ["markets", "yes", "markets"]
    assert out[0]["cost_usd"] == out[1]["cost_usd"] == pytest.approx(0.042 * 2000 / 1e6 / 2)


def test_batch_partial_failure_marks_only_failed_group():
    api = FakeAPI(auto_answer(), http_error(400))
    out = clf(api).decide_batch([{"question": "q", "text": "a"}, {"question": "q", "text": "b"}])
    assert "error" not in out[0]
    assert out[1] == {"error": "unavailable", "reason": "HTTP 400"}


def test_from_env():
    c = Classifier.from_env({"CASCADE_CLASSIFIER_API_KEY": "x", "CASCADE_CLASSIFIER_URL": "http://local"})
    assert c.api_key == "x" and c.url == "http://local"
    assert Classifier.from_env({}).api_key is None
