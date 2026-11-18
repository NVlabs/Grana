# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mock HTTP verifies billing bounds, resumability and strict annotation output."""
import base64
from concurrent.futures import ThreadPoolExecutor
import json
import threading
import time

import httpx
import pytest

from grana.annotation import api as api


SCHEMA = {"type": "object", "properties": {"stage": {"type": "integer"}},
          "required": ["stage"], "additionalProperties": False}
TEST_MODEL = "fixture/vision-model"
TEST_BASE_URL = "https://annotation.example/v1"
KEY = "test-key-never-persist-abcdefghijklmnopqrstuvwxyz"
IMAGE = "data:image/png;base64," + base64.b64encode(b"unique-image-evidence").decode()


@pytest.fixture(autouse=True)
def provider_configuration(monkeypatch):
    """Explicit synthetic provider for mocked requests; never a runtime default."""
    monkeypatch.setenv(api.MODEL_ENV, TEST_MODEL)
    monkeypatch.setenv(api.BASE_URL_ENV, TEST_BASE_URL)


@pytest.fixture(autouse=True)
def credential(monkeypatch):
    monkeypatch.setenv(api.KEY_ENV, KEY)


def response(**updates):
    return {"id": "resp-test", "status": "completed", "model": TEST_MODEL,
            "reasoning": {"effort": "xhigh"},
            "output": [{"type": "reasoning", "summary": [{"text": "private reasoning"}]},
                       {"type": "message", "content": [{"type": "output_text", "text": '{"stage":2}'}]}],
            "usage": {"input_tokens": 20, "output_tokens": 30, "total_tokens": 50,
                      "output_tokens_details": {"reasoning_tokens": 25}}, **updates}


def invoke(directory, handler, name="draft", **kwargs):
    values = {"instructions": "Label the observed stage without assuming success.",
              "content": [{"type": "input_text", "text": "frame 42"},
                          {"type": "input_image", "image_url": IMAGE, "detail": "high"}],
              "schema": SCHEMA, "transport": httpx.MockTransport(handler)}
    return api.request_json(directory, name, **(values | kwargs))


def read(path):
    return json.loads(path.read_text())


def test_exact_request_and_verified_offline_cache(tmp_path, monkeypatch):
    requests = []

    def handler(request):
        requests.append(request)
        assert str(request.url) == TEST_BASE_URL + "/responses"
        assert request.headers["Authorization"] == "Bearer " + KEY
        payload = json.loads(request.content)
        assert payload["model"] == TEST_MODEL
        assert payload["store"] is False
        assert payload["reasoning"] == {"effort": "xhigh"}
        assert payload["max_output_tokens"] == 16384
        assert payload["text"]["format"]["strict"] is True
        assert payload["text"]["format"]["schema"] == SCHEMA
        assert payload["input"][0]["content"][1]["image_url"] == IMAGE
        return httpx.Response(200, json=response())

    result = invoke(tmp_path, handler)
    assert result["data"] == {"stage": 2} and result["cache_hit"] is False
    assert result["usage"]["total_tokens"] == 50
    assert result["response_id"] == "resp-test" and result["model"] == TEST_MODEL
    assert len(read(tmp_path / "budget.json")["reservations"]) == 1
    monkeypatch.delenv(api.KEY_ENV)
    cached = invoke(tmp_path, lambda _: pytest.fail("cache must not contact HTTP"))
    assert cached == result | {"cache_hit": True}
    assert len(requests) == 1
    persisted = "\n".join(path.read_text() for path in tmp_path.rglob("*.json"))
    for forbidden in (KEY, IMAGE, "unique-image-evidence", "private reasoning", "Authorization"):
        assert forbidden not in persisted


@pytest.mark.parametrize("change", ["image", "instructions", "schema", "model", "effort", "output_tokens", "endpoint", "order"])
def test_fingerprint_covers_exact_request_and_destination(tmp_path, change):
    def handler(request):
        sent = json.loads(request.content)
        return httpx.Response(200, json=response(model=sent["model"], reasoning=sent["reasoning"]))

    first = invoke(tmp_path, handler)
    variants = {
        "image": {"content": [{"type": "input_image", "image_url": "data:image/png;base64,YWJj", "detail": "high"}]},
        "instructions": {"instructions": "Different instruction"},
        "schema": {"schema": SCHEMA | {"description": "Different standard"}},
        "model": {"model": "different/model"}, "effort": {"reasoning_effort": "high"},
        "output_tokens": {"max_output_tokens": 1024}, "endpoint": {"base_url": "https://another.example/v1"},
        "order": {"content": [{"type": "input_image", "image_url": IMAGE, "detail": "high"},
                               {"type": "input_text", "text": "frame 42"}]},
    }
    second = invoke(tmp_path, handler, **variants[change])
    assert first["fingerprint"] != second["fingerprint"]
    assert second["cache_hit"] is False
    assert len(read(tmp_path / "budget.json")["reservations"]) == 2


def test_missing_key_fails_locally_without_spending(tmp_path, monkeypatch):
    monkeypatch.delenv(api.KEY_ENV)
    with pytest.raises(api.AnnotationAPIError, match="missing_api_key"):
        invoke(tmp_path, lambda _: pytest.fail("missing key must never send"))
    assert not (tmp_path / "budget.json").exists()


@pytest.mark.parametrize("failure", ["timeout", "connection", "http408", "http500", "nonterminal", "malformed"])
def test_uncertain_attempt_does_not_automatically_retry(tmp_path, failure):
    calls = []

    def handler(request):
        calls.append(request)
        if failure == "timeout":
            raise httpx.ReadTimeout(KEY + IMAGE, request=request)
        if failure == "connection":
            raise httpx.ConnectError(KEY, request=request)
        if failure.startswith("http"):
            return httpx.Response(int(failure[4:]), text=KEY + IMAGE)
        if failure == "nonterminal":
            return httpx.Response(200, json=response(status="in_progress"))
        return httpx.Response(200, text="not terminal JSON " + KEY)

    with pytest.raises(api.AnnotationAPIError, match="transport_uncertain") as caught:
        invoke(tmp_path, handler, max_attempts=4)
    assert KEY not in str(caught.value) and IMAGE not in str(caught.value)
    assert len(calls) == 1
    with pytest.raises(api.AnnotationAPIError, match="requires_explicit_retry"):
        invoke(tmp_path, handler, max_attempts=100)
    assert len(calls) == 1
    saved = next(tmp_path.rglob("receipt.json")).read_text()
    assert KEY not in saved and IMAGE not in saved
    recovered = invoke(tmp_path, lambda _: httpx.Response(200, json=response()), retry_uncertain=True)
    assert recovered["data"] == {"stage": 2} and not recovered["cache_hit"]
    receipt = read(next(tmp_path.rglob("receipt.json")))
    assert len(receipt["attempts"]) == 2
    assert receipt["attempts"][1]["explicit_uncertain_retry"] is True


def test_interrupted_process_becomes_uncertain_and_stays_charged(tmp_path):
    def interrupted(_):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        invoke(tmp_path, interrupted)
    assert read(next(tmp_path.rglob("receipt.json")))["status"] == "in_flight"
    assert len(read(tmp_path / "budget.json")["reservations"]) == 1
    with pytest.raises(api.AnnotationAPIError, match="requires_explicit_retry"):
        invoke(tmp_path, lambda _: pytest.fail("interrupted attempt must not replay"))
    assert read(next(tmp_path.rglob("receipt.json")))["status"] == "transport_uncertain"


def test_attempt_limit_cannot_increase_after_uncertain(tmp_path):
    def handler(request):
        raise httpx.ReadTimeout("timeout", request=request)

    with pytest.raises(api.AnnotationAPIError, match="transport_uncertain"):
        invoke(tmp_path, handler, max_attempts=1)
    with pytest.raises(api.AnnotationAPIError, match="request_attempt_budget_exhausted"):
        invoke(tmp_path, lambda _: pytest.fail("original attempt limit applies"),
               max_attempts=100, retry_uncertain=True)
    assert len(read(tmp_path / "budget.json")["reservations"]) == 1


def test_campaign_limit_shared_across_requests_and_never_increases(tmp_path):
    invoke(tmp_path, lambda _: httpx.Response(200, json=response()), max_requests=1)
    with pytest.raises(api.AnnotationAPIError, match="campaign_budget_exhausted"):
        invoke(tmp_path, lambda _: pytest.fail("original campaign limit applies"), name="review", max_requests=100)
    budget = read(tmp_path / "budget.json")
    assert budget["max_requests"] == 1 and len(budget["reservations"]) == 1


def test_rate_limit_retries_are_counted_and_bounded(tmp_path, monkeypatch):
    monkeypatch.setattr(api.time, "sleep", lambda _: None)
    calls = []

    def handler(_):
        calls.append(1)
        return httpx.Response(429, text=KEY) if len(calls) == 1 else httpx.Response(200, json=response())

    assert invoke(tmp_path, handler)["data"] == {"stage": 2}
    assert len(calls) == len(read(tmp_path / "budget.json")["reservations"]) == 2


def test_rate_limit_cannot_exceed_campaign_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(api.time, "sleep", lambda _: None)
    calls = []

    def handler(_):
        calls.append(1)
        return httpx.Response(429, text="rate limit")

    with pytest.raises(api.AnnotationAPIError, match="campaign_budget_exhausted"):
        invoke(tmp_path, handler, max_attempts=3, max_requests=1)
    assert len(calls) == 1


@pytest.mark.parametrize("raw,code", [
    (response(status="incomplete"), "response_incomplete"),
    (response(status="failed"), "response_failed"),
    (response(output=[{"type": "message", "content": [{"type": "refusal", "refusal": KEY}]}]), "refusal"),
    (response(model="wrong"), "model_mismatch"),
    (response(reasoning={"effort": "low"}), "effort_mismatch"),
    (response(reasoning=None), "effort_mismatch"),
    (response(id=""), "missing_response_id"),
    (response(output=[]), "invalid_output_json"),
    (response(output=[{"type": "function_call"}]), "unexpected_output_type"),
    (response(output=[{"type": "message", "content": [{"type": "output_text", "text": '{"stage":"wrong"}'}]}]), "invalid_output_schema"),
    (response(output=[{"type": "message", "content": [{"type": "output_text", "text": '{"stage":1,"stage":2}'}]}]), "invalid_output_json"),
    (response(output=[{"type": "message", "content": [{"type": "output_text", "text": '{"stage":NaN}'}]}]), "invalid_output_json"),
])
def test_semantic_failures_never_fallback_or_retry(tmp_path, raw, code):
    calls = []

    def handler(_):
        calls.append(1)
        return httpx.Response(200, json=raw)

    with pytest.raises(api.AnnotationAPIError, match=code):
        invoke(tmp_path, handler, max_attempts=10)
    with pytest.raises(api.AnnotationAPIError, match=code):
        invoke(tmp_path, handler, retry_uncertain=True)
    assert len(calls) == 1
    assert not list(tmp_path.rglob("completed.json"))
    receipt = read(next(tmp_path.rglob("receipt.json")))
    assert receipt["attempts"][0]["response"]["usage"]["total_tokens"] == 50
    assert receipt["attempts"][0]["response"]["id"] == raw["id"]


@pytest.mark.parametrize("status", [302, 400, 401, 403, 404, 413, 422])
def test_http_failure_is_safe_and_terminal(tmp_path, status):
    calls = []

    def handler(_):
        calls.append(1)
        return httpx.Response(status, text=KEY + IMAGE, headers={"Location": "https://other.example"})

    with pytest.raises(api.AnnotationAPIError, match=f"http_status_{status}") as caught:
        invoke(tmp_path, handler)
    assert len(calls) == 1
    assert KEY not in str(caught.value)
    assert KEY not in next(tmp_path.rglob("receipt.json")).read_text()


@pytest.mark.parametrize("secret", [KEY, IMAGE])
def test_provider_echo_of_secret_or_image_is_not_persisted(tmp_path, secret):
    raw = response(usage={"unsafe": secret})
    with pytest.raises(api.AnnotationAPIError, match="sensitive_response_rejected"):
        invoke(tmp_path, lambda _: httpx.Response(200, json=raw))
    assert not list(tmp_path.rglob("completed.json"))
    assert secret not in next(tmp_path.rglob("receipt.json")).read_text()


def test_corrupt_completed_cache_fails_without_repair_or_network(tmp_path):
    invoke(tmp_path, lambda _: httpx.Response(200, json=response()))
    path = next(tmp_path.rglob("completed.json"))
    path.write_text(path.read_text().replace('"stage":2', '"stage":3'))
    with pytest.raises(api.AnnotationAPIError, match="invalid_completed_cache"):
        invoke(tmp_path, lambda _: pytest.fail("corrupt cache must not rebill"))


def test_corrupt_campaign_budget_fails_closed(tmp_path):
    (tmp_path / "budget.json").write_text("invalid")
    with pytest.raises(api.AnnotationAPIError, match="invalid_local_record"):
        invoke(tmp_path, lambda _: pytest.fail("corrupt budget must not reset"))


def test_concurrent_identical_requests_send_once(tmp_path):
    calls = []

    def handler(_):
        calls.append(1)
        time.sleep(0.02)
        return httpx.Response(200, json=response())

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: invoke(tmp_path, handler), range(4)))
    assert len(calls) == 1
    assert sum(not item["cache_hit"] for item in results) == 1
    assert len(read(tmp_path / "budget.json")["reservations"]) == 1


def test_concurrent_distinct_requests_share_atomic_budget(tmp_path):
    barrier = threading.Barrier(3)
    calls = []

    def handler(_):
        calls.append(1)
        return httpx.Response(200, json=response())

    def run(index):
        barrier.wait()
        try:
            return invoke(tmp_path, handler, name=f"request-{index}", max_requests=1)
        except api.AnnotationAPIError as exc:
            return exc.code

    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(run, range(3)))
    assert len(calls) == 1
    assert results.count("campaign_budget_exhausted") == 2
    assert len(read(tmp_path / "budget.json")["reservations"]) == 1


@pytest.mark.parametrize("override", [
    {"name": "../escape"}, {"max_attempts": True}, {"max_requests": 0},
    {"timeout_seconds": float("inf")}, {"timeout_seconds": 0}, {"max_output_tokens": 65537},
    {"reasoning_effort": "low"}, {"retry_uncertain": "yes"},
    {"base_url": "https://user:secret@example.com/v1"}, {"base_url": "http://example.com/v1"},
    {"base_url": "https://example.com/v1?key=value"},
    {"content": [{"type": "input_image", "image_url": "https://example.com/private", "detail": "high"}]},
    {"content": [{"type": "input_file", "file_id": "file-test"}]},
])
def test_local_validation_prevents_http(tmp_path, override):
    with pytest.raises(ValueError):
        invoke(tmp_path, lambda _: pytest.fail("invalid input must not send"), **override)
    assert not (tmp_path / "budget.json").exists()


@pytest.mark.parametrize("missing", ["model", "base_url", "both"])
def test_live_request_requires_explicit_provider_without_spending(tmp_path, monkeypatch, missing):
    if missing in ("model", "both"):
        monkeypatch.delenv(api.MODEL_ENV)
    if missing in ("base_url", "both"):
        monkeypatch.delenv(api.BASE_URL_ENV)
    with pytest.raises(ValueError, match="Annotation .* is required"):
        invoke(tmp_path, lambda _: pytest.fail("missing provider must never send"))
    assert not list(tmp_path.iterdir())


def test_no_provider_is_selected_when_environment_is_empty(monkeypatch):
    import inspect
    monkeypatch.delenv(api.MODEL_ENV)
    monkeypatch.delenv(api.BASE_URL_ENV)
    assert api.provider_settings(required=False) == {"model": None, "base_url": None}
    for name in ("model", "base_url"):
        assert inspect.signature(api.request_json).parameters[name].default is None
    assert api.KEY_ENV == "GRANA_ANNOTATION_API_KEY"


def test_explicit_provider_overrides_environment_and_replays_without_any_environment(tmp_path, monkeypatch):
    selected = {"model": "alternative/vision", "base_url": "https://alternate.example/custom/v2",
                "api_key_env": "ALTERNATE_PROVIDER_KEY"}
    monkeypatch.setenv(selected["api_key_env"], KEY)
    monkeypatch.delenv(api.KEY_ENV)
    seen = []

    def handler(request):
        seen.append(request)
        payload = json.loads(request.content)
        assert str(request.url) == selected["base_url"] + "/responses"
        assert payload["model"] == selected["model"]
        assert request.headers["authorization"] == "Bearer " + KEY
        return httpx.Response(200, json=response(model=selected["model"]))

    first = invoke(tmp_path, handler, **selected)
    for name in (api.MODEL_ENV, api.BASE_URL_ENV, selected["api_key_env"]):
        monkeypatch.delenv(name)
    cached = invoke(tmp_path, lambda _: pytest.fail("completed request must replay offline"), **selected)
    assert cached == first | {"cache_hit": True}
    assert len(seen) == 1
    assert read(tmp_path / "budget.json")["reservations"][0]["fingerprint"] == first["fingerprint"]


@pytest.mark.parametrize("settings", [
    {"model": "YOUR-MODEL-NAME"}, {"model": "${MODEL}"},
    {"base_url": "https://provider.invalid/v1"}, {"base_url": "https://YOUR-ENDPOINT/v1"},
])
def test_documentation_placeholders_cannot_send(tmp_path, settings):
    with pytest.raises(ValueError, match="placeholder"):
        invoke(tmp_path, lambda _: pytest.fail("placeholder must never send"), **settings)
    assert not list(tmp_path.iterdir())
