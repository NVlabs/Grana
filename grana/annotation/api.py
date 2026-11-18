# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Durable, bounded JSON Responses requests for offline annotation campaigns.

The campaign directory owns a non-increasing HTTP-attempt budget. A timeout or
interrupted attempt is never retried implicitly: the caller must explicitly set
``retry_uncertain=True`` on a later invocation, accepting possible duplicate
billing. Completed requests can be reused without credentials or network access.
"""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
import time
from urllib.parse import urlsplit

import httpx
import jsonschema

BASE_URL_ENV = "GRANA_ANNOTATION_BASE_URL"
MODEL_ENV = "GRANA_ANNOTATION_MODEL"
KEY_ENV = "GRANA_ANNOTATION_API_KEY"


def provider_settings(*, model=None, base_url=None, required=True):
    """Resolve explicit values before environment values; never select a provider.

    Callers freeze these resolved values in campaign inputs. Receipt validation
    uses recorded values directly, without consulting the current environment.
    Offline planning may leave either field unset via ``required=False``.
    """
    values = {"model": model if model is not None else (os.environ.get(MODEL_ENV) or None),
              "base_url": base_url if base_url is not None else (os.environ.get(BASE_URL_ENV) or None)}
    for name, env in (("model", MODEL_ENV), ("base_url", BASE_URL_ENV)):
        value = values[name]
        if value is None and not required:
            continue
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Annotation {name} is required; set --{name.replace('_', '-')} or {env} (or api.{name} in YAML)")
    if values["model"] is not None:
        _validate_model(values["model"])
    if values["base_url"] is not None:
        _validate_base_url(values["base_url"])
    return values


def _validate_model(model):
    if (not isinstance(model, str) or not model.strip() or model != model.strip()
            or not model.isprintable() or model.upper().startswith(("YOUR-", "YOUR_", "<"))
            or "${" in model):
        raise ValueError("model must be an explicit provider model name, not a placeholder")


def _validate_base_url(base_url):
    try:
        url = urlsplit(base_url) if isinstance(base_url, str) else urlsplit("")
        valid = (url.scheme == "https" and url.hostname and not url.username and not url.password
                 and not url.query and not url.fragment and base_url.isprintable() and " " not in base_url
                 and not url.hostname.endswith(".invalid") and url.hostname != "invalid"
                 and not url.hostname.lower().startswith(("your-", "your_", "<")) and "${" not in base_url)
    except ValueError:
        valid = False
    if not valid:
        raise ValueError("base_url must be an explicit HTTPS URL without credentials, query, fragment or placeholders")


class AnnotationAPIError(RuntimeError):
    """A safe error code and optional local receipt; no provider body or headers."""

    def __init__(self, code: str, receipt: Path | None = None, *, detail: str | None = None):
        self.code, self.receipt = code, str(receipt) if receipt else None
        super().__init__(code + (f"; {detail}" if detail else "") + (f"; receipt: {receipt}" if receipt else ""))


def _bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _digest(value):
    return hashlib.sha256(value).hexdigest()


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _json(text):
    value = json.loads(text, object_pairs_hook=_pairs,
                       parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
    _bytes(value)
    return value


def _read(path):
    try:
        return _json(path.read_bytes())
    except (OSError, ValueError, TypeError):
        raise AnnotationAPIError("invalid_local_record", path) from None


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = _bytes(value)
    descriptor, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@contextmanager
def _lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def _integer(value, name, maximum=None):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1 or (maximum and value > maximum):
        raise ValueError(f"{name} must be a positive integer" + (f" <= {maximum}" if maximum else ""))


def _prepare(instructions, content, schema, model, reasoning_effort, max_output_tokens, base_url):
    if not isinstance(instructions, str) or not instructions.strip():
        raise ValueError("instructions must be nonempty text")
    _validate_model(model)
    if reasoning_effort not in ("high", "xhigh"):
        raise ValueError("reasoning_effort must be high or xhigh")
    _integer(max_output_tokens, "max_output_tokens", 65536)
    if not isinstance(schema, dict):
        raise ValueError("schema must be a JSON object")
    jsonschema.Draft202012Validator.check_schema(schema)
    _validate_base_url(base_url)
    if not isinstance(content, list) or not content:
        raise ValueError("content must be a nonempty ordered list")
    for item in content:
        if not isinstance(item, dict):
            raise ValueError("content entries must be objects")
        if item.get("type") == "input_text":
            if set(item) != {"type", "text"} or not isinstance(item["text"], str):
                raise ValueError("input_text requires only type and text")
        elif item.get("type") == "input_image":
            if (set(item) != {"type", "image_url", "detail"} or item["detail"] != "high"
                    or not isinstance(item["image_url"], str)
                    or not re.fullmatch(r"data:image/(?:png|jpeg|webp);base64,[A-Za-z0-9+/]+={0,2}", item["image_url"])):
                raise ValueError("input_image requires a PNG/JPEG/WebP data URI and detail=high")
        else:
            raise ValueError("content supports only input_text and input_image")
    payload = {"model": model, "instructions": instructions, "store": False,
               "reasoning": {"effort": reasoning_effort}, "max_output_tokens": max_output_tokens,
               "input": [{"role": "user", "content": content}],
               "text": {"format": {"type": "json_schema", "name": "grana_annotation",
                                    "strict": True, "schema": schema}}}
    # Freeze caller-owned containers before hashing or sending them.
    return _json(_bytes(payload)), base_url.rstrip("/")


def _has_refusal(value):
    if isinstance(value, dict):
        return bool(value.get("refusal")) or value.get("type") == "refusal" or any(_has_refusal(v) for v in value.values())
    return isinstance(value, list) and any(_has_refusal(v) for v in value)


def _validated(raw, schema, model, effort):
    if not isinstance(raw, dict):
        raise AnnotationAPIError("invalid_response")
    if _has_refusal(raw):
        raise AnnotationAPIError("refusal")
    if raw.get("status") != "completed":
        raise AnnotationAPIError("response_" + raw["status"] if raw.get("status") in ("incomplete", "failed", "cancelled") else "transport_uncertain")
    if raw.get("model") != model:
        raise AnnotationAPIError("model_mismatch")
    if not isinstance(raw.get("reasoning"), dict) or raw["reasoning"].get("effort") != effort:
        raise AnnotationAPIError("effort_mismatch")
    if not isinstance(raw.get("id"), str) or not raw["id"].strip():
        raise AnnotationAPIError("missing_response_id")
    output = raw.get("output")
    if not isinstance(output, list):
        raise AnnotationAPIError("invalid_output")
    texts = []
    for item in output:
        if not isinstance(item, dict):
            raise AnnotationAPIError("invalid_output")
        if item.get("type") == "message":
            parts = item.get("content")
            if not isinstance(parts, list):
                raise AnnotationAPIError("invalid_output")
            for part in parts:
                if not isinstance(part, dict) or part.get("type") != "output_text" or not isinstance(part.get("text"), str):
                    raise AnnotationAPIError("invalid_output")
                texts.append(part["text"])
        elif item.get("type") != "reasoning":
            raise AnnotationAPIError("unexpected_output_type")
    try:
        data = _json("".join(texts))
    except (ValueError, TypeError):
        raise AnnotationAPIError("invalid_output_json") from None
    try:
        jsonschema.Draft202012Validator(schema).validate(data)
    except jsonschema.ValidationError:
        raise AnnotationAPIError("invalid_output_schema") from None
    return data


def _response_metadata(raw, key):
    """Retain billing evidence even for a terminal refusal/incomplete response."""
    if not isinstance(raw, dict):
        return {}
    metadata = {name: raw[name] for name in ("id", "status", "model", "usage") if name in raw}
    encoded = _bytes(metadata)
    if key.encode() in encoded or b"data:image/" in encoded:
        return {}
    return metadata


def _reserve(directory, max_requests, reservation=None):
    """Persist the campaign's lower limit, optionally reserving one HTTP attempt."""
    path = directory / "budget.json"
    with _lock(directory / ".budget.lock"):
        budget = _read(path) if path.exists() else {"version": 1, "max_requests": max_requests, "reservations": []}
        if (not isinstance(budget, dict) or budget.get("version") != 1
                or type(budget.get("max_requests")) is not int or budget["max_requests"] < 1
                or not isinstance(budget.get("reservations"), list)):
            raise AnnotationAPIError("invalid_local_budget", path)
        budget["max_requests"] = min(budget["max_requests"], max_requests)
        _write(path, budget)
        if reservation is None:
            return
        if len(budget["reservations"]) >= budget["max_requests"]:
            raise AnnotationAPIError("campaign_budget_exhausted", path)
        budget["reservations"].append(reservation)
        _write(path, budget)


def _result(receipt, path, completed, cache_hit):
    provider = completed["response"]
    return {"data": completed["data"], "usage": provider.get("usage"),
            "response_id": provider["id"], "model": provider["model"],
            "receipt": str(path), "cache_hit": cache_hit,
            "fingerprint": receipt["fingerprint"], "elapsed_seconds": receipt["elapsed_seconds"]}


def request_json(directory: Path, name: str, *, instructions: str, content: list[dict],
                 schema: dict, model: str | None = None, reasoning_effort: str = "xhigh",
                 max_output_tokens: int = 16384, timeout_seconds: float = 480,
                 max_attempts: int = 2, max_requests: int = 24, base_url: str | None = None,
                 retry_uncertain: bool = False, api_key_env: str = KEY_ENV,
                 transport: httpx.BaseTransport | None = None) -> dict:
    """Send or reuse one strict JSON request; ``max_requests`` counts HTTP attempts.

    Only a received HTTP 429 is automatically retried. Transport failures, 408,
    5xx, nonterminal responses and interrupted processes require an explicit
    later retry. The original per-request and campaign limits can only decrease.
    Receipts contain hashes/metadata and validated output, never input images,
    credentials, raw HTTP errors, or provider reasoning text.
    """
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,119}", name):
        raise ValueError("name must be a simple request identifier (1..120 characters)")
    for value, field in ((max_attempts, "max_attempts"), (max_requests, "max_requests")):
        _integer(value, field)
    if (isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds) or not 0 < timeout_seconds <= 900):
        raise ValueError("timeout_seconds must be positive and <= 900")
    if not isinstance(retry_uncertain, bool):
        raise ValueError("retry_uncertain must be a boolean")
    if not isinstance(api_key_env, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,127}", api_key_env):
        raise ValueError("api_key_env must name an environment variable")
    provider = provider_settings(model=model, base_url=base_url)
    model, base_url = provider["model"], provider["base_url"]
    payload, base_url = _prepare(instructions, content, schema, model, reasoning_effort, max_output_tokens, base_url)
    fingerprint = _digest(_bytes({"base_url": base_url, "payload": payload}))
    directory = Path(directory).expanduser().resolve()
    request_dir = directory / "requests" / name / fingerprint
    path = request_dir / "receipt.json"
    with _lock(request_dir / ".lock"):
        receipt = _read(path) if path.exists() else {
            "version": 1, "name": name, "fingerprint": fingerprint, "status": "prepared",
            "base_url": base_url, "model": model, "reasoning_effort": reasoning_effort,
            "max_output_tokens": max_output_tokens, "max_attempts": max_attempts,
            "payload_sha256": _digest(_bytes(payload)), "attempts": [],
        }
        if (not isinstance(receipt, dict) or receipt.get("version") != 1
                or receipt.get("fingerprint") != fingerprint or not isinstance(receipt.get("attempts"), list)
                or type(receipt.get("max_attempts")) is not int or receipt["max_attempts"] < 1):
            raise AnnotationAPIError("invalid_local_record", path)
        receipt["max_attempts"] = min(receipt["max_attempts"], max_attempts)
        if receipt["status"] == "completed":
            complete_path = request_dir / "completed.json"
            try:
                raw = complete_path.read_bytes()
                completed = _json(raw)
                if _digest(raw) != receipt["completed_sha256"] or completed["data"] != _validated(completed["response"], schema, model, reasoning_effort):
                    raise ValueError("cache mismatch")
            except (OSError, ValueError, KeyError, TypeError, AnnotationAPIError):
                raise AnnotationAPIError("invalid_completed_cache", path) from None
            # A cache hit spends no attempt, but a newly lowered campaign cap
            # still binds later requests (including concurrent workers).
            _reserve(directory, max_requests)
            return _result(receipt, path, completed, True)
        if receipt["status"] in ("in_flight", "transport_uncertain"):
            receipt["status"] = "transport_uncertain"
            if receipt["attempts"]:
                receipt["attempts"][-1]["status"] = "transport_uncertain"
            _write(path, receipt)
            if not retry_uncertain:
                raise AnnotationAPIError("transport_uncertain_requires_explicit_retry", path)
        elif receipt["status"] not in ("prepared", "rate_limited"):
            raise AnnotationAPIError(receipt.get("error_code", "request_failed"), path)
        try:
            key = os.environ[api_key_env]
        except KeyError:
            raise AnnotationAPIError("missing_api_key", detail=f"set environment variable {api_key_env}") from None
        if not key.strip():
            raise AnnotationAPIError("missing_api_key", detail=f"set environment variable {api_key_env}")
        _write(path, receipt)
        while len(receipt["attempts"]) < receipt["max_attempts"]:
            number = len(receipt["attempts"]) + 1
            attempt = {"number": number, "status": "in_flight", "timeout_seconds": timeout_seconds,
                       "started_unix_seconds": time.time(), "explicit_uncertain_retry": retry_uncertain}
            receipt["attempts"].append(attempt)
            receipt["status"] = "in_flight"
            # Both intent and campaign reservation are durable before any network I/O.
            _write(path, receipt)
            try:
                _reserve(directory, max_requests, {"name": name, "fingerprint": fingerprint, "attempt": number})
            except AnnotationAPIError as exc:
                attempt["status"] = receipt["status"] = "not_sent"
                receipt["error_code"] = exc.code
                _write(path, receipt)
                raise
            started = time.monotonic()
            try:
                with httpx.Client(transport=transport, timeout=httpx.Timeout(timeout_seconds, connect=min(10, timeout_seconds)),
                                  follow_redirects=False, trust_env=False) as client:
                    response = client.post(base_url + "/responses", content=_bytes(payload),
                                           headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
                attempt["http_status"] = response.status_code
                if response.status_code == 429:
                    raise AnnotationAPIError("rate_limited")
                if response.status_code == 408 or response.status_code >= 500:
                    raise AnnotationAPIError("transport_uncertain")
                if response.status_code != 200:
                    raise AnnotationAPIError(f"http_status_{response.status_code}")
                try:
                    raw = _json(response.content)
                except (ValueError, TypeError):
                    raise AnnotationAPIError("transport_uncertain") from None
                attempt["response"] = _response_metadata(raw, key)
                data = _validated(raw, schema, model, reasoning_effort)
                # Keep only output text, never chain-of-thought or arbitrary response extras.
                provider = {"id": raw["id"], "model": raw["model"], "status": "completed",
                            "reasoning": {"effort": reasoning_effort}, "usage": raw.get("usage"),
                            "output": [{"type": "message", "content": [{"type": "output_text", "text": _bytes(data).decode()}]}]}
                completed = {"response": provider, "data": data}
                output_bytes = _bytes(completed)
                if key.encode() in output_bytes or b"data:image/" in output_bytes:
                    raise AnnotationAPIError("sensitive_response_rejected")
                _write(request_dir / "completed.json", completed)
                receipt["completed_sha256"] = _digest(output_bytes)
                attempt["status"] = receipt["status"] = "completed"
                receipt["elapsed_seconds"] = time.monotonic() - started
                attempt["elapsed_seconds"] = receipt["elapsed_seconds"]
                _write(path, receipt)
                return _result(receipt, path, completed, False)
            except (httpx.TransportError, TimeoutError, ConnectionError):
                code = "transport_uncertain"
            except AnnotationAPIError as exc:
                code = exc.code
            attempt["status"] = receipt["status"] = code if code in ("transport_uncertain", "rate_limited") else "failed"
            attempt["error_code"] = receipt["error_code"] = code
            attempt["elapsed_seconds"] = time.monotonic() - started
            _write(path, receipt)
            if code != "rate_limited":
                raise AnnotationAPIError(code, path) from None
            if len(receipt["attempts"]) < receipt["max_attempts"]:
                time.sleep(min(2 ** (number - 1), 8))
        raise AnnotationAPIError("request_attempt_budget_exhausted", path)
