#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Submit prebuilt annotation windows with bounded workers and one API budget.

The manifest is a JSON list of {name, bundle, evidence, output_dir}; paths must
be absolute. All inputs are verified before workers start. Individual failures
are collected, while other jobs finish. This runner adds no HTTP or retry logic.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import re
import sys
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grana.annotation.api import AnnotationAPIError, KEY_ENV, provider_settings
from scripts import annotation_skill_api as driver


def _absolute(value):
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError("Paths must be nonempty and absolute")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError("Paths must be absolute")
    return path.resolve()


def _private_directory(path, raw_roots):
    if (path.is_relative_to(ROOT)
            or any((parent / ".git").exists() for parent in (path, *path.parents))
            or any(path.is_relative_to(raw) for raw in raw_roots)):
        raise ValueError("Private outputs must be outside code checkouts and raw sources")
    if any(parent.exists() and not parent.is_dir() for parent in (path, *path.parents)):
        raise ValueError("Output and campaign paths must be directories")


def _positive(value, name, maximum=None):
    if type(value) is not int or value < 1 or maximum is not None and value > maximum:
        raise ValueError(f"Invalid {name}")


def _preflight(manifest_path, api_dir, policy_dir, settings):
    jobs = json.loads(Path(manifest_path).read_text())
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("Manifest must be a nonempty list")
    names, outputs, normalized = set(), [], []
    for job in jobs:
        if not isinstance(job, dict) or set(job) != {"name", "bundle", "evidence", "output_dir"}:
            raise ValueError("Each job requires name, bundle, evidence and output_dir")
        name = job["name"]
        if (not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,119}", name)
                or name in names):
            raise ValueError("Job names must be unique simple request identifiers")
        names.add(name)
        item = {key: _absolute(job[key]) for key in ("bundle", "evidence", "output_dir")}
        if not item["bundle"].is_file() or not item["evidence"].is_file():
            raise ValueError("Bundle and evidence files must exist")
        output = item["output_dir"]
        if any(output.is_relative_to(other) or other.is_relative_to(output) for other in outputs):
            raise ValueError("Job output directories must be disjoint")
        outputs.append(output)
        normalized.append(item | {"name": name})

    raw_roots = []
    for job in normalized:
        bundle, content, images, template = driver.prepare_submission(
            job["bundle"], job["evidence"], policy_dir, settings.get("template_path"))
        raw_roots.extend(driver.input_roots(bundle, template))
        prepared_path = job["output_dir"] / "prepared.json"
        if prepared_path.exists():
            # Match the driver's immutable delivery record without creating it.
            api_settings = {key: settings[key] for key in (
                "model", "base_url", "reasoning_effort", "max_output_tokens",
                "timeout_seconds", "max_attempts", "max_requests")}
            if settings["api_key_env"] != KEY_ENV:
                api_settings["api_key_env"] = settings["api_key_env"]
            expected = {"schema_version": "grana.annotation_delivery.v1",
                        "bundle_sha256": driver.digest(bundle), "context_sha256": bundle["context_sha256"],
                        "policy": bundle["policy"], "images": images,
                        "frame_order": [row["frame"] for row in images], "source_identity_sent": False,
                        "api": api_settings, "request_name": job["name"], "api_dir": str(api_dir)}
            if template is not None:
                expected["template"] = template
            if json.loads(prepared_path.read_text()) != expected:
                raise ValueError("Existing prepared artifact differs from this job")
        del bundle, content, images  # Do not retain every image payload in the batch.
    for path in [api_dir, *outputs]:
        _private_directory(path, raw_roots)
    return normalized


def _error_code(error):
    if isinstance(error, AnnotationAPIError):
        code = error.code
        return code if isinstance(code, str) and re.fullmatch(r"[A-Za-z0-9_]{1,96}", code) else "api_error"
    if isinstance(error, (ValueError, TypeError)):
        return "validation_error"
    if isinstance(error, OSError):
        return "local_io_error"
    return "unexpected_error"


def _receipt_path(value, api_dir):
    if not isinstance(value, str) or len(value) > 4096:
        return None
    try:
        path = Path(value).resolve()
        return str(path) if path.is_relative_to(api_dir) and path.is_file() else None
    except (OSError, ValueError):
        return None


def _artifacts(output, api_dir, error=None):
    result = {}
    for name in ("prepared", "candidate", "validation", "delivery"):
        path = output / f"{name}.json"
        if path.is_file() and path.resolve().is_relative_to(output):
            result[name] = str(path)
    receipt = _receipt_path(getattr(error, "receipt", None), api_dir)
    if receipt is None and "delivery" in result:
        try:
            delivery = json.loads(Path(result["delivery"]).read_text())
            receipt = _receipt_path(delivery.get("receipt"), api_dir)
        except (OSError, ValueError, TypeError, AttributeError):
            pass
    if receipt is not None:
        result["receipt"] = receipt
    return result


def run(manifest_path, *, api_dir, max_requests, workers=16, policy_dir=driver.POLICY_DIR,
        model=None, base_url=None, reasoning_effort="xhigh", max_output_tokens=32768,
        timeout_seconds=480, max_attempts=1, retry_uncertain=False, run_worker=None,
        template_path=None, api_key_env=KEY_ENV):
    """Return ordered safe job summaries; ``run_worker`` permits network-free tests."""
    provider = provider_settings(model=model, base_url=base_url)
    model, base_url = provider["model"], provider["base_url"]
    _positive(workers, "workers", 32)
    _positive(max_requests, "max_requests")
    _positive(max_attempts, "max_attempts")
    _positive(max_output_tokens, "max_output_tokens", 65536)
    if not isinstance(api_key_env, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,127}", api_key_env):
        raise ValueError("api_key_env must be an environment variable name")
    if (isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds) or not 0 < timeout_seconds <= 900):
        raise ValueError("Invalid timeout_seconds")
    if not isinstance(model, str) or not model.strip() or reasoning_effort not in ("high", "xhigh"):
        raise ValueError("Invalid model or reasoning_effort")
    if type(retry_uncertain) is not bool or not isinstance(base_url, str):
        raise ValueError("Invalid retry_uncertain or base_url")
    url = urlsplit(base_url)
    if url.scheme != "https" or not url.hostname or url.username or url.password or url.query or url.fragment:
        raise ValueError("base_url must be HTTPS without credentials, query or fragment")
    api_dir = _absolute(api_dir)
    policy_dir = Path(policy_dir).expanduser().resolve()
    settings = dict(policy_dir=policy_dir, api_dir=api_dir, max_requests=max_requests,
                    model=model, base_url=base_url, reasoning_effort=reasoning_effort,
                    max_output_tokens=max_output_tokens, timeout_seconds=timeout_seconds,
                    max_attempts=max_attempts, retry_uncertain=retry_uncertain,
                    template_path=template_path, api_key_env=api_key_env)
    jobs = _preflight(manifest_path, api_dir, policy_dir, settings)
    worker = driver.run if run_worker is None else run_worker

    def execute(job):
        result = {"name": job["name"], "output_dir": str(job["output_dir"])}
        error = None
        try:
            answer = worker(job["bundle"], job["evidence"], job["output_dir"],
                            request_name=job["name"], **settings)
            if (not isinstance(answer, dict) or answer.get("status") != "valid_candidate"
                    or answer.get("accepted_for_training") is not False
                    or type(answer.get("cache_hit")) is not bool):
                raise ValueError("Worker did not return a validated candidate")
            result.update(status="valid", cache_hit=answer["cache_hit"])
        except Exception as exc:
            error = exc
            result.update(status="failed", error_code=_error_code(exc))
        result["artifacts"] = _artifacts(job["output_dir"], api_dir, error)
        return result

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(execute, jobs))
    return {"status": "completed" if all(row["status"] == "valid" for row in results) else "failed",
            "workers": workers, "max_requests": max_requests, "api_dir": str(api_dir),
            "jobs": results, "accepted_for_training": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--api-dir", type=Path, required=True, help="Shared private campaign with a non-increasing HTTP-attempt budget")
    parser.add_argument("--max-requests", type=int, required=True, help="Global campaign HTTP-attempt limit, including previous invocations")
    parser.add_argument("--workers", type=int, default=16, help="Concurrent jobs, from 1 to 32 (default: 16)")
    parser.add_argument("--policy-dir", type=Path, default=driver.POLICY_DIR)
    parser.add_argument("--template", dest="template_path", type=Path)
    parser.add_argument("--api-key-env", default=KEY_ENV)
    parser.add_argument("--model", help="Provider model name; otherwise GRANA_ANNOTATION_MODEL")
    parser.add_argument("--base-url", help="Responses API base URL; otherwise GRANA_ANNOTATION_BASE_URL")
    parser.add_argument("--reasoning-effort", choices=("high", "xhigh"), default="xhigh")
    parser.add_argument("--max-output-tokens", type=int, default=32768)
    parser.add_argument("--timeout-seconds", type=float, default=480)
    parser.add_argument("--max-attempts", type=int, default=1)
    parser.add_argument("--retry-uncertain", action="store_true", help="Explicitly permit a later uncertain-transport retry")
    args = vars(parser.parse_args(argv))
    args["manifest_path"] = args.pop("manifest")
    try:
        result = run(**args)
    except Exception as error:
        print(json.dumps({"status": "preflight_failed", "error_code": _error_code(error)}))
        raise SystemExit(2) from None
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    if result["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
