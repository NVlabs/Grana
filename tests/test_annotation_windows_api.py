# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Synthetic bundles and injected workers/transports; no provider calls."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import threading

import httpx
from PIL import Image
import pytest

from test_annotation_api import provider_configuration  # noqa: F401

from grana.annotation.api import AnnotationAPIError, KEY_ENV
from scripts import annotation_skill as skill
from scripts import annotation_skill_api as driver
from scripts import annotation_windows_api as batch


@pytest.fixture
def inputs(tmp_path):
    policy = tmp_path / "policy"
    for name in skill.POLICY_FILES:
        path = policy / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("Synthetic frozen annotation policy. Preserve unknown evidence.\n")
    context = json.loads((skill.POLICY_DIR / "references/synthetic-context.json").read_text())
    candidate = json.loads((skill.POLICY_DIR / "references/synthetic-candidate.json").read_text())
    jobs = []
    for number in range(6):
        folder = tmp_path / f"input-{number}"
        folder.mkdir()
        current = deepcopy(context)
        current["source_identity"] = {"source_root": str(tmp_path / f"raw-{number}")}
        bundle = folder / "bundle.json"
        bundle.write_text(json.dumps(skill.prepare_bundle(current, policy, mode="combined")))
        rows = []
        for frame in current["evidence_frames"]:
            image = folder / f"frame-{frame}.png"
            Image.new("RGB", (4, 3), (number, frame, 0)).save(image)
            rows.append({"frame": frame, "image_path": image.name,
                         "sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
                         "timestamp": current["time_edges"][frame], "view_names": ["top"]})
        evidence = folder / "evidence.json"
        evidence.write_text(json.dumps({"schema_version": driver.EVIDENCE_SCHEMA, "images": rows}))
        jobs.append({"name": f"window-{number}", "bundle": str(bundle),
                     "evidence": str(evidence), "output_dir": str(tmp_path / f"output-{number}")})
    manifest = tmp_path / "jobs.json"
    manifest.write_text(json.dumps(jobs))
    return {"manifest_path": manifest, "api_dir": tmp_path / "campaign", "max_requests": 3,
            "policy_dir": policy}, jobs, candidate


def valid():
    return {"status": "valid_candidate", "cache_hit": False, "accepted_for_training": False}


def test_workers_are_bounded_and_all_inputs_preflight_before_any_call(inputs, monkeypatch):
    args, jobs, _ = inputs
    prepared, calls = [], []
    original = driver.prepare_request

    def prepare(*values):
        answer = original(*values)
        prepared.append(values[0])
        return answer

    monkeypatch.setattr(driver, "prepare_request", prepare)
    barrier = threading.Barrier(2)
    lock = threading.Lock()
    active = peak = 0

    def worker(bundle, evidence, output, **settings):
        nonlocal active, peak
        assert len(prepared) == len(jobs)
        with lock:
            active += 1
            peak = max(peak, active)
            calls.append(settings)
        barrier.wait(timeout=5)
        with lock:
            active -= 1
        return valid()

    result = batch.run(**args, workers=2, run_worker=worker)
    assert result["status"] == "completed" and peak == 2
    assert [row["name"] for row in result["jobs"]] == [row["name"] for row in jobs]
    assert len(calls) == len(jobs)
    assert {row["request_name"] for row in calls} == {row["name"] for row in jobs}
    assert all(row["api_dir"] == args["api_dir"] and row["max_requests"] == 3 for row in calls)
    assert all(row["max_attempts"] == 1 and row["retry_uncertain"] is False for row in calls)
    assert result["accepted_for_training"] is False


@pytest.mark.parametrize("workers", [0, 33, True, 2.5])
def test_invalid_worker_limit_never_launches(inputs, workers):
    args, _, _ = inputs
    with pytest.raises(ValueError):
        batch.run(**args, workers=workers, run_worker=lambda *_a, **_k: pytest.fail("no calls"))


@pytest.mark.parametrize("damage", ["duplicate_name", "duplicate_output", "nested_output", "symlink_output", "relative",
                                    "missing", "fields", "unsafe_name", "raw_output", "repo_output",
                                    "api_source", "bad_image"])
def test_late_invalid_job_prevents_all_calls_and_output_creation(inputs, damage, tmp_path):
    args, jobs, _ = inputs
    if damage == "duplicate_name":
        jobs[-1]["name"] = jobs[0]["name"]
    elif damage == "duplicate_output":
        jobs[-1]["output_dir"] = jobs[0]["output_dir"]
    elif damage == "nested_output":
        jobs[-1]["output_dir"] = str(Path(jobs[0]["output_dir"]) / "child")
    elif damage == "symlink_output":
        alias = tmp_path / "output-alias"
        alias.symlink_to(jobs[0]["output_dir"], target_is_directory=True)
        jobs[-1]["output_dir"] = str(alias)
    elif damage == "relative":
        jobs[-1]["bundle"] = "bundle.json"
    elif damage == "missing":
        jobs[-1]["evidence"] = str(tmp_path / "missing.json")
    elif damage == "fields":
        jobs[-1]["extra"] = "unsupported"
    elif damage == "unsafe_name":
        jobs[-1]["name"] = "../other"
    elif damage == "raw_output":
        # A job may not write inside another job's raw source either.
        jobs[-1]["output_dir"] = str(tmp_path / "raw-0" / "outputs")
    elif damage == "repo_output":
        repo = tmp_path / "checkout"
        (repo / ".git").mkdir(parents=True)
        jobs[-1]["output_dir"] = str(repo / "outputs")
    elif damage == "api_source":
        args["api_dir"] = tmp_path / "raw-0" / "campaign"
    else:
        path = Path(jobs[-1]["evidence"])
        evidence = json.loads(path.read_text())
        evidence["images"][0]["sha256"] = "0" * 64
        path.write_text(json.dumps(evidence))
    args["manifest_path"].write_text(json.dumps(jobs))
    with pytest.raises(ValueError):
        batch.run(**args, run_worker=lambda *_a, **_k: pytest.fail("no calls"))
    assert not args["api_dir"].exists()
    assert not any(Path(row["output_dir"]).exists() for row in jobs)


def test_errors_are_collected_without_accepting_unknown_results_or_leaking_payloads(inputs):
    args, jobs, _ = inputs
    secret = "PRIVATE_KEY data:image/png;base64,PRIVATE_PROVIDER_PAYLOAD"
    receipt = args["api_dir"] / "requests" / "safe" / "receipt.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_text("{}")
    calls = []

    def worker(_bundle, _evidence, _output, **settings):
        name = settings["request_name"]
        calls.append(name)
        if name == "window-0":
            raise AnnotationAPIError("transport_uncertain", receipt)
        if name == "window-1":
            raise ValueError(secret)
        if name == "window-2":
            return {"status": "completed", "data": secret}
        if name == "window-3":
            raise RuntimeError(secret)
        if name == "window-4":
            raise AnnotationAPIError(secret, Path("/outside/private.json"))
        return valid()

    result = batch.run(**args, run_worker=worker)
    assert result["status"] == "failed" and len(calls) == len(jobs)
    assert [row["status"] for row in result["jobs"]] == ["failed"] * 5 + ["valid"]
    assert [row.get("error_code") for row in result["jobs"][:5]] == [
        "transport_uncertain", "validation_error", "validation_error", "unexpected_error", "api_error"]
    assert result["jobs"][0]["artifacts"]["receipt"] == str(receipt)
    assert "receipt" not in result["jobs"][4]["artifacts"]
    encoded = json.dumps(result)
    assert secret not in encoded and "/outside/private.json" not in encoded


def test_incompatible_existing_output_prevents_all_worker_calls(inputs):
    args, jobs, _ = inputs
    output = Path(jobs[-1]["output_dir"])
    output.mkdir()
    prepared = output / "prepared.json"
    prepared.write_text('{"bundle_sha256":"old-input"}')
    with pytest.raises(ValueError, match="Existing prepared artifact differs"):
        batch.run(**args, run_worker=lambda *_a, **_k: pytest.fail("no calls"))
    assert prepared.read_text() == '{"bundle_sha256":"old-input"}'
    assert not args["api_dir"].exists()


def test_shared_budget_bounds_mocked_http_and_completed_request_is_reused(inputs, monkeypatch):
    args, jobs, candidate = inputs
    jobs = jobs[:3]
    args["manifest_path"].write_text(json.dumps(jobs))
    args["max_requests"] = 1
    monkeypatch.setenv(KEY_ENV, "synthetic-test-key")
    seen = []

    def handler(request):
        payload = json.loads(request.content)
        seen.append(payload)
        return httpx.Response(200, json={
            "id": "resp-synthetic", "status": "completed", "model": payload["model"],
            "reasoning": payload["reasoning"],
            "output": [{"type": "message", "content": [{"type": "output_text", "text": json.dumps(candidate)}]}],
            "usage": {"input_tokens": 1, "output_tokens": 1}})

    def worker(*values, **settings):
        return driver.run(*values, **settings, transport=httpx.MockTransport(handler))

    result = batch.run(**args, run_worker=worker)
    assert result["status"] == "failed" and len(seen) == 1
    assert sum(row["status"] == "valid" for row in result["jobs"]) == 1
    assert all(row.get("error_code") == "campaign_budget_exhausted"
               for row in result["jobs"] if row["status"] == "failed")
    budget = json.loads((args["api_dir"] / "budget.json").read_text())
    assert budget["max_requests"] == 1 and len(budget["reservations"]) == 1
    completed = next(row for row in result["jobs"] if row["status"] == "valid")
    assert Path(completed["artifacts"]["receipt"]).is_file()
    args["manifest_path"].write_text(json.dumps([next(row for row in jobs if row["name"] == completed["name"])]))
    monkeypatch.delenv(KEY_ENV)
    resumed = batch.run(**args, run_worker=worker)
    assert resumed["status"] == "completed" and resumed["jobs"][0]["cache_hit"] is True
    assert len(seen) == 1


def test_two_concurrent_outputs_stay_isolated_and_resume_at_full_budget(inputs, monkeypatch):
    args, jobs, candidate = inputs
    jobs = jobs[:2]
    args["manifest_path"].write_text(json.dumps(jobs))
    args["max_requests"] = 2
    monkeypatch.setenv(KEY_ENV, "synthetic-test-key")
    barrier = threading.Barrier(2)
    seen = []

    def handler(request):
        payload = json.loads(request.content)
        text = payload["input"][0]["content"][0]["text"]
        input_id = json.loads(text.split("\n", 1)[1])["input_id"]
        seen.append(input_id)
        barrier.wait(timeout=5)
        answer = deepcopy(candidate)
        answer["stages"]["summary"] = f"Synthetic response for {input_id}"
        return httpx.Response(200, json={
            "id": f"resp-{input_id}", "status": "completed", "model": payload["model"],
            "reasoning": payload["reasoning"],
            "output": [{"type": "message", "content": [{"type": "output_text", "text": json.dumps(answer)}]}]})

    def worker(*values, **settings):
        return driver.run(*values, **settings, transport=httpx.MockTransport(handler))

    first = batch.run(**args, run_worker=worker)
    assert first["status"] == "completed" and len(set(seen)) == 2
    receipts = []
    for job, result in zip(jobs, first["jobs"], strict=True):
        input_id = json.loads(Path(job["bundle"]).read_text())["context_sha256"]
        output = Path(job["output_dir"])
        assert json.loads((output / "candidate.json").read_text())["stages"]["summary"] == f"Synthetic response for {input_id}"
        assert all(Path(path).is_relative_to(output) for key, path in result["artifacts"].items() if key != "receipt")
        receipts.append(result["artifacts"]["receipt"])
    assert len(set(receipts)) == 2
    budget = json.loads((args["api_dir"] / "budget.json").read_text())
    assert len(budget["reservations"]) == budget["max_requests"] == 2
    roots = [args["api_dir"], *(Path(job["output_dir"]) for job in jobs)]
    before = {path: path.read_bytes() for root in roots for path in root.rglob("*.json")}

    monkeypatch.delenv(KEY_ENV)

    def cached_worker(*values, **settings):
        return driver.run(*values, **settings, transport=httpx.MockTransport(
            lambda _: pytest.fail("Full-budget cache reuse must not send HTTP")))

    resumed = batch.run(**args, run_worker=cached_worker)
    assert resumed["status"] == "completed"
    assert all(row["cache_hit"] is True for row in resumed["jobs"])
    assert before == {path: path.read_bytes() for root in roots for path in root.rglob("*.json")}


def test_default_sixteen_workers_overlap_without_exceeding_shared_budget(inputs, monkeypatch, tmp_path):
    args, templates, candidate = inputs
    jobs = [templates[number % len(templates)] | {
        "name": f"parallel-{number}", "output_dir": str(tmp_path / f"parallel-output-{number}")}
        for number in range(20)]
    args["manifest_path"].write_text(json.dumps(jobs))
    args["max_requests"] = 16
    monkeypatch.setenv(KEY_ENV, "synthetic-test-key")
    barrier = threading.Barrier(16)
    lock = threading.Lock()
    active = peak = count = 0

    def handler(request):
        nonlocal active, peak, count
        payload = json.loads(request.content)
        with lock:
            active += 1
            count += 1
            peak = max(peak, active)
        barrier.wait(timeout=10)
        with lock:
            active -= 1
        return httpx.Response(200, json={
            "id": "resp-synthetic", "status": "completed", "model": payload["model"],
            "reasoning": payload["reasoning"],
            "output": [{"type": "message", "content": [{"type": "output_text", "text": json.dumps(candidate)}]}]})

    def worker(*values, **settings):
        return driver.run(*values, **settings, transport=httpx.MockTransport(handler))

    result = batch.run(**args, run_worker=worker)
    assert result["workers"] == peak == count == 16
    assert result["status"] == "failed"
    assert sum(row["status"] == "valid" for row in result["jobs"]) == 16
    failed = [row for row in result["jobs"] if row["status"] == "failed"]
    assert len(failed) == 4
    assert all(row["error_code"] == "campaign_budget_exhausted" for row in failed)
    budget = json.loads((args["api_dir"] / "budget.json").read_text())
    assert len(budget["reservations"]) == budget["max_requests"] == 16


def test_cli_returns_failure_and_only_safe_summary(inputs, monkeypatch, capsys):
    args, _, _ = inputs
    monkeypatch.setattr(batch, "run", lambda **_args: {"status": "failed", "jobs": []})
    cli = ["--manifest", str(args["manifest_path"]), "--api-dir", str(args["api_dir"]), "--max-requests", "1"]
    with pytest.raises(SystemExit) as exc:
        batch.main(cli)
    assert exc.value.code == 1 and json.loads(capsys.readouterr().out)["status"] == "failed"

    def broken(**_args):
        raise ValueError("PRIVATE_KEY data:image/png;base64,PRIVATE_PAYLOAD")

    monkeypatch.setattr(batch, "run", broken)
    with pytest.raises(SystemExit) as exc:
        batch.main(cli)
    assert exc.value.code == 2
    assert json.loads(capsys.readouterr().out) == {"status": "preflight_failed", "error_code": "validation_error"}
