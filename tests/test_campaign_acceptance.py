# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cloud eventual consistency and acceptance remain separate from model retries."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from grana.training import campaign, campaign_cli as cli, campaign_report as reports


@pytest.mark.parametrize("success_on", [1, 2, None])
def test_acceptance_persists_local_then_cloud_with_bounded_retry(tmp_path, monkeypatch, success_on):
    plan = {"campaign": "fixture", "jobs": []}
    local = {"passed": True, "failures": [], "jobs": [{"name": "job", "logging_mode": "online"}]}
    monkeypatch.setattr(cli, "load_campaign", lambda _: (tmp_path, plan, {}))
    monkeypatch.setattr(reports, "report_campaign", lambda _: local)
    monkeypatch.setattr(campaign.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("acceptance must not launch training"))
    attempts = []
    def cloud(_, *, local_report):
        # Local verification is already durably recorded before any API request.
        paths = list((tmp_path / "acceptance").glob("*/local-report.json"))
        assert len(paths) == 1 and json.loads(paths[0].read_text()) == local_report == local
        attempts.append(len(attempts) + 1)
        passed = success_on == len(attempts)
        return {"passed": passed, "failures": [] if passed else ["upload not yet visible"],
                "remote": [{"url": "https://fixture/run"}] if passed else []}
    monkeypatch.setattr(reports, "audit_wandb", cloud)
    result = reports.accept_campaign(tmp_path, max_attempts=3, retry_seconds=0)
    assert result["passed"] is (success_on is not None)
    assert len(attempts) == (success_on or 3)
    assert result["training_restarted"] is False
    assert json.loads((tmp_path / "acceptance.json").read_text()) == result
    assert all(Path(attempt["report"]).is_file() for attempt in result["cloud_attempts"])


def test_local_acceptance_failure_never_requests_cloud_or_training(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "load_campaign", lambda _: (tmp_path, {"campaign": "fixture"}, {}))
    monkeypatch.setattr(reports, "report_campaign", lambda _: {"passed": False, "jobs": [], "failures": ["incomplete eval"]})
    monkeypatch.setattr(reports, "audit_wandb", lambda *args, **kwargs: pytest.fail("failed local report must stop acceptance"))
    result = reports.accept_campaign(tmp_path, retry_seconds=0)
    assert result["passed"] is False and result["cloud_attempts"] == []
    assert result["failures"] == ["incomplete eval"]


def test_offline_acceptance_does_not_authenticate_or_contact_wandb(tmp_path, monkeypatch):
    def forbidden(**kwargs):
        pytest.fail("offline and disabled runs must not contact W&B")
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(login=forbidden, Api=forbidden))
    report = {"passed": True, "failures": [], "jobs": [
        {"name": "offline", "logging_mode": "offline"}, {"name": "disabled", "logging_mode": "disabled"}]}
    result = reports.audit_wandb(tmp_path, local_report=report)
    assert result["passed"] is True and result["remote"] == []
    assert [job["name"] for job in result["remote_skipped"]] == ["offline", "disabled"]
    assert "remote" not in report  # Caller keeps the original local report.


@pytest.mark.parametrize("acceptance,status", [(None, "awaiting_acceptance"), (False, "failed"), (True, "succeeded")])
def test_observers_do_not_claim_success_before_final_acceptance(tmp_path, monkeypatch, acceptance, status):
    plan = {"campaign": "fixture", "completion_command": ["audit"], "jobs": [], "deadline_utc": "later"}
    state = {"status": "succeeded", "jobs": {}}
    monkeypatch.setattr(cli, "load_campaign", lambda _: (tmp_path, plan, state))
    if acceptance is not None:
        campaign.atomic_json(tmp_path / "acceptance.json", {"passed": acceptance, "finished_utc": "now"})
    snapshot = cli.status_snapshot(tmp_path)
    assert snapshot["status"] == snapshot["completion_status"] == status
    assert snapshot["recorded_status"] == "succeeded"


def test_acceptance_rechecks_frozen_runtime_before_cloud_success(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "load_campaign", lambda _: (tmp_path, {"campaign": "fixture"}, {}))
    monkeypatch.setattr(reports, "report_campaign", lambda _: {"passed": True, "jobs": [], "failures": []})
    monkeypatch.setattr(reports, "audit_wandb", lambda *args, **kwargs: {"passed": True, "remote": [], "failures": []})
    calls = []
    def changed(_):
        calls.append(True)
        if len(calls) > 1:
            raise ValueError("runtime changed during acceptance")
        return {"verified": True}
    monkeypatch.setattr(campaign, "verify_frozen_execution", changed)
    result = reports.accept_campaign(tmp_path, retry_seconds=0)
    assert result["passed"] is False
    assert result["failures"] == ["ValueError: runtime changed during acceptance"]


@pytest.mark.parametrize("tamper", [None, "runtime", "config", "source"])
def test_actual_worker_provenance_must_match_frozen_campaign(tmp_path, tamper):
    import hashlib
    import yaml
    from grana.training.runtime_fingerprint import capture_runtime_fingerprint
    output = tmp_path / "output"
    output.mkdir()
    expected = capture_runtime_fingerprint(package_names=[])
    actual = expected if tamper != "runtime" else capture_runtime_fingerprint(package_names=["not-installed-fixture"])
    runtime = tmp_path / "runtime.json"
    runtime.write_text(json.dumps(expected))
    (output / "environment.json").write_text(json.dumps({"runtime_fingerprint": actual}))
    cfg = {"training": {"output_dir": str(output), "max_steps": 250}}
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(yaml.safe_dump(cfg))
    actual_config = cfg if tamper != "config" else {"training": {**cfg["training"], "max_steps": 500}}
    (output / "resolved_config.json").write_text(json.dumps(actual_config))
    source = tmp_path / "grana" / "engine.py"
    source.parent.mkdir()
    source.write_text("# frozen fixture")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    fingerprint = tmp_path / "sources.json"
    fingerprint.write_text(json.dumps({"files_sha256": {str(source): digest}}))
    (output / "ray_run.json").write_text(json.dumps({"code_sha256": {
        "grana/engine.py": digest if tamper != "source" else "different-source"}}))
    plan = {"repo": str(tmp_path), "runtime_fingerprint": str(runtime), "source_fingerprint": str(fingerprint)}
    job = {"output_dir": str(output), "config": str(recipe)}
    if tamper:
        with pytest.raises(ValueError):
            campaign.verify_job_provenance(plan, job)
    else:
        result = campaign.verify_job_provenance(plan, job)
        assert result == {"runtime_fingerprint_sha256": expected["sha256"],
                          "resolved_config_equal": True, "source_files_verified": 1}


def flattened_runtime_summary(value):
    result = {}
    def visit(node, path):
        if isinstance(node, dict) and node:
            for key, child in node.items():
                visit(child, path + "." + key)
        else:
            result[path] = node
    visit(value, "environment.runtime_fingerprint")
    return result


@pytest.mark.parametrize("shape", ["nested", "dotted_object", "flat"])
def test_remote_runtime_shapes_preserve_literal_package_dots_and_leaf_types(shape):
    expected = {"schema": "fixture", "sha256": "fixture-digest", "python": {"version": "3.12.12"},
                "packages": {"example.pkg": {"installed": False}, "native": {
                    "installed": True, "observed_file_count": 3, "missing_files": [],
                    "large_native_files": [{"path": "lib/runtime.so", "ctime_ns": 1793776455799193885}]}}}
    summary = ({"environment": {"runtime_fingerprint": expected}} if shape == "nested" else
               {"environment.runtime_fingerprint": expected} if shape == "dotted_object" else
               flattened_runtime_summary(expected))
    actual = reports.runtime_fingerprint_from_summary(summary, expected)
    assert actual == expected
    assert "example.pkg" in actual["packages"] and "example" not in actual["packages"]
    assert actual["packages"]["example.pkg"]["installed"] is False
    assert type(actual["packages"]["native"]["observed_file_count"]) is int
    timestamp = actual["packages"]["native"]["large_native_files"][0]["ctime_ns"]
    assert type(timestamp) is int and timestamp == 1793776455799193885


@pytest.mark.parametrize("change", ["absent", "missing", "extra"])
def test_remote_runtime_flattening_never_fills_missing_or_discards_extra_fields(change):
    expected = {"schema": "fixture", "sha256": "digest", "packages": {"example.pkg": {"installed": False}}}
    summary = flattened_runtime_summary(expected)
    if change == "absent":
        summary = {"optimizer_steps": 12}
    elif change == "missing":
        summary.pop("environment.runtime_fingerprint.packages.example.pkg.installed")
    else:
        summary["environment.runtime_fingerprint.packages.unexpected.installed"] = True
    with pytest.raises(ValueError, match="missing|leaves differ"):
        reports.runtime_fingerprint_from_summary(summary, expected)


def test_flat_runtime_rejects_colliding_literal_and_nested_dotted_keys():
    expected = {"packages": {"example.pkg": {"installed": True}, "example": {"pkg": {"installed": False}}}}
    with pytest.raises(ValueError, match="ambiguous dotted"):
        reports.runtime_fingerprint_from_summary(flattened_runtime_summary(expected), expected)


@pytest.mark.parametrize("payload", [None, [], "invalid"])
@pytest.mark.parametrize("dotted", [False, True])
def test_remote_runtime_object_cannot_be_missing_or_replaced_by_local_capture(payload, dotted):
    summary = ({"environment.runtime_fingerprint": payload} if dotted else
               {"environment": {"runtime_fingerprint": payload}})
    with pytest.raises(ValueError, match="must be an object"):
        reports.runtime_fingerprint_from_summary(summary, {"schema": "fixture"})
