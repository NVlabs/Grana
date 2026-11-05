# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit campaign acceptance must reject missing evaluations and weights."""
from copy import deepcopy
import json

import pytest
import torch

from grana.modeling.checkpoint import tensor_digest
from grana.training.campaign_report import summarize_job, verify_expected_job_contract, verify_remote_config


def save(path, value):
    path.write_text(json.dumps(value))


def test_remote_config_accepts_and_records_only_omitted_empty_mappings():
    expected = {"evaluation": {"deploy_dense": {"group_by_cache": {}, "enabled": True}},
                "data": {"caches": [{"path": "/cache", "task_weights": {"A": 1.0}}]}}
    remote = deepcopy(expected)
    del remote["evaluation"]["deploy_dense"]["group_by_cache"]
    remote["data"]["caches"][0]["task_weights"]["A"] = 1
    before = deepcopy(remote)
    assert verify_remote_config(remote, expected) == {
        "resolved_config_equal": True,
        "remote_omitted_empty_config_mappings": ["evaluation.deploy_dense.group_by_cache"]}
    assert remote == before and expected["evaluation"]["deploy_dense"]["group_by_cache"] == {}


@pytest.mark.parametrize("remote,expected", [
    ({}, {"mapping": {"cache": "holdout"}}),
    ({}, {"caches": []}),
    ({"seed": 41}, {"seed": 42}),
    ({"enabled": 1}, {"enabled": True}),
    ({"caches": ["b", "a"]}, {"caches": ["a", "b"]}),
    ({"mapping": {"extra": 1}}, {"mapping": {}}),
    ({"extra": {}}, {}),
])
def test_remote_config_still_rejects_actual_configuration_changes(remote, expected):
    with pytest.raises(ValueError, match="Remote resolved configuration"):
        verify_remote_config(remote, expected)


@pytest.fixture
def full_job(tmp_path):
    window = {"cache": "/exact/cache", "task": "unchanged", "trajectory_id": "heldout", "poi": 0, "roles": ["B", "C", "D"]}
    plan = {"settings": {"metric_scope": "full_heldout_dense"}, "window_count": 1, "model_calls": 3,
            "role_window_counts": {"A": 0, "B": 1, "C": 1, "D": 1}, "windows": [window], "plan_sha256": "fixed"}
    save(tmp_path / "evaluation_plan.json", plan)
    events = []
    for step in (250, 500):
        result = {"plan_sha256": "fixed", "provenance": {"mode": "during_training"},
                  "records": [dict(window, role=role) for role in ("B", "C", "D")]}
        save(tmp_path / f"evaluation-{step:06d}.json", result)
        save(tmp_path / f"deploy-dense-evaluation-{step:06d}.json", result)
        events.append({"optimizer_steps": step, "model_calls": 3, "deploy_dense_model_calls": 3,
                       "deploy_dense_reused_primary_predictions": True})
    return {"expected_eval_steps": [250, 500], "expected_pois_per_event": 1}, {"evaluation": {"mode": "dense"}}, {"evaluation_events": events}


def test_explicit_full_heldout_acceptance_requires_both_events_and_single_pass(full_job, tmp_path):
    job, cfg, result = full_job
    assert verify_expected_job_contract(job, cfg, result, tmp_path) == []
    truncated = deepcopy(result)
    truncated["evaluation_events"].pop(0)
    with pytest.raises(ValueError, match="required optimizer steps"):
        verify_expected_job_contract(job, cfg, truncated, tmp_path)
    repeated = deepcopy(result)
    repeated["evaluation_events"][0]["deploy_dense_reused_primary_predictions"] = False
    with pytest.raises(ValueError, match="exactly one full primary"):
        verify_expected_job_contract(job, cfg, repeated, tmp_path)


def test_full_heldout_acceptance_requires_identical_aliases_and_exact_plan_coverage(full_job, tmp_path):
    job, cfg, result = full_job
    path = tmp_path / "deploy-dense-evaluation-000250.json"
    saved = json.loads(path.read_text())
    saved["records"].pop()
    save(path, saved)
    with pytest.raises(ValueError, match="Primary and dense predictions differ"):
        verify_expected_job_contract(job, cfg, result, tmp_path)
    save(tmp_path / "evaluation-000250.json", saved)
    with pytest.raises(ValueError, match="exact full held-out plan"):
        verify_expected_job_contract(job, cfg, result, tmp_path)


def test_all_requested_checkpoints_require_real_tensors_matching_metadata(tmp_path):
    from safetensors.torch import save_file

    paths = []
    for step in (250, 500):
        path = tmp_path / f"checkpoint-{step:06d}"
        path.mkdir()
        tensors = {"weight": torch.tensor([float(step)])}
        save_file(tensors, str(path / "weights.safetensors"))
        save(path / "metadata.json", {"format": "grana-weights-v1", "source_step": step, "weights_sha256": tensor_digest(tensors)})
        paths.append(str(path))
    job = {"expected_checkpoints": paths}
    assert [row["optimizer_steps"] for row in verify_expected_job_contract(job, {}, {}, tmp_path)] == [250, 500]
    save_file({"weight": torch.tensor([-1.])}, str(tmp_path / "checkpoint-000250/weights.safetensors"))
    with pytest.raises(ValueError, match="tensor digest differs"):
        verify_expected_job_contract(job, {}, {}, tmp_path)


@pytest.fixture
def completed_job(tmp_path):
    output = tmp_path / "raw-hotwheels"
    output.mkdir()
    checkpoint = output / "checkpoint-000001"
    checkpoint.mkdir()
    save(checkpoint / "metadata.json", {"source_step": 1, "weights_sha256": "trained-hotwheels"})
    save(output / "result.json", {"status": "completed", "optimizer_steps": 1,
                                  "checkpoint_reload_equal": True, "checkpoint_prediction_parity": True,
                                  "checkpoint_load": {"weights_sha256": "trained-hotwheels"},
                                  "training_compute_seconds": 1., "total_elapsed_seconds": 2., "evaluation_events": []})
    save(output / "ray_run.json", {"status": "succeeded"})
    save(output / "resolved_config.json", {"evaluation": {"enabled": False, "renders": {"enabled": False}}})
    save(output / "wandb_run.json", {"mode": "disabled"})
    (output / "training_metrics.jsonl").write_text(json.dumps({"optimizer_steps": 1, "update_seconds": 1., "gpu_peak_allocated_gib": 2.}) + "\n")
    job = {"name": "hotwheels_d15", "size": "4B", "task": "hotwheels", "stage": "posttrain",
           "output_dir": str(output), "expected_steps": 1, "expected_checkpoint": str(checkpoint),
           "dependency": "battatplane_d15", "source_pretrain": None}
    upstream = {"name": "battatplane_d15", "size": "4B", "stage": "posttrain"}
    return job, {"battatplane_d15": upstream}


def test_raw_initialization_report_accepts_scheduling_dependency_without_loading_weights(completed_job):
    job, jobs = completed_job
    report = summarize_job(job, jobs)
    assert report["source_pretrain"] is None
    assert report["name"] == "hotwheels_d15" and report["optimizer_steps"] == 1
    assert report["final_evaluation_report"]["primary"]["status"] == "unavailable"


def test_campaign_reports_primary_accumulated_quality_without_rewriting_metrics(completed_job, monkeypatch):
    from pathlib import Path

    job, jobs = completed_job
    output = Path(job["output_dir"])
    original = json.loads((output / "result.json").read_text())
    original["evaluation_events"] = [{"optimizer_steps": 1, "training_elapsed_seconds": 1.}]
    save(output / "result.json", original)
    save(output / "resolved_config.json", {"evaluation": {"enabled": True, "mode": "dense", "renders": {"enabled": False}}})
    metrics = {"role_D_poi_rmse": .23, "accumulated/overall/raw_rmse": .4,
               "accumulated/overall/normalized_rmse": .04}
    # Verification has its own independent raw-record tests. Here its validated
    # boundary isolates report selection and compatibility of the original map.
    monkeypatch.setattr("grana.training.campaign_report.verify_evaluation", lambda *args: {
        "training_elapsed_seconds": 1., "elapsed_seconds": .5, "metrics": metrics})
    report = summarize_job(job, jobs)
    assert report["final_heldout_metrics"] == metrics
    assert report["evaluation_events"][0]["metrics"] == metrics
    assert report["final_evaluation_report"] == report["evaluation_events"][0]["evaluation_report"]
    assert report["final_evaluation_report"]["primary"]["raw_rmse"] == .4
    assert report["final_evaluation_report"]["primary"]["normalized_rmse"] == .04


@pytest.mark.parametrize("legacy", [False, True])
def test_pretrained_source_receipt_remains_required_for_explicit_and_legacy_transfer(completed_job, tmp_path, legacy):
    job, jobs = completed_job
    pretrained = tmp_path / "pretrain/checkpoint-000500"
    pretrained.mkdir(parents=True)
    save(pretrained / "metadata.json", {"weights_sha256": "loaded-pretrain"})
    jobs["pretrain"] = {"size": "4B", "stage": "pretrain", "expected_checkpoint": str(pretrained)}
    output = tmp_path / "raw-hotwheels"
    initialization = {"report": {"weights_sha256": "loaded-pretrain",
                                  "heads": {name: {"outcome": "loaded"} for name in ("progress_head", "temporal_head")}}}
    save(output / "initialization.json", initialization)
    if legacy:
        del job["source_pretrain"]
        job["dependency"] = "pretrain"
    else:
        job["source_pretrain"] = "pretrain"
    assert summarize_job(job, jobs)["source_pretrain"] == "pretrain"
    initialization["report"]["weights_sha256"] = "unrelated-weights"
    save(output / "initialization.json", initialization)
    with pytest.raises(ValueError, match="did not load its own pretrain"):
        summarize_job(job, jobs)
