# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standalone suites keep cohorts, checkpoint identity and completion explicit."""
from copy import deepcopy
import json
from types import SimpleNamespace
import sys

import numpy as np
import pytest

from grana.config import DEFAULTS
from grana.data.cache import Trajectory
from grana.training.distributed_evaluation import split_plan
from scripts.training import evaluate_checkpoint_suites as suites


@pytest.fixture
def cohort(tmp_path, monkeypatch):
    config = deepcopy(DEFAULTS)
    config["data"].update(caches=["train"], views=["top"], max_pixels=None)
    config["evaluation"].update(seed=71, stride_pairs=[[3, 1]])
    config["training"]["num_workers"] = 2
    rows = []
    for task in ("first", "second"):
        for index in range(3):
            path = tmp_path / f"{task}-{index}.npy"
            pixels = np.zeros((11, 2, 3, 3), dtype=np.uint8)
            np.save(path, pixels)
            rows.append(Trajectory(tmp_path / "heldout", path.stem, task, 11, (path,), pixels.shape,
                                   ("top",), ("A", "B", "C"), tuple(i / 10 for i in range(11)),
                                   (0,) * 5 + (1,) * 6, (.3,) * 11, ("First", "Second"), "Complete task."))
    monkeypatch.setattr("grana.data.cache.CachedTrajectories", lambda caches, **kwargs:
                        SimpleNamespace(records=() if caches == ["train"] else tuple(rows)))
    return config, {"name": "heldout", "caches": ["eval"], "roles": ["C", "D"]}


@pytest.mark.parametrize("workers", [1, 2, 4])
def test_full_cohort_keeps_routes_tail_and_configured_sampling(cohort, workers):
    config, spec = cohort
    config["training"]["num_workers"] = workers
    original = deepcopy(config)
    spec.update(batch_size=4, frame_stride=3)
    plans, summaries = suites.prepare(config, [spec])
    plan, summary = plans["heldout"], summaries["heldout"]
    assert config == original
    assert (summary["trajectories"], summary["windows"], summary["role_rows"]) == (6, 30, 60)
    assert summary["metric_scope"] == "full_heldout_dense"
    assert plan.settings["seed"] == 71
    assert {window.stride_pair for window in plan.windows} == {(3, 1)}
    assert {window.roles for window in plan.windows} == {("C", "D")}
    assert {window.poi for window in plan.windows} == {0, 3, 6, 9, 10}
    shards = split_plan(plan, workers)
    assert tuple(window for shard in shards for window in shard.plan.windows) == plan.windows
    assert shards[-1].plan.windows[-1] == plan.windows[-1]


def test_bounded_sampling_is_deterministic_and_allows_empty_ranks(cohort):
    config, spec = cohort
    config["training"]["num_workers"] = 4
    spec.update(roles=["A"], max_trajectories_per_task=1, points_per_trajectory=5)
    plans, summaries = suites.prepare(config, [spec])
    plan, summary = plans["heldout"], summaries["heldout"]
    assert summary["trajectories"] == 2 and summary["windows"] == 10
    assert summary["batch_size"] == 64 and summary["active_workers"] == 1
    assert summary["metric_scope"] == "fixed_heldout_uniform_poi_cohort"
    assert {window.poi for window in plan.windows} == {0, 2, 5, 7, 10}
    assert {window.roles for window in plan.windows} == {("A",)}
    assert suites.prepare(config, [spec])[0]["heldout"].manifest() == plan.manifest()


@pytest.mark.parametrize("bounded", [False, True])
@pytest.mark.parametrize("oracle", [False, True])
def test_dense_and_bounded_share_the_same_oracle_policy(cohort, bounded, oracle):
    config, spec = cohort
    spec.update(roles=["B", "C", "D"], include_oracle=oracle)
    spec.update({"points_per_trajectory": 3} if bounded else {"frame_stride": 4})
    plan = suites.prepare(config, [spec])[0]["heldout"]
    assert {window.roles for window in plan.windows} == ({("B", "C", "D")} if oracle else {("C", "D")})


@pytest.mark.parametrize("change", [
    {"name": "../outside"}, {"name": "plans"}, {"name": "driver.json"}, {"batch_size": 0}, {"batch_size": True},
    {"max_trajectories_per_task": 0}, {"points_per_trajectory": 1},
    {"points_per_trajectory": 4, "frame_stride": 2}, {"roles": ["D"]},
    {"roles": ["A", "A"]}, {"caches": []}, {"batch_szie": 64}, {"seed": True},
])
def test_invalid_specs_are_rejected_before_execution(cohort, change):
    config, spec = cohort
    with pytest.raises(ValueError):
        suites.normalize_specs(config, [{**spec, **change}])


def test_duplicate_and_empty_suites_rejected(cohort):
    config, spec = cohort
    for specs in ([], [spec, spec]):
        with pytest.raises(ValueError):
            suites.normalize_specs(config, specs)


@pytest.fixture
def source(tmp_path):
    config = deepcopy(DEFAULTS)
    source = tmp_path / "source"
    checkpoint = source / "checkpoint-000002"
    checkpoint.mkdir(parents=True)
    suites.write(checkpoint / "metadata.json", {"format": "grana-weights-v1", "source_step": 2,
                 "model": config["model"], "weights_sha256": "checkpoint-digest"})
    (checkpoint / "weights.safetensors").write_bytes(b"read by load_weights in the GPU worker")
    (source / "training_metrics.jsonl").write_text(json.dumps(
        {"optimizer_steps": 2, "checkpoint_path": str(checkpoint), "elapsed_seconds": 31.5}) + "\n")
    return config, source, checkpoint


def test_unfinished_local_checkpoint_needs_no_wandb_source_and_is_not_modified(source):
    config, root, checkpoint = source
    before = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
    path, receipt = suites.checkpoint_source(config, root, 2)
    assert path == checkpoint
    assert receipt == {"source_run": f"local:{root}", "training_elapsed_seconds": 31.5,
                       "weights_sha256": "checkpoint-digest"}
    assert before == {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
    config["model"]["revision"] = "different-backbone"
    with pytest.raises(ValueError, match="revision"):
        suites.checkpoint_source(config, root, 2)


def test_duplicate_checkpoint_receipt_is_rejected(source):
    config, root, _ = source
    path = root / "training_metrics.jsonl"
    path.write_text(path.read_text() * 2)
    with pytest.raises(ValueError, match="one committed"):
        suites.checkpoint_source(config, root, 2)


@pytest.mark.parametrize("failure", ["no_worker_report", "missing_suite", "wrong_plan", "unfinished_upload", "ray_error"])
def test_normal_ray_exit_is_not_sufficient_for_success(tmp_path, failure):
    spec = {"name": "heldout"}
    prepared = {"heldout": {"plan_sha256": "plan"}}
    result = SimpleNamespace(error=None, metrics={"completed_suites": 1})
    report = {"status": "succeeded", "plan_sha256": "plan", "wandb": {"status": "finished"}}
    if failure == "no_worker_report":
        result.metrics = {}
    elif failure == "wrong_plan":
        report["plan_sha256"] = "another-plan"
    elif failure == "unfinished_upload":
        report["wandb"]["status"] = "running"
    elif failure == "ray_error":
        result.error = "worker failed"
    if failure != "missing_suite":
        suites.write(tmp_path / "heldout" / "report.json", report)
    with pytest.raises(RuntimeError):
        suites.verify_completion(result, tmp_path, [spec], prepared)


def test_success_requires_all_suites_and_finished_logging(tmp_path):
    specs = [{"name": name} for name in ("full", "bounded")]
    prepared = {spec["name"]: {"plan_sha256": spec["name"]} for spec in specs}
    for name, status in (("full", "finished"), ("bounded", "disabled")):
        suites.write(tmp_path / name / "report.json",
                     {"status": "succeeded", "plan_sha256": name, "wandb": {"status": status}})
    suites.verify_completion(SimpleNamespace(error=None, metrics={"completed_suites": 2}),
                             tmp_path, specs, prepared)


@pytest.mark.parametrize("upload_error", [False, True])
def test_interruption_closes_only_this_attempts_unfinished_cloud_runs(tmp_path, monkeypatch, upload_error):
    specs = [{"name": name} for name in ("running", "complete", "disabled")]
    for name, mode, status in (("running", "online", "running"), ("complete", "online", "finished"),
                                ("disabled", "disabled", "disabled")):
        suites.write(tmp_path / name / "wandb_run.json",
                     {"id": name, "project": "project", "entity": "entity", "mode": mode, "status": status})
    calls, summary = [], {}

    def init(**kwargs):
        calls.append(kwargs)
        if upload_error:
            raise RuntimeError("network unavailable")
        return SimpleNamespace(summary=summary, finish=lambda **kwargs: calls.append(kwargs))

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=init, Settings=lambda **kwargs: kwargs))
    errors = suites.close_incomplete_runs(tmp_path, specs, status="cancelled", error="KeyboardInterrupt")
    assert calls[0]["id"] == "running" and calls[0]["resume"] == "must"
    assert len(errors) == int(upload_error)
    if not upload_error:
        assert summary["evaluation_complete"] is False and calls[1] == {"exit_code": 130}
        assert suites.read(tmp_path / "running" / "wandb_run.json")["evaluation_status"] == "cancelled"
    assert suites.read(tmp_path / "complete" / "wandb_run.json")["status"] == "finished"


def test_interrupted_receipt_write_does_not_hide_the_original_failure(tmp_path):
    path = tmp_path / "heldout" / "wandb_run.json"
    path.parent.mkdir()
    path.write_text('{"mode": "online",')
    errors = suites.close_incomplete_runs(tmp_path, [{"name": "heldout"}], status="cancelled", error="original error")
    assert len(errors) == 1 and "JSONDecodeError" in errors[0]["error"]


@pytest.mark.parametrize("outcome,expected_status", [("success", "succeeded"), ("partial", "incomplete"),
                                                     ("interrupt", "cancelled"), ("failure", "failed")])
def test_driver_cleanup_and_terminal_status(source, tmp_path, monkeypatch, outcome, expected_status):
    import ray
    import ray.train.torch
    import torch

    config, root, _ = source
    config["training"]["num_workers"] = 9  # Evaluation defaults to visible GPUs, not this training value.
    specs = [{"name": "heldout", "caches": ["eval"], "roles": ["A"]}]
    prepared = {"heldout": {"plan_sha256": "plan"}}
    specs_path, prepared_path, output = tmp_path / "specs.json", tmp_path / "prepared.json", tmp_path / "output"
    suites.write(specs_path, specs)
    suites.write(prepared_path, {"suites": prepared})
    monkeypatch.setattr("grana.config.load_config", lambda path: deepcopy(config))
    monkeypatch.setattr("grana.runtime.require_owned_environment", lambda: None)
    monkeypatch.setattr(suites, "prepare", lambda config, specs: ({}, prepared))
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda index: SimpleNamespace(
        uuid=f"00000000-0000-0000-0000-{index:012d}"))
    calls = []
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    monkeypatch.setenv("RAY_TRAIN_V2_ENABLED", "1")
    monkeypatch.setattr(ray, "init", lambda **kwargs: calls.append(("init", kwargs)))
    monkeypatch.setattr(ray, "shutdown", lambda: calls.append(("shutdown", None)))

    def trainer(train_loop_per_worker, **kwargs):
        import os
        assert os.environ["RAY_TRAIN_V2_ENABLED"] == "0"
        assert train_loop_per_worker is suites.worker
        payload = kwargs["train_loop_config"]
        assert payload["config"]["training"]["num_workers"] == 2
        assert len(payload["device_mappings"]) == 2

        def fit():
            if outcome == "interrupt":
                raise KeyboardInterrupt()
            if outcome == "failure":
                return SimpleNamespace(error="worker failed", metrics={})
            if outcome == "success":
                suites.write(output / "heldout" / "report.json",
                             {"status": "succeeded", "plan_sha256": "plan", "wandb": {"status": "finished"}})
            return SimpleNamespace(error=None, metrics={"completed_suites": 1} if outcome == "success" else {})
        return SimpleNamespace(fit=fit)

    monkeypatch.setattr(ray.train.torch, "TorchTrainer", trainer)
    args = ["--config", "unused", "--specs", str(specs_path), "--prepared", str(prepared_path),
            "--source-run", str(root), "--step", "2", "--output", str(output)]
    if outcome == "success":
        suites.main(args)
    else:
        with pytest.raises(KeyboardInterrupt if outcome == "interrupt" else RuntimeError):
            suites.main(args)
    assert suites.read(output / "driver.json")["status"] == expected_status
    assert calls[0][0] == "init"
    assert calls[0][1]["_node_ip_address"] == "127.0.0.1"
    assert calls[0][1]["dashboard_host"] == "127.0.0.1"
    assert calls[-1][0] == "shutdown"
    import os
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "2,3"
    assert os.environ["RAY_TRAIN_V2_ENABLED"] == "1"
