# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Logging boundaries: rank ownership, failure visibility, and artifact payloads."""
import copy
import hashlib
import json
import sys
from types import SimpleNamespace

import pytest

from grana.training.logging import RunLogger
from grana.training.evaluation_reporting import compact_evaluation_metrics


class FakeArtifact:
    def __init__(self, **kwargs):
        self.kwargs, self.files, self.waited = kwargs, [], False

    def add_file(self, path, name):
        self.files.append((path, name))

    def wait(self, timeout):
        self.waited = True


class FakeRun:
    def __init__(self):
        self.summary, self.rows, self.artifacts, self.finishes = {}, [], [], []
        self.log_calls, self.history_steps, self.dropped_rows = [], [], []
        self._step, self._partial = 0, {}
        self.id, self.entity, self.project, self.url, self.dir = "test", "team", "grana", "https://example.test/run", "/tmp/fake"

    def define_metric(self, *args, **kwargs):
        pass

    def log(self, row, *, step=None, commit=None):
        self.log_calls.append({"step": step, "commit": commit})
        if step is not None and step < self._step:
            self.dropped_rows.append(copy.deepcopy(row))
            return
        if step is not None and step > self._step:
            if self._partial:
                self.rows.append(self._partial)
                self.history_steps.append(self._step)
                self._partial = {}
            self._step = step
        self._partial.update(copy.deepcopy(row))
        if commit is True or (step is None and commit is None):
            self.rows.append(self._partial)
            self.history_steps.append(self._step)
            self._partial = {}
            self._step += 1

    def log_artifact(self, artifact, aliases):
        self.artifacts.append((artifact, aliases))
        return artifact

    def finish(self, exit_code):
        self.finishes.append(exit_code)


@pytest.fixture
def logging_setup(tmp_path, monkeypatch):
    cfg = {"training": {"output_dir": str(tmp_path), "stage": "pretrain"},
           "roles": {"A": 2, "B": 2, "C": 1, "D": 1},
           "loss": {"role_weights": {"A": 1, "B": 1, "C": 0.25, "D": 1}},
           "logging": {"mode": "online", "project": "grana", "entity": "team", "group": "pair",
                       "name": "pretrain", "job_type": None, "tags": ["test"], "upload_checkpoint_weights": False}}
    run, calls = FakeRun(), []

    def initialize(**kwargs):
        calls.append(kwargs)
        return run

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=initialize, Artifact=FakeArtifact, Settings=lambda **kw: kw))
    monkeypatch.setattr("grana.training.logging.environment_summary", lambda: {"python": "test"})
    return cfg, run, calls


@pytest.mark.parametrize("rank,mode", [(1, "online"), (0, "disabled")])
def test_non_primary_and_disabled_never_initialize(logging_setup, rank, mode):
    cfg, run, calls = logging_setup
    cfg["logging"]["mode"] = mode
    with RunLogger(cfg, rank=rank) as logger:
        logger.log_step({"loss": 1}, step=1)
        logger.log_evaluation({"loss": 1}, step=1)
        logger.log_checkpoint("/does/not/exist", step=1)
    assert calls == []
    assert run.rows == []
    assert run.finishes == []


@pytest.mark.parametrize("mode", ["online", "offline", "disabled"])
def test_worker_environment_is_captured_once_and_persisted_in_all_modes(logging_setup, monkeypatch, tmp_path, mode):
    cfg, run, calls = logging_setup
    cfg["logging"]["mode"] = mode
    environment = {"python": "test", "runtime_fingerprint": {"sha256": "worker-runtime"}}
    observed = []

    def capture():
        observed.append(True)
        return environment

    monkeypatch.setattr("grana.training.logging.environment_summary", capture)
    with RunLogger(cfg):
        pass
    assert observed == [True]
    payload = (tmp_path / "environment.json").read_bytes()
    assert json.loads(payload) == environment
    receipt = json.loads((tmp_path / "wandb_run.json").read_text())
    assert receipt["environment_path"] == str(tmp_path / "environment.json")
    assert receipt["environment_sha256"] == hashlib.sha256(payload).hexdigest()
    assert receipt["runtime_fingerprint_sha256"] == "worker-runtime"
    if mode == "disabled":
        assert not calls and not run.summary
    else:
        assert run.summary["environment"] is environment


def test_non_primary_rank_never_captures_or_writes_environment(logging_setup, monkeypatch, tmp_path):
    cfg, _, _ = logging_setup
    monkeypatch.setattr("grana.training.logging.environment_summary", lambda: pytest.fail("Rank 1 captured environment"))
    with RunLogger(cfg, rank=1):
        pass
    assert not (tmp_path / "environment.json").exists()


@pytest.mark.parametrize("mode", ["online", "disabled"])
def test_worker_environment_capture_failure_propagates_before_sdk_initialization(logging_setup, monkeypatch, tmp_path, mode):
    cfg, _, calls = logging_setup
    cfg["logging"]["mode"] = mode

    def fail():
        raise OSError("fixture environment capture failure")

    monkeypatch.setattr("grana.training.logging.environment_summary", fail)
    with pytest.raises(OSError, match="environment capture failure"):
        with RunLogger(cfg):
            pass
    assert not calls
    receipt = json.loads((tmp_path / "wandb_run.json").read_text())
    assert receipt["status"] == "failed" and receipt["failure_type"] == "OSError"


def test_history_and_eval_share_optimizer_axis_without_step_loss(logging_setup):
    cfg, run, calls = logging_setup
    with RunLogger(cfg, world_size=2) as logger:
        logger.start(model_summary={"size": "0.8B"}, data_summary={"tasks": ["a", "b"]})
        logger.log_step({"loss": 0.5, "role_C_loss": 2, "learning_rate": 1e-5}, step=1)
        logger.log_evaluation({"role_C_poi_accuracy": 0.75}, step=1)
        assert run.rows == []
        logger.commit_step(step=1)
        logger.log_step({"loss": 0.4, "role_C_loss": 1.6}, step=2)
        logger.finish({"checkpoint_prediction_parity": True})
    assert calls[0]["config"] == cfg
    assert calls[0]["settings"]["disable_git"] is True
    assert calls[0]["settings"]["symlink"] is False
    assert run.rows[0]["role_C_weighted_loss"] == 0.5
    assert run.rows[0]["eval/role_C_poi_accuracy"] == .75
    assert run.rows[1]["optimizer_steps"] == 2
    assert run.history_steps == [1, 2] and run.dropped_rows == []
    assert run.log_calls == [{"step": 1, "commit": True}, {"step": 2, "commit": True}]
    assert run.summary["logged_optimizer_steps"] == 2
    assert run.finishes == [0]


def test_eval_logs_confusion_and_zero_support_stages_on_optimizer_axis(logging_setup, monkeypatch):
    from grana.training.evaluation import record_diagnostics

    cfg, run, _ = logging_setup
    monkeypatch.setattr(sys.modules["wandb"], "Table", lambda **kwargs: kwargs, raising=False)
    diagnostics, metrics = record_diagnostics([
        {"cache": "/heldout/cache", "task": "unchanged", "trajectory_id": "heldout", "poi": 0,
         "role": "C", "group": "L1L2", "num_stages": 3, "predictions": [1] * 8, "targets": [0] * 8},
    ])
    with RunLogger(cfg) as logger:
        logger.log_evaluation(metrics, step=250, diagnostics=diagnostics)
    row = run.rows[0]
    assert row["optimizer_steps"] == 250
    assert [250, "per_group", "L1L2", 0, 1, 1] in row["eval/stage_confusion"]["data"]
    assert [250, "per_group", "L1L2", 2, 0, None] in row["eval/stage_support"]["data"]


def test_accumulated_primary_preserves_full_detail_in_artifact_and_survives_render_subset(logging_setup):
    cfg, run, _ = logging_setup
    cfg["evaluation"] = {"mode": "dense"}
    metrics = {"role_B_poi_rmse": .22, "role_D_poi_rmse": .23,
               "accumulated/overall/raw_rmse": .4, "accumulated/overall/normalized_rmse": .04}
    with RunLogger(cfg) as logger:
        receipt = logger.log_evaluation(metrics, step=250)
        primary = copy.deepcopy(run.summary["primary_evaluation"])
        report = copy.deepcopy(run.summary["evaluation_report"])
        logger.log_evaluation({"role_D_poi_rmse": .1}, step=250, purpose="renders")
    assert primary["raw_rmse"] == .4 and primary["normalized_rmse"] == .04
    assert primary["route"] == "predicted_stage_C_to_D" and primary["optimizer_steps"] == 250
    assert run.summary["primary_evaluation"] == primary and run.summary["evaluation_report"] == report
    assert run.rows == [{"optimizer_steps": 250,
                         **{"eval/" + key: value for key, value in compact_evaluation_metrics(metrics).items()}}]
    assert run.history_steps == [250]
    from pathlib import Path
    payload = Path(receipt["local_path"]).read_bytes()
    assert hashlib.sha256(payload).hexdigest() == receipt["sha256"]
    assert json.loads(payload)["metrics"] == metrics
    assert receipt["logged_scalar_keys"] == sorted(key for key in run.rows[0] if key != "optimizer_steps")
    assert receipt["file_name"] == "evaluation.json"
    assert run.artifacts[0][0].waited


def test_evaluation_artifact_preserves_all_records_runtime_and_provenance(logging_setup, tmp_path):
    cfg, run, _ = logging_setup
    metrics = {"role_B_poi_rmse": .2, "role_D_poi_rmse": .3,
               "accumulated/overall/raw_rmse": .8, "accumulated/overall/normalized_rmse": .08,
               "diagnostics/overall/D/incorrect_C/rmse": .9}
    evaluation = {"schema": "grana-heldout-eval-v1", "optimizer_steps": 2, "plan_sha256": "exact-plan",
                  "metrics": metrics, "records": [{"cache": "cache", "task": "task", "trajectory_id": "id", "poi": 5,
                                                   "role": "D", "predictions": [.2] * 8}],
                  "timing_seconds": {"forward_cuda": 1.2}, "provenance": {"mode": "during_training"}}
    before = copy.deepcopy(evaluation)
    event = {**metrics, "elapsed_seconds": 3., "evaluation_and_persistence_seconds": 4.}
    with RunLogger(cfg) as logger:
        receipt = logger.log_evaluation(event, step=2, evaluation=evaluation)
        with pytest.raises(ValueError, match="increase strictly"):
            logger.log_evaluation(event, step=2, evaluation=evaluation)
        logger.finish({"optimizer_steps": 2, "checkpoint_prediction_parity": True,
                       "evaluation_events": [event], "runtime_summary": {"large": "kept only in artifact"},
                       "eval/diagnostics/overall/D/incorrect_C/rmse": .9,
                       "eval/accumulated/overall/raw_rmse": .8})
    envelope = json.loads((tmp_path / "evaluation_artifacts/primary-000002.json").read_text())
    assert envelope["evaluation"] == before and evaluation == before
    assert envelope["metrics"] == event and envelope["purpose"] == "primary"
    assert receipt["plan_sha256"] == "exact-plan"
    assert run.rows[0]["eval/evaluation_and_persistence_seconds"] == 4.
    assert "eval/diagnostics/overall/D/incorrect_C/rmse" not in run.rows[0]
    assert "evaluation_events" not in run.summary and "runtime_summary" not in run.summary
    assert "eval/diagnostics/overall/D/incorrect_C/rmse" not in run.summary
    completion = json.loads((tmp_path / "evaluation_artifacts/completion.json").read_text())
    assert completion["summary"]["evaluation_events"] == [event]
    assert completion["summary"]["runtime_summary"] == {"large": "kept only in artifact"}
    assert run.summary["evaluation_logging_schema"] == "grana-compact-eval-v3"
    assert run.summary["history_step_schema"] == "grana-optimizer-step-history-v1"


def test_distinct_dense_population_has_its_own_namespace_and_artifact(logging_setup):
    cfg, run, _ = logging_setup
    cfg["evaluation"] = {"mode": "sampled"}
    with RunLogger(cfg) as logger:
        logger.log_evaluation({"role_D_poi_rmse": .2}, step=2)
        logger.log_evaluation({"accumulated/overall/raw_rmse": .7,
                               "accumulated/overall/normalized_rmse": .07}, step=2, purpose="dense")
    assert run.summary["primary_evaluation"]["status"] == "unavailable"
    compact = compact_evaluation_metrics({"accumulated/overall/raw_rmse": .7,
                                          "accumulated/overall/normalized_rmse": .07})
    assert run.rows == [{"optimizer_steps": 2, **{"eval/dense/" + key: value for key, value in compact.items()}}]
    assert [row["purpose"] for row in run.summary["evaluation_artifacts"]] == ["primary", "dense"]


@pytest.mark.parametrize("damage", ["metrics", "step", "purpose"])
def test_evaluation_logging_rejects_mismatched_artifact_identity_before_publication(logging_setup, damage):
    cfg, run, _ = logging_setup
    evaluation = {"optimizer_steps": 2, "metrics": {"role_C_poi_accuracy": .5}}
    metrics, step, purpose = dict(evaluation["metrics"]), 2, "primary"
    if damage == "metrics":
        metrics["role_C_poi_accuracy"] = .9
    elif damage == "step":
        step = 3
    else:
        purpose = "../escape"
    with RunLogger(cfg) as logger:
        with pytest.raises(ValueError):
            logger.log_evaluation(metrics, step=step, evaluation=evaluation, purpose=purpose)
    assert run.artifacts == [] and run.rows == []


@pytest.mark.parametrize("include_weights", [False, True])
def test_checkpoint_manifest_explicitly_labels_payload(logging_setup, tmp_path, include_weights):
    cfg, run, _ = logging_setup
    cfg["logging"]["upload_checkpoint_weights"] = include_weights
    checkpoint = tmp_path / "checkpoint-000001"
    checkpoint.mkdir()
    (checkpoint / "metadata.json").write_text(json.dumps({"weights_sha256": "digest"}))
    (checkpoint / "weights.safetensors").write_bytes(b"test-weights")
    with RunLogger(cfg) as logger:
        receipt = logger.log_checkpoint(checkpoint, step=1, final=True)
    artifact, aliases = run.artifacts[0]
    assert receipt["payload_included"] is include_weights
    assert artifact.kwargs["type"] == ("model" if include_weights else "checkpoint-manifest")
    assert ("weights.safetensors" in [name for _, name in artifact.files]) is include_weights
    assert artifact.waited
    assert "final" in aliases


def test_training_failure_propagates_and_finishes_nonzero(logging_setup, tmp_path):
    cfg, run, _ = logging_setup
    with pytest.raises(RuntimeError, match="training failed"):
        with RunLogger(cfg):
            raise RuntimeError("training failed")
    assert run.finishes == [1]
    assert run.summary["status"] == "failed"
    receipt = json.loads((tmp_path / "wandb_run.json").read_text())
    assert receipt["exit_code"] == 1
    assert receipt["failure_type"] == "RuntimeError"


def test_online_failure_never_falls_back(logging_setup, monkeypatch, tmp_path):
    cfg, _, calls = logging_setup

    def fail_init(**kwargs):
        calls.append(kwargs)
        raise ConnectionError("network failure")

    monkeypatch.setattr(sys.modules["wandb"], "init", fail_init)
    with pytest.raises(ConnectionError):
        with RunLogger(cfg):
            pass
    assert len(calls) == 1
    assert calls[0]["mode"] == "online"
    assert json.loads((tmp_path / "wandb_run.json").read_text())["status"] == "failed"


def test_finish_failure_is_visible(logging_setup, monkeypatch, tmp_path):
    cfg, run, _ = logging_setup

    def fail_finish(exit_code):
        raise TimeoutError("upload timeout")

    monkeypatch.setattr(run, "finish", fail_finish)
    with pytest.raises(TimeoutError):
        with RunLogger(cfg):
            pass
    assert json.loads((tmp_path / "wandb_run.json").read_text())["status"] == "failed"


def test_artifact_failure_finishes_failed_and_propagates(logging_setup, monkeypatch, tmp_path):
    cfg, run, _ = logging_setup
    checkpoint = tmp_path / "checkpoint-000001"
    checkpoint.mkdir()
    (checkpoint / "metadata.json").write_text(json.dumps({"weights_sha256": "digest"}))
    (checkpoint / "weights.safetensors").write_bytes(b"test-weights")

    def fail_wait(self, timeout):
        raise ConnectionError("artifact upload failed")

    monkeypatch.setattr(FakeArtifact, "wait", fail_wait)
    with pytest.raises(ConnectionError, match="artifact upload failed"):
        with RunLogger(cfg) as logger:
            logger.log_checkpoint(checkpoint, step=1)
    assert run.finishes == [1]
    assert run.summary["status"] == "failed"


def test_real_step_renders_are_published_with_time_metrics_and_manifest(logging_setup, monkeypatch, tmp_path):
    cfg, run, _ = logging_setup
    sdk = sys.modules["wandb"]
    monkeypatch.setattr(sdk, "Image", lambda path, **kwargs: {"image": path, **kwargs}, raising=False)
    monkeypatch.setattr(sdk, "Video", lambda path, **kwargs: {"video": path, **kwargs}, raising=False)
    monkeypatch.setattr(sdk, "Table", lambda **kwargs: kwargs, raising=False)
    bundle = {"schema": "grana-training-eval-renders-v1", "optimizer_steps": 25,
              "training_elapsed_seconds": 120.5, "rendering_seconds": 1.2, "plan_sha256": "held-out-plan",
              "replay_type": "sampled_heldout_windows_not_original_speed", "available_trajectories": 8,
              "rendered_trajectories": 1, "manifest": str(tmp_path / "manifest.json"),
              "items": [{"key": "stable-trajectory", "task": "pick", "trajectory_id": "held-out",
                         "image": "sample.png", "video": "sample.gif", "video_format": "gif",
                         "metrics": {"role_A_poi_rmse": .2, "role_C_poi_accuracy": .75},
                         "sampled_pois": [2, 5, 8], "replay_pois": [2, 8]}]}
    with RunLogger(cfg) as logger:
        logger.log_step({"loss": .5}, step=25)
        logger.log_evaluation_renders(bundle, step=25)
        with pytest.raises(ValueError, match="increase strictly"):
            logger.log_evaluation_renders(bundle, step=25)
        with pytest.raises(ValueError, match="match this optimizer"):
            logger.log_evaluation_renders(bundle, step=26)
        logger.commit_step(step=25)
        logger.log_step({"loss": .4}, step=26)
    row = run.rows[0]
    assert run.history_steps == [25, 26] and run.dropped_rows == []
    assert row["optimizer_steps"] == 25 and row["training_elapsed_seconds"] == 120.5
    assert row["eval/rendering_seconds"] == 1.2 and row["eval/rendered_trajectories"] == 1
    assert row["eval/renders/stable-trajectory"]["image"] == "sample.png"
    assert "not original-speed" in row["eval/replays/stable-trajectory"]["caption"]
    assert row["eval/replays/stable-trajectory"]["format"] == "gif"
    table = row["eval/render_examples"]
    table_row = dict(zip(table["columns"], table["data"][0], strict=True))
    assert table_row["A_whole_task_RMSE_diagnostic"] == .2 and table_row["C_stage_accuracy_diagnostic"] == .75
    assert table_row["D_predicted_focus_relative_RMSE_diagnostic"] is None
    assert table_row["evaluated_POIs"] == 3 and table_row["replay_POIs"] == 2
    artifact, aliases = run.artifacts[0]
    assert artifact.kwargs["type"] == "evaluation-renders" and artifact.waited
    assert artifact.files == [(bundle["manifest"], "manifest.json")]
    assert aliases == ["latest", "step-000025"]
    assert run.summary["evaluation_renders"][0]["plan_sha256"] == "held-out-plan"
    receipt = json.loads((tmp_path / "wandb_run.json").read_text())
    assert receipt["evaluation_renders"][0]["optimizer_steps"] == 25


def test_render_media_and_companion_metrics_use_disjoint_typed_collections_at_steps_2_4(logging_setup, monkeypatch, tmp_path):
    cfg, run, _ = logging_setup
    sdk = sys.modules["wandb"]
    monkeypatch.setattr(sdk, "Image", lambda path, **kwargs: {"image": path}, raising=False)
    monkeypatch.setattr(sdk, "Video", lambda path, **kwargs: {"video": path}, raising=False)
    monkeypatch.setattr(sdk, "Table", lambda **kwargs: kwargs, raising=False)
    collections, original = {}, run.log_artifact

    def enforce_collection_type(artifact, aliases):
        name, kind = artifact.kwargs["name"], artifact.kwargs["type"]
        if name in collections and collections[name] != kind:
            raise ValueError(f"Artifact {name} already exists with type {collections[name]}; cannot create {kind}")
        collections[name] = kind
        return original(artifact, aliases)

    monkeypatch.setattr(run, "log_artifact", enforce_collection_type)
    with RunLogger(cfg) as logger:
        for step in (2, 4):
            bundle = {"schema": "grana-training-eval-renders-v1", "optimizer_steps": step,
                      "training_elapsed_seconds": float(step), "rendering_seconds": .1, "plan_sha256": "plan",
                      "replay_type": "sampled_heldout_windows_not_original_speed", "available_trajectories": 1,
                      "rendered_trajectories": 1, "manifest": str(tmp_path / f"manifest-{step}.json"),
                      "items": [{"key": "trajectory", "task": "pick", "trajectory_id": "held-out",
                                 "image": "sample.png", "video": "sample.gif", "video_format": "gif",
                                 "metrics": {"role_C_poi_accuracy": .75}, "sampled_pois": [0, 1], "replay_pois": [0, 1]}]}
            logger.log_step({"loss": .5}, step=step)
            logger.log_evaluation_renders(bundle, step=step)
            logger.log_evaluation({"role_C_poi_accuracy": .75}, step=step, purpose="renders")
            logger.log_evaluation({"role_C_poi_accuracy": .8}, step=step)
            logger.commit_step(step=step)
    assert collections == {"grana-test-evaluation-render-media": "evaluation-renders",
                           "grana-test-evaluation-renders": "evaluation-details",
                           "grana-test-evaluation-primary": "evaluation-details"}
    assert run.history_steps == [2, 4] and not run.dropped_rows
    assert all(row["eval/render_subset/role_C_poi_accuracy"] == .75
               and row["eval/renders/trajectory"]["image"] == "sample.png" for row in run.rows)
    assert [artifact.kwargs["name"] for artifact, _ in run.artifacts].count("grana-test-evaluation-render-media") == 2
    assert all(artifact.waited for artifact, _ in run.artifacts)


@pytest.mark.parametrize("rank,mode", [(1, "online"), (0, "disabled")])
def test_disabled_render_logging_does_not_load_media_or_validate_bundle(logging_setup, rank, mode):
    cfg, run, calls = logging_setup
    cfg["logging"]["mode"] = mode
    with RunLogger(cfg, rank=rank) as logger:
        logger.log_evaluation_renders({}, step=1)
    assert calls == [] and run.rows == []


def replay_bundle():
    return {"optimizer_steps": 250, "plan_sha256": "execution-plan", "manifest": "/new/replay/manifest.json",
            "items": [{"image": "plot.png", "video": "replay.mp4"}],
            "provenance": {"mode": "checkpoint_replay", "source_optimizer_step": 250,
                           "new_optimizer_updates": 0, "source_run": "team/grana/old-run",
                           "checkpoint": "/old/checkpoint-000250", "weights_sha256": "digest"}}


def test_checkpoint_replay_uses_new_run_waits_for_artifacts_and_retains_training_guard(logging_setup, monkeypatch, tmp_path):
    cfg, run, calls = logging_setup
    cfg["logging"]["job_type"] = "checkpoint-evaluation"
    artifact, published = FakeArtifact(), []
    monkeypatch.setattr("grana.training.deploy_dense.log_dense_bundle",
                        lambda remote, sdk, bundle, **kwargs: published.append((remote, bundle)) or artifact)
    with RunLogger(cfg) as logger:
        logger.log_checkpoint_replay_dense(replay_bundle(), step=250)
        with pytest.raises(ValueError, match="increase strictly"):
            logger.log_checkpoint_replay_dense(replay_bundle(), step=250)
        with pytest.raises(ValueError, match="Training dense evaluation"):
            logger.log_deploy_dense(replay_bundle(), step=250)
    assert calls[0]["resume"] == "never" and calls[0]["reinit"] == "create_new"
    assert "id" not in calls[0]
    assert artifact.waited and published[0][0] is run
    assert run.summary["logged_optimizer_steps"] == 0
    assert "deploy_dense_evaluations" not in run.summary
    receipt = json.loads((tmp_path / "wandb_run.json").read_text())
    assert receipt["checkpoint_replay_evaluations"][0]["provenance"]["source_run"] == "team/grana/old-run"


@pytest.mark.parametrize("damage", ["job", "mode", "step", "updates", "identity", "missing_provenance"])
def test_checkpoint_replay_rejects_training_runs_and_false_provenance(logging_setup, monkeypatch, damage):
    cfg, _, _ = logging_setup
    cfg["logging"]["job_type"] = "checkpoint-evaluation"
    bundle = replay_bundle()
    if damage == "job":
        cfg["logging"]["job_type"] = "posttrain"
    elif damage == "mode":
        bundle["provenance"]["mode"] = "during_training"
    elif damage == "step":
        bundle["provenance"]["source_optimizer_step"] = 249
    elif damage == "updates":
        bundle["provenance"]["new_optimizer_updates"] = 1
    elif damage == "identity":
        bundle["provenance"]["source_run"] = None
    else:
        del bundle["provenance"]
    monkeypatch.setattr("grana.training.deploy_dense.log_dense_bundle", lambda *a: pytest.fail("invalid replay was uploaded"))
    with RunLogger(cfg) as logger:
        with pytest.raises(ValueError, match="dedicated evaluation run"):
            logger.log_checkpoint_replay_dense(bundle, step=250)


def test_explicit_commit_rejects_empty_duplicate_past_and_uncommitted_advance(logging_setup):
    cfg, run, _ = logging_setup
    with RunLogger(cfg) as logger:
        with pytest.raises(ValueError, match="without a history payload"):
            logger.commit_step(step=1)
        logger.log_step({"loss": .5}, step=1)
        with pytest.raises(ValueError, match="Commit pending"):
            logger.log_step({"loss": .4}, step=2)
        with pytest.raises(ValueError, match="Commit pending"):
            logger.commit_step(step=2)
        with pytest.raises(ValueError, match="increase strictly"):
            logger.log_step({"loss": .5}, step=1)
        logger.commit_step(step=1)
        for step in (0, 1):
            with pytest.raises(ValueError, match="already committed"):
                logger.log_evaluation({"role_C_poi_accuracy": .75}, step=step)
            with pytest.raises(ValueError, match="already committed"):
                logger.commit_step(step=step)
        logger.log_step({"loss": .4}, step=2)
    assert run.history_steps == [1, 2] and not run.dropped_rows
    assert run.artifacts == []  # Rejected late eval never publishes an artifact.


@pytest.mark.parametrize("step", [True, -1, 1.5, "1"])
def test_history_commit_rejects_invalid_step_without_writing(logging_setup, step):
    cfg, run, _ = logging_setup
    with RunLogger(cfg) as logger:
        with pytest.raises(ValueError, match="nonnegative integer"):
            logger.commit_step(step=step)
    assert not run.rows and not run.log_calls


def test_duplicate_payload_keys_are_rejected_before_artifact_or_summary_mutation(logging_setup):
    cfg, run, _ = logging_setup
    with RunLogger(cfg) as logger:
        logger.log_step({"loss": .5, "eval/role_C_poi_accuracy": .5}, step=1)
        with pytest.raises(ValueError, match="Duplicate history keys"):
            logger.log_evaluation({"role_C_poi_accuracy": .75}, step=1)
        assert run.artifacts == [] and "primary_evaluation" not in run.summary
    assert run.rows[0]["eval/role_C_poi_accuracy"] == .5


def test_failure_flushes_only_completed_measured_payload_once(logging_setup, tmp_path):
    cfg, run, _ = logging_setup
    with pytest.raises(RuntimeError, match="evaluation failed"):
        with RunLogger(cfg) as logger:
            logger.log_step({"loss": .5, "time/training_step": 3.}, step=2)
            raise RuntimeError("evaluation failed")
    assert run.rows == [{"optimizer_steps": 2, "loss": .5, "time/training_step": 3.}]
    assert run.history_steps == [2] and run.finishes == [1]
    assert run.summary["status"] == "failed"
    receipt = json.loads((tmp_path / "wandb_run.json").read_text())
    assert receipt["committed_optimizer_steps"] == [2] and receipt["pending_optimizer_step"] is None


@pytest.mark.parametrize("catch_write_failure", [False, True])
def test_ambiguous_history_write_is_never_retried_or_reported_as_success(
        logging_setup, monkeypatch, tmp_path, catch_write_failure):
    cfg, run, _ = logging_setup
    original_log = run.log

    def accepted_then_failed(row, **kwargs):
        original_log(row, **kwargs)
        raise ConnectionError("history acknowledgment failed")

    monkeypatch.setattr(run, "log", accepted_then_failed)
    expected = RuntimeError if catch_write_failure else ConnectionError
    with pytest.raises(expected):
        with RunLogger(cfg) as logger:
            logger.log_step({"loss": .5}, step=1)
            if catch_write_failure:
                with pytest.raises(ConnectionError):
                    logger.commit_step(step=1)
                logger.finish()
            else:
                logger.commit_step(step=1)
    assert len(run.log_calls) == 1 and run.history_steps == [1]
    assert run.finishes == [1] and run.summary["status"] == "failed"
    receipt = json.loads((tmp_path / "wandb_run.json").read_text())
    assert receipt["history_write_failed"] is True and receipt["committed_optimizer_steps"] == []


def test_dense_media_and_eval_share_one_real_history_step(logging_setup, monkeypatch, tmp_path):
    cfg, run, _ = logging_setup
    sdk = sys.modules["wandb"]
    monkeypatch.setattr(sdk, "Image", lambda path, **kwargs: {"image": path, **kwargs}, raising=False)
    monkeypatch.setattr(sdk, "Video", lambda path, **kwargs: {"video": path, **kwargs}, raising=False)
    monkeypatch.setattr(sdk, "Table", lambda **kwargs: kwargs, raising=False)
    monkeypatch.setattr(sdk, "Artifact", lambda name, **kwargs: FakeArtifact(name=name, **kwargs))
    bundle = {"schema": "grana-deploy-dense-v1", "optimizer_steps": 2,
              "training_elapsed_seconds": 1., "generated_at": "2026-11-04T00:00:00Z",
              "plan_sha256": "plan", "evaluated_pois": 2, "frame_stride": 1,
              "selection_shortfall_count": 0, "manifest": str(tmp_path / "manifest.json"),
              "provenance": {"mode": "during_training"}, "metrics": {},
              "items": [{"key": "task/best", "task": "task", "selection": "best",
                         "candidate_trajectories": 1, "trajectory_id": "traj", "units": "raw",
                         "route": "predicted_stage_route", "image": "sample.png", "video": "sample.mp4",
                         "video_pois": [0, 1], "metrics": {"raw_rmse": .2, "normalized_rmse": .1,
                                                                         "evaluated_pois": 2}}]}
    with RunLogger(cfg) as logger:
        logger.log_step({"loss": .5}, step=1)
        logger.commit_step(step=1)
        logger.log_step({"loss": .4}, step=2)
        logger.log_deploy_dense(bundle, step=2)
        logger.log_evaluation({"role_C_poi_accuracy": .75}, step=2)
        assert len(run.rows) == 1
        logger.commit_step(step=2)
        logger.finish()
    assert run.history_steps == [1, 2] and not run.dropped_rows
    row = run.rows[1]
    assert row["optimizer_steps"] == 2 and row["loss"] == .4
    assert row["eval/role_C_poi_accuracy"] == .75
    assert row["deploy_dense_samples/task/best"]["image"] == "sample.png"
    assert row["deploy_dense_videos/task/best"]["video"] == "sample.mp4"
    assert run.summary["committed_optimizer_steps"] == [1, 2]


def test_zero_step_evaluation_and_sparse_steps_remain_explicit(logging_setup):
    cfg, run, _ = logging_setup
    with RunLogger(cfg) as logger:
        logger.log_evaluation({"role_C_poi_accuracy": .5}, step=0)
        logger.commit_step(step=0)
        logger.log_evaluation({"role_C_poi_accuracy": .8}, step=250)
    assert run.history_steps == [0, 250]
    assert run.summary["logged_optimizer_steps"] == 0
    assert run.summary["committed_optimizer_steps"] == [0, 250]
