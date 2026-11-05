# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deployment errors must include wrong-stage routing and remain auditable."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from grana.data.cache import Trajectory
from grana.training.deploy_dense import build_dense_plan, dense_metrics, dense_series, log_dense_bundle, render_dense_evaluation
from grana.training.evaluation import build_evaluation_plan, evaluate
from grana.training.campaign_report import verify_dense_bundle, verify_evaluation


class Prediction(torch.nn.Module):
    def forward(self, inputs, roles, counts):
        if roles == ["C"]:
            logits = torch.zeros(1, 8, counts[0])
            logits[:, :, -1] = 5
            return {"stage_logits": logits}
        return {"progress_logits": torch.zeros(1, 8, 10)}


@pytest.fixture
def cohort(tmp_path):
    def make(roles=("A", "B", "C"), count=2):
        rows = []
        for index in range(count):
            path = tmp_path / f"rgb-{index}.npy"
            pixels = np.full((8, 20, 24, 3), (30 + index * 100) % 256, dtype=np.uint8)
            np.save(path, pixels)
            rows.append(Trajectory(tmp_path, f"heldout-{index}", "assemble", 8, (path,), pixels.shape,
                                   ("top",), roles, tuple(np.linspace(0, 1, 8)),
                                   (0, 0, 0, 0, 1, 1, 1, 1) if "C" in roles else (),
                                   (.1, .3, .6, .9, .1, .3, .6, .9) if "C" in roles else (),
                                   ("Reach", "Place") if "C" in roles else (), "Assemble."))
        inventory = SimpleNamespace(records=tuple(rows), by_role={role: {} for role in "ABC"})
        base = build_evaluation_plan(inventory, {"max_trajectories_per_task": None, "windows_per_trajectory": 2,
                                                "stride_pairs": [[1, 1]]},
                                     {"views": ["top"], "max_pixels": None, "stride_pairs": [[1, 1]]},
                                     training_dataset=SimpleNamespace(records=()))
        return base
    return make


def assessed(base, **kwargs):
    plan = build_dense_plan(base, frame_stride=3, **kwargs)
    result = evaluate(Prediction(), None, plan, device="cpu", max_pixels=None, collate_fn=lambda *args: {})
    return result, plan


def test_dense_plan_uses_only_deployment_route_and_measured_frames(cohort):
    base = cohort()
    result, plan = assessed(base)
    assert {w.roles for w in plan.windows} == {("B", "C", "D")}
    assert {w.poi for w in plan.windows} == {0, 3, 6, 7}
    assert plan.manifest()["model_calls"] == result["model_calls"] == 24
    assert plan.settings["source_plan_sha256"] == base.manifest()["plan_sha256"]
    _, without_oracle = assessed(base, include_oracle=False)
    assert {w.roles for w in without_oracle.windows} == {("C", "D")}
    assert {w.trajectory.trajectory_id for w in plan.windows} <= {w.trajectory.trajectory_id for w in base.windows}


def test_wrong_stage_error_is_included_in_accumulated_progress(cohort):
    from grana.training.evaluation_reporting import evaluation_report

    result, plan = assessed(cohort())
    rows = dense_series(result, plan)
    row = rows[0]
    assert [p["prediction"] for p in row["points"]] == pytest.approx([1.5] * 4)
    assert [p["target"] for p in row["points"]] == pytest.approx([.1, .9, 1.6, 1.9])
    expected = ((1.4 ** 2 + .6 ** 2 + .1 ** 2 + .4 ** 2) / 4) ** .5
    assert row["metrics"]["raw_rmse"] == pytest.approx(expected)
    assert row["metrics"]["normalized_rmse"] == pytest.approx(expected / 2)
    assert row["metrics"]["stage_accuracy"] == .5
    # A correct focus-relative target of zero cannot hide choosing the wrong stage.
    assert row["points"][0]["focus_target"] == 0
    assert row["points"][0]["prediction"] - row["points"][0]["target"] == pytest.approx(1.4)
    assert row["points"][0]["oracle"] == pytest.approx(.5)
    metrics = dense_metrics(rows)
    assert metrics["overlay_acc/assemble_predicted_stage_route_accumulated_progress_raw_rmse"] == pytest.approx(expected)
    assert metrics["eval_quality/assemble/predicted_stage_route_stage_accuracy"] == .5
    report = evaluation_report(result["metrics"], scope="dense")
    assert report["primary"]["raw_rmse"] == pytest.approx(expected)
    assert report["primary"]["normalized_rmse"] == pytest.approx(expected / 2)
    assert report["primary"]["raw_rmse"] != pytest.approx(result["metrics"]["role_D_poi_rmse"])
    assert report["oracle_accumulated_comparison"]["per_task"]["assemble"]["raw_rmse"] == pytest.approx(
        result["metrics"]["role_B_poi_rmse"])


@pytest.mark.parametrize("point_count", [1, 2])
def test_sparse_smoke_samples_do_not_draw_unmeasured_trajectory(cohort, point_count):
    from grana.training.deploy_dense import _background
    result, plan = assessed(cohort())
    row = dense_series(result, plan)[0]
    row["points"] = [row["points"][0], row["points"][-1]][:point_count]
    pixels = np.asarray(_background(row))
    # Endpoint-only checks cannot imply a curve or a stage boundary halfway
    # through a trajectory whose interior was never evaluated.
    blue_prediction = np.array([0x16, 0x70, 0xDC])
    assert not np.any(np.all(pixels[376:570, 200:1100] == blue_prediction, axis=-1))
    assert np.all(pixels[250, 200:1100] == [0xE1, 0xE6, 0xEE])
    assert np.all(pixels[280, 200:1100] == [0xE1, 0xE6, 0xEE])
    assert np.any(np.all(pixels[376:570, 61:70] == blue_prediction, axis=-1))


def test_a_only_inventory_stays_whole_task_and_has_no_fabricated_stages(cohort):
    result, plan = assessed(cohort(roles=("A",)))
    rows = dense_series(result, plan)
    assert {w.roles for w in plan.windows} == {("A",)}
    assert rows[0]["scale"] == 1 and rows[0]["units"] == "whole_task_[0,1]"
    assert all("predicted_stage" not in p for p in rows[0]["points"])
    assert "overlay_acc/assemble_compressed_whole_task_progress_raw_rmse" in dense_metrics(rows)


def test_dense_rejects_predictions_from_another_plan(cohort):
    result, plan = assessed(cohort())
    result["plan_sha256"] = "foreign"
    with pytest.raises(ValueError, match="exact held-out"):
        dense_series(result, plan)


@pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None, reason="MP4 integration needs ffmpeg/ffprobe")
def test_mp4_best_worst_selection_and_replay_provenance(cohort, tmp_path):
    result, plan = assessed(cohort())
    bundle = render_dense_evaluation(result, plan, output_dir=tmp_path / "dense", step=50,
                                     training_elapsed_seconds=123.5, max_frames=3,
                                     provenance={"mode": "checkpoint_replay", "checkpoint": "/saved/checkpoint-000050",
                                                 "weights_sha256": "digest", "source_run": "team/project/run"})
    result.update(optimizer_steps=50, training_elapsed_seconds=123.5)
    assert len(verify_dense_bundle(bundle["manifest"], result)) == 4
    assert [i["selection"] for i in bundle["items"]] == ["best_raw_rmse", "worst_raw_rmse"]
    assert len({i["trajectory_id"] for i in bundle["items"]}) == 2
    assert bundle["evaluated_pois"] == 8 and bundle["selection_shortfall_count"] == 0
    assert bundle["provenance"]["mode"] == "checkpoint_replay"
    for item in bundle["items"]:
        assert item["video_pois"] == [0, 6, 7]
        assert item["image_poi"] == 0  # largest actual error, not an arbitrary mid-frame
        video = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                                    "stream=codec_name,nb_frames,pix_fmt", "-of", "json", item["video"]]))
        assert video["streams"][0] == {"codec_name": "h264", "pix_fmt": "yuv420p", "nb_frames": "3"}
        assert hashlib.sha256(Path(item["video"]).read_bytes()).hexdigest() == item["video_sha256"]
    corrupted = deepcopy(bundle)
    corrupted["series"][0]["points"][0]["prediction"] -= 1
    Path(bundle["manifest"]).write_text(json.dumps(corrupted))
    with pytest.raises(ValueError, match="RMSE differs"):
        verify_dense_bundle(bundle["manifest"], result)


def test_single_trajectory_does_not_duplicate_best_as_worst(cohort, tmp_path, monkeypatch):
    import grana.training.deploy_dense as dense
    monkeypatch.setattr(dense, "require_encoder", lambda: "unused")
    monkeypatch.setattr(dense, "_video", lambda path, frames, fps: path.write_bytes(b"test-video"))
    result, plan = assessed(cohort(count=1))
    bundle = render_dense_evaluation(result, plan, output_dir=tmp_path / "one", step=2, training_elapsed_seconds=1)
    assert len(bundle["items"]) == 1 and bundle["selection_shortfall_count"] == 1
    assert bundle["items"][0]["selection"] == "best_raw_rmse"


def test_replay_requires_checkpoint_provenance(cohort, tmp_path):
    result, plan = assessed(cohort())
    with pytest.raises(ValueError, match="requires checkpoint"):
        render_dense_evaluation(result, plan, output_dir=tmp_path / "invalid", step=50,
                                training_elapsed_seconds=3, provenance={"mode": "checkpoint_replay"})


@pytest.mark.parametrize("buffered", [False, True])
def test_wandb_names_and_replay_axis_preserve_training_step(cohort, tmp_path, monkeypatch, buffered):
    import grana.training.deploy_dense as dense
    monkeypatch.setattr(dense, "require_encoder", lambda: "unused")
    monkeypatch.setattr(dense, "_video", lambda path, frames, fps: path.write_bytes(b"test-video"))
    result, plan = assessed(cohort())
    bundle = render_dense_evaluation(result, plan, output_dir=tmp_path / "log", step=50, training_elapsed_seconds=3)
    metrics, rows, files = [], [], []
    artifact = SimpleNamespace(add_file=lambda path, name: files.append((path, name)))
    sdk = SimpleNamespace(Image=lambda path, **kw: ("image-file", path), Video=lambda path, **kw: ("video-file", path),
                          Table=lambda **kw: kw, Artifact=lambda *args, **kw: artifact)
    run = SimpleNamespace(id="run", summary={}, define_metric=lambda *args, **kw: metrics.append((args, kw)), log=rows.append,
                          log_artifact=lambda *args, **kw: artifact)
    queued = []
    sink = (lambda row, *, step: queued.append((row, step))) if buffered else None
    log_dense_bundle(run, sdk, bundle, step_metric="deploy_dense/optimizer_steps", history_sink=sink)
    if buffered:
        assert rows == [] and queued[0][1] == 50
        row = queued[0][0]
    else:
        assert queued == []
        row = rows[0]
    assert "optimizer_steps" not in row and row["deploy_dense/optimizer_steps"] == 50
    assert row["deploy_dense_videos/assemble/worst_raw_rmse"][0] == "video-file"
    assert row["deploy_dense_samples/assemble/best_raw_rmse"][0] == "image-file"
    assert not any(key.startswith(("overlay_acc/", "accumulated/", "eval_quality/")) for key in row)
    assert {key for key, value in row.items() if isinstance(value, (int, float))} == {
        "deploy_dense/optimizer_steps", "eval_media/selection_shortfall_count"}
    assert all(kwargs["step_metric"] == "deploy_dense/optimizer_steps" for _, kwargs in metrics[1:])
    assert files == [(bundle["manifest"], "manifest.json")]
    table = row["eval_media/selection_manifest"]
    item = dict(zip(table["columns"], table["data"][0], strict=True))
    assert item["primary_route_raw_rmse"] == bundle["items"][0]["metrics"]["raw_rmse"]
    assert item["primary_route_normalized_rmse"] == bundle["items"][0]["metrics"]["normalized_rmse"]
    assert item["oracle_B_accumulated_raw_rmse_diagnostic"] == bundle["items"][0]["metrics"]["oracle_raw_rmse"]
    assert run.summary["deploy_dense_evaluation_report"]["primary"]["route"] == "predicted_stage_C_to_D"


@pytest.mark.parametrize("stride", [0, -1, True, 1.5])
def test_invalid_dense_stride_fails_before_inference(cohort, stride):
    with pytest.raises(ValueError, match="frame_stride"):
        build_dense_plan(cohort(), frame_stride=stride)


def grouped_base(cohort):
    original = cohort(count=6)
    roots = [original.windows[0].trajectory.cache_root / group for group in ("cache-one", "cache-two")]
    rows = [replace(row, cache_root=roots[index // 3]) for index, row in enumerate(original.dataset.records)]
    base = build_evaluation_plan(SimpleNamespace(records=tuple(rows)),
                                 {"max_trajectories_per_task": None, "windows_per_trajectory": 2, "stride_pairs": [[1, 1]]},
                                 {"views": ["top"], "max_pixels": None, "stride_pairs": [[1, 1]]},
                                 training_dataset=SimpleNamespace(records=()), roles=("B", "C", "D"))
    return base, {str(root): group for root, group in zip(roots, ("L1L2", "L3L4"), strict=True)}


def test_full_dense_groups_follow_exact_cache_identity_without_changing_prompts(cohort):
    base, mapping = grouped_base(cohort)
    before = base.manifest()
    plan = build_dense_plan(base, max_trajectories_per_task=None, frame_stride=3, group_by_cache=mapping)
    seen = []

    def capture(processor, examples, max_pixels, device):
        seen.extend(examples)
        return {}

    result = evaluate(Prediction(), None, plan, device="cpu", max_pixels=None, collate_fn=capture)
    assert plan.settings["metric_scope"] == "full_heldout_dense"
    assert plan.manifest()["trajectory_count"] == 6
    assert result["model_calls"] == len(seen) == 6 * 4 * 3
    assert {window.group for window in plan.windows} == {"L1L2", "L3L4"}
    assert all(window.group == mapping[str(window.trajectory.cache_root)] for window in plan.windows)
    assert all(example.task == "assemble" and "L1L2" not in example.prompt and "L3L4" not in example.prompt for example in seen)
    assert base.manifest() == before
    assert result["diagnostics"]["per_group"]["L1L2"]["C"]["support"] == 12
    assert result["metrics"]["accumulated/overall/support"] == 24
    assert "accumulated/group/L3L4/constant_0.5_normalized_rmse" in result["metrics"]
    corrupted = deepcopy(result)
    corrupted["records"][0]["group"] = "foreign"
    with pytest.raises(ValueError, match="exact held-out cache"):
        dense_series(corrupted, plan)


def test_exact_cache_group_mapping_rejects_partial_and_ambiguous_paths(cohort):
    base, mapping = grouped_base(cohort)
    with pytest.raises(ValueError, match="exactly cover"):
        build_dense_plan(base, max_trajectories_per_task=None, group_by_cache=dict(list(mapping.items())[:1]))
    root = next(iter(mapping))
    with pytest.raises(ValueError, match="duplicate canonical"):
        build_dense_plan(base, group_by_cache={**mapping, root + "/processed_dataset": "duplicate"})


def test_each_group_extremum_uses_same_during_training_records_and_is_audited(cohort, tmp_path, monkeypatch):
    import grana.training.deploy_dense as dense
    monkeypatch.setattr(dense, "require_encoder", lambda: "unused")
    monkeypatch.setattr(dense, "_video", lambda path, frames, fps: path.write_bytes(b"test-video"))
    base, mapping = grouped_base(cohort)
    result, plan = assessed(base, max_trajectories_per_task=None, group_by_cache=mapping)
    result.update(optimizer_steps=250, training_elapsed_seconds=123., provenance={"mode": "during_training"})
    saved = tmp_path / "evaluation-000250.json"
    saved.write_text(json.dumps(result))
    assert verify_evaluation(saved, 250)["records"] == result["records"]
    bundle = render_dense_evaluation(result, plan, output_dir=tmp_path / "groups", step=250, training_elapsed_seconds=123.)
    assert bundle["provenance"]["mode"] == "during_training"
    assert len(bundle["items"]) == 4 and len(verify_dense_bundle(bundle["manifest"], result)) == 8
    assert {(item["group"], item["selection"]) for item in bundle["items"]} == {
        (group, selection) for group in mapping.values() for selection in ("best_raw_rmse", "worst_raw_rmse")}
    for item in bundle["items"]:
        assert mapping[item["cache"]] == item["group"] and item["candidate_trajectories"] == 3
    corrupted = deepcopy(bundle)
    corrupted["items"][0]["cache"] = "foreign-cache"
    Path(bundle["manifest"]).write_text(json.dumps(corrupted))
    with pytest.raises(ValueError, match="best/worst selection"):
        verify_dense_bundle(bundle["manifest"], result)
    corrupted = deepcopy(result)
    corrupted["records"][0]["elapsed_seconds"] += 1.
    Path(bundle["manifest"]).write_text(json.dumps(bundle))
    with pytest.raises(ValueError, match="exact persisted"):
        verify_dense_bundle(bundle["manifest"], corrupted)
    corrupted = deepcopy(result)
    corrupted["diagnostics"]["overall"]["C"]["support"] += 1
    saved.write_text(json.dumps(corrupted))
    with pytest.raises(ValueError, match="Independent quality audit differs"):
        verify_evaluation(saved, 250)


def test_display_caps_at_96_measured_frames_without_any_extra_prediction(cohort, tmp_path, monkeypatch):
    from PIL import Image
    import grana.training.deploy_dense as dense

    base = cohort(roles=("A",), count=1)
    source = base.dataset.records[0]
    pixels = np.zeros((101, 2, 3, 3), dtype=np.uint8)
    rgb = tmp_path / "long-rgb.npy"
    np.save(rgb, pixels)
    row = replace(source, num_frames=101, frame_paths=(rgb,), frame_shape=pixels.shape,
                  progress=tuple(index / 100 for index in range(101)))
    base = build_evaluation_plan(SimpleNamespace(records=(row,)),
                                 {"max_trajectories_per_task": None, "windows_per_trajectory": 2, "stride_pairs": [[1, 1]]},
                                 {"views": ["top"], "max_pixels": None, "stride_pairs": [[1, 1]]},
                                 training_dataset=SimpleNamespace(records=()), roles=("A",))
    plan = build_dense_plan(base, max_trajectories_per_task=None, frame_stride=1)
    result = evaluate(Prediction(), None, plan, device="cpu", max_pixels=None, collate_fn=lambda *args: {})
    monkeypatch.setattr(dense, "require_encoder", lambda: "unused")
    monkeypatch.setattr(dense, "_frame", lambda *args, **kwargs: Image.new("RGB", dense.SIZE))
    encoded_counts = []

    def video(path, frames, fps):
        encoded_counts.append(sum(1 for _ in frames))
        path.write_bytes(b"test-video")

    monkeypatch.setattr(dense, "_video", video)
    bundle = render_dense_evaluation(result, plan, output_dir=tmp_path / "display", step=250, training_elapsed_seconds=1.)
    assert result["model_calls"] == 101
    assert bundle["evaluated_pois"] == 101 and encoded_counts == [96]
    assert len(bundle["items"][0]["video_pois"]) == 96
    assert bundle["items"][0]["video_pois"][0] == 0 and bundle["items"][0]["video_pois"][-1] == 100


def test_independent_audit_does_not_call_training_metric_helpers(cohort, tmp_path, monkeypatch):
    import grana.training.evaluation as native
    import grana.training.deploy_dense as dense

    result, _ = assessed(cohort())
    result.update(optimizer_steps=250, training_elapsed_seconds=123.)
    path = tmp_path / "independent.json"
    path.write_text(json.dumps(result))

    def forbidden(*args, **kwargs):
        pytest.fail("independent recomputation must not call production metric helpers")

    monkeypatch.setattr(native, "record_diagnostics", forbidden)
    monkeypatch.setattr(dense, "dense_metrics", forbidden)
    verify_evaluation(path, 250)
    result["metrics"]["accumulated/overall/constant_0.5_normalized_rmse"] += .01
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="Independent quality audit differs"):
        verify_evaluation(path, 250)
