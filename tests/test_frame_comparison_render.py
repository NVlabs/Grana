# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image, ImageDraw
import pytest

from grana.inference.window import compose_frame_indices
from scripts.inference import render_frame_comparison as render


def experiment(tmp_path):
    seeds, weights = [11, 29], "a" * 64
    trajectory = {"key": "example", "trajectory_id": "example-trajectory", "group": "L3L4",
                  "num_frames": 50, "stage_count": 10, "source_fps": 10,
                  "sources": {}, "request_path": str(tmp_path / "request.json"),
                  "metadata_path": str(tmp_path / "metadata.json")}
    paths = {}
    for index, view in enumerate(("top", "left", "right", "bottom")):
        path = tmp_path / f"{view}.npy"
        array = np.full((50, 24, 32, 3), 25 + index * 40, dtype=np.uint8)
        np.save(path, array)
        stat = path.stat()
        paths[view] = str(path)
        trajectory["sources"][view] = {"shape": list(array.shape), "path": str(path),
                                       "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    request = {"views": paths, "stage_prompts": [f"Stage {i} action description" for i in range(10)], "task": "test"}
    metadata = {"stage_ids": [6] * 50, "within_stage_progress": [.5] * 50}
    Path(trajectory["request_path"]).write_text(json.dumps(request))
    Path(trajectory["metadata_path"]).write_text(json.dumps(metadata))
    plan = {"seeds": seeds, "checkpoint": {"weights_sha256": weights}, "trajectories": [trajectory]}
    records = {}
    targets = {11: [10, 20, 30], 29: [10, 20, 40]}
    for seed in seeds:
        for frame in targets[seed]:
            for mode, slot in (("poi", 5), ("last", 7)):
                c = 3 if frame == 20 and mode == "last" else 6
                d = .5 if mode == "poi" else .7
                row = {"seed": seed, "key": trajectory["key"], "kind": "quality", "source_frame": frame,
                       "mode": mode, "slot": slot, "C_ids": [c] * 8, "D_values": [d] * 8,
                       "D_focus_stage": c, "frame_indices": list(compose_frame_indices(frame, 50, prediction_frame=mode)),
                       "gt": {"stage_id": 6, "within_stage_progress": .5},
                       "request_id": f"{seed}-{frame}-{mode}",
                       "input_sha256": hashlib.sha256(f"{frame}-{mode}".encode()).hexdigest()}
                row["prediction_sha256"] = render.summary.prediction_digest(row)
                records[seed, trajectory["key"], "quality", frame, mode] = row
    metrics = {mode: render.summary.metrics([r for identity, r in records.items() if identity[-1] == mode], 10)
               for mode in ("poi", "last")}
    report = {"checkpoint": plan["checkpoint"], "trajectories": [{"key": trajectory["key"],
               "trajectory_id": trajectory["trajectory_id"], "metrics": metrics,
               "delta_last_minus_poi": {"normalized_rmse": metrics["last"]["normalized_rmse"] - metrics["poi"]["normalized_rmse"]}}]}
    return plan, records, report


def test_unique_curves_preserve_original_repeat_weighting_and_catastrophic_stage_error(tmp_path):
    plan, records, report = experiment(tmp_path)
    case = render.select_cases(plan, records, report, 1)[0]
    assert case["unique_targets"] == 4
    assert case["original_samples_per_mode"] == 6
    assert [p["frame"] for p in case["points"]] == [10, 20, 30, 40]
    assert [p["repeat_count"] for p in case["points"]] == [2, 2, 1, 1]
    assert case["original_normalized_rmse"]["last"] != case["unique_normalized_rmse"]["last"]
    assert case["original_normalized_rmse"]["poi"] == 0
    event = case["gap_analysis"]["top_points"][0]
    assert event["frame"] == 20 and event["gt_stage"] == event["poi_C"] == 6 and event["last_C"] == 3
    assert event["weighted_gap_stage_units"] == pytest.approx(2 * (3.7 - 6.5) ** 2)
    assert event["diagnostic_rmse_if_this_target_removed"]["last"] == pytest.approx(.02)


def test_ranking_uses_original_metric_not_unique_union_metric(tmp_path):
    plan, records, report = experiment(tmp_path)
    cases = render.select_cases(plan, records, report, 1)
    assert cases[0]["rank"] == 1
    altered = deepcopy(report)
    altered["trajectories"][0]["metrics"]["last"]["normalized_rmse"] = .2
    with pytest.raises(ValueError, match="Ranking RMSE"):
        render.select_cases(plan, records, altered, 1)
    altered = deepcopy(report)
    altered["trajectories"][0]["delta_last_minus_poi"]["normalized_rmse"] = -1
    with pytest.raises(ValueError, match="Not enough"):
        render.select_cases(plan, records, altered, 1)


@pytest.mark.parametrize("mutation,message", [
    (lambda records: records.pop((29, "example", "quality", 10, "last")), "repeat counts"),
    (lambda records: records.pop((29, "example", "quality", 40, "last")), "same physical"),
    (lambda records: records[29, "example", "quality", 10, "last"].update(prediction_sha256="b" * 64), "nonidentical"),
    (lambda records: records[29, "example", "quality", 10, "last"].update(input_sha256="c" * 64), "inputs differ"),
])
def test_invalid_pairs_or_repeated_predictions_are_rejected(tmp_path, mutation, message):
    plan, records, report = experiment(tmp_path)
    mutation(records)
    with pytest.raises(ValueError, match=message):
        render.select_cases(plan, records, report, 1)


def test_render_retains_same_target_and_extra_poi_observations(tmp_path):
    plan, records, report = experiment(tmp_path)
    case = render.select_cases(plan, records, report, 1)[0]
    trajectory = case["trajectory"]
    metadata = json.loads(Path(trajectory["metadata_path"]).read_text())
    request = json.loads(Path(trajectory["request_path"]).read_text())
    arrays = {view: np.load(path) for view, path in request["views"].items()}
    base, gt = render.base_canvas(case, metadata, request)
    frame = render.render_frame(case, 1, arrays, metadata, request, base, gt, 8, hold=True)
    assert frame.size == (render.WIDTH, render.HEIGHT)
    assert frame.getpixel((30, 180)) == (25, 25, 25)
    assert frame.getpixel((500, 180)) == (65, 65, 65)
    assert case["points"][1]["poi"]["frame_indices"][5] == 20
    assert case["points"][1]["last"]["frame_indices"][7] == 20
    assert case["points"][1]["poi"]["frame_indices"][7] == 28
    assert max(case["points"][1]["last"]["frame_indices"]) == 20


def test_local_curve_limits_keep_ground_truth_between_measured_targets_visible(tmp_path):
    plan, records, report = experiment(tmp_path)
    case = render.select_cases(plan, records, report, 1)[0]
    gt = [6.5] * 50
    gt[25] = 9.5
    drawing = ImageDraw.Draw(Image.new("RGB", (render.WIDTH, render.HEIGHT)))
    xy = render.plot(drawing, render.LOCAL_PLOT, case["points"], gt, 10, 40,
                     stage_count=10, current=20, source_fps=10)
    _, y = xy(25, 9.5)
    assert render.LOCAL_PLOT[1] <= y <= render.LOCAL_PLOT[3]


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="H.264 tools unavailable")
def test_encoded_pair_video_is_fully_decodable_with_hotspot_positions(tmp_path):
    plan, records, report = experiment(tmp_path)
    case = render.select_cases(plan, records, report, 1)[0]
    output = tmp_path / "videos"
    output.mkdir()
    artifact = render.render_case(case, output, 2, .5)
    assert artifact["full_decode"] == "passed"
    assert artifact["frames"] == 7 and artifact["duration_seconds"] == 3.5
    assert Path(artifact["video"]).is_file() and Path(artifact["poster"]).is_file()
    main = next(point for point in artifact["hotspot_video_positions"] if point["event_rank"] == 1)
    assert main["source_frame"] == 20 and main["video_seconds"] == 1.0
