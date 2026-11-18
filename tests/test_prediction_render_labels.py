# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rendered readout labels must identify the role, actual input slot and source frame."""
from types import SimpleNamespace

from PIL import Image


def test_training_curve_labels_follow_the_record_slot(monkeypatch):
    from grana.training import rendering

    labels = []
    monkeypatch.setattr(rendering, "_text", lambda draw, xy, text, *args, **kwargs: labels.append(str(text)))
    samples = {128: {"D": {"poi_slot": 7, "predictions": [.4] * 8, "targets": [.5] * 8}}}
    trajectory = SimpleNamespace(num_frames=200, stage_prompts=("Reach", "Place"))
    rendering._plot(Image.new("RGB", (640, 200)), (0, 0, 630, 195), samples, trajectory, "D", 128)
    assert "D | POI slot 7 | predicted-focus progress" in labels
    assert any("Source frame index | cursor 128" in text for text in labels)


def test_dense_frame_identifies_both_roles_and_the_actual_readout(monkeypatch):
    from grana.training import deploy_dense

    labels = []
    monkeypatch.setattr(deploy_dense, "_text", lambda draw, xy, text, *args, **kwargs: labels.append(str(text)))
    point = {"poi": 128, "source_frame": 128, "poi_slot": 7, "prediction": 1.4, "target": 1.5,
             "predicted_stage": 1, "target_stage": 1, "focus_progress": .4, "focus_target": .5}
    row = {"task": "assemble", "trajectory_id": "example", "route": "predicted_stage_route",
           "poi_slots": [7], "num_frames": 200, "scale": 2, "points": [point],
           "stage_prompts": ["Reach", "Place"],
           "metrics": {"raw_rmse": .1, "normalized_rmse": .05, "stage_accuracy": 1.0}}
    trajectory = SimpleNamespace(view_names=("top",), images=lambda indices: (Image.new("RGB", (20, 20)),))
    background = deploy_dense._background(row)
    deploy_dense._frame(background, row, trajectory, point, step=10, elapsed=1,
                        provenance={"mode": "during_training"}, selection="example", frame_stride=10)
    assert "Blue: C + D, POI slot 7" in labels
    assert any("POI slot 7 | source frame 128" in text for text in labels)
    assert any("C[7] stage" in text and "D[7]" in text for text in labels)
    assert any("Offline POI replay" in text for text in labels)


def test_historical_comparison_curve_remains_explicit_poi(monkeypatch):
    from scripts import render_inference_comparison as comparison

    painter = comparison.Painter()
    labels = []
    monkeypatch.setattr(painter, "text", lambda draw, xy, text, *args, **kwargs: labels.append(str(text)))
    episode = {"id": "example", "stage_prompts": ["Reach", "Place"],
               "windows": [{"poi": 120, "gt": {"whole_progress": .4, "stage_id": 0}},
                           {"poi": 128, "gt": {"whole_progress": .5, "stage_id": 1}}]}
    comparison.plot_points(Image.new("RGB", (700, 280)), (0, 0, 690, 270),
                           episode, [], {}, 1, "C", painter)
    assert "C · POI slot 5 · first unfinished stage" in labels
    assert any("Source frame index · cursor 128" in text for text in labels)
