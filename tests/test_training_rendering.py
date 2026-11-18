# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Training replays must retain real RGB, held-out labels, and native C -> D."""
import copy
from dataclasses import replace
import hashlib
import json
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch

from grana.data.cache import Trajectory
from grana.data.sampling import stage_relative_targets
from grana.training.evaluation import build_evaluation_plan, evaluate
from grana.training.rendering import render_evaluation


class Predictions(torch.nn.Module):
    def forward(self, inputs, roles, stage_counts):
        if roles == ["C"]:
            logits = torch.zeros(1, 8, stage_counts[0])
            logits[:, :, -1] = 4
            return {"stage_logits": logits}
        return {"progress_logits": torch.zeros(1, 8, 10)}


def inventory(rows):
    return SimpleNamespace(records=tuple(rows), by_role={role: {} for role in "ABC"})


@pytest.fixture
def evaluated(tmp_path):
    def make(identity="heldout", task="pick", roles=("A", "B", "C"), windows=2):
        paths = []
        for view in ("top", "wrist"):
            pixels = np.zeros((8, 16, 20, 3), dtype=np.uint8)
            pixels[:] = (10, 20, 30)
            pixels[:, :, :, 0 if view == "top" else 1] = np.arange(8, dtype=np.uint8)[:, None, None] * 25
            path = tmp_path / f"{identity}-{view}.npy"
            np.save(path, pixels)
            paths.append(path)
        trajectory = Trajectory(tmp_path, identity, task, 8, tuple(paths), pixels.shape, ("top", "wrist"), roles,
                                tuple(i / 7 for i in range(8)), (0, 0, 0, 0, 1, 1, 1, 1),
                                (.1, .3, .6, .9, .1, .3, .6, .9), ("Reach", "Pick"), "Pick the object.")
        plan = build_evaluation_plan(inventory([trajectory]),
                                     {"windows_per_trajectory": windows, "max_trajectories_per_task": 1, "stride_pairs": [[1, 1]]},
                                     {"views": ["top"], "max_pixels": None, "stride_pairs": [[1, 1]]},
                                     training_dataset=inventory([]))
        result = evaluate(Predictions(), None, plan, device="cpu", max_pixels=None,
                          collate_fn=lambda *args: {})
        return result, plan, trajectory
    return make


def test_real_rgb_and_native_targets_survive_training_step_replay(evaluated, tmp_path):
    result, plan, trajectory = evaluated()
    originals = {path: path.read_bytes() for path in trajectory.frame_paths}
    bundle = render_evaluation(result, plan, output_dir=tmp_path / "renders", step=50, training_elapsed_seconds=123.5)
    assert bundle["optimizer_steps"] == 50 and bundle["training_elapsed_seconds"] == 123.5
    assert bundle["plan_sha256"] == result["plan_sha256"]
    assert bundle["replay_type"] == "sampled_heldout_windows_not_original_speed"
    assert bundle["rendered_trajectories"] == 1
    item = bundle["items"][0]
    assert item["view_names"] == ["top", "wrist"]
    assert item["sampled_pois"] == item["replay_pois"] == [2, 5]
    assert item["metrics"]["role_A_poi_rmse"] == pytest.approx(result["metrics"]["role_A_poi_rmse"])
    assert item["metrics"]["role_C_poi_accuracy"] == .5
    d_points = [point for point in item["points"] if point["role"] == "D"]
    assert [(point["focus_stage"], point["target"]) for point in d_points] == [(1, 0.), (1, .3)]
    assert all(point["prediction"] == pytest.approx(.5) for point in d_points)
    # The PNG is the middle replay POI (frame 5); both raw camera payloads survive.
    with Image.open(item["image"]) as image:
        assert image.size == (1280, 960)
        assert image.getpixel((150, 350)) == (125, 20, 30)
        assert image.getpixel((455, 350)) == (10, 125, 30)
    with Image.open(item["video"]) as video:
        assert video.format == "GIF" and video.n_frames == 2
        assert video.info["duration"] == 500
    assert item["image_sha256"] == hashlib.sha256(open(item["image"], "rb").read()).hexdigest()
    assert item["video_sha256"] == hashlib.sha256(open(item["video"], "rb").read()).hexdigest()
    assert json.loads(open(bundle["manifest"]).read()) == bundle
    assert all(path.read_bytes() == before for path, before in originals.items())
    with pytest.raises(ValueError, match="fresh render"):
        render_evaluation(result, plan, output_dir=tmp_path / "renders", step=50, training_elapsed_seconds=123.5)


@pytest.mark.parametrize("corruption", ["foreign_plan", "missing_prediction", "repeated_prediction", "wrong_camera", "wrong_poi_geometry", "relabelled_gt", "wrong_d_focus"])
def test_refuse_foreign_or_incomplete_predictions_before_any_media(evaluated, tmp_path, corruption):
    result, plan, _ = evaluated()
    result = copy.deepcopy(result)
    if corruption == "foreign_plan":
        result["plan_sha256"] = "foreign"
    elif corruption == "missing_prediction":
        result["records"].pop()
    elif corruption == "repeated_prediction":
        result["records"][-1] = result["records"][0]
    elif corruption == "wrong_camera":
        result["records"][0]["view_names"].reverse()
    elif corruption == "wrong_poi_geometry":
        result["records"][0]["frame_indices"][5] += 1
    elif corruption == "relabelled_gt":
        result["records"][0]["targets"][5] = .99
    else:
        d_record = next(row for row in result["records"] if row["role"] == "D")
        d_record["focus_stage"] = 0
        trajectory = plan.windows[0].trajectory
        d_record["targets"] = list(stage_relative_targets(tuple(trajectory.stage_ids[i] for i in d_record["frame_indices"]),
                                                          tuple(trajectory.within_stage_progress[i] for i in d_record["frame_indices"]),
                                                          0, len(trajectory.stage_prompts)))
    with pytest.raises(ValueError):
        render_evaluation(result, plan, output_dir=tmp_path / "renders", step=50, training_elapsed_seconds=123.5)
    assert not (tmp_path / "renders").exists()


def test_unassessed_roles_remain_absent_and_replay_cap_does_not_change_curve(evaluated, tmp_path):
    result, plan, _ = evaluated(roles=("A",), windows=8)
    bundle = render_evaluation(result, plan, output_dir=tmp_path / "renders", step=2,
                               training_elapsed_seconds=20, max_frames=3)
    item = bundle["items"][0]
    assert item["sampled_pois"] == list(range(8)) and item["replay_pois"] == [0, 4, 7]
    assert len(item["points"]) == 8 and all(point["role"] == "A" for point in item["points"])
    assert set(item["metrics"]) == {"role_A_poi_rmse"}
    with Image.open(item["video"]) as video:
        assert video.n_frames == 3


def test_trajectory_media_cap_is_task_balanced_and_does_not_cap_evaluation(evaluated, tmp_path):
    _, _, trajectory = evaluated()
    extra = [replace(trajectory, task=task, trajectory_id=identity)
             for task, identity in (("pick", "pick-second"), ("stack", "stack-first"))]
    plan = build_evaluation_plan(inventory([trajectory, *extra]), {"windows_per_trajectory": 2, "max_trajectories_per_task": None,
                                                                 "stride_pairs": [[1, 1]]},
                                 {"views": ["top"], "max_pixels": None, "stride_pairs": [[1, 1]]},
                                 training_dataset=inventory([]))
    result = evaluate(Predictions(), None, plan, device="cpu", max_pixels=None, collate_fn=lambda *args: {})
    bundle = render_evaluation(result, plan, output_dir=tmp_path / "renders", step=5,
                               training_elapsed_seconds=100, max_trajectories=2, max_frames=1)
    assert bundle["available_trajectories"] == 3 and bundle["rendered_trajectories"] == 2
    assert [item["task"] for item in bundle["items"]] == ["pick", "stack"]
    assert result["model_calls"] == 24 and bundle["evaluation_metrics"] == result["metrics"]
