# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from grana.data.cache import Trajectory
from grana.training.evaluation import build_evaluation_plan, evaluate
from grana.training.render_plan import build_render_plan


DATA = {"views": ["top"], "max_pixels": None, "stride_pairs": [[1, 1], [3, 2]]}


def dataset(rows):
    return SimpleNamespace(records=tuple(rows))


@pytest.fixture
def trajectory(tmp_path):
    def make(identity, *, task="task", frames=101, roles=("A", "B", "C")):
        path = tmp_path / f"{identity}.npy"
        pixels = np.zeros((frames, 2, 3, 3), dtype=np.uint8)
        np.save(path, pixels)
        return Trajectory(tmp_path, identity, task, frames, (path,), pixels.shape, ("top",), roles,
                          tuple(index / max(1, frames - 1) for index in range(frames)),
                          (0,) * frames, (.2,) * frames, ("First", "Second", "Third"), "Complete the task.")
    return make


def base_plan(rows, *, data=DATA):
    return build_evaluation_plan(dataset(rows),
                                 {"max_trajectories_per_task": None, "windows_per_trajectory": 2, "seed": 1729,
                                  "stride_pairs": data["stride_pairs"]},
                                 data, training_dataset=dataset([]))


def grouped_pois(plan):
    grouped = defaultdict(list)
    for window in plan.windows:
        grouped[(window.trajectory.task, window.trajectory.trajectory_id)].append(window.poi)
    return grouped


def test_full_range_uniform_real_samples_keep_base_plan_unchanged(trajectory, monkeypatch):
    row = trajectory("heldout")
    base = base_plan([row])
    before = base.manifest()
    monkeypatch.setattr(Trajectory, "images", lambda *args: pytest.fail("planning must not read RGB"))
    render = build_render_plan(base)
    pois = grouped_pois(render)[("task", "heldout")]
    assert len(pois) == 24 and pois[0] == 0 and pois[-1] == 100
    assert all(isinstance(poi, int) and 0 <= poi < row.num_frames for poi in pois)
    assert set(right - left for left, right in zip(pois, pois[1:])) == {4, 5}
    assert base.manifest() == before
    assert render.dataset is base.dataset and all(window.trajectory is row for window in render.windows)
    assert render.settings["purpose"] == "sampled_rendering"
    assert render.settings["metric_scope"] == "sampled_heldout_render_cohort"
    assert render.settings["source_plan_sha256"] == before["plan_sha256"]
    assert render.manifest()["plan_sha256"] != before["plan_sha256"]
    assert render.manifest()["model_calls"] == 24 * 4


def test_short_trajectories_use_each_physical_frame_once(trajectory):
    rows = [trajectory(f"short-{frames}", frames=frames) for frames in [1, 2, 7, 24]]
    render = build_render_plan(base_plan(rows))
    for (_, identity), pois in grouped_pois(render).items():
        count = int(identity.split("-")[-1])
        assert pois == list(range(count))
        assert len(pois) == len(set(pois))
    assert render.manifest()["trajectory_count"] == 4


def test_selection_determinism_caps_and_legal_strides(trajectory):
    rows = [trajectory(f"{task}-{index}", task=task) for task in ["one", "two"] for index in range(7)]
    base = base_plan(rows)
    render = build_render_plan(base, max_trajectories_per_task=3, points_per_trajectory=6)
    again = build_render_plan(base, max_trajectories_per_task=3, points_per_trajectory=6)
    assert render.manifest() == again.manifest()
    reversed_base = replace(base, windows=base.windows[::-1])
    reordered = build_render_plan(reversed_base, max_trajectories_per_task=3, points_per_trajectory=6)
    signature = lambda plan: [(w.trajectory.task, w.trajectory.trajectory_id, w.poi, w.stride_pair, w.roles)
                              for w in plan.windows]
    assert signature(render) == signature(reordered)
    assert len(render.windows) == 2 * 3 * 6
    assert all(task["selected_trajectories"] == 3 for task in render.tasks)
    assert all(len(pois) == 6 for pois in grouped_pois(render).values())
    assert all(window.stride_pair in {(1, 1), (3, 2)} for window in render.windows)
    assert all(window.roles == ("A", "B", "C", "D") for window in render.windows)
    assert all(window.manifest()["frame_indices"][5] == window.poi for window in render.windows)


def test_cap_preserves_rare_roles_and_rejects_impossible_coverage(trajectory):
    rows = [trajectory(f"atomic-{index}", roles=("A",)) for index in range(7)]
    rows.append(trajectory("native", roles=("B", "C")))
    base = base_plan(rows)
    render = build_render_plan(base, max_trajectories_per_task=2, points_per_trajectory=3)
    assert render.tasks[0]["roles"] == ["A", "B", "C", "D"]
    assert render.tasks[0]["selected_trajectories"] == 2
    assert set().union(*(set(window.roles) for window in render.windows)) == set("ABCD")
    assert all(window.roles == ("B", "C", "D") for window in render.windows
               if window.trajectory.trajectory_id == "native")
    with pytest.raises(ValueError, match="cap omits eligible roles"):
        build_render_plan(base, max_trajectories_per_task=1)
    full = trajectory("all-roles")
    covered = build_render_plan(base_plan([*rows, full]), max_trajectories_per_task=1)
    assert covered.tasks[0]["selected_ids"] == ["all-roles"]


def test_only_base_windows_are_sources_and_labels_are_not_inspected(trajectory):
    class ForbiddenLabels:
        def __iter__(self):
            pytest.fail("render selection must not inspect labels")

        def __getitem__(self, index):
            pytest.fail("render selection must not inspect labels")

    class ForbiddenDataset:
        @property
        def records(self):
            pytest.fail("render selection must not enumerate the dataset")

    base = base_plan([trajectory("heldout", roles=("A",)), trajectory("other-heldout", roles=("A",))])
    row = replace(base.windows[0].trajectory, progress=ForbiddenLabels(), stage_ids=ForbiddenLabels(),
                  within_stage_progress=ForbiddenLabels())
    selected = replace(base, dataset=ForbiddenDataset(),
                       windows=tuple(replace(window, trajectory=row) for window in base.windows
                                     if window.trajectory.trajectory_id == row.trajectory_id))
    render = build_render_plan(selected)
    assert {window.trajectory.trajectory_id for window in render.windows} == {row.trajectory_id}
    assert render.tasks[0]["selected_ids"] == [row.trajectory_id]
    assert all(window.roles == ("A",) for window in render.windows)


@pytest.mark.parametrize("max_pixels", [None, 76800])
def test_preprocessing_views_and_nested_settings_are_preserved_without_aliasing(trajectory, max_pixels):
    base = base_plan([trajectory("heldout")], data={**DATA, "max_pixels": max_pixels})
    render = build_render_plan(base)
    for key in ["preprocessing", "max_pixels", "views", "view_sets", "stride_pairs", "seed", "heldout_check"]:
        assert render.settings[key] == base.settings[key]
    render.settings["views"].append("new")
    render.settings["preprocessing"]["version"] = 99
    render.settings["stride_pairs"][0][0] = 999
    assert base.settings["views"] == ["top"]
    assert base.settings["preprocessing"]["version"] == 1
    assert base.settings["stride_pairs"][0][0] == 1


def test_render_plan_executes_with_existing_evaluate_and_predicted_d(trajectory):
    base = base_plan([trajectory("heldout", frames=5)])
    plan = build_render_plan(base, points_per_trajectory=3)

    class Model(nn.Module):
        def forward(self, inputs, roles, stage_counts):
            assert not self.training and not torch.is_grad_enabled()
            if roles == ["C"]:
                assert inputs["example"].focus_stage is None
                logits = torch.zeros(1, 8, stage_counts[0])
                logits[:, :, 2] = 5
                return {"stage_logits": logits}
            if roles == ["D"]:
                assert inputs["example"].focus_stage == 2
            return {"progress_logits": torch.zeros(1, 8, 10)}

    def collate(_processor, examples, max_pixels, _device):
        assert max_pixels is None
        return {"example": examples[0]}

    model = Model().train()
    result = evaluate(model, None, plan, device="cpu", max_pixels=None, collate_fn=collate)
    assert model.training
    assert result["plan_sha256"] == plan.manifest()["plan_sha256"]
    assert result["model_calls"] == 12 and result["task_count"] == 1
    assert {record["poi"] for record in result["records"]} == {0, 2, 4}
    assert all(record["focus_stage"] == 2 for record in result["records"] if record["role"] == "D")
    assert all(record["focus_stage"] is None for record in result["records"] if record["role"] == "C")
    assert base.manifest()["model_calls"] == 8


@pytest.mark.parametrize("keyword, value", [("max_trajectories_per_task", value) for value in [True, 0, -1, 1.5]]
                         + [("points_per_trajectory", value) for value in [True, 0, 1, -1, None, 1.5]])
def test_invalid_limits_fail_before_plan_construction(trajectory, keyword, value):
    with pytest.raises(ValueError, match=keyword):
        build_render_plan(base_plan([trajectory("heldout")]), **{keyword: value})


def test_null_trajectory_cap_keeps_every_admitted_trajectory(trajectory):
    rows = [trajectory(f"heldout-{index}") for index in range(7)]
    plan = build_render_plan(base_plan(rows), max_trajectories_per_task=None, points_per_trajectory=2)
    assert plan.manifest()["trajectory_count"] == 7
    assert len(plan.windows) == 14


def test_empty_base_plan_is_rejected(trajectory):
    base = base_plan([trajectory("heldout")])
    with pytest.raises(ValueError, match="nonempty held-out"):
        build_render_plan(replace(base, windows=()))
