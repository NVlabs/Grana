# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Saved query occurrences, rather than literal slot five, drive every consumer."""
import copy
from dataclasses import dataclass, replace
from hashlib import sha256
import json
from types import SimpleNamespace

import pytest

from grana.training import evaluation
from grana.training.campaign_report import independent_quality_audit, verify_evaluation
from grana.training.deploy_dense import dense_series
from grana.training.distributed_evaluation import _contribution
from grana.training.evaluation import EvaluationWindow, EvaluationPlan, record_diagnostics, record_poi_slot
from grana.training.losses import FRAME_WEIGHTS
from grana.training.rendering import _role_metrics, _validated_records, render_evaluation
from test_training_rendering import Predictions, evaluated


@dataclass(frozen=True)
class ReversedWindow(EvaluationWindow):
    def manifest(self):
        row = super().manifest()
        return dict(row, frame_indices=row["frame_indices"][::-1], poi_slot=2,
                    frame_weight_order=list(reversed(range(8))))


@dataclass(frozen=True)
class ReversedPlan(EvaluationPlan):
    """Explicit test-only plan; production held-out plans remain unaugmented."""
    def manifest(self):
        result = super().manifest()
        result.pop("plan_sha256")
        result["score_slot"] = 2
        result["plan_sha256"] = sha256(json.dumps(result, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        return result


def reverse_records(result, plan):
    windows = tuple(ReversedWindow(w.trajectory, w.poi, w.stride_pair, w.roles, w.group) for w in plan.windows)
    reordered_plan = ReversedPlan(plan.dataset, windows, plan.tasks, plan.settings)
    reordered = copy.deepcopy(result)
    for record in reordered["records"]:
        for name in ("frame_indices", "predictions", "targets", "slot_losses"):
            record[name] = record[name][::-1]
        record.update(poi_slot=2, frame_weight_order=list(reversed(range(8))))
    reordered.update(plan_sha256=reordered_plan.manifest()["plan_sha256"], score_slot=2)
    reordered["diagnostics"], metrics = record_diagnostics(reordered["records"])
    reordered["metrics"].update(metrics)
    return reordered, reordered_plan


def test_reordered_records_keep_native_queries_metrics_losses_and_media(evaluated, tmp_path):
    original, plan, _ = evaluated()
    original_plan = plan.manifest()
    reordered, reordered_plan = reverse_records(original, plan)
    assert reordered["diagnostics"]["score_slot"] == 2
    assert reordered["metrics"] == original["metrics"]
    assert independent_quality_audit(reordered)["status"] == "passed"
    reordered["optimizer_steps"] = 4
    path = tmp_path / "evaluation.json"
    path.write_text(json.dumps(reordered))
    assert verify_evaluation(path, 4) == reordered
    for before, after in zip(original["records"], reordered["records"], strict=True):
        assert _contribution(after, FRAME_WEIGHTS) == _contribution(before, FRAME_WEIGHTS)
    before = _validated_records(original, plan)
    after = _validated_records(reordered, reordered_plan)
    for identity in before:
        assert _role_metrics(after[identity]) == _role_metrics(before[identity])
    original_series = dense_series(original, plan)
    reordered_series = dense_series(reordered, reordered_plan)
    for original_row, reordered_row in zip(original_series, reordered_series, strict=True):
        # Predictions and physical source frames stay identical. The visible
        # slot provenance must change when the query occurrence is reordered.
        assert original_row.pop("poi_slots") == [5]
        assert reordered_row.pop("poi_slots") == [2]
        for original_point, reordered_point in zip(original_row["points"], reordered_row["points"], strict=True):
            assert original_point.pop("poi_slot") == 5
            assert reordered_point.pop("poi_slot") == 2
            assert original_point["source_frame"] == reordered_point["source_frame"] == original_point["poi"]
        assert original_row == reordered_row
    rendered = render_evaluation(reordered, reordered_plan, output_dir=tmp_path / "render", step=4,
                                 training_elapsed_seconds=12)
    old_points = {(r["poi"], r["role"]): r for r in original["records"]}
    for point in rendered["items"][0]["points"]:
        row = old_points[point["poi"], point["role"]]
        assert point["prediction"] == row["predictions"][5]
        assert point["target"] == row["targets"][5]
        assert point["poi_slot"] == 2 and point["source_frame"] == row["frame_indices"][5]
    assert plan.manifest() == original_plan


@pytest.mark.parametrize("slot", [True, -1, 8, 2.0, "2", None])
def test_records_reject_invalid_query_position(evaluated, slot):
    result, plan, _ = evaluated()
    result["records"][0]["poi_slot"] = slot
    with pytest.raises(ValueError, match="poi_slot"):
        record_diagnostics(result["records"])
    with pytest.raises(ValueError, match="poi_slot"):
        _validated_records(result, plan)


def test_record_slot_cannot_silently_reinterpret_a_frozen_plan(evaluated):
    result, plan, _ = evaluated()
    result["records"][0]["poi_slot"] = 2
    with pytest.raises(ValueError, match="query slot"):
        _validated_records(result, plan)


@pytest.mark.parametrize("damage", ["poi_slot", "frame_indices", "frame_weight_order"])
def test_training_geometry_cannot_enter_fixed_eval(evaluated, monkeypatch, damage):
    _, plan, _ = evaluated()
    original = evaluation.materialize_example

    def augmented(*args, **kwargs):
        assert kwargs["execution_mode"] == "eval"
        example = original(*args, **kwargs)
        changes = {"poi_slot": 2, "frame_indices": example.frame_indices[::-1],
                   "frame_weight_order": tuple(reversed(range(8)))}
        return replace(example, **{damage: changes[damage]})

    monkeypatch.setattr(evaluation, "materialize_example", augmented)
    with pytest.raises(ValueError, match="frozen evaluation geometry"):
        evaluation.evaluate(Predictions(), None, plan, device="cpu", max_pixels=None, collate_fn=lambda *args: {})


@pytest.mark.parametrize("order", [[False, 1, 2, 3, 4, 5, 6, 7], [0] * 8, list(range(7)), None])
def test_invalid_saved_weight_order_cannot_pass_media_audit(evaluated, order):
    result, plan, _ = evaluated()
    result["records"][0]["frame_weight_order"] = order
    with pytest.raises(ValueError, match="frame_weight_order"):
        _validated_records(result, plan)


def test_mixed_record_slots_are_scored_per_record(evaluated):
    result, plan, _ = evaluated()
    reversed_result, _ = reverse_records(result, plan)
    poi = result["records"][0]["poi"]
    mixed = [b if a["poi"] == poi else a
             for a, b in zip(result["records"], reversed_result["records"], strict=True)]
    diagnostics, metrics = record_diagnostics(mixed)
    assert diagnostics["score_slot"] is None
    assert diagnostics["overall"] == result["diagnostics"]["overall"]
    assert metrics == record_diagnostics(result["records"])[1]


def test_training_sampling_weights_do_not_change_frozen_evaluation_plan(evaluated):
    _, plan, _ = evaluated()
    settings = {"windows_per_trajectory": 2, "max_trajectories_per_task": 1, "stride_pairs": [[1, 1]]}
    data = {"views": ["top"], "max_pixels": None, "stride_pairs": [[1, 1]]}
    baseline = evaluation.build_evaluation_plan(plan.dataset, settings, data,
                                                training_dataset=SimpleNamespace(records=()))
    mixed = dict(data, train_frame_sampling={
        "frame_selection_profile_weights": {"regular_ctsp": .5, "irregular_stratified": .5},
        "query_order_profile_weights": {"forward": .5, "tail_rewind": .25, "backward": .25}})
    actual = evaluation.build_evaluation_plan(plan.dataset, settings, mixed,
                                              training_dataset=SimpleNamespace(records=()))
    assert actual.manifest() == baseline.manifest()
    assert all("poi_slot" not in window and "frame_weight_order" not in window
               for window in actual.manifest()["windows"])
    assert record_poi_slot({}) == 5
