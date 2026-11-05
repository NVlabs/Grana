# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Primary summaries must not substitute conditional progress errors."""
from copy import deepcopy
import math

import pytest

from grana.training.deploy_dense import dense_metrics
from grana.training.evaluation_reporting import (accumulated_scope_routes, compact_evaluation_metrics,
                                                diagnostic_metric_aliases, evaluation_report)


@pytest.mark.parametrize("partial", [{}, {"accumulated/overall/raw_rmse": .4},
                                    {"accumulated/overall/normalized_rmse": .04}])
def test_sampled_or_incomplete_metrics_never_invent_accumulated_quality(partial):
    metrics = {"role_B_poi_rmse": .22, "role_D_poi_rmse": .23, **partial}
    before = deepcopy(metrics)
    report = evaluation_report(metrics, scope="sampled")
    assert report["primary"]["status"] == "unavailable"
    assert report["primary"]["scope"] == "sampled"
    assert "raw_rmse" not in report["primary"] and "normalized_rmse" not in report["primary"]
    assert report["oracle_accumulated_comparison"]["per_task"] == {}
    assert report["conditional_diagnostics"]["diagnostics/progress/D_predicted_focus_relative_rmse"] == .23
    assert metrics == before


def test_primary_uses_measured_accumulated_pair_and_separates_oracle():
    metrics = {"role_B_poi_rmse": .22, "role_D_poi_rmse": .23,
               "accumulated/overall/raw_rmse": .4, "accumulated/overall/normalized_rmse": .04,
               "eval_quality/assemble/oracle_stage_route_accumulated_progress_raw_rmse": .22}
    before = deepcopy(metrics)
    report = evaluation_report(metrics, scope="dense")
    assert report["primary"]["route"] == "predicted_stage_C_to_D"
    assert report["primary"]["raw_rmse"] == .4
    assert report["primary"]["normalized_rmse"] == .04
    oracle = report["oracle_accumulated_comparison"]["per_task"]["assemble"]
    assert oracle["raw_rmse"] == .22 and "normalized_rmse" not in oracle
    assert metrics == before


def test_whole_task_and_mixed_routes_keep_pointwise_normalization():
    # Two A POIs and one staged POI: overall values must use the actual POI
    # weighting and each trajectory's own scale, not a mean of route RMSEs.
    whole_task = {"task": "single", "route": "compressed_whole_task", "scale": 1,
                  "points": [{"prediction": .2, "target": 0}, {"prediction": .2, "target": 0}]}
    staged = {"task": "staged", "route": "predicted_stage_route", "scale": 4,
              "points": [{"prediction": 2, "target": 0, "predicted_stage": 2, "target_stage": 0}]}
    a_report = evaluation_report(dense_metrics([whole_task]), scope="dense")
    assert a_report["primary"]["route"] == "whole_task_A"
    assert a_report["primary"]["raw_rmse"] == a_report["primary"]["normalized_rmse"] == pytest.approx(.2)
    assert a_report["oracle_accumulated_comparison"]["per_task"] == {}
    assert a_report["conditional_diagnostics"] == {}
    report = evaluation_report(dense_metrics([whole_task, staged]), scope="dense")
    assert report["primary"]["route"] == "mixed_C_to_D_and_A"
    assert report["primary"]["raw_rmse"] == pytest.approx(math.sqrt((.04 + .04 + 4) / 3))
    assert report["primary"]["normalized_rmse"] == pytest.approx(math.sqrt((.04 + .04 + .25) / 3))


def test_diagnostic_aliases_do_not_relabel_other_roles_or_mutate_legacy_keys():
    metrics = {"role_A_poi_rmse": .1, "role_B_poi_rmse": .2, "role_C_poi_accuracy": .7,
               "role_D_poi_rmse": .3}
    before = deepcopy(metrics)
    assert diagnostic_metric_aliases(metrics) == {
        "diagnostics/progress/B_oracle_focus_relative_rmse": .2,
        "diagnostics/progress/D_predicted_focus_relative_rmse": .3,
    }
    assert metrics == before


def test_compact_projection_keeps_quality_and_worker_metrics_without_alias_families():
    metrics = {"accumulated/overall/raw_rmse": .4, "accumulated/overall/normalized_rmse": .04,
               "accumulated/group/group/with/slashes/raw_rmse": .5,
               "accumulated/group/group/with/slashes/normalized_rmse": .05,
               "accumulated/overall/support": 17, "accumulated/overall/trajectory_support": 3,
               "role_C_poi_accuracy": .8, "execution_world_size": 3, "elapsed_seconds": 5.,
               "evaluation_and_persistence_seconds": 7., "windows_per_second": 3.4,
               "eval_quality/first/oracle_stage_route_accumulated_progress_raw_rmse": .2,
               "eval_quality/second/oracle_stage_route_accumulated_progress_raw_rmse": .3}
    for rank in (0, 1, 2):
        metrics.update({f"runtime/worker_{rank}/cuda_memory/peak_reserved_bytes": 100 + rank,
                        f"runtime/worker_{rank}/cuda_memory/device_free_after_bytes": 200 + rank})
    expected = {key: value for key, value in metrics.items()
                if not key.endswith(("raw_rmse", "normalized_rmse"))}
    expected.update({"overall/predicted_stage_route_accumulated_progress_poi_micro_raw_rmse": .4,
                     "overall/predicted_stage_route_accumulated_progress_poi_micro_normalized_rmse": .04,
                     "group/group/with/slashes/predicted_stage_route_accumulated_progress_poi_micro_raw_rmse": .5,
                     "group/group/with/slashes/predicted_stage_route_accumulated_progress_poi_micro_normalized_rmse": .05,
                     "task/first/oracle_stage_route_accumulated_progress_poi_micro_raw_rmse": .2,
                     "task/second/oracle_stage_route_accumulated_progress_poi_micro_raw_rmse": .3})
    metrics.update({"role_B_poi_rmse": .2, "role_D_poi_rmse": .3,
                    "overlay_acc/duplicate": .4, "diagnostics/overall/D/incorrect_C/rmse": .9,
                    "runtime/worker_0/cuda_memory/peak_allocated_bytes": 50,
                    "runtime/timing_seconds/forward_cuda": 3.})
    before = deepcopy(metrics)
    assert compact_evaluation_metrics(metrics) == expected
    assert metrics == before


def test_compact_multitask_and_a_only_routes_preserve_measured_pairs():
    metrics = {"accumulated/overall/raw_rmse": .4, "accumulated/overall/normalized_rmse": .4,
               "role_A_poi_rmse": .4}
    assert compact_evaluation_metrics(metrics) == {
        "overall/compressed_whole_task_progress_poi_micro_raw_rmse": .4,
        "overall/compressed_whole_task_progress_poi_micro_normalized_rmse": .4}
    for task in ("a", "b"):
        metrics[f"accumulated/task/{task}/raw_rmse"] = .2
        metrics[f"accumulated/task/{task}/normalized_rmse"] = .02
    projected = compact_evaluation_metrics(metrics)
    assert len(projected) == 6 and "role_A_poi_rmse" not in projected
    assert not any("oracle" in key or "role_C" in key for key in projected)
    assert all("compressed_whole_task_progress" in key for key in projected)


def _series(task, group, *, staged, errors):
    points = [{"prediction": error, "target": 0} for error in errors]
    if staged:
        for point in points:
            point.update(predicted_stage=int(point["prediction"]), target_stage=0, oracle=.1)
    return {"task": task, "group": group, "route": "predicted_stage_route" if staged else "compressed_whole_task",
            "scale": 4 if staged else 1, "points": points}


def _records(series, *, include_a_with_staged=False):
    return [{"cache": "cache", "task": row["task"], "trajectory_id": str(i), "poi": poi,
             "group": row["group"], "role": role}
            for i, row in enumerate(series) for poi, _ in enumerate(row["points"])
            for role in (("A", "B", "C", "D") if include_a_with_staged else ("B", "C", "D"))
            if row["route"] == "predicted_stage_route"] + [
        {"cache": "cache", "task": row["task"], "trajectory_id": str(i), "poi": poi,
         "group": row["group"], "role": "A"}
        for i, row in enumerate(series) for poi, _ in enumerate(row["points"])
        if row["route"] == "compressed_whole_task"]


def test_pretrain_macro_does_not_become_poi_weighted_or_a_staged_metric():
    series = [_series("frequent", "heldout", staged=False, errors=[.2, .2, .2]),
              _series("rare", "heldout", staged=False, errors=[.8])]
    metrics = {**dense_metrics(series), "role_A_task_macro_poi_rmse": .5}
    projected = compact_evaluation_metrics(metrics, evaluation={"records": _records(series)})
    assert projected["overall/compressed_whole_task_progress_task_macro_raw_rmse"] == .5
    assert projected["overall/compressed_whole_task_progress_poi_micro_raw_rmse"] == pytest.approx(math.sqrt(.19))
    legacy = compact_evaluation_metrics(metrics, evaluation={"records": _records(series)},
                                        schema="grana-compact-eval-v2")
    assert legacy["overall/compressed_whole_task_progress_task_macro_rmse"] == .5
    assert legacy["overall/compressed_whole_task_progress_raw_rmse"] == pytest.approx(math.sqrt(.19))
    assert not any("poi_micro" in key for key in legacy)
    assert "overall/compressed_whole_task_progress_raw_rmse" not in projected
    assert "overall/compressed_whole_task_progress_task_macro_rmse" not in projected
    # An A diagnostic alongside C->D is not an overall deployment macro.
    staged = [_series("staged", "heldout", staged=True, errors=[1])]
    projected = compact_evaluation_metrics(
        {**dense_metrics(staged), "role_A_task_macro_poi_rmse": .5},
        evaluation={"records": _records(staged, include_a_with_staged=True)})
    assert not any("task_macro" in key for key in projected)


def test_explicit_prediction_name_is_exact_measured_metric_without_duplicate_aliases():
    series = [_series("Assembly/Task", "group/one", staged=True, errors=[.1, 2, .4])]
    metrics = dense_metrics(series)
    before = deepcopy(metrics)
    projected = compact_evaluation_metrics(metrics, evaluation={"records": _records(series)})
    assert projected["overall/predicted_stage_route_accumulated_progress_poi_micro_raw_rmse"] == metrics[
        "eval_quality/assembly-task/predicted_stage_route_accumulated_progress_raw_rmse"]
    assert projected["overall/predicted_stage_route_accumulated_progress_poi_micro_normalized_rmse"] == metrics[
        "accumulated/overall/normalized_rmse"]
    assert projected["task/Assembly/Task/oracle_stage_route_accumulated_progress_poi_micro_raw_rmse"] == .1
    assert not any(key.startswith(("eval_quality/", "overlay_acc/")) or key.endswith("/raw_rmse")
                   or key.endswith("/normalized_rmse") for key in projected)
    # A single task's duplicate predicted pair is omitted; the distinct oracle remains.
    assert not any(key.startswith("task/") and "predicted_stage_route" in key for key in projected)
    assert metrics == before


def test_mixed_routes_are_classified_per_scope_and_pool_errors_not_task_rmse_means():
    series = [_series("single", "shared", staged=False, errors=[.2, .2]),
              _series("staged/one", "shared", staged=True, errors=[2]),
              _series("staged/two", "staged_only", staged=True, errors=[1, 1])]
    metrics = dense_metrics(series)
    evaluation = {"records": _records(series)}
    routes = accumulated_scope_routes(metrics, evaluation=evaluation)
    assert routes == {"overall": "mixed_C_to_D_and_A", "task/single": "whole_task_A",
                      "task/staged/one": "predicted_stage_C_to_D", "task/staged/two": "predicted_stage_C_to_D",
                      "group/shared": "mixed_C_to_D_and_A", "group/staged_only": "predicted_stage_C_to_D"}
    projected = compact_evaluation_metrics(metrics, evaluation=evaluation)
    assert projected["overall/mixed_route_accumulated_progress_poi_micro_raw_rmse"] == pytest.approx(math.sqrt(6.08 / 5))
    assert projected["overall/mixed_route_accumulated_progress_poi_micro_normalized_rmse"] == pytest.approx(math.sqrt(.455 / 5))
    assert projected["task/single/compressed_whole_task_progress_poi_micro_raw_rmse"] == pytest.approx(.2)
    assert projected["task/staged/one/predicted_stage_route_accumulated_progress_poi_micro_raw_rmse"] == 2
    assert projected["group/shared/mixed_route_accumulated_progress_poi_micro_raw_rmse"] == pytest.approx(math.sqrt(4.08 / 3))
    assert projected["group/staged_only/predicted_stage_route_accumulated_progress_poi_micro_raw_rmse"] == 1
    assert not any(key.startswith("overall/predicted_stage") or key.startswith("group/shared/predicted_stage")
                   or key.startswith("task/single/predicted_stage") for key in projected)


def test_a_scored_alongside_c_and_d_does_not_change_deployment_route():
    series = [_series("task", "group", staged=True, errors=[1])]
    metrics = {**dense_metrics(series), "role_A_poi_rmse": .1, "role_D_poi_rmse": .2}
    projected = compact_evaluation_metrics(metrics, evaluation={"records": _records(series, include_a_with_staged=True)})
    assert projected["overall/predicted_stage_route_accumulated_progress_poi_micro_raw_rmse"] == 1
    assert not any("mixed_route" in key or "compressed_whole_task" in key for key in projected)


def test_metrics_only_mixed_group_never_claims_predicted_stage_route():
    series = [_series("single", "a_group", staged=False, errors=[.2]),
              _series("staged", "c_group", staged=True, errors=[2])]
    projected = compact_evaluation_metrics(dense_metrics(series))
    assert projected["overall/mixed_route_accumulated_progress_poi_micro_raw_rmse"] == pytest.approx(math.sqrt(2.02))
    assert projected["task/single/compressed_whole_task_progress_poi_micro_raw_rmse"] == .2
    assert projected["task/staged/predicted_stage_route_accumulated_progress_poi_micro_raw_rmse"] == 2
    # Without records there is no proof which task belongs to each group.
    assert projected["group/a_group/unclassified_route_accumulated_progress_poi_micro_raw_rmse"] == .2
    assert projected["group/c_group/unclassified_route_accumulated_progress_poi_micro_raw_rmse"] == 2


def test_missing_route_evidence_is_explicit_and_does_not_drop_measured_quality():
    projected = compact_evaluation_metrics({"accumulated/overall/raw_rmse": .4,
                                            "accumulated/overall/normalized_rmse": .04})
    assert projected == {"overall/unclassified_route_accumulated_progress_poi_micro_raw_rmse": .4,
                         "overall/unclassified_route_accumulated_progress_poi_micro_normalized_rmse": .04}


def test_compact_sampled_coverage_deduplicates_roles_without_inventing_accumulated_error():
    records = [{"cache": "cache", "task": "task", "trajectory_id": trajectory, "poi": poi, "role": role}
               for trajectory, poi in (("first", 1), ("first", 2), ("second", 1)) for role in ("B", "C", "D")]
    result = compact_evaluation_metrics({"role_D_poi_rmse": .2}, evaluation={"records": records})
    assert result == {"coverage/windows": 3, "coverage/trajectories": 2}


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True])
def test_compact_projection_rejects_invalid_selected_scalars(value):
    with pytest.raises(ValueError, match="finite"):
        compact_evaluation_metrics({"elapsed_seconds": value})
