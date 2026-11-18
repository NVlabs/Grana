# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Independent formula checks for the Grana target construction."""
from copy import deepcopy

import pytest

from grana.data.targets.composition import PROGRESS_METHOD, compute_target_composition, validate_stage_parameters


PARAMS = {"rho": 5.0, "p_cap": .95, "t_min_seconds": 9.0}


def inputs(quality, outcome="high", *, timestamps=None, parameters=None):
    count = len(quality)
    return dict(labels=["1"] * count, scores=[""] * (count - 1) + [outcome],
                action_quality=quality, stage_prompts=["Open drawer"], label_map={"1": "Open drawer"},
                source_fps=30, timestamps=timestamps if timestamps is not None else [i / 30 for i in range(count)],
                stage_parameters=[parameters if parameters is not None else PARAMS])


def test_published_1117_frame_javascript_oracle():
    # Direct transcription of the website's targets: frame/fps for the first
    # attempt; count since recovery for the second. No production helper used.
    count = 1117
    data = inputs([0] * 250 + [2] * 571 + [0] * 296)
    data["stage_prompts"] = [f"Stage {i}" for i in range(9)]
    data["label_map"] = {"1": "Stage 6"}
    data["stage_parameters"] = [PARAMS.copy() for _ in range(9)]
    row = compute_target_composition(**data)
    endpoint = .95 * ((250 / 30) / 9)
    oracle = [endpoint * (frame / 30) / (249 / 30) if frame < 250 else 0.0 if frame <= 820
              else (frame - 820) / 296 for frame in range(count)]
    assert max(abs(actual - expected) for actual, expected in zip(row["within_stage_progress"], oracle)) < 1e-14
    assert max(abs(actual - (6 + expected) / 9) for actual, expected in zip(row["progress"], oracle)) < 1e-14
    assert row["within_stage_progress"][0] == 0
    assert row["within_stage_progress"][249] == pytest.approx(95 / 108)
    assert row["within_stage_progress"][250:821] == [0] * 571
    assert row["within_stage_progress"][821] == pytest.approx(1 / 296)
    assert row["within_stage_progress"][-1] == 1
    assert row["stage_ids"] == [6] * count and row["method"] == PROGRESS_METHOD
    assert row["chunks"][0]["case"] == "A_interrupted"
    assert row["chunks"][0]["duration_seconds"] == pytest.approx(250 / 30)
    assert row["chunks"][1]["anchor_time"] == pytest.approx(820 / 30)
    assert row["chunks"][1]["duration_seconds"] == pytest.approx(296 / 30)


def test_nonuniform_timestamps_and_quality_changes_share_one_chunk():
    # Weighted increments: 5*1, 1*2, 5*3 => 5,2,15. Endpoint is the explicit .5.
    row = compute_target_composition(**inputs([0, 0, 1, 0], "mid", timestamps=[10, 11, 13, 16]))
    assert row["within_stage_progress"] == pytest.approx([0, .5 * 5 / 22, .5 * 7 / 22, .5])
    assert len(row["chunks"]) == 1 and row["chunks"][0]["weighted_duration"] == 22


def test_interrupted_endpoint_uses_reset_duration_then_saturates():
    params = {"rho": 2, "p_cap": .8, "t_min_seconds": 4}
    short = compute_target_composition(**inputs([0, 0, 2], timestamps=[0, 1, 2], parameters=params))
    long = compute_target_composition(**inputs([0, 0, 2], timestamps=[0, 5, 6], parameters=params))
    assert short["within_stage_progress"] == [0, .4, 0]
    assert long["within_stage_progress"] == [0, .8, 0]
    # The outcome cannot override a terminal regressed observation.
    assert short["stage_partial_successes"] == [1]


def test_regression_remains_zero_even_with_mid_outcome():
    row = compute_target_composition(**inputs([0, 0, 2, 2], "mid"))
    assert row["within_stage_progress"][-2:] == [0, 0]


def test_multiple_attempts_reset_independently_and_ignore_failure_duration():
    args = inputs([0, 0, 2, 2, 0, 1, 2, 0, 0], timestamps=[0, 1, 2, 20, 21, 22, 23, 24, 25],
                  parameters={"rho": 3, "p_cap": .8, "t_min_seconds": 4})
    row = compute_target_composition(**args)
    # First reset is2s, second3s from last regressed20, final2s from regressed23.
    assert [chunk["endpoint"] for chunk in row["chunks"]] == pytest.approx([.4, .6, 1])
    assert row["within_stage_progress"] == pytest.approx([0, .4, 0, 0, .45, .6, 0, .5, 1])
    assert [chunk["anchor_time"] for chunk in row["chunks"]] == [0, 20, 23]


def test_low_boundary_outcome_does_not_erase_earlier_attempt_credit():
    row = compute_target_composition(**inputs([0, 0, 2, 0, 0], "low"))
    assert row["within_stage_progress"][1] > 0
    assert row["within_stage_progress"][2:] == [0, 0, 0]


def test_initial_regression_and_single_recovery_frame():
    row = compute_target_composition(**inputs([2, 2, 0]))
    assert row["within_stage_progress"] == [0, 0, 1]
    assert row["chunks"][0]["anchor_time"] == 1 / 30
    all_failed = compute_target_composition(**inputs([2, 2], "high"))
    assert all_failed["within_stage_progress"] == [0, 0] and all_failed["chunks"] == []


def test_stage_boundaries_reset_with_no_short_run_merge_or_tail_absorption():
    args = inputs([0] * 8)
    args.update(labels=["leading", "1", "1", "2", "2", "1", "1", "trailing"],
                scores=["", "", "high", "", "mid", "", "low", ""],
                stage_prompts=["Open drawer", "Close drawer"],
                label_map={"1": "Open drawer", "2": "Close drawer"}, stage_parameters=[PARAMS, PARAMS])
    row = compute_target_composition(**args)
    assert row["source_indices"] == [1, 2, 3, 4, 5, 6]
    assert row["stage_ids"] == [0, 0, 1, 1, 0, 0]
    assert row["within_stage_progress"] == [0, 1, 0, .5, 0, 0]
    assert row["progress"] == [0, .5, .5, .75, 0, 0]
    assert len(row["stage_runs"]) == 3 and row["trailing_absorbed_n_frames"] == 0


def test_stage_specific_parameters_and_translation_invariance():
    args = inputs([0, 0, 2, 0, 0, 2])
    args.update(labels=["1"] * 3 + ["2"] * 3, scores=["", "", "high", "", "", "high"],
                stage_prompts=["Open drawer", "Close drawer"],
                label_map={"1": "Open drawer", "2": "Close drawer"},
                stage_parameters=[{"rho": 2, "p_cap": .5, "t_min_seconds": 1},
                                  {"rho": 3, "p_cap": .9, "t_min_seconds": 2}])
    row = compute_target_composition(**args)
    assert row["chunks"][0]["endpoint"] == pytest.approx(.5 * (2 / 30))
    assert row["chunks"][1]["endpoint"] == pytest.approx(.9 * (2 / 30) / 2)
    shifted = compute_target_composition(**(args | {"timestamps": [time + 100 for time in args["timestamps"]]}))
    assert shifted["within_stage_progress"] == pytest.approx(row["within_stage_progress"], abs=1e-12)


@pytest.mark.parametrize("quality,outcome", [([0], "high"), ([0, 2], "high")])
def test_positive_endpoint_without_observed_update_duration_is_rejected(quality, outcome):
    with pytest.raises(ValueError, match="zero observed"):
        compute_target_composition(**inputs(quality, outcome))
    assert compute_target_composition(**inputs([0], "low"))["within_stage_progress"] == [0]


@pytest.mark.parametrize("changes,match", [
    ({"scores": ["", "", ""]}, "missing explicit"),
    ({"scores": ["high", "", "low"]}, "conflicting explicit"),
    ({"scores": ["", "", "unknown"]}, "scores"),
    ({"labels": ["1", "empty", "1"]}, "gaps"),
    ({"labels": ["1", "unknown", "1"]}, "Unknown raw"),
    ({"timestamps": [0, 0, 1]}, "strictly increasing"),
    ({"timestamps": [0, float("nan"), 1]}, "finite"),
    ({"timestamps": [0, float("inf"), 1]}, "finite"),
    ({"timestamps": [0, 1]}, "equal"),
    ({"action_quality": [0, True, 0]}, "action_quality"),
    ({"action_quality": [0, 3, 0]}, "action_quality"),
    ({"source_fps": 0}, "positive"),
    ({"source_fps": True}, "finite"),
    ({"label_map": {"1": "Unknown stage"}}, "label_map"),
    ({"label_map": {"trailing": "Open drawer"}}, "cannot be mapped"),
])
def test_invalid_or_incomplete_annotations_are_rejected(changes, match):
    with pytest.raises(ValueError, match=match):
        compute_target_composition(**(inputs([0, 0, 0]) | changes))


@pytest.mark.parametrize("parameters", [None, [], [PARAMS, PARAMS], [{}],
    [PARAMS | {"rho": 1}], [PARAMS | {"rho": True}], [PARAMS | {"rho": float("inf")}],
    [PARAMS | {"p_cap": 0}], [PARAMS | {"p_cap": 1}], [PARAMS | {"t_min_seconds": 0}],
    [PARAMS | {"t_min_seconds": float("nan")}], [PARAMS | {"unknown": 1}]])
def test_parameters_are_explicit_and_strict(parameters):
    with pytest.raises(ValueError):
        validate_stage_parameters(parameters, 1)


def test_input_objects_are_not_mutated_and_repeated_same_score_is_unambiguous():
    args = inputs([0, 1, 0])
    args["scores"] = ["high", "empty", "high"]
    original = deepcopy(args)
    row = compute_target_composition(**args)
    assert args == original and row["within_stage_progress"] == pytest.approx([0, 1 / 6, 1])


def test_finite_inputs_cannot_overflow_into_nan_targets():
    with pytest.raises(ValueError, match="finite numeric range"):
        compute_target_composition(**inputs([0, 0], timestamps=[0, 100],
                                            parameters=PARAMS | {"rho": 1e308}))
