# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from grana.data.targets.legacy import PROGRESS_METHOD, compute_progress


def targets(quality, scores=None, **options):
    return compute_progress(labels=["1"] * len(quality),
                            scores=scores if scores is not None else [""] * (len(quality) - 1) + ["high"],
                            action_quality=quality, stage_prompts=["Open"], label_map={"1": "Open"},
                            source_fps=30, **options)


def test_weighted_stage_and_sparse_authoritative_score():
    row = targets([0, 1, 0], ["low", "empty", "mid"])
    assert row["method"] == PROGRESS_METHOD
    assert row["source_indices"] == [0, 1, 2]
    assert row["within_stage_progress"] == pytest.approx([2.5 / 10.1, 2.55 / 10.1, .5])
    assert row["stage_partial_successes"] == [.5]


def test_failure_resets_and_terminal_recovery_owns_endpoint():
    row = targets([0, 0, 2, 0, 1], progress_cap_seconds=1 / 30)
    values = row["within_stage_progress"]
    assert values[:3] == pytest.approx([.45, .9, 0])
    assert values[3:] == pytest.approx([5 / 5.1, 1])
    assert row["stage_quality_labels"] == ["suboptimal"]


def test_low_score_cannot_recover_after_failure():
    row = targets([0, 2, 0, 1], ["", "", "", "low"])
    assert row["within_stage_progress"][0] > 0
    assert row["within_stage_progress"][1:] == [0, 0, 0]


def test_only_terminal_failure_block_holds_mid_target():
    row = targets([0, 2, 0, 2, 2], ["", "", "", "", "mid"])
    assert row["within_stage_progress"][1] == 0
    assert row["within_stage_progress"][-2:] == [.5, .5]


@pytest.mark.parametrize("quality,expected", [([0, 1, 0], 1), ([0, 2, 0], .5), ([2, 0, 0], 0), ([0, 0, 2], 0)])
def test_missing_score_uses_legacy_quality_outcome(quality, expected):
    row = targets(quality, ["empty"] * len(quality))
    assert row["stage_partial_successes"] == [expected]
    assert row["within_stage_progress"][-1] == pytest.approx(expected)


def test_stage_regression_unmatched_projection_and_capped_trailing():
    labels = ["prefix", "2", "2", "gap", "1", "1", "trailing", "trailing", "trailing"]
    row = compute_progress(labels=labels, scores=["high"] * len(labels), action_quality=[0] * len(labels),
                           stage_prompts=["Open", "Close"], label_map={"1": "Open", "2": "Close"},
                           source_fps=10, min_segment_frames=0, max_trailing_seconds=.2)
    assert row["source_indices"] == [1, 2, 4, 5, 6, 7]
    assert row["stage_ids"] == [1, 1, 0, 0, 0, 0]
    assert row["within_stage_progress"] == [.5, 1, .5, 1, 1, 1]
    assert row["progress"] == [.75, 1, .25, .5, .5, .5]
    assert row["trailing_absorbed_n_frames"] == 2


def test_short_later_runs_merge_before_scoring():
    row = compute_progress(labels=["1"] * 10 + ["2"] * 2, scores=[""] * 9 + ["high", "", "low"],
                           action_quality=[0] * 12, stage_prompts=["Open", "Close"],
                           label_map={"1": "Open", "2": "Close"}, source_fps=30)
    # The short label-2 run belongs to label-1, including its authoritative low score.
    assert row["stage_ids"] == [0] * 12
    assert row["within_stage_progress"] == [0] * 12


def test_first_short_run_is_not_merged_and_suffix_match_is_case_insensitive():
    row = compute_progress(labels=["prefix OPEN", "close", "close"], scores=["high"] * 3,
                           action_quality=[0] * 3, stage_prompts=["Open", "Close"],
                           source_fps=30, min_segment_frames=2, max_trailing_seconds=0)
    assert row["stage_ids"] == [0, 1, 1]
    assert row["progress"] == [.5, .75, 1]


@pytest.mark.parametrize("changes,match", [
    ({"action_quality": [3]}, "action_quality"),
    ({"action_quality": [True]}, "action_quality"),
    ({"scores": ["unknown"]}, "scores"),
    ({"scores": []}, "equal"),
    ({"labels": []}, "equal"),
    ({"labels": ["unmapped"]}, "No annotation"),
    ({"source_fps": 0}, "source_fps"),
    ({"progress_cap_seconds": float("nan")}, "progress_cap_seconds"),
    ({"progress_cap_seconds": .001}, "one source frame"),
    ({"max_trailing_seconds": -1}, "max_trailing_seconds"),
    ({"min_segment_frames": True}, "min_segment_frames"),
    ({"label_map": {"*": "Open"}}, "wildcard"),
])
def test_invalid_policy_inputs_are_rejected(changes, match):
    args = dict(labels=["Open"], scores=["high"], action_quality=[0], stage_prompts=["Open"], source_fps=30)
    with pytest.raises(ValueError, match=match):
        compute_progress(**(args | changes))


@pytest.mark.parametrize("prompts", [["Open", "Then Open"], ["Open", "OPEN"]])
def test_stage_prompts_cannot_silently_alias_a_different_stage(prompts):
    with pytest.raises(ValueError, match="ambiguous"):
        compute_progress(labels=["2"], scores=["high"], action_quality=[0],
                         stage_prompts=prompts, label_map={"2": prompts[1]}, source_fps=30)
