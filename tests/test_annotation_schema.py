# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy

import jsonschema
import pytest

from grana.annotation.schema import candidate_schema, coalesce_candidate, coverage, validate_candidate


@pytest.fixture
def candidate():
    return {
        "segments": [
            {"start_frame": 0, "end_frame": 2, "stage_id": "S2", "identity_status": "resolved", "reason": "Visible insertion", "evidence_frames": [0, 1]},
            {"start_frame": 2, "end_frame": 3, "stage_id": None, "identity_status": "unresolved", "reason": "Occlusion", "evidence_frames": [2]},
            {"start_frame": 3, "end_frame": 5, "stage_id": "S1", "identity_status": "uncertain", "reason": "Possible renewed approach", "evidence_frames": [3, 4]},
        ],
        "boundaries": [
            {"frame": 2, "lower_frame": 1, "upper_frame": 3, "reason": "Contact not visible", "evidence_frames": [1, 2, 3]},
            {"frame": 3, "lower_frame": 2, "upper_frame": 4, "reason": "Reappearance", "evidence_frames": [2, 3, 4]},
        ],
        "omitted_stages": [{"stage_id": "S3", "reason": "Not visible in this partial clip"}],
        "invalid_intervals": [{"start_frame": 2, "end_frame": 4, "reason": "Corrupted source view"}],
        "summary": "Partial clip with revisit and uncertainty; success unassessed.",
    }


def test_stage_only_schema_and_validation_preserve_revisit_and_uncertainty(candidate):
    original = deepcopy(candidate)
    jsonschema.Draft202012Validator.check_schema(candidate_schema(["S1", "S2", "S3"]))
    jsonschema.validate(candidate, candidate_schema(["S1", "S2", "S3"]))
    result = validate_candidate(candidate, ["S1", "S2", "S3"], 5)
    assert candidate == original == result and result is not candidate
    assert [s["stage_id"] for s in result["segments"]] == ["S2", None, "S1"]
    assert "action_quality" not in result and "outcome" not in result


def test_duration_uses_recorded_clock_and_unions_boundary_brackets(candidate):
    result = coverage(validate_candidate(candidate, ["S1", "S2", "S3"], 5), [10, 11, 13, 16, 20, 25])
    assert result["source_frames"] == 5 and result["source_seconds"] == 15
    assert result["unresolved_identity"]["frames"] == 1
    assert result["unresolved_identity"]["seconds"] == 3
    assert result["unresolved_identity"]["percent_frames"] == 20
    assert result["uncertain_identity"]["seconds"] == 9
    assert result["boundary_uncertainty"]["intervals"] == [[1, 4]]
    assert result["boundary_uncertainty"]["seconds"] == 9
    assert result["source_invalid"]["frames"] == 2
    assert result["source_invalid"]["seconds"] == 7
    # Invalid source remains a separate axis even when identity is known.
    assert result["source_invalid"]["percent_seconds"] == pytest.approx(100 * 7 / 15)


@pytest.mark.parametrize("edit,match", [
    (lambda x: x["segments"][1].update(start_frame=1), "partition"),
    (lambda x: x["segments"][1].update(start_frame=3), "partition"),
    (lambda x: x["segments"][0].update(end_frame=True), "integer"),
    (lambda x: x["segments"][0].update(stage_id=None), "null"),
    (lambda x: x["segments"][1].update(stage_id="S1"), "null"),
    (lambda x: x["segments"][0].update(stage_id="new"), "registry"),
    (lambda x: x["segments"][0].update(evidence_frames=[5]), "evidence"),
    (lambda x: x["segments"][0].update(evidence_frames=[]), "evidence"),
    (lambda x: x["segments"][0].update(outcome="high"), "exactly"),
    (lambda x: x["boundaries"].pop(), "cuts"),
    (lambda x: x["boundaries"][0].update(lower_frame=3), "bracket"),
    (lambda x: x["omitted_stages"].append({"stage_id": "S1", "reason": "invented omission"}), "absent"),
    (lambda x: x["omitted_stages"].clear(), "absent"),
    (lambda x: x["invalid_intervals"][0].update(end_frame=6), "within"),
    (lambda x: x.update(summary=" "), "nonempty"),
])
def test_malformed_candidates_are_never_repaired(candidate, edit, match):
    edit(candidate)
    with pytest.raises(ValueError, match=match):
        validate_candidate(candidate, ["S1", "S2", "S3"], 5)


def test_coalescing_is_explicit_and_keeps_different_identity_status(candidate):
    candidate["segments"][0]["stage_id"] = "S1"
    candidate["segments"][1].update(stage_id="S1", identity_status="resolved")
    candidate["omitted_stages"].append({"stage_id": "S2", "reason": "Not observed"})
    validated = validate_candidate(candidate, ["S1", "S2", "S3"], 5)
    assert len(validated["segments"]) == 3
    merged = coalesce_candidate(validated, ["S1", "S2", "S3"], 5)
    assert len(merged["segments"]) == 2
    assert merged["segments"][0]["end_frame"] == 3
    assert merged["segments"][0]["evidence_frames"] == [0, 1, 2]
    assert [b["frame"] for b in merged["boundaries"]] == [3]
    assert len(validated["segments"]) == 3


@pytest.mark.parametrize("times", [[0], [0, 0, 1], [0, float("nan")], [0, True]])
def test_bad_source_clock_is_not_turned_into_fps_duration(candidate, times):
    with pytest.raises(ValueError, match="time_edges"):
        coverage(candidate, times)
