# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit synthetic review windows; no private examples or network access."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("evaluate_annotation_events", ROOT / "scripts/evaluate_annotation_events.py")
scorer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(scorer)


def span(start, end, **kwargs):
    return {"start_frame": start, "end_frame": end, **kwargs}


def event(name, start, end, quality="failure", start_bracket=None, end_bracket=None, **kwargs):
    return span(start, end, id=name, quality=quality,
                start_boundary_bracket=start_bracket or [start, start],
                end_boundary_bracket=end_bracket or [end, end], **kwargs)


def document(predictions, *, n=12, windows=None, events=None, stage_windows=None, edges=None):
    return {
        "schema_version": scorer.INPUT_VERSION,
        "time_edges": edges or list(range(n + 1)),
        "candidate": {
            "stages": {"segments": [span(0, n, stage_id="S1", identity_status="resolved")]},
            "aq_intervals": [span(start, end, quality=quality) for start, end, quality in predictions],
        },
        "reference": {"status": "independent_agent_candidate", "policy_version": "synthetic-draft-v1",
                      "aq_windows": windows or [], "stage_windows": stage_windows or [], "events": events or []},
    }


def test_normal_false_alarm_and_unknown_are_separate_and_gaps_not_normal():
    source = document([(0, 2, "failure"), (2, 4, "suboptimal"), (4, 6, "unknown"), (6, 12, "nominal")],
                      windows=[span(1, 6, id="reviewed-normal", quality="nominal", role="normal")])
    before = deepcopy(source)
    report = scorer.evaluate_events(source)
    assert source == before
    assert report["reviewed_normal"]["seconds"] == 5
    assert report["reviewed_normal"]["predicted_seconds"] == {
        "failure": 1, "suboptimal": 2, "unknown": 2, "nominal": 0}
    assert report["reviewed_normal"]["false_alarm_seconds"] == 3
    assert report["reviewed_normal"]["unknown_seconds"] == 2
    assert report["aq"]["unreviewed_seconds"] == 7
    assert report["accepted_for_training"] is False
    assert report["reference_status_verified"] is False


def test_one_long_prediction_cannot_detect_two_events_or_gain_duration_weight():
    source = document([(0, 12, "failure")], events=[event("short", 1, 2), event("longer", 5, 8)])
    report = scorer.evaluate_events(source)
    assert report["events"]["by_class"]["failure"]["detection_fraction"] == .5
    assert report["events"]["matches"][0]["event_id"] == "longer"
    assert report["events"]["missed_reference_event_ids"] == ["short"]
    assert "precision" not in report["events"]  # Partial reference does not define all negatives.


def test_matching_maximizes_count_before_iou_and_reassigns_greedy_choice():
    # Event A's uncertain envelope can match either prediction, preferring P1;
    # event B can only match P1. A greedy IoU choice would miss B.
    source = document([(0, 2, "failure"), (2, 3, "nominal"), (3, 4, "failure"), (4, 12, "nominal")],
                      events=[event("A", 2, 3, start_bracket=[0, 3], end_bracket=[2, 4]),
                              event("B", 0, 1)])
    report = scorer.evaluate_events(source)
    assert report["events"]["by_class"]["failure"]["matched_events"] == 2
    matches = {row["event_id"]: row for row in report["events"]["matches"]}
    assert matches["A"]["predicted_span"] == [3, 4]
    assert matches["A"]["matching_used_uncertain_envelope"] is True
    assert matches["B"]["predicted_span"] == [0, 2]


def test_event_matching_uses_guaranteed_core_and_requires_correct_class():
    source = document([(0, 3, "failure"), (3, 5, "suboptimal"), (5, 12, "unknown")],
                      events=[event("core", 2, 6, start_bracket=[1, 3], end_bracket=[5, 7])])
    report = scorer.evaluate_events(source)
    assert report["events"]["matches"] == []  # Same-class run touches but does not overlap [3,5).
    assert report["events"]["missed_reference_event_ids"] == ["core"]
    assert len(report["events"]["unmatched_predictions"]) == 2


def test_unknown_is_not_a_positive_event_prediction():
    report = scorer.evaluate_events(document([(0, 12, "unknown")], events=[event("e", 3, 5)]))
    assert report["events"]["prediction_count_after_merging_same_class_adjacency"] == 0
    assert report["events"]["by_class"]["failure"]["missed_events"] == 1


def test_adjacent_same_quality_is_one_run_not_extra_event_credit():
    source = document([(0, 4, "failure"), (4, 8, "failure"), (8, 12, "nominal")],
                      events=[event("a", 1, 2), event("b", 5, 6)])
    report = scorer.evaluate_events(source)
    assert report["events"]["prediction_count_after_merging_same_class_adjacency"] == 1
    assert len(report["events"]["matches"]) == 1


def test_boundary_distance_uses_reference_brackets_and_native_nonuniform_clock():
    edges = [100, 100.2, 100.5, 101, 102, 104, 107, 111, 116, 122, 129, 137, 146]
    source = document([(0, 1, "nominal"), (1, 9, "failure"), (9, 12, "nominal")], edges=edges,
                      events=[event("drop", 3, 7, start_bracket=[2, 4], end_bracket=[6, 8])])
    source["candidate"]["aq_boundaries"] = [
        {"frame": 1, "lower_frame": 0, "upper_frame": 3},
        {"frame": 9, "lower_frame": 8, "upper_frame": 10}]
    report = scorer.evaluate_events(source)
    match = report["events"]["matches"][0]
    assert match["start_boundary"]["signed_distance_to_reference_bracket_seconds"] == pytest.approx(-.3)
    assert match["end_boundary"]["signed_distance_to_reference_bracket_seconds"] == 6
    # Broad prediction brackets overlap the reference, but do not erase nominal-cut error.
    assert match["start_boundary"]["bracket_distance_seconds"] == 0
    assert match["end_boundary"]["bracket_distance_seconds"] == 0
    assert report["source_seconds"] == 46
    assert report["native_start_time"] == 100


def test_recovery_continuation_vs_new_false_alarm_and_unknown():
    source = document([(0, 2, "nominal"), (2, 7, "failure"), (7, 8, "nominal"),
                       (8, 9, "suboptimal"), (9, 10, "unknown"), (10, 12, "nominal")],
                      windows=[span(5, 10, id="recovery", quality="nominal", role="recovery")],
                      events=[event("drop", 2, 5, recovery_window_id="recovery")])
    report = scorer.evaluate_events(source)
    assert report["reviewed_recovery"]["false_alarm_seconds"] == 3
    assert report["reviewed_recovery"]["matched_event_continuation_seconds"] == 2
    assert report["reviewed_recovery"]["unknown_seconds"] == 1
    assert report["events"]["matches"][0]["linked_recovery_continuation_seconds"] == 2


def test_stage_agreement_only_explicit_windows_and_uncertain_is_not_resolved():
    source = document([(0, 12, "nominal")], stage_windows=[span(2, 10, id="stage", stage_id="S1")])
    source["candidate"]["stages"]["segments"] = [
        span(0, 4, stage_id="S1", identity_status="resolved"),
        span(4, 6, stage_id="S1", identity_status="uncertain"),
        span(6, 12, stage_id="S2", identity_status="resolved")]
    report = scorer.evaluate_events(source)
    assert report["stage"]["reviewed_seconds"] == 8
    assert report["stage"]["resolved_agreement"] == .25
    assert report["stage"]["unresolved_or_uncertain_seconds"] == 2
    assert report["aq"]["known_reference_agreement"] is None


def test_reference_unknown_is_excluded_not_changed_to_nominal():
    source = document([(0, 12, "failure")], windows=[
        span(0, 5, id="ambiguous", quality="unknown", role="other"),
        span(5, 7, id="event", quality="failure", role="event")])
    report = scorer.evaluate_events(source)
    assert report["aq"]["unknown_reference_seconds"] == 5
    assert report["aq"]["known_reference_seconds"] == 2
    assert report["aq"]["known_reference_agreement"] == 1
    assert report["reviewed_normal"]["seconds"] == 0
    assert report["reviewed_normal"]["false_alarm_fraction"] is None


@pytest.mark.parametrize("edit,error", [
    (lambda x: x["time_edges"].__setitem__(2, 1), "strictly increasing"),
    (lambda x: x["time_edges"].__setitem__(2, float("nan")), "finite"),
    (lambda x: x["time_edges"].__setitem__(2, True), "numbers"),
    (lambda x: x["candidate"]["aq_intervals"][0].update(start_frame=1), "partition"),
    (lambda x: x["candidate"]["stages"]["segments"][0].update(end_frame=11), "partition"),
    (lambda x: x["candidate"]["aq_intervals"][0].update(quality="regressed"), "no label mapping"),
    (lambda x: x["reference"].update(status="verified_ground_truth"), "reference.status"),
    (lambda x: x["reference"]["events"].append(event("one", 8, 10)), "IDs must be unique"),
    (lambda x: x["reference"]["events"][0].update(start_boundary_bracket=[4, 5]), "contain"),
    (lambda x: x["reference"]["events"][0].update(quality="unknown"), "positive event"),
    (lambda x: x["reference"]["events"][0].update(recovery_window_id="missing"), "recovery_window_id"),
])
def test_invalid_clock_partition_reference_and_events_fail_closed(edit, error):
    source = document([(0, 12, "nominal")], events=[event("one", 2, 4)])
    edit(source)
    with pytest.raises(ValueError, match=error):
        scorer.evaluate_events(source)


def test_overlapping_review_windows_are_rejected_but_cross_channel_overlap_allowed():
    source = document([(0, 12, "nominal")],
                      windows=[span(1, 4, id="a", quality="nominal", role="normal"),
                               span(3, 5, id="b", quality="failure", role="event")])
    with pytest.raises(ValueError, match="AQ windows must not overlap"):
        scorer.evaluate_events(source)
    source["reference"]["aq_windows"].pop()
    source["reference"]["stage_windows"] = [span(1, 4, id="stage", stage_id="S1")]
    assert scorer.evaluate_events(source)["stage"]["resolved_agreement"] == 1


def test_candidate_boundary_gap_and_invalid_bracket_are_rejected():
    source = document([(0, 4, "failure"), (4, 12, "nominal")])
    source["candidate"]["aq_boundaries"] = []
    with pytest.raises(ValueError, match="every AQ partition cut"):
        scorer.evaluate_events(source)
    source["candidate"]["aq_boundaries"] = [{"frame": 4, "lower_frame": 5, "upper_frame": 8}]
    with pytest.raises(ValueError, match="contain"):
        scorer.evaluate_events(source)


def test_human_reviewed_remains_a_declared_status_not_automatic_acceptance():
    source = document([(0, 12, "nominal")])
    source["reference"]["status"] = "human_reviewed"
    report = scorer.evaluate_events(source)
    assert report["reference_status"] == "human_reviewed"
    assert report["reference_status_verified"] is False
    assert report["semantic_accuracy_verified"] is False
    assert report["accepted_for_training"] is False


def test_cli_synthetic_example_and_existing_output_is_not_overwritten(tmp_path):
    source = document([(0, 12, "nominal")], windows=[span(1, 4, id="a", quality="nominal", role="normal")])
    source_path, output_path = tmp_path / "input.json", tmp_path / "report.json"
    source_path.write_text(json.dumps(source))
    command = [sys.executable, str(ROOT / "scripts/evaluate_annotation_events.py"),
               "--input", str(source_path), "--output", str(output_path)]
    subprocess.run(command, check=True, capture_output=True, text=True)
    report = json.loads(output_path.read_text())
    assert report["reviewed_normal"]["seconds"] == 3
    before = output_path.read_bytes()
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 2
    assert "File exists" in result.stderr
    assert output_path.read_bytes() == before
