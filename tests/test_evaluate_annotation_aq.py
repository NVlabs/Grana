# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicitly synthetic AQ references: no media, private labels or API calls."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/evaluate_annotation_aq.py"
SPEC = importlib.util.spec_from_file_location("evaluate_annotation_aq", SCRIPT)
scorer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(scorer)


def human(start, end, label):
    return {"start_frame": start, "end_frame": end, "label": label}


def prediction(start, end, quality):
    return {"start_frame": start, "end_frame": end, "quality": quality}


def document(edges, references, predictions):
    return {"schema_version": scorer.INPUT_VERSION, "time_edges": edges,
            "human_intervals": references, "predicted_intervals": predictions}


@pytest.fixture
def sample():
    return document([10, 10.5, 12, 15],
                    [human(0, 2, "failure"), human(2, 3, "nominal")],
                    [prediction(0, 1, "failure"), prediction(1, 2, "unknown"), prediction(2, 3, "suboptimal")])


def test_nonuniform_native_clock_weights_seconds_and_keeps_abstention(sample):
    before = deepcopy(sample)
    report = scorer.evaluate_aq(sample)
    assert sample == before
    assert report == scorer.evaluate_aq(sample)
    assert report["source_seconds"] == 5
    assert report["confusion_seconds"]["failure"] == {"nominal": 0, "suboptimal": 0, "failure": .5, "unknown": 1.5}
    assert report["confusion_seconds"]["nominal"]["suboptimal"] == 3
    assert report["per_class"]["failure"] == {
        "support_seconds": 2, "predicted_seconds": .5, "true_positive_seconds": .5,
        "false_positive_seconds": 0, "false_negative_seconds": 1.5,
        "precision": 1, "recall": .25, "iou": .25}
    assert report["accuracy"] == .1
    assert report["scoring_policy"]["accuracy_domain"] == (
        "Time-weighted agreement on human nominal/suboptimal/failure seconds only; "
        "not whole-recording accuracy or evidence of held-out generalization.")
    assert report["abstention"]["scored_fraction"] == .3
    # No human suboptimal support: no manufactured zero/one metric, but FP is visible.
    unsupported = report["per_class"]["suboptimal"]
    assert unsupported["false_positive_seconds"] == 3
    assert all(unsupported[name] is None for name in ("precision", "recall", "iou"))
    assert report["per_class"]["nominal"]["precision"] is None
    assert report["per_class"]["nominal"]["recall"] == 0
    assert report["boundary_evaluation"]["status"] == "not_implemented"


def test_seconds_intersections_do_not_round_human_endpoints_to_frames():
    data = document([100, 101, 103],
                    [{"start_time": 100.25, "end_time": 101.5, "label": "failure"}],
                    [prediction(0, 1, "nominal"), prediction(1, 2, "failure")])
    report = scorer.evaluate_aq(data)
    assert report["confusion_seconds"]["failure"]["nominal"] == .75
    assert report["confusion_seconds"]["failure"]["failure"] == .5
    assert report["human_coverage"]["seconds_by_label"]["blank"] == 1.75
    assert report["per_class"]["failure"]["recall"] == .4
    assert report["per_class"]["failure"]["predicted_seconds"] == .5  # Blank time is not a negative.


def test_human_overlap_is_conflict_even_for_same_class_or_unknown():
    data = document(list(range(8)),
                    [human(0, 3, "failure"), human(1, 2, "failure"), human(2, 4, "suboptimal"),
                     human(3, 5, "unknown"), human(5, 6, "unlabeled"), human(6, 7, "conflict")],
                    [prediction(0, 7, "failure")])
    report = scorer.evaluate_aq(data)
    assert report["human_coverage"]["seconds_by_label"] == {
        "nominal": 0, "suboptimal": 0, "failure": 1,
        "blank": 0, "unlabeled": 1, "unknown": 1, "conflict": 4}
    assert report["human_coverage"]["scored_seconds"] == 1
    assert report["human_coverage"]["excluded_seconds"] == 6
    assert report["confusion_seconds"]["failure"]["failure"] == 1
    assert report["per_class"]["failure"]["precision"] == 1
    assert scorer.evaluate_aq({**data, "human_intervals": list(reversed(data["human_intervals"]))})["confusion_seconds"] == report["confusion_seconds"]


def test_all_excluded_categories_and_gaps_remain_distinct():
    data = document(list(range(7)),
                    [human(0, 1, ""), human(1, 2, "blank"), human(2, 3, "unlabeled"),
                     human(3, 4, "unknown"), human(4, 5, "conflict")],
                    [prediction(0, 6, "unknown")])
    report = scorer.evaluate_aq(data)
    assert report["human_coverage"]["seconds_by_label"] == {
        "nominal": 0, "suboptimal": 0, "failure": 0,
        "blank": 3, "unlabeled": 1, "unknown": 1, "conflict": 1}
    assert report["accuracy"] is None
    assert report["abstention"] == {"all_source_seconds": 6, "all_source_fraction": 1,
                                    "scored_seconds": 0, "scored_fraction": None}
    assert all(row[key] is None for row in report["per_class"].values() for key in ("precision", "recall", "iou"))


def test_no_human_annotations_means_blank_not_nominal():
    report = scorer.evaluate_aq(document([0, 2], [], [prediction(0, 1, "nominal")]))
    assert report["human_coverage"]["seconds_by_label"]["blank"] == 2
    assert report["human_coverage"]["scored_seconds"] == 0
    assert report["per_class"]["nominal"]["predicted_seconds"] == 0


def test_all_unknown_predictions_are_full_misses_on_supported_classes():
    report = scorer.evaluate_aq(document([0, 2, 5], [human(0, 1, "failure"), human(1, 2, "nominal")],
                                        [prediction(0, 2, "unknown")]))
    assert report["accuracy"] == 0
    assert report["abstention"]["scored_fraction"] == 1
    assert report["per_class"]["failure"]["false_negative_seconds"] == 2
    assert report["per_class"]["failure"]["recall"] == 0
    assert report["per_class"]["nominal"]["false_negative_seconds"] == 3
    assert report["per_class"]["nominal"]["iou"] == 0


def test_adjacent_subdivision_does_not_change_metrics():
    base = document([0, .5, 2, 5], [human(0, 3, "failure")], [prediction(0, 3, "failure")])
    split = document(base["time_edges"], [human(0, 1, "failure"), human(1, 3, "failure")],
                     [prediction(0, 2, "failure"), prediction(2, 3, "failure")])
    first, second = scorer.evaluate_aq(base), scorer.evaluate_aq(split)
    assert first["confusion_seconds"] == second["confusion_seconds"]
    assert first["per_class"] == second["per_class"]
    assert sum(first["human_coverage"]["seconds_by_label"].values()) == first["source_seconds"]


def test_precision_recall_and_iou_have_distinct_denominators():
    data = document([0, 1, 3, 6], [human(0, 2, "failure"), human(2, 3, "nominal")],
                    [prediction(0, 1, "failure"), prediction(1, 2, "nominal"), prediction(2, 3, "failure")])
    metrics = scorer.evaluate_aq(data)["per_class"]["failure"]
    assert metrics["precision"] == .25
    assert metrics["recall"] == pytest.approx(1 / 3)
    assert metrics["iou"] == pytest.approx(1 / 6)


def test_native_integer_end_is_exclusive_and_adjacent_humans_do_not_conflict():
    data = document([0, 1, 2], [human(0, 1, "failure"), human(1, 2, "nominal")],
                    [prediction(0, 1, "failure"), prediction(1, 2, "nominal")])
    report = scorer.evaluate_aq(data)
    assert report["human_coverage"]["seconds_by_label"]["conflict"] == 0
    assert report["accuracy"] == 1


def test_raw_candidate_optional_reason_and_evidence_are_not_scored(sample):
    expected = scorer.evaluate_aq(sample)
    sample["predicted_intervals"][0].update(reason="Synthetic explanation only.", evidence_frames=[0])
    got = scorer.evaluate_aq(sample)
    assert got["confusion_seconds"] == expected["confusion_seconds"]
    assert got["input_sha256"] != expected["input_sha256"]
    assert "Synthetic explanation" not in json.dumps(got)


def test_nonfinite_optional_metadata_is_rejected(sample):
    sample["predicted_intervals"][0]["evidence_frames"] = [float("nan")]
    with pytest.raises(ValueError):
        scorer.evaluate_aq(sample)


@pytest.mark.parametrize("edges", [[], [0], [0, 0], [1, 0], [0, True], [0, "1"],
                                   [0, float("nan")], [0, float("inf")], [-1e308, 1e308],
                                   [0, 10 ** 1000], [2 ** 60, 2 ** 60 + 1]])
def test_invalid_clocks_are_rejected(sample, edges):
    sample["time_edges"] = edges
    with pytest.raises(ValueError):
        scorer.evaluate_aq(sample)


@pytest.mark.parametrize("rows", [[], [prediction(0, 2, "failure")], [prediction(1, 3, "failure")],
                                  [prediction(0, 2, "failure"), prediction(1, 3, "failure")],
                                  [prediction(0, 1, "failure"), prediction(2, 3, "failure")],
                                  [prediction(1, 3, "failure"), prediction(0, 1, "failure")],
                                  [prediction(0, 4, "failure")], [prediction(0, 0, "failure")],
                                  [prediction(False, 3, "failure")], [prediction(0, 3., "failure")],
                                  [prediction(0, 3, "regressed")], [prediction(0, 3, "unlabeled")],
                                  [prediction(0, 3, ["failure"])]])
def test_invalid_prediction_partitions_and_classes_fail(sample, rows):
    sample["predicted_intervals"] = rows
    with pytest.raises(ValueError):
        scorer.evaluate_aq(sample)


@pytest.mark.parametrize("rows", [[human(-1, 1, "failure")], [human(1, 1, "failure")],
                                  [human(0, 4, "failure")], [human(0, 1, "regressed")],
                                  [human(0, 1, None)], [human(0, 1, " failure ")],
                                  [{"start_time": 9, "end_time": 11, "label": "failure"}],
                                  [{"start_time": 10, "end_time": 16, "label": "failure"}],
                                  [{"start_time": 11, "end_time": 10, "label": "failure"}],
                                  [{"start_time": 10, "end_time": float("nan"), "label": "failure"}],
                                  [{"start_frame": 0, "end_time": 11, "label": "failure"}]])
def test_invalid_human_intervals_and_units_fail(sample, rows):
    sample["human_intervals"] = rows
    with pytest.raises(ValueError):
        scorer.evaluate_aq(sample)


@pytest.mark.parametrize("change", [lambda d: d.update(schema_version="other"),
                                    lambda d: d.update(private_source="not-accepted"),
                                    lambda d: d.update(human_intervals={}),
                                    lambda d: d.update(predicted_intervals={}),
                                    lambda d: d["predicted_intervals"][0].update(extra=True)])
def test_invalid_document_shape_fails(sample, change):
    change(sample)
    with pytest.raises(ValueError):
        scorer.evaluate_aq(sample)


def test_cli_round_trip_and_existing_output_is_not_overwritten(tmp_path, sample):
    input_path, output_path = tmp_path / "synthetic-input.json", tmp_path / "synthetic-report.json"
    input_path.write_text(json.dumps(sample))
    command = [sys.executable, "-B", str(SCRIPT), "--input", str(input_path)]
    stdout = subprocess.run(command, capture_output=True, text=True, check=True)
    assert json.loads(stdout.stdout) == scorer.evaluate_aq(sample)
    subprocess.run([*command, "--output", str(output_path)], capture_output=True, text=True, check=True)
    original = output_path.read_bytes()
    rejected = subprocess.run([*command, "--output", str(output_path)], capture_output=True, text=True)
    assert rejected.returncode == 2 and "File exists" in rejected.stderr
    assert output_path.read_bytes() == original
    assert json.loads(original) == json.loads(stdout.stdout)


@pytest.mark.parametrize("text", ['{"time_edges": [], "time_edges": [0,1]}', '{"time_edges": [0,NaN]}',
                                  '{"time_edges": [0,Infinity]}'])
def test_ambiguous_json_rejected_before_scoring(tmp_path, text):
    path = tmp_path / "synthetic-invalid.json"
    path.write_text(text)
    result = subprocess.run([sys.executable, "-B", str(SCRIPT), "--input", str(path)],
                            capture_output=True, text=True)
    assert result.returncode == 2
    assert not result.stdout
