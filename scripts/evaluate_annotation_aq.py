#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare AQ intervals on a native clock, without reading media or using an API.

Input JSON has exactly schema_version="grana.aq_evaluation_input.v1",
time_edges (N+1 native timestamps), human_intervals and predicted_intervals.
Human rows have label and either start_frame/end_frame or start_time/end_time.
Prediction rows have start_frame/end_frame/quality; optional reason and
evidence_frames from an annotation candidate are accepted but are not scored.
All intervals are half-open. Frame indices are zero-based native row ranks.
Human times are native-clock seconds, not clip-relative video times; this tool
does not infer offsets, FPS, inclusive endpoints, or label-taxonomy mappings.

Human nominal/suboptimal/failure are scored. Gaps and explicit blank (including
the empty string), unlabeled, unknown and conflict are excluded and reported
separately. Any overlap of human rows is conflict, even for identical labels.
Predictions must partition [0,N) in order. Predicted unknown is abstention: it
remains in the confusion matrix and counts against the reference class recall.

Precision, recall and IoU use seconds, not counts of frames or intervals. A
class with no human support has null metrics; precision is also null when the
class is never predicted. False-positive seconds remain reported in both cases.
Boundary and event-detection scores are deliberately not implemented: they
require a separately declared event-matching and endpoint convention.

Example (explicitly synthetic):
  {"schema_version":"grana.aq_evaluation_input.v1", "time_edges":[0,0.1,0.3],
   "human_intervals":[{"start_frame":0,"end_frame":2,"label":"failure"}],
   "predicted_intervals":[{"start_frame":0,"end_frame":2,"quality":"unknown"}]}

Run: python scripts/evaluate_annotation_aq.py --input INPUT.json [--output NEW.json]
Without --output, the report is printed. Existing output files are never replaced.
The report includes a canonical input digest, but no source paths or identifiers.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path


INPUT_VERSION = "grana.aq_evaluation_input.v1"
REPORT_VERSION = "grana.aq_evaluation_report.v1"
CLASSES = ("nominal", "suboptimal", "failure")
PREDICTIONS = (*CLASSES, "unknown")
EXCLUDED = ("blank", "unlabeled", "unknown", "conflict")
HUMAN_LABELS = (*CLASSES, *EXCLUDED)


def _keys(value, required, name, optional=()):
    if (not isinstance(value, dict) or not set(required) <= set(value)
            or set(value) - set(required) - set(optional)):
        raise ValueError(f"{name} requires {sorted(required)}; optional fields: {sorted(optional)}")


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} must be a finite number") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _frame(value, name):
    if type(value) is not int:
        raise ValueError(f"{name} must be an integer frame rank")
    return value


def _frame_span(row, edges, name):
    start, end = (_frame(row[key], f"{name}.{key}") for key in ("start_frame", "end_frame"))
    if not 0 <= start < end < len(edges):
        raise ValueError(f"{name} must be a nonempty half-open interval within [0,N)")
    return start, end


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def evaluate_aq(document: dict) -> dict:
    """Score one recording's declared labels; never infer nominal from blanks.

    This evaluates agreement with supplied reference labels, not their validity.
    No input is mutated, repaired, reordered or written by this function.
    """
    _keys(document, ("schema_version", "time_edges", "human_intervals", "predicted_intervals"), "input")
    if document["schema_version"] != INPUT_VERSION:
        raise ValueError(f"schema_version must be {INPUT_VERSION}")
    raw_edges = document["time_edges"]
    if not isinstance(raw_edges, list) or len(raw_edges) < 2:
        raise ValueError("time_edges must contain N+1 finite strictly increasing native times")
    edges = [_number(value, "time_edges value") for value in raw_edges]
    if any(b <= a for a, b in zip(edges, edges[1:])):
        raise ValueError("time_edges must be strictly increasing")
    duration = edges[-1] - edges[0]
    if not math.isfinite(duration):
        raise ValueError("Native-clock duration must be finite")
    if not isinstance(document["human_intervals"], list):
        raise ValueError("human_intervals must be an array")
    if not isinstance(document["predicted_intervals"], list):
        raise ValueError("predicted_intervals must be an array")

    human = []
    for index, row in enumerate(document["human_intervals"]):
        name = f"human_intervals[{index}]"
        if isinstance(row, dict) and set(row) == {"label", "start_frame", "end_frame"}:
            first, last = _frame_span(row, edges, name)
            start, end = edges[first], edges[last]
        else:
            _keys(row, ("label", "start_time", "end_time"), name)
            start, end = (_number(row[key], f"{name}.{key}") for key in ("start_time", "end_time"))
            if not edges[0] <= start < end <= edges[-1]:
                raise ValueError(f"{name} times must form a nonempty interval within the native clock")
        label = row["label"]
        if not isinstance(label, str) or label not in (*HUMAN_LABELS, ""):
            raise ValueError(f"{name}.label must be one of {HUMAN_LABELS} or an empty blank label")
        human.append((start, end, label or "blank"))

    predictions, cursor = [], 0
    for index, row in enumerate(document["predicted_intervals"]):
        name = f"predicted_intervals[{index}]"
        _keys(row, ("start_frame", "end_frame", "quality"), name, ("reason", "evidence_frames"))
        first, last = _frame_span(row, edges, name)
        if first != cursor:
            raise ValueError("predicted_intervals must partition [0,N) in order, without gaps or overlaps")
        if not isinstance(row["quality"], str) or row["quality"] not in PREDICTIONS:
            raise ValueError(f"{name}.quality must be one of {PREDICTIONS}")
        predictions.append((edges[first], edges[last], row["quality"]))
        cursor = last
    if cursor != len(edges) - 1:
        raise ValueError("predicted_intervals must completely partition [0,N)")

    # Sweep only interval endpoints, not every frame. Human rows are independent
    # tracks: simultaneous rows are conflict, without class precedence or voting.
    starts, ends = defaultdict(list), defaultdict(list)
    cuts = {edges[0], edges[-1]}
    for index, (start, end, label) in enumerate(human):
        starts[start].append((index, label))
        ends[end].append(index)
        cuts.update((start, end))
    for start, end, _ in predictions:
        cuts.update((start, end))
    cuts = sorted(cuts)
    matrix_parts = {label: {quality: [] for quality in PREDICTIONS} for label in CLASSES}
    human_parts = {label: [] for label in HUMAN_LABELS}
    predicted_parts = {quality: [] for quality in PREDICTIONS}
    active, prediction_index = {}, 0
    for start, end in zip(cuts, cuts[1:]):
        for index in ends[start]:
            del active[index]
        active.update(starts[start])
        label = "conflict" if len(active) > 1 else next(iter(active.values()), "blank")
        while predictions[prediction_index][1] <= start:
            prediction_index += 1
        quality = predictions[prediction_index][2]
        seconds = end - start
        human_parts[label].append(seconds)
        predicted_parts[quality].append(seconds)
        if label in CLASSES:
            matrix_parts[label][quality].append(seconds)

    confusion = {label: {quality: math.fsum(parts) for quality, parts in row.items()}
                 for label, row in matrix_parts.items()}
    human_seconds = {label: math.fsum(parts) for label, parts in human_parts.items()}
    predicted_seconds = {quality: math.fsum(parts) for quality, parts in predicted_parts.items()}
    scored_seconds = math.fsum(human_seconds[label] for label in CLASSES)
    excluded_seconds = math.fsum(human_seconds[label] for label in EXCLUDED)
    scored_predictions = {quality: math.fsum(confusion[label][quality] for label in CLASSES)
                          for quality in PREDICTIONS}
    per_class = {}
    for label in CLASSES:
        support = human_seconds[label]
        true_positive = confusion[label][label]
        false_positive = math.fsum(confusion[other][label] for other in CLASSES if other != label)
        false_negative = math.fsum(confusion[label][other] for other in PREDICTIONS if other != label)
        per_class[label] = {
            "support_seconds": support, "predicted_seconds": scored_predictions[label],
            "true_positive_seconds": true_positive, "false_positive_seconds": false_positive,
            "false_negative_seconds": false_negative,
            "precision": _ratio(true_positive, scored_predictions[label]) if support else None,
            "recall": _ratio(true_positive, support),
            "iou": _ratio(true_positive, support + false_positive) if support else None,
        }
    # Canonicalization also rejects non-JSON/nonfinite optional candidate metadata.
    encoded = json.dumps(document, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return {
        "schema_version": REPORT_VERSION, "input_sha256": hashlib.sha256(encoded.encode()).hexdigest(),
        "unit": "seconds", "source_frames": len(edges) - 1, "source_seconds": duration,
        "human_coverage": {
            "seconds_by_label": human_seconds,
            "fraction_by_label": {label: seconds / duration for label, seconds in human_seconds.items()},
            "scored_seconds": scored_seconds, "excluded_seconds": excluded_seconds,
            "scored_fraction": scored_seconds / duration,
        },
        "prediction_coverage": {"all_source_seconds": predicted_seconds,
                                "scored_seconds": scored_predictions},
        "confusion_seconds": confusion, "per_class": per_class,
        "accuracy": _ratio(math.fsum(confusion[label][label] for label in CLASSES), scored_seconds),
        "abstention": {
            "all_source_seconds": predicted_seconds["unknown"],
            "all_source_fraction": predicted_seconds["unknown"] / duration,
            "scored_seconds": scored_predictions["unknown"],
            "scored_fraction": _ratio(scored_predictions["unknown"], scored_seconds),
        },
        "scoring_policy": {
            "accuracy_domain": "Time-weighted agreement on human nominal/suboptimal/failure seconds only; not whole-recording accuracy or evidence of held-out generalization.",
            "human_overlap": "any simultaneous human rows are conflict, including identical labels",
            "human_excluded_labels": list(EXCLUDED),
            "predicted_unknown": "abstention included as a false negative for the supported human class",
            "unsupported_class_metrics": "null when human support is zero; false-positive seconds are retained",
            "precision_domain": "only human nominal/suboptimal/failure intervals; excluded spans are not negatives",
        },
        "boundary_evaluation": {
            "status": "not_implemented",
            "reason": "Onset/offset and event scores require a declared endpoint convention and event matching; no nearest-boundary matching is inferred.",
        },
    }


def _read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def nonfinite(value):
        raise ValueError(f"Nonfinite JSON value: {value}")

    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique, parse_constant=nonfinite)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="Create a new report JSON; default: stdout")
    args = parser.parse_args(argv)
    try:
        report = evaluate_aq(_read_json(args.input))
        encoded = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        if args.output:
            with args.output.open("x", encoding="utf-8") as handle:
                handle.write(encoded)
        else:
            print(encoded, end="")
    except (ValueError, TypeError, OSError, OverflowError) as error:
        parser.exit(2, f"evaluate_annotation_aq: {error}\n")


if __name__ == "__main__":
    main()
