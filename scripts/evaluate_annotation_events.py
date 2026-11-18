#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Diagnose stage/AQ events against explicit reviewed windows on a native clock.

Input (see tests/test_evaluate_annotation_events.py for runnable synthetic cases):
  {"schema_version":"grana.annotation_event_evaluation_input.v1",
   "time_edges":[10,11,12],
   "candidate":{"stages":{"segments":[{"start_frame":0,"end_frame":2,
     "stage_id":"S1","identity_status":"resolved"}]},
     "aq_intervals":[{"start_frame":0,"end_frame":2,"quality":"nominal"}]},
   "reference":{"status":"independent_agent_candidate","policy_version":"draft-v1",
     "aq_windows":[{"id":"normal1","start_frame":0,"end_frame":2,
       "quality":"nominal","role":"normal"}],"stage_windows":[],"events":[]}}

Reference status is a caller declaration, never verified human acceptance.
AQ windows require id/start_frame/end_frame/quality/role; role is normal,
recovery, event, or other. Normal and recovery require explicit nominal quality.
Stage windows require id/start_frame/end_frame/stage_id. Each window channel is
nonoverlapping; the two channels can overlap. Gaps are unreviewed, never normal.
Reference unknown AQ is reported but excluded from known-class agreement.

Events require id/quality/start_frame/end_frame/start_boundary_bracket/
end_boundary_bracket, with optional recovery_window_id linking an explicit
normal recovery window. Brackets are inclusive possible cuts [lower,upper].
Event nominal spans must not overlap; uncertain brackets may overlap. All spans
are half-open native frame ranks. There is no FPS or clock-offset inference.

Same-class events match one-to-one against maximal predicted bad-class runs.
Eligibility requires positive overlap with the guaranteed core
[start_bracket.upper,end_bracket.lower); if empty, use the possible envelope
and flag the match as uncertain. Maximum match count is optimized first, then
total temporal IoU. Thus one long prediction cannot detect multiple events.
Unmatched predictions are not automatically false positives in a partial review.
False alarms use only explicit nominal windows. Predicted unknown is separate.

CLI: python scripts/evaluate_annotation_events.py --input INPUT.json --output NEW.json
Without --output, print JSON. Existing output files are never replaced. No media,
network, labels, or files beyond the specified input are accessed by the scorer.
This diagnostic does not replace full candidate/schema/evidence validation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


INPUT_VERSION = "grana.annotation_event_evaluation_input.v1"
REPORT_VERSION = "grana.annotation_event_evaluation_report.v1"
QUALITIES = ("nominal", "suboptimal", "failure", "unknown")
KNOWN = QUALITIES[:-1]
BAD = ("suboptimal", "failure")


def _keys(value, required, name, optional=()):
    if (not isinstance(value, dict) or not set(required) <= set(value)
            or set(value) - set(required) - set(optional)):
        raise ValueError(f"{name} requires {sorted(required)}; optional: {sorted(optional)}")


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty text")
    return value


def _frame(value, n, name):
    if type(value) is not int or not 0 <= value <= n:
        raise ValueError(f"{name} must be an integer frame cut within [0,N]")
    return value


def _span(row, n, name):
    start, end = (_frame(row.get(key), n, name + "." + key)
                  for key in ("start_frame", "end_frame"))
    if start >= end:
        raise ValueError(f"{name} must be a nonempty half-open interval")
    return start, end


def _bracket(value, nominal, n, name):
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{name} requires [lower_frame,upper_frame]")
    lower, upper = (_frame(x, n, name) for x in value)
    if not lower <= nominal <= upper:
        raise ValueError(f"{name} must contain its nominal boundary")
    return lower, upper


def _rows(value, name):
    if not isinstance(value, list):
        raise ValueError(f"{name} must be an array")
    return value


def _nonoverlap(rows, name):
    ordered = sorted(rows, key=lambda x: (x["start_frame"], x["end_frame"]))
    if any(a["end_frame"] > b["start_frame"] for a, b in zip(ordered, ordered[1:])):
        raise ValueError(f"{name} must not overlap")
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{name} IDs must be unique")


def _partition(rows, n, name, stage=False):
    cursor = 0
    for index, row in enumerate(_rows(rows, name)):
        if not isinstance(row, dict):
            raise ValueError(f"{name}[{index}] must be an object")
        first, last = _span(row, n, name)
        if first != cursor:
            raise ValueError(f"{name} must partition [0,N) in order without gaps or overlaps")
        if stage:
            identity = row.get("identity_status")
            stage_id = row.get("stage_id")
            if identity not in ("resolved", "uncertain", "unresolved"):
                raise ValueError("Stage identity_status must be resolved, uncertain or unresolved")
            if identity == "unresolved":
                if stage_id is not None:
                    raise ValueError("Unresolved stage identity requires stage_id null")
            else:
                _text(stage_id, "stage_id")
        elif row.get("quality") not in QUALITIES:
            raise ValueError(f"AQ quality must be one of {QUALITIES}; no label mapping is inferred")
        cursor = last
    if cursor != n:
        raise ValueError(f"{name} must completely partition [0,N)")


def _overlap(a, b, edges):
    first, last = max(a[0], b[0]), min(a[1], b[1])
    return edges[last] - edges[first] if first < last else 0.0


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def _match_events(events, predictions, edges):
    """Min-cost maximum bipartite flow: cardinality first, then temporal IoU."""
    count_e, count_p = len(events), len(predictions)
    sink = count_e + count_p + 1
    graph = [[] for _ in range(sink + 1)]

    def add(a, b, cost):
        forward = [b, len(graph[b]), 1, cost]
        backward = [a, len(graph[a]), 0, -cost]
        graph[a].append(forward)
        graph[b].append(backward)
        return forward

    for i in range(count_e):
        add(0, i + 1, 0.0)
    for j in range(count_p):
        add(count_e + j + 1, sink, 0.0)
    links = []
    for i, event in enumerate(events):
        core = (event["start_boundary_bracket"][1], event["end_boundary_bracket"][0])
        uncertain = core[0] >= core[1]
        support = ((event["start_boundary_bracket"][0], event["end_boundary_bracket"][1])
                   if uncertain else core)
        for j, pred in enumerate(predictions):
            span = pred["start_frame"], pred["end_frame"]
            overlap = _overlap(support, span, edges)
            if event["quality"] != pred["quality"] or overlap <= 0:
                continue
            union = (edges[support[1]] - edges[support[0]]
                     + edges[span[1]] - edges[span[0]] - overlap)
            iou = overlap / union
            link = add(i + 1, count_e + j + 1, -iou)
            links.append((i, j, link, iou, overlap, uncertain, support))
    # Bellman-Ford handles negative forward costs and reassignment residual edges.
    while True:
        distances = [math.inf] * len(graph)
        previous = [None] * len(graph)
        distances[0] = 0.0
        for _ in range(len(graph) - 1):
            changed = False
            for node, outgoing in enumerate(graph):
                if not math.isfinite(distances[node]):
                    continue
                for edge_index, edge in enumerate(outgoing):
                    target, _, capacity, cost = edge
                    value = distances[node] + cost
                    if capacity and value < distances[target] - 1e-12:
                        distances[target] = value
                        previous[target] = node, edge_index
                        changed = True
            if not changed:
                break
        if previous[sink] is None:
            break
        node = sink
        while node:
            parent, edge_index = previous[node]
            edge = graph[parent][edge_index]
            edge[2] -= 1
            graph[node][edge[1]][2] += 1
            node = parent
    return [(i, j, iou, overlap, uncertain, support)
            for i, j, edge, iou, overlap, uncertain, support in links if edge[2] == 0]


def _boundary_diagnostic(point, predicted_bracket, reference_bracket, edges):
    value = edges[point]
    lo, hi = (edges[x] for x in reference_bracket)
    pred_lo, pred_hi = (edges[x] for x in predicted_bracket)
    signed = value - lo if value < lo else value - hi if value > hi else 0.0
    return {
        "predicted_frame": point, "predicted_bracket": list(predicted_bracket),
        "reference_bracket": list(reference_bracket),
        "signed_distance_to_reference_bracket_seconds": signed,
        "absolute_distance_to_reference_bracket_seconds": abs(signed),
        "bracket_distance_seconds": max(0.0, lo - pred_hi, pred_lo - hi),
        "predicted_bracket_width_seconds": pred_hi - pred_lo,
        "reference_bracket_width_seconds": hi - lo,
    }


def evaluate_events(document):
    """Return agreement diagnostics without accepting or mutating either input."""
    _keys(document, ("schema_version", "time_edges", "candidate", "reference"), "input")
    if document["schema_version"] != INPUT_VERSION:
        raise ValueError(f"schema_version must be {INPUT_VERSION}")
    raw_edges = document["time_edges"]
    if (not isinstance(raw_edges, list) or len(raw_edges) < 2
            or any(type(x) not in (int, float) for x in raw_edges)):
        raise ValueError("time_edges must contain N+1 finite strictly increasing numbers")
    try:
        edges = [float(x) for x in raw_edges]
    except OverflowError as error:
        raise ValueError("time_edges must be finite") from error
    if (any(not math.isfinite(x) for x in edges)
            or any(b <= a for a, b in zip(edges, edges[1:]))
            or not math.isfinite(edges[-1] - edges[0])):
        raise ValueError("time_edges must be finite and strictly increasing")
    n = len(edges) - 1
    candidate = document["candidate"]
    if not isinstance(candidate, dict) or not isinstance(candidate.get("stages"), dict):
        raise ValueError("candidate requires stages.segments and aq_intervals")
    stages = candidate["stages"].get("segments")
    predictions = candidate.get("aq_intervals")
    _partition(stages, n, "candidate.stages.segments", stage=True)
    _partition(predictions, n, "candidate.aq_intervals")
    boundary_map = {}
    if "aq_boundaries" in candidate:
        boundaries = _rows(candidate["aq_boundaries"], "candidate.aq_boundaries")
        cuts = [row["start_frame"] for row in predictions[1:]]
        if [row.get("frame") if isinstance(row, dict) else None for row in boundaries] != cuts:
            raise ValueError("candidate.aq_boundaries must match every AQ partition cut in order")
        for boundary in boundaries:
            cut = _frame(boundary["frame"], n, "AQ boundary")
            bracket = _bracket([boundary.get("lower_frame"), boundary.get("upper_frame")],
                               cut, n, "AQ boundary bracket")
            boundary_map[cut] = bracket

    reference = document["reference"]
    _keys(reference, ("status", "policy_version", "aq_windows", "stage_windows", "events"), "reference")
    if reference["status"] not in ("independent_agent_candidate", "human_reviewed"):
        raise ValueError("reference.status must be independent_agent_candidate or human_reviewed")
    _text(reference["policy_version"], "reference.policy_version")
    aq_windows = _rows(reference["aq_windows"], "reference.aq_windows")
    stage_windows = _rows(reference["stage_windows"], "reference.stage_windows")
    events = _rows(reference["events"], "reference.events")
    for row in aq_windows:
        _keys(row, ("id", "start_frame", "end_frame", "quality", "role"), "AQ window")
        _text(row["id"], "AQ window.id")
        _span(row, n, "AQ window")
        if row["quality"] not in QUALITIES or row["role"] not in ("normal", "recovery", "event", "other"):
            raise ValueError("AQ window has unsupported quality or role")
        if row["role"] in ("normal", "recovery") and row["quality"] != "nominal":
            raise ValueError("Normal and recovery windows require explicit nominal quality")
    _nonoverlap(aq_windows, "AQ windows")
    for row in stage_windows:
        _keys(row, ("id", "start_frame", "end_frame", "stage_id"), "stage window")
        _text(row["id"], "stage window.id")
        _text(row["stage_id"], "stage window.stage_id")
        _span(row, n, "stage window")
    _nonoverlap(stage_windows, "stage windows")
    windows_by_id = {row["id"]: row for row in aq_windows}
    for row in events:
        _keys(row, ("id", "quality", "start_frame", "end_frame", "start_boundary_bracket",
                    "end_boundary_bracket"), "event", ("recovery_window_id",))
        _text(row["id"], "event.id")
        start, end = _span(row, n, "event")
        if row["quality"] not in BAD:
            raise ValueError("Reference events require suboptimal or failure; unknown is not a positive event")
        _bracket(row["start_boundary_bracket"], start, n, "event start bracket")
        _bracket(row["end_boundary_bracket"], end, n, "event end bracket")
        if "recovery_window_id" in row:
            _text(row["recovery_window_id"], "event.recovery_window_id")
            recovery = windows_by_id.get(row["recovery_window_id"])
            if recovery is None or recovery["role"] != "recovery":
                raise ValueError("recovery_window_id must name an explicit normal recovery window")
            if recovery["start_frame"] < end:
                raise ValueError("Linked recovery must not begin before the event nominal end")
    _nonoverlap(events, "events")

    matrix = {label: {quality: 0.0 for quality in QUALITIES} for label in QUALITIES}
    recovery_parts = {quality: [] for quality in QUALITIES}
    for window in aq_windows:
        span = window["start_frame"], window["end_frame"]
        for pred in predictions:
            overlap = _overlap(span, (pred["start_frame"], pred["end_frame"]), edges)
            matrix[window["quality"]][pred["quality"]] += overlap
            if window["role"] == "recovery":
                recovery_parts[pred["quality"]].append(overlap)
    support = {label: math.fsum(matrix[label].values()) for label in QUALITIES}
    known_seconds = math.fsum(support[label] for label in KNOWN)
    reviewed_seconds = math.fsum(support.values())
    source_seconds = edges[-1] - edges[0]
    stage_matrix = {}
    stage_seconds = 0.0
    for window in stage_windows:
        span = window["start_frame"], window["end_frame"]
        stage_seconds += edges[span[1]] - edges[span[0]]
        row = stage_matrix.setdefault(window["stage_id"], {})
        for pred in stages:
            overlap = _overlap(span, (pred["start_frame"], pred["end_frame"]), edges)
            if overlap:
                key = pred["stage_id"] if pred["identity_status"] == "resolved" else "unresolved_or_uncertain"
                row[key] = row.get(key, 0.0) + overlap

    # Merge same-quality adjacency before counting events; output fragmentation
    # must not create additional detection opportunities.
    runs = []
    for pred in predictions:
        if runs and runs[-1]["quality"] == pred["quality"]:
            runs[-1]["end_frame"] = pred["end_frame"]
        else:
            runs.append({key: pred[key] for key in ("start_frame", "end_frame", "quality")})
    predicted_events = [row for row in runs if row["quality"] in BAD]
    matches = _match_events(events, predicted_events, edges)
    event_matches = []
    for i, j, iou, overlap, uncertain, match_span in matches:
        event, pred = events[i], predicted_events[j]
        diagnostic = {
            "event_id": event["id"], "quality": event["quality"], "prediction_index": j,
            "predicted_span": [pred["start_frame"], pred["end_frame"]],
            "matching_span": list(match_span), "matching_used_uncertain_envelope": uncertain,
            "matching_overlap_seconds": overlap, "matching_temporal_iou": iou,
        }
        for name in ("start", "end"):
            point = pred[name + "_frame"]
            diagnostic[name + "_boundary"] = _boundary_diagnostic(
                point, boundary_map.get(point, (point, point)), event[name + "_boundary_bracket"], edges)
        diagnostic["linked_recovery_continuation_seconds"] = None
        if "recovery_window_id" in event:
            recovery = windows_by_id[event["recovery_window_id"]]
            diagnostic["linked_recovery_window_id"] = recovery["id"]
            diagnostic["linked_recovery_continuation_seconds"] = _overlap(
                (pred["start_frame"], pred["end_frame"]),
                (recovery["start_frame"], recovery["end_frame"]), edges)
        event_matches.append(diagnostic)
    matched_refs = {i for i, *_ in matches}
    matched_predictions = {j for _, j, *_ in matches}
    class_events = {}
    for quality in BAD:
        total = sum(event["quality"] == quality for event in events)
        matched = sum(events[i]["quality"] == quality for i in matched_refs)
        class_events[quality] = {"reference_events": total, "matched_events": matched,
                                 "missed_events": total - matched, "detection_fraction": _ratio(matched, total)}
    normal = matrix["nominal"]
    recovery = {quality: math.fsum(parts) for quality, parts in recovery_parts.items()}
    encoded = json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return {
        "schema_version": REPORT_VERSION, "input_sha256": hashlib.sha256(encoded.encode()).hexdigest(),
        "reference_status": reference["status"], "reference_policy_version": reference["policy_version"],
        "reference_status_verified": False, "semantic_accuracy_verified": False, "accepted_for_training": False,
        "source_frames": n, "source_seconds": source_seconds, "native_start_time": edges[0],
        "aq": {
            "reviewed_seconds": reviewed_seconds, "unreviewed_seconds": max(0.0, source_seconds - reviewed_seconds),
            "known_reference_seconds": known_seconds, "unknown_reference_seconds": support["unknown"],
            "confusion_seconds": matrix, "reference_support_seconds": support,
            "known_reference_agreement": _ratio(math.fsum(matrix[q][q] for q in KNOWN), known_seconds),
            "predicted_unknown_on_known_reference_seconds": math.fsum(matrix[q]["unknown"] for q in KNOWN),
        },
        "reviewed_normal": {
            "seconds": support["nominal"], "predicted_seconds": normal,
            "false_alarm_seconds": math.fsum(normal[q] for q in BAD),
            "false_alarm_fraction": _ratio(math.fsum(normal[q] for q in BAD), support["nominal"]),
            "unknown_seconds": normal["unknown"],
        },
        "reviewed_recovery": {
            "seconds": math.fsum(recovery.values()), "predicted_seconds": recovery,
            "false_alarm_seconds": math.fsum(recovery[q] for q in BAD),
            "unknown_seconds": recovery["unknown"],
            "matched_event_continuation_seconds": math.fsum(
                row["linked_recovery_continuation_seconds"] or 0.0 for row in event_matches),
        },
        "stage": {
            "reviewed_seconds": stage_seconds, "confusion_seconds": stage_matrix,
            "resolved_agreement": _ratio(math.fsum(row.get(label, 0.0) for label, row in stage_matrix.items()), stage_seconds),
            "unresolved_or_uncertain_seconds": math.fsum(row.get("unresolved_or_uncertain", 0.0) for row in stage_matrix.values()),
        },
        "events": {
            "by_class": class_events, "matches": event_matches,
            "missed_reference_event_ids": [row["id"] for i, row in enumerate(events) if i not in matched_refs],
            "unmatched_predictions": [{"prediction_index": i, **row} for i, row in enumerate(predicted_events)
                                       if i not in matched_predictions],
            "prediction_count_after_merging_same_class_adjacency": len(predicted_events),
        },
        "interpretation": {
            "agreement": "Agreement with supplied reviewed windows, not verified accuracy or automatic human acceptance.",
            "scope": "Unreviewed gaps and reference unknowns are not nominal negatives; event proposals can be incomplete.",
            "unmatched_predictions": "Unmatched does not establish false positive; use explicit normal-window false alarms.",
            "event_matching": "Maximum one-to-one same-class match count, then total temporal IoU; positive core overlap or flagged uncertain-envelope fallback.",
            "boundary_distance": "Signed point-to-reference-bracket distance: negative early, positive late, zero inside; bracket distance and widths reported separately.",
            "recovery": "Global false alarm uses every explicit normal recovery window; continuation uses only the matched contiguous bad run and its linked recovery window.",
            "candidate_validation": "Clock/partition/minimum shape checked; full schema, evidence, semantics and outcomes require their separate validator.",
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        document = json.loads(args.input.read_text())
        report = evaluate_events(document)
        output = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        if args.output:
            with args.output.open("x", encoding="utf-8") as handle:
                handle.write(output)
        else:
            print(output, end="")
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    main()
