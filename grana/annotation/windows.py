# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plan bounded annotation clips from predicted objectives, without reading labels.

Stage hypotheses select clip boundaries; they do not constrain the next model's
answer. Core ranges partition the original clip. Context ranges may overlap.
This pure planner does not validate delivery, inspect media, or accept labels;
the source-checkout preparation CLI binds those artifacts before using a plan.
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from fractions import Fraction
import math


def _parameter(value, name, *, allow_zero=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0 or (not allow_zero and value == 0)):
        raise ValueError(f"{name} must be finite and {'nonnegative' if allow_zero else 'positive'}")
    return Fraction(str(value))


def _frame(value, count, name):
    if type(value) is not int or not 0 <= value <= count:
        raise ValueError(f"{name} must be an integer native-frame cut within the clip")
    return value


def _sample(clock, start, end, stride, limit):
    selected, frame = {start, end - 1}, start
    while frame < end - 1:
        tick = (clock[frame] - clock[start]) // stride + 1
        frame = bisect_left(clock, clock[start] + tick * stride, lo=frame + 1, hi=end)
        if frame < end:
            selected.add(frame)
        if len(selected) > limit:
            break
    if len(selected) > limit:
        raise ValueError("A context window exceeds max_images; reduce core/context duration or increase stride. No frames were dropped.")
    return sorted(selected)


def plan_stage_windows(context, candidate, *, max_core_seconds=30.0, context_seconds=2.0,
                       stride_seconds=0.5, max_images=128):
    """Build a deterministic plan; require already declared native frame mapping.

    Adjacent same-ID stage fragments are one predicted run even when confidence
    differs. Long runs split by elapsed native time. A single native frame wider
    than max_core_seconds cannot satisfy the cap and is rejected explicitly.
    """
    core_limit = _parameter(max_core_seconds, "max_core_seconds")
    padding = _parameter(context_seconds, "context_seconds", allow_zero=True)
    stride = _parameter(stride_seconds, "stride_seconds")
    if type(max_images) is not int or max_images < 1:
        raise ValueError("max_images must be a positive integer")
    if not isinstance(context, dict):
        raise ValueError("context must be an object")
    times = context.get("time_edges")
    if (not isinstance(times, list) or len(times) < 2
            or any(type(x) not in (int, float) or not math.isfinite(x) for x in times)
            or any(b <= a for a, b in zip(times, times[1:]))):
        raise ValueError("time_edges must contain N+1 finite strictly increasing native timestamps")
    clock = [Fraction(str(t)) for t in times]
    count = len(times) - 1
    identity = context.get("source_identity")
    source_range = identity.get("source_frame_range") if isinstance(identity, dict) else None
    if (not isinstance(source_range, list) or len(source_range) != 2
            or any(type(x) is not int for x in source_range)
            or source_range[0] < 0 or source_range[1] - source_range[0] != count):
        raise ValueError("source_identity.source_frame_range must bind the exact clip-to-native offset")
    if identity.get("frame_mapping") != "native_frame = candidate_frame + source_frame_range[0]":
        raise ValueError("source_identity requires the explicit clip-to-native frame_mapping")
    registry = context.get("registry")
    if not isinstance(registry, list) or not registry:
        raise ValueError("context requires an explicit registry")
    ids = {row.get("id") for row in registry if isinstance(row, dict)}
    if (len(ids) != len(registry) or any(not isinstance(x, str) or not x for x in ids)):
        raise ValueError("Registry IDs must be distinct nonempty strings")
    stages = candidate.get("stages") if isinstance(candidate, dict) else None
    segments = stages.get("segments") if isinstance(stages, dict) else None
    if not isinstance(segments, list) or not segments:
        raise ValueError("candidate requires stages.segments")
    runs, cursor = [], 0
    for segment in segments:
        if not isinstance(segment, dict):
            raise ValueError("Stage segment must be an object")
        start = _frame(segment.get("start_frame"), count, "stage start")
        end = _frame(segment.get("end_frame"), count, "stage end")
        if start != cursor or start >= end:
            raise ValueError("Stage segments must partition the clip in order without gaps or overlaps")
        stage_id, status = segment.get("stage_id"), segment.get("identity_status")
        if ((status == "unresolved" and stage_id is not None)
                or status not in ("resolved", "uncertain", "unresolved")
                or (status != "unresolved" and stage_id not in ids)):
            raise ValueError("Stage hypothesis has incompatible identity_status or registry ID")
        if runs and runs[-1]["stage_id"] == stage_id:
            runs[-1]["end_frame"] = end
            runs[-1]["statuses"].add(status)
        else:
            runs.append({"start_frame": start, "end_frame": end, "stage_id": stage_id, "statuses": {status}})
        cursor = end
    if cursor != count:
        raise ValueError("Stage segments must completely partition the clip")
    invalid = stages.get("invalid_intervals", [])
    boundaries = stages.get("boundaries", [])
    if not isinstance(invalid, list) or not isinstance(boundaries, list):
        raise ValueError("Stage invalid_intervals and boundaries must be arrays")
    invalid_spans = []
    for row in invalid:
        if not isinstance(row, dict):
            raise ValueError("Invalid source interval must be an object")
        start, end = (_frame(row.get(k), count, k) for k in ("start_frame", "end_frame"))
        if start >= end:
            raise ValueError("Invalid source interval must be nonempty")
        invalid_spans.append((start, end))
    uncertain_cuts = []
    for row in boundaries:
        if not isinstance(row, dict):
            raise ValueError("Stage boundary must be an object")
        lower, cut, upper = (_frame(row.get(k), count, k) for k in ("lower_frame", "frame", "upper_frame"))
        if not lower <= cut <= upper:
            raise ValueError("Stage boundary bracket must contain its nominal cut")
        if lower < upper:
            uncertain_cuts.append((lower, upper))
    offset = source_range[0]
    windows = []
    for run_index, run in enumerate(runs):
        start, run_end = run["start_frame"], run["end_frame"]
        while start < run_end:
            end = bisect_right(clock, clock[start] + core_limit, lo=start + 1, hi=run_end + 1) - 1
            if end <= start:
                raise ValueError("A single native frame exceeds max_core_seconds")
            left = bisect_left(clock, clock[start] - padding, hi=start + 1)
            right = bisect_right(clock, clock[end] + padding, lo=end, hi=count + 1) - 1
            frames = _sample(clock, left, right, stride, max_images)
            flags = []
            if run["statuses"] != {"resolved"}:
                flags.append("coarse_stage_identity_uncertain_or_unresolved")
            if any(a < right and left < b for a, b in invalid_spans):
                flags.append("coarse_source_invalid_interval_intersects_context")
            if any(a < right and left <= b for a, b in uncertain_cuts):
                flags.append("coarse_stage_boundary_uncertainty_intersects_context")
            windows.append({
                "window_id": f"window_{len(windows):04d}", "predicted_run_index": run_index,
                "stage_hypothesis": {"stage_id": run["stage_id"], "identity_statuses": sorted(run["statuses"]),
                                     "enforced": False},
                "predicted_run_frame_range": [run["start_frame"], run_end],
                "core_frame_range": [start, end], "context_frame_range": [left, right],
                "source_core_frame_range": [offset + start, offset + end],
                "source_frame_range": [offset + left, offset + right],
                "window_local_core_range": [start - left, end - left],
                "native_core_time_range": [times[start], times[end]],
                "native_context_time_range": [times[left], times[right]],
                "predicted_run_was_split": start != run["start_frame"] or end != run_end,
                "source_evidence_frames": [offset + f for f in frames],
                "window_evidence_frames": [f - left for f in frames],
                "expected_images": len(frames), "requires_review": bool(flags), "review_reasons": flags,
                "outcome_scope": "This context clip only; a clipped predicted-stage outcome is not a whole-stage outcome.",
            })
            start = end
    return {"schema_version": "grana.annotation_stage_window_plan.v1", "planner_version": "predicted-stage-windows-v1",
            "parameters": {"max_core_seconds": max_core_seconds, "context_seconds": context_seconds,
                           "stride_seconds": stride_seconds, "max_images": max_images},
            "source_frame_range": list(source_range), "native_time_range": [times[0], times[-1]],
            "windows": windows, "core_partition_verified": True, "stage_hints_are_hypotheses": True,
            "human_annotations_read": False, "accepted_for_training": False,
            "scope": "Predicted stage cores cover the input clip once. Context may overlap. No automatic stitching, stage enforcement, or acceptance."}
