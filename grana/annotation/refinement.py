# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic, bounded extra RGB observations from a model candidate only.

This planner does not accept human labels or infer that more frames improve a
prediction. A selected window is a request for evidence, never an AQ decision.
"""
from __future__ import annotations

from bisect import bisect_left
from collections import deque
from fractions import Fraction
import hashlib
import json
import math


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def plan_refinement(bundle, candidate, validation, source_receipt, *, policy_dir,
                    max_extra_frames=64, max_total_images=256, radius_seconds=1.0,
                    dense_hz=10.0, seed="grana-refinement-v1"):
    """Validate inputs and reserve whole local windows without dropping the grid.

    One hash-selected nominal audit is considered first. Remaining trigger
    families take turns; hash ordering within each family prevents a preference
    for early events. Overlap is free, and an unaffordable window is skipped in
    full rather than silently reducing its sampling rate. Long uncertainty
    brackets produce at most three fixed-width windows, never a dense full span.
    Frames in the output are clip-local unless explicitly named source_frames.
    """
    from scripts.annotation_skill import stored_validation_receipt, validate_bundle_candidate

    actual = validate_bundle_candidate(bundle, candidate, policy_dir)
    if bundle.get("mode") == "quality":
        raise ValueError("Refine stage evidence before preparing the quality bundle")
    if digest(actual) != digest(stored_validation_receipt(validation, actual)):
        raise ValueError("Baseline validation receipt differs from the supplied bundle/candidate")
    for name, value, minimum in (("max_extra_frames", max_extra_frames, 0),
                                 ("max_total_images", max_total_images, 1)):
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    for name, value in (("radius_seconds", radius_seconds), ("dense_hz", dense_hz)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")
    if not isinstance(seed, str) or not seed:
        raise ValueError("seed must be nonempty text")
    context = bundle["context"]
    times, baseline = context["time_edges"], sorted(context["evidence_frames"])
    count = len(times) - 1
    identity = context["source_identity"]
    if not isinstance(source_receipt, dict) or source_receipt.get("source_identity") != identity:
        raise ValueError("Source receipt must bind the exact clip source identity")
    if source_receipt.get("annotation_columns_read") is not False:
        raise ValueError("Source receipt must declare annotation_columns_read=false")
    native_range = identity.get("source_frame_range")
    if (not isinstance(native_range, list) or len(native_range) != 2
            or any(type(v) is not int for v in native_range)
            or native_range[0] < 0 or native_range[1] - native_range[0] != count):
        raise ValueError("Source identity needs the native source_frame_range for this clip")
    offset = native_range[0]
    sampling = source_receipt.get("sampling", {})
    if (not isinstance(sampling, dict) or sampling.get("candidate_frames") != baseline
            or sampling.get("source_frames") != [f + offset for f in baseline]
            or sampling.get("grid_origin_seconds") != times[0]):
        raise ValueError("Source receipt frame mapping or native clock differs from the bundle")
    if len(baseline) > max_total_images:
        raise ValueError("max_total_images is smaller than the original evidence; no grid frames may be dropped")
    budget = min(max_extra_frames, max_total_images - len(baseline))
    rational_times = [Fraction(str(t)) for t in times]
    radius, step = Fraction(str(radius_seconds)), 1 / Fraction(str(dense_hz))
    windows = {}

    def add_window(family, frame, trigger, clip_start=0, clip_end=count):
        center = rational_times[frame]
        lower = max(rational_times[clip_start], center - radius)
        upper = min(rational_times[clip_end], center + radius)
        start = bisect_left(rational_times, lower, lo=clip_start, hi=clip_end)
        end = bisect_left(rational_times, upper, lo=start, hi=clip_end)
        if start >= end:
            return
        key = (family, start, end)
        if key in windows:
            windows[key]["triggers"].append(trigger)
            return
        # Fixed 10 Hz grid anchored to the native clip clock, plus the local
        # window endpoints. Decimal rational arithmetic avoids float drift.
        selected = {start, end - 1}
        first_tick = (rational_times[start] - rational_times[0]) // step
        if rational_times[0] + first_tick * step < rational_times[start]:
            first_tick += 1
        target = rational_times[0] + first_tick * step
        last = start - 1
        while target < rational_times[end]:
            frame_index = bisect_left(rational_times, target, lo=last + 1, hi=end)
            if frame_index >= end:
                break
            selected.add(frame_index)
            last = frame_index
            tick = (rational_times[frame_index] - rational_times[0]) // step + 1
            target = rational_times[0] + tick * step
        windows[key] = {"family": family, "start_frame": start, "end_frame": end,
                        "native_time_range": [times[start], times[end]],
                        "requested_frames": sorted(selected), "triggers": [trigger]}

    for boundary in candidate.get("aq_boundaries", []):
        for key in ("frame", "lower_frame", "upper_frame"):
            add_window("aq_boundary", boundary[key], {"type": "aq_boundary", "field": key,
                                                       "frame": boundary[key], "boundary": boundary["frame"]})
    for interval in candidate.get("aq_intervals", []):
        if interval["quality"] != "nominal":
            for key in ("start_frame", "end_frame"):
                add_window("quality_endpoint", interval[key], {"type": "quality_endpoint", "field": key,
                    "frame": interval[key], "quality": interval["quality"]})
    for boundary in candidate["stages"]["boundaries"]:
        for key in ("frame", "lower_frame", "upper_frame"):
            add_window("stage_transition", boundary[key], {"type": "stage_transition", "field": key,
                                                            "frame": boundary[key], "boundary": boundary["frame"]})
    nominal_runs = [(s["start_frame"], s["end_frame"]) for s in candidate.get("aq_intervals", [])
                    if s["quality"] == "nominal"]
    nominal_count = sum(end - start for start, end in nominal_runs)
    order_seed = digest({"seed": seed, "context": bundle["context_sha256"], "candidate": digest(candidate)})
    if nominal_count:
        rank = int(hashlib.sha256((order_seed + ":nominal").encode()).hexdigest(), 16) % nominal_count
        for start, end in nominal_runs:
            if rank < end - start:
                frame = start + rank
                add_window("nominal_audit", frame, {"type": "nominal_audit", "frame": frame}, start, end)
                break
            rank -= end - start
    families = ("nominal_audit", *sorted(("aq_boundary", "quality_endpoint", "stage_transition"),
                key=lambda family: digest({"seed": order_seed, "family": family})))
    queues = {}
    for family in families:
        values = [w for w in windows.values() if w["family"] == family]
        values.sort(key=lambda w: digest({"seed": order_seed, "window": w}))
        queues[family] = deque(values)
    selected_frames, decisions = set(baseline), []
    while any(queues.values()):
        for family in families:
            if not queues[family]:
                continue
            window = queues[family].popleft()
            additional = set(window["requested_frames"]) - selected_frames
            used = len(selected_frames) - len(baseline)
            accept = used + len(additional) <= budget
            decisions.append(window | {"status": "selected" if accept else "skipped",
                "decision_reason": "within_budget" if accept else "whole_window_exceeds_remaining_budget",
                "marginal_extra_frames": len(additional), "remaining_extra_budget_before": budget - used})
            if accept:
                selected_frames.update(additional)
    # Record final coverage, including overlap from other accepted windows.
    for window in decisions:
        covered = set(window["requested_frames"]) & selected_frames
        window["covered_requested_frames"] = len(covered)
        window["requested_frame_count"] = len(window["requested_frames"])
        window["fully_covered"] = len(covered) == window["requested_frame_count"]
    extras = sorted(selected_frames - set(baseline))
    return {"schema_version": "grana.annotation_refinement_plan.v1", "planner_version": "bounded-windows-v1",
        "inputs": {"bundle_sha256": digest(bundle), "candidate_sha256": digest(candidate),
                   "validation_sha256": digest(validation), "source_receipt_sha256": digest(source_receipt)},
        "parameters": {"max_extra_frames": max_extra_frames, "max_total_images": max_total_images,
                       "radius_seconds": radius_seconds, "dense_hz": dense_hz, "seed": seed},
        "family_order": list(families),
        "source_frame_range": native_range, "native_time_range": [times[0], times[-1]],
        "baseline_frames": baseline, "extra_frames": extras,
        "extra_source_frames": [f + offset for f in extras], "expanded_frames": sorted(selected_frames),
        "budget": {"effective_extra_limit": budget, "extra_frames": len(extras),
                   "total_images": len(selected_frames), "unused_extra_budget": budget - len(extras)},
        "windows": decisions, "nominal_audit_available": bool(nominal_count),
        "human_annotations_read": False, "accepted_for_training": False,
        "scope": "Candidate-only local evidence requests. Unselected spans and semantic accuracy remain unverified."}
