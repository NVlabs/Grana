# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict stage-only candidates; uncertainty axes never imply reward labels."""
from copy import deepcopy
from math import isfinite


def _registry(stage_ids):
    if (not isinstance(stage_ids, (list, tuple)) or not 1 <= len(stage_ids) <= 26
            or any(not isinstance(x, str) or not x.strip() for x in stage_ids)
            or len(set(stage_ids)) != len(stage_ids)):
        raise ValueError("stage_ids must contain 1..26 distinct nonempty strings")
    return list(stage_ids)


def candidate_schema(stage_ids):
    """JSON Schema for a strict structured-output request (all keys required)."""
    ids = _registry(stage_ids)
    integer = {"type": "integer", "minimum": 0}
    text = {"type": "string", "minLength": 1}
    evidence = {"type": "array", "items": integer, "minItems": 1}

    def obj(fields):
        return {"type": "object", "properties": fields, "required": list(fields), "additionalProperties": False}

    return obj({
        "segments": {"type": "array", "minItems": 1, "items": obj({
            "start_frame": integer, "end_frame": integer,
            "stage_id": {"type": ["string", "null"], "enum": [*ids, None]},
            "identity_status": {"type": "string", "enum": ["resolved", "uncertain", "unresolved"]},
            "reason": text, "evidence_frames": evidence})},
        "boundaries": {"type": "array", "items": obj({
            "frame": integer, "lower_frame": integer, "upper_frame": integer,
            "reason": text, "evidence_frames": evidence})},
        "omitted_stages": {"type": "array", "items": obj({
            "stage_id": {"type": "string", "enum": ids}, "reason": text})},
        "invalid_intervals": {"type": "array", "items": obj({
            "start_frame": integer, "end_frame": integer, "reason": text})},
        "summary": text,
    })


def _keys(value, names, label):
    if not isinstance(value, dict) or set(value) != set(names):
        raise ValueError(f"{label} requires exactly {sorted(names)}")


def _integer(value, label):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return value


def _text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be nonempty text")


def _evidence(values, count):
    if (not isinstance(values, list) or not values
            or any(not 0 <= _integer(x, "evidence frame") < count for x in values)
            or len(set(values)) != len(values)):
        raise ValueError("evidence_frames must be distinct real source-frame indices")


def validate_candidate(data, stage_ids, frame_count):
    """Validate exact coverage; do not repair, reorder or merge API output."""
    ids = _registry(stage_ids)
    if _integer(frame_count, "frame_count") < 1:
        raise ValueError("frame_count must be positive")
    _keys(data, ("segments", "boundaries", "omitted_stages", "invalid_intervals", "summary"), "candidate")
    for name in ("segments", "boundaries", "omitted_stages", "invalid_intervals"):
        if not isinstance(data[name], list):
            raise ValueError(f"{name} must be an array")
    _text(data["summary"], "summary")
    cursor, present, cuts = 0, set(), []
    for segment in data["segments"]:
        _keys(segment, ("start_frame", "end_frame", "stage_id", "identity_status", "reason", "evidence_frames"), "segment")
        start, end = (_integer(segment[k], k) for k in ("start_frame", "end_frame"))
        if start != cursor or not 0 <= start < end <= frame_count:
            raise ValueError("segments must partition [0, frame_count) with no gaps or overlaps")
        stage, status = segment["stage_id"], segment["identity_status"]
        if status not in ("resolved", "uncertain", "unresolved") or (stage is None) != (status == "unresolved"):
            raise ValueError("stage_id is null if and only if identity_status is unresolved")
        if stage is not None:
            if stage not in ids:
                raise ValueError("stage_id must belong to the registry")
            present.add(stage)
        _text(segment["reason"], "segment reason")
        _evidence(segment["evidence_frames"], frame_count)
        if start:
            cuts.append(start)
        cursor = end
    if cursor != frame_count:
        raise ValueError("segments must cover every source frame")
    boundary_frames = []
    for boundary in data["boundaries"]:
        _keys(boundary, ("frame", "lower_frame", "upper_frame", "reason", "evidence_frames"), "boundary")
        frame, low, high = (_integer(boundary[k], k) for k in ("frame", "lower_frame", "upper_frame"))
        if not 0 <= low <= frame <= high <= frame_count:
            raise ValueError("boundary bracket must contain its nominal frame within [0,N]")
        boundary_frames.append(frame)
        _text(boundary["reason"], "boundary reason")
        _evidence(boundary["evidence_frames"], frame_count)
    if boundary_frames != cuts:
        raise ValueError("boundaries must exactly match all internal segment cuts in order")
    omitted = []
    for item in data["omitted_stages"]:
        _keys(item, ("stage_id", "reason"), "omitted stage")
        if item["stage_id"] not in ids:
            raise ValueError("omitted stage must belong to the registry")
        omitted.append(item["stage_id"])
        _text(item["reason"], "omission reason")
    if len(set(omitted)) != len(omitted) or set(omitted) != set(ids) - present:
        raise ValueError("omitted_stages must name exactly the absent registry IDs")
    for item in data["invalid_intervals"]:
        _keys(item, ("start_frame", "end_frame", "reason"), "invalid interval")
        start, end = (_integer(item[k], k) for k in ("start_frame", "end_frame"))
        if not 0 <= start < end <= frame_count:
            raise ValueError("invalid interval must lie within the source")
        _text(item["reason"], "invalid interval reason")
    return deepcopy(data)


def _union(intervals):
    merged = []
    for start, end in sorted(intervals):
        if start == end:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(end, merged[-1][1])
        else:
            merged.append([start, end])
    return merged


def coverage(candidate, time_edges):
    """Source-clock duration and union coverage, independently for each axis.

    Boundary rank brackets [low, high] measure the time interval
    [edge[low], edge[high]); an exact boundary has zero uncertain duration.
    This function consumes a previously validated candidate.
    """
    if (not isinstance(time_edges, (list, tuple)) or len(time_edges) < 2
            or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not isfinite(x) for x in time_edges)
            or any(b <= a for a, b in zip(time_edges, time_edges[1:]))):
        raise ValueError("time_edges must contain N+1 finite strictly increasing times")
    count, duration = len(time_edges) - 1, time_edges[-1] - time_edges[0]
    if not candidate["segments"] or candidate["segments"][0]["start_frame"] != 0 or candidate["segments"][-1]["end_frame"] != count:
        raise ValueError("Candidate and source clock disagree")
    axes = {
        "unresolved_identity": [(s["start_frame"], s["end_frame"]) for s in candidate["segments"] if s["identity_status"] == "unresolved"],
        "uncertain_identity": [(s["start_frame"], s["end_frame"]) for s in candidate["segments"] if s["identity_status"] == "uncertain"],
        "boundary_uncertainty": [(b["lower_frame"], b["upper_frame"]) for b in candidate["boundaries"]],
        "source_invalid": [(s["start_frame"], s["end_frame"]) for s in candidate["invalid_intervals"]],
    }
    result = {"source_frames": count, "source_seconds": duration}
    for name, intervals in axes.items():
        if any(not 0 <= a <= b <= count for a, b in intervals):
            raise ValueError("Candidate and source clock disagree")
        spans = _union(intervals)
        frames = sum(b - a for a, b in spans)
        seconds = sum(time_edges[b] - time_edges[a] for a, b in spans)
        result[name] = {"intervals": spans, "frames": frames, "seconds": seconds,
                        "percent_frames": 100 * frames / count, "percent_seconds": 100 * seconds / duration}
    return result


def coalesce_candidate(candidate, stage_ids, frame_count):
    """Explicitly coalesce equal adjacent identities; never repair bad coverage."""
    result = validate_candidate(candidate, stage_ids, frame_count)
    segments, removed = [], set()
    for segment in result["segments"]:
        if segments and all(segments[-1][key] == segment[key] for key in ("stage_id", "identity_status")):
            previous = segments[-1]
            removed.add(segment["start_frame"])
            previous["end_frame"] = segment["end_frame"]
            previous["reason"] += " | " + segment["reason"]
            previous["evidence_frames"] = sorted(set(previous["evidence_frames"] + segment["evidence_frames"]))
        else:
            segments.append(segment)
    result["segments"] = segments
    result["boundaries"] = [b for b in result["boundaries"] if b["frame"] not in removed]
    return validate_candidate(result, stage_ids, frame_count)
