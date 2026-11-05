# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit reviewed stage + AQ/outcome sidecars, never inferred reward labels."""
import hashlib
import json
from pathlib import Path

from .schema import validate_candidate
from .source import LeRobotSource


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be explicit nonempty text")


def _keys(value, fields, name):
    if not isinstance(value, dict) or set(value) != set(fields):
        raise ValueError(f"{name} requires exactly {sorted(fields)}")


def _integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _read(path):
    raw = Path(path).read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def read_bridge_annotations(source, views, episode_index, stages, specification):
    """Return the same native arrays as preprocess._episode_annotations.

    Revalidate actual source bytes and clocks, the accepted revision, the exact
    registry and the independent supplement before producing any dense labels.
    Consecutive same-ID candidate fragments form one maximal raw stage run.
    """
    _keys(specification, ("revision", "supplement"), "annotation specification")
    revision, revision_hash = _read(specification["revision"])
    supplement, supplement_hash = _read(specification["supplement"])
    _keys(revision, ("candidate", "candidate_sha256", "review", "source_identity", "stages", "time_edges", "created_at"), "revision")
    _keys(supplement, ("source_identity", "reviewer", "provenance", "boundary_policy", "aq_intervals", "outcomes"), "reward supplement")
    episode = LeRobotSource(Path(source), views).load_episode(episode_index)
    if revision["source_identity"] != episode.source_identity or supplement["source_identity"] != episode.source_identity:
        raise ValueError("Revision/supplement source_identity must exactly match the current source")
    edges = revision["time_edges"]
    if not isinstance(edges, list) or edges != list(episode.time_edges) or any(isinstance(x, bool) for x in edges):
        raise ValueError("Revision time_edges must exactly match the native source clock")
    registry = revision["stages"]
    if not isinstance(registry, list) or len(registry) != len(stages):
        raise ValueError("Revision registry must match configured stages")
    for item, configured in zip(registry, stages, strict=True):
        _keys(item, ("id", "label", "prompt"), "revision stage")
        _text(item["id"], "stage id")
        if {key: item[key] for key in ("label", "prompt")} != configured:
            raise ValueError("Revision registry order/labels/prompts must exactly match configured stages")
    ids = [stage["id"] for stage in registry]
    candidate = validate_candidate(revision["candidate"], ids, episode.frame_count)
    canonical = json.dumps(candidate, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    if revision["candidate_sha256"] != hashlib.sha256(canonical).hexdigest():
        raise ValueError("Revision candidate_sha256 does not match its candidate")
    review = revision["review"]
    _keys(review, ("reviewer", "kind", "decision", "notes", "evidence_inspected"), "review")
    if review["kind"] not in ("human", "agent") or review["decision"] != "accepted":
        raise ValueError("Reward export requires an explicitly accepted human or agent review")
    for key in ("reviewer", "notes"):
        _text(review[key], f"review.{key}")
    _text(revision["created_at"], "revision.created_at")
    inspected = review["evidence_inspected"]
    if not isinstance(inspected, list) or not inspected:
        raise ValueError("Review must describe the evidence inspected")
    for item in inspected:
        _text(item, "review.evidence_inspected entry")
    if len(set(inspected)) != len(inspected):
        raise ValueError("Review evidence descriptions must be unique")
    if any(segment["identity_status"] != "resolved" for segment in candidate["segments"]):
        raise ValueError("Reward export cannot fill unresolved or uncertain stage identity")
    if candidate["invalid_intervals"]:
        raise ValueError("Reward export cannot train on source-invalid intervals")
    if supplement["boundary_policy"] != "nominal_dense":
        raise ValueError("Reward export requires explicit nominal_dense boundary policy; masking is unsupported")
    for key in ("reviewer", "provenance"):
        _text(supplement[key], f"supplement.{key}")

    quality = []
    quality_values = {"nominal": 0, "suboptimal": 1, "regressed": 2}
    if not isinstance(supplement["aq_intervals"], list):
        raise ValueError("aq_intervals must explicitly cover every native frame")
    for interval in supplement["aq_intervals"]:
        _keys(interval, ("start_frame", "end_frame", "quality", "reason"), "AQ interval")
        start, end = (_integer(interval[key], key) for key in ("start_frame", "end_frame"))
        if start != len(quality) or not 0 <= start < end <= episode.frame_count:
            raise ValueError("AQ intervals must partition the native source with no gaps or overlaps")
        if interval["quality"] not in quality_values:
            raise ValueError("AQ quality must be nominal, suboptimal or regressed")
        _text(interval["reason"], "AQ evidence reason")
        quality.extend([quality_values[interval["quality"]]] * (end - start))
    if len(quality) != episode.frame_count:
        raise ValueError("AQ intervals must explicitly cover every native frame")

    runs = []
    for segment in candidate["segments"]:
        if runs and runs[-1]["stage_id"] == segment["stage_id"]:
            runs[-1]["end_frame"] = segment["end_frame"]
        else:
            runs.append({key: segment[key] for key in ("start_frame", "end_frame", "stage_id")})
    outcomes = supplement["outcomes"]
    if not isinstance(outcomes, list) or len(outcomes) != len(runs):
        raise ValueError("Outcomes must cover every maximal contiguous stage run exactly")
    labels, scores = [], [""] * episode.frame_count
    label_for_id = {stage["id"]: stage["label"] for stage in registry}
    for run, outcome in zip(runs, outcomes, strict=True):
        _keys(outcome, ("start_frame", "end_frame", "stage_id", "outcome", "reason"), "stage outcome")
        for key in ("start_frame", "end_frame"):
            _integer(outcome[key], key)
        if {key: outcome[key] for key in run} != run:
            raise ValueError("Outcome identities and boundaries must exactly match stage runs")
        if outcome["outcome"] not in ("high", "mid", "low"):
            raise ValueError("Every stage run needs an explicit high/mid/low outcome")
        _text(outcome["reason"], "outcome evidence reason")
        labels.extend([label_for_id[run["stage_id"]]] * (run["end_frame"] - run["start_frame"]))
        scores[run["end_frame"] - 1] = outcome["outcome"]
    provenance = {"annotation_source": "reviewed_stage_and_explicit_reward_sidecars_v1",
                  "quality_source": "explicit_supplement.aq_intervals", "quality_disagreement_frames": None,
                  "stage_revision": str(Path(specification["revision"]).resolve()), "stage_revision_sha256": revision_hash,
                  "reward_supplement": str(Path(specification["supplement"]).resolve()), "reward_supplement_sha256": supplement_hash,
                  "candidate_sha256": revision["candidate_sha256"], "stage_review": review,
                  "supplement_reviewer": supplement["reviewer"], "supplement_provenance": supplement["provenance"],
                  "boundary_policy": "nominal_dense", "boundary_brackets": candidate["boundaries"],
                  "source_identity": episode.source_identity, "clock_receipt": episode.clock_receipt}
    return labels, scores, quality, list(episode.timestamps), provenance
