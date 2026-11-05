# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure annotation-to-target policy, independent of video and cache I/O.

The current policy preserves the source xDOF concat quality-aware defaults.
Replace/version ``compute_progress`` when the target definition changes; its
``source_indices`` projection keeps every label aligned with the original RGB.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import accumulate
from math import isfinite
from numbers import Integral, Real


PROGRESS_METHOD = "legacy_concat_quality_v1"
_WEIGHTS = {0: 5, 1: 0.1, 2: 0}
_SCORES = {"high": 1.0, "mid": 0.5, "low": 0.0}


def _quality_progress(quality: list[int], target: float, cap: int) -> list[float]:
    """Legacy stage kernel: 0 normal, 1 suboptimal, 2 failure.

    Nonfailure weights are 5 and .1. Failure resets progress. A successful or
    partial terminal recovery owns its endpoint; an earlier attempt is bounded
    by .9. Low-scored stages cannot recover after their first failure. A final
    failure block with an explicit mid score holds .5.
    """
    count = len(quality)
    if 2 not in quality:
        weights = [_WEIGHTS[q] for q in quality]
        total = sum(weights)
        return list(accumulate(w / total * target for w in weights))

    terminal_start = count
    for index in range(count - 1, -1, -1):
        if quality[index] == 2:
            break
        terminal_start = index
    has_terminal = terminal_start < count and target > 0
    terminal_weight = sum(_WEIGHTS[q] for q in quality[terminal_start:]) if has_terminal else 0
    mid_failure_start = count
    if quality[-1] == 2 and target == 0.5:
        mid_failure_start = count - 1
        while mid_failure_start > 0 and quality[mid_failure_start - 1] == 2:
            mid_failure_start -= 1

    # Failure contributes normal weight to earlier attempts' suffix denominator.
    suffix = [0] * (count + 1)
    for index in range(count - 1, -1, -1):
        suffix[index] = suffix[index + 1] + (5 if quality[index] == 2 else _WEIGHTS[quality[index]])

    def denominator(start: int) -> float:
        return suffix[start] - suffix[min(count, start + cap)]

    attempt_end = terminal_start if has_terminal else count
    denominators = [0.0] * count
    previous = denominator(0)
    index = 0
    while index < attempt_end:
        if quality[index] == 2:
            index += 1
            continue
        start = index
        while index < attempt_end and quality[index] != 2:
            index += 1
        followed_by_failure = index < count and quality[index] == 2
        value = denominator(start) if followed_by_failure else previous
        block_weight = sum(_WEIGHTS[quality[j]] for j in range(start, index))
        value = max(value, block_weight / 0.9)
        denominators[start:index] = [value] * (index - start)
        if followed_by_failure:
            previous = value

    result = []
    accumulated = terminal_accumulated = 0.0
    failure_seen = False
    for index, q in enumerate(quality):
        if q == 2:
            accumulated = 0.0
            failure_seen = True
            value = 0.5 if index >= mid_failure_start else 0.0
        elif failure_seen and target <= 0:
            value = 0.0
        elif has_terminal and index >= terminal_start:
            terminal_accumulated += _WEIGHTS[q]
            value = target if index == count - 1 else target * terminal_accumulated / terminal_weight
        else:
            denom = denominators[index] if denominators[index] > 0 else denominator(0)
            accumulated += _WEIGHTS[q] / denom
            value = accumulated
        result.append(value)
    return result


def _segments(labels: list[str], minimum: int) -> list[tuple[int, int, str]]:
    """Merge each short later annotation run into the immediately prior run."""
    raw = []
    start = 0
    for end in range(1, len(labels) + 1):
        if end == len(labels) or labels[end] != labels[start]:
            raw.append((start, end, labels[start]))
            start = end
    merged = [raw[0]]
    for start, end, label in raw[1:]:
        if end - start < minimum:
            previous_start, _, previous_label = merged[-1]
            merged[-1] = (previous_start, end, previous_label)
        else:
            merged.append((start, end, label))
    return merged


def _outcome(scores: list[str], quality: list[int]) -> tuple[str, float]:
    score = next((s for s in reversed(scores) if s in _SCORES), None)
    has_failure = 2 in quality
    if score is None:
        if quality[0] != 2 and quality[-1] != 2:
            score = "mid" if has_failure else "high"
        else:
            score = "low"
    category = "failure" if score == "low" else "suboptimal" if score == "mid" or has_failure else "successful"
    return category, _SCORES[score]


def _real(value: object, name: str, *, allow_zero: bool = False) -> float:
    if (isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value)
            or (value < 0 if allow_zero else value <= 0)):
        raise ValueError(f"{name} must be finite and {'nonnegative' if allow_zero else 'positive'}")
    return float(value)


def compute_progress(
    *,
    labels: Sequence[str],
    scores: Sequence[str],
    action_quality: Sequence[int],
    stage_prompts: Sequence[str],
    source_fps: float,
    label_map: Mapping[str, str] | None = None,
    progress_cap_seconds: float = 40.0,
    min_segment_frames: int = 10,
    max_trailing_seconds: float | None = None,
) -> dict:
    """Return frame-aligned supervision and its original-frame projection.

    Inputs are complete raw-episode arrays, before frame selection. ``labels``
    and sparse ``scores`` have already been decoded through LeRobot tasks.jsonl.
    Scores are high/mid/low, or empty/'' when absent. The last explicit score in
    each annotation run wins; absent scores use the legacy quality heuristic.
    ``action_quality`` is 0 normal, 1 suboptimal, 2 failure, never task IDs.

    ``label_map`` performs exact renaming. Ordered prompts use the source's
    case-insensitive suffix matching; stage identity may regress. Unmatched
    prefix/interior frames are omitted. Unmatched final frames hold the last
    stage's endpoint, optionally capped in native-source seconds. Short later
    runs merge into the previous run before any of these operations.

    Result arrays use ``source_indices`` coordinates. Whole-task progress is
    (stage_id + within_stage_progress) / number_of_stages. This bounded policy
    excludes source-specific affine ranges, stage subsets and per-episode label
    overrides. Video decoding, resizing and downsampling belong to the caller.
    """
    sequences = {"labels": labels, "scores": scores, "action_quality": action_quality, "stage_prompts": stage_prompts}
    if any(isinstance(value, (str, bytes)) or not isinstance(value, Sequence) for value in sequences.values()):
        raise ValueError("labels, scores, action_quality and stage_prompts must be sequences")
    labels, scores, quality, prompts = list(labels), list(scores), list(action_quality), list(stage_prompts)
    if not labels or len(scores) != len(labels) or len(quality) != len(labels):
        raise ValueError("labels, scores and action_quality must have equal, nonzero lengths")
    if any(not isinstance(value, str) for value in labels + scores):
        raise ValueError("labels and scores must contain decoded strings")
    scores = [score.strip().lower() for score in scores]
    if any(score not in ("", "empty", "high", "mid", "low") for score in scores):
        raise ValueError("scores must be high, mid, low, empty or ''")
    if any(isinstance(q, bool) or not isinstance(q, Integral) or q not in _WEIGHTS for q in quality):
        raise ValueError("action_quality must contain integers 0 normal, 1 suboptimal, 2 failure")
    if (not 1 <= len(prompts) <= 26 or any(not isinstance(p, str) or not p.strip() for p in prompts)
            or len(set(prompts)) != len(prompts)):
        raise ValueError("stage_prompts must contain 1..26 unique nonempty strings")
    normalized = [prompt.lower() for prompt in prompts]
    if any(left.endswith(right) for i, left in enumerate(normalized)
           for j, right in enumerate(normalized) if i != j):
        raise ValueError("stage_prompts have ambiguous case-insensitive suffix matches")
    if label_map is not None and (not isinstance(label_map, Mapping)
            or any(not isinstance(k, str) or not isinstance(v, str) for k, v in label_map.items())
            or "*" in label_map):
        raise ValueError("label_map must map exact strings to stage prompts; wildcard mapping is unsupported")
    rename = dict(label_map or {})
    fps = _real(source_fps, "source_fps")
    cap = int(_real(progress_cap_seconds, "progress_cap_seconds") * fps)
    if cap < 1:
        raise ValueError("progress_cap_seconds must cover at least one source frame")
    if isinstance(min_segment_frames, bool) or not isinstance(min_segment_frames, Integral) or min_segment_frames < 0:
        raise ValueError("min_segment_frames must be a nonnegative integer")
    trailing_cap = None if max_trailing_seconds is None else int(_real(max_trailing_seconds, "max_trailing_seconds", allow_zero=True) * fps)

    def stage_id(label: str) -> int:
        resolved = rename.get(label, label).lower()
        return next((index for index, prompt in enumerate(prompts) if resolved.endswith(prompt.lower())), -1)

    segments = _segments(labels, int(min_segment_frames))
    matches = [(start, end, stage_id(label)) for start, end, label in segments if stage_id(label) >= 0]
    if not matches:
        raise ValueError("No annotation stage matches the configured stage_prompts")
    kept, stages, within = [], [], []
    partials, categories = [0.0] * len(prompts), ["failure"] * len(prompts)
    trailing_count = 0
    for run_index, (start, end, stage) in enumerate(matches):
        category, target = _outcome(scores[start:end], quality[start:end])
        values = _quality_progress(quality[start:end], target, cap)
        run_end = end
        if run_index == len(matches) - 1:
            run_end = len(labels) if trailing_cap is None else min(len(labels), end + trailing_cap)
            trailing_count = run_end - end
            values += [values[-1]] * trailing_count
        kept.extend(range(start, run_end))
        stages.extend([stage] * (run_end - start))
        within.extend(min(max(float(value), 0.0), 1.0) for value in values)
        partials[stage], categories[stage] = target, category
    return {
        "method": PROGRESS_METHOD,
        "source_indices": kept,
        "stage_ids": stages,
        "within_stage_progress": within,
        "progress": [(stage + value) / len(prompts) for stage, value in zip(stages, within, strict=True)],
        "stage_partial_successes": partials,
        "stage_quality_labels": categories,
        "trailing_absorbed_n_frames": trailing_count,
    }
