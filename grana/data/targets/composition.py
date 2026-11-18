# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Grana's explicitly parameterized sparse-annotation target construction.

This policy is separate from ``progress.py``'s historical cache semantics.
The discrete reset convention starts a stage at its first observed timestamp
(zero initial update); recovery starts at the last regressed timestamp.
No synthetic time interval is introduced.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import accumulate
from math import fsum, isfinite
from numbers import Integral, Real

PROGRESS_METHOD = "grana_target_composition"
_OUTCOMES = {"high": 1.0, "mid": 0.5, "low": 0.0}
_UNSCORED = {"", "empty", "leading", "trailing"}


def _number(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def validate_stage_parameters(stage_parameters, num_stages: int) -> list[dict]:
    """Require explicit parameters for every ordered stage; infer no calibration."""
    if isinstance(num_stages, bool) or not isinstance(num_stages, Integral) or not 1 <= num_stages <= 26:
        raise ValueError("num_stages must be an integer in 1..26")
    if (not isinstance(stage_parameters, (list, tuple)) or len(stage_parameters) != num_stages):
        raise ValueError("stage_parameters must contain one parameter object per stage")
    result = []
    for index, item in enumerate(stage_parameters):
        if not isinstance(item, Mapping) or set(item) != {"rho", "p_cap", "t_min_seconds"}:
            raise ValueError(f"stage_parameters[{index}] requires exactly rho, p_cap, t_min_seconds")
        values = {key: _number(item[key], f"stage_parameters[{index}].{key}") for key in item}
        if values["rho"] <= 1 or not 0 < values["p_cap"] < 1 or values["t_min_seconds"] <= 0:
            raise ValueError("Stage parameters require rho > 1, 0 < p_cap < 1, t_min_seconds > 0")
        result.append(values)
    return result


def compute_target_composition(
    *, labels: Sequence[str], scores: Sequence[str], action_quality: Sequence[int],
    stage_prompts: Sequence[str], label_map: Mapping[str, str], source_fps: float,
    timestamps: Sequence[float], stage_parameters: Sequence[dict],
) -> dict:
    """Construct native-frame local targets, then compose ``(stage_id+p)/K``.

    Every contiguous raw stage-label run requires an explicit high/mid/low
    outcome. Distinct outcomes in one run, unknown labels, and unlabelled gaps
    between stages are rejected. Prefix/suffix unscored frames are omitted.
    Nominal/suboptimal updates remain in one chunk, weighted rho/1. Regressed
    frames are always zero; each following chunk gets a fresh reset anchor.

    Case A uses the next regressed frame's timestamp to set its duration and
    ``p_cap * min(1, duration/t_min_seconds)`` endpoint. Case B ends at its last
    observed stage frame and uses that stage run's annotated outcome. Both
    normalize by the sum of non-regressed weighted observed time increments.
    A positive endpoint with zero observed duration is undefined and rejected.
    """
    arrays = {"labels": labels, "scores": scores, "action_quality": action_quality,
              "stage_prompts": stage_prompts, "timestamps": timestamps}
    if any(isinstance(value, (str, bytes)) or not isinstance(value, Sequence) for value in arrays.values()):
        raise ValueError("Labels, scores, quality, prompts and timestamps must be sequences")
    labels, scores, quality, prompts, times = (list(arrays[key]) for key in arrays)
    if not labels or any(len(value) != len(labels) for value in (scores, quality, times)):
        raise ValueError("Labels, scores, action_quality and timestamps need equal nonzero lengths")
    if (not 1 <= len(prompts) <= 26 or any(not isinstance(p, str) or not p.strip() for p in prompts)
            or len(set(prompts)) != len(prompts)):
        raise ValueError("stage_prompts must contain 1..26 distinct nonempty descriptions")
    parameters = validate_stage_parameters(stage_parameters, len(prompts))
    if not isinstance(label_map, Mapping) or any(not isinstance(k, str) or v not in prompts for k, v in label_map.items()):
        raise ValueError("label_map must map exact text labels to configured stage_prompts")
    if any(key in _UNSCORED for key in label_map):
        raise ValueError("Unscored prefix/suffix labels cannot be mapped to a stage")
    if any(not isinstance(value, str) for value in labels + scores):
        raise ValueError("labels and scores must contain decoded strings")
    scores = [score.strip().lower() for score in scores]
    if any(score not in {"", "empty", *_OUTCOMES} for score in scores):
        raise ValueError("scores must be empty/high/mid/low")
    if any(isinstance(q, bool) or not isinstance(q, Integral) or q not in (0, 1, 2) for q in quality):
        raise ValueError("action_quality must be 0 nominal, 1 suboptimal, or 2 regressed")
    fps = _number(source_fps, "source_fps")
    if fps <= 0:
        raise ValueError("source_fps must be positive")
    times = [_number(value, "timestamp") for value in times]
    if any(right <= left for left, right in zip(times, times[1:])):
        raise ValueError("timestamps must be strictly increasing")
    prompt_ids = {prompt: index for index, prompt in enumerate(prompts)}
    stage_ids = []
    for label in labels:
        resolved = label_map.get(label, label)
        if resolved in prompt_ids:
            stage_ids.append(prompt_ids[resolved])
        elif label in _UNSCORED:
            stage_ids.append(-1)
        else:
            raise ValueError(f"Unknown raw stage label: {label!r}")
    selected = [index for index, stage in enumerate(stage_ids) if stage >= 0]
    if not selected:
        raise ValueError("No annotated stage frames were found")
    first, last = selected[0], selected[-1]
    if any(stage < 0 for stage in stage_ids[first:last + 1]):
        raise ValueError("Unlabelled gaps between annotated stage runs are unsupported")

    within, kept_stages, chunks, stage_runs = [], [], [], []
    partials, categories = [0.0] * len(prompts), ["failure"] * len(prompts)
    run_start = first
    while run_start <= last:
        run_end = run_start + 1
        while run_end <= last and labels[run_end] == labels[run_start]:
            run_end += 1
        stage = stage_ids[run_start]
        outcomes = {score for score in scores[run_start:run_end] if score in _OUTCOMES}
        if len(outcomes) != 1:
            reason = "missing explicit outcome" if not outcomes else "conflicting explicit outcomes"
            raise ValueError(f"Stage run [{run_start}, {run_end}) has {reason}")
        outcome = _OUTCOMES[next(iter(outcomes))]
        params = parameters[stage]
        values = [0.0] * (run_end - run_start)
        index = run_start
        run_index = len(stage_runs)
        while index < run_end:
            if quality[index] == 2:
                index += 1
                continue
            start = index
            while index < run_end and quality[index] != 2:
                index += 1
            end = index
            anchor = times[run_start] if start == run_start else times[start - 1]
            interrupted = end < run_end
            end_time = times[end] if interrupted else times[end - 1]
            duration = end_time - anchor
            endpoint = params["p_cap"] * min(1.0, duration / params["t_min_seconds"]) if interrupted else outcome
            updates = []
            previous = anchor
            for frame in range(start, end):
                updates.append((params["rho"] if quality[frame] == 0 else 1.0) * (times[frame] - previous))
                previous = times[frame]
            try:
                weighted_duration = fsum(updates)
            except OverflowError as error:
                raise ValueError("Weighted chunk duration exceeds finite numeric range") from error
            if not isfinite(duration) or not isfinite(weighted_duration):
                raise ValueError("Weighted chunk duration exceeds finite numeric range")
            if weighted_duration <= 0 and endpoint > 0:
                raise ValueError(f"Chunk [{start}, {end}) has a positive endpoint but zero observed update duration")
            if weighted_duration > 0:
                local = [endpoint * total / weighted_duration for total in accumulate(updates)]
                local[-1] = endpoint
                values[start - run_start:end - run_start] = local
            chunks.append({"stage_id": stage, "stage_run_index": run_index,
                           "frame_start": start, "frame_end": end,
                           "case": "A_interrupted" if interrupted else "B_boundary",
                           "anchor_time": anchor, "end_time": end_time, "duration_seconds": duration,
                           "weighted_duration": weighted_duration, "endpoint": endpoint,
                           "outcome": outcome, **params})
        category = ("failure" if outcome == 0 else "suboptimal"
                    if outcome < 1 or 2 in quality[run_start:run_end] else "successful")
        partials[stage], categories[stage] = outcome, category
        stage_runs.append({"stage_id": stage, "frame_start": run_start, "frame_end": run_end,
                           "outcome": outcome, "quality_label": category})
        within.extend(values)
        kept_stages.extend([stage] * len(values))
        run_start = run_end
    return {"method": PROGRESS_METHOD, "source_indices": selected, "stage_ids": kept_stages,
            "within_stage_progress": within,
            "progress": [(stage + value) / len(prompts) for stage, value in zip(kept_stages, within, strict=True)],
            "stage_partial_successes": partials, "stage_quality_labels": categories,
            "stage_runs": stage_runs, "chunks": chunks, "source_fps": fps,
            "trailing_absorbed_n_frames": 0}
