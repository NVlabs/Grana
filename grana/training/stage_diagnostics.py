# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Consumed-row stage diagnostics, reduced as counts before computing ratios."""
from collections import Counter

COUNT_KEYS = ("c_rows", "augmented_c_rows", "augmented_c_poi_distractor_predictions", "paired_c_rows", "d_rows")
HISTOGRAM_KEYS = ("candidate_count_histogram", "offset_histogram", "choice_target_histogram")


def empty_diagnostics() -> dict:
    return {"schema": "grana-stage-augmentation-diagnostics-v1",
            **dict.fromkeys(COUNT_KEYS, 0), "per_task": {},
            **{key: {} for key in HISTOGRAM_KEYS}, "examples": []}


def record_stage_rows(payload: dict, examples: list, requests: tuple, stage_indices: list,
                      choices: list | None, *, rank: int) -> None:
    """Record model-consumed rows, never queued CPU work or targets in prompts."""
    by_index = dict(zip(stage_indices, choices or (), strict=True))
    for index, (example, request) in enumerate(zip(examples, requests, strict=True)):
        if example.role not in ("C", "D"):
            continue
        task = payload["per_task"].setdefault(example.task, dict.fromkeys(COUNT_KEYS, 0))

        def add(key):
            payload[key] += 1
            task[key] += 1

        if example.role == "D":
            add("d_rows")
            continue
        add("c_rows")
        if request == "C_paired":
            add("paired_c_rows")
        mapping = example.stage_choices
        count = len(mapping.candidate_prompts) if mapping is not None else example.num_stages
        hist = payload["candidate_count_histogram"]
        hist[str(count)] = hist.get(str(count), 0) + 1
        if mapping is not None:
            add("augmented_c_rows")
            if mapping.choice_to_stage[by_index[index]] == -1:
                add("augmented_c_poi_distractor_predictions")
            offsets = payload["offset_histogram"]
            offsets[str(mapping.offset)] = offsets.get(str(mapping.offset), 0) + 1
            targets = payload["choice_target_histogram"]
            for target in example.targets:
                targets[str(target)] = targets.get(str(target), 0) + 1
        # A cycle starts with many paired rows. Reserve replay evidence for
        # augmented and unpaired canonical rows that occur later in the phase.
        category = "augmented" if mapping else ("paired" if request == "C_paired" else "canonical")
        if sum(row["category"] == category for row in payload["examples"]) < 2:
            payload["examples"].append({
                "category": category,
                "rank": rank, "task": example.task, "trajectory_id": example.trajectory_id,
                "lane": request, "frame_indices": example.frame_indices, "poi_slot": example.poi_slot,
                "num_stages": example.num_stages, "stage_answer_count": count,
                "canonical_stage_ids": example.stage_ids, "choice_targets": example.targets,
                "draw_id": mapping.draw_id if mapping else None,
                "offset": mapping.offset if mapping else None,
                "stage_to_choice": mapping.stage_to_choice if mapping else None,
                "choice_to_stage": mapping.choice_to_stage if mapping else None,
            })


def merge_diagnostics(parts: list[dict]) -> dict:
    result = empty_diagnostics()
    for part in parts:
        for key in COUNT_KEYS:
            result[key] += part[key]
        for key in HISTOGRAM_KEYS:
            merged = Counter(result[key])
            merged.update(part[key])
            result[key] = dict(sorted(merged.items()))
        for task, counts in part["per_task"].items():
            merged = result["per_task"].setdefault(task, dict.fromkeys(COUNT_KEYS, 0))
            for key in COUNT_KEYS:
                merged[key] += counts[key]
    # Interleave ranks so the bounded artifact demonstrates multiple workers.
    categories = ("augmented", "paired", "canonical")
    for index in range(2):
        for category in categories:
            for part in parts:
                rows = [row for row in part["examples"] if row["category"] == category]
                if index < len(rows) and len(result["examples"]) < 8:
                    result["examples"].append(rows[index])
    return result
