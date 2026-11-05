# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict, task-specific policy shared by training and canonical evaluation.

Every function accepts the ``data.train_stage_augmentation`` section, not the
whole data recipe. Applying augmentation and permitting Role D are separate
decisions: evaluation is canonical but retains the current job's D eligibility.
"""
from __future__ import annotations

from collections.abc import Collection
import copy


def default_train_stage_augmentation() -> dict:
    return {"enabled": False,
            "contiguous_shift": {"enabled": False, "candidate_count": 10},
            "default_task_enabled": False, "task_overrides": {}}


def normalize_train_stage_augmentation(raw: dict | None = None) -> dict:
    result = default_train_stage_augmentation()
    if raw is None:
        return result
    if not isinstance(raw, dict):
        raise ValueError("data.train_stage_augmentation must be a mapping")
    unknown = set(raw) - set(result)
    if unknown:
        raise ValueError(f"Unknown train_stage_augmentation keys: {sorted(unknown)}")
    for key in ("enabled", "default_task_enabled"):
        if key in raw:
            if not isinstance(raw[key], bool):
                raise ValueError(f"train_stage_augmentation.{key} must be boolean")
            result[key] = raw[key]
    method = raw.get("contiguous_shift", {})
    if not isinstance(method, dict):
        raise ValueError("train_stage_augmentation.contiguous_shift must be a mapping")
    unknown = set(method) - {"enabled", "candidate_count"}
    if unknown:
        raise ValueError(f"Unknown contiguous_shift keys: {sorted(unknown)}")
    result["contiguous_shift"].update(method)
    if not isinstance(result["contiguous_shift"]["enabled"], bool):
        raise ValueError("contiguous_shift.enabled must be boolean")
    count = result["contiguous_shift"]["candidate_count"]
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 26:
        raise ValueError("contiguous_shift.candidate_count must be an integer in 1..26")
    overrides = raw.get("task_overrides", {})
    if not isinstance(overrides, dict):
        raise ValueError("train_stage_augmentation.task_overrides must be a mapping")
    for task, override in overrides.items():
        if not isinstance(task, str) or not task.strip():
            raise ValueError("task_overrides keys must be nonempty task names")
        if not isinstance(override, dict) or set(override) != {"enabled"}:
            raise ValueError(f"task_overrides[{task!r}] must contain only enabled")
        if not isinstance(override["enabled"], bool):
            raise ValueError(f"task_overrides[{task!r}].enabled must be boolean")
    result["task_overrides"] = copy.deepcopy(overrides)
    return result


def task_stage_aug_enabled(policy: dict, task: str) -> bool:
    """Resolve an already-normalized policy; never gate this on model.training."""
    requested = policy["task_overrides"].get(task, {}).get(
        "enabled", policy["default_task_enabled"])
    return bool(policy["enabled"] and policy["contiguous_shift"]["enabled"] and requested)


def task_may_supply_d(policy: dict, task: str) -> bool:
    return not task_stage_aug_enabled(policy, task)


def validate_task_overrides(policy: dict, task_names: Collection[str]) -> None:
    unknown = set(policy["task_overrides"]) - set(task_names)
    if unknown:
        raise ValueError(f"Unknown task_overrides tasks in training inventory: {sorted(unknown)}")
