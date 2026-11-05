# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
from pathlib import Path

import pytest

from grana.config import load_config, forward_batch_sizes, supervision_budget
from grana.data.stage_augmentation_config import (normalize_train_stage_augmentation,
    task_stage_aug_enabled, task_may_supply_d, validate_task_overrides)
from grana.training.losses import window_role_counts
from grana.training.prefetch import forward_role_requests, independent_role_order
from grana.training.stage_diagnostics import empty_diagnostics, merge_diagnostics

RECIPE = Path(__file__).resolve().parents[1] / "configs/pretrain.yaml"


@pytest.mark.parametrize("master", [False, True])
@pytest.mark.parametrize("method", [False, True])
@pytest.mark.parametrize("default", [False, True])
def test_task_scope_and_method_gates(master, method, default):
    policy = normalize_train_stage_augmentation({"enabled": master, "default_task_enabled": default,
        "contiguous_shift": {"enabled": method},
        "task_overrides": {"task1": {"enabled": True}, "task2": {"enabled": False}}})
    assert task_stage_aug_enabled(policy, "task1") == (master and method)
    assert task_may_supply_d(policy, "task1") == (not (master and method))
    assert not task_stage_aug_enabled(policy, "task2")
    assert task_may_supply_d(policy, "task2")
    assert task_stage_aug_enabled(policy, "Tractor") == (master and method and default)


@pytest.mark.parametrize("raw", [
    {"typo": False}, {"enabled": 1}, {"default_task_enabled": "true"},
    {"contiguous_shift": None}, {"contiguous_shift": {"enabled": 1}},
    *({"contiguous_shift": {"candidate_count": count}} for count in (True, 0, 27, 10., "10")),
    {"task_overrides": []}, {"task_overrides": {"": {"enabled": True}}},
    {"task_overrides": {"task1": {"enabled": 1}}}, {"task_overrides": {"task1": {"typo": True}}},
])
def test_strict_policy_schema(raw):
    with pytest.raises(ValueError):
        normalize_train_stage_augmentation(raw)


def test_dynamic_task_keys_keep_unknown_key_validation():
    policy = load_config(RECIPE, [
        'data.train_stage_augmentation.task_overrides={"task name": {enabled: true}}',
        "data.train_stage_augmentation.enabled=true", "data.train_stage_augmentation.contiguous_shift.enabled=true",
    ])["data"]["train_stage_augmentation"]
    assert task_stage_aug_enabled(policy, "task name")
    validate_task_overrides(policy, ["task name", "other"])
    with pytest.raises(ValueError, match="Unknown task"):
        validate_task_overrides(policy, ["other"])
    with pytest.raises(ValueError, match="Unknown configuration key"):
        load_config(RECIPE, ["data.train_stage_augmentation.typo=true"])


@pytest.mark.parametrize("workers", [1, 2, 4])
@pytest.mark.parametrize("independent", [0, 1, 2, 7, 8])
@pytest.mark.parametrize("cycles", [1, 2])
def test_exact_budget_including_unequal_forward_sizes(workers, independent, cycles):
    roles = dict(A=16, B=8, C=16 + independent, D=16)
    total = sum(roles.values()) * workers * cycles
    cfg = load_config(RECIPE, [f"roles={roles}", f"training.num_workers={workers}",
                               f"training.global_batch_size={total}"])
    assert cfg["training"]["gradient_accumulation_steps"] == 2 * cycles
    sizes = forward_batch_sizes(cfg)
    assert sizes == (28 + (independent + 1) // 2, 28 + independent // 2)
    phases = forward_role_requests(roles)
    assert tuple(map(len, phases)) == sizes
    actual = {role: sum(request.split('_')[0] == role for phase in phases for request in phase) * cycles
              for role in roles}
    assert window_role_counts(roles, 2 * cycles) == actual
    budget = supervision_budget(cfg)
    assert budget["global_batch_size"] == total
    assert budget["role_rows_per_update_global"] == {role: rows * workers for role, rows in actual.items()}
    assert budget["physical_batch_size_per_worker"] == max(sizes)
    stream = independent_role_order(roles, 2 * cycles)
    assert stream.count("C_paired") == 16 * cycles
    assert stream.count("C_independent") == independent * cycles
    assert "D" not in stream


def test_bs128_mixed_contract_and_impossible_budget():
    cfg = load_config(RECIPE, ["roles={A: 16, B: 8, C: 24, D: 16}", "training.num_workers=2"])
    assert forward_batch_sizes(cfg) == (32, 32)
    assert supervision_budget(cfg)["role_rows_per_update_global"] == dict(A=32, B=16, C=48, D=32)
    with pytest.raises(ValueError, match="complete even forward cycles"):
        load_config(RECIPE, ["roles={A: 16, B: 8, C: 23, D: 16}", "training.num_workers=2"])
    with pytest.raises(ValueError, match="C>=D"):
        load_config(RECIPE, ["roles={A: 0, B: 2, C: 1, D: 2}"])


def test_diagnostics_reduce_counts_not_rank_ratios():
    left, right = empty_diagnostics(), empty_diagnostics()
    left.update(c_rows=10, augmented_c_rows=1, augmented_c_poi_distractor_predictions=1)
    right.update(c_rows=30, augmented_c_rows=9, augmented_c_poi_distractor_predictions=0)
    left["offset_histogram"] = {"0": 1}
    right["offset_histogram"] = {"0": 2, "3": 7}
    original = copy.deepcopy([left, right])
    result = merge_diagnostics([left, right])
    assert result["augmented_c_rows"] / result["c_rows"] == .25
    assert result["augmented_c_poi_distractor_predictions"] / result["augmented_c_rows"] == .1
    assert result["offset_histogram"] == {"0": 3, "3": 7}
    assert [left, right] == original
