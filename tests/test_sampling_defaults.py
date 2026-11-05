# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Independent checks for four train strides, two selectors and three orders."""
from collections import Counter
from math import sqrt
from pathlib import Path
from types import SimpleNamespace

import pytest

from grana.config import load_config
from grana.data.frame_sampling import FRAME_SELECTION_PROFILES, QUERY_ORDER_PROFILES
from grana.data.sampling import RoleSampler

TRAIN_PAIRS = ((10, 2), (16, 4), (24, 6), (32, 8))


@pytest.fixture
def index_only_dataset(monkeypatch):
    # Exercise the real RoleSampler request/RNG/geometry path without RGB I/O.
    # View/target materialization and C->D inheritance are covered in test_roles.
    def materialize(trajectory, role, poi, pair, *, frame_decision, execution_mode, stage_choices):
        assert execution_mode == "train" and stage_choices is None
        return SimpleNamespace(trajectory_id=trajectory.trajectory_id, role=role, poi=poi,
                               stride_pair=pair, **vars(frame_decision))

    monkeypatch.setattr("grana.data.sampling.materialize_example", materialize)
    rows = [SimpleNamespace(trajectory_id=f"trajectory-{index}", task="task", num_frames=301 + index)
            for index in range(3)]
    return SimpleNamespace(records=tuple(rows), by_role={role: {"task": rows} for role in "ABC"},
                           task_weights={role: {"task": 1.0} for role in "ABC"})


def sampler(dataset, selection, order, *, stride_pairs=None):
    config = {"train_frame_sampling": {
        "frame_selection_profile_weights": dict(zip(FRAME_SELECTION_PROFILES, selection, strict=True)),
        "query_order_profile_weights": dict(zip(QUERY_ORDER_PROFILES, order, strict=True))}}
    if stride_pairs is not None:
        config["stride_pairs"] = [list(pair) for pair in stride_pairs]
    return RoleSampler(dataset, config, seed=73)


def source_identity(example):
    return example.trajectory_id, example.poi, example.stride_pair


def base_frames(example):
    return tuple(example.frame_indices[example.frame_weight_order.index(i)] for i in range(8))


@pytest.mark.parametrize("role", "ABC")
@pytest.mark.parametrize("config_source", ["explicit", "python-defaults", "pretrain", "posttrain"])
def test_mixture_has_product_joint_law(index_only_dataset, role, config_source):
    if config_source == "explicit":
        selection, order = (.8, .2), (.1, .3, .6)
        selected = sampler(index_only_dataset, selection, order)
    else:
        selection, order = (.6, .4), (.5, .2, .3)
        config = {} if config_source == "python-defaults" else load_config(
            Path(__file__).resolve().parents[1] / "configs" / f"{config_source}.yaml")["data"]
        # This index-only fixture isolates frame sampling. Real phase recipes
        # with stage-choice materialization are covered in test_stage_augmentation.
        config.setdefault("train_stage_augmentation", {})["enabled"] = False
        selected = RoleSampler(index_only_dataset, config, seed=73)
    draws = 6000
    counts, stride_counts, joint_counts = Counter(), Counter(), Counter()
    for _ in range(draws):
        example = selected.sample(role)
        counts[example.frame_selection_profile, example.query_order_profile] += 1
        stride_counts[example.stride_pair] += 1
        joint_counts[example.stride_pair, example.frame_selection_profile, example.query_order_profile] += 1
    assert set(counts) == {(family, order_name) for family in FRAME_SELECTION_PROFILES
                           for order_name in QUERY_ORDER_PROFILES}
    assert set(stride_counts) == set(TRAIN_PAIRS)
    assert len(joint_counts) == 4 * 2 * 3
    for pair in TRAIN_PAIRS:
        assert abs(stride_counts[pair] - draws / 4) <= 6 * sqrt(draws * .25 * .75)
    for family, selection_probability in zip(FRAME_SELECTION_PROFILES, selection, strict=True):
        for order_name, order_probability in zip(QUERY_ORDER_PROFILES, order, strict=True):
            probability = selection_probability * order_probability
            # Six binomial standard deviations: deterministic seed, while rejecting
            # coupled selector/order draws or replacement with a uniform mixture.
            assert abs(counts[family, order_name] - draws * probability) <= 6 * sqrt(draws * probability * (1 - probability))
            for pair in TRAIN_PAIRS:
                joint_probability = probability / 4
                assert abs(joint_counts[pair, family, order_name] - draws * joint_probability) <= 6 * sqrt(draws * joint_probability * (1 - joint_probability))


@pytest.mark.parametrize("role", "ABC")
def test_changing_order_mixture_preserves_each_selected_occurrence(index_only_dataset, role):
    first = sampler(index_only_dataset, (.3, .7), (.1, .3, .6))
    second = sampler(index_only_dataset, (.3, .7), (.55, .35, .1))
    observed_families, changed_orders = set(), 0
    for _ in range(100):
        left, right = first.sample(role), second.sample(role)
        assert source_identity(left) == source_identity(right)
        assert left.frame_selection_profile == right.frame_selection_profile
        assert base_frames(left) == base_frames(right)
        observed_families.add(left.frame_selection_profile)
        changed_orders += left.query_order_profile != right.query_order_profile
    assert observed_families == set(FRAME_SELECTION_PROFILES)
    assert changed_orders > 0


@pytest.mark.parametrize("role", "ABC")
def test_changing_selection_mixture_preserves_each_order_draw(index_only_dataset, role):
    first = sampler(index_only_dataset, (.9, .1), (.2, .2, .6))
    second = sampler(index_only_dataset, (.2, .8), (.2, .2, .6))
    observed_orders, changed_families = set(), 0
    for _ in range(100):
        left, right = first.sample(role), second.sample(role)
        assert source_identity(left) == source_identity(right)
        assert left.query_order_profile == right.query_order_profile
        assert left.frame_weight_order == right.frame_weight_order
        observed_orders.add(left.query_order_profile)
        changed_families += left.frame_selection_profile != right.frame_selection_profile
        if left.frame_selection_profile == right.frame_selection_profile:
            assert left.frame_indices == right.frame_indices
    assert observed_orders == set(QUERY_ORDER_PROFILES)
    assert changed_families > 0


@pytest.mark.parametrize("role", "ABC")
def test_changing_stride_support_preserves_source_selection_and_order_draws(index_only_dataset, role):
    samplers = [sampler(index_only_dataset, (.3, .7), (.2, .2, .6), stride_pairs=pairs)
                for pairs in (None, ((16, 4),), tuple(reversed(TRAIN_PAIRS)))]
    observed_default_pairs = set()
    changed_geometry = False
    for _ in range(100):
        examples = [selected.sample(role) for selected in samplers]
        assert len({(x.trajectory_id, x.poi, x.frame_selection_profile, x.query_order_profile)
                    for x in examples}) == 1
        assert len({x.frame_weight_order for x in examples}) == 1
        observed_default_pairs.add(examples[0].stride_pair)
        assert examples[1].stride_pair == (16, 4)
        for other in examples[1:]:
            if examples[0].stride_pair == other.stride_pair:
                assert examples[0].frame_indices == other.frame_indices
            elif examples[0].frame_indices != other.frame_indices:
                changed_geometry = True
    assert observed_default_pairs == set(TRAIN_PAIRS)
    assert changed_geometry
