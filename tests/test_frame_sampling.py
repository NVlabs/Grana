# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections import Counter
import random

import pytest

from grana.data.frame_sampling import (
    FRAME_SELECTION_PROFILES, QUERY_ORDER_PROFILES, FRAME_SELECTION_KEY, QUERY_ORDER_KEY,
    QUERY_ORDERS, normalize_train_frame_sampling, sample_frame_indices,
)
from grana.data.sampling import CTSP9, frame_indices


def draw(poi, size, pair=(40, 4), family="regular_ctsp", order="forward", seed=42):
    return sample_frame_indices(poi, size, *pair, frame_selection_profile=family,
                                query_order_profile=order, rng=random.Random(seed))


def test_default_regular_selection_retains_historical_base_order():
    for size in (1, 2, 7, 8, 40, 300):
        for poi in {0, min(size - 1, 2), size // 2, size - 1}:
            for pair in (*CTSP9, (1, 4), (1000, 2000)):
                actual = draw(poi, size, pair)
                c, t = pair
                past = (poi - 3*c, poi - 2*c, poi - c) if poi >= 3*c else (0, poi//3, 2*poi//3)
                local = tuple(max(0, min(size-1, poi + offset*t)) for offset in (-2, -1, 0, 1, 2))
                assert actual.frame_indices == past + local == frame_indices(poi, size, *pair)
                assert actual.poi_slot == 5 and actual.frame_weight_order == tuple(range(8))
    assert draw(9, 20, (1, 4)).frame_indices == (6, 7, 8, 1, 5, 9, 13, 17)


@pytest.mark.parametrize("family", FRAME_SELECTION_PROFILES)
@pytest.mark.parametrize("order", QUERY_ORDER_PROFILES)
@pytest.mark.parametrize("size", [1, 2, 7, 8, 9, 500])
def test_every_selector_order_preserves_eight_occurrences_and_designated_poi(family, order, size):
    for poi in {0, size // 2, size - 1}:
        for pair in ((40, 4), (1, 1), (1, 4), (10**30, 10**30)):
            base = draw(poi, size, pair, family)
            result = draw(poi, size, pair, family, order)
            permutation = QUERY_ORDERS[order]
            assert result.frame_indices == tuple(base.frame_indices[index] for index in permutation)
            assert result.frame_weight_order == permutation
            assert len(result.frame_indices) == 8 and all(0 <= i < size for i in result.frame_indices)
            assert Counter(result.frame_indices) == Counter(base.frame_indices)
            assert result.poi_slot == permutation.index(base.poi_slot)
            assert result.frame_indices[result.poi_slot] == poi
            c, t = pair
            lower, upper = max(0, poi - max(3*c, 2*t+3)), min(size-1, poi+2*t)
            assert all(lower <= index <= upper for index in result.frame_indices)
            assert (result.frame_selection_profile, result.query_order_profile) == (family, order)


def test_three_order_examples_are_literal_permutations_of_same_frames():
    assert draw(200, 500).frame_indices == (80, 120, 160, 192, 196, 200, 204, 208)
    assert draw(200, 500, order="backward").frame_indices == (208, 204, 200, 196, 192, 160, 120, 80)
    assert draw(200, 500, order="tail_rewind").frame_indices == (80, 120, 160, 208, 204, 200, 196, 192)
    assert draw(200, 500, order="backward").poi_slot == 2


def test_irregular_context_strata_and_local_distances_are_respected():
    outputs = set()
    for seed in range(100):
        result = draw(200, 500, family="irregular_stratified", seed=seed)
        a, b, c, left_far, left_near, poi, right_near, right_far = result.frame_indices
        assert 80 <= a < 117 <= b < 154 <= c < 192
        assert 192 <= left_far < left_near < poi == 200 < right_near < right_far <= 208
        outputs.add(result.frame_indices)
    assert len(outputs) > 90
    assert draw(2, 100, family="irregular_stratified").frame_indices[:3] == (0, 0, 1)


def test_retired_uniform_window_is_not_a_supported_selector_or_configuration_key():
    assert FRAME_SELECTION_PROFILES == ("regular_ctsp", "irregular_stratified")
    with pytest.raises(ValueError, match="Unknown frame_selection_profile"):
        draw(200, 500, family="uniform_window")
    # Even an old disabled lane must fail instead of silently accepting a retired API.
    for weight in (0, 1):
        with pytest.raises(ValueError, match="Unknown configuration key.*uniform_window"):
            normalize_train_frame_sampling({FRAME_SELECTION_KEY: {"uniform_window": weight}})


def test_configuration_weights_are_canonical_and_robust_to_large_finite_values():
    actual = normalize_train_frame_sampling({FRAME_SELECTION_KEY: {
        "irregular_stratified": 1e308, "regular_ctsp": 1e308},
        QUERY_ORDER_KEY: {"tail_rewind": 2, "backward": 1, "forward": 1}})
    assert list(actual[FRAME_SELECTION_KEY]) == list(FRAME_SELECTION_PROFILES)
    assert list(actual[QUERY_ORDER_KEY]) == list(QUERY_ORDER_PROFILES)
    assert tuple(actual[FRAME_SELECTION_KEY].values()) == (.5, .5)
    assert tuple(actual[QUERY_ORDER_KEY].values()) == (.25, .25, .5)


@pytest.mark.parametrize("bad", [None, True, [], "uniform"])
def test_invalid_sampling_mappings_fail(bad):
    with pytest.raises(ValueError, match="train_frame_sampling"):
        normalize_train_frame_sampling(bad)
    with pytest.raises(ValueError, match="frame_selection_profile_weights"):
        normalize_train_frame_sampling({FRAME_SELECTION_KEY: bad})


@pytest.mark.parametrize("bad", [True, False, -1, float("inf"), float("nan"), "1", None, 10**1000])
def test_invalid_selection_weights_fail(bad):
    with pytest.raises(ValueError, match="regular_ctsp"):
        normalize_train_frame_sampling({FRAME_SELECTION_KEY: {"regular_ctsp": bad}})


def test_invalid_or_empty_strategy_support_fails():
    for supplied in ({"profile": "old"}, {FRAME_SELECTION_KEY: {"typo": 1}},
                     {FRAME_SELECTION_KEY: dict.fromkeys(FRAME_SELECTION_PROFILES, 0)},
                     {QUERY_ORDER_KEY: dict.fromkeys(QUERY_ORDER_PROFILES, 0)}):
        with pytest.raises(ValueError):
            normalize_train_frame_sampling(supplied)
