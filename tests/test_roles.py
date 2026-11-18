# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace
from types import SimpleNamespace
import random

import numpy as np
import pytest

from grana.data.cache import Trajectory
from grana.modeling.prompts import PROGRESS_MARKER, STAGE_MARKER, stage_prompt
from grana.data.sampling import CTSP9, RoleSampler, frame_indices, materialize_example, stage_relative_targets
from grana.data.frame_sampling import DEFAULT_TRAIN_STRIDE_PAIRS, FRAME_SELECTION_PROFILES, QUERY_ORDER_PROFILES, QUERY_ORDERS


@pytest.fixture
def dataset(tmp_path):
    views = ("top", "left", "right", "bottom")
    paths = []
    for view_index, view in enumerate(views):
        path = tmp_path / f"{view}.npy"
        pixels = np.zeros((7, 2, 3, 3), dtype=np.uint8)
        for frame in range(7):
            pixels[frame] = view_index * 20 + frame
        np.save(path, pixels)
        paths.append(path)
    row = Trajectory(tmp_path, "id1", "task", 7, tuple(paths), (7, 2, 3, 3), views, ("A", "B", "C"),
                     (0., .1, .2, .3, .4, .8, 1.), (0, 0, 1, 1, 1, 2, 2),
                     (0., .7, .1, .3, .9, .2, 1.), ("First operation", "Second operation", "Third operation"),
                     "The complete canonical task.")
    return SimpleNamespace(by_role={role: {"task": (row,)} for role in "ABC"}, records=(row,),
                           task_weights={role: {"task": 1.0} for role in "ABC"})


def test_exact_forward_geometry_and_literal_poi_slot():
    assert frame_indices(120, 200, 40, 4) == (0, 40, 80, 112, 116, 120, 124, 128)
    assert frame_indices(119, 200, 40, 4) == (0, 39, 79, 111, 115, 119, 123, 127)
    assert frame_indices(0, 1, 40, 4) == (0,) * 8
    assert frame_indices(1, 3, 40, 4) == (0, 0, 0, 0, 0, 1, 2, 2)
    # Sparse context can overlap the dense tail; never deduplicate or sort.
    assert frame_indices(9, 20, 1, 4) == (6, 7, 8, 1, 5, 9, 13, 17)
    for size in range(1, 15):
        for poi in range(size):
            for pair in CTSP9:
                indices = frame_indices(poi, size, *pair)
                assert len(indices) == 8 and indices[5] == poi
                assert all(0 <= index < size for index in indices)


def test_stage_relative_direction_and_off_focus_wsp_validation():
    assert stage_relative_targets((0, 1, 2, 1), (.4, .3, .7, .8), 1, 3) == (0., .3, 1., .8)
    with pytest.raises(ValueError, match="finite"):
        stage_relative_targets((0, 2), (float("nan"), .7), 1, 3)
    with pytest.raises(ValueError, match="focus"):
        stage_relative_targets((0, 2), (.2, .7), True, 3)


def test_a_uses_cached_global_progress_and_b_uses_poi_focus(dataset):
    sampler = RoleSampler(dataset, {"stride_pairs": [[40, 4]]}, seed=12)
    a = sampler.sample("A")
    source = dataset.records[0]
    assert a.targets == tuple(source.progress[index] for index in a.frame_indices)
    assert "The complete canonical task." in a.prompt
    assert "First operation" not in a.prompt
    b = sampler.sample("B")
    assert b.focus_stage == source.stage_ids[b.poi]
    assert b.targets == stage_relative_targets(b.stage_ids, b.within_stage_progress, b.focus_stage, b.num_stages)


def test_c_has_dense_labels_and_no_focus_or_answers_in_messages(dataset):
    c = RoleSampler(dataset, seed=3).sample("C")
    assert c.focus_stage is None
    assert c.targets == tuple(dataset.records[0].stage_ids[index] for index in c.frame_indices)
    content = c.messages()[0]["content"]
    assert len([item for item in content if item["type"] == "image"]) == 32
    assert sum(item.get("text") == STAGE_MARKER for item in content) == 8
    assert not any(item.get("text") == PROGRESS_MARKER for item in content)
    assert "Focus stage" not in c.prompt and "DONE" not in c.prompt
    assert all(f"{letter}: Stage {index}:" in c.prompt for index, letter in enumerate("ABC"))
    assert "select the final stage" in c.prompt
    # Full view groups precede each marker; each camera keeps its source pixels.
    for slot in range(8):
        for view_index in range(4):
            assert np.asarray(c.images[slot * 4 + view_index])[0, 0, 0] == view_index * 20 + c.frame_indices[slot]
    changed_labels = replace(c, targets=(2,) * 8, stage_ids=(2,) * 8)
    assert changed_labels.messages() == c.messages()


def test_d_reuses_exact_c_observations_and_changes_only_progress_query(dataset):
    sampler = RoleSampler(dataset, seed=14)
    c = sampler.sample("C")
    focus = (c.stage_ids[c.poi_slot] + 1) % c.num_stages
    d = sampler.make_d(c, focus)
    assert d.role == "D" and d.focus_stage == focus
    assert d.images is c.images and d.frame_indices is c.frame_indices
    assert all(a is b for a, b in zip(c.images, d.images, strict=True))
    assert d.stride_pair is c.stride_pair and d.poi == c.poi
    assert d.targets == stage_relative_targets(c.stage_ids, c.within_stage_progress, focus, c.num_stages)
    assert sum(x.get("text") == PROGRESS_MARKER for x in d.messages()[0]["content"]) == 8
    with pytest.raises(ValueError, match="derive"):
        sampler.sample("D")
    with pytest.raises(ValueError, match="unfocused C"):
        sampler.make_d(sampler.sample("B"), 0)
    c.images[0].putpixel((0, 0), (255, 255, 255))
    with pytest.raises(ValueError, match="observations changed"):
        sampler.make_d(c, focus)


def test_independent_role_streams_and_stride_invariance(dataset):
    first, second = RoleSampler(dataset, seed=4), RoleSampler(dataset, seed=4)
    second.sample("A")
    second.sample("B")
    assert first.sample("C").frame_indices == second.sample("C").frame_indices
    fixed = RoleSampler(dataset, {"stride_pairs": [[40, 4]]}, seed=4)
    varying = RoleSampler(dataset, seed=4)
    for _ in range(30):
        a, b = fixed.sample("A"), varying.sample("A")
        assert (a.trajectory_id, a.poi) == (b.trajectory_id, b.poi)


def test_task_uniform_sampling_before_trajectory_uniform_sampling(dataset):
    original = dataset.records[0]
    many = tuple(replace(original, task="many", trajectory_id=f"many{i}") for i in range(50))
    small = replace(original, task="small", trajectory_id="small")
    dataset.by_role["A"] = {"many": many, "small": (small,)}
    dataset.task_weights["A"] = {"many": 1.0, "small": 1.0}
    sampler = RoleSampler(dataset, seed=42)
    counts = {"many": 0, "small": 0}
    for _ in range(400):
        counts[sampler.sample("A").task] += 1
    assert 160 < counts["small"] < 240


def test_task_weights_apply_before_trajectory_selection(dataset):
    original = dataset.records[0]
    many = tuple(replace(original, task="many", trajectory_id=f"many{i}") for i in range(50))
    small = replace(original, task="small", trajectory_id="small")
    dataset.by_role["A"] = {"many": many, "small": (small,)}
    dataset.task_weights["A"] = {"many": 1.0, "small": 3.0}
    sampler = RoleSampler(dataset, seed=42)
    small_count = sum(sampler.sample("A").task == "small" for _ in range(400))
    assert 270 < small_count < 330


def test_native_choices_and_reserved_prompt_markers():
    with pytest.raises(ValueError, match="1..26"):
        stage_prompt("task", ())
    with pytest.raises(ValueError, match="reserved"):
        stage_prompt("task", ("bad <|stage_token|>", "two"))


def sampling_config(family="regular_ctsp", order="forward"):
    return {"train_frame_sampling": {
        "frame_selection_profile_weights": {name: float(name == family) for name in FRAME_SELECTION_PROFILES},
        "query_order_profile_weights": {name: float(name == order) for name in QUERY_ORDER_PROFILES}}}


def test_explicit_historical_stride_support_matches_original_rng_images_labels_and_fixed_evaluation(dataset):
    config = {**sampling_config(), "stride_pairs": [list(pair) for pair in CTSP9]}
    sampler = RoleSampler(dataset, config, seed=42)
    sources = {role: random.Random(f"grana:42:{role}:selection") for role in "ABC"}
    strides = {role: random.Random(f"grana:42:{role}:stride") for role in "ABC"}
    for _ in range(20):
        for role in "CAB":
            rng = sources[role]
            task = rng.choice(tuple(dataset.by_role[role]))
            trajectory = rng.choice(dataset.by_role[role][task])
            poi, pair = rng.randrange(trajectory.num_frames), strides[role].choice(CTSP9)
            expected = materialize_example(trajectory, role, poi, pair)
            actual = sampler.sample(role)
            assert (actual.frame_indices, actual.targets, actual.stage_ids, actual.within_stage_progress,
                    actual.focus_stage, actual.prompt, actual.observation_digest) == (
                        expected.frame_indices, expected.targets, expected.stage_ids, expected.within_stage_progress,
                        expected.focus_stage, expected.prompt, expected.observation_digest)
            assert actual.poi == poi and actual.poi_slot == 5 and actual.stride_pair == pair
            assert actual.frame_weight_order == tuple(range(8))
            assert [image.tobytes() for image in actual.images] == [image.tobytes() for image in expected.images]


def test_sampler_without_explicit_stride_uses_four_training_pairs(dataset):
    assert DEFAULT_TRAIN_STRIDE_PAIRS == ((10, 2), (16, 4), (24, 6), (32, 8))
    implicit = RoleSampler(dataset, seed=42)
    explicit = RoleSampler(dataset, {"stride_pairs": [list(pair) for pair in DEFAULT_TRAIN_STRIDE_PAIRS]}, seed=42)
    assert implicit.stride_pairs == explicit.stride_pairs == DEFAULT_TRAIN_STRIDE_PAIRS
    observed = set()
    for role in "ABC" * 10:
        actual, expected = implicit.sample(role), explicit.sample(role)
        observed.add(actual.stride_pair)
        assert actual.stride_pair == expected.stride_pair
        assert (actual.poi, actual.frame_indices, actual.targets) == (expected.poi, expected.frame_indices, expected.targets)
    assert observed == set(DEFAULT_TRAIN_STRIDE_PAIRS)


@pytest.mark.parametrize("family", FRAME_SELECTION_PROFILES)
@pytest.mark.parametrize("order", QUERY_ORDER_PROFILES)
def test_sampler_materializes_aligned_views_labels_and_d_metadata(dataset, family, order):
    base_sampler = RoleSampler(dataset, sampling_config(family), seed=17)
    sampler = RoleSampler(dataset, sampling_config(family, order), seed=17)
    for role in "ABC":
        for _ in range(6):
            base, actual = base_sampler.sample(role), sampler.sample(role)
            permutation = QUERY_ORDERS[order]
            assert actual.poi == base.poi and actual.frame_weight_order == permutation
            assert actual.targets == tuple(base.targets[index] for index in permutation)
            assert actual.stage_ids == tuple(base.stage_ids[index] for index in permutation)
            assert actual.within_stage_progress == tuple(base.within_stage_progress[index] for index in permutation)
            assert actual.frame_indices == tuple(base.frame_indices[index] for index in permutation)
            assert actual.focus_stage == base.focus_stage
            assert (actual.frame_selection_profile, actual.query_order_profile) == (family, order)
            for slot, base_slot in enumerate(permutation):
                for view in range(4):
                    assert actual.images[slot*4+view].tobytes() == base.images[base_slot*4+view].tobytes()
            if role == "C":
                d = sampler.make_d(actual, (actual.stage_ids[actual.poi_slot] + 1) % actual.num_stages)
                assert d.images is actual.images and d.frame_indices is actual.frame_indices
                assert d.poi_slot == actual.poi_slot and d.frame_weight_order is actual.frame_weight_order
                assert (d.frame_selection_profile, d.query_order_profile) == (family, order)


def test_selection_changes_preserve_source_stride_and_order_streams(dataset):
    samplers = []
    for family in FRAME_SELECTION_PROFILES:
        config = sampling_config(family)
        config["train_frame_sampling"]["query_order_profile_weights"] = dict.fromkeys(QUERY_ORDER_PROFILES, 1.)
        samplers.append(RoleSampler(dataset, config, seed=91))
    for _ in range(40):
        examples = [sampler.sample("C") for sampler in samplers]
        assert len({(x.task, x.trajectory_id, x.poi, x.stride_pair, x.query_order_profile) for x in examples}) == 1


def test_order_changes_preserve_selection_and_do_not_touch_global_randomness(dataset):
    samplers = []
    before = random.getstate()
    for order in QUERY_ORDER_PROFILES:
        config = sampling_config(order=order)
        config["train_frame_sampling"]["frame_selection_profile_weights"] = dict.fromkeys(FRAME_SELECTION_PROFILES, 1.)
        samplers.append(RoleSampler(dataset, config, seed=97))
    for _ in range(40):
        examples = [sampler.sample("B") for sampler in samplers]
        bases = [tuple(example.frame_indices[example.frame_weight_order.index(i)] for i in range(8)) for example in examples]
        assert bases[0] == bases[1] == bases[2]
        assert len({(x.poi, x.stride_pair, x.frame_selection_profile) for x in examples}) == 1
    assert random.getstate() == before


def test_sampler_snapshot_restores_all_random_streams_and_is_independent(dataset):
    config = {"train_frame_sampling": {
        "frame_selection_profile_weights": dict.fromkeys(FRAME_SELECTION_PROFILES, 1.),
        "query_order_profile_weights": dict.fromkeys(QUERY_ORDER_PROFILES, 1.)}}
    sampler = RoleSampler(dataset, config, seed=74)
    for role in "BAC":
        sampler.sample(role)
    snapshot = sampler.state_dict()
    first = [sampler.sample(role) for role in "ABCABCABC"]
    sampler.load_state_dict(snapshot)
    repeated = [sampler.sample(role) for role in "ABCABCABC"]
    for a, b in zip(first, repeated, strict=True):
        assert (a.frame_indices, a.poi_slot, a.targets, a.frame_selection_profile, a.query_order_profile) == (
            b.frame_indices, b.poi_slot, b.targets, b.frame_selection_profile, b.query_order_profile)
    with pytest.raises(ValueError, match="configuration"):
        RoleSampler(dataset, seed=74).load_state_dict(snapshot)
    snapshot["train_frame_sampling"]["frame_selection_profile_weights"]["regular_ctsp"] = 999
    assert sampler.train_frame_sampling["frame_selection_profile_weights"]["regular_ctsp"] < 1


@pytest.mark.parametrize("field,value", [
    ("poi_slot", True), ("poi_slot", -1), ("poi_slot", 8), ("poi_slot", 1.5),
    ("frame_weight_order", (0, 1, 2, 3, 4, 5, 6, 6)),
    ("frame_weight_order", (False, 1, 2, 3, 4, 5, 6, 7)),
    ("frame_weight_order", (0, 1, 2, 3, 4, 5, 6, 8)),
    ("frame_selection_profile", "legacy"), ("frame_selection_profile", "uniform_window"),
    ("query_order_profile", "reverse"),
])
def test_role_example_rejects_invalid_sampling_metadata(dataset, field, value):
    example = RoleSampler(dataset).sample("C")
    with pytest.raises(ValueError, match=field):
        replace(example, **{field: value})
