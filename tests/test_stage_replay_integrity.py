# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Replay binds sampling metadata without touching pixels or mapping RNGs."""
from dataclasses import replace
import random

import numpy as np
import pytest

from grana.data.cache import Trajectory
from grana.data.sampling import RoleSampler, materialize_example
from grana.data.stage_augmentation import StageChoiceMapping
from test_stage_augmentation import inventory, _dataset, _policy


def sampler(data):
    return RoleSampler(data, {"train_stage_augmentation": _policy("task1")}, seed=193, worker_rank=2)


@pytest.fixture
def replay_inventory(inventory):
    return _dataset((*inventory.records, replace(inventory.records[0], trajectory_id="task1:extra")))


def replace_first(data, **changes):
    return _dataset((replace(data.records[0], **changes), *data.records[1:]))


def test_replay_digest_preserves_same_inventory_sequence_without_pixel_reads(replay_inventory, monkeypatch):
    first = sampler(replay_inventory)
    order = ("A", "C_paired", "C_independent", "B", "C_independent") * 3
    for request in order:
        first.sample(request)
    rng_before = random.getstate()
    with monkeypatch.context() as patch:
        patch.setattr(Trajectory, "images", lambda *a, **k: pytest.fail("snapshot decoded images"))
        patch.setattr(np, "load", lambda *a, **k: pytest.fail("snapshot reread frame data"))
        snapshot = first.state_dict()
        copy = sampler(_dataset(tuple(replace(row) for row in replay_inventory.records)))
        copy.load_state_dict(snapshot)
        assert copy.state_dict()["replay_inventory_sha256"] == snapshot["replay_inventory_sha256"]
    assert random.getstate() == rng_before
    assert len(snapshot["replay_inventory_sha256"]) == 64
    for request in order:
        left, right = first.sample(request), copy.sample(request)
        assert (left.task, left.trajectory_id, left.frame_indices, left.targets, left.prompt, left.stage_choices) == (
            right.task, right.trajectory_id, right.frame_indices, right.targets, right.prompt, right.stage_choices)
        assert [image.tobytes() for image in left.images] == [image.tobytes() for image in right.images]


@pytest.mark.parametrize("change", ["weights", "task_order", "trajectory_order", "record_order",
                                   "stage_order", "trajectory_id", "cache_root", "frame_paths",
                                   "frame_geometry", "num_frames", "views", "roles", "task_prompt",
                                   "progress", "stage_ids", "within_stage_progress"])
def test_replay_rejects_changed_sampling_law_or_loaded_metadata(replay_inventory, change):
    original = sampler(replay_inventory)
    for _ in range(5):
        original.sample("C_independent")
    snapshot = original.state_dict()
    changed = _dataset(tuple(replace(row) for row in replay_inventory.records))
    row = changed.records[0]
    if change == "weights":
        changed.task_weights["C"]["task1"] = 1000.
    elif change == "task_order":
        changed.by_role["C"] = dict(reversed(tuple(changed.by_role["C"].items())))
    elif change == "trajectory_order":
        changed.by_role["C"]["task1"] = tuple(reversed(changed.by_role["C"]["task1"]))
    elif change == "record_order":
        changed.records = tuple(reversed(changed.records))
    elif change == "stage_order":
        changed = replace_first(changed, stage_prompts=tuple(reversed(row.stage_prompts)))
    elif change == "trajectory_id":
        changed = replace_first(changed, trajectory_id="new-identity")
    elif change == "cache_root":
        changed = replace_first(changed, cache_root=row.cache_root / "different-cache")
    elif change == "frame_paths":
        changed = replace_first(changed, frame_paths=(row.cache_root / "different-frames.npy",))
    elif change == "frame_geometry":
        changed = replace_first(changed, frame_shape=(row.num_frames, 4, 5, 3))
    elif change == "num_frames":
        changed = replace_first(changed, num_frames=30, frame_shape=(30, 2, 3, 3),
                                progress=row.progress[:30], stage_ids=row.stage_ids[:30],
                                within_stage_progress=row.within_stage_progress[:30])
    elif change == "views":
        changed = replace_first(changed, view_names=("side",))
    elif change == "roles":
        changed = replace_first(changed, roles=("B", "C"))
    elif change == "task_prompt":
        changed = replace_first(changed, compressed_task_prompt="A changed whole-task instruction.")
    elif change == "progress":
        changed = replace_first(changed, progress=(.5,) * row.num_frames)
    elif change == "stage_ids":
        changed = replace_first(changed, stage_ids=tuple(reversed(row.stage_ids)))
    elif change == "within_stage_progress":
        changed = replace_first(changed, within_stage_progress=(.5,) * row.num_frames)
    restored = sampler(changed)
    # In particular, real-stage permutations do not change the unordered donor
    # digest, so the new replay inventory check must catch them independently.
    if change in ("weights", "stage_order", "task_order", "trajectory_order"):
        assert restored.stage_distractor_pool.digest == original.stage_distractor_pool.digest
    before = {role: rng.getstate() for role, rng in restored._rng.items()}
    with pytest.raises(ValueError, match="configuration does not match"):
        restored.load_state_dict(snapshot)
    assert {role: rng.getstate() for role, rng in restored._rng.items()} == before
    assert restored._draw_ordinals == dict.fromkeys(restored._streams, 0)


def test_replay_detects_in_place_weight_change_without_cached_digest(replay_inventory):
    current = sampler(replay_inventory)
    snapshot = current.state_dict()
    replay_inventory.task_weights["C"]["task1"] = 1000.
    with pytest.raises(ValueError, match="configuration does not match"):
        current.load_state_dict(snapshot)


@pytest.mark.parametrize("weight", [0, -1, True, float("nan"), float("inf"), "1"])
def test_replay_digest_rejects_nonpositive_or_invalid_sampling_weights(replay_inventory, weight):
    current = sampler(replay_inventory)
    replay_inventory.task_weights["C"]["task1"] = weight
    with pytest.raises(ValueError, match="finite and positive"):
        current.state_dict()


def test_materialize_rejects_mapping_that_claims_the_wrong_true_stage_block_before_images(inventory, monkeypatch):
    current = sampler(inventory)
    draws = (current.sample("C_independent") for _ in range(100))
    actual = next(row for row in draws if row.stage_choices is not None and row.stage_choices.offset > 0)
    mapping = actual.stage_choices
    count = actual.num_stages
    forged = StageChoiceMapping(
        candidate_prompts=mapping.candidate_prompts,
        stage_to_choice=tuple(range(count)),
        choice_to_stage=tuple(range(count)) + (-1,) * (len(mapping.candidate_prompts) - count),
        offset=0, pool_digest=mapping.pool_digest,
        algorithm_version=mapping.algorithm_version, draw_id=mapping.draw_id)
    source = next(row for row in inventory.records if row.trajectory_id == actual.trajectory_id)
    monkeypatch.setattr(Trajectory, "images", lambda *a, **k: pytest.fail("invalid mapping read images"))
    with pytest.raises(ValueError, match="preserve the canonical task stages before materialization"):
        materialize_example(source, "C", actual.poi, actual.stride_pair,
                            execution_mode="train", stage_choices=forged)
