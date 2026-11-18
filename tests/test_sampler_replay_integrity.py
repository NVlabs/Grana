# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Replay binds sampling metadata without touching pixels or frame RNGs."""
from dataclasses import replace
import random

import numpy as np
import pytest

from grana.data.cache import Trajectory
from grana.data.sampling import RoleSampler
from types import SimpleNamespace


def _dataset(records):
    by_role = {role: {task: tuple(row for row in records if row.task == task and role in row.roles)
                      for task in sorted({row.task for row in records if role in row.roles})} for role in "ABC"}
    return SimpleNamespace(records=tuple(records), by_role=by_role,
                           task_weights={role: dict.fromkeys(by_role[role], 1.) for role in "ABC"})


@pytest.fixture
def inventory(tmp_path):
    pixels = np.arange(31 * 2 * 3 * 3, dtype=np.uint8).reshape((31, 2, 3, 3))
    path = tmp_path / "frames.npy"
    np.save(path, pixels)
    records = []
    for task, count in (("task1", 3), ("task2", 8), ("tractor", 5)):
        records.append(Trajectory(tmp_path, task + ":trajectory", task, 31, (path,), pixels.shape,
                                  ("top",), tuple("ABC"), tuple(np.linspace(0, 1, 31)),
                                  tuple(index * count // 31 for index in range(31)),
                                  tuple((index % 5) / 4 for index in range(31)),
                                  tuple(f"{task} action {index}" for index in range(count)),
                                  f"Perform {task}."))
    records.append(replace(records[0], task="a-only", trajectory_id="a-only", roles=("A",),
                           stage_prompts=(), stage_ids=(), within_stage_progress=()))
    return _dataset(records)


def sampler(data):
    return RoleSampler(data, seed=193, worker_rank=2)


@pytest.mark.parametrize("lane", ["C", "C_paired", "C_independent"])
def test_native_stage_targets_and_d_pairing_cover_all_eligible_tasks(inventory, lane):
    source = sampler(inventory)
    source.validate_role_budget({"C": 1, "D": 1})
    seen = set()
    for _ in range(40):
        c = source.sample(lane)
        seen.add(c.task)
        assert c.targets == c.stage_ids
        assert all(0 <= target < c.num_stages for target in c.targets)
        focus = (c.stage_ids[c.poi_slot] + 1) % c.num_stages
        d = source.make_d(c, focus)
        assert d.focus_stage == focus
        assert d.images is c.images and d.frame_indices is c.frame_indices
    assert seen == set(inventory.by_role["C"])


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
        assert (left.task, left.trajectory_id, left.frame_indices, left.targets, left.prompt) == (
            right.task, right.trajectory_id, right.frame_indices, right.targets, right.prompt)
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
