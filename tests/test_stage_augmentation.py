# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import FrozenInstanceError, replace
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import pytest

from grana.config import load_config
from grana.data.cache import CachedTrajectories, Trajectory
from grana.data.frame_sampling import FRAME_SELECTION_PROFILES, QUERY_ORDER_PROFILES
from grana.data.sampling import RoleSampler, materialize_example, stage_answer_count
from grana.data.stage_augmentation import (ALGORITHM_VERSION, StageDistractorPool,
                                         make_stage_choice_mapping, normalized_stage_text)


def _policy(*tasks, candidate_count=10):
    return {"enabled": True, "contiguous_shift": {"enabled": True, "candidate_count": candidate_count},
            "default_task_enabled": False, "task_overrides": {task: {"enabled": True} for task in tasks}}


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


@pytest.mark.parametrize("phase", ["pretrain", "posttrain"])
@pytest.mark.parametrize("mode", ["train", "eval", "inference"])
def test_recommended_phase_policy_augments_only_pretraining_c(inventory, phase, mode):
    recipe = Path(__file__).resolve().parents[1] / "configs" / f"{phase}.yaml"
    config = load_config(recipe)
    sampler = RoleSampler(inventory, config["data"], seed=42, execution_mode=mode)
    sampler.validate_role_budget(config["roles"])
    seen_c_tasks = set()
    for role in "ABC" * 30:
        sample = sampler.sample(role)
        if role == "C":
            seen_c_tasks.add(sample.task)
        if phase == "pretrain" and mode == "train" and role == "C":
            assert stage_answer_count(sample) == 10
            assert sample.stage_choices is not None
            assert sample.targets == tuple(sample.stage_choices.stage_to_choice[stage]
                                           for stage in sample.stage_ids)
            assert all(sample.task not in donors for donors in sample.stage_choices.candidate_source_tasks)
        else:
            assert sample.stage_choices is None
            if role == "C":
                assert sample.targets == sample.stage_ids
    assert seen_c_tasks == {"task1", "task2", "tractor"}


def test_pretraining_recipe_checks_donor_capacity_and_accepts_task_exclusions(inventory):
    recipe = Path(__file__).resolve().parents[1] / "configs/pretrain.yaml"
    config = load_config(recipe)
    with pytest.raises(ValueError, match="other-training-task distractors"):
        RoleSampler(_dataset([inventory.records[0]]), config["data"])
    excluded = load_config(recipe, [
        "data.train_stage_augmentation.task_overrides={task1: {enabled: false}}"])
    sampler = RoleSampler(inventory, excluded["data"], seed=42)
    seen = set()
    for _ in range(30):
        sample = sampler.sample("C")
        seen.add(sample.task)
        assert (sample.stage_choices is not None) == (sample.task != "task1")
    assert seen == {"task1", "task2", "tractor"}


def _make(pool, task, stages, n=10, *, seed=42, ordinal=0, rank=0, lane="C_independent"):
    return make_stage_choice_mapping(task, stages, n, pool_digest=pool.digest,
                                     donors=pool.eligible(task, stages, n), seed=seed,
                                     worker_rank=rank, lane=lane, draw_ordinal=ordinal)


@pytest.mark.parametrize("count", [1, 3, 8, 10])
def test_all_offsets_fill_complete_choices_and_preserve_real_stage_order(inventory, count):
    task = "synthetic"
    stages = tuple(f"Real stage {index}" for index in range(count))
    pool = StageDistractorPool.from_records(inventory.records)
    observed = set()
    for ordinal in range(300):
        mapping = _make(pool, task, stages, ordinal=ordinal)
        observed.add(mapping.offset)
        assert len(mapping.candidate_prompts) == 10
        assert len({normalized_stage_text(value) for value in mapping.candidate_prompts}) == 10
        assert mapping.stage_to_choice == tuple(range(mapping.offset, mapping.offset + count))
        assert mapping.candidate_prompts[mapping.offset:mapping.offset + count] == stages
        for choice, stage in enumerate(mapping.choice_to_stage):
            if stage == -1:
                assert mapping.candidate_source_tasks[choice]
                assert task not in mapping.candidate_source_tasks[choice]
            else:
                assert mapping.stage_to_choice[stage] == choice
                assert not mapping.candidate_source_tasks[choice]
    assert observed == set(range(11 - count))


def test_def_mapping_pads_abc_and_ghij(inventory):
    pool = StageDistractorPool.from_records(inventory.records)
    row = inventory.records[0]
    candidates = (_make(pool, row.task, row.stage_prompts, ordinal=ordinal) for ordinal in range(100))
    mapping = next(mapping for mapping in candidates if mapping.offset == 3)
    assert mapping.choice_to_stage == (-1, -1, -1, 0, 1, 2, -1, -1, -1, -1)
    assert all(mapping.candidate_prompts[index] for index in (0, 1, 2, 6, 7, 8, 9))
    with pytest.raises(FrozenInstanceError):
        mapping.offset = 0
    with pytest.raises(ValueError, match="immutable"):
        replace(mapping, candidate_source_tasks=[])


def test_uniform_start_draws_do_not_imply_uniform_target_letters(inventory):
    pool = StageDistractorPool.from_records(inventory.records)
    row = inventory.records[0]
    offsets = [0] * 8
    for ordinal in range(2400):
        offsets[_make(pool, row.task, row.stage_prompts, ordinal=ordinal).offset] += 1
    assert all(230 < count < 370 for count in offsets)
    # A fixed first-stage target has support A..H, not a misleading uniform A..J claim.
    assert all(_make(pool, row.task, row.stage_prompts, ordinal=i).stage_to_choice[0] <= 7 for i in range(100))


def test_pool_uses_unique_training_text_and_stable_source_provenance():
    rows = [SimpleNamespace(task="current", stage_prompts=("Real one", "Real two")),
            SimpleNamespace(task="donor-b", stage_prompts=("  Cafe\u0301   step  ", "Real one", "Other")),
            SimpleNamespace(task="donor-a", stage_prompts=("Café step", "Other")),
            SimpleNamespace(task="no-stages", stage_prompts=())]
    pool = StageDistractorPool.from_records(rows)
    assert pool == StageDistractorPool.from_records(list(reversed(rows)) + [rows[1]] * 20)
    eligible = pool.eligible("current", rows[0].stage_prompts, 4)
    assert len(eligible) == 2
    assert {normalized_stage_text(item.prompt) for item in eligible} == {"Café step", "Other"}
    assert all(item.source_tasks == ("donor-a", "donor-b") for item in eligible)
    assert all("held-out" not in item.source_tasks for item in eligible)
    with pytest.raises(ValueError, match="unique other-training-task"):
        pool.eligible("current", rows[0].stage_prompts, 5)
    # A description present solely in the current task is not an eligible donor.
    assert all(item.prompt not in rows[0].stage_prompts for item in eligible)


@pytest.mark.parametrize("count", [0, 11])
def test_invalid_real_stage_capacity_fails(inventory, count):
    pool = StageDistractorPool.from_records(inventory.records)
    with pytest.raises(ValueError, match="require 1 <= K <= N"):
        pool.eligible("invalid", tuple(f"Stage {index}" for index in range(count)), 10)


def test_duplicate_real_text_is_rejected_without_merging_stages(inventory):
    pool = StageDistractorPool.from_records(inventory.records)
    with pytest.raises(ValueError, match="duplicate normalized real"):
        pool.eligible("ambiguous", ("Café step", "  Cafe\u0301    step  "), 10)


def test_pool_preflight_does_not_load_images_and_rejects_shortage(inventory, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("preflight must not read pixels")
    monkeypatch.setattr(Trajectory, "images", forbidden)
    sampler = RoleSampler(inventory, {"train_stage_augmentation": _policy("task1")})
    assert sampler.stage_distractor_pool.digest
    only_current = _dataset([inventory.records[0]])
    with pytest.raises(ValueError, match="other-training-task distractors"):
        RoleSampler(only_current, {"train_stage_augmentation": _policy("task1")})


@pytest.mark.parametrize("selector", ["include_ids", "exclude_ids", "hash_split"])
def test_filtered_cache_inventory_is_the_only_donor_source(inventory, tmp_path, selector):
    from datasets import Dataset

    root = tmp_path / "filtered-cache"
    root.mkdir()
    rows = []
    for record in inventory.records[:3]:
        rows.append({"id": record.trajectory_id, "task": record.task, "num_frames": record.num_frames,
                     "frames_shape": list(record.frame_shape), "frames": str(record.frame_paths[0]),
                     "stage_prompts": list(record.stage_prompts), "stage_ids": list(record.stage_ids),
                     "within_stage_progress": list(record.within_stage_progress)})
    Dataset.from_list(rows).save_to_disk(str(root / "processed_dataset"))
    spec = {"path": str(root), "roles": ["B", "C"]}
    if selector == "include_ids":
        spec[selector] = [record.trajectory_id for record in inventory.records[:2]]
    elif selector == "exclude_ids":
        spec[selector] = [inventory.records[2].trajectory_id]
    else:
        spec[selector] = {"seed": "stage-donor-isolation", "eval_count": 1, "subset": "train"}
    train = CachedTrajectories([spec], views=["top"])
    assert len(train.records) == 2
    pool = StageDistractorPool.from_records(train.records)
    included_tasks = {record.task for record in train.records}
    heldout = next(record for record in inventory.records[:3] if record.task not in included_tasks)
    observed_sources = {task for _, sources in pool.entries for task, _ in sources}
    observed_text = {text for _, sources in pool.entries for _, text in sources}
    assert observed_sources == included_tasks
    assert not observed_text.intersection(heldout.stage_prompts)
    # The text index is defined entirely by selected records, not the cache
    # directory, which also contains the held-out row and its stage metadata.
    assert pool == StageDistractorPool.from_records(tuple(record for record in inventory.records[:3]
                                                          if record.task in included_tasks))


def test_k_equals_n_needs_no_donor_task(inventory):
    only_current = _dataset([inventory.records[0]])
    sampler = RoleSampler(only_current, {"train_stage_augmentation": _policy("task1", candidate_count=3)})
    sample = sampler.sample("C")
    assert sample.stage_choices.offset == 0
    assert sample.stage_choices.candidate_prompts == sample.stage_prompts
    assert sample.stage_choices.choice_to_stage == (0, 1, 2)


def test_augmentation_does_not_consume_task_poi_stride_or_global_rng(inventory):
    before = random.getstate()
    baseline = RoleSampler(inventory, seed=17)
    augmented = RoleSampler(inventory, {"train_stage_augmentation": _policy("task1", "tractor")}, seed=17)
    seen_augmented = set()
    for request in ("A", "B", "C") * 30:
        left, right = baseline.sample(request), augmented.sample(request)
        assert (left.task, left.trajectory_id, left.poi, left.frame_indices, left.stride_pair, left.stage_ids,
                left.stage_prompts, left.num_stages, left.within_stage_progress, left.frame_weight_order,
                left.observation_digest) == (
                    right.task, right.trajectory_id, right.poi, right.frame_indices, right.stride_pair, right.stage_ids,
                    right.stage_prompts, right.num_stages, right.within_stage_progress, right.frame_weight_order,
                    right.observation_digest)
        assert [image.tobytes() for image in left.images] == [image.tobytes() for image in right.images]
        if request == "C" and right.task in {"task1", "tractor"}:
            seen_augmented.add(right.task)
            assert right.stage_choices is not None and stage_answer_count(right) == 10
            assert right.targets == tuple(stage + right.stage_choices.offset for stage in left.targets)
        else:
            assert right.stage_choices is None
            assert right.prompt == left.prompt and right.targets == left.targets
    assert seen_augmented == {"task1", "tractor"}
    assert random.getstate() == before


@pytest.mark.parametrize("family", FRAME_SELECTION_PROFILES)
@pytest.mark.parametrize("order", QUERY_ORDER_PROFILES)
def test_eight_slots_share_mapping_for_all_frame_choices_and_orders(inventory, family, order):
    config = {"train_stage_augmentation": _policy("task1", "task2", "tractor"),
              "train_frame_sampling": {
                  "frame_selection_profile_weights": {name: float(name == family) for name in FRAME_SELECTION_PROFILES},
                  "query_order_profile_weights": {name: float(name == order) for name in QUERY_ORDER_PROFILES}}}
    sampler = RoleSampler(inventory, config, seed=23)
    duplicated = False
    for _ in range(20):
        example = sampler.sample("C")
        duplicated |= len(set(example.frame_indices)) < 8
        assert example.targets == tuple(example.stage_choices.stage_to_choice[stage] for stage in example.stage_ids)
        assert example.poi == example.frame_indices[example.poi_slot]
        assert example.frame_selection_profile == family and example.query_order_profile == order
        for i in range(8):
            for j in range(8):
                if example.frame_indices[i] == example.frame_indices[j]:
                    assert example.targets[i] == example.targets[j]
    assert duplicated


def test_mapping_and_prompt_do_not_depend_on_gt_stage(inventory):
    original = RoleSampler(inventory, {"train_stage_augmentation": _policy("task1", "task2", "tractor")}, seed=35)
    different_labels = _dataset([replace(row, stage_ids=(0,) * row.num_frames) if row.stage_ids else row
                                 for row in inventory.records])
    changed = RoleSampler(different_labels, {"train_stage_augmentation": _policy("task1", "task2", "tractor")}, seed=35)
    for _ in range(10):
        left, right = original.sample("C"), changed.sample("C")
        assert left.stage_choices == right.stage_choices
        assert left.prompt == right.prompt
        assert left.frame_indices == right.frame_indices


def test_pool_changes_do_not_change_offset_seed(inventory):
    pool = StageDistractorPool.from_records(inventory.records)
    expanded = StageDistractorPool.from_records(inventory.records + (
        SimpleNamespace(task="new-donor", stage_prompts=("Previously unseen donor description",)),))
    row = inventory.records[0]
    assert pool.digest != expanded.digest
    for ordinal in range(30):
        assert _make(pool, row.task, row.stage_prompts, ordinal=ordinal).offset == _make(
            expanded, row.task, row.stage_prompts, ordinal=ordinal).offset


@pytest.mark.parametrize("mode", ["eval", "inference"])
def test_eval_and_inference_never_augment_but_task_d_policy_survives(inventory, mode):
    sampler = RoleSampler(inventory, {"train_stage_augmentation": _policy("task1")}, execution_mode=mode)
    assert sampler.stage_distractor_pool is None
    seen = set()
    for _ in range(20):
        c = sampler.sample("C")
        seen.add(c.task)
        assert c.stage_choices is None and c.targets == c.stage_ids
        assert stage_answer_count(c) == c.num_stages
        if c.task == "task1":
            with pytest.raises(ValueError, match="forbidden"):
                sampler.make_d(c, 0)
        else:
            assert sampler.make_d(c, 0).role == "D"
    assert seen == {"task1", "task2", "tractor"}
    mapping = _make(StageDistractorPool.from_records(inventory.records), "task1", inventory.records[0].stage_prompts)
    with pytest.raises(ValueError, match="training Role C"):
        materialize_example(inventory.records[0], "C", 8, (16, 4), execution_mode=mode, stage_choices=mapping)


def test_unknown_overrides_rejected_in_training_only(inventory):
    config = {"train_stage_augmentation": _policy("missing-train-task")}
    with pytest.raises(ValueError, match="Unknown task_overrides"):
        RoleSampler(inventory, config)
    assert RoleSampler(inventory, config, execution_mode="eval").stage_distractor_pool is None


@pytest.mark.parametrize("mode", ["eval", "inference"])
def test_metadata_only_evaluation_inventory_keeps_d_policy(inventory, mode):
    dataset = SimpleNamespace(records=inventory.records)
    sampler = RoleSampler(dataset, {"train_stage_augmentation": _policy("task1")}, execution_mode=mode)
    assert sampler.stage_distractor_pool is None
    allowed = materialize_example(inventory.records[1], "C", 5, (16, 4), execution_mode=mode)
    assert sampler.make_d(allowed, 0).role == "D"
    blocked = materialize_example(inventory.records[0], "C", 5, (16, 4), execution_mode=mode)
    with pytest.raises(ValueError, match="forbidden"):
        sampler.make_d(blocked, 0)


def test_paired_lane_filters_tasks_before_weighted_sampling(inventory):
    inventory.task_weights["C"].update(task1=100., task2=1., tractor=3.)
    sampler = RoleSampler(inventory, {"train_stage_augmentation": _policy("task1")}, seed=81)
    sampler.validate_role_budget(dict(A=16, B=8, C=24, D=16))
    counts = dict(task2=0, tractor=0)
    for _ in range(400):
        c = sampler.sample("C_paired")
        assert c.task != "task1" and c.role == "C" and c.stage_choices is None
        counts[c.task] += 1
        d = sampler.make_d(c, 0)
        assert d.images is c.images and d.frame_indices is c.frame_indices
    assert 65 < counts["task2"] < 135
    assert 265 < counts["tractor"] < 335
    independent = [sampler.sample("C_independent") for _ in range(20)]
    assert any(row.task == "task1" and row.stage_choices is not None for row in independent)
    assert all(row.role == "C" for row in independent)


def test_plain_c_and_paired_c_preserve_old_sequence_when_disabled(inventory):
    old, new = RoleSampler(inventory, seed=45), RoleSampler(inventory, seed=45)
    for _ in range(15):
        left, right = old.sample("C"), new.sample("C_paired")
        assert (left.task, left.poi, left.frame_indices, left.targets, left.prompt) == (
            right.task, right.poi, right.frame_indices, right.targets, right.prompt)


def test_independent_and_paired_lane_randomness_are_separate(inventory):
    config = {"train_stage_augmentation": _policy("task1")}
    first, second = RoleSampler(inventory, config, seed=71), RoleSampler(inventory, config, seed=71)
    for _ in range(10):
        second.sample("C_paired")
        left, right = first.sample("C_independent"), second.sample("C_independent")
        assert (left.task, left.poi, left.frame_indices, left.targets, left.stage_choices) == (
            right.task, right.poi, right.frame_indices, right.targets, right.stage_choices)


def test_make_d_rejects_augmented_task_and_mapping_even_when_forged(inventory):
    sampler = RoleSampler(inventory, {"train_stage_augmentation": _policy("task1")})
    examples = (sampler.sample("C") for _ in range(30))
    c = next(example for example in examples if example.task == "task1")
    with pytest.raises(ValueError, match="forbidden"):
        sampler.make_d(c, 0)
    stripped_mapping = replace(c, targets=c.stage_ids, stage_choices=None)
    with pytest.raises(ValueError, match="forbidden"):
        sampler.make_d(stripped_mapping, 0)
    with pytest.raises(ValueError, match="forbidden"):
        RoleSampler(inventory).make_d(c, 0)
    with pytest.raises(ValueError, match="only valid on Role C"):
        replace(c, role="D")


def test_preflight_rejects_unfillable_pair_budget(inventory):
    sampler = RoleSampler(inventory, {"train_stage_augmentation": _policy("task1")})
    with pytest.raises(ValueError, match="C > D"):
        sampler.validate_role_budget(dict(A=0, B=0, C=16, D=16))
    with pytest.raises(ValueError, match="at least as many C"):
        sampler.validate_role_budget(dict(A=0, B=0, C=4, D=16))
    all_augmented = RoleSampler(inventory, {"train_stage_augmentation": _policy("task1", "task2", "tractor")})
    with pytest.raises(ValueError, match="no eligible unaugmented"):
        all_augmented.validate_role_budget(dict(A=0, B=0, C=24, D=16))
    with pytest.raises(ValueError, match="No eligible"):
        all_augmented.sample("C_paired")
    all_augmented.validate_role_budget(dict(A=0, B=0, C=16, D=0))


def test_snapshot_replays_mixed_lanes_and_rejects_policy_pool_or_rank_changes(inventory):
    config = {"train_stage_augmentation": _policy("task1")}
    sampler = RoleSampler(inventory, config, seed=94, worker_rank=1)
    order = ("A", "C_paired", "C_independent", "B", "C_independent", "C") * 4
    for request in order:
        sampler.sample(request)
    snapshot = sampler.state_dict()
    before = [sampler.sample(request) for request in order]
    sampler.load_state_dict(snapshot)
    after = [sampler.sample(request) for request in order]
    for left, right in zip(before, after, strict=True):
        assert (left.task, left.frame_indices, left.targets, left.stage_choices) == (
            right.task, right.frame_indices, right.targets, right.stage_choices)
        if right.stage_choices is not None:
            assert right.stage_choices.draw_id.startswith("rank=1:")
    with pytest.raises(ValueError, match="configuration"):
        RoleSampler(inventory, config, seed=94, worker_rank=0).load_state_dict(snapshot)
    with pytest.raises(ValueError, match="configuration"):
        RoleSampler(inventory, seed=94, worker_rank=1).load_state_dict(snapshot)
    changed_pool = _dataset(inventory.records + (replace(inventory.records[1], task="new-donor"),))
    with pytest.raises(ValueError, match="configuration"):
        RoleSampler(changed_pool, config, seed=94, worker_rank=1).load_state_dict(snapshot)
    summary = sampler.augmentation_summary()
    assert summary["algorithm_version"] == ALGORITHM_VERSION
    assert summary["task_policies"]["task1"] == {"enabled": True, "d_allowed": False}
    assert summary["task_policies"]["task2"] == {"enabled": False, "d_allowed": True}
    summary["policy"]["enabled"] = False
    assert sampler.train_stage_augmentation["enabled"] is True
