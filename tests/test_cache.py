# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import numpy as np
import pytest
from datasets import Dataset

from grana.data.cache import CachedTrajectories, resolve_frame_path


def write_cache(root: Path, ids=("episode",), **changes):
    (root / "frames").mkdir(parents=True)
    frames = np.arange(6 * 3 * 4 * 3, dtype=np.uint8).reshape(6, 3, 4, 3)
    row = dict(id="episode", task="assemble", num_frames=6, frames_shape=[6, 3, 4, 3],
               frames="/old-machine/data/frames/episode.npy",
               progress=[0., .1, .3, .5, .8, 1.], stage_ids=[0, 0, 1, 1, 2, 2],
               within_stage_progress=[0., .3, .1, .7, .6, 1.],
               stage_prompts=["Open box", "Insert part", "Close box"],
               compressed_task_prompt="Open the box, insert the part, and close it.")
    rows = []
    for identity in ids:
        np.save(root / "frames" / f"{identity}.npy", frames)
        rows.append({**row, "id": identity, "frames": f"/old-machine/data/frames/{identity}.npy", **changes})
    Dataset.from_list(rows).save_to_disk(str(root / "processed_dataset"))
    return frames


def test_relocated_cache_paths_and_pixels(tmp_path):
    expected = write_cache(tmp_path / "cache")
    data = CachedTrajectories([str(tmp_path / "cache")], views=["top"])
    record = data.records[0]
    indices = (0, 0, 1, 2, 3, 3, 4, 5)
    images = record.images(indices)
    assert len(images) == 8
    for image, index in zip(images, indices, strict=True):
        np.testing.assert_array_equal(np.asarray(image), expected[index])
    assert data.summary()["eligible_rows"] == {"A": 1, "B": 1, "C": 1}
    assert data.summary()["caches"][0]["tasks"] == ["assemble"]
    assert record.frame_paths == (tmp_path / "cache" / "frames" / "episode.npy",)


def test_per_cache_views_keep_camera_order_pixels_and_paired_d(tmp_path):
    from grana.data.sampling import RoleSampler, materialize_example

    one, two = tmp_path / "one", tmp_path / "two"
    expected_one = write_cache(one)
    left_path, wrist_path = two / "frames/left.npy", two / "frames/wrist.npy"
    expected_left = write_cache(two, frames_left=str(left_path), frames_wrist=str(wrist_path))
    expected_wrist = (expected_left.astype(np.uint16) + 40).astype(np.uint8)
    np.save(left_path, expected_left)
    np.save(wrist_path, expected_wrist)
    data = CachedTrajectories([str(one), {"path": str(two), "views": ["wrist", "left"]}], views=["top"])
    first, second = data.records
    assert first.view_names == ("top",) and second.view_names == ("wrist", "left")
    assert data.summary()["views"] == ["top"]
    assert data.summary()["view_sets"] == [["top"], ["wrist", "left"]]
    assert data.summary()["caches"][1]["views"] == ["wrist", "left"]
    a = materialize_example(first, "A", 3, (1, 1))
    c = materialize_example(second, "C", 3, (1, 1))
    assert len(a.images) == 8 and len(c.images) == 16
    for slot, frame in enumerate(c.frame_indices):
        np.testing.assert_array_equal(np.asarray(c.images[slot * 2]), expected_wrist[frame])
        np.testing.assert_array_equal(np.asarray(c.images[slot * 2 + 1]), expected_left[frame])
        np.testing.assert_array_equal(np.asarray(a.images[slot]), expected_one[a.frame_indices[slot]])
    sampler = RoleSampler(data, {"stride_pairs": [[1, 1]]})
    d = sampler.make_d(c, 1)
    assert d.images is c.images and d.view_names is c.view_names
    assert d.frame_indices is c.frame_indices
    assert sum(x.get("text") == "<|stage_token|>" for x in c.messages()[0]["content"]) == 8
    assert sum(x.get("text") == "<|prog_token|>" for x in d.messages()[0]["content"]) == 8
    sampled = {sampler.sample("C").view_names for _ in range(30)}
    assert sampled == {("top",), ("wrist", "left")}


@pytest.mark.parametrize("location", ["default", "cache"])
@pytest.mark.parametrize("views", [[], "top", ["top", "top"], ["top", " top "], [None], [""],
                                  ["<|stage_token|>"], ["<|image_pad|>"], ["left<|"], ["|>right"],
                                  ["x" * 129], [" left"], ["left "], ["\tleft\n"]])
def test_invalid_camera_inventory_is_rejected(tmp_path, location, views):
    root = tmp_path / "cache"
    write_cache(root)
    with pytest.raises(ValueError, match="view"):
        if location == "default":
            CachedTrajectories([str(root)], views=views)
        else:
            CachedTrajectories([{"path": str(root), "views": views}], views=["top"])


def test_camera_name_length_boundary_and_internal_spaces_preserve_identity(tmp_path):
    root = tmp_path / "cache"
    names = ["x" * 128, "left camera"]
    frame_path = str(root / "frames/episode.npy")
    write_cache(root, **{f"frames_{name}": frame_path for name in names})
    for entry, defaults in ((str(root), names), ({"path": str(root), "views": names}, ["top"])):
        record = CachedTrajectories([entry], views=defaults).records[0]
        assert record.view_names == tuple(names)
        assert len(record.images((0,))) == 2


def test_per_cache_missing_camera_is_not_filled_from_another_view(tmp_path):
    root = tmp_path / "cache"
    write_cache(root)
    with pytest.raises(ValueError, match="explicit NPY"):
        CachedTrajectories([{"path": str(root), "views": ["wrist"]}], views=["top"])


def test_explicit_absolute_path_precedes_same_cache_basename(tmp_path):
    root, external = tmp_path / "cache", tmp_path / "external"
    write_cache(root)
    external.mkdir()
    path = external / "episode.npy"
    np.save(path, np.zeros((6, 3, 4, 3), dtype=np.uint8))
    assert resolve_frame_path(str(path), root) == path
    assert resolve_frame_path("frames/episode.npy", root) == root / "frames" / "episode.npy"
    with pytest.raises(FileNotFoundError, match="same-cache fallback"):
        resolve_frame_path("/somewhere/missing.npy", root)


@pytest.mark.parametrize("changes,match", [
    ({"progress": [0., .1]}, "frame-aligned"),
    ({"progress": [0., .1, .2, .3, .4, float("nan")]}, "finite"),
    ({"within_stage_progress": [0., .1, .2, .3, .4, 1.1]}, "finite"),
    ({"stage_ids": [0, 0, 1, 1, 2, 3]}, "stage_ids"),
    ({"stage_ids": [0., 0., 1., 1., 2., 2.]}, "stage_ids"),
    ({"compressed_task_prompt": None}, "compressed_task_prompt"),
    ({"compressed_task_prompt": "bad <|prog_token|> prompt"}, "reserved"),
    ({"frames_shape": [5, 3, 4, 3]}, "frames_shape"),
])
def test_invalid_supervision_rejected_before_sampling(tmp_path, changes, match):
    write_cache(tmp_path / "cache", **changes)
    with pytest.raises(ValueError, match=match):
        CachedTrajectories([str(tmp_path / "cache")], views=["top"])


def test_a_only_cache_needs_no_stage_metadata_and_manifest_owns_prompt(tmp_path):
    root = tmp_path / "cache"
    write_cache(root, stage_ids=None, stage_prompts=None, within_stage_progress=None, compressed_task_prompt=None)
    data = CachedTrajectories([{"path": str(root), "prompt": "Canonical whole task.", "roles": ["A"]}], views=["top"])
    assert data.records[0].compressed_task_prompt == "Canonical whole task."
    assert data.by_role["B"] == data.by_role["C"] == {}


def test_bc_only_does_not_use_task_id_as_compressed_prompt(tmp_path):
    root = tmp_path / "cache"
    write_cache(root, progress=None, compressed_task_prompt=None)
    data = CachedTrajectories([{"path": str(root), "roles": ["B", "C"]}], views=["top"])
    assert data.records[0].compressed_task_prompt is None
    assert data.records[0].progress == ()
    assert data.by_role["A"] == {}


def test_invalid_image_dtype_rejected(tmp_path):
    root = tmp_path / "cache"
    write_cache(root)
    np.save(root / "frames" / "episode.npy", np.zeros((6, 3, 4, 3), dtype=np.float32))
    data = CachedTrajectories([str(root)], views=["top"])
    with pytest.raises(ValueError, match="uint8"):
        data.records[0].images((0,))


def test_missing_view_never_silently_copies_another_camera(tmp_path):
    root = tmp_path / "cache"
    write_cache(root)
    with pytest.raises(ValueError, match="explicit NPY"):
        CachedTrajectories([str(root)], views=["top", "left"])


def test_duplicate_inventory_and_independent_d_rejected(tmp_path):
    root = tmp_path / "cache"
    write_cache(root)
    with pytest.raises(ValueError, match="Duplicate"):
        CachedTrajectories([str(root), str(root)], views=["top"])
    with pytest.raises(ValueError, match="D derives"):
        CachedTrajectories([{"path": str(root), "roles": ["D"]}], views=["top"])


def test_include_exclude_ids_select_only_exact_trajectories(tmp_path):
    root = tmp_path / "cache"
    write_cache(root, ids=("first", "second", "third"))
    original_files = {path: path.read_bytes() for path in (root / "frames").iterdir()}
    included = CachedTrajectories([{"path": str(root), "include_ids": ["second"]}], views=["top"])
    excluded = CachedTrajectories([{"path": str(root), "exclude_ids": ["first", "third"]}], views=["top"])
    assert [row.trajectory_id for row in included.records] == ["second"]
    assert included.records == excluded.records
    summary = included.summary()["caches"][0]
    assert summary["source_rows"] == 3 and summary["rows"] == 1
    assert included.records[0].frame_paths == (root / "frames/second.npy",)
    assert original_files == {path: path.read_bytes() for path in (root / "frames").iterdir()}


@pytest.mark.parametrize("selectors, message", [
    ({"include_ids": []}, "nonempty list"),
    ({"include_ids": ["first", "first"]}, "unique trajectory IDs"),
    ({"include_ids": ["absent"]}, "absent trajectory IDs"),
    ({"exclude_ids": ["absent"]}, "absent trajectory IDs"),
    ({"include_ids": ["first"], "exclude_ids": ["second"]}, "Choose include_ids or exclude_ids"),
    ({"exclude_ids": ["first", "second"]}, "selection is empty"),
])
def test_invalid_trajectory_selection_fails_closed(tmp_path, selectors, message):
    root = tmp_path / "cache"
    write_cache(root, ids=("first", "second"))
    with pytest.raises(ValueError, match=message):
        CachedTrajectories([{"path": str(root), **selectors}], views=["top"])


def test_hash_split_reproduces_membership_independently_of_row_order(tmp_path):
    identities = ("third", "first", "fourth", "second", "fifth")
    expected = {"eval": {"first", "fourth"}, "train": {"third", "second", "fifth"}}
    for ordering in (identities, tuple(reversed(identities))):
        root = tmp_path / ("forward" if ordering == identities else "reverse")
        write_cache(root, ids=ordering)
        cohorts = {}
        for subset in ("train", "eval"):
            spec = {"path": str(root), "hash_split": {
                "seed": "public-example", "eval_count": 2, "subset": subset}}
            first = CachedTrajectories([spec], views=["top"])
            repeated = CachedTrajectories([spec], views=["top"])
            selected = [row.trajectory_id for row in first.records]
            assert first.records == repeated.records
            assert selected == [identity for identity in ordering if identity in expected[subset]]
            cohorts[subset] = set(selected)
        assert not cohorts["train"] & cohorts["eval"]
        assert cohorts["train"] | cohorts["eval"] == set(identities)


@pytest.mark.parametrize("split,message", [
    (None, "exactly seed"),
    ({}, "exactly seed"),
    ({"seed": "example", "eval_count": 1, "subset": "train", "unexpected": 1}, "exactly seed"),
    ({"seed": "", "eval_count": 1, "subset": "train"}, "seed"),
    ({"seed": "   ", "eval_count": 1, "subset": "train"}, "seed"),
    ({"seed": 12, "eval_count": 1, "subset": "train"}, "seed"),
    ({"seed": "example", "eval_count": 1, "subset": "test"}, "subset"),
    ({"seed": "example", "eval_count": 1, "subset": []}, "subset"),
    ({"seed": "example", "eval_count": True, "subset": "train"}, "positive integer"),
    ({"seed": "example", "eval_count": 1.5, "subset": "train"}, "positive integer"),
    ({"seed": "example", "eval_count": 0, "subset": "train"}, "positive integer"),
    ({"seed": "example", "eval_count": -1, "subset": "train"}, "positive integer"),
    ({"seed": "example", "eval_count": 3, "subset": "train"}, "smaller than"),
    ({"seed": "example", "eval_count": 4, "subset": "eval"}, "smaller than"),
])
def test_invalid_hash_split_rejected(tmp_path, split, message):
    root = tmp_path / "cache"
    write_cache(root, ids=("first", "second", "third"))
    with pytest.raises(ValueError, match=message):
        CachedTrajectories([{"path": str(root), "hash_split": split}], views=["top"])


@pytest.mark.parametrize("key,value", [("include_ids", ["first"]), ("exclude_ids", ["first"]),
                                      ("include_ids", None), ("exclude_ids", None)])
def test_hash_split_cannot_mix_with_explicit_selectors(tmp_path, key, value):
    root = tmp_path / "cache"
    write_cache(root, ids=("first", "second"))
    with pytest.raises(ValueError, match="Choose hash_split"):
        CachedTrajectories([{"path": str(root), key: value, "hash_split": {
            "seed": "example", "eval_count": 1, "subset": "train"}}], views=["top"])


@pytest.mark.parametrize("subset", ["train", "eval"])
def test_hash_split_rejects_duplicate_source_ids_before_selecting(tmp_path, subset):
    root = tmp_path / "cache"
    write_cache(root, ids=("first", "first", "second", "third"))
    with pytest.raises(ValueError, match="Duplicate cached trajectory IDs"):
        CachedTrajectories([{"path": str(root), "hash_split": {
            "seed": "example", "eval_count": 1, "subset": subset}}], views=["top"])


def test_role_specific_task_weights_and_conflicts(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    write_cache(first)
    write_cache(second, ids=("different",))
    data = CachedTrajectories([{"path": str(first), "task_weights": {"A": 3., "C": .5}}], views=["top"])
    assert data.task_weights == {"A": {"assemble": 3.}, "B": {"assemble": 1.}, "C": {"assemble": .5}}
    with pytest.raises(ValueError, match="Conflicting declared weight"):
        CachedTrajectories([{"path": str(first), "task_weights": {"A": 3.}}, str(second)], views=["top"])


@pytest.mark.parametrize("weights", [{"A": 0}, {"A": -1}, {"A": float("nan")}, {"A": True}, {"B": 2}, {"D": 2}])
def test_invalid_or_ineligible_task_weights_rejected(tmp_path, weights):
    root = tmp_path / "cache"
    write_cache(root)
    with pytest.raises(ValueError, match="task_weights|task weights"):
        CachedTrajectories([{"path": str(root), "roles": ["A"], "task_weights": weights}], views=["top"])


def test_single_stage_source_row_remains_eligible_for_all_roles(tmp_path):
    root = tmp_path / "cache"
    write_cache(root, stage_prompts=["Only operation"], stage_ids=[0] * 6)
    data = CachedTrajectories([str(root)], views=["top"])
    assert data.records[0].stage_ids == (0,) * 6
    assert data.summary()["eligible_rows"] == {"A": 1, "B": 1, "C": 1}


@pytest.mark.parametrize("stage,expected_roles", [
    ("pretrain", {"A": 4, "B": 4, "C": 4, "D": 0}),
    ("posttrain", {"A": 0, "B": 4, "C": 4, "D": 4}),
])
def test_cli_check_data_emits_fixed_eval_plan_and_rejects_leakage(tmp_path, monkeypatch, capsys, stage, expected_roles):
    import json
    import sys
    import yaml
    from grana.cli import main

    train, heldout = tmp_path / "train", tmp_path / "heldout"
    write_cache(train, ids=("train1",))
    write_cache(heldout, ids=("heldout1", "heldout2"))
    config = {"data": {"caches": [str(train)], "views": ["top"]},
              "training": {"stage": stage, "init_checkpoint": "/pretrain" if stage == "posttrain" else None},
              "evaluation": {"enabled": True, "caches": [str(heldout)],
                             "max_trajectories_per_task": 2, "windows_per_trajectory": 2}}
    path = tmp_path / "recipe.yaml"
    path.write_text(yaml.safe_dump(config))
    capsys.readouterr()
    monkeypatch.setattr(sys, "argv", ["grana", "validate", "--config", str(path), "--check-data"])
    main()
    report = json.loads(capsys.readouterr().out)
    assert report["data"]["trajectories"] == 1
    assert report["evaluation_data"]["trajectories"] == 2
    assert report["evaluation_plan"]["model_calls"] == 12
    assert report["evaluation_plan"]["role_window_counts"] == expected_roles
    config["evaluation"]["caches"] = [str(train)]
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="overlaps training"):
        main()
