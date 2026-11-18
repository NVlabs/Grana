# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from grana.data.sampling import frame_indices


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/compare_inference.py"
SPEC = importlib.util.spec_from_file_location("compare_inference_stride_tests", SCRIPT)
comparison = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(comparison)
VIEWS = ["top", "left", "right", "bottom"]


def manifest(pair):
    return {"schema_version": 1, "roles": ["A", "C", "D"], "view_names": VIEWS,
            "ctsp": {"context_stride": pair[0], "tail_stride": pair[1], "poi_slot": 5},
            "arms": [{"id": "arm"}], "episodes": [{
                "id": "episode", "frame_paths": dict.fromkeys(VIEWS, "unused.npy"),
                "num_frames": 500, "stage_prompts": ["stage one", "stage two"],
                "windows": [{"poi": 200, "frame_indices": list(frame_indices(200, 500, *pair)),
                             "gt": {"stage_id": 0, "whole_progress": .2, "within_stage_progress": .4}}]}]}


@pytest.mark.parametrize("pair", [(16, 4), (40, 4), (10, 1)])
def test_manifest_uses_declared_stride_without_rewriting_historical_frames(pair):
    frozen = manifest(pair)
    before = deepcopy(frozen)
    assert comparison.validate_manifest(frozen) == VIEWS
    assert frozen == before


@pytest.mark.parametrize("key,value", [
    ("context_stride", 0), ("context_stride", -1), ("context_stride", True),
    ("context_stride", 16.), ("tail_stride", False), ("tail_stride", "4"),
    ("tail_stride", None), ("poi_slot", 2), ("poi_slot", True), ("poi_slot", 5.),
])
def test_invalid_manifest_stride_or_query_slot_is_rejected(key, value):
    frozen = manifest((16, 4))
    frozen["ctsp"][key] = value
    with pytest.raises(ValueError, match="explicit positive integer strides"):
        comparison.validate_manifest(frozen)


def test_declared_stride_must_match_frozen_frame_indices():
    frozen = manifest((40, 4))
    frozen["ctsp"]["context_stride"] = 16
    with pytest.raises(ValueError, match="Window differs"):
        comparison.validate_manifest(frozen)
    frozen = manifest((16, 4))
    frozen["episodes"][0]["windows"][0]["frame_indices"][0] += 1
    with pytest.raises(ValueError, match="Window differs"):
        comparison.validate_manifest(frozen)


def test_new_comparison_manifest_uses_shared_c16_t4_default_without_real_cache(tmp_path, monkeypatch):
    import datasets
    import grana.data.cache

    cache = tmp_path / "cache"
    trajectory = SimpleNamespace(
        cache_root=cache, trajectory_id="episode", task="task", num_frames=500,
        view_names=tuple(reversed(VIEWS)), frame_paths=tuple(cache / f"{view}.npy" for view in reversed(VIEWS)),
        compressed_task_prompt="Complete task", stage_prompts=("stage one", "stage two"),
        progress=(.2,) * 500, stage_ids=(0,) * 500, within_stage_progress=(.4,) * 500)
    config = {"data": {"views": VIEWS}, "evaluation": {"caches": [{"path": str(cache)}]}}

    def read(path):
        path = Path(path)
        if path.name == "evaluation_plan.json":
            return {"windows": [{"cache": str(cache), "trajectory_id": "episode", "task": "task"}]}
        if path.name == "result.json":
            return {"checkpoint_path": str(path.parent / "checkpoint")}
        if path.name == "metadata.json":
            return {"config": config, "weights_sha256": "a" * 64}
        raise AssertionError(path)

    monkeypatch.setattr(comparison, "read", read)
    monkeypatch.setattr(comparison, "digest", lambda path: "b" * 64)
    monkeypatch.setattr(comparison, "arrays_for", lambda episode, views: {})
    monkeypatch.setattr(datasets, "load_from_disk", lambda path: [{"id": "episode", "data_gen_output_fps": 10}])
    def selected_cache(entries, views):
        assert entries == [{"path": str(cache)}]
        return SimpleNamespace(records=(trajectory,))

    monkeypatch.setattr(grana.data.cache, "CachedTrajectories", selected_cache)
    target = tmp_path / "new-manifest.json"
    spec_path = tmp_path / "comparison.json"
    spec = {"cohorts": [{"id": "fixture", "run_dir": "runs/any-user-run"}],
            "arms": [{"id": "custom-model", "label": "My checkpoint", "checkpoint": "models/final",
                      "cohorts": ["fixture"]}]}
    spec_path.write_text(json.dumps(spec))
    original_read = comparison.read
    monkeypatch.setattr(comparison, "read", lambda path: spec if Path(path) == spec_path else original_read(path))
    comparison.prepare(SimpleNamespace(spec=spec_path, manifest=target, episodes_per_task=1, windows=3))
    actual = json.loads(target.read_text())
    assert len(actual["arms"]) == 1
    assert actual["arms"][0]["checkpoint"] == str(tmp_path / "models/final")
    assert actual["arms"][0]["label"] == "My checkpoint"
    assert actual["expected_model_calls"] == 9
    assert actual["ctsp"] == {"context_stride": 16, "tail_stride": 4, "poi_slot": 5}
    for episode in actual["episodes"]:
        assert episode["frame_paths"] == {view: str(cache / f"{view}.npy") for view in VIEWS}
        for window in episode["windows"]:
            assert window["frame_indices"] == list(frame_indices(window["poi"], 500, 16, 4))
    assert comparison.validate_manifest(actual) == VIEWS


@pytest.mark.parametrize("change", ["duplicate", "unknown-cohort", "unsafe-id", "empty-path", "unused-cohort"])
def test_comparison_spec_rejects_invalid_scope(tmp_path, change):
    spec = {"cohorts": [{"id": "target", "run_dir": "run"}],
            "arms": [{"id": "model", "checkpoint": "weights", "cohorts": ["target"]}]}
    if change == "duplicate":
        spec["arms"].append(dict(spec["arms"][0]))
    elif change == "unknown-cohort":
        spec["arms"][0]["cohorts"] = ["missing"]
    elif change == "unsafe-id":
        spec["arms"][0]["id"] = "../model"
    elif change == "empty-path":
        spec["cohorts"][0]["run_dir"] = ""
    else:
        spec["cohorts"].append({"id": "unused", "run_dir": "other"})
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec))
    with pytest.raises(ValueError):
        comparison.comparison_spec(path)


def test_named_cohort_scope_does_not_expand_to_other_cohorts_with_same_task():
    from scripts.render_inference_comparison import arm_label, task_arms

    arm = {"id": "model", "label": "User checkpoint", "cohorts": ["heldout"], "tasks": ["assemble"]}
    episode = {"task": "assemble", "task_alias": "heldout"}
    other = {"task": "assemble", "task_alias": "other"}
    assert comparison.arm_includes_episode(arm, episode)
    assert not comparison.arm_includes_episode(arm, other)
    assert task_arms({"arms": [arm]}, episode) == [arm]
    with pytest.raises(ValueError, match="one to four"):
        task_arms({"arms": [arm]}, other)
    assert arm_label(arm) == "User checkpoint"


@pytest.fixture
def saved_cohort(tmp_path):
    import numpy as np
    from datasets import Dataset

    cache, run, checkpoint = (tmp_path / name for name in ("cache", "run", "checkpoint"))
    for path in (cache, run, checkpoint):
        path.mkdir()
    frames = {}
    for index, view in enumerate(VIEWS):
        path = cache / f"{view}.npy"
        np.save(path, np.full((8, 2, 3, 3), index, dtype=np.uint8))
        frames[f"frames_{view}"] = str(path)
    rows = [{"id": identity, "task": "Task", "num_frames": 8, "frames_shape": [8, 2, 3, 3],
             "compressed_task_prompt": "Cached task", "progress": [.25] * 8,
             "stage_prompts": ["Prepare", "Finish"], "stage_ids": [0] * 8,
             "within_stage_progress": [.5] * 8, "data_gen_output_fps": 10, **frames}
            for identity in ("first", "second")]
    Dataset.from_list(rows).save_to_disk(str(cache / "processed_dataset"))
    (run / "evaluation_plan.json").write_text(json.dumps({"windows": [
        {"cache": str(cache), "trajectory_id": row["id"], "task": row["task"]} for row in rows]}))
    (run / "result.json").write_text(json.dumps({"checkpoint_path": str(checkpoint)}))
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"cohorts": [{"id": "heldout", "run_dir": "run"}],
                               "arms": [{"id": "model", "checkpoint": "checkpoint", "cohorts": ["heldout"]}]}))
    return SimpleNamespace(cache=cache, checkpoint=checkpoint,
                           args=SimpleNamespace(spec=spec, manifest=tmp_path / "manifest.json",
                                                episodes_per_task=2, windows=2))


def save_cohort_config(source, entries):
    (source.checkpoint / "metadata.json").write_text(json.dumps({
        "config": {"data": {"views": VIEWS}, "evaluation": {"caches": entries}},
        "weights_sha256": "a" * 64}))


@pytest.mark.parametrize("mapping", [False, True])
def test_prepare_accepts_processed_dataset_cache_path(saved_cohort, mapping):
    path = str(saved_cohort.cache / "processed_dataset")
    save_cohort_config(saved_cohort, [{"path": path} if mapping else path])
    comparison.prepare(saved_cohort.args)
    result = json.loads(saved_cohort.args.manifest.read_text())
    assert [episode["trajectory_id"] for episode in result["episodes"]] == ["first", "second"]
    assert all(episode["cache"] == str(saved_cohort.cache) for episode in result["episodes"])


@pytest.mark.parametrize("hash_split", [False, True])
def test_prepare_preserves_disjoint_cache_entry_scope(saved_cohort, hash_split):
    entries = [{"path": str(saved_cohort.cache), "include_ids": ["first"], "prompt": "First scope"},
               {"path": str(saved_cohort.cache / "processed_dataset"), "exclude_ids": ["first"],
                "prompt": "Second scope", "views": list(reversed(VIEWS))}]
    if hash_split:
        entries[0].pop("include_ids")
        entries[1].pop("exclude_ids")
        for entry, subset in zip(entries, ("eval", "train"), strict=True):
            entry["hash_split"] = {"seed": "public-example", "eval_count": 1, "subset": subset}
    save_cohort_config(saved_cohort, entries)
    comparison.prepare(saved_cohort.args)
    result = json.loads(saved_cohort.args.manifest.read_text())
    assert [episode["compressed_task_prompt"] for episode in result["episodes"]] == ["First scope", "Second scope"]
    assert all(episode["frame_paths"] == {view: str(saved_cohort.cache / f"{view}.npy") for view in VIEWS}
               for episode in result["episodes"])


@pytest.mark.parametrize("ambiguous", [False, True])
def test_prepare_rejects_missing_or_ambiguous_cohort_membership(saved_cohort, ambiguous):
    entries = [{"path": str(saved_cohort.cache), "include_ids": ["first"]}]
    if ambiguous:
        entries.append({"path": str(saved_cohort.cache / "processed_dataset"), "include_ids": ["first"]})
    save_cohort_config(saved_cohort, entries)
    with pytest.raises(ValueError, match="Ambiguous|absent from checkpoint evaluation caches"):
        comparison.prepare(saved_cohort.args)
    assert not saved_cohort.args.manifest.exists()
