# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
import hashlib
from io import StringIO
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from grana.inference import client
from grana.inference.selection import select_prediction
from scripts.inference import evaluate_http_frames as evaluation
from scripts.inference import summarize_http_frames as summary


def write_json(path, value):
    path.write_text(json.dumps(value))


def test_cache_manifest_resolves_user_paths_and_keeps_source_unchanged(tmp_path, monkeypatch):
    monkeypatch.setenv("USER_CACHE", str(tmp_path / "external"))
    source = tmp_path / "caches.yaml"
    source.write_text('caches:\n- relative/cache\n- path: ${USER_CACHE}/eval\n  roles: [C]\n- path: ${GRANA_DATA_ROOT}/other\n')
    before = source.read_bytes()
    root = tmp_path / "data"
    entries = evaluation.cache_entries(source, root)
    assert entries == [{"path": str(root / "relative/cache")},
                       {"path": str(tmp_path / "external/eval"), "roles": ["C"]},
                       {"path": str(root / "other")}]
    assert source.read_bytes() == before
    source.write_text('caches:\n- relative/cache\n')
    assert evaluation.cache_entries(source) == [{"path": str(tmp_path / "relative/cache")}]


def test_cache_manifest_rejects_unresolved_variables(tmp_path, monkeypatch):
    monkeypatch.delenv("MISSING_CACHE_ROOT", raising=False)
    source = tmp_path / "caches.yaml"
    source.write_text('caches:\n- ${MISSING_CACHE_ROOT}/cache\n')
    with pytest.raises(ValueError, match="Unresolved"):
        evaluation.cache_entries(source)


@pytest.mark.parametrize("frames,samples", [(70, 128), (77, 128), (201, 128), (6261, 128), (17, 2)])
def test_stratified_targets_cover_disjoint_time_bins_reproducibly(frames, samples):
    targets = evaluation.stratified_targets(frames, samples, "cache/trajectory", 11)
    assert targets == evaluation.stratified_targets(frames, samples, "cache/trajectory", 11)
    count = min(frames, samples)
    assert len(targets) == len(set(targets)) == count
    assert targets == sorted(targets)
    assert all(i * frames // count <= target < (i + 1) * frames // count
               for i, target in enumerate(targets))
    if samples >= frames:
        assert targets == list(range(frames))


def test_seed_and_trajectory_identity_change_only_frame_sampling():
    first = evaluation.stratified_targets(6261, 128, "cache/one", 11)
    assert first != evaluation.stratified_targets(6261, 128, "cache/one", 29)
    assert first != evaluation.stratified_targets(6261, 128, "cache/two", 11)


@pytest.mark.parametrize("frames,samples", [(True, 4), (10, True), (1, 4), (10, 1), (2.5, 2), (10, 3.5)])
def test_stratified_sampling_rejects_invalid_sizes(frames, samples):
    with pytest.raises(ValueError):
        evaluation.stratified_targets(frames, samples, "one", 11)


@pytest.fixture
def campaign(tmp_path):
    frames = np.arange(201, dtype=np.uint8)[:, None, None, None] * np.ones((201, 2, 3, 3), dtype=np.uint8)
    source = tmp_path / "top.npy"
    np.save(source, frames)
    request_path, metadata_path = tmp_path / "request.json", tmp_path / "metadata.json"
    write_json(request_path, {"views": {"top": str(source)}, "task": "Assemble",
                              "stage_prompts": ["Prepare", "Finish"]})
    # Labels intentionally disagree with the mock model's C prediction after frame 100.
    # A valid D request must still focus on predicted C, never this local GT.
    metadata = {"trajectory_id": "fixture", "num_frames": 201, "source_fps": 10,
                "stage_ids": [0] * 201, "within_stage_progress": [.125] * 201,
                "whole_progress": [.0625] * 201}
    write_json(metadata_path, metadata)
    stat = source.stat()
    trajectory = {"key": "fixture-key", "trajectory_id": "fixture", "group": "L1L2",
                  "num_frames": 201, "stage_count": 2, "request_path": str(request_path),
                  "metadata_path": str(metadata_path), "request_sha256": evaluation.file_digest(request_path),
                  "metadata_sha256": evaluation.file_digest(metadata_path),
                  "sources": {"top": {"path": str(source), "shape": [201, 2, 3, 3],
                                       "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}},
                  "anchors": [0, 100, 200], "targets": {"11": [3, 100, 197], "29": [1, 99, 199]}}
    plan = {"schema": "grana-http-frame-comparison-plan-v1", "seeds": [11, 29],
            "stride_pair": [16, 4], "checkpoint": {"weights_sha256": "a" * 64},
            "trajectories": [trajectory]}
    plan_path = tmp_path / "plan.json"
    write_json(plan_path, plan)
    args = SimpleNamespace(plan=plan_path, seed=11, server="http://localhost:9876", timeout=30)
    return args, plan, metadata


def response_for(request, roles, request_id):
    ids = [int(frame >= 100) for frame in request.frame_indices]
    values = [frame / 200 for frame in request.frame_indices]
    focus = ids[request.poi_slot]
    result = {"checkpoint": {"weights_sha256": "a" * 64}, "roles": list(roles),
              "prediction_frame": "poi" if request.poi_slot == 5 else "last",
              "poi_slot": request.poi_slot, "frame_indices": list(request.frame_indices),
              "stage": {"ids": ids}, "D_focus_stage": focus,
              "progress": {"D": {"values": values, "focus_stage": focus}}}
    result["selected_prediction"] = select_prediction(result)
    return {"schema": client.SCHEMA, "request_id": request_id, "elapsed_seconds": .01, "result": result}


def install_server(monkeypatch, health_change=None, reply_change=None):
    health = {"ready": True, "wire_formats": ["grana-binary-v1"], "checkpoint": {
        "weights_sha256": "a" * 64,
        "inference_runtime": {"gpu_image_resize": True, "policy": "cuda_optimized_v1"}}}
    if health_change:
        health_change(health)
    monkeypatch.setattr(evaluation, "urlopen", lambda *a, **kw: StringIO(json.dumps(health)))
    calls = []

    def predict(server, request, *, roles, request_id, timeout):
        assert roles == ("C", "D")
        assert request.focus_stage is None
        assert not hasattr(request, "stage_ids") and not hasattr(request, "within_stage_progress")
        body = client.build_binary(request, roles, request_id=request_id)
        magic, version, length = client.BINARY_PREAMBLE.unpack_from(body)
        assert magic == b"GRNB" and version == 1
        offset = client.BINARY_PREAMBLE.size
        header = json.loads(body[offset:offset + length])
        assert not {"gt", "stage_ids", "within_stage_progress", "whole_progress", "focus_stage"} & header.keys()
        assert [int(frame[0, 0, 0]) for frame in request.images["top"]] == list(request.frame_indices)
        calls.append((request_id, request, header))
        response = response_for(request, roles, request_id)
        if reply_change:
            reply_change(response)
        return response

    monkeypatch.setattr(evaluation, "predict", predict)
    return calls


def test_paired_http_run_preserves_target_identity_gt_isolation_and_journal_receipt(campaign, monkeypatch):
    args, plan, metadata = campaign
    calls = install_server(monkeypatch)
    first = evaluation.run(args)
    output = args.plan.parent / "runs/seed_11"
    records = [json.loads(line) for line in (output / "predictions.jsonl").read_text().splitlines()]
    assert len(calls) == len(records) == first["requests_written"] == 12
    assert first["plan_sha256"] == evaluation.file_digest(args.plan)
    assert first["journal_sha256"] == evaluation.file_digest(output / "predictions.jsonl")
    assert json.loads((output / "completion.json").read_text()) == first
    assert len({row["request_id"] for row in records}) == 12
    for row, (_, request, header) in zip(records, calls, strict=True):
        slot, frame = row["slot"], row["source_frame"]
        assert slot == (5 if row["mode"] == "poi" else 7)
        assert row["frame_indices"][slot] == frame
        assert row["D_focus_stage"] == row["C_ids"][slot] == int(frame >= 100)
        assert row["gt"] == {"stage_id": metadata["stage_ids"][frame],
                              "within_stage_progress": metadata["within_stage_progress"][frame]}
        assert row["checkpoint_sha256"] == "a" * 64
        assert row["input_sha256"] == hashlib.sha256(client.build_binary(request, ("C", "D"), request_id="identity")).hexdigest()
        prediction = {key: row[key] for key in ("C_ids", "D_values", "D_focus_stage")}
        assert row["prediction_sha256"] == evaluation.digest(prediction)
        assert header["frame_indices"] == row["frame_indices"]
        if row["mode"] == "last":
            assert max(row["frame_indices"]) <= frame
    # The same source can be an anchor and a quality observation; keep distinct IDs/tags.
    assert {(r["kind"], r["mode"]) for r in records if r["source_frame"] == 100} == {
        (kind, mode) for kind in ("anchor", "quality") for mode in ("poi", "last")}
    target_100 = {r["mode"]: r for r in records if r["kind"] == "quality" and r["source_frame"] == 100}
    assert target_100["poi"]["frame_indices"] == [52, 68, 84, 92, 96, 100, 104, 108]
    assert target_100["last"]["frame_indices"] == [44, 60, 76, 84, 88, 92, 96, 100]
    assert target_100["last"]["C_ids"][5] == 0 and target_100["last"]["D_focus_stage"] == 1

    args.seed = 29
    evaluation.run(args)
    second = [json.loads(line) for line in (args.plan.parent / "runs/seed_29/predictions.jsonl").read_text().splitlines()]
    anchor_hashes = lambda rows: {(r["source_frame"], r["mode"]): (r["input_sha256"], r["prediction_sha256"])
                                  for r in rows if r["kind"] == "anchor"}
    assert anchor_hashes(records) == anchor_hashes(second)
    assert records[0]["mode"] == "poi" and second[0]["mode"] == "last"
    with pytest.raises(FileExistsError, match="fresh run"):
        evaluation.run(args)


@pytest.mark.parametrize("change", [
    lambda h: h.update(ready=False),
    lambda h: h.update(wire_formats=["json"]),
    lambda h: h["checkpoint"].update(weights_sha256="b" * 64),
    lambda h: h["checkpoint"]["inference_runtime"].update(gpu_image_resize=False),
    lambda h: h["checkpoint"]["inference_runtime"].update(policy="fallback"),
])
def test_noncanonical_server_is_rejected_before_inference(campaign, monkeypatch, change):
    args, _, _ = campaign
    calls = install_server(monkeypatch, health_change=change)
    with pytest.raises(ValueError, match="canonical"):
        evaluation.run(args)
    assert not calls


@pytest.mark.parametrize("change", [
    lambda e: e.update(request_id="other-request"),
    lambda e: e["result"]["checkpoint"].update(weights_sha256="b" * 64),
    lambda e: e["result"].update(roles=["C"]),
    lambda e: e["result"]["selected_prediction"].update(frame_index=200),
    lambda e: e["result"].update(D_focus_stage=1),
    lambda e: e["result"]["progress"]["D"].update(focus_stage=1),
])
def test_wrong_checkpoint_selection_or_d_focus_cannot_enter_completed_journal(campaign, monkeypatch, change):
    args, _, _ = campaign
    calls = install_server(monkeypatch, reply_change=change)
    with pytest.raises(ValueError):
        evaluation.run(args)
    assert len(calls) == 1
    output = args.plan.parent / "runs/seed_11"
    assert not (output / "completion.json").exists()
    assert (output / "predictions.jsonl").read_text() == ""


@pytest.mark.parametrize("source", ["request", "metadata", "rgb"])
def test_changed_source_or_gt_metadata_is_rejected_before_model_calls(campaign, monkeypatch, source):
    args, plan, _ = campaign
    calls = install_server(monkeypatch)
    trajectory = plan["trajectories"][0]
    if source == "rgb":
        path = Path(trajectory["sources"]["top"]["path"])
        stat = path.stat()
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    else:
        path = Path(trajectory[f"{source}_path"])
        path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="changed after plan"):
        evaluation.run(args)
    assert not calls


@pytest.mark.parametrize("change", [lambda p: p.update(stride_pair=[30, 4]),
                                   lambda p: p.update(schema="wrong-schema"),
                                   lambda p: p.update(seeds=[29])])
def test_incompatible_plan_is_rejected_before_http(campaign, monkeypatch, change):
    args, plan, _ = campaign
    change(plan)
    write_json(args.plan, plan)
    monkeypatch.setattr(evaluation, "urlopen", lambda *a, **kw: pytest.fail("Invalid plan reached HTTP"))
    with pytest.raises(ValueError):
        evaluation.run(args)


def test_prepare_uses_all_manifest_trajectories_and_sampling_ignores_labels(tmp_path, monkeypatch):
    from datasets import Dataset
    import yaml

    data_root = tmp_path / "data"
    cache = data_root / "assembly-heldout"
    (cache / "frames").mkdir(parents=True)
    rows = []
    for identity, count in (("short", 70), ("long", 201)):
        path = cache / "frames" / f"{identity}.npy"
        np.save(path, np.zeros((count, 2, 3, 3), dtype=np.uint8))
        rows.append({"id": identity, "task": "Assembly", "num_frames": count,
                     "frames_shape": [count, 2, 3, 3], "frames_top": str(path),
                     "stage_prompts": ["Prepare", "Finish"], "stage_ids": [0] * count,
                     "within_stage_progress": [.25] * count, "data_gen_output_fps": 10})
    Dataset.from_list(rows).save_to_disk(str(cache / "processed_dataset"))
    config = tmp_path / "eval.yaml"
    config.write_text(yaml.safe_dump({"caches": [{"path": "${GRANA_POSTTRAIN_DATA_ROOT}/assembly-heldout",
                                                 "roles": ["B", "C"], "views": ["top"]}]}))
    monkeypatch.setattr(evaluation.subprocess, "check_output", lambda *a, **kw: "f" * 40 + "\n")
    args = SimpleNamespace(config=config, data_root=data_root, checkpoint=tmp_path / "checkpoint",
                           weights_sha256="a" * 64, output=tmp_path / "first", seeds=[11, 29, 47],
                           samples=128, anchors=8)
    first = evaluation.prepare(args)
    summary.validate_plan(first)
    assert {row["group"] for row in first["trajectories"]} == {"assembly-heldout"}
    assert len(first["trajectories"]) == 2 and first["stride_pair"] == [16, 4]
    for t in first["trajectories"]:
        raw = json.loads(Path(t["request_path"]).read_text())
        labels = json.loads(Path(t["metadata_path"]).read_text())
        assert set(raw) == {"views", "task", "stage_prompts"}
        assert labels["whole_progress"] == [.125] * t["num_frames"]
        assert t["request_sha256"] == evaluation.file_digest(t["request_path"])
        assert t["metadata_sha256"] == evaluation.file_digest(t["metadata_path"])
        assert len(t["anchors"]) == 8
        assert all(len(set(targets)) == min(128, t["num_frames"]) for targets in t["targets"].values())
    changed = deepcopy(rows)
    for row in changed:
        row["stage_ids"] = [1] * row["num_frames"]
        row["within_stage_progress"] = [.75] * row["num_frames"]
    # Change labels while preserving the source inventory and time bins.
    Dataset.from_list(changed).save_to_disk(str(cache / "processed_dataset"))
    args.output = tmp_path / "second"
    second = evaluation.prepare(args)
    assert [(t["key"], t["targets"], t["anchors"]) for t in first["trajectories"]] == [
        (t["key"], t["targets"], t["anchors"]) for t in second["trajectories"]]
    assert [t["metadata_sha256"] for t in first["trajectories"]] != [
        t["metadata_sha256"] for t in second["trajectories"]]


def test_prepare_supports_same_named_caches_without_git_metadata(tmp_path, monkeypatch):
    from datasets import Dataset
    from scripts.inference import evaluate_frames as offline
    import yaml

    caches = [tmp_path / "source-a/eval", tmp_path / "source-b/eval"]
    for cache in caches:
        (cache / "frames").mkdir(parents=True)
        pixels = cache / "frames/episode.npy"
        np.save(pixels, np.zeros((4, 2, 3, 3), dtype=np.uint8))
        Dataset.from_list([{"id": "same-id", "task": "Synthetic task", "num_frames": 4,
                            "frames_shape": [4, 2, 3, 3], "frames_top": str(pixels),
                            "stage_prompts": ["Place"], "stage_ids": [0] * 4,
                            "within_stage_progress": [.25] * 4, "data_gen_output_fps": 10}]).save_to_disk(str(cache / "processed_dataset"))
    config = tmp_path / "caches.yaml"
    config.write_text(yaml.safe_dump({"caches": [{"path": str(cache), "views": ["top"], "roles": ["C"]} for cache in caches]}))
    def no_git(*args, **kwargs):
        raise FileNotFoundError("git is not installed")
    monkeypatch.setattr(evaluation.subprocess, "check_output", no_git)
    args = SimpleNamespace(config=config, data_root=None, checkpoint=tmp_path / "weights",
                           weights_sha256="a" * 64, output=tmp_path / "prepared", seeds=[1], samples=2, anchors=2)
    plan = evaluation.prepare(args)
    summary.validate_plan(plan)
    assert plan["source_commit"] is None
    assert "scripts/inference/evaluate_http_frames.py" in plan["source_files_sha256"]
    assert len({row["key"] for row in plan["trajectories"]}) == 2
    assert len({row["group"] for row in plan["trajectories"]}) == 2
    assert all(Path(row["request_path"]).is_file() for row in plan["trajectories"])
    assert {row["group"] for row in plan["trajectories"]} == {str(cache) for cache in caches}
    offline_args = SimpleNamespace(plan=args.output / "plan.json", output=tmp_path / "offline",
                                   devices="0", batch_size=1, memory_fraction=.9,
                                   max_batch_size=None, seeds=None)
    prepared, schedule, _ = offline.prepare_campaign(
        offline_args, device_count=1,
        get_device_properties=lambda _: SimpleNamespace(uuid="GPU-00000000-0000-0000-0000-000000000000"),
    )
    assert prepared["trajectories"] == plan["trajectories"]
    assert schedule["devices"] == [0]
    assert sum(task["jobs"] for task in schedule["shards"]["0"]) == 16
