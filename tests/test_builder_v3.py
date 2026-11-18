# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared-file episodes preserve the v2 cache's RGB, supervision and sampling contract."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from datasets import load_from_disk
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from grana.annotation.label_server import LabelProject
from grana.annotation.source import LeRobotSource
from grana.data.builder import build_cache
from test_label_v3_clip import FIRST, make_source as boundary_source, recipe as boundary_recipe


FPS, LENGTHS = 10, (8, 12, 8)
WORDS = ["1", "2", "empty", "high", "low", "normal", "suboptimal", "failure"]


def encode(path, rgb):
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-v", "error", "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", "32x24",
                    "-framerate", str(FPS), "-i", "pipe:0", "-c:v", "libx264rgb", "-crf", "0", "-g", "4",
                    "-threads", "1", str(path)], input=rgb.tobytes(), capture_output=True, check=True)


@pytest.fixture
def paired(tmp_path):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("needs ffmpeg and ffprobe")
    views = {view: f"observation.images.{view}" for view in ("top", "wrist")}
    rows = [{"frame_index": i, "episode_index": ep, "timestamp": 7 + i / FPS,
             "annotation.sub_goal": int(i >= length // 2),
             "annotation.sub_goal_score": 3 if i == length // 2 - 1 else 4 if i == length - 1 else 2,
             "annotation.frame_type": 6 if i == 2 else 7 if i == 3 else 5}
            for ep, length in enumerate(LENGTHS) for i in range(length)]
    info = {"fps": FPS, "chunks_size": 1000,
            "features": {key: {"dtype": "video", "shape": [24, 32, 3]} for key in views.values()}}
    v3, v2 = tmp_path / "v3", tmp_path / "v2"
    for root in (v3, v2):
        (root / "meta").mkdir(parents=True)
        (root / "data/chunk-000").mkdir(parents=True)
    (v3 / "meta/info.json").write_text(json.dumps(info | {
        "codebase_version": "v3.0", "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"}))
    (v2 / "meta/info.json").write_text(json.dumps(info | {
        "codebase_version": "v2.1", "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/{video_key}/episode_{episode_index:06d}.mp4"}))
    # Shared parquet is deliberately out of frame order and includes other episodes.
    pq.write_table(pa.Table.from_pylist(rows[::-1]), v3 / "data/chunk-000/file-000.parquet")
    pq.write_table(pa.Table.from_pylist([row for row in rows if row["episode_index"] == 1]),
                   v2 / "data/chunk-000/episode_000001.parquet")
    tasks = [{"task_index": i, "task": word} for i, word in enumerate(WORDS)]
    pq.write_table(pa.Table.from_pylist(tasks), v3 / "meta/tasks.parquet")
    (v2 / "meta/tasks.jsonl").write_text("".join(json.dumps(row) + "\n" for row in tasks))
    episodes = [{"episode_index": ep, "length": length, "trajectory_id": f"trajectory-{ep}",
                 "data/chunk_index": 0, "data/file_index": 0} for ep, length in enumerate(LENGTHS)]
    pixels = {}
    edges = np.cumsum([0, *LENGTHS]).tolist()
    for camera, (view, key) in enumerate(views.items()):
        prefix = camera * 3  # Each camera has its own nonzero video offset.
        rgb = np.zeros((prefix + sum(LENGTHS), 24, 32, 3), np.uint8)
        rgb[..., 0] = np.arange(len(rgb))[:, None, None] * 7
        rgb[..., 1] = 30 + camera * 100
        rgb[..., 2] = 230 - np.arange(len(rgb))[:, None, None] * 5
        pixels[view] = rgb[prefix + edges[1]:prefix + edges[2]]
        encode(v3 / f"videos/{key}/chunk-000/file-000.mp4", rgb)
        encode(v2 / f"videos/{key}/episode_000001.mp4", pixels[view])
        for ep, row in enumerate(episodes):
            row.update({f"videos/{key}/chunk_index": 0, f"videos/{key}/file_index": 0,
                        f"videos/{key}/from_timestamp": (prefix + edges[ep]) / FPS,
                        f"videos/{key}/to_timestamp": (prefix + edges[ep + 1]) / FPS})
    metadata = v3 / "meta/episodes/chunk-000/file-000.parquet"
    metadata.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(episodes), metadata)
    (v2 / "meta/episodes.jsonl").write_text(json.dumps({key: episodes[1][key] for key in ("episode_index", "length", "trajectory_id")}) + "\n")
    config = {"source": str(v3), "output": str(tmp_path / "cache-v3"), "episodes": [1], "task": "assembly",
              "compressed_task_prompt": "Place and secure the part.", "views": views, "fps": 5,
              "stages": [{"label": "1", "prompt": "Place the part"}, {"label": "2", "prompt": "Secure the part"}],
              "progress": {"method": "grana_target_composition", "parameter_status": "example",
                           "parameter_source": "Synthetic fixture", "stage_parameters": [
                               {"rho": 2, "p_cap": .8, "t_min_seconds": 1}] * 2}}
    return config, v2, pixels


def row_at(output):
    return load_from_disk(str(Path(output) / "processed_dataset"))[0]


def source_hashes(source):
    return {str(path.relative_to(source)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in Path(source).rglob("*") if path.is_file()}


@pytest.mark.parametrize("resize", [None, [16, 12]])
@pytest.mark.parametrize("legacy", [False, True])
def test_v3_shared_parquet_and_camera_ranges_match_v2_cache(paired, tmp_path, resize, legacy):
    config, v2, pixels = paired
    if resize:
        config["image_size"] = resize
    if legacy:
        config["progress"] = {"min_segment_frames": 1}
    before = source_hashes(config["source"])
    v3_report = build_cache(config)
    v2_cfg = config | {"source": str(v2), "output": str(tmp_path / "cache-v2")}
    build_cache(v2_cfg)
    actual, expected = row_at(config["output"]), row_at(v2_cfg["output"])
    for key in actual.keys() - {"frames", "frames_top", "frames_wrist"}:
        assert actual[key] == expected[key], key
    for view in config["views"]:
        np.testing.assert_array_equal(np.load(actual[f"frames_{view}"]), np.load(expected[f"frames_{view}"]))
        if resize is None:
            np.testing.assert_array_equal(np.load(actual[f"frames_{view}"]), pixels[view][actual["source_frame_indices"]])
    assert v3_report["source_version"] == "v3.0"
    assert set(v3_report["metadata_sha256"]) == {"info.json", "tasks.parquet", "episodes/chunk-000/file-000.parquet"}
    assert v3_report["episodes"][0]["videos"]["wrist"]["from_timestamp"] == 1.1
    assert before == source_hashes(config["source"])


def test_v3_resolves_independent_metadata_data_and_camera_locations(paired):
    config, _, pixels = paired
    source = Path(config["source"])
    metadata = source / "meta/episodes/chunk-000/file-000.parquet"
    episodes = pq.read_table(metadata).to_pylist()
    data = source / "data/chunk-002/file-007.parquet"
    camera = source / "videos/observation.images.wrist/chunk-004/file-006.mp4"
    for destination in (data, camera):
        destination.parent.mkdir(parents=True)
    (source / "data/chunk-000/file-000.parquet").rename(data)
    (source / "videos/observation.images.wrist/chunk-000/file-000.mp4").rename(camera)
    for episode in episodes:
        episode.update({"data/chunk_index": 2, "data/file_index": 7,
                        "videos/observation.images.wrist/chunk_index": 4,
                        "videos/observation.images.wrist/file_index": 6})
    relocated = source / "meta/episodes/chunk-003/file-009.parquet"
    relocated.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(episodes), relocated)
    metadata.unlink()
    tasks = source / "meta/tasks.parquet"
    pq.write_table(pq.read_table(tasks).rename_columns(["task_index", "__index_level_0__"]), tasks)
    report = build_cache(config)
    row = row_at(config["output"])
    for view in config["views"]:
        np.testing.assert_array_equal(np.load(row[f"frames_{view}"]), pixels[view][row["source_frame_indices"]])
    assert report["episodes"][0]["parquet"] == str(data)
    assert report["episodes"][0]["videos"]["wrist"]["path"] == str(camera)
    assert "episodes/chunk-003/file-009.parquet" in report["metadata_sha256"]


@pytest.mark.parametrize("range_shift", [0, .4, .49])
def test_v3_submitted_human_labels_build_only_the_selected_episode(paired, tmp_path, range_shift):
    config, _, pixels = paired
    source = Path(config["source"])
    parquet = source / "data/chunk-000/file-000.parquet"
    table = pq.read_table(parquet)
    pq.write_table(table.select(["frame_index", "episode_index", "timestamp"]), parquet)
    metadata = source / "meta/episodes/chunk-000/file-000.parquet"
    episodes = pq.read_table(metadata).to_pylist()
    for key in config["views"].values():
        for edge in ("from_timestamp", "to_timestamp"):
            episodes[1][f"videos/{key}/{edge}"] += range_shift / FPS
    pq.write_table(pa.Table.from_pylist(episodes), metadata)
    config.update(labels=str(tmp_path / "labels"), episodes="all", fps=FPS)
    project = LabelProject(config)
    try:
        project.save(1, {"base_revision": None, "project_revision": project.project()["revision"],
                         "subtasks": [{"start_frame": 0, "end_frame": 6, "stage": "1", "outcome": "high"},
                                      {"start_frame": 6, "end_frame": 12, "stage": "2", "outcome": "low"}],
                         "quality": [{"start_frame": 2, "end_frame": 3, "label": "suboptimal"},
                                     {"start_frame": 3, "end_frame": 4, "label": "regressed"}],
                         "annotator": "Fixture", "done": True, "review_confirmed": True})
    finally:
        project.close()
    report = build_cache(config)
    assert [row["episode_index"] for row in report["episodes"]] == [1]
    actual = row_at(config["output"])
    assert actual["source_frame_indices"] == list(range(12))
    assert actual["stage_ids"] == [0] * 6 + [1] * 6
    assert actual["action_quality"] == [0, 0, 1, 2] + [0] * 8
    assert actual["within_stage_progress"][3] == 0
    assert actual["within_stage_progress"][5] == 1
    assert actual["within_stage_progress"][-1] == 0
    assert actual["progress"][-1] == .5
    for view in config["views"]:
        np.testing.assert_array_equal(np.load(actual[f"frames_{view}"]), pixels[view])


def test_v3_reviewed_sidecar_has_the_same_targets_as_native_annotations(paired, tmp_path):
    config, _, _ = paired
    build_cache(config)
    expected = row_at(config["output"])
    # This input now has no native supervision; only the explicit sidecars can
    # supply AQ, stage identities and outcomes.
    parquet = Path(config["source"]) / "data/chunk-000/file-000.parquet"
    pq.write_table(pq.read_table(parquet).select(["frame_index", "episode_index", "timestamp"]), parquet)
    episode = LeRobotSource(Path(config["source"]), config["views"], allow_shared_videos=True).load_episode(1)
    candidate = {"segments": [
        {"start_frame": start, "end_frame": end, "stage_id": stage, "identity_status": "resolved",
         "reason": "Authored fixture", "evidence_frames": [start, end - 1]}
        for start, end, stage in ((0, 6, "S1"), (6, 12, "S2"))],
        "boundaries": [{"frame": 6, "lower_frame": 6, "upper_frame": 6, "reason": "Authored fixture", "evidence_frames": [5, 6]}],
        "omitted_stages": [], "invalid_intervals": [], "summary": "Authored fixture"}
    revision = {"candidate": candidate, "candidate_sha256": hashlib.sha256(json.dumps(candidate, sort_keys=True,
                separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest(),
                "stages": [{"id": f"S{i + 1}", **stage} for i, stage in enumerate(config["stages"])],
                "source_identity": episode.source_identity, "time_edges": list(episode.time_edges), "created_at": "fixture",
                "review": {"reviewer": "Fixture", "kind": "human", "decision": "accepted", "notes": "Authored fixture",
                           "evidence_inspected": ["Synthetic video"]}}
    supplement = {"source_identity": episode.source_identity, "reviewer": "Fixture", "provenance": "Authored fixture",
                  "boundary_policy": "nominal_dense", "aq_intervals": [
                      {"start_frame": start, "end_frame": end, "quality": quality, "reason": "Authored fixture"}
                      for start, end, quality in ((0, 2, "nominal"), (2, 3, "suboptimal"), (3, 4, "regressed"), (4, 12, "nominal"))],
                  "outcomes": [{"start_frame": start, "end_frame": end, "stage_id": stage, "outcome": outcome, "reason": "Authored fixture"}
                               for start, end, stage, outcome in ((0, 6, "S1", "high"), (6, 12, "S2", "low"))]}
    spec = {"revision": str(tmp_path / "revision.json"), "supplement": str(tmp_path / "supplement.json")}
    for key, doc in (("revision", revision), ("supplement", supplement)):
        Path(spec[key]).write_text(json.dumps(doc))
    config.update(output=str(tmp_path / "cache-sidecars"), annotations={"1": spec})
    build_cache(config)
    actual = row_at(config["output"])
    for key in ("source_frame_indices", "source_timestamps", "stage_ids", "within_stage_progress", "progress", "action_quality"):
        assert actual[key] == expected[key], key
    bad = deepcopy(supplement)
    bad["source_identity"]["episode_index"] = 0
    Path(spec["supplement"]).write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="source_identity"):
        build_cache(config | {"output": str(tmp_path / "cache-wrong-episode")})
    assert not (tmp_path / "cache-wrong-episode").exists()


@pytest.mark.parametrize("problem,match", [("bounds", "frame count/PTS"), ("rows", "contiguous"),
                                            ("clock", "align"), ("outcome", "outcome")])
def test_invalid_v3_alignment_or_supervision_never_publishes(paired, problem, match):
    config, _, _ = paired
    root = Path(config["source"])
    if problem == "bounds":
        path = root / "meta/episodes/chunk-000/file-000.parquet"
        rows = pq.read_table(path).to_pylist()
        rows[1]["videos/observation.images.top/to_timestamp"] -= 1 / FPS
    else:
        path = root / "data/chunk-000/file-000.parquet"
        rows = pq.read_table(path).to_pylist()
        row = next(row for row in rows if row["episode_index"] == 1 and row["frame_index"] == 5)
        if problem == "rows":
            row["frame_index"] = 4
        elif problem == "clock":
            row["timestamp"] += .01
        else:
            row["annotation.sub_goal_score"] = 2
    pq.write_table(pa.Table.from_pylist(rows), path)
    with pytest.raises(ValueError, match=match):
        build_cache(config)
    output = Path(config["output"])
    assert not output.exists()
    failures = list(output.parent.glob(f".{output.name}.partial-*/build_status.json"))
    assert len(failures) == 1 and json.loads(failures[0].read_text())["status"] == "failed"


def test_v3_decoder_refuses_an_ambiguous_adjacent_episode_frame(tmp_path):
    # Metadata excludes the preceding frame, but an irregular PTS puts it within
    # the decoder's match tolerance. Publishing one shifted frame would corrupt labels.
    source = boundary_source(tmp_path / "raw", shift=.4, stray=(FIRST - 1, FIRST - .2))
    config = boundary_recipe(source, tmp_path)
    config.update(episodes=[1])
    del config["labels"]
    path = source / "data/chunk-000/file-000.parquet"
    rows = pq.read_table(path).to_pylist()
    for row in rows:
        row.update({"annotation.sub_goal": 0, "annotation.sub_goal_score": 3 if row["frame_index"] == 14 else 2,
                    "annotation.frame_type": 5})
    pq.write_table(pa.Table.from_pylist(rows), path)
    pq.write_table(pa.Table.from_pylist([{"task_index": i, "task": word} for i, word in enumerate(WORDS)]), source / "meta/tasks.parquet")
    with pytest.raises(ValueError, match="length/frame shape|decode/alignment"):
        build_cache(config)
    assert not Path(config["output"]).exists()
