# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from grana.annotation.source import CLOCK_COLUMNS, LeRobotSource


@pytest.fixture
def raw(tmp_path, monkeypatch):
    ffmpeg = Path("/usr/bin/ffmpeg")
    if not ffmpeg.is_file():
        found = shutil.which("ffmpeg")
        if not found:
            pytest.skip("Requires ffmpeg")
        ffmpeg = Path(found)
    monkeypatch.setenv("PATH", str(ffmpeg.parent) + ":" + os.environ["PATH"])
    if not shutil.which("ffprobe"):
        pytest.skip("Requires ffprobe")
    source = tmp_path / "unannotated"
    (source / "meta").mkdir(parents=True)
    views = {"left": "observation.images.left", "top": "observation.images.top"}
    info = {"codebase_version": "v2.0", "fps": 10, "chunks_size": 1000,
            "data_path": "data/{episode_chunk}/episode_{episode_index}.parquet",
            "video_path": "videos/{video_key}/{episode_index}.mp4",
            "features": {key: {"dtype": "video", "shape": [24, 32, 3]} for key in views.values()}}
    (source / "meta/info.json").write_text(json.dumps(info))
    (source / "meta/episodes.jsonl").write_text(json.dumps({"episode_index": 0, "length": 6, "trajectory_id": "native-id"}) + "\n")
    table = {"frame_index": list(range(6)), "episode_index": [[0]] * 6,
             "timestamp": [[10 + i / 10] for i in range(6)]}
    parquet = source / "data/0/episode_0.parquet"
    parquet.parent.mkdir(parents=True)
    pq.write_table(pa.table(table), parquet)
    pixels = {}
    for camera, (view, key) in enumerate(views.items()):
        video = source / f"videos/{key}/0.mp4"
        video.parent.mkdir(parents=True)
        rgb = np.zeros((6, 24, 32, 3), dtype=np.uint8)
        rgb[..., 0] = np.arange(6)[:, None, None] * 30
        rgb[..., 1] = 50 + camera * 100
        rgb[..., 2] = 25
        subprocess.run([str(ffmpeg), "-v", "error", "-f", "rawvideo", "-pixel_format", "rgb24",
                        "-video_size", "32x24", "-framerate", "10", "-i", "pipe:0",
                        "-c:v", "libx264rgb", "-crf", "0", "-threads", "1", str(video)],
                       input=rgb.tobytes(), capture_output=True, check=True)
        pixels[view] = rgb
    return source, views, parquet, table, pixels


def test_unannotated_source_and_exact_multicamera_frame_projection(raw, tmp_path, monkeypatch):
    source, views, _, _, pixels = raw
    before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source.rglob("*") if p.is_file()}
    original, reads = pq.read_table, []
    def limited(path, **kwargs):
        reads.append(kwargs["columns"])
        return original(path, **kwargs)
    monkeypatch.setattr(pq, "read_table", limited)
    reader = LeRobotSource(source, views)
    episode = reader.load_episode(0)
    assert reads == [list(CLOCK_COLUMNS)]
    assert episode.frame_count == 6 and episode.view_names == ("left", "top")
    assert episode.timestamps == pytest.approx([10 + i / 10 for i in range(6)])
    assert episode.time_edges[-1] == pytest.approx(10.6)
    assert episode.source_identity["trajectory_id"] == "native-id"
    assert episode.clock_receipt["cameras"]["left"]["source_minus_video_offset_seconds"] == 10
    evidence = reader.make_evidence(episode, [4, 0, 5], tmp_path / "evidence", [32, 24])
    assert [x["frame"] for x in evidence] == [4, 0, 5]
    for item in evidence:
        image = np.asarray(Image.open(item["image_path"]))
        assert image.shape == (48, 64, 3)
        for camera, view in enumerate(views):
            np.testing.assert_array_equal(image[24:48, camera * 32:(camera + 1) * 32], pixels[view][item["frame"]])
        assert item["timestamp"] == episode.timestamps[item["frame"]]
        assert item["sha256"] == hashlib.sha256(Path(item["image_path"]).read_bytes()).hexdigest()
    assert before == {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source.rglob("*") if p.is_file()}
    with pytest.raises(FileExistsError):
        reader.make_evidence(episode, [4], tmp_path / "evidence", [32, 24])


def test_existing_ground_truth_and_actions_are_never_read(raw, monkeypatch):
    source, views, parquet, table, _ = raw
    table.update({"annotation.sub_goal": ["secret-label"] * 6,
                  "rl_info.abs_progress": [999.0] * 6, "action": [[999.0]] * 6})
    pq.write_table(pa.table(table), parquet)
    original = pq.read_table
    def guarded(path, **kwargs):
        assert kwargs["columns"] == list(CLOCK_COLUMNS)
        return original(path, **kwargs)
    monkeypatch.setattr(pq, "read_table", guarded)
    episode = LeRobotSource(source, views).load_episode(0)
    assert "secret-label" not in repr(episode) and "999.0" not in repr(episode)


@pytest.mark.parametrize("column,values,match", [
    ("frame_index", [0, 1, 2, 3, 4, 4], "contiguous"),
    ("episode_index", [0, 0, 1, 0, 0, 0], "contiguous"),
    ("timestamp", [0, .1, .1, .3, .4, .5], "strictly increasing"),
    ("timestamp", [0, .1, float("nan"), .3, .4, .5], "finite"),
    ("timestamp", [0, .1, .25, .3, .4, .5], "PTS does not align"),
])
def test_bad_native_clock_or_row_mapping_rejected_before_evidence(raw, column, values, match):
    source, views, parquet, table, _ = raw
    table[column] = values
    pq.write_table(pa.table(table), parquet)
    with pytest.raises(ValueError, match=match):
        LeRobotSource(source, views).load_episode(0)


def test_video_shape_and_changed_payload_fail_closed(raw, tmp_path):
    source, views, _, _, _ = raw
    reader = LeRobotSource(source, views)
    episode = reader.load_episode(0)
    video = episode.video_paths["left"]
    with video.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="changed"):
        reader.make_evidence(episode, [0], tmp_path / "evidence")
    metadata = json.loads((source / "meta/info.json").read_text())
    metadata["features"][views["top"]]["shape"] = [12, 32, 3]
    (source / "meta/info.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="shape"):
        LeRobotSource(source, views).load_episode(0)


def test_evidence_never_writes_into_source_or_checkout(raw):
    source, views, _, _, _ = raw
    reader = LeRobotSource(source, views)
    episode = reader.load_episode(0)
    for path in (source / "evidence", Path(__file__).resolve().parents[1] / ".local/forbidden-annotation-test"):
        with pytest.raises(ValueError, match="outside"):
            reader.make_evidence(episode, [0], path)
        assert not path.exists()


def test_source_path_escape_is_rejected(raw):
    source, views, _, _, _ = raw
    info_path = source / "meta/info.json"
    info = json.loads(info_path.read_text())
    info["data_path"] = "../outside.parquet"
    info_path.write_text(json.dumps(info))
    with pytest.raises(ValueError, match="inside"):
        LeRobotSource(source, views).load_episode(0)
