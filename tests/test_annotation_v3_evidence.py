# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evidence-reader coverage selected from the unified annotation branch.

Only source reading/decoding is adopted here, independently of its UI/cache refactor.
"""
from functools import partial
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from grana.annotation.source import LeRobotSource as SourceReader
from grana.data.lerobot import LeRobotDataset

FPS, LENGTHS, VIEWS = 10, {0: 20, 1: 30}, {"top": "observation.images.top", "wrist": "observation.images.wrist"}
LeRobotSource = partial(SourceReader, allow_shared_videos=True)

def pixels(episode, view, length=None):
    """Distinct lossless frames: red counts frames, green tells episodes apart, blue tells cameras apart."""
    length = LENGTHS[episode] if length is None else length
    rgb = np.zeros((length, 24, 32, 3), np.uint8)
    rgb[..., 0] = np.arange(length)[:, None, None] * 7
    rgb[..., 1] = (40 + episode * 120) % 256
    rgb[..., 2] = 30 if view == "top" else 200
    return rgb



def encode(path, rgb):
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-v", "error", "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", "32x24",
                    "-framerate", str(FPS), "-i", "pipe:0", "-c:v", "libx264rgb", "-crf", "0", "-g", "4", str(path)],
                   input=rgb.tobytes(), check=True, capture_output=True)



def info(version, data_path, video_path):
    return dict(codebase_version=version, fps=FPS, chunks_size=1000, data_path=data_path, video_path=video_path,
                features={key: {"dtype": "video", "shape": [24, 32, 3]} for key in VIEWS.values()})



@pytest.fixture
def datasets(tmp_path):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("needs ffmpeg and ffprobe")
    rows = lambda ep: {"frame_index": list(range(LENGTHS[ep])), "episode_index": [ep] * LENGTHS[ep],
                       "timestamp": [i / FPS for i in range(LENGTHS[ep])]}
    # v2: one parquet and one video per camera for each episode.
    v2 = tmp_path / "v2"
    (v2 / "meta").mkdir(parents=True)
    (v2 / "meta/info.json").write_text(json.dumps(info("v2.1", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
                                                       "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4")))
    (v2 / "meta/episodes.jsonl").write_text("".join(json.dumps(dict(episode_index=ep, length=n, trajectory_id=f"trajectory-{ep}")) + "\n"
                                                    for ep, n in LENGTHS.items()))
    for ep in LENGTHS:
        path = v2 / f"data/chunk-000/episode_{ep:06d}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table(rows(ep)), path)
        for view, key in VIEWS.items():
            encode(v2 / f"videos/chunk-000/{key}/episode_{ep:06d}.mp4", pixels(ep, view))
    # v3: both episodes in one parquet and one video per camera, located through meta/episodes.
    v3 = tmp_path / "v3"
    (v3 / "meta/episodes/chunk-000").mkdir(parents=True)
    (v3 / "meta/info.json").write_text(json.dumps(info("v3.0", "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
                                                       "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4")))
    (v3 / "data/chunk-000").mkdir(parents=True)
    pq.write_table(pa.concat_tables([pa.table(rows(ep)) for ep in LENGTHS]), v3 / "data/chunk-000/file-000.parquet")
    for view, key in VIEWS.items():
        encode(v3 / f"videos/{key}/chunk-000/file-000.mp4", np.concatenate([pixels(ep, view) for ep in LENGTHS]))
    starts = {0: 0.0, 1: LENGTHS[0] / FPS}
    episodes = [{"episode_index": ep, "length": n, "trajectory_id": f"trajectory-{ep}", "tasks": ["demo"],
                 "data/chunk_index": 0, "data/file_index": 0,
                 **{f"videos/{key}/{name}": value for key in VIEWS.values()
                    for name, value in (("chunk_index", 0), ("file_index", 0), ("from_timestamp", starts[ep]),
                                        ("to_timestamp", starts[ep] + n / FPS))}} for ep, n in LENGTHS.items()]
    pq.write_table(pa.Table.from_pylist(episodes), v3 / "meta/episodes/chunk-000/file-000.parquet")
    pq.write_table(pa.table({"task_index": [0], "task": ["demo"]}), v3 / "meta/tasks.parquet")
    return v2, v3



def test_v3_reads_rows_and_video_ranges(datasets):
    _, v3 = datasets
    dataset = LeRobotDataset(v3)
    assert dataset.version == 3 and sorted(dataset.episodes) == [0, 1] and dataset.tasks == {0: "demo"}
    rows = dataset.rows(1, ["frame_index", "timestamp"])
    assert rows["frame_index"] == list(range(LENGTHS[1])) and rows["timestamp"][-1] == pytest.approx((LENGTHS[1] - 1) / FPS)
    segment = dataset.video(1, VIEWS["top"])
    assert segment.shared and segment.start == pytest.approx(2.0) and segment.end == pytest.approx(5.0)
    assert LeRobotDataset(datasets[0]).video(1, VIEWS["top"]).shared is False



def append_episode(v3, ep, length):
    """Add an episode the way LeRobot v3 does: rows appended to the shared parquet, packets stream-copied onto each video
    by FFmpeg's concat demuxer with its defaults, which also copy the H.264 parameter sets into every keyframe."""
    dataset = LeRobotDataset(v3)
    start = sum(dataset.length(index) for index in dataset.episodes) / FPS
    rows = {"frame_index": list(range(length)), "episode_index": [ep] * length, "timestamp": [i / FPS for i in range(length)]}
    data = v3 / "data/chunk-000/file-000.parquet"
    pq.write_table(pa.concat_tables([pq.read_table(data), pa.table(rows)]), data)
    for view, key in VIEWS.items():
        shared = v3 / f"videos/{key}/chunk-000/file-000.mp4"
        extra, listing, joined = (shared.with_name(name) for name in ("extra.mp4", "list.txt", "joined.mp4"))
        encode(extra, pixels(ep, view, length))
        listing.write_text(f"file '{shared}'\nfile '{extra}'\n")
        subprocess.run(["ffmpeg", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(listing),
                        "-c", "copy", str(joined)], check=True, capture_output=True)
        joined.replace(shared)
        extra.unlink()
        listing.unlink()
    meta = v3 / "meta/episodes/chunk-000/file-000.parquet"
    rows = pq.read_table(meta).to_pylist()
    rows.append({**rows[-1], "episode_index": ep, "length": length, "trajectory_id": f"trajectory-{ep}",
                 **{f"videos/{key}/{name}": value for key in VIEWS.values()
                    for name, value in (("from_timestamp", start), ("to_timestamp", start + length / FPS))}})
    pq.write_table(pa.Table.from_pylist(rows), meta)



@pytest.mark.parametrize("index,offset", [(0, 0), (1, 3)])
def test_v3_evidence_uses_exact_native_ranks_and_preserves_clip_clock(datasets, tmp_path, index, offset):
    selected = [LENGTHS[index] - 1, offset, offset + 4]
    outputs = []
    for source in datasets:
        reader = LeRobotSource(source, VIEWS)
        episode = reader.load_episode(index)
        directory = tmp_path / f"evidence-{source.name}"
        records = reader.make_evidence(episode, selected, directory, (32, 24), frame_offset=offset)
        assert [record["frame"] for record in records] == [frame - offset for frame in selected]
        assert [record["timestamp"] for record in records] == [episode.timestamps[frame] for frame in selected]
        for frame, record in zip(selected, records, strict=True):
            mosaic = np.asarray(Image.open(record["image_path"]))
            for camera, view in enumerate(VIEWS):
                np.testing.assert_array_equal(mosaic[24:48, camera * 32:(camera + 1) * 32], pixels(index, view)[frame])
        assert sorted(path.name for path in directory.iterdir()) == sorted(Path(record["image_path"]).name for record in records)
        outputs.append([Path(record["image_path"]).read_bytes() for record in records])
    assert outputs[0] == outputs[1]



@pytest.mark.parametrize("change", ["clock", "video"])
def test_v3_evidence_revalidates_its_own_episode_before_output(datasets, tmp_path, change):
    _, v3 = datasets
    reader = LeRobotSource(v3, VIEWS)
    episode = reader.load_episode(1)
    if change == "video":
        top, wrist = (v3 / f"videos/{key}/chunk-000/file-000.mp4" for key in VIEWS.values())
        shutil.copyfile(wrist, top)
    else:
        data = v3 / "data/chunk-000/file-000.parquet"
        rows = pq.read_table(data).to_pydict()
        rows["timestamp"] = [t + 5 if ep == 1 else t for t, ep in zip(rows["timestamp"], rows["episode_index"])]
        pq.write_table(pa.table(rows), data)
    directory = tmp_path / "evidence"
    with pytest.raises(ValueError, match="Source episode changed"):
        reader.make_evidence(episode, [0, LENGTHS[1] - 1], directory)
    assert not directory.exists()


def test_v3_evidence_never_reads_supervision_state_or_action(datasets, tmp_path, monkeypatch):
    _, v3 = datasets
    reader = LeRobotSource(v3, VIEWS)
    original = reader.load_episode(1)
    path = v3 / "data/chunk-000/file-000.parquet"
    rows = pq.read_table(path).to_pydict()
    for name in ("annotation.stage", "rl_info.failure", "observation.state", "action"):
        rows[name] = ["PRIVATE_HIDDEN_VALUE"] * len(rows["frame_index"])
    pq.write_table(pa.table(rows), path)
    read, accesses = pq.read_table, []

    def clock_only(path, *args, **kwargs):
        if Path(path).is_relative_to(v3 / "data"):
            columns = kwargs.get("columns")
            assert columns is not None and set(columns) <= {"frame_index", "episode_index", "timestamp"}
            accesses.append(columns)
        return read(path, *args, **kwargs)

    monkeypatch.setattr(pq, "read_table", clock_only)
    current = LeRobotSource(v3, VIEWS).load_episode(1)
    assert current.source_identity == original.source_identity
    records = reader.make_evidence(original, [0, 29], tmp_path / "evidence", (32, 24))
    assert accesses and len(records) == 2
    assert "PRIVATE_HIDDEN_VALUE" not in json.dumps(current.source_identity)



def test_v3_evidence_allows_appending_other_episodes(datasets, tmp_path):
    _, v3 = datasets
    reader = LeRobotSource(v3, VIEWS)
    episode = reader.load_episode(1)
    append_episode(v3, 2, 10)
    records = reader.make_evidence(episode, [0, LENGTHS[1] - 1], tmp_path / "evidence", (32, 24))
    for record in records:
        mosaic = np.asarray(Image.open(record["image_path"]))
        np.testing.assert_array_equal(mosaic[24:48, :32], pixels(1, "top")[record["frame"]])



def test_v3_evidence_seeks_nonkeyframe_episode_start_with_nonzero_source_clock(datasets, tmp_path):
    _, v3 = datasets
    for view, key in VIEWS.items():
        video = v3 / f"videos/{key}/chunk-000/file-000.mp4"
        rgb = np.concatenate([pixels(ep, view) for ep in LENGTHS])
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pixel_format", "rgb24",
                        "-video_size", "32x24", "-framerate", str(FPS), "-i", "pipe:0",
                        "-c:v", "libx264rgb", "-crf", "0", "-g", "60", "-sc_threshold", "0", str(video)],
                       input=rgb.tobytes(), capture_output=True, check=True)
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "frame=key_frame",
                            "-of", "json", str(video)], capture_output=True, text=True, check=True)
    assert json.loads(probe.stdout)["frames"][LENGTHS[0]]["key_frame"] == 0
    data = v3 / "data/chunk-000/file-000.parquet"
    rows = pq.read_table(data).to_pydict()
    rows["timestamp"] = [10 + t for t in rows["timestamp"]]
    pq.write_table(pa.table(rows), data)
    reader = LeRobotSource(v3, VIEWS)
    selected = [0, 9, LENGTHS[1] - 1]
    records = reader.make_evidence(1, selected, tmp_path / "evidence", (32, 24))
    assert [record["timestamp"] for record in records] == pytest.approx([10 + frame / FPS for frame in selected])
    for frame, record in zip(selected, records, strict=True):
        mosaic = np.asarray(Image.open(record["image_path"]))
        for camera, view in enumerate(VIEWS):
            np.testing.assert_array_equal(mosaic[24:48, camera * 32:(camera + 1) * 32], pixels(1, view)[frame])



def cameras(record, size=(32, 24)):
    """Each camera's image in an evidence mosaic: two to a row, each under a 24-pixel caption."""
    mosaic, (width, height) = np.asarray(Image.open(record["image_path"])), size
    return {view: mosaic[(i // 2) * (height + 24) + 24:(i // 2 + 1) * (height + 24), (i % 2) * width:(i % 2 + 1) * width]
            for i, view in enumerate(record["view_names"])}



def test_v3_evidence_shows_each_episodes_own_frames(datasets, tmp_path):
    """The first episode of the shared files, one 2 s into them, and one appended the way LeRobot appends."""
    v2, v3 = datasets
    files = lambda root: {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
    second = LeRobotSource(v3, VIEWS).load_episode(1)   # read before the append, which leaves its packets as they were
    append_episode(v3, 2, 10)
    before = files(v3)
    reader, records = LeRobotSource(v3, VIEWS), {}
    for episode, frames in ((reader.load_episode(0), [19, 0, 7]), (second, [0, 29, 13, 14]), (reader.load_episode(2), [9, 0])):
        ep = episode.episode_index
        records[ep] = reader.make_evidence(episode, frames, tmp_path / f"v3-{ep}", (32, 24))
        assert [row["frame"] for row in records[ep]] == frames   # in the order asked
        for row in records[ep]:
            assert row["timestamp"] == episode.timestamps[row["frame"]] and row["view_names"] == list(VIEWS)
            for view, image in cameras(row).items():
                np.testing.assert_array_equal(image, pixels(ep, view, episode.frame_count)[row["frame"]])
        assert sorted(path.name for path in (tmp_path / f"v3-{ep}").iterdir()) == [
            f"episode_{ep:06d}_frame_{frame:06d}.png" for frame in sorted(frames)]
    assert files(v3) == before   # nothing written beside the dataset
    # The same frames as LeRobot v2 evidence, at the native size and resized: the same mosaic bytes.
    same = LeRobotSource(v2, VIEWS).make_evidence(1, [0, 29, 13, 14], tmp_path / "v2-1", (32, 24))
    assert [row["sha256"] for row in same] == [row["sha256"] for row in records[1]]
    resized = [LeRobotSource(source, VIEWS).make_evidence(1, [3, 28], tmp_path / f"{name}-resized", (48, 40), frame_offset=3)
               for name, source in (("v2", v2), ("v3", v3))]
    assert [row["sha256"] for row in resized[0]] == [row["sha256"] for row in resized[1]]
    assert Image.open(resized[1][0]["image_path"]).size == (96, 64) and [row["frame"] for row in resized[1]] == [0, 25]



def test_v3_evidence_takes_frames_only_at_their_presentation_times(datasets, tmp_path, monkeypatch):
    """Episode 0 runs on into episode 1, so a decoder counting frames from a seek point would show each frame's successor."""
    import grana.annotation.source as module

    _, v3 = datasets
    probe = module._segment_probe

    def half_a_frame_late(segment, fps):   # no frame is shown at these times, so none may be taken by its position
        width, height, pts, packets = probe(segment, fps)
        return width, height, [time + 0.5 / fps for time in pts], packets

    # In place before the episode is read, so its clock and the evidence revalidation agree: only decoding can refuse.
    monkeypatch.setattr(module, "_segment_probe", half_a_frame_late)
    reader = LeRobotSource(v3, VIEWS)
    with pytest.raises(ValueError, match="Could not decode exactly the requested frames of top from file-000.mp4"):
        reader.make_evidence(reader.load_episode(0), [3, 4], tmp_path / "evidence")
    assert not (tmp_path / "evidence").exists()



def test_v3_evidence_takes_any_number_of_frames_at_once(datasets, tmp_path):
    """One FFmpeg select per camera: written as a flat sum, FFmpeg would refuse 99 or more frames."""
    _, v3 = datasets
    append_episode(v3, 2, 130)   # after both episodes in the shared files
    records = LeRobotSource(v3, VIEWS).make_evidence(2, list(range(130)), tmp_path / "evidence", (32, 24))
    assert [row["frame"] for row in records] == list(range(130))
    for row in records:
        for view, image in cameras(row).items():
            np.testing.assert_array_equal(image, pixels(2, view, 130)[row["frame"]])
