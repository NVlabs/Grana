# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""grana label plays a LeRobot v3 episode as a clip of exactly the frames its source identity covers."""
import json
import shutil
import subprocess

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from grana._ffmpeg import passthrough
from grana.annotation.label_server import LabelProject
from grana.annotation.source import LeRobotSource
from grana.data.builder import validate_build_config

FPS, LENGTHS, KEY = 10, (11, 15, 12), "observation.images.top"   # three episodes in one shared video
FIRST = LENGTHS[0]   # episode 1's first frame in the shared video


def pictures():
    """Distinct lossless frames: red counts the frames of the shared video, green tells episodes apart."""
    rgb = np.zeros((sum(LENGTHS), 24, 32, 3), np.uint8)
    rgb[..., 0] = np.arange(sum(LENGTHS))[:, None, None] * 6
    rgb[..., 1] = np.repeat([60, 150, 240], LENGTHS)[:, None, None]
    rgb[..., 2] = 100
    return rgb


def write_ranges(root, shift=0.0):
    """meta/episodes, with episode 1's time range starting ``shift`` frames after its first frame."""
    edges = np.cumsum((0, *LENGTHS)) + np.array([0, shift, shift, 0])
    rows = [{"episode_index": ep, "length": n, "trajectory_id": f"trajectory-{ep}", "tasks": ["demo"],
             "data/chunk_index": 0, "data/file_index": 0, f"videos/{KEY}/chunk_index": 0, f"videos/{KEY}/file_index": 0,
             f"videos/{KEY}/from_timestamp": edges[ep] / FPS, f"videos/{KEY}/to_timestamp": edges[ep + 1] / FPS}
            for ep, n in enumerate(LENGTHS)]
    pq.write_table(pa.Table.from_pylist(rows), root / "meta/episodes/chunk-000/file-000.parquet")


def make_source(root, shift=0.0, stray=None):
    """A v3 dataset whose shared video shows frame i at i / FPS, except frame ``stray[0]`` at ``stray[1]`` / FPS."""
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("needs ffmpeg and ffprobe")
    (root / "meta/episodes/chunk-000").mkdir(parents=True)
    (root / "data/chunk-000").mkdir(parents=True)
    (root / "meta/info.json").write_text(json.dumps(dict(
        codebase_version="v3.0", fps=FPS, chunks_size=1000, data_path="data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        video_path="videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        features={KEY: {"dtype": "video", "shape": [24, 32, 3]}})))
    pq.write_table(pa.table({"frame_index": [i for n in LENGTHS for i in range(n)],
                             "episode_index": [ep for ep, n in enumerate(LENGTHS) for _ in range(n)],
                             "timestamp": [i / FPS for n in LENGTHS for i in range(n)]}), root / "data/chunk-000/file-000.parquet")
    pq.write_table(pa.table({"task_index": [0], "task": ["demo"]}), root / "meta/tasks.parquet")
    write_ranges(root, shift)
    video = root / f"videos/{KEY}/chunk-000/file-000.mp4"
    video.parent.mkdir(parents=True)
    timing = (["-vf", f"settb=1/10240,setpts='if(eq(N,{stray[0]}),{stray[1]},N)/({FPS}*TB)'", *passthrough("ffmpeg"),
               "-enc_time_base", "1:10240"] if stray else [])
    subprocess.run(["ffmpeg", "-v", "error", "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", "32x24",
                    "-framerate", str(FPS), "-i", "pipe:0", *timing, "-c:v", "libx264rgb", "-crf", "0", "-g", "4", str(video)],
                   input=pictures().tobytes(), check=True, capture_output=True)
    return root


def recipe(source, tmp_path):
    return validate_build_config({
        "source": str(source), "output": str(tmp_path / "cache"), "labels": str(tmp_path / "labels"), "episodes": "all",
        "task": "assembly", "views": {"top": KEY}, "compressed_task_prompt": "Place, then secure.", "fps": 5,
        "stages": [{"label": "1", "prompt": "Place part"}, {"label": "2", "prompt": "Secure part"}],
        "progress": {"method": "grana_target_composition", "parameter_status": "example", "parameter_source": "Synthetic test",
                     "stage_parameters": [{"rho": 20, "p_cap": .95, "t_min_seconds": 1}] * 2}})


def shown(clip):
    """The frame of the shared video that each frame of a clip shows: the nearest picture."""
    raw = subprocess.check_output(["ffmpeg", "-v", "error", "-i", str(clip), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
    rgb = np.frombuffer(raw, np.uint8).reshape(-1, 1, 24, 32, 3).astype(int)
    return np.abs(rgb - pictures()[None].astype(int)).mean(axis=(2, 3, 4)).argmin(axis=1).tolist()


@pytest.mark.parametrize("shift", [0.0, 0.4, 0.49])
def test_v3_clip_plays_exactly_the_frames_its_identity_covers(tmp_path, shift):
    """An episode's frames are those up to half a frame before its time range. The editor checked episode 1's clock with
    the range at its first frame; moving the range less than half a frame later keeps the same frames and identity. A cut
    that seeks a quarter frame early used to drop the first frame and end the clip with episode 2's first frame."""
    source = make_source(tmp_path / "raw")
    project = LabelProject(recipe(source, tmp_path))
    try:
        project.clock(1)
        write_ranges(source, shift)
        assert LeRobotSource(source, {"top": KEY}, allow_shared_videos=True).load_episode(1).source_identity[
            "video_packets_sha256"] == project.dataset.load_episode(1).source_identity["video_packets_sha256"]
        assert shown(project.video(1, "top")) == list(range(FIRST, FIRST + LENGTHS[1]))
    finally:
        project.close()


def test_v3_clip_refuses_a_frame_its_identity_does_not_cover(tmp_path):
    """A frame outside the episode's range yet within a quarter frame of its first frame cannot be told apart from it
    while cutting: refuse rather than play a clip one frame off its labels."""
    from grana.annotation.label_server import _cut
    from grana.annotation.source import episode_frame_times

    source = make_source(tmp_path / "raw", shift=0.4, stray=(FIRST - 1, FIRST - 0.2))
    episode = LeRobotSource(source, {"top": KEY}, allow_shared_videos=True).load_episode(1)
    segment = episode.video_segments["top"]
    times = episode_frame_times(segment, FPS, episode.source_identity["video_packets_sha256"]["top"])
    assert list(times) == pytest.approx([i / FPS for i in range(FIRST, FIRST + LENGTHS[1])], abs=1e-4)
    clip = tmp_path / "clips/episode.mp4"
    with pytest.raises(ValueError, match=f"Could not cut the episode's {LENGTHS[1]} frames"):
        _cut(segment, times, FPS, clip)
    assert list(clip.parent.iterdir()) == []
