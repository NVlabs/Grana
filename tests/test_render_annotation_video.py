# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The review movie keeps every real frame and rejects invented time bases."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from scripts.render_annotation_video import render, uniform_fps

FFMPEG = "/usr/bin/ffmpeg" if Path("/usr/bin/ffmpeg").is_file() else shutil.which("ffmpeg")


@pytest.fixture
def annotation(tmp_path):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("System FFmpeg/FFprobe are required for rendering qualification")
    source, root = tmp_path / "source", tmp_path / "run"
    source.mkdir()
    (root / "episodes/000001").mkdir(parents=True)
    views = {}
    for view, color in (("exterior_1", "red"), ("exterior_2", "green"), ("wrist", "blue")):
        path = source / f"{view}.mp4"
        subprocess.run([FFMPEG, "-v", "error", "-nostdin", "-n", "-f", "lavfi", "-i",
                        f"color=c={color}:s=32x24:r=10", "-frames:v", "6", "-c:v", "libx264",
                        "-threads", "1", "-pix_fmt", "yuv420p", str(path)], check=True, capture_output=True)
        views[view] = str(path)
    candidate = {
        "segments": [
            {"start_frame": 0, "end_frame": 3, "stage_id": "S1", "identity_status": "resolved",
             "reason": "Synthetic visible stage", "evidence_frames": [0, 2]},
            {"start_frame": 3, "end_frame": 6, "stage_id": None, "identity_status": "unresolved",
             "reason": "Synthetic occlusion", "evidence_frames": [3, 5]},
        ],
        "boundaries": [{"frame": 3, "lower_frame": 2, "upper_frame": 4,
                        "reason": "Synthetic boundary bracket", "evidence_frames": [2, 3, 4]}],
        "omitted_stages": [], "invalid_intervals": [{"start_frame": 4, "end_frame": 5, "reason": "Test invalid span"}],
        "summary": "Synthetic renderer fixture; never a real candidate.",
    }
    (root / "episodes/000001/final.json").write_text(json.dumps(candidate))
    manifest = {
        "schema_version": 1, "task": "Synthetic renderer test", "stages": [{"id": "S1", "label": "1", "prompt": "Open"}],
        "episodes": [{"episode_index": 1, "frame_count": 6, "time_edges": [10 + i / 10 for i in range(7)],
                      "video_paths": views, "candidate_file": "episodes/000001/final.json",
                      "source_identity": {"source_root": str(source), "video_sha256": {
                          view: hashlib.sha256(Path(path).read_bytes()).hexdigest() for view, path in views.items()}},
                      "clock_receipt": {"cameras": {view: {"source_minus_video_offset_seconds": 10} for view in views}}}],
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root, source, manifest, tmp_path / "rendered/episode.mp4"


def test_real_cpu_encode_keeps_all_frames_three_views_and_immutable_inputs(annotation):
    root, source, manifest, output = annotation
    originals = {p: p.read_bytes() for p in [root / "manifest.json", root / "episodes/000001/final.json", *source.glob("*.mp4")]}
    result = render(root, 1, output)
    assert result["frames"] == 6 and result["fps"] == "10"
    assert result["human_review"] == "pending" and result["source_seconds"] == pytest.approx(.6)
    assert result["source_identity"] == manifest["episodes"][0]["source_identity"]
    assert result["video_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    raw = subprocess.run([FFMPEG, "-v", "error", "-i", str(output), "-frames:v", "1", "-f", "rawvideo",
                          "-pix_fmt", "rgb24", "pipe:1"], check=True, capture_output=True).stdout
    # Sample each view away from overlaid text; all three original views survive composition.
    for x, channel in ((160, 0), (480, 1), (800, 2)):
        position = (225 * 960 + x) * 3
        pixel = raw[position:position + 3]
        assert pixel[channel] > 100 and all(pixel[other] < 15 for other in range(3) if other != channel)
    assert all(p.read_bytes() == data for p, data in originals.items())
    with pytest.raises(ValueError, match="fresh"):
        render(root, 1, output)
    assert not list(output.parent.glob(".*.mp4"))


@pytest.mark.parametrize("view_count", [1, 4])
def test_one_and_four_camera_movies_preserve_frame_count(annotation, view_count):
    root, _, manifest, output = annotation
    episode = manifest["episodes"][0]
    if view_count == 1:
        episode["video_paths"] = {"exterior_1": episode["video_paths"]["exterior_1"]}
    else:
        episode["video_paths"]["bottom"] = episode["video_paths"]["exterior_1"]
        episode["source_identity"]["video_sha256"]["bottom"] = episode["source_identity"]["video_sha256"]["exterior_1"]
        episode["clock_receipt"]["cameras"]["bottom"] = {"source_minus_video_offset_seconds": 10}
    (root / "manifest.json").write_text(json.dumps(manifest))
    result = render(root, 1, output)
    assert result["frames"] == 6 and result["fps"] == "10"
    probe = json.loads(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                      "stream=width,height,nb_frames", "-of", "json", str(output)],
                                     check=True, capture_output=True, text=True).stdout)["streams"][0]
    assert probe["width"] == max(960, view_count * 320) and probe["height"] == 450
    assert int(probe["nb_frames"]) == 6


@pytest.mark.parametrize("edges", [[0, .1, .25, .35], [0, .1, .1], [0, float("nan"), .2], [0, True, 2]])
def test_nonuniform_or_invalid_source_clock_is_rejected(edges):
    with pytest.raises(ValueError):
        uniform_fps(edges)


def test_fractional_source_rate_is_preserved():
    assert str(uniform_fps([4 + i * 1001 / 30000 for i in range(101)])) == "30000/1001"


@pytest.mark.parametrize("kind", ["nonuniform", "camera_offset"])
def test_mismatched_real_clock_refuses_output(annotation, kind):
    root, _, manifest, output = annotation
    episode = manifest["episodes"][0]
    if kind == "nonuniform":
        episode["time_edges"][2] += .01
    else:
        episode["clock_receipt"]["cameras"]["wrist"]["source_minus_video_offset_seconds"] += .01
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="clock"):
        render(root, 1, output)
    assert not output.exists()


def test_renderer_never_writes_into_source_or_annotation_run(annotation):
    root, source, _, _ = annotation
    for output in (source / "new.mp4", root / "new.mp4"):
        with pytest.raises(ValueError, match="outside"):
            render(root, 1, output)
        assert not output.exists()
