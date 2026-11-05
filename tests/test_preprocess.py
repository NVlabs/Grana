# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise native annotation/video alignment through the public cache command."""
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
import yaml

from grana.data.cache import CachedTrajectories
from grana.data.builder import build_cache, load_build_config, validate_build_config


@pytest.fixture
def native(tmp_path, monkeypatch):
    # Use a real, lossless two-camera video so wrong frame/view order is visible.
    ffmpeg = "/usr/bin/ffmpeg" if Path("/usr/bin/ffmpeg").exists() else shutil.which("ffmpeg")
    if not ffmpeg or not shutil.which("ffprobe"):
        pytest.skip("Native video integration requires ffmpeg and ffprobe")
    monkeypatch.setenv("PATH", str(Path(ffmpeg).parent) + ":" + __import__("os").environ["PATH"])
    source = tmp_path / "raw"
    (source / "meta").mkdir(parents=True)
    keys = {view: f"observation.images.{view}" for view in ("top", "left")}
    info = dict(codebase_version="v2.1", fps=10, chunks_size=1000,
                data_path="data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
                video_path="videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
                features={key: {"dtype": "video", "shape": [24, 32, 3]} for key in keys.values()})
    (source / "meta/info.json").write_text(json.dumps(info))
    (source / "meta/episodes.jsonl").write_text(json.dumps(dict(episode_index=0, length=30, trajectory_id="unique-source-episode")) + "\n")
    words = ["1", "2", "trailing", "empty", "high", "normal", "suboptimal", "failure"]
    (source / "meta/tasks.jsonl").write_text("".join(json.dumps(dict(task_index=i, task=word)) + "\n" for i, word in enumerate(words)))
    table = {"frame_index": list(range(30)), "episode_index": [[0]] * 30,
             "timestamp": [[i / 10] for i in range(30)],
             "annotation.sub_goal": [[0]] * 12 + [[1]] * 12 + [[2]] * 6,
             "annotation.sub_goal_score": [4 if i in (11, 23) else 3 for i in range(30)],
             "annotation.frame_type": [5] * 30,
             "rl_info.action_quality_mask": [int(i == 5) for i in range(30)],
             "rl_info.abs_progress": [99.0] * 30}
    parquet = source / "data/chunk-000/episode_000000.parquet"
    parquet.parent.mkdir(parents=True)
    pq.write_table(pa.table(table), parquet)
    videos, pixels = {}, {}
    for offset, (view, key) in enumerate(keys.items()):
        path = source / f"videos/chunk-000/{key}/episode_000000.mp4"
        path.parent.mkdir(parents=True)
        rgb = np.zeros((30, 24, 32, 3), np.uint8)
        rgb[..., 0] = np.arange(30)[:, None, None] * 5
        rgb[..., 1] = 50 + offset * 100
        rgb[..., 2] = 25
        subprocess.run([ffmpeg, "-v", "error", "-f", "rawvideo", "-pixel_format", "rgb24",
                        "-video_size", "32x24", "-framerate", "10", "-i", "pipe:0",
                        "-c:v", "libx264rgb", "-crf", "0", "-threads", "1", str(path)],
                       input=rgb.tobytes(), capture_output=True, check=True)
        videos[view], pixels[view] = path, rgb
    config = dict(source=str(source), output=str(tmp_path / "cache"), episodes=[0], task="drawer",
                  compressed_task_prompt="Open and close the drawer.",
                  stages=[dict(label="1", prompt="Open drawer"), dict(label="2", prompt="Close drawer")],
                  views=keys, fps=5, progress=dict(min_segment_frames=1, max_trailing_seconds=.2))
    return config, table, parquet, videos, pixels


def test_cli_build_preserves_pixels_labels_and_source_identity(native, tmp_path, monkeypatch, capsys):
    from grana.cli import main

    config, _, _, _, pixels = native
    source = Path(config["source"])
    before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source.rglob("*") if p.is_file()}
    recipe = tmp_path / "cache.yaml"
    recipe.write_text(yaml.safe_dump(config, sort_keys=False))
    monkeypatch.setattr("sys.argv", ["grana", "build-cache", "--config", str(recipe)])
    main()
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "complete"
    assert report["sampling_method"] == "linspace_v1"
    assert report["episodes"][0]["quality_source"] == "rl_info.action_quality_mask"
    assert report["episodes"][0]["quality_disagreement_frames"] == 1
    cache = Path(config["output"])
    row = load_from_disk(str(cache / "processed_dataset"))[0]
    indices = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 25]
    assert row["id"] == "unique-source-episode"
    assert row["source_frame_indices"] == indices
    assert row["source_timestamps"] == pytest.approx([i / 10 for i in indices])
    assert row["stage_ids"] == [0] * 6 + [1] * 7
    assert row["progress"][0] == pytest.approx(5 / 55.1 / 2)
    assert row["progress"][-1] == 1
    assert row["frames_shape"] == [13, 24, 32, 3]
    for view, rgb in pixels.items():
        np.testing.assert_array_equal(np.load(row[f"frames_{view}"]), rgb[indices])
    data = CachedTrajectories([cache], views=["top", "left"])
    assert data.summary()["eligible_rows"] == {"A": 1, "B": 1, "C": 1}
    assert len(data.records[0].images((0, 6, 12))) == 6
    assert before == {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source.rglob("*") if p.is_file()}
    with pytest.raises(FileExistsError):
        build_cache(config)


def test_frame_type_fallback_and_explicit_resize(native):
    config, table, path, _, _ = native
    del table["rl_info.action_quality_mask"]
    table["annotation.frame_type"][5] = 6
    pq.write_table(pa.table(table), path)
    config["image_size"] = [16, 12]
    report = build_cache(config)
    assert report["episodes"][0]["quality_source"] == "annotation.frame_type"
    row = load_from_disk(str(Path(config["output"]) / "processed_dataset"))[0]
    assert row["frames_shape"] == [13, 12, 16, 3]
    assert row["progress"][0] == pytest.approx(5 / 55.1 / 2)


@pytest.mark.parametrize("column,value,match", [
    ("annotation.sub_goal", [99], "absent tasks"),
    ("annotation.sub_goal", [5], "Unrecognized raw stage"),
    ("annotation.sub_goal_score", 5, "scores"),
    ("rl_info.action_quality_mask", 3, "0/1/2"),
    ("frame_index", 1, "contiguous"),
    ("episode_index", [2], "episode_index"),
    ("timestamp", [.5], "timestamps"),
])
def test_malformed_annotations_never_publish_a_cache(native, column, value, match):
    config, table, path, _, _ = native
    table[column][0] = value
    pq.write_table(pa.table(table), path)
    with pytest.raises(ValueError, match=match):
        build_cache(config)
    output = Path(config["output"])
    assert not output.exists()
    status = list(output.parent.glob(".cache.partial-*/build_status.json"))
    assert len(status) == 1 and json.loads(status[0].read_text())["status"] == "failed"


def test_decoder_checks_unselected_tail(native, monkeypatch):
    import grana.data.builder as module

    config, _, _, videos, _ = native
    # Hide the container count; full decoding must still catch an extra tail frame.
    probe = module._probe_video
    monkeypatch.setattr(module, "_probe_video", lambda path: {**probe(path), "frames": None})
    original = module._decode_video
    monkeypatch.setattr(module, "_decode_video", lambda path, dest, indices, count, size, image_size=None:
                        original(path, dest, indices, count - 1, size, image_size))
    with pytest.raises(ValueError, match="length/frame shape"):
        build_cache(config)
    assert not Path(config["output"]).exists()


def test_config_paths_and_scope_are_explicit(native, tmp_path, monkeypatch):
    config, _, _, _, _ = native
    monkeypatch.setenv("TEST_GRANA_RAW", config["source"])
    recipe = tmp_path / "portable.yaml"
    recipe.write_text(yaml.safe_dump(config | {"source": "${TEST_GRANA_RAW}", "output": "relative-cache"}))
    loaded = load_build_config(recipe)
    assert loaded["source"] == config["source"]
    assert loaded["output"] == str(tmp_path / "relative-cache")
    for changes in ({"source": "${UNDEFINED_GRANA_TEST_VAR}"}, {"output": str(Path(config["source"]) / "cache")},
                    {"unknown": True}, {"episodes": [0, 0]}, {"fps": float("nan")},
                    {"stages": [dict(label="1", prompt="Open"), dict(label="2", prompt="Then Open")]}):
        with pytest.raises(ValueError):
            validate_build_config(config | changes)
    info_path = Path(config["source"]) / "meta/info.json"
    info = json.loads(info_path.read_text())
    info["codebase_version"] = "v3.0"
    info_path.write_text(json.dumps(info))
    with pytest.raises(ValueError, match="v2"):
        build_cache(config)


def composition_config(config):
    return config | {"progress": {
        "method": "grana_target_composition",
        "stage_parameters": [{"rho": 2, "p_cap": .8, "t_min_seconds": 1}] * 2,
        "parameter_status": "example", "parameter_source": "Synthetic test, not task calibration",
    }}


def test_composition_build_uses_native_aq_before_sampling_and_keeps_exact_rgb(native):
    config, table, path, _, pixels = native
    # First attempt: t=0..0.2, failure at t=.3 -> endpoint .8*.3=.24.
    # Failure at .3/.4; recovery .5..1.1 has one weight-1 then six weight-2 updates.
    table["annotation.frame_type"][0] = 3  # Unmarked AQ is nominal.
    table["annotation.frame_type"][3:6] = [7, 7, 6]
    table["rl_info.action_quality_mask"] = [0] * 30  # Stale mask must not replace annotations.
    pq.write_table(pa.table(table), path)
    config = composition_config(config)
    report = build_cache(config)
    assert report["episodes"][0]["quality_source"] == "annotation.frame_type"
    assert report["episodes"][0]["quality_disagreement_frames"] == 3
    assert report["episodes"][0]["chunks"]
    row = load_from_disk(str(Path(config["output"]) / "processed_dataset"))[0]
    indices = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 23]
    assert row["source_frame_indices"] == indices  # No legacy trailing hold.
    expected = [0, .24, 0, 3 / 13, 7 / 13, 11 / 13, 0, 2 / 11, 4 / 11, 6 / 11, 8 / 11, 1]
    assert row["within_stage_progress"] == pytest.approx(expected)
    assert row["progress"] == pytest.approx([(stage + p) / 2 for stage, p in zip([0]*6+[1]*6, expected)])
    assert row["progress_method"] == "grana_target_composition"
    assert json.loads(row["progress_config_json"]) == report["config"]["progress"]
    for view, rgb in pixels.items():
        np.testing.assert_array_equal(np.load(row[f"frames_{view}"]), rgb[indices])
    assert CachedTrajectories([config["output"]], views=["top", "left"]).summary()["eligible_rows"] == {"A": 1, "B": 1, "C": 1}


@pytest.mark.parametrize("change", [
    {"method": "unknown"}, {"stage_parameters": []}, {"parameter_status": "assumed"},
    {"parameter_source": ""}, {"min_segment_frames": 10}, {"max_trailing_seconds": .2},
])
def test_composition_config_rejects_undefined_or_legacy_settings(native, change):
    config = composition_config(native[0])
    config["progress"].update(change)
    with pytest.raises(ValueError):
        validate_build_config(config)


@pytest.mark.parametrize("missing", ["stage_parameters", "parameter_status", "parameter_source"])
def test_composition_requires_explicit_parameters_and_provenance(native, missing):
    config = composition_config(native[0])
    del config["progress"][missing]
    with pytest.raises(ValueError):
        validate_build_config(config)


def test_composition_missing_outcome_never_publishes_cache(native):
    config, table, path, _, _ = native
    table["annotation.sub_goal_score"][11] = 3
    pq.write_table(pa.table(table), path)
    with pytest.raises(ValueError, match="outcome"):
        build_cache(composition_config(config))
    assert not Path(config["output"]).exists()
