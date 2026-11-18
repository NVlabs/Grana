# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LeRobot v2/v3 annotation/video reader producing the Grana cache contract.

The selected target policy runs on native frames before uniform frame selection.
This intentionally omits the legacy failure-aware oversampler.
"""
from __future__ import annotations

from copy import deepcopy
from fractions import Fraction
import hashlib
import json
import math
from numbers import Integral, Real
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import uuid

import numpy as np
import yaml

from grana._paths import PROJECT_ROOT

from grana.data.targets.legacy import PROGRESS_METHOD as LEGACY_METHOD, compute_progress
from grana.data.targets.composition import PROGRESS_METHOD as COMPOSITION_METHOD, compute_target_composition, validate_stage_parameters
from grana.modeling.prompts import text_field

PROGRESS_DEFAULTS = {"progress_cap_seconds": 40.0, "min_segment_frames": 10, "max_trailing_seconds": None}
ANNOTATIONS = ("annotation.sub_goal", "annotation.sub_goal_score", "annotation.frame_type")
QUALITY = {"normal": 0, "suboptimal": 1, "failure": 2}


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


def _path(value, name, base):
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError(f"{name} must be an explicit path")
    expanded = os.path.expandvars(os.path.expanduser(str(value)))
    if "$" in expanded:
        raise ValueError(f"Unresolved environment variable in {name}: {value}")
    path = Path(expanded)
    if name == "output" and path.is_symlink():
        raise ValueError("Output must be a new directory, not an existing symbolic link")
    return (path if path.is_absolute() else base / path).resolve()


def validate_build_config(config: dict, base_dir: str | Path | None = None) -> dict:
    """Return a normalized copy without reading data or creating directories."""
    required = {"source", "output", "episodes", "task", "compressed_task_prompt", "stages", "views", "fps"}
    if not isinstance(config, dict) or required - config.keys() or config.keys() - required - {"image_size", "progress", "annotations", "labels"}:
        raise ValueError(f"Build config requires {sorted(required)}; optional keys: image_size, progress, annotations, labels")
    cfg = deepcopy(config)
    base = Path(base_dir or Path.cwd()).resolve()
    source, output = (_path(cfg[name], name, base) for name in ("source", "output"))
    project = PROJECT_ROOT
    if output.is_relative_to(project) or output.is_relative_to(source):
        raise ValueError("Cache output must be outside the project checkout and raw source dataset")
    if any((parent / ".git").exists() for parent in (output, *output.parents)):
        raise ValueError("Cache output must be outside Git checkouts")
    cfg.update(source=str(source), output=str(output))
    for key in ("task", "compressed_task_prompt"):
        cfg[key] = text_field(cfg[key], key)
    episodes = cfg["episodes"]
    if episodes == "all" and "labels" in cfg:
        pass
    elif not isinstance(episodes, list) or not episodes or any(isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in episodes) or len(set(episodes)) != len(episodes):
        raise ValueError("episodes must be a nonempty list of unique nonnegative integers")
    if "labels" in cfg:
        if "annotations" in cfg:
            raise ValueError("Use either labels (human annotation) or annotations (reviewed sidecars), not both")
        labels = _path(cfg["labels"], "labels", base)
        if labels.is_relative_to(source):
            raise ValueError("labels must be outside the raw source dataset")
        if labels.is_relative_to(project) or any((parent / ".git").exists() for parent in (labels, *labels.parents)):
            raise ValueError("labels must be outside Git checkouts")
        if labels.is_relative_to(output):
            raise ValueError("labels must be outside the new cache output directory")
        cfg["labels"] = str(labels)
    if "annotations" in cfg:
        annotations = cfg["annotations"]
        if not isinstance(annotations, dict) or not annotations:
            raise ValueError("annotations must map every selected episode to revision and supplement paths")
        normalized = {}
        for key, item in annotations.items():
            if (isinstance(key, bool) or not isinstance(key, (str, int))
                    or not str(key).isdigit() or str(int(key)) != str(key)):
                raise ValueError("annotations keys must be canonical nonnegative episode indices")
            identity = str(key)
            if identity in normalized or not isinstance(item, dict) or set(item) != {"revision", "supplement"}:
                raise ValueError("Each annotation entry requires exactly revision and supplement")
            normalized[identity] = {name: str(_path(value, f"annotations.{identity}.{name}", base)) for name, value in item.items()}
        if set(normalized) != {str(ep) for ep in episodes}:
            raise ValueError("annotations must cover exactly the selected episodes")
        cfg["annotations"] = normalized
    stages = cfg["stages"]
    if not isinstance(stages, list) or not 1 <= len(stages) <= 26:
        raise ValueError("stages must contain 1..26 ordered {label, prompt} entries")
    for stage in stages:
        if not isinstance(stage, dict) or set(stage) != {"label", "prompt"}:
            raise ValueError("Each stage must contain exactly label and prompt")
        for key in ("label", "prompt"):
            stage[key] = text_field(stage[key], f"stage {key}")
        if stage["label"] in ("leading", "trailing"):
            raise ValueError("leading/trailing are reserved unscored labels")
    if len({stage["label"] for stage in stages}) != len(stages) or len({stage["prompt"] for stage in stages}) != len(stages):
        raise ValueError("Stage labels and canonical prompts must each be unique")
    prompts = [stage["prompt"].casefold() for stage in stages]
    if any(left.endswith(right) for i, left in enumerate(prompts) for j, right in enumerate(prompts) if i != j):
        raise ValueError("Canonical stage prompts cannot have case-insensitive suffix ambiguity")
    views = cfg["views"]
    if not isinstance(views, dict) or not views or any(not isinstance(view, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", view) for view in views):
        raise ValueError("views must map unique simple camera names to explicit video feature keys")
    for view, key in views.items():
        views[view] = text_field(key, f"views.{view}")
    if len(set(views.values())) != len(views):
        raise ValueError("Each named view must use a distinct video feature")
    _positive(cfg["fps"], "fps")
    size = cfg.get("image_size")
    if size is not None and (not isinstance(size, list) or len(size) != 2 or any(isinstance(x, bool) or not isinstance(x, int) or x <= 0 for x in size)):
        raise ValueError("image_size must be [width, height] positive integers, or omitted")
    options = cfg.get("progress", {})
    if not isinstance(options, dict):
        raise ValueError("progress must be a mapping")
    method = options.get("method", LEGACY_METHOD)
    if method == COMPOSITION_METHOD:
        required_progress = {"method", "stage_parameters", "parameter_status", "parameter_source"}
        if set(options) != required_progress:
            raise ValueError(f"{COMPOSITION_METHOD} requires exactly {sorted(required_progress)}")
        if options["parameter_status"] not in ("example", "configured", "calibrated"):
            raise ValueError("progress.parameter_status must be example, configured or calibrated")
        options["parameter_source"] = text_field(options["parameter_source"], "progress.parameter_source")
        options["stage_parameters"] = validate_stage_parameters(options["stage_parameters"], len(stages))
        cfg["progress"] = options
    elif method == LEGACY_METHOD:
        if options.keys() - PROGRESS_DEFAULTS.keys() - {"method"}:
            raise ValueError(f"Legacy progress accepts only method and {list(PROGRESS_DEFAULTS)}")
        cfg["progress"] = {**PROGRESS_DEFAULTS, **options, "method": method}
        _positive(cfg["progress"]["progress_cap_seconds"], "progress.progress_cap_seconds")
        minimum = cfg["progress"]["min_segment_frames"]
        if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 1:
            raise ValueError("progress.min_segment_frames must be a positive integer")
        tail = cfg["progress"]["max_trailing_seconds"]
        if tail is not None and (isinstance(tail, bool) or not isinstance(tail, Real) or not math.isfinite(tail) or tail < 0):
            raise ValueError("progress.max_trailing_seconds must be nonnegative or null")
    else:
        raise ValueError(f"Unknown progress.method: {method}")
    return cfg


def load_build_config(path: str | Path) -> dict:
    path = Path(path).expanduser().resolve()
    return validate_build_config(yaml.safe_load(path.read_text()), path.parent)


def validate_label_clock(episode, source_fps):
    """Human UI uses native frame/fps positions; shared videos play as episode clips."""
    _positive(source_fps, "source fps")
    times = np.asarray(episode.timestamps, dtype=np.float64)
    if not np.allclose(times - times[0], np.arange(episode.frame_count) / source_fps, rtol=0, atol=1e-4):
        raise ValueError("Human-label source timestamps must agree with the native frame rate")
    if not getattr(episode, "video_segments", None):
        for camera in episode.clock_receipt["cameras"].values():
            first_pts = times[0] - camera["source_minus_video_offset_seconds"]
            if abs(first_pts) > 1e-4:
                raise ValueError("Human-label videos must start at presentation timestamp zero for browser seeking")


def episode_targets(cfg: dict, labels, scores, quality, times, source_fps) -> dict:
    """One shared target calculation for the unsaved human preview and cache construction."""
    label_map = {stage["label"]: stage["prompt"] for stage in cfg["stages"]}
    inputs = dict(labels=labels, scores=scores, action_quality=quality,
                  stage_prompts=[stage["prompt"] for stage in cfg["stages"]], label_map=label_map, source_fps=source_fps)
    if cfg["progress"]["method"] == COMPOSITION_METHOD:
        return compute_target_composition(**inputs, timestamps=times, stage_parameters=cfg["progress"]["stage_parameters"])
    return compute_progress(**inputs, **{k: v for k, v in cfg["progress"].items() if k != "method"})


def _jsonl(path, key):
    result = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        identity = row[key]
        if isinstance(identity, bool) or not isinstance(identity, int) or identity < 0 or identity in result:
            raise ValueError(f"{path.name} needs unique nonnegative integer {key}")
        result[identity] = row
    return result


def _scalar(value):
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return value[0]
    return value


def _integers(values, name):
    values = [_scalar(value) for value in values]
    if any(isinstance(value, bool) or not isinstance(value, Integral) for value in values):
        raise ValueError(f"{name} must contain integer scalars or singleton integer lists")
    return [int(value) for value in values]


def _source_file(source, template, **fields):
    path = (source / template.format(**fields)).resolve()
    if not path.is_relative_to(source) or not path.is_file():
        raise ValueError(f"Source metadata template must resolve to an existing file within the dataset: {path}")
    return path


def _probe_video(path: Path) -> dict:
    executable = shutil.which("ffprobe")
    if not executable:
        raise RuntimeError("ffprobe is required to verify source videos")
    result = subprocess.run([executable, "-v", "error", "-select_streams", "v:0", "-show_entries",
                             "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames", "-of", "json", str(path)],
                            capture_output=True, text=True, check=True)
    streams = json.loads(result.stdout)["streams"]
    if len(streams) != 1:
        raise ValueError(f"Expected a video stream in {path}")
    stream = streams[0]
    return {"width": int(stream["width"]), "height": int(stream["height"]),
            "fps": float(Fraction(stream["avg_frame_rate"])), "nominal_fps": float(Fraction(stream["r_frame_rate"])),
            "frames": int(stream["nb_frames"]) if stream.get("nb_frames", "N/A") != "N/A" else None}


def _decode_video(path: Path, destination: Path, indices: list[int], source_frames: int, source_size: tuple[int, int], image_size=None,
                  *, frame_times=None, source_fps=None):
    """Decode every native frame; save only explicit selected frames to RGB NPY."""
    executable = shutil.which("ffmpeg")
    if not executable:
        raise RuntimeError("ffmpeg is required to decode source videos")
    width, height = tuple(image_size or source_size)
    seek, filters = [], []
    if frame_times is not None:
        from grana.annotation.source import MATCH_FRAMES, read_around

        # Shared videos are selected by the verified episode's presentation times,
        # never by frame numbers counted from a seek point. Decode the whole
        # fixed-cadence episode so an extra/missing frame fails even if unsampled.
        if len(frame_times) != source_frames:
            raise ValueError("Video presentation times must cover every native episode frame")
        seek = read_around(frame_times, source_fps)
        margin = MATCH_FRAMES / source_fps
        filters.append(f"select=between(t\\,{frame_times[0] - margin:.6f}\\,{frame_times[-1] + margin:.6f})")
    command = [executable, "-v", "error", "-nostdin", "-threads", "1", "-noautorotate", *seek, "-i", str(path),
               "-map", "0:v:0", "-an", "-sn", "-dn", "-threads", "1", "-vsync", "0"]
    if (width, height) != source_size:
        filters.append(f"scale={width}:{height}:flags=area")
    if filters:
        command += ["-vf", ",".join(filters)]
    command += ["-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"]
    frames = np.lib.format.open_memmap(destination, mode="w+", dtype=np.uint8, shape=(len(indices), height, width, 3))
    with tempfile.TemporaryFile() as error:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=error)
        native_index = selected = 0
        try:
            while raw := process.stdout.read(width * height * 3):
                if len(raw) != width * height * 3 or native_index >= source_frames:
                    raise ValueError(f"Video length/frame shape disagrees with episode metadata: {path}")
                if selected < len(indices) and native_index == indices[selected]:
                    frames[selected] = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)
                    selected += 1
                native_index += 1
            return_code = process.wait()
            if return_code or native_index != source_frames or selected != len(indices):
                error.seek(0)
                raise ValueError(f"Video decode/alignment failed for {path}: decoded={native_index}, expected={source_frames}; {error.read().decode()[-2000:]}")
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            process.stdout.close()
            frames.flush()
            del frames
    return [len(indices), height, width, 3]


def _episode_annotations(path, episode_index, length, tasks, source_fps, labels, *, native_annotations=False, dataset=None):
    import pyarrow.parquet as parquet

    names = parquet.read_schema(path).names
    required = ["frame_index", "episode_index", "timestamp", *ANNOTATIONS]
    if set(required) - set(names):
        raise ValueError(f"Missing native frame/annotation columns: {sorted(set(required) - set(names))}")
    mask_name = "rl_info.action_quality_mask"
    columns = required + ([mask_name] if mask_name in names else [])
    table = (dataset.rows(episode_index, columns) if dataset is not None
             else parquet.read_table(path, columns=columns).to_pydict())
    if len(table["frame_index"]) != length or _integers(table["frame_index"], "frame_index") != list(range(length)):
        raise ValueError("Parquet rows must match episode length and contiguous native frame indices")
    if set(_integers(table["episode_index"], "episode_index")) != {episode_index}:
        raise ValueError("Parquet episode_index disagrees with selected episode")
    times = np.asarray([_scalar(value) for value in table["timestamp"]], dtype=np.float64)
    if not np.isfinite(times).all() or not np.allclose(times - times[0], np.arange(length) / source_fps, rtol=0, atol=1e-4):
        raise ValueError("Parquet timestamps must agree with the native frame rate")
    decoded = {}
    for name in ANNOTATIONS:
        ids = _integers(table[name], name)
        if set(ids) - tasks.keys():
            raise ValueError(f"{name} references absent tasks metadata entries")
        values = [tasks[index]["task"] for index in ids]
        if any(not isinstance(value, str) for value in values):
            raise ValueError("Annotation dictionary values must be text")
        decoded[name] = values
        if name == ANNOTATIONS[0] and len(set(ids)) != len(set(values)):
            raise ValueError("Distinct raw label IDs cannot decode to the same stage text")
    unscored = {"", "leading", "trailing"} | ({"empty"} if native_annotations else set())
    unknown = set(decoded[ANNOTATIONS[0]]) - labels - unscored
    if unknown:
        raise ValueError(f"Unrecognized raw stage labels: {sorted(unknown)}")
    scores = ["" if value == "empty" else value for value in decoded[ANNOTATIONS[1]]]
    if set(scores) - {"", "high", "mid", "low"}:
        raise ValueError("Stage scores must be empty/high/mid/low")
    quality_labels = QUALITY | ({"nominal": 0, "regressed": 2, "": 0, "empty": 0} if native_annotations else {})
    if set(decoded[ANNOTATIONS[2]]) - quality_labels.keys():
        raise ValueError("Frame quality must be normal/suboptimal/failure")
    frame_quality = [quality_labels[value] for value in decoded[ANNOTATIONS[2]]]
    mask_quality = _integers(table[mask_name], mask_name) if mask_name in table else frame_quality
    if set(mask_quality) - {0, 1, 2}:
        raise ValueError("rl_info.action_quality_mask must contain only 0/1/2")
    use_mask = mask_name in table and not native_annotations
    quality = mask_quality if use_mask else frame_quality
    provenance = {"quality_source": mask_name if use_mask else ANNOTATIONS[2],
                  "quality_disagreement_frames": sum(a != b for a, b in zip(mask_quality, frame_quality, strict=True))}
    return decoded[ANNOTATIONS[0]], scores, quality, times.tolist(), provenance


def build_cache(config: dict) -> dict:
    """Create one new external cache atomically, retaining failed staging folders."""
    from datasets import Dataset

    cfg = validate_build_config(config)
    source, output = Path(cfg["source"]), Path(cfg["output"])
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite cache output: {output}")
    info = json.loads((source / "meta/info.json").read_text())
    version = str(info.get("codebase_version", ""))
    shared = None
    if version.startswith("v3."):
        from .lerobot import LeRobotDataset

        shared = LeRobotDataset(source)
    elif not version.startswith("v2."):
        raise ValueError("Cache construction supports native LeRobot v2.x and v3.0")
    source_fps = info["fps"]
    _positive(source_fps, "source fps")
    if cfg["fps"] > source_fps:
        raise ValueError("Output fps cannot exceed source fps; cache building never interpolates frames")
    chunks = info["chunks_size"]
    if isinstance(chunks, bool) or not isinstance(chunks, int) or chunks < 1:
        raise ValueError("LeRobot chunks_size must be a positive integer")
    episodes = shared.episodes if shared is not None else _jsonl(source / "meta/episodes.jsonl", "episode_index")
    tasks = {}
    if "annotations" not in cfg and "labels" not in cfg:
        tasks = ({index: {"task": task} for index, task in shared.tasks.items()} if shared is not None
                 else _jsonl(source / "meta/tasks.jsonl", "task_index"))
    label_source = None
    if "labels" in cfg or shared is not None:
        from grana.annotation.source import LeRobotSource

        label_source = LeRobotSource(source, cfg["views"], allow_shared_videos=True)
    if "labels" in cfg:
        from grana.annotation.labels import finished_episodes

        if cfg["episodes"] == "all":
            cfg["episodes"] = sorted(set(finished_episodes(cfg["labels"])) & episodes.keys())
            if not cfg["episodes"]:
                raise ValueError("No submitted episode labels; submit episodes in `grana label` first")
    if set(cfg["episodes"]) - episodes.keys():
        raise ValueError("Requested episode index is absent from episode metadata")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(f".{output.name}.partial-{uuid.uuid4().hex}")
    (staging / "frames").mkdir(parents=True)
    report = {"status": "building", "source": str(source), "output": str(output), "source_version": info["codebase_version"],
              "sampling_method": "linspace_v1", "failure_aware_oversampling": False, "config": cfg, "episodes": []}
    status_file = staging / "build_status.json"
    status_file.write_text(json.dumps(report, indent=2) + "\n")
    rows = []
    label_map = {stage["label"]: stage["prompt"] for stage in cfg["stages"]}
    try:
        for index in cfg["episodes"]:
            metadata = episodes[index]
            length = metadata["length"]
            if isinstance(length, bool) or not isinstance(length, int) or length < 1:
                raise ValueError("Episode length must be a positive integer")
            fields = {"episode_index": index, "episode_chunk": index // chunks}
            parquet_path = shared.data_path(index) if shared is not None else _source_file(source, info["data_path"], **fields)
            episode = label_source.load_episode(index) if label_source is not None else None
            composition = cfg["progress"]["method"] == COMPOSITION_METHOD
            if "annotations" in cfg:
                from grana.annotation.bridge import read_bridge_annotations

                labels, scores, quality, times, annotation_provenance = read_bridge_annotations(
                    source, cfg["views"], index, cfg["stages"], cfg["annotations"][str(index)])
                if len(times) != length or not np.allclose(np.asarray(times) - times[0], np.arange(length) / source_fps, rtol=0, atol=1e-4):
                    raise ValueError("Sidecar source timestamps must agree with the native frame rate")
            elif "labels" in cfg:
                from grana.annotation.labels import is_legacy_file, read_episode_labels, read_legacy_episode_labels

                validate_label_clock(episode, source_fps)
                if is_legacy_file(cfg["labels"], index):  # 11-key file from the first grana label, read as it is
                    labels, scores, quality, annotation_provenance = read_legacy_episode_labels(
                        cfg["labels"], index, trajectory_id=episode.source_identity["trajectory_id"],
                        times=episode.timestamps, stages=cfg["stages"])
                else:
                    labels, scores, quality, annotation_provenance = read_episode_labels(cfg["labels"], episode, cfg["stages"])
                times = list(episode.timestamps)
            else:
                labels, scores, quality, times, annotation_provenance = _episode_annotations(
                    parquet_path, index, length, tasks, source_fps, set(label_map), native_annotations=composition, dataset=shared)
            targets = episode_targets(cfg, labels, scores, quality, times, source_fps)
            retained = targets["source_indices"]
            if not retained or retained != sorted(set(retained)) or min(retained) < 0 or max(retained) >= length:
                raise ValueError("Progress kernel must return increasing unique native source_indices")
            target_count = min(len(retained), max(1, round(len(retained) * cfg["fps"] / source_fps)))
            selection = np.linspace(0, len(retained) - 1, target_count, dtype=int).tolist()
            selected = [retained[i] for i in selection]
            source_identity = metadata.get("trajectory_id")
            identity = text_field(source_identity, "source trajectory_id") if source_identity else f"{source.name}__episode_{index:06d}"
            identity_method = "source_trajectory_id" if source_identity else "dataset_episode_fallback"
            row = {"id": identity, "identity_method": identity_method, "task": cfg["task"],
                   "compressed_task_prompt": cfg["compressed_task_prompt"], "stage_prompts": list(label_map.values()),
                   "source_episode_index": index, "source_trajectory_id": str(metadata.get("trajectory_id", "")),
                   "source_frame_indices": selected, "source_timestamps": [times[i] for i in selected],
                   "source_fps": float(source_fps), "data_gen_output_fps": float(cfg["fps"]),
                   "progress_method": targets["method"], "sampling_method": "linspace_v1", "view_names": list(cfg["views"]),
                   "action_quality": [quality[i] for i in selected], "num_frames": target_count}
            row["progress_config_json"] = json.dumps(cfg["progress"], sort_keys=True, allow_nan=False)
            for key in ("stage_ids", "within_stage_progress", "progress"):
                if len(targets[key]) != len(retained):
                    raise ValueError(f"Progress kernel returned misaligned {key}")
                row[key] = [targets[key][i] for i in selection]
            probes, source_shape = {}, None
            for view, video_key in cfg["views"].items():
                feature = info["features"].get(video_key, {})
                if feature.get("dtype") != "video":
                    raise ValueError(f"View {view} does not name a native video feature: {video_key}")
                decode_options = {}
                if shared is not None:
                    from grana.annotation.source import episode_frame_times

                    segment = episode.video_segments[view]
                    video = segment.path
                    frame_times = episode_frame_times(segment, source_fps, episode.source_identity["video_packets_sha256"][view])
                    height, width, _ = episode.clock_receipt["cameras"][view]["shape"]
                    probe = {"width": width, "height": height, "frames": len(frame_times),
                             "fps": source_fps, "nominal_fps": source_fps,
                             "from_timestamp": segment.start, "to_timestamp": segment.end,
                             "video_packets_sha256": episode.source_identity["video_packets_sha256"][view]}
                    decode_options = {"frame_times": frame_times, "source_fps": source_fps}
                else:
                    video = _source_file(source, info["video_path"], video_key=video_key, **fields)
                    probe = _probe_video(video)
                shape = (probe["width"], probe["height"])
                if feature.get("shape") != [shape[1], shape[0], 3] or (source_shape is not None and shape != source_shape):
                    raise ValueError("Video feature shape or multiview geometry mismatch")
                if any(not math.isclose(probe[key], source_fps, rel_tol=0, abs_tol=1e-4) for key in ("fps", "nominal_fps")) or probe["frames"] not in (None, length):
                    raise ValueError("Video frame count/fps differs from native episode annotations")
                source_shape = shape
                filename = f"episode_{index:06d}_{view}.npy"
                row["frames_shape"] = _decode_video(video, staging / "frames" / filename, selected, length, shape, cfg.get("image_size"),
                                                     **decode_options)
                row[f"frames_{view}"] = str(output / "frames" / filename)
                probes[view] = {"path": str(video), **probe}
            row["frames"] = row[f"frames_{next(iter(cfg['views']))}"]
            rows.append(row)
            report["episodes"].append({"episode_index": index, "source_frames": length, "retained_frames": len(retained),
                                       "cache_frames": target_count, "source_frame_indices": selected, "progress_method": targets["method"],
                                       "trajectory_id": identity, "identity_method": identity_method,
                                       "parquet": str(parquet_path), "videos": probes, **annotation_provenance})
            if composition:
                report["episodes"][-1]["chunks"] = targets["chunks"]
        Dataset.from_list(rows).save_to_disk(str(staging / "processed_dataset"))
        # The production reader validates unit-range targets and complete row
        # semantics; its same-cache fallback resolves the intended final paths.
        from .cache import CachedTrajectories
        checked = CachedTrajectories([str(staging)], views=list(cfg["views"]))
        for trajectory in checked.records:
            trajectory.images((0,))
        report.update(status="complete", rows=len(rows), total_cache_frames=sum(row["num_frames"] for row in rows),
                      reader_validation="passed",
                      metadata_sha256={str(path.relative_to(source / "meta")): hashlib.sha256(path.read_bytes()).hexdigest()
                                       for path in (shared.metadata_files() if shared is not None else
                                                    [source / "meta" / name for name in ("info.json", "episodes.jsonl", "tasks.jsonl")])
                                       if path.is_file()})
        (staging / "provenance.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        status_file.write_text(json.dumps({"status": "complete", "rows": len(rows)}, indent=2) + "\n")
        if output.exists():
            raise FileExistsError(f"Output appeared while building: {output}")
        staging.rename(output)
    except BaseException as error:
        status_file.write_text(json.dumps({"status": "failed", "error_type": type(error).__name__, "error": str(error), "output": str(output)}, indent=2) + "\n")
        raise
    return report
