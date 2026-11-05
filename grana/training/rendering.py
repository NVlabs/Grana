# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU held-out replays of predictions made at a particular training step.

These are sampled evaluation windows, not continuous original-speed movies.
No model is loaded and no additional predictions are manufactured for rendering.
"""
from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import time

from PIL import Image, ImageDraw, ImageFont, ImageOps

from grana.data.sampling import POI_SLOT, stage_relative_targets
from .evaluation import record_poi_slot


WIDTH, HEIGHT = 1280, 960
INK, MUTED, BORDER = "#17243b", "#52627a", "#dce4ef"
PREDICTION, GROUND_TRUTH = "#0767d3", "#17243b"
ROLE_TITLES = {"A": "A | whole-task progress", "B": "B | ground-truth focus progress",
               "C": "C | first unfinished stage", "D": "D | predicted-focus progress"}


@lru_cache(maxsize=16)
def _font(size, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(name, size)
    except OSError:
        return ImageFont.load_default(size=size)


def _text(draw, xy, text, size=18, fill=INK, bold=False, max_width=None):
    text = str(text)
    font = _font(size, bold)
    if max_width is not None:
        while text and draw.textlength(text, font=font) > max_width:
            text = text[:-2] + "…" if len(text) > 1 else ""
    draw.text(xy, text, fill=fill, font=font)


def _elapsed(seconds):
    total = int(seconds)
    return f"{total // 3600:02d}:{total % 3600 // 60:02d}:{total % 60:02d}"


def _key(trajectory):
    return str(trajectory.cache_root), trajectory.task, trajectory.trajectory_id


def _validated_records(result, plan):
    """Reject foreign, incomplete, or relabelled predictions before reading RGB."""
    manifest = plan.manifest()
    if result.get("plan_sha256") != manifest["plan_sha256"] or result.get("score_slot") != manifest.get("score_slot", POI_SLOT):
        raise ValueError("Render predictions must belong to this exact held-out evaluation plan")
    expected = {(*_key(window.trajectory), window.poi, role): window
                for window in plan.windows for role in window.roles}
    records = result.get("records", [])
    if len(records) != len(expected) or result.get("model_calls") != len(expected):
        raise ValueError("Render records must cover every model call in the held-out plan")
    found, grouped = {}, defaultdict(dict)
    for record in records:
        identity = (record["cache"], record["task"], record["trajectory_id"], record["poi"], record["role"])
        if identity not in expected or identity in found:
            raise ValueError("Render records contain an unknown or repeated held-out window/role")
        window = expected[identity]
        trajectory = window.trajectory
        geometry = window.manifest()
        indices = geometry["frame_indices"]
        slot = record_poi_slot(record)
        weight_order = record.get("frame_weight_order", list(range(8)))
        if (not isinstance(weight_order, (list, tuple)) or len(weight_order) != 8
                or any(type(index) is not int for index in weight_order) or sorted(weight_order) != list(range(8))):
            raise ValueError("Invalid record frame_weight_order")
        if (slot != record_poi_slot(geometry) or indices[slot] != record["poi"]
                or tuple(weight_order) != tuple(geometry.get("frame_weight_order", range(8)))):
            raise ValueError("Render query slot or frame weights differ from the held-out plan")
        if (record["frame_indices"] != indices or record["view_names"] != list(trajectory.view_names)
                or record["num_stages"] != len(trajectory.stage_prompts)):
            raise ValueError("Render record RGB geometry differs from the held-out plan")
        predictions, targets = record["predictions"], record["targets"]
        if len(predictions) != 8 or len(targets) != 8 or any(not math.isfinite(x) for x in predictions):
            raise ValueError("Render records need eight finite predictions and targets")
        role = record["role"]
        if record.get("stage_augmentation_applied", False) is not False:
            raise ValueError("Rendered evaluation records must never contain stage augmentation")
        if role == "C":
            answer_count = record.get("stage_answer_count", len(trajectory.stage_prompts))
            if type(answer_count) is not int or answer_count != len(trajectory.stage_prompts):
                raise ValueError("Rendered C answer count must equal the true stage count")
            if any(isinstance(x, bool) or int(x) != x or not 0 <= x < len(trajectory.stage_prompts) for x in predictions):
                raise ValueError("Rendered C predictions must be valid task-local stages")
            expected_targets = [trajectory.stage_ids[i] for i in indices]
        elif role == "A":
            expected_targets = [trajectory.progress[i] for i in indices]
        else:
            expected_targets = stage_relative_targets(tuple(trajectory.stage_ids[i] for i in indices),
                                                      tuple(trajectory.within_stage_progress[i] for i in indices),
                                                      record["focus_stage"], len(trajectory.stage_prompts))
            if role == "B" and record["focus_stage"] != trajectory.stage_ids[record["poi"]]:
                raise ValueError("Rendered B must use the ground-truth POI focus")
        if role != "C" and any(not 0 <= x <= 1 + 1e-6 for x in predictions):
            raise ValueError("Rendered progress predictions must lie in [0, 1]")
        if any(not math.isfinite(x) or not math.isclose(x, y, abs_tol=1e-6)
               for x, y in zip(targets, expected_targets, strict=True)):
            raise ValueError("Render record targets differ from the actual held-out labels")
        found[identity] = record
        grouped[identity[:3]].setdefault(record["poi"], {})[role] = record
    for identity, record in found.items():
        if record["role"] == "D":
            c_record = found.get((*identity[:-1], "C"))
            if (c_record is None or record_poi_slot(record) != record_poi_slot(c_record)
                    or record["focus_stage"] != int(c_record["predictions"][record_poi_slot(c_record)])):
                raise ValueError("Rendered D must use this window's C predicted focus")
    return grouped


def _role_metrics(samples):
    metrics = {}
    for role in "ABCD":
        rows = [sample[role] for sample in samples.values() if role in sample]
        if not rows:
            continue
        if role == "C":
            metrics["role_C_poi_accuracy"] = sum(row["predictions"][record_poi_slot(row)] == row["targets"][record_poi_slot(row)]
                                                   for row in rows) / len(rows)
        else:
            metrics[f"role_{role}_poi_rmse"] = math.sqrt(sum((row["predictions"][record_poi_slot(row)] - row["targets"][record_poi_slot(row)]) ** 2
                                                              for row in rows) / len(rows))
    return metrics


def _plot(canvas, box, samples, trajectory, role, current):
    draw = ImageDraw.Draw(canvas)
    left, top, right, bottom = box
    draw.rounded_rectangle(box, radius=10, fill="white", outline=BORDER)
    rows = [(poi, records[role]) for poi, records in sorted(samples.items()) if role in records]
    slots = ",".join(str(slot) for slot in sorted({record_poi_slot(row) for _, row in rows}))
    title = f"{role} | POI slot {slots} | {ROLE_TITLES[role].split(' | ', 1)[1]}" if rows else ROLE_TITLES[role]
    _text(draw, (left + 14, top + 8), title, 18, bold=True)
    if not rows:
        _text(draw, (left + 20, top + 72), "Unassessed for this trajectory", 18, MUTED)
        return
    x0, x1, y0, y1 = left + 43, right - 18, top + 47, bottom - 34
    maximum = max(1, len(trajectory.stage_prompts) - 1) if role == "C" else 1

    def xy(poi, value):
        return x0 + poi / max(1, trajectory.num_frames - 1) * (x1 - x0), y1 - value / maximum * (y1 - y0)

    ticks = (0, maximum) if role == "C" else (0, .5, 1)
    for tick in ticks:
        _, y = xy(0, tick)
        draw.line((x0, y, x1, y), fill=BORDER)
        _text(draw, (left + 8, y - 9), f"{tick:g}", 14, MUTED)
    for poi in (0, trajectory.num_frames - 1):
        x, _ = xy(poi, 0)
        _text(draw, (x - 8, y1 + 3), poi, 13, MUTED)
    x, _ = xy(current, 0)
    draw.line((x, y0, x, y1), fill="#b7c6d9", width=2)
    # Show measured samples only. No interpolated predictions between windows.
    for poi, row in rows:
        x, prediction = xy(poi, row["predictions"][record_poi_slot(row)])
        _, target = xy(poi, row["targets"][record_poi_slot(row)])
        draw.ellipse((x - 4, prediction - 4, x + 4, prediction + 4), outline=PREDICTION, width=2)
        draw.line((x - 4, target - 4, x + 4, target + 4), fill=GROUND_TRUTH, width=2)
        draw.line((x - 4, target + 4, x + 4, target - 4), fill=GROUND_TRUTH, width=2)
    _text(draw, (left + 14, bottom - 19), f"Source frame index | cursor {current} | circles: prediction | crosses: GT", 12, MUTED)


def _frame(trajectory, samples, poi, *, step, training_elapsed_seconds, sample_index, sample_count, replay_fps, metrics):
    canvas = Image.new("RGB", (WIDTH, HEIGHT), "#f3f6fb")
    draw = ImageDraw.Draw(canvas)
    _text(draw, (20, 12), trajectory.task, 25, bold=True, max_width=WIDTH - 40)
    _text(draw, (20, 47), f"STEP {step} | TRAINING ELAPSED {_elapsed(training_elapsed_seconds)} ({training_elapsed_seconds:.1f}s) | held-out {trajectory.trajectory_id}",
          18, max_width=WIDTH - 40)
    current = samples[poi]
    readouts = ", ".join(f"{role}[{record_poi_slot(row)}]" for role, row in sorted(current.items()))
    _text(draw, (20, 77), f"OFFLINE POI REPLAY | {readouts} | source frame {poi}/{trajectory.num_frames - 1} | sample {sample_index + 1}/{sample_count}",
          17, MUTED, max_width=WIDTH - 40)
    _text(draw, (20, 104), f"Brackets identify input slots; POI windows may include later frames. Playback {replay_fps:g} samples/s, not original speed.", 16, MUTED)
    # Preserve all effective ordered cameras, including cache-specific view sets.
    columns = 1 if len(trajectory.view_names) == 1 else 2
    rows = math.ceil(len(trajectory.view_names) / columns)
    cell_width, cell_height = 610 // columns, 404 // rows
    images = trajectory.images((poi,))
    for index, (name, image) in enumerate(zip(trajectory.view_names, images, strict=True)):
        left, top = 20 + index % columns * cell_width, 140 + index // columns * cell_height
        box = (left, top, left + cell_width - 8, top + cell_height - 8)
        draw.rounded_rectangle(box, radius=8, fill="#101827")
        image_box = (left + 4, top + 28, box[2] - 4, box[3] - 4)
        resized = ImageOps.contain(image, (image_box[2] - image_box[0], image_box[3] - image_box[1]), Image.Resampling.LANCZOS)
        canvas.paste(resized, (image_box[0] + (image_box[2] - image_box[0] - resized.width) // 2,
                               image_box[1] + (image_box[3] - image_box[1] - resized.height) // 2))
        _text(draw, (left + 9, top + 4), name, 15, "white", max_width=cell_width - 20)
    draw = ImageDraw.Draw(canvas)
    for index, role in enumerate("ABCD"):
        left, top, right, bottom = 647, 140 + index * 103, 1260, 235 + index * 103
        draw.rounded_rectangle((left, top, right, bottom), radius=10, fill="white", outline=BORDER)
        title = (f"{role} | POI slot {record_poi_slot(current[role])} | {ROLE_TITLES[role].split(' | ', 1)[1]}"
                 if role in current else ROLE_TITLES[role])
        _text(draw, (left + 14, top + 8), title, 18, bold=True)
        if role not in current:
            _text(draw, (left + 14, top + 43), "Unassessed for this window", 17, MUTED)
            continue
        row = current[role]
        predicted, target = row["predictions"][record_poi_slot(row)], row["targets"][record_poi_slot(row)]
        if role == "C":
            metric = f"stage accuracy diagnostic {metrics['role_C_poi_accuracy']:.3f}"
        else:
            label = {"A": "whole-task", "B": "oracle-focus", "D": "predicted-focus"}[role]
            metric = f"{label} RMSE diagnostic {metrics[f'role_{role}_poi_rmse']:.4f}"
        values = f"prediction {int(predicted)} | GT {int(target)}" if role == "C" else f"prediction {predicted:.4f} | GT {target:.4f}"
        _text(draw, (left + 14, top + 35), f"{metric} | {values}", 14, PREDICTION, max_width=right - left - 28)
        if role in ("B", "D"):
            _text(draw, (left + 14, top + 65), f"Focus stage {row['focus_stage']} | GT is relative to this focus", 15, MUTED)
        elif role == "C":
            _text(draw, (left + 14, top + 65), trajectory.stage_prompts[int(predicted)], 15, MUTED, max_width=right - left - 28)
    for role, box in (("A", (20, 568, 630, 746)), ("C", (647, 568, 1260, 746)),
                      ("B", (20, 764, 630, 942)), ("D", (647, 764, 1260, 942))):
        _plot(canvas, box, samples, trajectory, role, poi)
    return canvas


def render_evaluation(result, plan, *, output_dir, step: int, training_elapsed_seconds: float,
                      max_trajectories: int = 4, max_frames: int = 24, replay_fps: float = 2) -> dict:
    """Persist PNGs, GIF replays, and auditable numeric samples for this eval.

    Select trajectories in deterministic task round-robin order; cap media only,
    never the scalar evaluation. Each curve uses *all* evaluated POIs for that
    trajectory. If necessary the replay samples those POIs evenly up to
    ``max_frames``. GIF files are encoded with Pillow, so W&B needs no moviepy or
    external encoder. The caller owns rank-zero execution and W&B logging.
    """
    for name, value in (("step", step), ("max_trajectories", max_trajectories), ("max_frames", max_frames)):
        if isinstance(value, bool) or not isinstance(value, int) or value < (0 if name == "step" else 1):
            raise ValueError(f"render {name} must be a {'nonnegative' if name == 'step' else 'positive'} integer")
    if (isinstance(training_elapsed_seconds, bool) or not math.isfinite(training_elapsed_seconds) or training_elapsed_seconds < 0
            or isinstance(replay_fps, bool) or not math.isfinite(replay_fps) or not .1 <= replay_fps <= 20):
        raise ValueError("render elapsed time must be finite/nonnegative and replay_fps must be in [0.1, 20]")
    started = time.perf_counter()
    grouped = _validated_records(result, plan)
    trajectories = {_key(window.trajectory): window.trajectory for window in plan.windows}
    by_task = defaultdict(list)
    for identity in sorted(grouped):
        by_task[identity[1]].append(identity)
    ordered = [identities[index] for index in range(max(map(len, by_task.values())))
               for _, identities in sorted(by_task.items()) if index < len(identities)]
    directory = Path(output_dir).expanduser().resolve() / f"step-{step:06d}"
    if directory.exists():
        raise ValueError(f"Use a fresh render output directory: {directory}")
    directory.mkdir(parents=True)
    rendered = []
    for identity in ordered[:max_trajectories]:
        trajectory, samples = trajectories[identity], grouped[identity]
        pois = sorted(samples)
        if len(pois) <= max_frames:
            replay_pois = pois
        elif max_frames == 1:
            replay_pois = [pois[len(pois) // 2]]
        else:
            replay_pois = [pois[round(index * (len(pois) - 1) / (max_frames - 1))] for index in range(max_frames)]
        # A stable per-trajectory media key makes W&B's step slider comparable.
        digest = sha256(json.dumps(identity).encode()).hexdigest()[:16]
        task_slug = re.sub(r"[^a-zA-Z0-9]+", "-", trajectory.task).strip("-").lower()[:64] or "task"
        media_key = f"{task_slug}/{digest}"
        image_path, video_path = directory / f"{digest}.png", directory / f"{digest}.gif"
        metrics = _role_metrics(samples)
        frames = [_frame(trajectory, samples, poi, step=step, training_elapsed_seconds=training_elapsed_seconds,
                         sample_index=index, sample_count=len(replay_pois), replay_fps=replay_fps, metrics=metrics)
                  for index, poi in enumerate(replay_pois)]
        frames[len(frames) // 2].save(image_path)
        frames[0].save(video_path, save_all=True, append_images=frames[1:], duration=round(1000 / replay_fps), loop=0)
        rendered.append({"key": media_key, "task": trajectory.task, "trajectory_id": trajectory.trajectory_id,
                         "cache": str(trajectory.cache_root), "view_names": list(trajectory.view_names),
                         "image": str(image_path), "video": str(video_path), "video_format": "gif",
                         "image_sha256": sha256(image_path.read_bytes()).hexdigest(),
                         "video_sha256": sha256(video_path.read_bytes()).hexdigest(),
                         "sampled_pois": pois, "replay_pois": replay_pois, "metrics": metrics,
                         "points": [{"poi": poi, "role": role, "poi_slot": record_poi_slot(row),
                                     "source_frame": row["frame_indices"][record_poi_slot(row)],
                                     "prediction": row["predictions"][record_poi_slot(row)],
                                     "target": row["targets"][record_poi_slot(row)], "focus_stage": row["focus_stage"]}
                                    for poi, records in sorted(samples.items()) for role, row in sorted(records.items())]})
    bundle = {"schema": "grana-training-eval-renders-v1", "optimizer_steps": step,
              "training_elapsed_seconds": float(training_elapsed_seconds), "plan_sha256": result["plan_sha256"],
              "replay_type": "sampled_heldout_windows_not_original_speed", "replay_fps": replay_fps,
              "prediction_readout": "saved evaluation POI; input slots and source frames are recorded per point",
              "trajectory_selection": "task round-robin then cache/task/trajectory identity order; media cap only",
              "available_trajectories": len(grouped), "rendered_trajectories": len(rendered),
              "evaluation_metrics": result["metrics"], "rendering_seconds": time.perf_counter() - started,
              "items": rendered, "manifest": str(directory / "manifest.json")}
    Path(bundle["manifest"]).write_text(json.dumps(bundle, indent=2) + "\n")
    return bundle
