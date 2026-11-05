# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dense deployment-route diagnostics, with measured predictions and explicit units.

Staged tasks use C -> D, with optional B as an oracle diagnostic. A is used
only for trajectories whose evaluation contract has no staged route. Every
plotted prediction comes from an evaluated POI; video sampling never adds calls.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from functools import lru_cache
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time

from PIL import Image, ImageDraw, ImageOps

from .evaluation import EvaluationPlan, EvaluationWindow, record_poi_slot
from .render_plan import build_render_plan
from .rendering import _text, _validated_records


SCHEMA = "grana-deploy-dense-v1"
SIZE = (1280, 720)
COLORS = ("#377cd2", "#14a078", "#e79624", "#c94b73", "#825bc4", "#23aebc", "#a27240", "#718440")


@lru_cache(maxsize=1)
def require_encoder() -> str:
    override = os.environ.get("GRANA_FFMPEG")
    candidates = [override] if override else [str(Path(directory) / "ffmpeg") for directory in os.get_exec_path()]
    for candidate in dict.fromkeys(candidates):
        if not candidate or not os.access(candidate, os.X_OK):
            continue
        result = subprocess.run([candidate, "-hide_banner", "-encoders"], capture_output=True, text=True, timeout=10)
        if result.returncode == 0 and "libx264" in result.stdout:
            return candidate
    raise RuntimeError("Dense MP4 rendering requires ffmpeg with libx264; set GRANA_FFMPEG to a compatible binary")


def build_dense_plan(base_plan: EvaluationPlan, *, max_trajectories_per_task=4,
                     frame_stride=10, include_oracle=True, group_by_cache=None) -> EvaluationPlan:
    if isinstance(frame_stride, bool) or not isinstance(frame_stride, int) or frame_stride < 1:
        raise ValueError("deploy_dense.frame_stride must be a positive integer")
    if not isinstance(include_oracle, bool):
        raise ValueError("deploy_dense.include_oracle must be boolean")
    selected = build_render_plan(base_plan, max_trajectories_per_task=max_trajectories_per_task,
                                 points_per_trajectory=2)
    group_by_cache = {} if group_by_cache is None else group_by_cache
    if not isinstance(group_by_cache, dict):
        raise ValueError("deploy_dense.group_by_cache must map exact cache paths to group labels")
    groups = {}
    for path, label in group_by_cache.items():
        if not isinstance(path, str) or not path or not isinstance(label, str) or not label.strip() or label != label.strip():
            raise ValueError("deploy_dense.group_by_cache needs nonempty exact paths and group labels")
        root = Path(os.path.expandvars(path)).expanduser().resolve()
        if root.name == "processed_dataset":
            root = root.parent
        key = str(root)
        if key in groups:
            raise ValueError("deploy_dense.group_by_cache has duplicate canonical cache paths")
        groups[key] = label
    admitted_caches = {str(w.trajectory.cache_root) for w in base_plan.windows}
    if groups and set(groups) != admitted_caches:
        raise ValueError("deploy_dense.group_by_cache must exactly cover the held-out cache paths")
    labels = set(groups.values())
    if len({_slug(label) for label in labels}) != len(labels):
        raise ValueError("Dense evaluation group aliases must be unique")
    seen, windows, task_roles = set(), [], defaultdict(set)
    settings = deepcopy(selected.settings)
    pairs, seed = settings["stride_pairs"], settings["seed"]
    for source in selected.windows:
        row = source.trajectory
        identity = (str(row.cache_root), row.task, row.trajectory_id)
        if identity in seen:
            continue
        seen.add(identity)
        if "C" in source.roles and "D" in source.roles:
            roles = (("B",) if include_oracle and "B" in source.roles else ()) + ("C", "D")
        elif "A" in source.roles:
            task_policy = settings.get("stage_routing", {}).get("task_policies", {}).get(row.task, {})
            # Keep the requested canonical C diagnostic for a task whose D
            # route is forbidden; accumulated progress still comes only from A.
            retain_c = task_policy.get("d_allowed") is False and "C" in source.roles
            roles = ("A", "C") if retain_c else ("A",)
        else:
            raise ValueError(f"Dense deployment evaluation needs C+D or an A-only route for task {row.task!r}; "
                             "use ordinary stage evaluation or explicitly select a supported dense cohort")
        task_roles[row.task].update(roles)
        pois = sorted(set(range(0, row.num_frames, frame_stride)) | {row.num_frames - 1})
        for poi in pois:
            key = f"{seed}:{row.task}:{row.cache_root}:{row.trajectory_id}:{poi}"
            pair = pairs[int.from_bytes(sha256(key.encode()).digest()[:8], "big") % len(pairs)]
            windows.append(EvaluationWindow(row, poi, tuple(pair), roles, groups.get(str(row.cache_root), source.group)))
    tasks = tuple(dict(task, roles=sorted(task_roles[task["task"]])) for task in selected.tasks)
    for task in tasks:
        if "stage_routing" in task:
            task["stage_routing"] = deepcopy(task["stage_routing"])
            task["stage_routing"]["role_window_counts"] = {
                role: sum(role in window.roles for window in windows if window.trajectory.task == task["task"])
                for role in ("A", "B", "C", "D")}
    full = max_trajectories_per_task is None and all(
        task["selected_trajectories"] == task["available_trajectories"] for task in base_plan.tasks)
    settings.update(purpose="deploy_dense", metric_scope="full_heldout_dense" if full else "fixed_heldout_dense_cohort",
                    windows_per_trajectory=None, frame_stride=frame_stride,
                    source_fps=10, include_final_frame=True, group_by_cache=groups,
                    roles=sorted(set().union(*task_roles.values())), include_oracle=include_oracle,
                    selection="fixed base-plan trajectories; every frame_stride source frames plus final frame",
                    route_policy="native C -> D; optional B oracle; A only without an eligible C+D route")
    if "stage_routing" in settings:
        settings["route_policy"] += "; tasks forbidding D retain requested C only as a canonical stage diagnostic"
    return EvaluationPlan(base_plan.dataset, tuple(windows), tasks, settings)


def dense_series(result, plan) -> list[dict]:
    expected_groups = {(str(w.trajectory.cache_root), w.trajectory.task, w.trajectory.trajectory_id): w.group
                       for w in plan.windows}
    for record in result.get("records", []):
        identity = record["cache"], record["task"], record["trajectory_id"]
        if identity in expected_groups and record.get("group") != expected_groups[identity]:
            raise ValueError("Dense record group differs from its exact held-out cache identity")
    grouped = _validated_records(result, plan)
    trajectories = {(str(w.trajectory.cache_root), w.trajectory.task, w.trajectory.trajectory_id): w.trajectory
                    for w in plan.windows}
    series = []
    for identity, samples in sorted(grouped.items()):
        trajectory = trajectories[identity]
        points = []
        staged = "D" in next(iter(samples.values()))
        for poi, records in sorted(samples.items()):
            if staged:
                c, d = records["C"], records["D"]
                predicted_stage, true_stage = int(c["predictions"][record_poi_slot(c)]), int(c["targets"][record_poi_slot(c)])
                point = {"poi": poi, "prediction": predicted_stage + d["predictions"][record_poi_slot(d)],
                         "target": true_stage + trajectory.within_stage_progress[poi],
                         "predicted_stage": predicted_stage, "target_stage": true_stage,
                         "focus_progress": d["predictions"][record_poi_slot(d)], "focus_target": d["targets"][record_poi_slot(d)]}
                if "B" in records:
                    point["oracle"] = true_stage + records["B"]["predictions"][record_poi_slot(records["B"])]
            else:
                a = records["A"]
                point = {"poi": poi, "prediction": a["predictions"][record_poi_slot(a)], "target": a["targets"][record_poi_slot(a)]}
            readout = records["C"] if staged else records["A"]
            point.update(poi_slot=record_poi_slot(readout),
                         source_frame=readout["frame_indices"][record_poi_slot(readout)])
            points.append(point)
        scale = len(trajectory.stage_prompts) if staged else 1
        rmse = math.sqrt(sum((p["prediction"] - p["target"]) ** 2 for p in points) / len(points))
        metrics = {"raw_rmse": rmse, "normalized_rmse": rmse / scale, "evaluated_pois": len(points)}
        if staged:
            metrics["stage_accuracy"] = sum(p["predicted_stage"] == p["target_stage"] for p in points) / len(points)
            metrics["focus_progress_rmse"] = math.sqrt(sum((p["focus_progress"] - p["focus_target"]) ** 2 for p in points) / len(points))
        if "oracle" in points[0]:
            metrics["oracle_raw_rmse"] = math.sqrt(sum((p["oracle"] - p["target"]) ** 2 for p in points) / len(points))
        first_record = next(iter(next(iter(samples.values())).values()))
        series.append({"task": trajectory.task, "group": first_record.get("group", trajectory.task),
                       "trajectory_id": trajectory.trajectory_id,
                       "cache": str(trajectory.cache_root), "view_names": list(trajectory.view_names),
                       "num_frames": trajectory.num_frames, "stage_prompts": list(trajectory.stage_prompts),
                       "route": "predicted_stage_route" if staged else "compressed_whole_task",
                       "prediction_roles": ["C", "D"] if staged else ["A"],
                       "poi_slots": sorted({point["poi_slot"] for point in points}),
                       "units": "stage_units_[0,N]" if staged else "whole_task_[0,1]",
                       "scale": scale, "points": points, "metrics": metrics})
    return series


def _slug(task):
    return re.sub(r"[^a-zA-Z0-9]+", "-", task).strip("-").lower() or "task"


def dense_metrics(series) -> dict:
    grouped = defaultdict(list)
    for row in series:
        grouped[row["task"]].append(row)
    metrics, names = {}, set()
    for task, members in sorted(grouped.items()):
        alias = _slug(task)
        if alias in names:
            raise ValueError("Dense task aliases must be unique")
        names.add(alias)
        if len({row["route"] for row in members}) != 1:
            raise ValueError("A dense task must have one consistent deployment route")
        points = [p for row in members for p in row["points"]]
        staged = members[0]["route"] == "predicted_stage_route"
        suffix = "predicted_stage_route_accumulated_progress" if staged else "compressed_whole_task_progress"
        rmse = math.sqrt(sum((p["prediction"] - p["target"]) ** 2 for p in points) / len(points))
        metrics[f"overlay_acc/{alias}_{suffix}_raw_rmse"] = rmse
        metrics[f"eval_quality/{alias}/{suffix}_raw_rmse"] = rmse
        normalized_mse = sum((p["prediction"] - p["target"]) ** 2 / row["scale"] ** 2
                             for row in members for p in row["points"]) / len(points)
        metrics[f"overlay_acc/{alias}_{suffix}_normalized_rmse"] = math.sqrt(normalized_mse)
        metrics[f"eval_quality/{alias}/evaluated_pois"] = len(points)
        metrics[f"eval_quality/{alias}/evaluated_trajectories"] = len(members)
        if staged:
            metrics[f"eval_quality/{alias}/predicted_stage_route_stage_accuracy"] = sum(
                p["predicted_stage"] == p["target_stage"] for p in points) / len(points)
            if all("oracle" in p for p in points):
                key = "oracle_stage_route_accumulated_progress_raw_rmse"
                value = math.sqrt(sum((p["oracle"] - p["target"]) ** 2 for p in points) / len(points))
                metrics[f"overlay_acc/{alias}_{key}"] = value
                metrics[f"eval_quality/{alias}/{key}"] = value
    # Group labels come solely from exact cache identity. They never replace
    # the task string passed to the model or alter any stage/focus prompt.
    groups = defaultdict(list)
    for row in series:
        groups[row.get("group", row["task"])].append(row)

    def accumulated(members):
        values = [(point, row["scale"]) for row in members for point in row["points"]]
        if not values:
            raise ValueError("Dense accumulated metrics require measured POIs")
        support = len(values)
        errors = [abs(point["prediction"] - point["target"]) for point, _ in values]
        normalized_errors = [error / scale for error, (_, scale) in zip(errors, values, strict=True)]
        baseline = [abs(.5 - point["target"] / scale) for point, scale in values]
        result = {"support": support, "trajectory_support": len(members),
                  "raw_rmse": math.sqrt(sum(error ** 2 for error in errors) / support),
                  "raw_mae": sum(errors) / support,
                  "normalized_rmse": math.sqrt(sum(error ** 2 for error in normalized_errors) / support),
                  "normalized_mae": sum(normalized_errors) / support,
                  "normalized_accuracy_within_0.1": sum(error <= .1 + 1e-8 for error in normalized_errors) / support,
                  "constant_0.5_normalized_rmse": math.sqrt(sum(error ** 2 for error in baseline) / support),
                  "constant_0.5_normalized_mae": sum(baseline) / support,
                  "constant_0.5_normalized_accuracy_within_0.1": sum(error <= .1 + 1e-8 for error in baseline) / support}
        low = [(point, scale) for point, scale in values if point["target"] / scale < .05]
        result["normalized_gt_lt_0.05_support"] = len(low)
        if low:
            low_errors = [abs(point["prediction"] - point["target"]) / scale for point, scale in low]
            result["normalized_gt_lt_0.05_rmse"] = math.sqrt(sum(error ** 2 for error in low_errors) / len(low))
            result["normalized_gt_lt_0.05_accuracy_within_0.1"] = sum(error <= .1 + 1e-8 for error in low_errors) / len(low)
        return result

    for prefix, members in [("overall", series),
                            *((f"task/{task}", rows) for task, rows in sorted(grouped.items())),
                            *((f"group/{group}", rows) for group, rows in sorted(groups.items()))]:
        for key, value in accumulated(members).items():
            metrics[f"accumulated/{prefix}/{key}"] = value
    for group, members in sorted(groups.items()):
        alias = _slug(group)
        if not all(row.get("group", row["task"]) == row["task"] for row in members):
            value = accumulated(members)
            routes = {row["route"] for row in members}
            suffix = ("predicted_stage_route_accumulated_progress" if routes == {"predicted_stage_route"}
                      else "compressed_whole_task_progress" if routes == {"compressed_whole_task"}
                      else "mixed_route_accumulated_progress")
            metrics[f"overlay_acc/{alias}_{suffix}_raw_rmse"] = value["raw_rmse"]
            metrics[f"overlay_acc/{alias}_{suffix}_normalized_rmse"] = value["normalized_rmse"]
    return metrics


def _background(row):
    canvas = Image.new("RGB", SIZE, "#f4f6fa")
    draw = ImageDraw.Draw(canvas)
    sparse = len(row["points"]) <= 2
    slots = ",".join(map(str, row["poi_slots"]))
    roles = "C + D" if row["route"] == "predicted_stage_route" else "A"
    _text(draw, (16, 310), f"Blue: {roles}, POI slot {slots}", 17, "#1670dc")
    _text(draw, (355, 310), "Green: ground truth", 17, "#139967")
    _text(draw, (650, 310), f"Orange: B oracle, POI slot {slots}" if "oracle" in row["points"][0] else "", 17, "#cc7c16")
    quantity = "accumulated progress (stage units)" if row["route"] == "predicted_stage_route" else "whole-task progress [0, 1]"
    _text(draw, (16, 337), f"{roles} | POI slot {slots} | {quantity}", 18, bold=True)
    if sparse:
        _text(draw, (620, 339), f"Only {len(row['points'])} POIs; intermediate frames unevaluated", 17, "#9c6010", max_width=635)
    box = (65, 375, 1255, 570)
    draw.rectangle(box, fill="white", outline="#b7c5d5")
    maximum = row["scale"]
    for tick in sorted({0, maximum / 2, maximum}):
        y = box[3] - tick / maximum * (box[3] - box[1])
        draw.line((box[0], y, box[2], y), fill="#e1e6ee")
        _text(draw, (12, y - 10), f"{tick:g}", 15)
    for field, color, width in (("oracle", "#dc8b25", 2), ("target", "#15996b", 3), ("prediction", "#1670dc", 2)):
        points = [(box[0] + p["poi"] / max(1, row["num_frames"] - 1) * (box[2] - box[0]),
                   box[3] - p[field] / maximum * (box[3] - box[1])) for p in row["points"] if field in p]
        if len(points) >= 3:
            draw.line(points, fill=color, width=width)
        for x, y in points:
            radius = 4 if sparse else 1
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
    _text(draw, (65, 578), "Source frame 0", 14)
    _text(draw, (1060, 578), f"Source frame {row['num_frames'] - 1}", 14)
    if row["route"] == "predicted_stage_route":
        for field, y, label in (("target_stage", 247, "C GT"), ("predicted_stage", 276, "C pred")):
            _text(draw, (16, y), label, 16)
            if sparse:
                draw.rectangle((75, y, 1255, y + 19), fill="#e1e6ee")
            for index, point in enumerate(row["points"]):
                if sparse:
                    center = 75 + point["poi"] / max(1, row["num_frames"] - 1) * 1180
                    left, right = max(75, center - 4), min(1255, center + 4)
                else:
                    left = 75 + index / len(row["points"]) * 1180
                    right = 75 + (index + 1) / len(row["points"]) * 1180
                draw.rectangle((left, y, right, y + 19), fill=COLORS[point[field] % len(COLORS)])
    return canvas


def _frame(background, row, trajectory, point, *, step, elapsed, provenance, selection, frame_stride):
    canvas = background.copy()
    draw = ImageDraw.Draw(canvas)
    label = row.get("group", row["task"])
    _text(draw, (16, 7), f"{label} | {row['task']} | {selection} | step {step} | {row['trajectory_id']}", 20, bold=True, max_width=1248)
    mode = "DURING TRAINING" if provenance["mode"] == "during_training" else "CHECKPOINT REPLAY (generated after training)"
    _text(draw, (16, 36), f"{mode} | POI slot {point['poi_slot']} | source frame {point['source_frame']} | {len(row['points'])} points, stride {frame_stride} | elapsed {elapsed:.1f}s", 16, max_width=1248)
    views = trajectory.images((point["poi"],))
    cell = 1248 // len(views)
    for index, (name, source) in enumerate(zip(trajectory.view_names, views, strict=True)):
        left = 16 + cell * index
        draw.rectangle((left, 65, left + cell - 6, 235), fill="#132139")
        resized = ImageOps.contain(source, (cell - 10, 166), Image.Resampling.LANCZOS)
        canvas.paste(resized, (left + (cell - 6 - resized.width) // 2, 67 + (166 - resized.height) // 2))
        ImageDraw.Draw(canvas).rectangle((left + 3, 69, left + min(cell - 10, 100), 90), fill="#132139")
        _text(ImageDraw.Draw(canvas), (left + 6, 68), name, 15, "white", max_width=cell - 18)
    draw = ImageDraw.Draw(canvas)
    x = 65 + point["poi"] / max(1, row["num_frames"] - 1) * 1190
    draw.line((x, 375, x, 570), fill="#9caaba", width=2)
    error = point["prediction"] - point["target"]
    metric_label = "Accumulated RMSE" if row["route"] == "predicted_stage_route" else "Whole-task RMSE"
    readout = f"C[{point['poi_slot']}] + D[{point['poi_slot']}]" if row["route"] == "predicted_stage_route" else f"A[{point['poi_slot']}]"
    _text(draw, (16, 606), f"{readout} prediction {point['prediction']:.3f} | GT {point['target']:.3f} | {metric_label} raw {row['metrics']['raw_rmse']:.4f}, normalized {row['metrics']['normalized_rmse']:.4f}",
          20, "#b83f49" if abs(error) > .1 else "#146b54", bold=True, max_width=1248)
    if "predicted_stage" in point:
        _text(draw, (16, 638), f"C[{point['poi_slot']}] stage {point['predicted_stage']} / GT {point['target_stage']} (zero-based) | accuracy {row['metrics']['stage_accuracy']:.1%} | D[{point['poi_slot']}] {point['focus_progress']:.3f} / focus-relative GT {point['focus_target']:.3f}", 17, max_width=1248)
        _text(draw, (16, 667), f"Predicted stage: {row['stage_prompts'][point['predicted_stage']]}", 16, max_width=1248)
    coverage = "Markers show measured points only; intermediate frames unevaluated." if len(row["points"]) <= 2 else "Lines connect measured points; no smoothing."
    _text(draw, (16, 696), f"Offline POI replay; inputs may include later frames. {coverage} Playback is not original speed.", 14, "#57657a", max_width=1248)
    return canvas


def _video(path, frames, fps):
    command = [require_encoder(), "-hide_banner", "-loglevel", "error", "-nostdin", "-f", "rawvideo",
               "-pix_fmt", "rgb24", "-s", f"{SIZE[0]}x{SIZE[1]}", "-r", str(fps), "-i", "pipe:0",
               "-an", "-c:v", "libx264", "-threads", "2", "-preset", "fast", "-crf", "23",
               "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-n", str(path)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for frame in frames:
            process.stdin.write(frame.tobytes())
        process.stdin.close()
        error = process.stderr.read()
        if process.wait() != 0:
            raise RuntimeError(f"MP4 encoder failed: {error.decode(errors='replace')[-1000:]}")
    except BrokenPipeError as error:
        detail = process.stderr.read().decode(errors="replace")[-1000:]
        process.wait()
        raise RuntimeError(f"MP4 encoder closed its input: {detail}") from error
    except BaseException:
        process.kill()
        process.wait()
        raise
    finally:
        if not process.stdin.closed:
            try:
                process.stdin.close()
            except OSError:
                pass
        process.stderr.close()


def render_dense_evaluation(result, plan, *, output_dir, step, training_elapsed_seconds,
                            max_frames=96, replay_fps=8, provenance=None):
    require_encoder()
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError("Dense optimizer step must be a nonnegative integer")
    if isinstance(max_frames, bool) or not isinstance(max_frames, int) or max_frames < 1:
        raise ValueError("Dense max_frames must be positive")
    if (isinstance(replay_fps, bool) or not isinstance(replay_fps, (int, float))
            or not math.isfinite(replay_fps) or not .1 <= replay_fps <= 20):
        raise ValueError("Dense replay_fps must be in [0.1, 20]")
    if (isinstance(training_elapsed_seconds, bool) or not isinstance(training_elapsed_seconds, (int, float))
            or not math.isfinite(training_elapsed_seconds) or training_elapsed_seconds < 0):
        raise ValueError("Dense training elapsed time must be finite and nonnegative")
    provenance = dict(provenance or {"mode": "during_training"})
    if provenance.get("mode") not in ("during_training", "checkpoint_replay"):
        raise ValueError("Dense provenance must distinguish training from checkpoint replay")
    if provenance["mode"] == "checkpoint_replay" and not all(provenance.get(k) for k in ("checkpoint", "weights_sha256", "source_run")):
        raise ValueError("Checkpoint replay requires checkpoint, weights digest, and source run")
    started = time.perf_counter()
    series = dense_series(result, plan)
    metrics = dense_metrics(series)
    directory = Path(output_dir).expanduser().resolve() / f"step-{step:06d}"
    directory.mkdir(parents=True, exist_ok=False)
    grouped = defaultdict(list)
    for row in series:
        grouped[row.get("group", row["task"])].append(row)
    trajectories = {(str(w.trajectory.cache_root), w.trajectory.task, w.trajectory.trajectory_id): w.trajectory for w in plan.windows}
    items, shortfall = [], 0
    for group, rows in sorted(grouped.items()):
        ranked = sorted(rows, key=lambda row: (row["metrics"]["raw_rmse"], row["cache"], row["trajectory_id"]))
        choices = [("best_raw_rmse", ranked[0])]
        if len(ranked) > 1:
            choices.append(("worst_raw_rmse", ranked[-1]))
        else:
            shortfall += 1
        for selection, row in choices:
            key = f"{_slug(group)}/{selection}"
            stem = directory / key
            stem.parent.mkdir(parents=True, exist_ok=True)
            image_path, video_path = stem.with_suffix(".png"), stem.with_suffix(".mp4")
            points = row["points"]
            count = min(max_frames, len(points))
            indices = [len(points) // 2] if count == 1 else [round(i * (len(points) - 1) / (count - 1)) for i in range(count)]
            background = _background(row)
            trajectory = trajectories[row["cache"], row["task"], row["trajectory_id"]]
            def frame(point):
                return _frame(background, row, trajectory, point, step=step, elapsed=training_elapsed_seconds,
                              provenance=provenance, selection=selection, frame_stride=plan.settings["frame_stride"])
            worst = max(points, key=lambda p: (abs(p["prediction"] - p["target"]), -p["poi"]))
            frame(worst).save(image_path)
            _video(video_path, (frame(points[i]) for i in indices), replay_fps)
            items.append({"key": key, "task": row["task"], "group": group, "cache": row["cache"],
                          "trajectory_id": row["trajectory_id"], "selection": selection,
                          "route": row["route"], "units": row["units"], "metrics": row["metrics"],
                          "prediction_roles": row["prediction_roles"], "poi_slots": row["poi_slots"],
                          "candidate_trajectories": len(rows), "image": str(image_path), "video": str(video_path),
                          "image_sha256": sha256(image_path.read_bytes()).hexdigest(),
                          "video_sha256": sha256(video_path.read_bytes()).hexdigest(),
                          "image_poi": worst["poi"], "image_poi_slot": worst["poi_slot"],
                          "video_pois": [points[i]["poi"] for i in indices],
                          "video_poi_slots": [points[i]["poi_slot"] for i in indices]})
    bundle = {"schema": SCHEMA, "optimizer_steps": step, "training_elapsed_seconds": training_elapsed_seconds,
              "generated_at": datetime.now(timezone.utc).isoformat(), "provenance": provenance,
              "plan_sha256": result["plan_sha256"], "source_plan_sha256": plan.settings["source_plan_sha256"],
              "records_sha256": sha256(json.dumps(result["records"], sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest(),
              "frame_stride": plan.settings["frame_stride"], "replay_fps": replay_fps,
              "source_fps": plan.settings.get("source_fps", 10), "include_final_frame": True,
              "metric_scope": plan.settings["metric_scope"],
              "group_by_cache": plan.settings.get("group_by_cache", {}),
              "selection_policy": "best/worst raw RMSE per exact-cache evaluation group within the recorded dense cohort",
              "accumulated_progress": "C_POI_stage_id + D_POI_progress; GT_stage_id + GT_within_stage_progress; raw units [0,N]",
              "prediction_readout": "saved evaluation POI; input slot and source frame recorded per point; offline replay",
              "available_trajectories": len(series), "evaluated_pois": sum(len(row["points"]) for row in series),
              "selection_shortfall_count": shortfall, "items": items, "series": series,
              "metrics": metrics, "rendering_seconds": time.perf_counter() - started,
              "manifest": str(directory / "manifest.json")}
    Path(bundle["manifest"]).write_text(json.dumps(bundle, indent=2, allow_nan=False) + "\n")
    return bundle


def log_dense_bundle(run, sdk, bundle, *, step_metric="optimizer_steps", history_sink=None):
    """Publish dense media, optionally into a caller-owned optimizer-step row.

    RunLogger supplies a sink that buffers all scalar/media producers until its
    explicit commit_step. The default keeps the standalone archived-run replay
    behavior: append a new history row, never write to a completed past step.
    """
    if bundle.get("schema") != SCHEMA:
        raise ValueError("Invalid dense evaluation bundle")
    # Scalars are emitted exactly once by log_evaluation under their population
    # namespace. The manifest still contains every original dense metric.
    row = {step_metric: bundle["optimizer_steps"],
           "eval_media/selection_shortfall_count": bundle["selection_shortfall_count"]}
    data = []
    for item in bundle["items"]:
        metric_label = ("Predicted-stage C→D accumulated RMSE" if item["route"] == "predicted_stage_route"
                        else "A whole-task RMSE")
        caption = (f"{item.get('group', item['task'])} | {item['task']} | {item['selection']} among {item['candidate_trajectories']} held-out trajectories | "
                   f"step {bundle['optimizer_steps']} | {bundle['provenance']['mode']} | {item['units']} | "
                   f"POI slots {item.get('poi_slots', [5])}, source frames {item['video_pois'][0]}–{item['video_pois'][-1]} | "
                   f"{metric_label}: raw {item['metrics']['raw_rmse']:.4f}, normalized {item['metrics']['normalized_rmse']:.4f}; "
                   "sampled playback, not original speed")
        row[f"deploy_dense_samples/{item['key']}"] = sdk.Image(item["image"], caption=caption)
        row[f"deploy_dense_videos/{item['key']}"] = sdk.Video(item["video"], format="mp4", caption=caption)
        data.append([bundle["optimizer_steps"], item["task"], item.get("group", item["task"]), item["selection"], item["trajectory_id"],
                     item["route"], item["units"], item["metrics"]["raw_rmse"], item["metrics"]["normalized_rmse"],
                     item["metrics"].get("oracle_raw_rmse"), item["metrics"].get("focus_progress_rmse"),
                     item["metrics"].get("stage_accuracy"),
                     item["candidate_trajectories"], item["metrics"]["evaluated_pois"], len(item["video_pois"]),
                     bundle["frame_stride"], bundle["training_elapsed_seconds"], bundle["generated_at"],
                     bundle["provenance"]["mode"], bundle["provenance"].get("weights_sha256")])
    row["eval_media/selection_manifest"] = sdk.Table(
        columns=["optimizer_steps", "task", "group", "selection", "trajectory_id", "route", "units",
                 "primary_route_raw_rmse", "primary_route_normalized_rmse", "oracle_B_accumulated_raw_rmse_diagnostic",
                 "D_predicted_focus_relative_rmse_diagnostic", "stage_accuracy_diagnostic",
                 "candidate_trajectories", "evaluated_pois", "video_frames", "frame_stride", "training_elapsed_seconds",
                 "generated_at", "mode", "weights_sha256"], data=data)
    from .evaluation_reporting import evaluation_report

    # Keep optional dense-cohort reports separate from the primary evaluation's
    # scope (which may use a different, sampled held-out plan).
    run.summary["deploy_dense_evaluation_report"] = evaluation_report(
        bundle["metrics"], scope=bundle.get("metric_scope", "dense"))
    run.define_metric(step_metric)
    for key in row:
        if key != step_metric:
            run.define_metric(key, step_metric=step_metric)
    if history_sink is None:
        run.log(row)
    else:
        history_sink(row, step=bundle["optimizer_steps"])
    artifact = sdk.Artifact(f"grana-{run.id}-deploy-dense", type="evaluation-dense",
                            metadata={"optimizer_steps": bundle["optimizer_steps"], "provenance": bundle["provenance"],
                                      "plan_sha256": bundle["plan_sha256"]})
    artifact.add_file(bundle["manifest"], name="manifest.json")
    return run.log_artifact(artifact, aliases=["latest", f"step-{bundle['optimizer_steps']:06d}"])
