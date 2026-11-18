#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Freeze held-out windows, then compare existing Grana checkpoints without training.

Prepare from explicit cohort runs and checkpoints with ``--prepare --spec FILE``.
Review the generated manifest before ``--run --arm NAME``.
All image payloads stay in their existing external NPY files. Ground truth is
used only for offline metrics after the label-free predictor returns.
"""
from __future__ import annotations

import argparse
import datetime
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def input_digest(episode, window, images):
    """Bind actual label-free inputs, including full RGB bytes and slot order."""
    descriptor = {key: episode[key] for key in ("task", "compressed_task_prompt", "stage_prompts")}
    descriptor["frame_indices"] = window["frame_indices"]
    hashed = hashlib.sha256(json.dumps(descriptor, sort_keys=True).encode())
    for view, frames in images.items():
        hashed.update(view.encode())
        for frame in frames:
            hashed.update(str(frame.shape).encode())
            hashed.update(frame.tobytes())
    return hashed.hexdigest()


def arrays_for(episode, views):
    import numpy as np
    arrays = {}
    for view in views:
        path = Path(episode["frame_paths"][view])
        frames = np.load(path, mmap_mode="r", allow_pickle=False)
        if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[0] != episode["num_frames"] or frames.shape[-1] != 3:
            raise ValueError(f"Invalid RGB frame array: {path}")
        arrays[view] = frames
    return arrays


def select_images(arrays, indices):
    import numpy as np
    return {view: tuple(np.array(frames[index], copy=True) for index in indices) for view, frames in arrays.items()}


def comparison_spec(path):
    """Resolve user-selected cohorts and checkpoints without campaign conventions."""
    if path is None:
        raise ValueError("--prepare requires --spec with explicit cohorts and arms")
    path = Path(path).expanduser().resolve()
    spec = read(path)
    if not isinstance(spec, dict) or set(spec) != {"cohorts", "arms"}:
        raise ValueError("Comparison spec requires exactly cohorts and arms")
    for category in ("cohorts", "arms"):
        entries = spec[category]
        if not isinstance(entries, list) or not entries or any(not isinstance(item, dict) for item in entries):
            raise ValueError(f"{category} must be a nonempty list of objects")
        identifiers = [item.get("id") for item in entries]
        if any(not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", value) for value in identifiers):
            raise ValueError("Comparison identifiers must be path-safe names")
        if len(set(identifiers)) != len(identifiers):
            raise ValueError(f"Duplicate {category} identifier")
    cohorts = {item["id"] for item in spec["cohorts"]}
    for item in spec["cohorts"]:
        if set(item) != {"id", "run_dir"}:
            raise ValueError("Each cohort requires id and run_dir")
    for item in spec["arms"]:
        if not {"id", "checkpoint", "cohorts"} <= set(item) or set(item) - {"id", "label", "checkpoint", "cohorts"}:
            raise ValueError("Each arm requires id, checkpoint and cohorts; label is optional")
        selected = item["cohorts"]
        if (not isinstance(selected, list) or not selected or any(not isinstance(value, str) for value in selected)
                or len(set(selected)) != len(selected) or not set(selected) <= cohorts):
            raise ValueError("Arm cohorts must name distinct declared cohorts")
        if "label" in item and (not isinstance(item["label"], str) or not item["label"].strip()):
            raise ValueError("Arm label must be nonempty text")
    if {cohort for item in spec["arms"] for cohort in item["cohorts"]} != cohorts:
        raise ValueError("Every cohort must be selected by at least one checkpoint arm")
    for category, field in (("cohorts", "run_dir"), ("arms", "checkpoint")):
        for item in spec[category]:
            value = item[field]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field} must be a nonempty path")
            resolved = Path(value).expanduser()
            item[field] = str((path.parent / resolved).resolve())
    return path, spec


def prepare(args):
    import numpy as np
    from datasets import load_from_disk
    from grana.data.cache import CachedTrajectories
    from grana.data.sampling import frame_indices
    from grana.data.frame_sampling import DEFAULT_EVAL_STRIDE_PAIRS

    spec_path, selection = comparison_spec(args.spec)
    if args.manifest.exists():
        raise FileExistsError(f"Use a new manifest: {args.manifest}")
    context_stride, tail_stride = DEFAULT_EVAL_STRIDE_PAIRS[0]
    views = None
    episodes, provenance, task_names = [], [], {}
    for cohort in selection["cohorts"]:
        task_alias, run = cohort["id"], Path(cohort["run_dir"])
        plan_path = run / "evaluation_plan.json"
        plan = read(plan_path)
        # Selection follows saved cohort order, before reading any predictions.
        selected, seen = [], set()
        for window in plan["windows"]:
            key = (window["cache"], window["trajectory_id"])
            if key not in seen:
                selected.append(window)
                seen.add(key)
            if len(selected) == args.episodes_per_task:
                break
        if len(selected) != args.episodes_per_task:
            raise ValueError(f"Insufficient distinct held-out trajectories: {task_alias}")
        final_path = Path(read(run / "result.json")["checkpoint_path"])
        metadata = read(final_path / "metadata.json")
        config = metadata["config"]
        checkpoint_views = config["data"]["views"]
        if views is None:
            views = checkpoint_views
        if (not isinstance(views, list) or len(views) != 4 or any(not isinstance(view, str) or not view for view in views)
                or len(set(views)) != 4 or checkpoint_views != views):
            raise ValueError("Comparison requires the same four ordered checkpoint views")
        trajectories = {}
        for entry in config["evaluation"]["caches"]:
            original = entry if isinstance(entry, dict) else {"path": entry}
            # Reuse native path normalization and selectors, including disjoint
            # entries with different prompts or camera orders in one cache.
            for trajectory in CachedTrajectories([original], views).records:
                key = (trajectory.cache_root, trajectory.trajectory_id)
                if key in trajectories:
                    raise ValueError(f"Ambiguous checkpoint evaluation cache membership: {key}")
                trajectories[key] = trajectory, original
        task_names[task_alias] = sorted({item["task"] for item in selected})
        for number, selected_window in enumerate(selected, 1):
            cache = Path(selected_window["cache"]).expanduser().resolve()
            key = (cache, selected_window["trajectory_id"])
            if key not in trajectories:
                raise ValueError(f"Saved trajectory is absent from checkpoint evaluation caches: {key}")
            trajectory, original = trajectories[key]
            frame_paths = dict(zip(trajectory.view_names, map(str, trajectory.frame_paths), strict=True))
            if set(frame_paths) != set(views):
                raise ValueError("Cohort camera aliases must match the checkpoint views")
            data = load_from_disk(str(cache / "processed_dataset"))
            raw = next(row for row in data if row["id"] == trajectory.trajectory_id)
            indices = sorted(set(np.linspace(0, trajectory.num_frames - 1, args.windows, dtype=int).tolist()))
            episode = {"id": f"{task_alias}_{number:02d}", "task": trajectory.task,
                       "task_alias": task_alias, "cache": str(cache), "trajectory_id": trajectory.trajectory_id,
                       "num_frames": trajectory.num_frames, "view_names": views,
                       "frame_paths": {view: frame_paths[view] for view in views},
                       "compressed_task_prompt": trajectory.compressed_task_prompt,
                       "compressed_prompt_origin": "manifest override" if "prompt" in original else "cached task definition",
                       "stage_prompts": list(trajectory.stage_prompts),
                       "cached_fps": raw.get("data_gen_output_fps"), "original_timestamps_known": False,
                       "poi_indices": indices, "windows": []}
            arrays = arrays_for(episode, views)
            for poi in indices:
                window = {"poi": poi, "frame_indices": list(frame_indices(poi, trajectory.num_frames, context_stride, tail_stride)),
                          "gt": {"whole_progress": trajectory.progress[poi], "stage_id": trajectory.stage_ids[poi],
                                 "within_stage_progress": trajectory.within_stage_progress[poi]}}
                window["input_sha256"] = input_digest(episode, window, select_images(arrays, window["frame_indices"]))
                episode["windows"].append(window)
            episode["all_observed_gt_stages"] = sorted(set(trajectory.stage_ids))
            episode["sampled_gt_stages"] = sorted({window["gt"]["stage_id"] for window in episode["windows"]})
            episode["unsampled_gt_stages"] = sorted(set(trajectory.stage_ids) - set(episode["sampled_gt_stages"]))
            episode["gt_stage_backtracks"] = sum(b < a for a, b in zip(trajectory.stage_ids, trajectory.stage_ids[1:]))
            episodes.append(episode)
        provenance.append({"task_alias": task_alias, "source_plan": str(plan_path), "source_plan_sha256": digest(plan_path),
                           "checkpoint_metadata": str(final_path / "metadata.json"),
                           "checkpoint_metadata_sha256": digest(final_path / "metadata.json")})
    arms = []
    for source in selection["arms"]:
        checkpoint = Path(source["checkpoint"])
        metadata = read(checkpoint / "metadata.json")
        if metadata["config"]["data"]["views"] != views:
            raise ValueError("Every comparison checkpoint must use the selected ordered views")
        arms.append({"id": source["id"], "label": source.get("label", source["id"]),
                     "cohorts": source["cohorts"],
                     "checkpoint": str(checkpoint), "checkpoint_metadata_sha256": digest(checkpoint / "metadata.json"),
                     "weights_sha256": metadata["weights_sha256"],
                     "tasks": sorted({task for cohort_id in source["cohorts"] for task in task_names[cohort_id]})})
    total = sum(len(episode["windows"]) * 3 for arm in arms for episode in episodes if arm_includes_episode(arm, episode))
    manifest = {"schema_version": 1, "created_utc": now(), "view_names": views, "roles": ["A", "C", "D"],
                "ctsp": {"context_stride": context_stride, "tail_stride": tail_stride, "poi_slot": 5},
                "selection": {"episodes_per_task": args.episodes_per_task, "windows_per_episode": args.windows,
                              "rule": "First distinct trajectories in saved held-out plan order; np.linspace(0,T-1,N,dtype=int), deduplicated; independent of predictions."},
                "interpretation": {"A": "whole-task progress", "C": "task-local stage ID",
                                   "D": "progress relative to model C focus; target 1/WSP/0 for focus earlier/same/later than GT",
                                   "sampling": "offline CTSP replay includes two observations after POI; not causal online inference",
                                   "scope": "fixed selected held-out trajectories, not a full-corpus benchmark"},
                "expected_model_calls": total, "arms": arms, "episodes": episodes, "provenance": provenance,
                "source_spec": str(spec_path), "source_spec_sha256": digest(spec_path)}
    write_new(args.manifest, manifest)
    print(json.dumps({"status": "prepared", "manifest": str(args.manifest), "manifest_sha256": digest(args.manifest),
                      "episodes": len(episodes), "arms": len(arms), "model_calls": total}), flush=True)


def relative_target(focus, gt_stage, within):
    return 1.0 if focus < gt_stage else float(within) if focus == gt_stage else 0.0


def arm_includes_episode(arm, episode):
    """Use explicit cohort identity; older manifests carry task scope only."""
    if "cohorts" in arm:
        return episode["task_alias"] in arm["cohorts"]
    return episode["task"] in arm["tasks"]


def metrics(records):
    n = len(records)
    return {"poi_count": n, "A_rmse": math.sqrt(sum((r["A"] - r["gt"]["A"]) ** 2 for r in records) / n),
            "C_accuracy": sum(r["C"] == r["gt"]["stage_id"] for r in records) / n,
            "C_stage_mae": sum(abs(r["C"] - r["gt"]["stage_id"]) for r in records) / n,
            "D_relative_rmse": math.sqrt(sum((r["D"] - r["gt"]["D"]) ** 2 for r in records) / n)}


def check_output(output, num_stages):
    if output["roles"] != ["A", "C", "D"] or output["poi_slot"] != 5:
        raise ValueError("Unexpected inference readout")
    for role in ("A", "D"):
        values = output["progress"][role]["values"]
        if len(values) != 8 or not all(math.isfinite(x) and 0 <= x <= 1 for x in values):
            raise ValueError("Invalid progress predictions")
        if output["progress"][role]["poi_score"] != values[5]:
            raise ValueError("Progress POI must be slot 5")
    ids = output["stage"]["ids"]
    if len(ids) != 8 or any(type(x) is not int or not 0 <= x < num_stages for x in ids):
        raise ValueError("Invalid task-local stage IDs")
    if output["D_focus_stage"] != ids[5] or output["progress"]["D"]["focus_stage"] != ids[5]:
        raise ValueError("D focus must be the model's C prediction at slot 5")
    matrices = [(output["progress"][role]["bin_probabilities"], 10) for role in ("A", "D")]
    matrices.append((output["stage"]["probabilities"], num_stages))
    for matrix, columns in matrices:
        if len(matrix) != 8 or any(len(row) != columns or any(not math.isfinite(x) or not 0 <= x <= 1 for x in row)
                                   or abs(sum(row) - 1) > 1e-5 for row in matrix):
            raise ValueError("Invalid probability distribution")
    json.dumps(output, allow_nan=False)


def validate_manifest(manifest):
    from grana.data.sampling import frame_indices
    if manifest["schema_version"] != 1 or manifest["roles"] != ["A", "C", "D"]:
        raise ValueError("Unsupported manifest schema or roles")
    views = manifest["view_names"]
    ctsp = manifest.get("ctsp")
    if (not isinstance(views, list) or len(views) != 4
            or any(not isinstance(view, str) or not view for view in views) or len(set(views)) != 4 or not isinstance(ctsp, dict)
            or set(ctsp) != {"context_stride", "tail_stride", "poi_slot"}
            or any(type(ctsp[key]) is not int or ctsp[key] <= 0 for key in ("context_stride", "tail_stride"))
            or type(ctsp["poi_slot"]) is not int or ctsp["poi_slot"] != 5):
        raise ValueError("Expected full four-view CTSP replay with explicit positive integer strides and poi_slot 5")
    context_stride, tail_stride = ctsp["context_stride"], ctsp["tail_stride"]
    for category in ("arms", "episodes"):
        ids = [entry["id"] for entry in manifest[category]]
        if len(ids) != len(set(ids)) or any(not re.fullmatch(r"[A-Za-z0-9_-]+", value) for value in ids):
            raise ValueError("Identifiers must be unique and path-safe")
    for episode in manifest["episodes"]:
        if list(episode["frame_paths"]) != views:
            raise ValueError("Camera order mismatch")
        for window in episode["windows"]:
            if window["frame_indices"] != list(frame_indices(window["poi"], episode["num_frames"], context_stride, tail_stride)):
                raise ValueError("Window differs from fixed CTSP plan")
            gt = window["gt"]
            if not 0 <= gt["stage_id"] < len(episode["stage_prompts"]) or any(not math.isfinite(gt[key]) or not 0 <= gt[key] <= 1
                                                                             for key in ("whole_progress", "within_stage_progress")):
                raise ValueError("Invalid offline ground truth")
    return views


def run(args):
    import torch
    from grana.inference.predictor import GranaPredictor, InferenceRequest
    from grana.runtime import require_owned_environment

    require_owned_environment()
    if args.arm is None:
        raise ValueError("--run requires --arm")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.arm):
        raise ValueError("--arm must be a path-safe name")
    started = time.monotonic()
    manifest_hash = digest(args.manifest)
    manifest = read(args.manifest)
    views = validate_manifest(manifest)
    output_root = args.output_root or args.manifest.parent
    receipt_path = output_root / "receipts" / f"{args.arm}.json"
    event_path = output_root / "logs" / f"{args.arm}.jsonl"
    if receipt_path.exists() or event_path.exists():
        raise FileExistsError("Use a fresh output root; completed or partial evidence is never overwritten")
    arms = [arm for arm in manifest["arms"] if arm["id"] == args.arm]
    if not arms:
        raise ValueError("No matching model arms")
    files = [*sorted((ROOT / "grana").rglob("*.py")), Path(__file__).resolve()]
    code = {str(path.relative_to(ROOT)): digest(path) for path in files}
    receipt = {"schema_version": 1, "started_utc": now(), "pid": os.getpid(), "argv": sys.argv,
               "executable": sys.executable, "device": args.device, "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
               "manifest": str(args.manifest), "manifest_sha256": manifest_hash, "code_sha256": code,
               "runtime_policy": "cuda_optimized_v1", "inference_runtime": {},
               "model_loads": 0, "model_calls": 0, "results": [], "status": "running"}
    event_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with event_path.open("x") as events:
            for arm in arms:
                metadata_path = Path(arm["checkpoint"]) / "metadata.json"
                if digest(metadata_path) != arm["checkpoint_metadata_sha256"]:
                    raise ValueError("Checkpoint metadata changed after manifest freeze")
                predictor = GranaPredictor(arm["checkpoint"], device=args.device)
                receipt["model_loads"] += 1
                receipt["inference_runtime"][arm["id"]] = predictor.inference_runtime
                if predictor.view_names != tuple(views) or predictor.checkpoint["weights_sha256"] != arm["weights_sha256"]:
                    raise ValueError("Checkpoint input geometry or tensor identity mismatch")
                report = predictor.checkpoint["load_report"]
                if any(report["heads"][head]["outcome"] != "loaded" for head in ("progress_head", "temporal_head")):
                    raise ValueError("Complete checkpoint heads must load")
                for episode in manifest["episodes"]:
                    if not arm_includes_episode(arm, episode):
                        continue
                    output_path = output_root / "results" / arm["id"] / f"{episode['id']}.json"
                    if output_path.exists():
                        raise FileExistsError(output_path)
                    arrays = arrays_for(episode, views)
                    records = []
                    for index, window in enumerate(episode["windows"]):
                        tick = time.monotonic()
                        images = select_images(arrays, window["frame_indices"])
                        actual_input_hash = input_digest(episode, window, images)
                        if actual_input_hash != window["input_sha256"]:
                            raise ValueError("RGB/task inputs changed after manifest freeze")
                        # This constructor is the model boundary: no GT, cached
                        # stage IDs, timestamps, or oracle focus can enter it.
                        request = InferenceRequest(images=images, task=episode["task"],
                                                   compressed_task_prompt=episode["compressed_task_prompt"],
                                                   stage_prompts=episode["stage_prompts"])
                        output = predictor.predict(request, roles=("A", "C", "D"))
                        receipt["model_calls"] += 3
                        check_output(output, len(episode["stage_prompts"]))
                        focus = output["D_focus_stage"]
                        gt = window["gt"]
                        record = {"poi": window["poi"], "frame_indices": window["frame_indices"],
                                  "input_sha256": actual_input_hash, "A": output["progress"]["A"]["poi_score"],
                                  "C": output["stage"]["poi_stage_id"], "D": output["progress"]["D"]["poi_score"],
                                  "stage_probabilities": output["stage"]["probabilities"][5], "D_focus": focus,
                                  "gt": {"A": gt["whole_progress"], "stage_id": gt["stage_id"],
                                         "within_stage_progress": gt["within_stage_progress"],
                                         "D": relative_target(focus, gt["stage_id"], gt["within_stage_progress"])},
                                  "output": output, "elapsed_seconds": time.monotonic() - tick}
                        records.append(record)
                        events.write(json.dumps({"arm": arm["id"], "episode_id": episode["id"], "record": record}, allow_nan=False) + "\n")
                        events.flush()
                        if index == 0 or (index + 1) % 8 == 0:
                            print(json.dumps({"arm": arm["id"], "episode": episode["id"], "windows_done": index + 1,
                                              "windows_total": len(episode["windows"]), "model_calls": receipt["model_calls"]}), flush=True)
                    result = {"schema_version": 1, "arm": arm, "episode_id": episode["id"], "task": episode["task"],
                              "manifest_sha256": manifest_hash, "checkpoint": predictor.checkpoint,
                              "records": records, "metrics": metrics(records),
                              "provenance": {"code_sha256": code, "label_free_model_input": True, "GT_used_only_offline": True}}
                    write_new(output_path, result)
                    receipt["results"].append({"path": str(output_path), "sha256": digest(output_path), "arm": arm["id"],
                                               "episode_id": episode["id"], "metrics": result["metrics"]})
                    del arrays
                if digest(metadata_path) != arm["checkpoint_metadata_sha256"]:
                    raise ValueError("Checkpoint metadata changed during inference")
                del predictor
                gc.collect()
                torch.cuda.empty_cache()
        if digest(args.manifest) != manifest_hash or any(digest(ROOT / path) != value for path, value in code.items()):
            raise ValueError("Manifest or executing source changed during inference")
        receipt["status"] = "passed"
    except BaseException as exc:
        receipt["status"] = "failed"
        receipt["error"] = repr(exc)
        raise
    finally:
        receipt["finished_utc"] = now()
        receipt["elapsed_seconds"] = time.monotonic() - started
        write_new(receipt_path, receipt)
    print(json.dumps({"status": receipt["status"], "receipt": str(receipt_path), "model_calls": receipt["model_calls"]}), flush=True)


def main():

    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare", action="store_true")
    action.add_argument("--run", action="store_true")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--spec", type=Path, help="JSON selecting cohort run directories and named checkpoint arms")
    parser.add_argument("--episodes-per-task", type=int, default=2)
    parser.add_argument("--windows", type=int, default=32)
    parser.add_argument("--arm", help="Named checkpoint arm from the prepared manifest")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    args.manifest = args.manifest.expanduser().resolve()
    if args.episodes_per_task <= 0 or args.windows < 2:
        parser.error("Require positive episodes-per-task and at least two windows")
    prepare(args) if args.prepare else run(args)


if __name__ == "__main__":
    main()
