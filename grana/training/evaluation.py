# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixed held-out windows, native stage routing, and explicit metric denominators."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha256
import json
import math
import time

import torch

from grana.data.cache import CachedTrajectories, Trajectory
from grana.modeling.inputs import collate_examples, preprocessing_spec
from grana.modeling.image_preprocessing import configure_image_preprocessing
from .losses import FRAME_WEIGHTS, ROLES, progress_loss, stage_loss
from grana.data.sampling import POI_SLOT, RoleSampler, frame_indices, materialize_example
from grana.data.frame_sampling import DEFAULT_EVAL_STRIDE_PAIRS


def record_poi_slot(record: dict) -> int:
    """An explicit per-record query position, with the historical slot-five default."""
    slot = record.get("poi_slot", POI_SLOT)
    if type(slot) is not int or not 0 <= slot < 8:
        raise ValueError("Record poi_slot must be an integer in 0..7")
    return slot


@dataclass(frozen=True)
class EvaluationWindow:
    trajectory: Trajectory
    poi: int
    stride_pair: tuple[int, int]
    roles: tuple[str, ...]
    group: str | None = None

    def manifest(self) -> dict:
        row = self.trajectory
        value = {"cache": str(row.cache_root), "task": row.task, "trajectory_id": row.trajectory_id,
                "num_stages": len(row.stage_prompts),
                "view_names": list(row.view_names),
                "poi": self.poi, "frame_indices": list(frame_indices(self.poi, row.num_frames, *self.stride_pair)),
                "stride_pair": list(self.stride_pair), "roles": list(self.roles)}
        if self.group is not None:
            value["group"] = self.group
        return value


@dataclass(frozen=True)
class EvaluationPlan:
    dataset: CachedTrajectories
    windows: tuple[EvaluationWindow, ...]
    tasks: tuple[dict, ...]
    settings: dict

    def manifest(self) -> dict:
        counts = {role: sum(role in window.roles for window in self.windows) for role in ROLES}
        value = {"schema": "grana-heldout-eval-v1", "settings": self.settings,
                 "tasks": list(self.tasks), "task_count": len(self.tasks),
                 "trajectory_count": sum(task["selected_trajectories"] for task in self.tasks),
                 "window_count": len(self.windows), "role_window_counts": counts,
                 # Historical model_calls counts role rows, independent of batching.
                 "model_calls": sum(counts.values()), "batch_size": self.settings.get("batch_size", 1),
                 "score_slot": POI_SLOT, "dense_loss_slots_per_role_window": 8,
                 "windows": [window.manifest() for window in self.windows]}
        value["plan_sha256"] = sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        return value


def build_evaluation_plan(dataset: CachedTrajectories, evaluation_config: dict, data_config: dict,
                          *, training_dataset: CachedTrajectories, roles=ROLES) -> EvaluationPlan:
    """Enumerate all work without reading RGB or touching the training sampler.

    Each held-out task participates. The cap is per task, and each selected
    trajectory contributes distinct interior quantile POIs (one-third and
    two-thirds for two windows); short trajectories use all physical frames.
    Cache eligibility may make a task A-only; missing role metrics
    remain explicitly unassessed instead of being filled with zero.
    """
    maximum = evaluation_config.get("max_trajectories_per_task", 2)
    requested = evaluation_config.get("windows_per_trajectory", 2)
    seed = evaluation_config.get("seed", 1729)
    batch_size = evaluation_config.get("batch_size", 1)
    for name, value in (("windows_per_trajectory", requested), ("max_trajectories_per_task", maximum),
                        ("batch_size", batch_size)):
        if value is None and name == "max_trajectories_per_task":
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"evaluation.{name} must be a positive integer" + (" or null" if name.startswith("max_") else ""))
    if isinstance(roles, dict):
        roles = tuple(role for role in ROLES if roles.get(role, 0) > 0)
    else:
        roles = tuple(roles)
    if not roles or len(set(roles)) != len(roles) or any(role not in ROLES for role in roles):
        raise ValueError("Evaluation roles must be a nonempty subset of A/B/C/D")
    if "D" in roles and "C" not in roles:
        raise ValueError("Predicted-focus D evaluation requires native C")
    # Training's stride distribution and frame sampling never select evaluation inputs.
    sampler = RoleSampler(dataset, {"stride_pairs": evaluation_config.get("stride_pairs", DEFAULT_EVAL_STRIDE_PAIRS)},
                          seed=seed, execution_mode="eval")
    train_ids = {(row.task, row.trajectory_id) for row in training_dataset.records}
    train_payloads = {path.resolve() for row in training_dataset.records for path in row.frame_paths}
    grouped, identities = defaultdict(list), set()
    for row in dataset.records:
        identity = (row.task, row.trajectory_id)
        if identity in train_ids or any(path.resolve() in train_payloads for path in row.frame_paths):
            raise ValueError(f"Held-out trajectory overlaps training identity or RGB payload: {identity}")
        if identity in identities:
            raise ValueError(f"Duplicate held-out trajectory identity: {identity}")
        identities.add(identity)
        eligible = tuple(role for role in ROLES if role in roles and ("C" if role == "D" else role) in row.roles)
        if not eligible:
            raise ValueError(f"Held-out trajectory has no requested evaluation roles: {identity}")
        grouped[row.task].append((row, eligible))
    if not grouped:
        raise ValueError("Held-out evaluation inventory must be nonempty")
    windows, task_reports = [], []
    for task, candidates in sorted(grouped.items()):
        candidates.sort(key=lambda item: sha256(f"{seed}:{task}:{item[0].cache_root}:{item[0].trajectory_id}".encode()).digest())
        selected = candidates if maximum is None else candidates[:maximum]
        task_roles = set().union(*(set(eligible) for _, eligible in candidates))
        selected_roles = set().union(*(set(eligible) for _, eligible in selected))
        if task_roles != selected_roles:
            raise ValueError(f"Trajectory cap omits eligible roles for {task!r}; raise max_trajectories_per_task")
        for row, eligible in selected:
            count = min(requested, row.num_frames)
            pois = list(range(row.num_frames)) if count == row.num_frames else [(i + 1) * row.num_frames // (count + 1) for i in range(count)]
            for poi in pois:
                key = f"{seed}:{task}:{row.cache_root}:{row.trajectory_id}:{poi}"
                pair = sampler.stride_pairs[int.from_bytes(sha256(key.encode()).digest()[:8], "big") % len(sampler.stride_pairs)]
                windows.append(EvaluationWindow(row, poi, pair, eligible))
        task_reports.append({"task": task, "available_trajectories": len(candidates),
                             "selected_trajectories": len(selected), "roles": sorted(task_roles),
                             "cache_paths": sorted({str(row.cache_root) for row, _ in candidates}),
                             "selected_ids": [row.trajectory_id for row, _ in selected]})
    settings = {"max_trajectories_per_task": maximum, "windows_per_trajectory": requested,
                "seed": seed, "roles": list(roles), "views": list(data_config["views"]),
                "view_sets": [list(names) for names in sorted({window.trajectory.view_names for window in windows})],
                "views_policy": "data.views is the default; each window records its effective ordered view_names",
                "max_pixels": data_config["max_pixels"],
                "preprocessing": preprocessing_spec(data_config["max_pixels"],
                                                     gpu_image_resize=data_config.get("gpu_image_resize", False)),
                "stride_pairs": [list(pair) for pair in sampler.stride_pairs],
                "selection": "sha256-ranked trajectories per task; distinct interior quantile POIs; all frames when shorter than requested",
                "heldout_check": "all eval task/id identities and resolved RGB payload paths disjoint from training",
                "training_trajectories_checked": len(training_dataset.records)}
    # Historical resolved configs acquire the new default during load_config.
    # Omitting the no-op default here preserves their exact v1 manifest/hash,
    # including dense/render plans whose identity includes this plan's hash.
    if data_config.get("gpu_image_resize", False):
        settings["gpu_image_resize"] = True
    if batch_size != 1:
        settings["batch_size"] = batch_size
    return EvaluationPlan(dataset, tuple(windows), tuple(task_reports), settings)


def _empty_accumulator() -> dict:
    return {"windows": 0, "dense_slots": 0, "weighted_loss_sum": 0.0, "dense_weight_mass": 0.0,
            "poi_squared_error_sum": 0.0, "poi_correct": 0, "dense_correct": 0,
            "poi_absolute_stage_error_sum": 0, "dense_absolute_stage_error_sum": 0,
            "poi_far_stage_errors": 0, "dense_far_stage_errors": 0}


def _summarize(accumulators: dict) -> dict:
    summaries = {}
    for role in ROLES:
        totals = accumulators.get(role, _empty_accumulator())
        result = {**totals, "status": "assessed" if totals["windows"] else "unassessed"}
        if totals["windows"]:
            result["dense_weighted_loss"] = totals["weighted_loss_sum"] / totals["dense_weight_mass"]
            if role == "C":
                result["poi_accuracy"] = totals["poi_correct"] / totals["windows"]
                result["dense_accuracy"] = totals["dense_correct"] / totals["dense_slots"]
                result["poi_mae"] = totals["poi_absolute_stage_error_sum"] / totals["windows"]
                result["dense_mae"] = totals["dense_absolute_stage_error_sum"] / totals["dense_slots"]
                result["poi_far_error_rate"] = totals["poi_far_stage_errors"] / totals["windows"]
                result["dense_far_error_rate"] = totals["dense_far_stage_errors"] / totals["dense_slots"]
                result["stage_error_units"] = "stage_index_distance; far_error means absolute error >= 2"
            else:
                result["poi_rmse"] = math.sqrt(totals["poi_squared_error_sum"] / totals["windows"])
                result["progress_domain"] = "whole_task_[0,1]" if role == "A" else "focus_relative_[0,1]"
        summaries[role] = result
    return summaries


def record_diagnostics(records: list[dict]) -> tuple[dict, dict]:
    """Summarize measured POIs, retaining zero-support strata as unassessed.

    D targets are relative to its predicted focus, so its native progress
    quality must be split by whether C picked the true stage. Accumulated
    task progress is scored separately by the dense deployment evaluator.
    """
    slots = {record_poi_slot(row) for row in records}
    slot = next(iter(slots)) if len(slots) == 1 else POI_SLOT if not slots else None

    def progress(rows):
        support = len(rows)
        if not support:
            return {"status": "unassessed", "support": 0, "rmse": None, "mae": None,
                    "accuracy_within_0.1": None, "constant_0.5_rmse": None,
                    "constant_0.5_mae": None, "constant_0.5_accuracy_within_0.1": None}
        errors = [abs(float(row["predictions"][record_poi_slot(row)]) - float(row["targets"][record_poi_slot(row)])) for row in rows]
        baseline = [abs(.5 - float(row["targets"][record_poi_slot(row)])) for row in rows]
        return {"status": "assessed", "support": support,
                "rmse": math.sqrt(sum(error ** 2 for error in errors) / support),
                "mae": sum(errors) / support,
                "accuracy_within_0.1": sum(error <= .1 + 1e-8 for error in errors) / support,
                "constant_0.5_rmse": math.sqrt(sum(error ** 2 for error in baseline) / support),
                "constant_0.5_mae": sum(baseline) / support,
                "constant_0.5_accuracy_within_0.1": sum(error <= .1 + 1e-8 for error in baseline) / support}

    def identity(row):
        return row["cache"], row["task"], row["trajectory_id"], row["poi"]

    def summarize(rows):
        c_rows = [row for row in rows if row["role"] == "C"]
        stage_count = max((row["num_stages"] for row in c_rows), default=0)
        confusion = [[0] * stage_count for _ in range(stage_count)]
        correct_c = {}
        for row in c_rows:
            target, prediction = int(row["targets"][record_poi_slot(row)]), int(row["predictions"][record_poi_slot(row)])
            confusion[target][prediction] += 1
            correct_c[identity(row)] = prediction == target
        support = [sum(row) for row in confusion]
        accuracy = [confusion[index][index] / count if count else None for index, count in enumerate(support)]
        observed_accuracy = [value for value in accuracy if value is not None]
        distances = [abs(int(row["predictions"][record_poi_slot(row)]) - int(row["targets"][record_poi_slot(row)])) for row in c_rows]
        result = {"C": {"status": "assessed" if c_rows else "unassessed", "support": len(c_rows),
                        "poi_accuracy": sum(correct_c.values()) / len(c_rows) if c_rows else None,
                        "stage_macro_accuracy": sum(observed_accuracy) / len(observed_accuracy) if observed_accuracy else None,
                        "stage_macro_supported_classes": len(observed_accuracy),
                        "stage_support": support, "stage_accuracy": accuracy,
                        "confusion_matrix": confusion, "confusion_axes": "rows=GT, columns=predicted; zero-based stage indices",
                        "stage_mae": sum(distances) / len(distances) if distances else None,
                        "abs_error_ge_2_rate": sum(error >= 2 for error in distances) / len(distances) if distances else None}}
        for role in ("A", "B", "D"):
            role_rows = [row for row in rows if row["role"] == role]
            value = progress(role_rows)
            value["gt_lt_0.05"] = progress([row for row in role_rows if row["targets"][record_poi_slot(row)] < .05])
            if role == "D":
                if any(identity(row) not in correct_c for row in role_rows):
                    raise ValueError("D progress diagnostics require the matching native C record")
                for label, correct in (("correct_C", True), ("incorrect_C", False)):
                    selected = [row for row in role_rows if correct_c[identity(row)] == correct]
                    value[label] = progress(selected)
                    value[label]["gt_lt_0.05"] = progress([row for row in selected if row["targets"][record_poi_slot(row)] < .05])
            result[role] = value
        return result

    per_group, per_task = defaultdict(list), defaultdict(list)
    for row in records:
        per_task[row["task"]].append(row)
        if "group" in row:
            per_group[row["group"]].append(row)
    diagnostics = {"score_slot": slot, "progress_threshold": .1, "low_progress_threshold": .05,
                   "stage_macro_policy": "arithmetic mean over GT stages with nonzero support; all stage supports retained",
                   "overall": summarize(records),
                   "per_task": {task: summarize(rows) for task, rows in sorted(per_task.items())},
                   "per_group": {group: summarize(rows) for group, rows in sorted(per_group.items())}}
    metrics = {}

    def flatten(prefix, value):
        for key, item in value.items():
            name = f"{prefix}/{key}" if prefix else key
            if isinstance(item, dict):
                flatten(name, item)
            elif isinstance(item, (int, float)) and not isinstance(item, bool):
                metrics[name] = item
            elif isinstance(item, list) and key in ("stage_support", "stage_accuracy"):
                for index, number in enumerate(item):
                    if number is not None:
                        metrics[f"{prefix}/stage_{index}/{key.removeprefix('stage_')}"] = number

    flatten("diagnostics/overall", diagnostics["overall"])
    for scope in ("per_task", "per_group"):
        for name, value in diagnostics[scope].items():
            flatten(f"diagnostics/{scope}/{name}", value)
    return diagnostics, metrics


def evaluate(model, processor, plan: EvaluationPlan, *, device, max_pixels: int | None,
             frame_weights=FRAME_WEIGHTS, role_weights=None, loss_config=None,
             collate_fn=None, progress_callback=None) -> dict:
    """Run on one rank's unwrapped model; the caller owns sharding/persistence.

    C receives no focus/GT stage. Its detached slot-5 argmax supplies D's focus
    on the exact same images. Conditional-head RMSE/accuracy score only slot 5; objective
    diagnostics retain all eight slots with the training configuration's weights.
    Stage MAE and far-error rates use unweighted stage-index distances, with
    both POI and all-slot summaries; far errors are distances of at least two.

    Each forward batches one role over up to evaluation.batch_size windows.
    Records retain their window/role order and model_calls retains its historical
    role-row count; forward_calls reports actual batched model invocations.
    Per-record elapsed_seconds amortizes its batch's preparation/scoring time.
    """
    gpu_image_resize = plan.settings.get("gpu_image_resize", False)
    image_preprocessing = preprocessing_spec(max_pixels, gpu_image_resize=gpu_image_resize)
    if image_preprocessing != plan.settings["preprocessing"]:
        raise ValueError("Evaluation max_pixels and GPU resize must match the fixed plan's preprocessing contract")
    batch_size = plan.settings.get("batch_size", 1)
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("evaluation.batch_size must be a positive integer")
    if collate_fn is None:
        collate_fn = collate_examples
    device = torch.device(device)
    processor = configure_image_preprocessing(processor, device, resize_on_device=gpu_image_resize)
    weights = torch.tensor(frame_weights, device=device, dtype=torch.float64)
    if weights.shape != (8,) or not weights.isfinite().all() or (weights <= 0).any():
        raise ValueError("Evaluation needs eight finite positive loss frame weights")
    weight_mass = float(weights.sum())
    role_weights = {role: 1.0 for role in ROLES} if role_weights is None else role_weights
    loss_config = {} if loss_config is None else loss_config
    stage_options = {"cdf_weight": 1.0, **loss_config.get("stage", {})}
    progress_options = {"cdf_weight": 1.0, "ceiling_weight": 0.0,
                        **loss_config.get("progress", {})}
    sampler = RoleSampler(plan.dataset, {"stride_pairs": plan.settings["stride_pairs"]}, seed=plan.settings["seed"],
                          execution_mode="eval")
    aggregate, per_task, records = defaultdict(_empty_accumulator), {}, []
    manifest = plan.manifest()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    previous_mode = model.training
    forward_calls, max_forward_batch_size = 0, 0
    forward_batches, window_batches = Counter(), []
    timings = {name: 0.0 for name in ("prepare_examples_host", "collate_host", "forward_host",
                                     "score_and_transfer_host", "aggregate_host", "summarize_host")}
    forward_cuda_seconds = 0.0

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def score(examples, windows, preparation_started):
        nonlocal forward_calls, max_forward_batch_size, forward_cuda_seconds
        role = examples[0].role
        phase_started = time.perf_counter()
        timings["prepare_examples_host"] += phase_started - preparation_started
        inputs = collate_fn(processor, examples, max_pixels, device)
        timings["collate_host"] += time.perf_counter() - phase_started
        phase_started = time.perf_counter()
        if device.type == "cuda":
            cuda_start, cuda_end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            cuda_start.record(torch.cuda.current_stream(device))
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            output = model(inputs, [example.role for example in examples],
                           [example.num_stages for example in examples])
        if device.type == "cuda":
            cuda_end.record(torch.cuda.current_stream(device))
        timings["forward_host"] += time.perf_counter() - phase_started
        phase_started = time.perf_counter()
        forward_calls += 1
        forward_batches[len(examples)] += 1
        max_forward_batch_size = max(max_forward_batch_size, len(examples))
        if role == "C":
            logits = output["stage_logits"]
            targets = torch.tensor([example.targets for example in examples], device=device, dtype=torch.long)
            losses = stage_loss(logits, targets, **stage_options)
            predictions = logits.argmax(-1)
        else:
            logits = output["progress_logits"]
            targets = torch.tensor([example.targets for example in examples], device=device, dtype=torch.float32)
            losses = progress_loss(logits, targets, **progress_options)
            centers = torch.arange(10, device=device, dtype=torch.float32) / 9
            predictions = (logits.float().softmax(-1) * centers).sum(-1)
        if not losses.isfinite().all() or not predictions.isfinite().all():
            raise FloatingPointError("Non-finite held-out prediction or objective")
        # Transfer the small batch outputs once; no per-row CUDA scalar reads.
        weighted_losses = (losses.double() * weights).sum(-1).detach().cpu().tolist()
        slot_losses = losses.detach().cpu().tolist()
        predictions, targets = predictions.detach().cpu(), targets.cpu()
        if device.type == "cuda":
            # The small blocking D2H copies above completed this forward.
            forward_cuda_seconds += cuda_start.elapsed_time(cuda_end) / 1000
        batch_records = []
        for example, window, prediction, target, weighted_loss, row_losses in zip(
                examples, windows, predictions, targets, weighted_losses, slot_losses, strict=True):
            geometry = window.manifest()
            poi_slot = getattr(example, "poi_slot", POI_SLOT)
            if (poi_slot != record_poi_slot(geometry) or list(example.frame_indices) != geometry["frame_indices"]
                    or tuple(getattr(example, "frame_weight_order", range(8))) != tuple(range(8))):
                raise ValueError("Held-out materialization differs from the frozen evaluation geometry")
            correct = (prediction == target) if role == "C" else None
            stage_error = (prediction - target).abs() if correct is not None else None
            contribution = {"windows": 1, "dense_slots": 8,
                            "weighted_loss_sum": weighted_loss, "dense_weight_mass": weight_mass,
                            "poi_squared_error_sum": 0.0 if correct is not None else float((prediction[poi_slot].double() - example.targets[poi_slot]).square()),
                            "poi_correct": int(correct[poi_slot]) if correct is not None else 0,
                            "dense_correct": int(correct.sum()) if correct is not None else 0,
                            "poi_absolute_stage_error_sum": int(stage_error[poi_slot]) if stage_error is not None else 0,
                            "dense_absolute_stage_error_sum": int(stage_error.sum()) if stage_error is not None else 0,
                            "poi_far_stage_errors": int(stage_error[poi_slot] >= 2) if stage_error is not None else 0,
                            "dense_far_stage_errors": int((stage_error >= 2).sum()) if stage_error is not None else 0}
            batch_records.append({**window.manifest(), "role": role, "focus_stage": example.focus_stage,
                                  "true_within_stage_progress": (window.trajectory.within_stage_progress[window.poi]
                                                                 if window.trajectory.within_stage_progress else None),
                                  "targets": list(example.targets), "predictions": prediction.tolist(),
                                  "slot_losses": row_losses,
                                  **contribution})
        elapsed_per_row = (time.perf_counter() - preparation_started) / len(examples)
        for record in batch_records:
            record["elapsed_seconds"] = elapsed_per_row
        timings["score_and_transfer_host"] += time.perf_counter() - phase_started
        return batch_records

    try:
        model.eval()
        # Preserve training's torch RNG even if a future deterministic processor
        # or model implementation performs incidental random operations.
        devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices), torch.inference_mode():
            for offset in range(0, len(plan.windows), batch_size):
                windows = plan.windows[offset:offset + batch_size]
                window_batches.append(len(windows))
                scored = {}
                for role in ROLES:
                    selected = [(i, window) for i, window in enumerate(windows) if role in window.roles]
                    if role == "D" or not selected:
                        continue
                    prepared = time.perf_counter()
                    examples = [materialize_example(window.trajectory, role, window.poi, window.stride_pair,
                                                    execution_mode="eval")
                                for _, window in selected]
                    batch_records = score(examples, [window for _, window in selected], prepared)
                    for (i, _), record in zip(selected, batch_records, strict=True):
                        scored[i, role] = record
                    paired = [(i, window, example, record) for (i, window), example, record
                              in zip(selected, examples, batch_records, strict=True)
                              if role == "C" and "D" in window.roles]
                    if paired:
                        prepared = time.perf_counter()
                        derived = [sampler.make_d(example, int(record["predictions"][record_poi_slot(record)]))
                                   for _, _, example, record in paired]
                        d_records = score(derived, [window for _, window, _, _ in paired], prepared)
                        for (i, _, _, _), record in zip(paired, d_records, strict=True):
                            scored[i, "D"] = record
                        del derived
                    del examples, paired
                # Accumulate in the original order, preserving metric denominators
                # and serialized window/role ordering even across partial batches.
                phase_started = time.perf_counter()
                for i, window in enumerate(windows):
                    for role in window.roles:
                        record = scored[i, role]
                        task_totals = per_task.setdefault(record["task"], defaultdict(_empty_accumulator))
                        for destination in (aggregate[role], task_totals[role]):
                            for key in destination:
                                destination[key] += record[key]
                        records.append(record)
                        if progress_callback is not None:
                            progress_callback(len(records), manifest["model_calls"])
                timings["aggregate_host"] += time.perf_counter() - phase_started
    finally:
        model.train(previous_mode)
    synchronize()
    phase_started = time.perf_counter()
    summaries = _summarize(aggregate)
    if len(records) != manifest["model_calls"]:
        raise RuntimeError("Evaluation model-call coverage differs from the fixed plan")
    metrics = {}
    quality_keys = {role: ("poi_accuracy", "dense_accuracy", "poi_mae", "dense_mae",
                          "poi_far_error_rate", "dense_far_error_rate") if role == "C" else ("poi_rmse",)
                    for role in ROLES}
    for role, totals in summaries.items():
        if totals["status"] == "unassessed":
            continue
        metrics[f"role_{role}_dense_weighted_loss"] = totals["dense_weighted_loss"]
        for key in quality_keys[role]:
            metrics[f"role_{role}_{key}"] = totals[key]
    task_summaries = {task: _summarize(totals) for task, totals in sorted(per_task.items())}
    for role in ROLES:
        for key in quality_keys[role]:
            values = []
            for task, task_roles in task_summaries.items():
                if task_roles[role]["status"] == "assessed":
                    value = task_roles[role][key]
                    metrics[f"task/{task}/role_{role}_{key}"] = value
                    values.append(value)
            if values:
                metrics[f"role_{role}_task_macro_{key}"] = sum(values) / len(values)
    metrics["assessed_roles_weighted_loss"] = sum(totals["dense_weighted_loss"] * role_weights[role]
                                                   for role, totals in summaries.items() if totals["status"] == "assessed")
    diagnostics, diagnostic_metrics = record_diagnostics(records)
    metrics.update(diagnostic_metrics)
    result = {"schema": "grana-heldout-eval-v1", "plan_sha256": manifest["plan_sha256"],
            "score_slot": manifest["score_slot"], "elapsed_seconds": time.perf_counter() - started,
            "model_calls": len(records), "forward_calls": forward_calls,
            "batch_size": batch_size, "max_forward_batch_size": max_forward_batch_size,
            "forward_batch_sizes": {str(size): count for size, count in sorted(forward_batches.items())},
            "window_batch_sizes": window_batches,
            "task_count": len(per_task), "metrics": metrics,
            "image_preprocessing": image_preprocessing,
            "loss_config": {"stage": stage_options, "progress": progress_options,
                            "frame_weights": list(frame_weights), "role_weights": dict(role_weights)},
            "roles": summaries, "per_task": task_summaries,
            "diagnostics": diagnostics, "records": records}
    if plan.settings.get("purpose") == "deploy_dense":
        # These are the very same records later used for PNG/MP4 selection;
        # no extra model pass is needed for headline accumulated quality.
        from .deploy_dense import dense_metrics, dense_series
        metrics.update(dense_metrics(dense_series(result, plan)))
    timings["summarize_host"] = time.perf_counter() - phase_started
    if device.type == "cuda":
        timings["forward_cuda"] = forward_cuda_seconds
        free, total = torch.cuda.mem_get_info(device)
        result["cuda_memory"] = {"peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                                 "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                                 "allocated_after_bytes": torch.cuda.memory_allocated(device),
                                 "reserved_after_bytes": torch.cuda.memory_reserved(device),
                                 "device_free_after_bytes": free, "device_total_bytes": total}
    else:
        result["cuda_memory"] = None
    result["timing_seconds"] = timings
    result["timing_semantics"] = "Host spans include waits; CUDA forward spans may overlap host work and must not be added to them."
    result["elapsed_seconds"] = time.perf_counter() - started
    result["windows_per_second"] = len(plan.windows) / result["elapsed_seconds"]
    result["role_rows_per_second"] = len(records) / result["elapsed_seconds"]
    return result
