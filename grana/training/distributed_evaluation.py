# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Disjoint evaluation batches and record-based aggregation across training ranks."""
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from hashlib import sha256
import json
import math
from pathlib import Path
import time

import torch.distributed as dist

from grana.training.evaluation import (_empty_accumulator, _summarize, record_diagnostics,
                                      record_poi_slot, evaluation_stage_diagnostics)
from grana.training.losses import ROLES
from grana.training.rendering import _validated_records


def require(condition, message):
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class PlanShard:
    rank: int
    window_start: int
    window_stop: int
    chunk_start: int
    chunk_stop: int
    plan: object
    parent_plan_sha256: str

    def manifest(self):
        manifest = self.plan.manifest()
        return {"rank": self.rank, "window_start": self.window_start, "window_stop": self.window_stop,
                "chunk_start": self.chunk_start, "chunk_stop": self.chunk_stop,
                "window_count": self.window_stop - self.window_start,
                "chunk_count": self.chunk_stop - self.chunk_start,
                "parent_plan_sha256": self.parent_plan_sha256,
                "plan_sha256": manifest["plan_sha256"], "model_calls": manifest["model_calls"],
                "batch_size": manifest["batch_size"]}


def split_plan(plan, workers=2):
    """Divide contiguous native window batches; only the last shard can tail."""
    batch = plan.settings.get("batch_size", 1)
    require(type(batch) is int and batch > 0, "Invalid plan batch size")
    require(type(workers) is int and workers >= 1, "At least one worker is required")
    chunks = (len(plan.windows) + batch - 1) // batch
    require(chunks > 0, "Evaluation plan must contain windows")
    digest = plan.manifest()["plan_sha256"]
    result, chunk_start = [], 0
    for rank in range(workers):
        count = chunks // workers + (rank < chunks % workers)
        chunk_stop = chunk_start + count
        start, stop = min(chunk_start * batch, len(plan.windows)), min(chunk_stop * batch, len(plan.windows))
        shard = replace(plan, windows=tuple(plan.windows[start:stop]))
        result.append(PlanShard(rank, start, stop, chunk_start, chunk_stop, shard, digest))
        chunk_start = chunk_stop
    require(tuple(window for shard in result for window in shard.plan.windows) == plan.windows,
            "Shard windows differ from the full ordered plan")
    return tuple(result)


def expected_batches(plan):
    batch = plan.settings.get("batch_size", 1)
    windows, forwards = [], Counter()
    for offset in range(0, len(plan.windows), batch):
        chunk = plan.windows[offset:offset + batch]
        windows.append(len(chunk))
        for role in ROLES:
            count = sum(role in window.roles for window in chunk)
            if count:
                forwards[count] += 1
    return windows, {str(size): count for size, count in sorted(forwards.items())}


def verify_ordered_records(result, plan):
    manifest = plan.manifest()
    expected = [(window, role) for window in manifest["windows"] for role in window["roles"]]
    records = result["records"]
    require(len(records) == len(expected) == result["model_calls"], "Missing or extra role rows")
    identities = set()
    for record, (window, role) in zip(records, expected, strict=True):
        identity = (record["cache"], record["task"], record["trajectory_id"], record["poi"], record["role"])
        require(identity not in identities, "Duplicate role/window identity")
        identities.add(identity)
        require(record["role"] == role and all(record.get(key) == value for key, value in window.items()),
                "Records differ from exact ordered plan inputs")
    # Also verifies actual labels, image geometry, valid stages/progress and
    # native C(slot5) -> D focus for every window, without materializing RGB.
    _validated_records(result, plan)
    evidence = evaluation_stage_diagnostics(records, plan)
    if "stage_augmentation" in result:
        require(result["stage_augmentation"] == evidence,
                "Worker evaluation stage evidence differs from canonical records")


def _contribution(record, frame_weights):
    predictions, targets, losses = (record[key] for key in ("predictions", "targets", "slot_losses"))
    require(len(predictions) == len(targets) == len(losses) == 8, "Expected eight slots per role row")
    require(all(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
                for values in (predictions, targets, losses) for value in values), "Nonfinite row values")
    slot = record_poi_slot(record)
    order = record.get("frame_weight_order", list(range(8)))
    require(isinstance(order, (list, tuple)) and len(order) == 8
            and all(type(index) is int for index in order) and sorted(order) == list(range(8)),
            "Invalid record frame_weight_order")
    frame_weights = [frame_weights[index] for index in order]
    stage = record["role"] == "C"
    errors = [abs(predicted - target) for predicted, target in zip(predictions, targets, strict=True)]
    values = {"windows": 1, "dense_slots": 8,
              "weighted_loss_sum": sum(loss * weight for loss, weight in zip(losses, frame_weights, strict=True)),
              "dense_weight_mass": sum(frame_weights),
              "poi_squared_error_sum": 0.0 if stage else errors[slot] ** 2,
              "poi_correct": int(errors[slot] == 0) if stage else 0,
              "dense_correct": sum(error == 0 for error in errors) if stage else 0,
              "poi_absolute_stage_error_sum": int(errors[slot]) if stage else 0,
              "dense_absolute_stage_error_sum": int(sum(errors)) if stage else 0,
              "poi_far_stage_errors": int(errors[slot] >= 2) if stage else 0,
              "dense_far_stage_errors": sum(error >= 2 for error in errors) if stage else 0}
    for key, value in values.items():
        require(isinstance(record.get(key), (int, float)) and math.isfinite(record[key])
                and math.isclose(record[key], value, rel_tol=1e-7, abs_tol=1e-9),
                f"Saved contribution differs from raw slots: {key}")
    return values


def _summary_metrics(records, frame_weights, role_weights):
    aggregate, per_task = defaultdict(_empty_accumulator), {}
    for record in records:
        values = _contribution(record, frame_weights)
        role = record["role"]
        task = per_task.setdefault(record["task"], defaultdict(_empty_accumulator))
        for destination in (aggregate[role], task[role]):
            for key, value in values.items():
                destination[key] += value
    roles = _summarize(aggregate)
    tasks = {task: _summarize(totals) for task, totals in sorted(per_task.items())}
    quality_keys = {role: ("poi_accuracy", "dense_accuracy", "poi_mae", "dense_mae",
                           "poi_far_error_rate", "dense_far_error_rate") if role == "C" else ("poi_rmse",)
                    for role in ROLES}
    metrics = {}
    for role, totals in roles.items():
        if totals["status"] == "assessed":
            metrics[f"role_{role}_dense_weighted_loss"] = totals["dense_weighted_loss"]
            for key in quality_keys[role]:
                metrics[f"role_{role}_{key}"] = totals[key]
    for role in ROLES:
        for key in quality_keys[role]:
            values = []
            for task, totals in tasks.items():
                if totals[role]["status"] == "assessed":
                    metrics[f"task/{task}/role_{role}_{key}"] = totals[role][key]
                    values.append(totals[role][key])
            if values:
                metrics[f"role_{role}_task_macro_{key}"] = sum(values) / len(values)
    metrics["assessed_roles_weighted_loss"] = sum(totals["dense_weighted_loss"] * role_weights[role]
                                                for role, totals in roles.items() if totals["status"] == "assessed")
    diagnostics, quality = record_diagnostics(records)
    metrics.update(quality)
    return roles, tasks, diagnostics, metrics


def _merge_vision(results):
    if not any("vision_sdpa" in result for result in results):
        return None
    require(all("vision_sdpa" in result for result in results), "Missing worker vision stats")
    values = [result["vision_sdpa"] for result in results]
    descriptors = ("attention_modules", "vision_encoders", "evaluation_enabled", "training_enabled")
    require(all(all(value[key] == values[0][key] for key in descriptors) for value in values),
            "Worker vision configurations differ")
    merged = {key: values[0][key] for key in descriptors}
    merged.update(worker_count=len(values), module_count_semantics="identical per-worker module counts; execution counters summed")
    for phase in ("training", "evaluation"):
        counts = {key: sum(value[phase][key] for value in values) for key in ("calls", "hits", "images")}
        fallbacks = Counter()
        for value in values:
            fallbacks.update(value[phase]["fallbacks"])
        merged[phase] = {**counts, "fallbacks": dict(fallbacks)}
    merged["profile"] = {"enabled": all(value["profile"]["enabled"] for value in values),
                         **{key: sum(value["profile"][key] for value in values) for key in
                            ("vision_calls", "host_seconds", "cuda_seconds", "pending_cuda_spans")}}
    require(merged["profile"]["pending_cuda_spans"] == 0, "Worker vision spans were not synchronized")
    return merged


def merge_results(plan, shard_results, *, elapsed_seconds, frame_weights, role_weights, loss_config):
    """Recompute every quality metric from all raw rows; elapsed is shared wall time."""
    require(isinstance(elapsed_seconds, (int, float)) and not isinstance(elapsed_seconds, bool)
            and math.isfinite(elapsed_seconds) and elapsed_seconds > 0, "Invalid concurrent elapsed time")
    require(len(frame_weights) == 8 and all(math.isfinite(value) and value > 0 for value in frame_weights),
            "Expected eight finite positive frame weights")
    shards = split_plan(plan, workers=len(shard_results))
    require(set(shard_results) == {shard.rank for shard in shards}, "Missing or duplicate worker ranks")
    expected_loss = {"stage": {"cdf_weight": 1.0, **loss_config.get("stage", {})},
                     "progress": {"cdf_weight": 1.0, "ceiling_weight": 0.0, **loss_config.get("progress", {})},
                     "frame_weights": list(frame_weights), "role_weights": dict(role_weights)}
    records, summaries, results = [], [], []
    memory = {}
    forward_histogram, timings = Counter(), Counter()
    for shard in shards:
        result = shard_results[shard.rank]
        memory[str(shard.rank)] = result.get("cuda_memory") if result is not None else None
        if not shard.plan.windows:
            require(result is None, "Empty shard must not contain evaluated records")
            summaries.append({**shard.manifest(), "elapsed_seconds": 0.0,
                              "forward_calls": 0, "forward_batch_sizes": {}})
            continue
        require(result is not None, "Nonempty shard requires an evaluation result")
        require(result["plan_sha256"] == shard.plan.manifest()["plan_sha256"], "Worker shard plan hash differs")
        require(result["loss_config"] == expected_loss, "Worker loss configuration differs")
        require(result.get("image_preprocessing") == plan.settings["preprocessing"],
                "Worker image preprocessing differs from the fixed plan")
        require(result["batch_size"] == plan.settings.get("batch_size", 1), "Worker batch size differs")
        verify_ordered_records(result, shard.plan)
        windows, forwards = expected_batches(shard.plan)
        require(result["window_batch_sizes"] == windows and result["forward_batch_sizes"] == forwards
                and result["forward_calls"] == sum(forwards.values())
                and result["max_forward_batch_size"] == max(map(int, forwards)), "Worker batch boundaries/forwards differ")
        require(math.isfinite(result["elapsed_seconds"]) and 0 < result["elapsed_seconds"] <= elapsed_seconds + 1e-6,
                "Worker elapsed exceeds combined concurrent wall time")
        require(all(math.isfinite(value) and value >= 0 for value in result["timing_seconds"].values()),
                "Invalid worker timing")
        forward_histogram.update(result["forward_batch_sizes"])
        timings.update(result["timing_seconds"])
        records.extend(result["records"])
        summaries.append({**shard.manifest(), "elapsed_seconds": result["elapsed_seconds"],
                          "forward_calls": result["forward_calls"], "forward_batch_sizes": result["forward_batch_sizes"]})
        results.append(result)
    roles, tasks, diagnostics, metrics = _summary_metrics(records, frame_weights, role_weights)
    manifest = plan.manifest()
    window_batches, expected_forwards = expected_batches(plan)
    require(dict(forward_histogram) == expected_forwards, "Merged batches differ from native full-plan batches")
    result = {"schema": "grana-heldout-eval-v1", "plan_sha256": manifest["plan_sha256"], "score_slot": manifest["score_slot"],
              "elapsed_seconds": elapsed_seconds, "model_calls": len(records),
              "forward_calls": sum(forward_histogram.values()), "batch_size": manifest["batch_size"],
              "max_forward_batch_size": max(map(int, forward_histogram)),
              "forward_batch_sizes": {key: forward_histogram[key] for key in sorted(forward_histogram, key=int)},
              "window_batch_sizes": window_batches, "task_count": len(tasks), "metrics": metrics,
              "loss_config": expected_loss, "roles": roles, "per_task": tasks, "diagnostics": diagnostics,
              "stage_augmentation": evaluation_stage_diagnostics(records, plan),
              "image_preprocessing": dict(plan.settings["preprocessing"]),
              "records": records, "timing_seconds": dict(timings), "cuda_memory": None,
              "cuda_memory_by_worker": memory,
              "timing_semantics": "elapsed_seconds is coordinator-measured concurrent evaluation wall time. timing_seconds sums worker spans, not wall time; CUDA/host spans overlap.",
              "windows_per_second": len(plan.windows) / elapsed_seconds,
              "role_rows_per_second": len(records) / elapsed_seconds,
              "execution_world_size": len(shards),
              "parallel_execution": {"world_size": len(shards), "batch_membership_preserved": True,
                                     "coordinator_wall_seconds": elapsed_seconds, "shards": summaries,
                                     "worker_elapsed_seconds_sum": sum(worker["elapsed_seconds"] for worker in results),
                                     "worker_elapsed_seconds_max": max(worker["elapsed_seconds"] for worker in results)}}
    verify_ordered_records(result, plan)
    if plan.settings.get("purpose") == "deploy_dense":
        from grana.training.deploy_dense import dense_metrics, dense_series
        metrics.update(dense_metrics(dense_series(result, plan)))
    vision = _merge_vision(results)
    if vision is not None:
        result["vision_sdpa"] = vision
    return result


def evaluate_distributed(plan, evaluate_fn, *, output_dir, frame_weights, role_weights, loss_config):
    """Every training rank scores disjoint batches using its existing raw model.

    Large CPU record payloads travel through the training job's shared output
    directory, avoiding NCCL object collectives that stage serialized records on
    GPU. Collectives exchange only plan digests, errors and file digests. Rank 0
    returns the merged result; other ranks return None after merge validation.
    A failure on any live rank is reported to all peers before they can train.
    """
    rank, world_size = dist.get_rank(), dist.get_world_size()
    preflight = {"rank": rank, "error": None, "plan_sha256": None}
    try:
        preflight["plan_sha256"] = plan.manifest()["plan_sha256"]
        shard = split_plan(plan, workers=world_size)[rank]
        directory = Path(output_dir)
    except Exception as error:
        preflight["error"] = f"{type(error).__name__}: {error}"
    headers = [None] * world_size
    dist.all_gather_object(headers, preflight)
    errors = [f"rank {header['rank']}: {header['error']}" for header in headers if header["error"]]
    if errors:
        raise RuntimeError("Distributed evaluation preflight failed: " + "; ".join(errors))
    if len({header["plan_sha256"] for header in headers}) != 1:
        raise RuntimeError("Training workers reconstructed different evaluation plans")
    started = time.perf_counter()
    dist.barrier()
    status = {"rank": rank, "error": None, "empty": not shard.plan.windows}
    try:
        if shard.plan.windows:
            result = evaluate_fn(shard.plan)
            directory.mkdir(parents=True, exist_ok=True)
            payload = json.dumps(result).encode()
            target = directory / f"rank-{rank:03d}.json"
            temporary = target.with_suffix(".json.tmp")
            temporary.write_bytes(payload)
            temporary.replace(target)
            status["sha256"] = sha256(payload).hexdigest()
    except Exception as error:
        status["error"] = f"{type(error).__name__}: {error}"
    statuses = [None] * world_size
    dist.all_gather_object(statuses, status)
    errors = [f"rank {s['rank']}: {s['error']}" for s in statuses if s["error"]]
    if errors:
        raise RuntimeError("Distributed evaluation failed: " + "; ".join(errors))
    result, merge_error = None, [None]
    if rank == 0:
        try:
            shards = {}
            for status in statuses:
                worker_rank = status["rank"]
                if status["empty"]:
                    shards[worker_rank] = None
                    continue
                payload = (directory / f"rank-{worker_rank:03d}.json").read_bytes()
                require(sha256(payload).hexdigest() == status["sha256"], "Worker result digest differs")
                shards[worker_rank] = json.loads(payload)
            result = merge_results(plan, shards, elapsed_seconds=time.perf_counter() - started,
                                   frame_weights=frame_weights, role_weights=role_weights,
                                   loss_config=loss_config)
            elapsed = time.perf_counter() - started
            result.update(elapsed_seconds=elapsed, windows_per_second=len(plan.windows) / elapsed,
                          role_rows_per_second=result["model_calls"] / elapsed)
            result["parallel_execution"].update(coordinator_wall_seconds=elapsed,
                                                transport="shared_output_directory",
                                                active_workers=sum(not s["empty"] for s in statuses))
            result["timing_semantics"] = (
                "elapsed_seconds is rank-zero wall time from the start barrier through concurrent "
                "scoring, shard persistence, collection and merge. Worker timing_seconds are summed "
                "work spans, not wall time; CUDA and host spans overlap.")
        except Exception as error:
            merge_error[0] = f"{type(error).__name__}: {error}"
    dist.broadcast_object_list(merge_error, src=0)
    if merge_error[0]:
        raise RuntimeError("Distributed evaluation merge failed: " + merge_error[0])
    return result
