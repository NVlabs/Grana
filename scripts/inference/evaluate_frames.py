#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluate a prepared paired-frame plan on one GPU with physical batch size 1.

The default device is parent logical CUDA index 0. Select --devices auto to
balance whole trajectories over all visible GPUs, or specify logical indices.
Each independent worker sees its assigned GPU alone as cuda:0. Optional
--batch-size auto probes memory capacity, starting at 32 when the workload and
user cap allow it. Batch sizes and limits are per GPU.
These are local throughput evaluations, not HTTP latency benchmarks.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from grana.inference.client import build_binary
from grana.inference.resources import autotune_batch_size, balanced_shards, is_cuda_oom, select_devices
from grana.inference.window import compose_frame_indices
from scripts.inference.evaluate_http_frames import digest, file_digest
from scripts.inference.replay_http import validate_reply, write_json
from scripts.inference.summarize_http_frames import strict_json, validate_metadata, validate_plan


def identity(job):
    return job["seed"], job["key"], job["kind"], job["source_frame"], job["mode"]


def jobs_for(plan, keys=None):
    selected = None if keys is None else set(keys)
    return [{"seed": seed, "key": row["key"], "kind": kind, "source_frame": frame, "mode": mode}
            for row in plan["trajectories"] if selected is None or row["key"] in selected
            for seed in plan["seeds"]
            for kind, frames in (("anchor", row["anchors"]), ("quality", row["targets"][str(seed)]))
            for frame in frames for mode in ("poi", "last")]


def representative_jobs(jobs, limit=32):
    """Spread distinct physical targets across the shard, retaining both modes."""
    targets = {}
    for job in jobs:
        targets.setdefault((job["key"], job["source_frame"]), {}).setdefault(job["mode"], job)
    ordered = sorted(targets)
    count = min(len(ordered), max(1, limit // 2))
    selected = [ordered[i * (len(ordered) - 1) // max(1, count - 1)] for i in range(count)]
    return [targets[key][mode] for key in selected for mode in ("poi", "last") if mode in targets[key]][:limit]


def batch_size_value(value):
    if value == "auto":
        return value
    try:
        size = int(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError("batch-size must be auto or a positive integer") from None
    if size < 1:
        raise argparse.ArgumentTypeError("batch-size must be auto or a positive integer")
    return size


def smaller_auto_batch(size, max_batch_size=None):
    candidate = max(1, size // 2)
    if size > 32 and (max_batch_size is None or max_batch_size >= 32):
        candidate = max(32, candidate)
    return candidate if max_batch_size is None else min(candidate, max_batch_size)


def cuda_uuid(properties):
    value = getattr(properties, "uuid", None)
    # PyTorch's _CUuuid exposes a bare UUID through str() on physical H100s.
    # Explicit MIG-prefixed selectors remain distinct; a bare CUDA UUID alone
    # does not provide enough information to infer a MIG instance selector.
    match = None if value is None else re.fullmatch(
        r"(?:(GPU|MIG)-)?([0-9a-fA-F]{8}-(?:[0-9a-fA-F]{4}-){3}[0-9a-fA-F]{12})", str(value))
    if match is None:
        raise RuntimeError("CUDA properties must expose a complete GPU/MIG UUID for worker isolation")
    return f"{match.group(1) or 'GPU'}-{match.group(2).lower()}"


def isolated_device_mappings(devices, get_properties):
    """Resolve parent logical devices through CUDA, never as physical ordinals."""
    mappings, used = {}, set()
    parent_visibility = os.environ.get("CUDA_VISIBLE_DEVICES")
    for device in devices:
        uuid = cuda_uuid(get_properties(device))
        if uuid in used:
            raise ValueError("Selected parent CUDA devices resolved to a duplicate GPU UUID")
        used.add(uuid)
        mappings[str(device)] = {"parent_logical_device": device, "device_uuid": uuid,
            "parent_cuda_visible_devices": parent_visibility, "worker_cuda_visible_devices": uuid,
            "worker_local_device": 0}
    return mappings


def prepare_campaign(args, device_count, *, get_device_properties=None):
    source = args.plan.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Use a new offline evaluation directory: {output}")
    if not 0 < args.memory_fraction <= 1 or not math.isfinite(args.memory_fraction):
        raise ValueError("memory-fraction must be in (0,1]")
    if args.max_batch_size is not None and args.max_batch_size < 1:
        raise ValueError("max-batch-size must be positive")
    if args.batch_size != "auto" and args.max_batch_size is not None and args.batch_size > args.max_batch_size:
        raise ValueError("Explicit batch-size exceeds max-batch-size")
    plan = deepcopy(validate_plan(strict_json(source.read_bytes())))
    if args.seeds is not None:
        if not args.seeds or len(set(args.seeds)) != len(args.seeds) or set(args.seeds) - set(plan["seeds"]):
            raise ValueError("seeds must be distinct seeds from the prepared plan")
        plan["seeds"] = list(args.seeds)
        for row in plan["trajectories"]:
            row["targets"] = {str(seed): row["targets"][str(seed)] for seed in args.seeds}
    for row in plan["trajectories"]:
        for field in ("request_path", "metadata_path"):
            path = Path(row[field]).expanduser()
            row[field] = str(path.resolve() if path.is_absolute() else (source.parent / path).resolve())
    tasks = []
    for row in plan["trajectories"]:
        pixels = sum(math.prod(spec["shape"][1:3]) for spec in row.get("sources", {}).values()) or 1
        count = sum(2 * (len(row["targets"][str(seed)]) + len(row["anchors"])) for seed in plan["seeds"])
        tasks.append({"key": row["key"], "cost": count * pixels, "jobs": count})
    devices = select_devices(args.devices, device_count, len(tasks))
    if get_device_properties is None:
        import torch
        get_device_properties = torch.cuda.get_device_properties
    mappings = isolated_device_mappings(devices, get_device_properties)
    shards = balanced_shards(tasks, devices)
    batching = {"batch_size_scope": "per_gpu", "requested_batch_size_per_gpu": args.batch_size,
                "requested_aggregate_batch_capacity": len(devices) * args.batch_size if type(args.batch_size) is int else None,
                "workers_synchronized": False}
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1
    plan["execution"] = {"mode": "local_batched", "source_plan": str(source),
                         "source_plan_sha256": file_digest(source), "requested_devices": args.devices,
                         "devices": devices, "device_mappings": mappings, "requested_batch_size": args.batch_size,
                         "memory_fraction": args.memory_fraction, "max_batch_size": args.max_batch_size, **batching}
    output.mkdir(parents=True)
    write_json(output / "plan.json", plan)
    schedule = {"devices": devices, "cpu_threads_per_worker": max(1, min(8, cpus // len(devices))),
                "device_mappings": mappings,
                "shards": {str(device): members for device, members in shards.items()},
                "plan_sha256": file_digest(output / "plan.json"), **batching}
    write_json(output / "schedule.json", schedule)
    return plan, schedule, output


class WindowCache:
    """Keep read-only camera mmaps and label-free request views for one worker."""

    def __init__(self, plan):
        self.rows = {row["key"]: row for row in plan["trajectories"]}
        self.sources, self.requests, self.hashes, self.labels = {}, {}, {}, {}

    def request(self, job):
        import numpy as np
        from grana.inference.cli import load_request
        from grana.inference.predictor import InferenceRequest

        key, frame, mode = job["key"], job["source_frame"], job["mode"]
        cache_key = key, frame, mode
        if cache_key in self.requests:
            return self.requests[cache_key]
        row = self.rows[key]
        if key not in self.sources:
            path = Path(row["request_path"])
            if "request_sha256" in row and file_digest(path) != row["request_sha256"]:
                raise ValueError("Request source changed after plan creation")
            # Validate the public file interface once, then reuse its RGB mmaps.
            load_request(path, frame_index=0, prediction_frame="last")
            raw = strict_json(path.read_bytes())
            if set(raw) - {"views", "task", "stage_prompts", "compressed_task_prompt"}:
                raise ValueError("Offline plan requests must contain full videos and label-free prompts only")
            arrays = {}
            for view, filename in raw["views"].items():
                filename = Path(filename).expanduser()
                filename = filename if filename.is_absolute() else path.parent / filename
                source = row.get("sources", {}).get(view)
                if source:
                    stat = filename.stat()
                    if (stat.st_size, stat.st_mtime_ns) != (source["size_bytes"], source["mtime_ns"]):
                        raise ValueError("RGB source changed after plan creation")
                array = np.load(filename, mmap_mode="r", allow_pickle=False)
                if array.dtype != np.uint8 or array.ndim != 4 or array.shape[0] != row["num_frames"] or array.shape[-1] != 3:
                    raise ValueError("Expected frame-aligned RGB uint8 camera arrays")
                if source and list(array.shape) != source["shape"]:
                    raise ValueError("RGB shape differs from the prepared plan")
                arrays[view] = array
            self.sources[key] = raw, arrays
        raw, arrays = self.sources[key]
        indices = compose_frame_indices(frame, row["num_frames"], prediction_frame=mode)
        request = InferenceRequest(images={view: tuple(array[index] for index in indices) for view, array in arrays.items()},
                                   task=raw.get("task"), stage_prompts=raw.get("stage_prompts", ()),
                                   compressed_task_prompt=raw.get("compressed_task_prompt"),
                                   prediction_frame=mode, frame_indices=indices)
        if indices[request.poi_slot] != frame or (mode == "last" and max(indices) > frame):
            raise ValueError("Selected physical target or no-future contract changed")
        self.requests[cache_key] = request
        return request

    def input_digest(self, job, request):
        key = job["key"], job["source_frame"], job["mode"]
        if key not in self.hashes:
            self.hashes[key] = hashlib.sha256(build_binary(request, ("C", "D"), request_id="identity")).hexdigest()
        return self.hashes[key]

    def ground_truth(self, job):
        key, frame = job["key"], job["source_frame"]
        if key not in self.labels:
            row = self.rows[key]
            path = Path(row["metadata_path"])
            data = path.read_bytes()
            if "metadata_sha256" in row and hashlib.sha256(data).hexdigest() != row["metadata_sha256"]:
                raise ValueError("Ground-truth metadata changed after plan creation")
            self.labels[key] = strict_json(data)
            validate_metadata(row, self.labels[key])
        labels = self.labels[key]
        return {"stage_id": labels["stage_ids"][frame], "within_stage_progress": labels["within_stage_progress"][frame]}


    def revalidate_sources(self):
        """Reject sources modified while cached views or input hashes were live."""
        checks = {"trajectories": len(self.sources), "request_hashes": 0, "metadata_hashes": 0, "rgb_stats": 0}
        for key, (_, arrays) in self.sources.items():
            row = self.rows[key]
            for name in ("request", "metadata"):
                if f"{name}_sha256" in row:
                    if file_digest(Path(row[f"{name}_path"])) != row[f"{name}_sha256"]:
                        raise ValueError(f"{name} source changed while the offline worker was running")
                    checks[f"{name}_hashes"] += 1
            for view, array in arrays.items():
                source = row.get("sources", {}).get(view)
                if source:
                    stat = Path(array.filename).stat()
                    if (stat.st_size, stat.st_mtime_ns) != (source["size_bytes"], source["mtime_ns"]):
                        raise ValueError("RGB source changed while the offline worker was running")
                    checks["rgb_stats"] += 1
        return checks


def run_worker(args, *, predictor_factory=None, cuda=None):
    import torch
    from grana.inference.predictor import GranaPredictor

    cuda = torch.cuda if cuda is None else cuda
    predictor_factory = GranaPredictor if predictor_factory is None else predictor_factory
    output = args.output.expanduser().resolve()
    plan = validate_plan(strict_json((output / "plan.json").read_bytes()))
    schedule = strict_json((output / "schedule.json").read_bytes())
    device = args.worker_device
    mapping = schedule["device_mappings"][str(device)]
    if (mapping != plan["execution"]["device_mappings"].get(str(device))
            or mapping["parent_logical_device"] != device or mapping["worker_local_device"] != 0
            or mapping["worker_cuda_visible_devices"] != mapping["device_uuid"]
            or os.environ.get("CUDA_VISIBLE_DEVICES") != mapping["worker_cuda_visible_devices"]):
        raise ValueError("Worker CUDA visibility differs from the declared isolated device mapping")
    if cuda.device_count() != 1:
        raise ValueError("Each offline worker must see exactly one CUDA device")
    local_device = 0
    cuda.set_device(local_device)
    actual_uuid = cuda_uuid(cuda.get_device_properties(local_device))
    if actual_uuid != mapping["device_uuid"]:
        raise ValueError("Worker CUDA device UUID differs from the parent device mapping")
    keys = [task["key"] for task in schedule["shards"][str(device)]]
    jobs = jobs_for(plan, keys)
    folder = output / "workers" / f"gpu_{device}"
    folder.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    predictor = predictor_factory(plan["checkpoint"]["path"], model_name_or_path=args.model_name_or_path, device="cuda:0")
    weights = plan["checkpoint"]["weights_sha256"]
    if predictor.checkpoint["weights_sha256"] != weights:
        raise ValueError("Loaded checkpoint differs from the immutable plan")
    runtime = predictor.checkpoint.get("inference_runtime", {})
    if runtime.get("policy") != "cuda_optimized_v1" or runtime.get("gpu_image_resize") is not True:
        raise ValueError("Offline evaluation requires the canonical CUDA/GPU-resize runtime")
    cpu_threads = schedule["cpu_threads_per_worker"]
    torch.set_num_threads(cpu_threads)
    if runtime.get("cpu_threads") != cpu_threads:
        runtime.update(cpu_threads=cpu_threads, cpu_thread_budget_source="offline_worker_cpu_share")
        predictor.inference_runtime = deepcopy(runtime)
    cache = WindowCache(plan)

    def cleanup():
        gc.collect()
        cuda.empty_cache()

    def measure(batch):
        requests = [cache.request(job) for job in batch]
        cuda.synchronize(local_device)
        cuda.reset_peak_memory_stats(local_device)
        began = time.perf_counter()
        results = predictor.predict_batch(requests, roles=("C", "D"), batch_size=len(requests))
        cuda.synchronize(local_device)
        seconds = time.perf_counter() - began
        return requests, results, seconds, int(cuda.max_memory_allocated(local_device))

    cleanup()
    free, total = cuda.mem_get_info(local_device)
    baseline = int(cuda.memory_allocated(local_device))
    budget = int(baseline + args.memory_fraction * free)
    limit = min(len(jobs), args.max_batch_size or len(jobs))
    if args.batch_size == "auto":
        representatives = representative_jobs(jobs)

        def probe(size):
            print(json.dumps({"device": device, "phase": "batch_probe", "batch_size": size,
                              "status": "warming"}), flush=True)
            if size == 1:
                indices = [len(representatives) // 2]
            elif size <= len(representatives):
                indices = [i * (len(representatives) - 1) // (size - 1) for i in range(size)]
            else:
                indices = [i % len(representatives) for i in range(size)]
            batch = [representatives[index] for index in indices]
            elapsed, warmup_seconds, maximum = 0., 0., baseline
            # Reuse the same representative input shape after warming it so
            # compilation is excluded from the measured throughput.
            for iteration in range(3):
                requests, results, seconds, peak = measure(batch)
                if len(results) != len(batch):
                    raise ValueError("Probe result count differs from the requested windows")
                for job, request, result in zip(batch, requests, results, strict=True):
                    validate_reply({"request_id": "probe", "result": result}, request, ("C", "D"),
                                   job["mode"], "probe", weights)
                if iteration == 0:
                    warmup_seconds = seconds
                else:
                    elapsed += seconds
                maximum = max(maximum, peak)
            measurement = {"peak_memory_bytes": maximum, "elapsed_seconds": elapsed,
                "windows_measured": 2 * size, "timed_batches": 2,
                "warmup_batches": 1, "warmup_elapsed_seconds": warmup_seconds,
                "representative_modes": sorted({job["mode"] for job in batch})}
            print(json.dumps({"device": device, "phase": "batch_probe", "batch_size": size,
                              "status": "measured", **measurement}), flush=True)
            return measurement

        selected, sizing = autotune_batch_size(probe, limit, budget, cleanup=cleanup)
        sizing["probe_policy"] = {"warmup_batches_per_size": 1, "timed_batches_per_size": 2,
            "representative_windows": len(representatives),
            "representative_modes": dict(Counter(job["mode"] for job in representatives)),
            "objective": "memory_capacity", "preferred_start_batch_size": 32, "throughput_role": "report_only",
            "timing_note": "Warm throughput is reported per candidate; memory fit determines batch capacity. Peak memory includes warmup."}
    else:
        selected = min(args.batch_size, len(jobs))
        sizing = {"mode": "explicit", "selected_batch_size": selected, "requested_batch_size": args.batch_size,
                  "memory_budget_bytes": budget, "probes": []}
    write_json(folder / "sizing.json", sizing)
    initial = selected
    offset, batches, inference_seconds, maximum_peak = 0, 0, 0., baseline
    histogram, retries, budget_exceedances = Counter(), [], []
    with (folder / "predictions.jsonl").open("x") as stream:
        while offset < len(jobs):
            physical_size = min(selected, len(jobs) - offset)
            batch = jobs[offset:offset + physical_size]
            retry = False
            try:
                requests, results, seconds, peak = measure(batch)
            except Exception as error:
                if args.batch_size != "auto" or not is_cuda_oom(error) or len(batch) == 1:
                    raise
                selected = smaller_auto_batch(len(batch), args.max_batch_size)
                retries.append({"offset": offset, "failed_batch_size": len(batch), "retry_batch_size": selected,
                                "reason": "cuda_oom", "error_type": type(error).__name__})
                retry = True
            # Leave the exception scope first so its traceback no longer keeps
            # failed CUDA tensors alive during garbage collection/cache release.
            if retry:
                cleanup()
                continue
            if len(results) != len(batch):
                raise ValueError("Batch result count differs from the requested windows")
            maximum_peak = max(maximum_peak, peak)
            if args.batch_size == "auto" and peak > budget:
                if len(batch) == 1:
                    raise RuntimeError(f"Automatic physical batch size 1 exceeded the GPU memory budget: {peak} > {budget}")
                selected = smaller_auto_batch(len(batch), args.max_batch_size)
                retries.append({"offset": offset, "failed_batch_size": len(batch), "retry_batch_size": selected,
                                "reason": "memory_budget", "peak_memory_bytes": peak, "memory_budget_bytes": budget})
                cleanup()
                continue
            batch_id = f"gpu-{device}-batch-{batches:06d}"
            if peak > budget:
                budget_exceedances.append({"batch_id": batch_id, "batch_size": len(batch),
                                           "peak_memory_bytes": peak, "memory_budget_bytes": budget,
                                           "accepted_explicit_batch_size": True})
            records = []
            for job, request, result in zip(batch, requests, results, strict=True):
                request_id = f"offline-{job['seed']}-{job['key']}-{job['kind']}-{job['source_frame']}-{job['mode']}"
                validate_reply({"request_id": request_id, "result": result}, request, ("C", "D"), job["mode"], request_id, weights)
                predictions = {"C_ids": result["stage"]["ids"], "D_values": result["progress"]["D"]["values"],
                               "D_focus_stage": result["D_focus_stage"]}
                records.append({**job, "slot": request.poi_slot, "frame_indices": list(request.frame_indices),
                    "request_id": request_id, "input_sha256": cache.input_digest(job, request),
                    "prediction_sha256": digest(predictions), **predictions, "gt": cache.ground_truth(job),
                    "checkpoint_sha256": weights, "execution": "local_batched", "transport": "local",
                    "input_identity_only": True, "worker_device": device, "batch_id": batch_id, "batch_size": len(batch),
                    "worker_local_device": local_device, "worker_device_uuid": actual_uuid,
                    "inference_batch_seconds": seconds, "inference_amortized_ms": seconds * 1000 / len(batch)})
            stream.writelines(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n" for record in records)
            stream.flush()
            offset += len(batch)
            batches += 1
            histogram[len(batch)] += 1
            inference_seconds += seconds
            if batches == 1 or batches % 10 == 0 or offset == len(jobs):
                print(json.dumps({"device": device, "completed": offset, "total": len(jobs), "batch_size": selected,
                                  "elapsed_seconds": round(time.monotonic() - start, 2)}), flush=True)
    source_checks = cache.revalidate_sources()
    receipt = {"execution": "local_batched", "transport": "local", "worker_device": device,
               "batch_size_scope": "per_gpu", "requested_batch_size_per_gpu": args.batch_size,
               "device_mapping": {**mapping, "actual_worker_device_uuid": actual_uuid},
               "final_source_revalidation": source_checks,
               "checkpoint": predictor.checkpoint, "plan_sha256": file_digest(output / "plan.json"),
               "requests_written": offset, "trajectory_keys": keys, "cpu_threads": cpu_threads,
               "cuda_memory": {"total_bytes": int(total), "free_before_probes_bytes": int(free),
                               "baseline_allocated_bytes": baseline, "memory_budget_bytes": budget,
                               "maximum_evaluation_peak_bytes": maximum_peak},
               "sizing": sizing, "initial_batch_size": initial, "final_batch_size": selected,
               "batch_histogram": dict(histogram), "runtime_batch_retries": retries,
               "runtime_oom_retries": [retry for retry in retries if retry["reason"] == "cuda_oom"],
               "memory_budget_exceeded_batches": budget_exceedances,
               "inference_batch_seconds_total": inference_seconds, "worker_wall_seconds": time.monotonic() - start,
               "timing_note": "Local predict_batch plus CUDA synchronization; excludes input loading, identity hashing, validation and journaling. Not HTTP latency.",
               "journal_sha256": file_digest(folder / "predictions.jsonl"),
               "source_sha256": {name: file_digest(ROOT / name) for name in (
                   "scripts/inference/evaluate_frames.py", "grana/inference/resources.py",
                   "grana/inference/predictor.py", "grana/inference/acceleration.py", "grana/inference/window.py")}}
    write_json(folder / "receipt.json", receipt)
    return receipt


def merge_workers(plan, schedule, output, elapsed_seconds):
    expected = {identity(job) for job in jobs_for(plan)}
    records, receipts = {}, []
    plan_sha = file_digest(output / "plan.json")
    weights = plan["checkpoint"]["weights_sha256"]
    for device in schedule["devices"]:
        folder = output / "workers" / f"gpu_{device}"
        receipt = strict_json((folder / "receipt.json").read_bytes())
        data = (folder / "predictions.jsonl").read_bytes()
        rows = [strict_json(line) for line in data.splitlines()]
        keys = {task["key"] for task in schedule["shards"][str(device)]}
        mapping = schedule["device_mappings"][str(device)]
        if (receipt["plan_sha256"] != plan_sha or receipt["checkpoint"]["weights_sha256"] != weights
                or receipt.get("worker_device") != device
                or receipt.get("device_mapping") != {**mapping, "actual_worker_device_uuid": mapping["device_uuid"]}
                or receipt["journal_sha256"] != hashlib.sha256(data).hexdigest() or receipt["requests_written"] != len(rows)):
            raise ValueError("Worker receipt/journal identity differs from the campaign")
        for row in rows:
            key = identity(row)
            if (key not in expected or key in records or row["key"] not in keys
                    or row["worker_device"] != device or row["checkpoint_sha256"] != weights
                    or row.get("worker_local_device") != 0 or row.get("worker_device_uuid") != mapping["device_uuid"]
                    or row.get("execution") != "local_batched" or row.get("transport") != "local"):
                raise ValueError("Unexpected, duplicate or incorrectly assigned worker result")
            records[key] = row
        receipts.append(receipt)
    if set(records) != expected:
        raise ValueError(f"Workers omitted {len(expected - records.keys())} planned predictions")
    # Publish the entire seed set together only after all worker receipts and
    # staged seed completions have passed. A failed merge remains incomplete.
    destination = output / "runs"
    if destination.exists():
        raise FileExistsError(f"Merged evaluation already exists: {destination}")
    staging = output / ".runs.pending"
    staging.mkdir(exist_ok=False)
    for seed in plan["seeds"]:
        directory = staging / f"seed_{seed}"
        directory.mkdir(parents=True, exist_ok=False)
        selected = [records[identity(job)] for job in jobs_for(plan) if job["seed"] == seed]
        with (directory / "predictions.jsonl").open("x") as stream:
            stream.writelines(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n" for row in selected)
        write_json(directory / "completion.json", {"seed": seed, "plan_sha256": plan_sha, "checkpoint_sha256": weights,
            "requests_written": len(selected), "execution": "local_batched", "transport": "local",
            "journal_sha256": file_digest(directory / "predictions.jsonl"),
            "inference_amortized_seconds": math.fsum(row["inference_amortized_ms"] for row in selected) / 1000,
            "finished_utc": datetime.now(timezone.utc).isoformat()})
    initial_batches = {str(row["worker_device"]): row["initial_batch_size"] for row in receipts}
    final_batches = {str(row["worker_device"]): row["final_batch_size"] for row in receipts}
    receipt = {"execution": "local_batched", "transport": "local", "devices": schedule["devices"],
               "device_mappings": schedule["device_mappings"],
               "batch_size_scope": "per_gpu", "requested_batch_size_per_gpu": schedule["requested_batch_size_per_gpu"],
               "requested_aggregate_batch_capacity": schedule["requested_aggregate_batch_capacity"],
               "initial_per_gpu_batch_sizes": initial_batches, "final_per_gpu_batch_sizes": final_batches,
               "initial_aggregate_batch_capacity": sum(initial_batches.values()),
               "final_aggregate_batch_capacity": sum(final_batches.values()), "workers_synchronized": False,
               "batch_capacity_note": "Sum of independent per-GPU batch caps; workers execute asynchronously.",
               "requests_written": len(records), "trajectory_count": len(plan["trajectories"]),
               "seeds": plan["seeds"], "campaign_wall_seconds": elapsed_seconds, "plan_sha256": plan_sha,
               "workers": receipts, "all_expected_predictions_present": True}
    write_json(output / ".execution.pending.json", receipt)
    staging.rename(destination)
    (output / ".execution.pending.json").rename(output / "execution.json")
    return receipt


def run(args):
    import torch

    plan, schedule, output = prepare_campaign(args, torch.cuda.device_count())
    (output / "logs").mkdir()
    processes, handles = [], []
    started = time.monotonic()
    try:
        for device in schedule["devices"]:
            command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--worker-device", str(device),
                       "--plan", str(output / "plan.json"), "--output", str(output), "--batch-size", str(args.batch_size),
                       "--memory-fraction", str(args.memory_fraction)]
            if args.max_batch_size is not None:
                command.extend(["--max-batch-size", str(args.max_batch_size)])
            if args.model_name_or_path:
                command.extend(["--model-name-or-path", args.model_name_or_path])
            handle = (output / "logs" / f"gpu_{device}.log").open("x")
            handles.append(handle)
            env = {**os.environ, "OMP_NUM_THREADS": str(schedule["cpu_threads_per_worker"]),
                   "CUDA_VISIBLE_DEVICES": schedule["device_mappings"][str(device)]["worker_cuda_visible_devices"],
                   "MKL_NUM_THREADS": str(schedule["cpu_threads_per_worker"]), "OPENBLAS_NUM_THREADS": "1",
                   "TORCHINDUCTOR_COMPILE_THREADS": "2", "USE_TF": "0"}
            processes.append((device, subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle,
                              stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)))
        pending = list(processes)
        while pending:
            for item in pending[:]:
                code = item[1].poll()
                if code is not None:
                    if code:
                        raise RuntimeError(f"GPU worker {item[0]} failed ({code}); inspect {output / 'logs' / f'gpu_{item[0]}.log'}")
                    pending.remove(item)
            if pending:
                time.sleep(.2)
    finally:
        for _, process in processes:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=10)
        for handle in handles:
            handle.close()
    result = merge_workers(plan, schedule, output, time.monotonic() - started)
    print(json.dumps({"output": str(output), "devices": result["devices"], "requests": result["requests_written"],
                      "batch_size_scope": "per_gpu", "per_gpu_batch_sizes": result["final_per_gpu_batch_sizes"],
                      "aggregate_batch_capacity": result["final_aggregate_batch_capacity"],
                      "elapsed_seconds": result["campaign_wall_seconds"]}), flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New campaign directory")
    parser.add_argument("--devices", default="0", help="Parent logical CUDA indices (default: 0, the first visible GPU); auto uses all parent-visible GPUs; workers isolate each assigned UUID as cuda:0")
    parser.add_argument("--batch-size", type=batch_size_value, default=1, help="Per-GPU physical batch size (default: fixed 1). Optional auto probes memory capacity starting at 32; fixed sizes are never reduced on OOM")
    parser.add_argument("--memory-fraction", type=float, default=.90, help="Fraction of currently free memory available for auto-sized activations")
    parser.add_argument("--max-batch-size", type=int, help="Optional upper bound on physical batch size per GPU")
    parser.add_argument("--seeds", type=int, nargs="+", help="Optional subset of the prepared sampling seeds")
    parser.add_argument("--model-name-or-path")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-device", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    return run_worker(args) if args.worker else run(args)


if __name__ == "__main__":
    main()
