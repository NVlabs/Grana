# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit device overrides and measured memory capacity for bulk inference.

Callers select device allocation and fixed or automatic batch sizing explicitly.
Latency benchmarks keep their declared batch size.
They do not change CUDA visibility or fall back to another model backend.
"""
from __future__ import annotations

import math


def select_devices(spec, device_count, job_count):
    """Return logical CUDA indices, respecting an explicit visible-device list."""
    if type(device_count) is not int or device_count < 1:
        raise RuntimeError("Bulk inference needs at least one visible CUDA device")
    if type(job_count) is not int or job_count < 1:
        raise ValueError("There must be at least one independent task")
    if spec == "auto":
        devices = list(range(device_count))
    elif isinstance(spec, str):
        parts = spec.split(",")
        if not parts or any(not part.strip().isdigit() for part in parts):
            raise ValueError("devices must be auto or comma-separated logical CUDA indices")
        devices = [int(part.strip()) for part in parts]
        if len(devices) != len(set(devices)) or any(index >= device_count for index in devices):
            raise ValueError("Device indices must be distinct and visible to this process")
    else:
        raise ValueError("devices must be auto or comma-separated logical CUDA indices")
    # A task cannot be split by this scheduler. Never duplicate work merely to
    # allocate a GPU when there are fewer independent tasks than devices.
    return devices[:min(len(devices), job_count)]


def balanced_shards(tasks, devices):
    """Greedy cost balancing preserves complete trajectory tasks and identities."""
    if not tasks or not devices or len(devices) != len(set(devices)):
        raise ValueError("Tasks and distinct devices must be nonempty")
    keys = [task.get("key") for task in tasks]
    if any(not isinstance(key, str) or not key for key in keys) or len(keys) != len(set(keys)):
        raise ValueError("Task keys must be unique nonempty strings")
    if any(type(task.get("cost")) not in (int, float) or not math.isfinite(task["cost"])
           or task["cost"] <= 0 for task in tasks):
        raise ValueError("Task costs must be finite and positive")
    shards, loads = {device: [] for device in devices}, {device: 0. for device in devices}
    order = {device: index for index, device in enumerate(devices)}
    for task in sorted(tasks, key=lambda item: (-item["cost"], item["key"])):
        device = min(devices, key=lambda index: (loads[index], order[index]))
        shards[device].append(task)
        loads[device] += task["cost"]
    return shards


def is_cuda_oom(error):
    """Recognize CUDA allocation failures, including wrapped compiler errors."""
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        message = str(error).lower()
        if ("cuda out of memory" in message or "cuda error: out of memory" in message
                or (type(error).__name__ == "OutOfMemoryError"
                    and (type(error).__module__ == "torch" or "cuda" in type(error).__module__))):
            return True
        error = (error.__cause__ if error.__cause__ is not None else
                 None if error.__suppress_context__ else error.__context__)
    return False


def compare_batch_predictions(reference, candidate, d_absolute_tolerance=0.001):
    """Check all eight C/D outputs against singleton predictions on the same inputs.

    This is a numerical qualification sample, not an accuracy guarantee. Hard
    stage decisions and the D focus must match exactly; the D tolerance is in
    within-stage [0,1] units. A discrepancy must remain visible in the receipt.
    """
    if (not reference or len(reference) != len(candidate)
            or type(d_absolute_tolerance) not in (int, float)
            or not math.isfinite(d_absolute_tolerance) or d_absolute_tolerance < 0):
        raise ValueError("Need paired predictions and a finite nonnegative D tolerance")
    same_ids, same_focus, maximum = True, True, 0.
    for expected, actual in zip(reference, candidate, strict=True):
        expected_ids, actual_ids = expected["stage"]["ids"], actual["stage"]["ids"]
        expected_values = expected["progress"]["D"]["values"]
        actual_values = actual["progress"]["D"]["values"]
        if any(len(values) != 8 for values in (expected_ids, actual_ids, expected_values, actual_values)):
            raise ValueError("Numerical qualification needs all eight C/D outputs")
        if any(type(value) is not int for value in (*expected_ids, *actual_ids)):
            raise ValueError("Numerical qualification needs integer C IDs")
        if any(type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1
               for value in (*expected_values, *actual_values)):
            raise ValueError("Numerical qualification needs finite D scores in [0,1]")
        same_ids = same_ids and expected_ids == actual_ids
        same_focus = same_focus and expected["D_focus_stage"] == actual["D_focus_stage"]
        maximum = max(maximum, *(abs(a - b) for a, b in zip(expected_values, actual_values, strict=True)))
    return {"predictions_compatible": same_ids and same_focus and maximum <= d_absolute_tolerance,
            "matched_C_ids": same_ids, "matched_D_focus": same_focus,
            "max_abs_D_difference": maximum, "d_absolute_tolerance": d_absolute_tolerance,
            "windows_checked": len(reference)}


def autotune_batch_size(probe, max_batch_size, memory_budget_bytes, cleanup=None, refine_steps=2):
    """Select the largest measured physical batch within the memory budget.

    ``probe(n)`` warms the exact physical batch before timing it, then returns
    peak_memory_bytes (including warmup), elapsed_seconds (warm timing only),
    and windows_measured. Start at min(32, max_batch_size) per GPU. A fitting
    batch doubles until the workload/caller cap or an allocation boundary;
    an initial failure halves until a batch fits. The measured fitting/failing
    interval then gets at most ``refine_steps`` additional probes. Only CUDA
    OOM is recoverable; other failures propagate.

    Warm throughput is recorded only: neither short/noisy timings nor output
    differences can reduce the chosen fitting batch. These are local capacity
    probes, not HTTP latency benchmarks.
    """
    if type(max_batch_size) is not int or max_batch_size < 1:
        raise ValueError("max_batch_size must be a positive integer")
    if type(memory_budget_bytes) is not int or memory_budget_bytes < 1:
        raise ValueError("memory_budget_bytes must be a positive integer")
    if type(refine_steps) is not int or refine_steps < 0:
        raise ValueError("refine_steps must be a nonnegative integer")
    probes, fitting = [], []
    largest_fitting, failed = 0, None
    initial_probe = min(32, max_batch_size)
    stop_reason = "max_batch_size"

    def attempt(size):
        try:
            measurement = probe(size)
            if not isinstance(measurement, dict):
                raise ValueError("Probe must return a measurement dictionary")
            peak, seconds = measurement.get("peak_memory_bytes"), measurement.get("elapsed_seconds")
            windows = measurement.get("windows_measured")
            if (type(peak) is not int or peak < 0 or type(seconds) not in (int, float)
                    or not math.isfinite(seconds) or seconds <= 0
                    or type(windows) is not int or windows < size or windows % size):
                raise ValueError("Probe needs nonnegative memory, positive finite warm timing, "
                                 "and a positive whole number of measured physical batches")
            throughput = windows / seconds
            if not math.isfinite(throughput):
                raise ValueError("Probe throughput must be finite")
            outcome = "memory_budget" if peak > memory_budget_bytes else "ok"
            row = {**measurement, "batch_size": size, "outcome": outcome,
                   "throughput_requests_per_second": throughput}
        except Exception as error:
            if not is_cuda_oom(error):
                raise
            row = {"batch_size": size, "outcome": "cuda_oom", "error_type": type(error).__name__}
        finally:
            if cleanup is not None:
                cleanup()
        probes.append(row)
        return row

    size = initial_probe
    while True:
        row = attempt(size)
        if row["outcome"] != "ok":
            failed = size
            stop_reason = row["outcome"]
            if fitting or size == 1:
                break
            size = max(1, size // 2)
            continue
        fitting.append(row)
        largest_fitting = size
        if failed is not None or size == max_batch_size:
            break
        size = min(max_batch_size, size * 2)
    if not fitting:
        raise RuntimeError("No physical batch fits the requested GPU memory budget; batch size 1 failed")
    for _ in range(refine_steps):
        if failed is None or failed - largest_fitting <= 1:
            break
        candidate = (largest_fitting + failed) // 2
        row = attempt(candidate)
        if row["outcome"] == "ok":
            fitting.append(row)
            largest_fitting = candidate
        else:
            failed = candidate
    maximum = max(row["throughput_requests_per_second"] for row in fitting)
    chosen = max(fitting, key=lambda row: row["batch_size"])
    selected = chosen["batch_size"]
    return selected, {"mode": "auto", "selected_batch_size": selected,
        "selected_peak_memory_bytes": chosen["peak_memory_bytes"],
        "selected_throughput_requests_per_second": chosen["throughput_requests_per_second"],
        "maximum_measured_throughput_requests_per_second": maximum,
        "largest_fitting_batch_size": largest_fitting, "memory_budget_bytes": memory_budget_bytes,
        "max_batch_size": max_batch_size, "initial_probe_batch_size": initial_probe,
        "probes": probes, "stop_reason": stop_reason,
        "selection_policy": {"objective": "memory_capacity", "preferred_start_batch_size": 32,
                             "refine_steps": refine_steps, "throughput_role": "report_only"},
        "timing_note": "Warm throughput is recorded, not used for capacity selection. Warmup and compilation "
                       "are excluded from elapsed_seconds. These are not HTTP latency measurements."}
