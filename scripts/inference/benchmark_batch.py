#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare one local physical batch with serial singleton public predictions.

One selected POI is one complete eight-frame window, with all camera views.
The same distinct windows are loaded once, warmed and reused for both variants.
Timing includes GPU resize and predictor postprocessing, but no HTTP, file
loading, hashing, output validation or ground-truth access.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np

from scripts.inference.benchmark_http import ROLE_SELECTIONS, file_digest, json_digest, prepare_input
from scripts.inference.replay_http import validate_reply
from scripts.inference.summarize_http_frames import strict_json, validate_plan

VARIANTS = ("batch", "serialized")
SOURCE_FILES = (
    "scripts/inference/benchmark_batch.py", "scripts/inference/benchmark_http.py",
    "scripts/inference/replay_http.py", "scripts/inference/summarize_http_frames.py",
    "grana/inference/predictor.py", "grana/inference/acceleration.py", "grana/inference/optimization.py",
    "grana/inference/cli.py", "grana/inference/window.py", "grana/inference/selection.py",
    "grana/modeling/model.py", "grana/modeling/inputs.py", "grana/modeling/image_preprocessing.py",
    "grana/modeling/vision_sdpa.py", "grana/modeling/backends.py",
)


def uniform_items(items, count):
    if not 1 <= count <= len(items):
        raise ValueError("Insufficient distinct real targets for the requested batch size")
    indices = [len(items) // 2] if count == 1 else [i * (len(items) - 1) // (count - 1) for i in range(count)]
    return [items[index] for index in indices]


def video_identity(path):
    raw = strict_json(path.read_bytes())
    if "frame_indices" in raw or not isinstance(raw.get("views"), dict) or not raw["views"]:
        raise ValueError("Benchmark input must contain full-video .npy views without frame_indices")
    sources = []
    lengths = set()
    for view, filename in raw["views"].items():
        if not isinstance(filename, str):
            raise ValueError("Benchmark requires full-video .npy view paths")
        source = Path(filename).expanduser()
        source = (path.parent / source).resolve() if not source.is_absolute() else source.resolve()
        if source.suffix.lower() != ".npy":
            raise ValueError("Benchmark requires full-video .npy view paths")
        array = np.load(source, mmap_mode="r", allow_pickle=False)
        if array.dtype != np.uint8 or array.ndim != 4 or array.shape[-1] != 3 or not len(array):
            raise ValueError("Expected nonempty RGB uint8 full-video camera arrays")
        sources.append((view, str(source)))
        lengths.add(len(array))
    if len(lengths) != 1:
        raise ValueError("Camera videos must have the same frame count")
    return tuple(sources), lengths.pop()


def plan_targets(plan, plan_path, count, seed):
    """Allocate targets evenly across trajectories, then spread within each one."""
    if seed not in plan["seeds"]:
        raise ValueError("seed must belong to the evaluation plan")
    groups, seen = [], set()
    for row in plan["trajectories"]:
        path = Path(row["request_path"]).expanduser()
        path = (plan_path.parent / path).resolve() if not path.is_absolute() else path.resolve()
        if row.get("request_sha256") and file_digest(path) != row["request_sha256"]:
            raise ValueError("Request changed after plan creation")
        sources, frames = video_identity(path)
        if frames != row["num_frames"]:
            raise ValueError("Video frame count differs from the plan")
        targets = []
        for frame in sorted(row["targets"][str(seed)]):
            identity = sources, frame
            if identity not in seen:
                seen.add(identity)
                targets.append(frame)
        if targets:
            groups.append({"row": row, "request_path": path, "targets": targets, "quota": 0})
    if sum(len(group["targets"]) for group in groups) < count:
        raise ValueError("Insufficient distinct real quality targets in this seed; no windows are duplicated")
    remaining = count
    while remaining:
        for group in groups:
            if group["quota"] < len(group["targets"]):
                group["quota"] += 1
                remaining -= 1
                if not remaining:
                    break
    selected = []
    for group in groups:
        if not group["quota"]:
            continue
        row = group["row"]
        selected.extend({"key": row["key"], "trajectory_id": row["trajectory_id"],
                         "request_path": group["request_path"], "source_frame": frame, "kind": "quality", "seed": seed}
                        for frame in uniform_items(group["targets"], group["quota"]))
    # Interleave trajectories so the workload order also covers the full shard.
    by_key = {group["row"]["key"]: [item for item in selected if item["key"] == group["row"]["key"]]
              for group in groups if group["quota"]}
    return [items[index] for index in range(max(map(len, by_key.values())))
            for items in by_key.values() if index < len(items)]


def prepare_workload(args):
    roles = tuple(args.roles)
    expected_weights = None
    if args.plan is not None:
        path = args.plan.expanduser().resolve()
        plan = validate_plan(strict_json(path.read_bytes()))
        seed = plan["seeds"][0] if args.seed is None else args.seed
        targets = plan_targets(plan, path, args.batch_size, seed)
        checkpoint = args.checkpoint or Path(plan["checkpoint"]["path"])
        if not checkpoint.is_absolute():
            checkpoint = path.parent / checkpoint
        expected_weights = plan["checkpoint"]["weights_sha256"]
        source = {"kind": "plan_quality_targets", "path": str(path), "sha256": file_digest(path), "seed": seed}
    else:
        if args.checkpoint is None or args.seed is not None:
            raise ValueError("--input requires --checkpoint and does not use --seed")
        path = args.input.expanduser().resolve()
        _, length = video_identity(path)
        targets = [{"key": "input", "trajectory_id": path.stem, "request_path": path,
                    "source_frame": frame, "kind": "uniform_full_video"}
                   for frame in uniform_items(list(range(length)), args.batch_size)]
        checkpoint = args.checkpoint
        source = {"kind": "full_video_input", "path": str(path), "sha256": file_digest(path)}
    requests, descriptors = [], []
    for target in targets:
        request, identity, geometry = prepare_input(target["request_path"], target["source_frame"],
                                                     args.prediction_frame, roles)
        requests.append(request)
        descriptors.append({**target, "request_path": str(target["request_path"]),
                            "identity": identity, "geometry": geometry})
    workload = {"source": source, "windows": descriptors, "count": len(requests),
                "trajectory_counts": dict(Counter(item["key"] for item in descriptors)),
                "prediction_frame": args.prediction_frame, "roles": list(roles),
                "input_sha256": json_digest([item["identity"]["input_sha256"] for item in descriptors]),
                "ordering": "trajectory-interleaved, uniformly spread source targets; identical in both variants"}
    return requests, workload, checkpoint.expanduser().resolve(), expected_weights


def counter_delta(before, after):
    return {key: counter_delta(before.get(key, {}), value) if isinstance(value, dict) else value - before.get(key, 0)
            for key, value in after.items() if isinstance(value, dict) or type(value) is int}


def round_order(index):
    return VARIANTS if index % 2 == 0 else tuple(reversed(VARIANTS))


def summarize(rows, windows):
    measured = [row for row in rows if row["phase"] == "measured"]
    values = sorted(row["batch_latency_ms"] for row in measured)
    def stats(samples):
        def quantile(q):
            position = (len(samples) - 1) * q
            low, high = math.floor(position), math.ceil(position)
            return samples[low] + (samples[high] - samples[low]) * (position - low)
        return {"mean": statistics.fmean(samples), "p50": quantile(.5), "p95": quantile(.95),
                "minimum": samples[0], "maximum": samples[-1]}
    return {"rounds": len(measured), "batch_latency_ms": stats(values),
            "per_selected_poi_amortized_ms": stats([value / windows for value in values]),
            "windows_per_second": windows * len(measured) * 1000 / sum(values),
            "peak_allocated_bytes": max(row["peak_allocated_bytes"] for row in measured),
            "peak_reserved_bytes": max(row["peak_reserved_bytes"] for row in measured)}


def source_identity():
    result = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    status = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=normal"],
                            capture_output=True, text=True, check=False)
    return {"git_commit": result.stdout.strip() if result.returncode == 0 else None,
            "git_dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
            "sha256": {name: file_digest(ROOT / name) for name in SOURCE_FILES}}


def run(args, *, predictor_factory=None, cuda=None, clock=None):
    if tuple(args.roles) not in ROLE_SELECTIONS:
        raise ValueError("roles must be A, C, or C D")
    if any(type(value) is not int or value < 1 for value in (args.batch_size, args.warmup, args.repeats)):
        raise ValueError("batch-size, warmup and repeats must be positive integers")
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Use a fresh benchmark output: {output}")
    requests, workload, checkpoint, expected_weights = prepare_workload(args)
    import torch
    if cuda is None:
        cuda = torch.cuda
    if cuda.device_count() != 1:
        raise RuntimeError("Isolate one GPU with CUDA_VISIBLE_DEVICES before starting this benchmark")
    cuda.set_device(0)
    if predictor_factory is None:
        from grana.inference.predictor import GranaPredictor
        predictor_factory = GranaPredictor
    clock = time.perf_counter if clock is None else clock
    source = source_identity()
    predictor = predictor_factory(checkpoint, model_name_or_path=args.model_name_or_path, device="cuda:0")
    weights = predictor.checkpoint["weights_sha256"]
    if expected_weights is not None and weights != expected_weights:
        raise ValueError("Loaded checkpoint differs from the evaluation plan")
    runtime = predictor.checkpoint.get("inference_runtime", {})
    if runtime.get("policy") != "cuda_optimized_v1" or runtime.get("gpu_image_resize") is not True:
        raise ValueError("Benchmark requires the canonical CUDA/GPU-resize predictor")
    rows, roles = [], tuple(args.roles)
    for phase, count, base in (("warmup", args.warmup, 0), ("measured", args.repeats, args.warmup)):
        for index in range(count):
            order = round_order(base + index)
            for order_index, variant in enumerate(order):
                before = predictor.runtime_counters()
                cuda.synchronize(0)
                cuda.reset_peak_memory_stats(0)
                began = clock()
                if variant == "batch":
                    results = predictor.predict_batch(requests, roles=roles, batch_size=len(requests))
                else:
                    results = [predictor.predict(request, roles=roles) for request in requests]
                cuda.synchronize(0)
                seconds = clock() - began
                if not math.isfinite(seconds) or seconds <= 0:
                    raise ValueError("Measured elapsed time must be positive and finite")
                after = predictor.runtime_counters()
                if len(results) != len(requests):
                    raise ValueError("Predictor returned a different number of windows")
                for request, result in zip(requests, results, strict=True):
                    validate_reply({"request_id": "benchmark", "result": result}, request, roles,
                                   args.prediction_frame, "benchmark", weights)
                row = {"phase": phase, "round": index, "variant": variant, "variant_order": list(order),
                       "order_index": order_index, "windows": len(requests),
                       "physical_batch_size": len(requests) if variant == "batch" else 1,
                       "public_calls": 1 if variant == "batch" else len(requests),
                       "elapsed_seconds": seconds, "batch_latency_ms": seconds * 1000,
                       "per_selected_poi_amortized_ms": seconds * 1000 / len(requests),
                       "windows_per_second": len(requests) / seconds,
                       "peak_allocated_bytes": int(cuda.max_memory_allocated(0)),
                       "peak_reserved_bytes": int(cuda.max_memory_reserved(0)),
                       "runtime_before": before, "runtime_after": after, "runtime_delta": counter_delta(before, after)}
                rows.append(row)
                print(json.dumps({key: row[key] for key in ("phase", "round", "variant", "batch_latency_ms",
                                                           "per_selected_poi_amortized_ms")}), flush=True)
    summaries = {variant: summarize([row for row in rows if row["variant"] == variant], len(requests))
                 for variant in VARIANTS}
    props = cuda.get_device_properties(0)
    report = {"schema": "grana-local-batch-throughput-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
              "execution": "local_public_predictor", "transport": "local", "workload": workload,
              "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
              "checkpoint": predictor.checkpoint, "checkpoint_metadata_sha256": file_digest(checkpoint / "metadata.json"),
              "source": source, "device": {"logical_device": 0, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                                          "name": props.name, "uuid": str(props.uuid), "total_memory_bytes": props.total_memory},
              "environment": {"python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda},
              "warmup_rounds_per_variant": args.warmup, "measured_rounds_per_variant": args.repeats,
              "timing": {"includes": ["complete public predictor calls", "GPU image resize", "CPU postprocessing", "CUDA synchronization"],
                         "excludes": ["HTTP transport", "frame loading", "hashing", "result validation", "ground-truth loading"],
                         "poi_unit": "One selected POI is one complete eight-frame multi-camera window; LAST selects slot 7, POI slot 5.",
                         "amortization_divisor": len(requests), "no_division_by_frame_or_camera_count": True,
                         "batch_latency_definition": "One batch call, or the complete serial sequence of singleton calls, over the same windows.",
                         "summary_scope": "Only measured rounds; warmup rounds including initial compilation are excluded from summaries and retained in raw_rounds.",
                         "note": "Local timings are not directly comparable to historical HTTP request-to-response latency."},
              "variants": summaries, "raw_rounds": rows, "runtime_counters": predictor.runtime_counters(),
              "throughput_speedup_batch_over_serialized": summaries["batch"]["windows_per_second"] / summaries["serialized"]["windows_per_second"]}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--plan", type=Path, help="Evaluation plan; select distinct quality targets from one seed")
    inputs.add_argument("--input", type=Path, help="Full-video .npy request JSON")
    parser.add_argument("--checkpoint", type=Path, help="Required with --input; defaults to the plan checkpoint")
    parser.add_argument("--seed", type=int, help="Plan quality seed; defaults to its first seed")
    parser.add_argument("--batch-size", type=int, default=64, help="Exact distinct workload size and per-GPU physical batch")
    parser.add_argument("--prediction-frame", choices=("poi", "last"), default="poi")
    parser.add_argument("--roles", nargs="+", choices=("A", "C", "D"), default=["C", "D"])
    parser.add_argument("--warmup", type=int, default=3, help="Complete warmup rounds per variant")
    parser.add_argument("--repeats", type=int, default=5, help="Measured rounds per variant; execution order alternates")
    parser.add_argument("--model-name-or-path")
    parser.add_argument("--output", type=Path, required=True, help="New JSON report")
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
