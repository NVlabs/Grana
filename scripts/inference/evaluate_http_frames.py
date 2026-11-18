#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare and execute paired POI/LAST C,D evaluations over an entire cache manifest.

Each seed uses independently stratified target frames. Separate fixed anchors
check repeatability; they are excluded from quality metrics. Public HTTP calls
receive RGB and task prompts only. A separate summary script computes metrics.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
from string import Template
import subprocess
import sys
import time
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from grana.data.frame_sampling import DEFAULT_EVAL_STRIDE_PAIRS
from grana.inference.cli import load_request
from grana.inference.client import build_binary, predict
from grana.inference.window import compose_frame_indices
from scripts.inference.replay_http import uniform_targets, validate_reply, write_json


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_revision():
    """Git is optional in an extracted source release; never use an enclosing repo."""
    if not (ROOT / ".git").exists():
        return None
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def stratified_targets(num_frames, samples, identity, seed):
    """One draw in each disjoint time bin, without using labels or predictions."""
    if type(num_frames) is not int or num_frames < 2 or type(samples) is not int or samples < 2:
        raise ValueError("num_frames and samples must be integers >= 2")
    count = min(samples, num_frames)
    rng = random.Random(int(hashlib.sha256(f"{seed}|{identity}".encode()).hexdigest(), 16))
    return [rng.randrange(i * num_frames // count, (i + 1) * num_frames // count)
            for i in range(count)]


def cache_entries(config_path, data_root=None):
    """Read a portable cache manifest, resolving paths without modifying it."""
    import yaml

    config_path = Path(config_path).expanduser().resolve()
    body = yaml.safe_load(config_path.read_text())
    if not isinstance(body, dict) or set(body) != {"caches"} or not isinstance(body["caches"], list) or not body["caches"]:
        raise ValueError("Manifest must contain a nonempty caches list")
    environment = dict(os.environ)
    base = config_path.parent
    if data_root is not None:
        base = Path(data_root).expanduser().resolve()
        environment.update(GRANA_DATA_ROOT=str(base), GRANA_POSTTRAIN_DATA_ROOT=str(base))
    entries = []
    for item in body["caches"]:
        entry = {"path": item} if isinstance(item, str) else dict(item) if isinstance(item, dict) else {}
        if not isinstance(entry.get("path"), str) or not entry["path"].strip():
            raise ValueError("Cache manifest entry must be a path or mapping with path")
        try:
            expanded = Template(entry["path"]).substitute(environment)
        except (KeyError, ValueError) as error:
            raise ValueError("Unresolved or invalid environment variable in cache path") from error
        entry["path"] = str((base / Path(expanded).expanduser()).resolve())
        entries.append(entry)
    return entries


def prepare(args):
    import numpy as np
    from datasets import load_from_disk
    from grana.data.cache import CachedTrajectories

    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Use a new experiment directory: {output}")
    if len(args.seeds) != len(set(args.seeds)) or not args.seeds:
        raise ValueError("Seeds must be nonempty and unique")
    if args.anchors < 2:
        raise ValueError("At least two separate repeatability anchors are required")
    if len(args.weights_sha256) != 64 or any(c not in "0123456789abcdef" for c in args.weights_sha256):
        raise ValueError("Expected a lowercase SHA256 checkpoint identity")
    config_path = args.config.expanduser().resolve()
    entries = cache_entries(config_path, args.data_root)
    dataset = CachedTrajectories(entries)
    if any("C" not in trajectory.roles for trajectory in dataset.records):
        raise ValueError("Every trajectory in this comparison must support C and derived D")
    raw_rows = {}
    caches = {t.cache_root for t in dataset.records}
    cache_names = Counter(cache.name for cache in caches)
    for cache in caches:
        for row in load_from_disk(str(cache / "processed_dataset")):
            raw_rows[(str(cache), row["id"])] = row
    output.mkdir(parents=True)
    (output / "inputs").mkdir()
    trajectories = []
    for trajectory in sorted(dataset.records, key=lambda t: (str(t.cache_root), t.trajectory_id)):
        identity = json.dumps([str(trajectory.cache_root.resolve()), trajectory.trajectory_id])
        key = hashlib.sha256(identity.encode()).hexdigest()[:16]
        row = raw_rows[(str(trajectory.cache_root), trajectory.trajectory_id)]
        fps = row["data_gen_output_fps"]
        if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not np.isfinite(fps) or fps <= 0:
            raise ValueError("Every source needs a finite positive data_gen_output_fps")
        sources = {}
        for view, path in zip(trajectory.view_names, trajectory.frame_paths, strict=True):
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            if array.dtype != np.uint8 or tuple(array.shape) != trajectory.frame_shape:
                raise ValueError(f"Invalid RGB source: {path}")
            stat = path.stat()
            sources[view] = {"path": str(path), "shape": list(array.shape),
                             "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        request_path = output / "inputs" / f"{key}.request.json"
        metadata_path = output / "inputs" / f"{key}.metadata.json"
        request = {"views": {v: str(p) for v, p in zip(trajectory.view_names, trajectory.frame_paths, strict=True)},
                   "task": trajectory.task, "stage_prompts": list(trajectory.stage_prompts)}
        stage_count = len(trajectory.stage_prompts)
        metadata = {"trajectory_id": trajectory.trajectory_id, "num_frames": trajectory.num_frames,
                    "source_fps": fps, "stage_ids": list(trajectory.stage_ids),
                    "within_stage_progress": list(trajectory.within_stage_progress),
                    "whole_progress": [(c + d) / stage_count for c, d in
                                       zip(trajectory.stage_ids, trajectory.within_stage_progress, strict=True)]}
        write_json(request_path, request)
        write_json(metadata_path, metadata)
        trajectories.append({"key": key, "trajectory_id": trajectory.trajectory_id,
            "group": (trajectory.cache_root.name if cache_names[trajectory.cache_root.name] == 1
                      else str(trajectory.cache_root)), "cache": str(trajectory.cache_root),
            "task": trajectory.task, "num_frames": trajectory.num_frames, "stage_count": stage_count,
            "source_fps": fps, "request_path": str(request_path), "metadata_path": str(metadata_path),
            "request_sha256": file_digest(request_path), "metadata_sha256": file_digest(metadata_path),
            "sources": sources,
            "targets": {str(seed): stratified_targets(trajectory.num_frames, args.samples, identity, seed)
                        for seed in args.seeds},
            "anchors": uniform_targets(trajectory.num_frames, min(args.anchors, trajectory.num_frames))})
    plan = {"schema": "grana-http-frame-comparison-plan-v1", "seeds": args.seeds,
            "samples_per_trajectory": args.samples, "anchors_per_trajectory": args.anchors,
            "stride_pair": list(DEFAULT_EVAL_STRIDE_PAIRS[0]),
            "checkpoint": {"path": str(args.checkpoint.expanduser().resolve()), "weights_sha256": args.weights_sha256},
            "config_path": str(config_path), "config_sha256": file_digest(config_path),
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source_commit": source_revision(),
            "source_files_sha256": {str(path.relative_to(ROOT)): file_digest(path)
                                    for path in sorted([*ROOT.joinpath("grana").rglob("*.py"),
                                                        *ROOT.joinpath("scripts/inference").glob("*.py")])},
            "sampling": "One uniform draw per disjoint time bin; min(samples,num_frames); seeds share no label-dependent selection. Separate uniform anchors are excluded from quality metrics.",
            "primary_metric": "For each trajectory, combine quality squared errors across seeds, then take RMSE; macro is the arithmetic mean of trajectory RMSEs.",
            "ground_truth_policy": "Separate local metadata, used only after the HTTP response; no GT or oracle stage in request.",
            "trajectories": trajectories}
    write_json(output / "plan.json", plan)
    requests = sum(2 * (len(t["targets"][str(seed)]) + len(t["anchors"]))
                   for seed in args.seeds for t in trajectories)
    print(json.dumps({"plan": str(output / "plan.json"), "trajectories": len(trajectories),
                      "requests": requests, "source_frames": sum(t["num_frames"] for t in trajectories)}), flush=True)
    return plan


def run(args):
    plan_path = args.plan.expanduser().resolve()
    plan = json.loads(plan_path.read_text())
    if plan["schema"] != "grana-http-frame-comparison-plan-v1" or args.seed not in plan["seeds"]:
        raise ValueError("Unknown plan or seed")
    if plan["stride_pair"] != list(DEFAULT_EVAL_STRIDE_PAIRS[0]):
        raise ValueError("Plan geometry differs from the current evaluation default")
    output = plan_path.parent / "runs" / f"seed_{args.seed}"
    if output.exists():
        raise FileExistsError(f"Use a fresh run output directory: {output}")
    with urlopen(args.server.rstrip("/") + "/healthz", timeout=30) as stream:
        health = json.load(stream)
    weights = plan["checkpoint"]["weights_sha256"]
    runtime = health["checkpoint"]["inference_runtime"]
    if (not health.get("ready") or health.get("wire_formats") != ["grana-binary-v1"]
            or health["checkpoint"]["weights_sha256"] != weights
            or not runtime["gpu_image_resize"] or runtime.get("policy") != "cuda_optimized_v1"):
        raise ValueError("Expected the plan checkpoint on a ready canonical binary/GPU-resize server")
    output.mkdir(parents=True)
    write_json(output / "health.json", health)
    count, started = 0, time.monotonic()
    with (output / "predictions.jsonl").open("x") as journal:
        for trajectory in plan["trajectories"]:
            request_path, metadata_path = Path(trajectory["request_path"]), Path(trajectory["metadata_path"])
            if (file_digest(request_path) != trajectory["request_sha256"]
                    or file_digest(metadata_path) != trajectory["metadata_sha256"]):
                raise ValueError("Input/metadata changed after plan creation")
            for source in trajectory["sources"].values():
                stat = Path(source["path"]).stat()
                if (stat.st_size, stat.st_mtime_ns) != (source["size_bytes"], source["mtime_ns"]):
                    raise ValueError("RGB source changed after plan creation")
            labels = json.loads(metadata_path.read_text())
            for kind, targets in (("anchor", trajectory["anchors"]),
                                  ("quality", trajectory["targets"][str(args.seed)])):
                for index, frame in enumerate(targets):
                    # Alternate order; both requests score exactly the same physical target.
                    modes = ("poi", "last") if (index + plan["seeds"].index(args.seed)) % 2 == 0 else ("last", "poi")
                    for mode in modes:
                        request = load_request(request_path, frame_index=frame, prediction_frame=mode)
                        expected = compose_frame_indices(frame, trajectory["num_frames"], prediction_frame=mode)
                        if request.frame_indices != expected or expected[request.poi_slot] != frame:
                            raise ValueError("Request window/selected source differs from canonical composition")
                        if mode == "last" and max(expected) > frame:
                            raise ValueError("LAST observed a future source frame")
                        request_id = f"eval-{args.seed}-{trajectory['key']}-{kind}-{frame}-{mode}"
                        start = time.perf_counter()
                        envelope = predict(args.server, request, roles=("C", "D"),
                                           request_id=request_id, timeout=args.timeout)
                        elapsed = (time.perf_counter() - start) * 1000
                        result = validate_reply(envelope, request, ("C", "D"), mode, request_id, weights)
                        predictions = {"C_ids": result["stage"]["ids"],
                                       "D_values": result["progress"]["D"]["values"],
                                       "D_focus_stage": result["D_focus_stage"]}
                        record = {"seed": args.seed, "key": trajectory["key"], "kind": kind,
                            "source_frame": frame, "mode": mode, "slot": request.poi_slot,
                            "frame_indices": list(expected), "request_id": request_id,
                            "input_sha256": hashlib.sha256(build_binary(request, ("C", "D"), request_id="identity")).hexdigest(),
                            "prediction_sha256": digest(predictions), **predictions,
                            "gt": {"stage_id": labels["stage_ids"][frame],
                                   "within_stage_progress": labels["within_stage_progress"][frame]},
                            "client_total_ms": elapsed, "server_handler_ms": envelope["elapsed_seconds"] * 1000,
                            "checkpoint_sha256": weights}
                        journal.write(json.dumps(record, allow_nan=False) + "\n")
                        journal.flush()
                        count += 1
                    if (index + 1) % 32 == 0 or index + 1 == len(targets):
                        print(json.dumps({"seed": args.seed, "trajectory": trajectory["trajectory_id"],
                                          "group": trajectory["group"], "kind": kind,
                                          "targets_completed": index + 1, "targets_total": len(targets),
                                          "requests_total": count, "elapsed_seconds": round(time.monotonic() - started, 1)}), flush=True)
    completion = {"seed": args.seed, "plan_sha256": file_digest(plan_path), "checkpoint_sha256": weights,
                  "requests_written": count, "elapsed_seconds": time.monotonic() - started,
                  "journal_sha256": file_digest(output / "predictions.jsonl"),
                  "finished_utc": datetime.now(timezone.utc).isoformat()}
    write_json(output / "completion.json", completion)
    print(json.dumps({"complete": completion}), flush=True)
    return completion


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--config", required=True, type=Path)
    prepare_parser.add_argument("--data-root", type=Path, help="Base for relative cache paths and GRANA_DATA_ROOT; defaults to the manifest directory")
    prepare_parser.add_argument("--checkpoint", required=True, type=Path)
    prepare_parser.add_argument("--weights-sha256", required=True)
    prepare_parser.add_argument("--output", required=True, type=Path)
    prepare_parser.add_argument("--seeds", nargs="+", type=int, default=[11, 29, 47])
    prepare_parser.add_argument("--samples", type=int, default=128)
    prepare_parser.add_argument("--anchors", type=int, default=8)
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--plan", required=True, type=Path)
    run_parser.add_argument("--seed", required=True, type=int)
    run_parser.add_argument("--server", required=True)
    run_parser.add_argument("--timeout", type=float, default=1200)
    args = parser.parse_args(argv)
    return prepare(args) if args.command == "prepare" else run(args)


if __name__ == "__main__":
    main()
