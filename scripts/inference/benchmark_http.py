#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Measure public binary HTTP prediction on the current evaluation CTSP window.

The external server stays loaded. Primary latency runs from sending an already
packed request through receipt of the full response, before JSON parsing.
Complete-client timing, including packing and parsing, is retained separately.
Input loading, hashes, result validation and report writing are outside both.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from numbers import Real
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np

from grana.data.frame_sampling import DEFAULT_EVAL_STRIDE_PAIRS
from grana.inference import client
from grana.inference.cli import load_request
from grana.inference.selection import select_prediction
from grana.inference.window import compose_frame_indices

ROLE_SELECTIONS = (("A",), ("C",), ("C", "D"))
TRANSPORT_METRICS = ("client_send_to_receive_ms", "client_send_ms", "client_send_complete_to_receive_ms",
                     "http_round_trip_ms")
SOURCE_FILES = (
    "scripts/inference/benchmark_http.py", "grana/data/frame_sampling.py",
    "grana/inference/cli.py", "grana/inference/client.py", "grana/inference/server.py",
    "grana/inference/selection.py", "grana/inference/window.py", "grana/inference/predictor.py",
    "grana/inference/acceleration.py", "grana/inference/optimization.py",
)


def json_digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_input(path: Path, frame_index: int, prediction_frame: str, roles: tuple[str, ...]):
    """Load a canonical window once and identify both sources and transmitted RGB."""
    path = path.expanduser().resolve()
    request = load_request(path, frame_index=frame_index, prediction_frame=prediction_frame)
    raw_bytes = path.read_bytes()
    raw = client.json_loads(raw_bytes)
    sources, pixels = {}, {}
    for view, frames in request.images.items():
        source = Path(raw["views"][view]).expanduser()
        source = (path.parent / source).resolve() if not source.is_absolute() else source.resolve()
        array = np.load(source, mmap_mode="r", allow_pickle=False)
        expected = compose_frame_indices(frame_index, len(array), prediction_frame=prediction_frame)
        if request.frame_indices != expected:
            raise ValueError("Loaded input differs from the shared evaluation frame composition")
        stat = source.stat()
        sources[view] = {"path": str(source),
                         "shape": list(array.shape), "dtype": str(array.dtype),
                         "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        digest = hashlib.sha256()
        for frame in frames:
            digest.update(np.ascontiguousarray(frame).tobytes())
        pixels[view] = {"shape": [8, *frames[0].shape], "rgb_sha256": digest.hexdigest()}
    if request.frame_indices[request.poi_slot] != frame_index:
        raise ValueError("The selected slot must identify the requested source frame")
    if prediction_frame == "last" and max(request.frame_indices) > frame_index:
        raise ValueError("Last-frame benchmark cannot include future observations")
    descriptor = {"roles": list(roles), "prediction_frame": prediction_frame,
                  "poi_slot": request.poi_slot, "frame_indices": list(request.frame_indices),
                  "view_names": list(request.images), "pixels": pixels,
                  "task": request.task, "compressed_task_prompt": request.compressed_task_prompt,
                  "stage_prompts": list(request.stage_prompts), "focus_stage": request.focus_stage}
    context_stride, tail_stride = DEFAULT_EVAL_STRIDE_PAIRS[0]
    geometry = {"profile": "regular_ctsp", "query_order": "forward",
                "stride_source": "DEFAULT_EVAL_STRIDE_PAIRS[0]", "context_stride": context_stride,
                "tail_stride": tail_stride, "prediction_frame": prediction_frame,
                "poi_slot": request.poi_slot, "source_frame_index": frame_index,
                "frame_indices": list(request.frame_indices),
                "has_future_observations": max(request.frame_indices) > frame_index}
    identity = {"request_path": str(path), "request_json_sha256": hashlib.sha256(raw_bytes).hexdigest(),
                "input_sha256": json_digest(descriptor), "descriptor": descriptor, "sources": sources}
    return request, identity, geometry


def _finite(value, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return float(value)


def _rows(value, rows: int, columns: int, label: str, *, probabilities=False):
    if not isinstance(value, list) or len(value) != rows:
        raise ValueError(f"{label} must preserve all {rows} rows")
    for row in value:
        if not isinstance(row, list) or len(row) != columns:
            raise ValueError(f"{label} must have {columns} columns")
        numbers = [_finite(item, label) for item in row]
        if probabilities and (any(not 0 <= item <= 1 for item in numbers)
                              or not math.isclose(sum(numbers), 1.0, abs_tol=1e-5)):
            raise ValueError(f"{label} must contain normalized probabilities")


def validate_response(response: dict, request, roles: tuple[str, ...], mode: str, request_id: str) -> dict:
    """Reject workload, runtime or output changes instead of reporting their speed."""
    if response.get("schema") != client.SCHEMA or response.get("request_id") != request_id:
        raise ValueError("Response schema or request ID changed")
    if _finite(response.get("elapsed_seconds"), "server elapsed time") < 0:
        raise ValueError("Server elapsed time must be nonnegative")
    result = response["result"]
    expected = {"roles": list(roles), "poi_slot": request.poi_slot, "prediction_frame": mode,
                "frame_indices": list(request.frame_indices), "view_names": list(request.images)}
    if any(result.get(key) != value for key, value in expected.items()):
        raise ValueError("Response changed roles, frame selection, source indices or camera order")
    checkpoint = result["checkpoint"]
    weights = checkpoint.get("weights_sha256")
    if not isinstance(weights, str) or len(weights) != 64 or any(char not in "0123456789abcdef" for char in weights):
        raise ValueError("Checkpoint must identify its SHA256 weights")
    runtime = checkpoint.get("inference_runtime", {})
    required = ("gpu_image_resize", "gpu_image_preprocessing", "batched_vision_sdpa", "reuse_images",
                "merge_lora", "cpu_position_ids", "compile_decoder", "compile_vision")
    if (runtime.get("policy") != "cuda_optimized_v1" or any(runtime.get(key) is not True for key in required)
            or runtime.get("attn_implementation") != "sdpa"
            or runtime.get("delta_rule_backend", {}).get("resolved") != "fla"
            or runtime.get("causal_conv1d", {}).get("active") is not True
            or runtime.get("text_input_cache", {}).get("enabled") is not True
            or runtime.get("fla_output_guard", {}).get("applied") is not True):
        raise ValueError("Benchmark requires the supported CUDA runtime with binary HTTP and GPU resize")
    progress_roles = set(roles) - {"C"}
    if set(result.get("progress", {})) != progress_roles:
        raise ValueError("Response progress heads differ from the requested roles")
    for role in progress_roles:
        output = result["progress"][role]
        _rows([output["values"]], 1, 8, f"{role} progress")
        if any(not 0 <= value <= 1 for value in output["values"]):
            raise ValueError("Progress must be in [0, 1]")
        _rows(output["bin_probabilities"], 8, 10, f"{role} bins", probabilities=True)
        if output["poi_score"] != output["values"][request.poi_slot]:
            raise ValueError("Selected progress does not match its slot")
    if "C" in roles:
        stages = result["stage"]
        count = len(request.stage_prompts)
        ids = stages["ids"]
        if (not isinstance(ids, list) or len(ids) != 8
                or any(type(value) is not int or not 0 <= value < count for value in ids)):
            raise ValueError("Stage IDs must preserve all eight valid task-local predictions")
        _rows(stages["logits"], 8, count, "stage logits")
        _rows(stages["probabilities"], 8, count, "stage probabilities", probabilities=True)
        focus = ids[request.poi_slot]
        if stages["poi_stage_id"] != focus:
            raise ValueError("Selected C stage does not match its slot")
        if "D" in roles and (result["D_focus_stage"] != focus or result["progress"]["D"]["focus_stage"] != focus):
            raise ValueError("D focus must equal C at the selected slot")
    elif "stage" in result or "D_focus_stage" in result:
        raise ValueError("A-only benchmark received an unrequested C/D result")
    if result.get("selected_prediction") != select_prediction(result):
        raise ValueError("Selected prediction metadata disagrees with the full arrays")
    return {"checkpoint": checkpoint, "image_preprocessing": result["image_preprocessing"],
            "view_policy": result["view_policy"], "max_pixels": result["max_pixels"]}


def summarize(samples: list[dict]) -> dict:
    """Mean and linearly interpolated P50/P95, with warmup samples kept separate."""
    if not samples:
        return {}
    summary = {}
    for key in (*TRANSPORT_METRICS, "client_total_ms", "server_handler_ms"):
        values = sorted(sample[key] for sample in samples)

        def quantile(q):
            index = (len(values) - 1) * q
            low, high = math.floor(index), math.ceil(index)
            return values[low] + (values[high] - values[low]) * (index - low)

        summary[key] = {"mean": statistics.fmean(values), "p50": quantile(.5), "p95": quantile(.95)}
    return summary


def source_identity() -> dict:
    def git(*args):
        completed = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=False)
        return completed.stdout.strip() if completed.returncode == 0 else None

    status = git("status", "--porcelain", "--untracked-files=normal")
    return {"scope": "local client checkout; server code revision is not exposed by the HTTP API",
            "git_commit": git("rev-parse", "HEAD"), "git_dirty": bool(status) if status is not None else None,
            "sha256": {name: file_digest(ROOT / name) for name in SOURCE_FILES},
            "python": sys.version, "platform": platform.platform(), "numpy": np.__version__}


def run(args) -> dict:
    roles = tuple(args.roles)
    if roles not in ROLE_SELECTIONS:
        raise ValueError("Benchmark roles must be A, C, or C D")
    if args.warmup < 0 or args.repeats <= 0 or not math.isfinite(args.timeout) or args.timeout <= 0:
        raise ValueError("Require warmup >= 0, repeats > 0 and a positive finite timeout")
    output = args.output.expanduser()
    if output.exists():
        raise FileExistsError(f"Use a fresh benchmark output: {output}")
    request, identity, geometry = prepare_input(args.input, args.frame_index, args.prediction_frame, roles)
    source = source_identity()
    created = datetime.now(timezone.utc).isoformat()
    run_id = uuid.uuid4().hex
    warmups, measured, baseline, sample_result = [], [], None, None
    for index in range(args.warmup + args.repeats):
        request_id = f"bench-{run_id}-{index}"
        timing = {}
        started = time.perf_counter_ns()
        response = client.predict(args.server, request, roles=roles, request_id=request_id, timeout=args.timeout,
                                  timing=timing)
        elapsed = (time.perf_counter_ns() - started) / 1_000_000
        transport = {key: _finite(timing.get(key), f"client {key} time") for key in TRANSPORT_METRICS}
        if any(value < 0 for value in transport.values()):
            raise ValueError("Client transport time must be nonnegative")
        receipt = validate_response(response, request, roles, args.prediction_frame, request_id)
        if baseline is None:
            baseline = receipt
        elif receipt != baseline:
            raise ValueError("Checkpoint, runtime or preprocessing changed during the benchmark")
        result = response["result"]
        predictions = {key: result[key] for key in ("progress", "stage", "selected_prediction") if key in result}
        sample = {"request_id": request_id, "client_total_ms": elapsed,
                  **transport,
                  "server_handler_ms": response["elapsed_seconds"] * 1000,
                  "prediction_sha256": json_digest(predictions)}
        if index < args.warmup:
            warmups.append(sample)
        else:
            measured.append(sample)
            if sample_result is None:
                sample_result = result
    report = {"schema": "grana.http_benchmark.v3", "created_utc": created, "server": args.server,
              "roles": list(roles), "wire_format": client.WIRE_BINARY, "geometry": geometry,
              "input": identity, "model": baseline, "client_source": source,
              "timing": {"unit": "milliseconds", "primary_metric": "client_send_to_receive_ms",
                         "boundary": "first request socket.sendall entry through response.read completion, before JSON parsing",
                         "includes": ["complete HTTP request headers and body transmission", "server handling",
                                      "server response serialization", "complete HTTP response transfer"],
                         "excludes": ["binary packing", "HTTP Request construction", "DNS/TCP/TLS/proxy connection setup",
                                      "response JSON parsing",
                                      "input file loading", "source and input hashing", "validation", "rendering",
                                      "server startup", "warmup"],
                         "client_total_boundary": "public client.predict call, including binary packing and response JSON parsing",
                         "http_round_trip_boundary": "HTTP opener start through response.read return, including connection setup; before JSON parsing",
                         "send_complete_boundary": "final request socket.sendall return; client-side completion, not server acknowledgement",
                         "server_handler_boundary": "server POST handler start through prediction; before response serialization",
                         "server_handler_relation": "contained within client_send_to_receive_ms and http_round_trip_ms; do not add the durations",
                         "historical_client_total_ms": "v1 measured the full public client.predict call, including packing and parsing; it is a different metric",
                         "percentiles": "linear interpolation over sorted samples"},
              "warmup": {"count": len(warmups), "server_cold_state": "unknown; externally managed server is not reset",
                         "samples": warmups, "summary": summarize(warmups)},
              "measured": {"count": len(measured), "samples": measured, "summary": summarize(measured)},
              "sample_result": sample_result}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return report


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", required=True)
    parser.add_argument("--input", type=Path, required=True,
                        help="Full .npy video request JSON without frozen frame_indices")
    parser.add_argument("--frame-index", type=int, required=True, help="Source frame to query using the evaluation CTSP geometry")
    parser.add_argument("--roles", nargs="+", choices=("A", "C", "D"), default=["C", "D"],
                        help="A, C, or C D (default)")
    parser.add_argument("--prediction-frame", choices=("poi", "last"), default="poi")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = run(args)
    print(json.dumps({"output": str(args.output), "geometry": report["geometry"],
                      "summary": report["measured"]["summary"]}, allow_nan=False))


if __name__ == "__main__":
    main()
