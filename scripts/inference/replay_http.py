#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Record A/C/C,D × POI/LAST HTTP replays using shared evaluation frame sampling.

Ground truth is read from a separate display metadata file and is never sent
to the model. Render the resulting manifest with render_http_replay.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import time
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from grana.data.frame_sampling import DEFAULT_EVAL_STRIDE_PAIRS
from grana.inference.cli import load_request
from grana.inference.client import build_binary, json_loads, predict
from grana.inference.presentation import prediction_chart_data

ROLE_CASES = (("A",), ("C",), ("C", "D"))
FRAME_CASES = ("poi", "last")


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def uniform_targets(num_frames: int, samples: int) -> list[int]:
    if type(num_frames) is not int or type(samples) is not int or not 2 <= samples <= num_frames:
        raise ValueError("Replay needs 2 <= samples <= number of source frames")
    return [index * (num_frames - 1) // (samples - 1) for index in range(samples)]


def load_episode(input_path: Path, metadata_path: Path) -> tuple[dict, dict]:
    """Validate source arrays and display-only labels before making requests."""
    import numpy as np

    raw = json_loads(input_path.read_text())
    if "frame_indices" in raw:
        raise ValueError("Replay input must contain full .npy videos, without frozen frame_indices")
    # Validate the same public request path used by clients, including camera order.
    sample = load_request(input_path, frame_index=0, prediction_frame="last")
    if not sample.task or not sample.compressed_task_prompt or not sample.stage_prompts:
        raise ValueError("Six-mode replay needs task, stage_prompts and compressed_task_prompt")
    metadata = json_loads(metadata_path.read_text())
    count, fps = metadata.get("num_frames"), metadata.get("source_fps")
    if type(count) is not int or count < 2:
        raise ValueError("Display metadata needs num_frames >= 2")
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not math.isfinite(fps) or fps <= 0:
        raise ValueError("Display metadata needs a positive finite source_fps")
    identity = metadata.get("trajectory_id")
    if not isinstance(identity, str) or not identity.strip():
        raise ValueError("Display metadata needs a trajectory_id")
    note = metadata.get("A_accuracy_note")
    if note is not None and (not isinstance(note, str) or not 1 <= len(note) <= 400):
        raise ValueError("Optional A_accuracy_note must contain 1..400 text characters")
    for key in ("whole_progress", "stage_ids", "within_stage_progress"):
        values = metadata.get(key)
        if not isinstance(values, list) or len(values) != count:
            raise ValueError(f"Display metadata {key} must have one value per source frame")
        if key == "stage_ids":
            if any(type(v) is not int or not 0 <= v < len(sample.stage_prompts) for v in values):
                raise ValueError("Display stage_ids must be valid task-local stage IDs")
        elif any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                 or not 0 <= v <= 1 for v in values):
            raise ValueError(f"Display {key} must contain finite values in [0,1]")
    paths = {}
    shapes = {}
    for view, source in raw["views"].items():
        path = Path(source).expanduser()
        path = path if path.is_absolute() else input_path.parent / path
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if array.dtype != np.uint8 or array.ndim != 4 or array.shape[0] != count or array.shape[-1] != 3:
            raise ValueError("Every camera must match the display metadata frame count and use RGB uint8")
        paths[view], shapes[view] = str(path.resolve()), list(array.shape)
    episode = {"trajectory_id": identity, "task": sample.task, "num_frames": count,
               "source_fps": fps, "view_names": list(paths), "frame_paths": paths, "frame_shapes": shapes,
               "stage_prompts": list(sample.stage_prompts), "compressed_task_prompt": sample.compressed_task_prompt}
    return episode, metadata


def validate_reply(envelope, request, roles, mode, request_id, weights_sha256):
    if envelope.get("request_id") != request_id:
        raise ValueError("Response request_id differs from the request")
    result = envelope["result"]
    if result["checkpoint"]["weights_sha256"] != weights_sha256:
        raise ValueError("Server checkpoint changed during replay")
    if result["roles"] != list(roles) or result["prediction_frame"] != mode:
        raise ValueError("Server roles or prediction frame differ from the replay case")
    if result["poi_slot"] != request.poi_slot or result.get("frame_indices") != list(request.frame_indices):
        raise ValueError("Server selected a different query slot or source window")
    selected = result["selected_prediction"]
    if selected["slot"] != request.poi_slot or selected["frame_index"] != request.frame_indices[request.poi_slot]:
        raise ValueError("Selected prediction does not identify the requested source frame")
    # Also checks complete eight-slot arrays and D's selected-frame focus.
    chart = prediction_chart_data(result, request)
    for series in chart["series"]:
        if series["role"] == "C":
            if selected["stage_id"] != series["selected_value"]:
                raise ValueError("Selected stage differs from the per-frame result")
        elif selected["progress"][series["role"]] != series["selected_value"]:
            raise ValueError("Selected progress differs from the per-frame result")
    return result


def run(args) -> Path:
    input_path, metadata_path = args.input.expanduser().resolve(), args.metadata.expanduser().resolve()
    episode, labels = load_episode(input_path, metadata_path)
    targets = uniform_targets(episode["num_frames"], args.samples)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.name_prefix):
        raise ValueError("name-prefix must contain only letters, digits, underscores or hyphens")
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Use a fresh replay output directory: {output}")
    with urlopen(args.server.rstrip("/") + "/healthz", timeout=30) as response:
        health = json.load(response)
    if not health.get("ready") or "grana-binary-v1" not in health.get("wire_formats", []):
        raise ValueError("Replay requires a ready binary HTTP server")
    checkpoint = health["checkpoint"]
    weights = checkpoint["weights_sha256"]
    if not checkpoint["inference_runtime"]["gpu_image_resize"]:
        raise ValueError("Replay requires the canonical GPU resize runtime")
    output.mkdir(parents=True)
    context_stride, tail_stride = DEFAULT_EVAL_STRIDE_PAIRS[0]
    manifest = {"schema": "grana-http-six-role-replays-v1", "model_label": args.model_label,
                "checkpoint": {"path": checkpoint["path"], "weights_sha256": weights,
                               "step": checkpoint["source_step"]},
                "sampling": {"context_stride": context_stride, "tail_stride": tail_stride,
                             "source_fps": episode["source_fps"], "target_frames": targets},
                "episode": episode, "cases": [],
                "ground_truth_policy": "Display only; never supplied to the server. Uniform targets do not use predictions.",
                "input_request": str(input_path), "display_metadata": str(metadata_path),
                "input_request_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
                "display_metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest()}
    if labels.get("A_accuracy_note") is not None:
        manifest["A_accuracy_note"] = labels["A_accuracy_note"]
    write_json(output / "health.json", health)
    for roles in ROLE_CASES:
        for mode in FRAME_CASES:
            name = f"{args.name_prefix}_{''.join(roles).lower()}_{mode}"
            records = []
            with (output / f"{name}.journal.jsonl").open("x") as journal:
                for index, source_frame in enumerate(targets):
                    request = load_request(input_path, frame_index=source_frame, prediction_frame=mode)
                    request_id = f"{name}-{index:03d}"
                    start = time.perf_counter()
                    reply = predict(args.server, request, roles=roles, request_id=request_id, timeout=args.timeout)
                    elapsed = (time.perf_counter() - start) * 1000
                    validate_reply(reply, request, roles, mode, request_id, weights)
                    if mode == "last" and max(request.frame_indices) > source_frame:
                        raise ValueError("Last-frame replay cannot observe a future source frame")
                    record = {"frame_index": source_frame, "frame_indices": list(request.frame_indices),
                              "slot": request.poi_slot, "request_id": request_id, "client_total_ms": elapsed,
                              "input_sha256": hashlib.sha256(build_binary(request, roles, request_id="identity")).hexdigest(),
                              "response": reply, "gt": {"whole_progress": labels["whole_progress"][source_frame],
                              "stage_id": labels["stage_ids"][source_frame],
                              "within_stage_progress": labels["within_stage_progress"][source_frame]}}
                    records.append(record)
                    journal.write(json.dumps(record, allow_nan=False) + "\n")
                    journal.flush()
                    if index == 0 or (index + 1) % 8 == 0 or index + 1 == len(targets):
                        print(json.dumps({"case": name, "completed": index + 1, "total": len(targets)}), flush=True)
            destination = output / f"{name}.json"
            write_json(destination, {"roles": list(roles), "prediction_frame": mode, "records": records})
            manifest["cases"].append({"name": name, "roles": list(roles), "prediction_frame": mode,
                                      "slot": 5 if mode == "poi" else 7, "records_path": str(destination),
                                      "records_sha256": hashlib.sha256(destination.read_bytes()).hexdigest()})
    path = output / "manifest.json"
    write_json(path, manifest)
    print(json.dumps({"manifest": str(path), "cases": len(manifest["cases"]),
                      "predictions": len(targets) * len(manifest["cases"])}), flush=True)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", required=True)
    parser.add_argument("--input", type=Path, required=True, help="Full .npy video request; no frame_indices")
    parser.add_argument("--metadata", type=Path, required=True,
                        help="Display-only trajectory_id, source_fps, num_frames and whole_progress/stage_ids/within_stage_progress arrays")
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--name-prefix", default="replay")
    parser.add_argument("--model-label", default="Grana checkpoint | binary HTTP")
    parser.add_argument("--timeout", type=float, default=1200)
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
