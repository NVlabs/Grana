#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plan stage-based clips, or prepare one clip, from a completed coarse candidate.

Source-checkout workflow; no API calls or automatic result stitching. Required
--policy-dir identifies the coarse experiment's exact frozen policy snapshot.
Without --window-id, create a fresh bound plan. Then use --resume --window-id
window_0000 to prepare one window. Independent windows can be prepared separately.
Incomplete/tampered output is rejected, never silently overwritten or resumed.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grana.annotation.refinement import digest
from grana.annotation.source import LeRobotSource
from grana.annotation.windows import plan_stage_windows
from scripts.annotation_skill import _read_json, prepare_bundle, stored_validation_receipt, validate_bundle_candidate
from scripts.annotation_skill_api import prepare_request
from scripts.prepare_annotation_evidence import prepare, select_frames


def _atomic_new(path, document):
    """Publish a complete JSON file atomically without replacing an existing file."""
    path = Path(path)
    raw = json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    descriptor, name = tempfile.mkstemp(prefix=".pending-", suffix=".json", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)  # Atomic create, fails if another writer won.
    finally:
        temporary.unlink(missing_ok=True)


def run(baseline_dir, candidate_dir, settings, output, *, policy_dir, window_id=None,
        max_core_seconds=30.0, context_seconds=2.0, stride_seconds=0.5, max_images=128,
        resume=False):
    """Bind original artifacts, plan all cores, and optionally prepare one window."""
    baseline_dir = Path(baseline_dir).expanduser().resolve()
    candidate_dir = Path(candidate_dir).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    bundle, _, verified = prepare_request(baseline_dir / "bundle.json", baseline_dir / "evidence.json", policy_dir)
    context = bundle["context"]
    candidate = _read_json(candidate_dir / "candidate.json")
    validation = _read_json(candidate_dir / "validation.json")
    delivery = _read_json(candidate_dir / "delivery.json")
    receipt = _read_json(baseline_dir / "source_receipt.json")
    actual = validate_bundle_candidate(bundle, candidate, policy_dir)
    if digest(actual) != digest(stored_validation_receipt(validation, actual)):
        raise ValueError("Coarse validation receipt differs from the supplied bundle/candidate")
    if (delivery.get("status") != "completed" or delivery.get("accepted_for_training") is not False
            or delivery.get("bundle_sha256") != digest(bundle)
            or delivery.get("context_sha256") != bundle["context_sha256"]
            or delivery.get("candidate_sha256") != digest(candidate)
            or delivery.get("frame_order") != sorted(context["evidence_frames"])
            or delivery.get("images") != verified):
        raise ValueError("Completed coarse delivery must bind the exact bundle, candidate and evidence")
    required = {"views", "task", "task_instruction", "stage_semantics", "task_policy", "registry"}
    if (not isinstance(settings, dict) or set(settings) != required
            or any(settings[key] != context[key] for key in required - {"views"})):
        raise ValueError("Settings must preserve the coarse task, registry, semantics and policy")
    identity = context["source_identity"]
    if receipt.get("source_identity") != identity or receipt.get("annotation_columns_read") is not False:
        raise ValueError("Source receipt must bind the source identity and declare no annotation columns read")
    source = Path(identity["source_root"]).expanduser().resolve()
    if (output.is_relative_to(ROOT) or output.is_relative_to(source)
            or output.is_relative_to(baseline_dir) or output.is_relative_to(candidate_dir)
            or any((p / ".git").exists() for p in (output, *output.parents))):
        raise ValueError("Window output must be outside code checkouts, raw source and coarse input directories")
    reader = LeRobotSource(source, settings["views"], allow_shared_videos=True)
    episode = reader.load_episode(identity["episode_index"])
    # The pure planner rejects missing or ambiguous native-frame mapping.
    plan = plan_stage_windows(context, candidate, max_core_seconds=max_core_seconds,
                              context_seconds=context_seconds, stride_seconds=stride_seconds, max_images=max_images)
    start, end = plan["source_frame_range"]
    expected_identity = episode.source_identity | {"source_frame_range": [start, end],
        "frame_mapping": "native_frame = candidate_frame + source_frame_range[0]"}
    if (identity != expected_identity or receipt.get("clock") != episode.clock_receipt
            or context["time_edges"] != list(episode.time_edges[start:end + 1])
            or any(row["view_names"] != list(episode.view_names) for row in verified)):
        raise ValueError("Raw source, views or clock changed since coarse evidence preparation")
    sampling = receipt.get("sampling", {})
    original = select_frames(episode.time_edges, start, end, sampling["stride_seconds"], sampling["max_images"],
                             extra_frames=sampling.get("extra_source_frames", []))
    if (original != sampling.get("source_frames")
            or sampling.get("candidate_frames") != sorted(context["evidence_frames"])
            or original != [f + start for f in sorted(context["evidence_frames"])]
            or sampling.get("grid_origin_seconds") != context["time_edges"][0]):
        raise ValueError("Coarse source receipt does not reproduce the supplied evidence frame mapping")
    for window in plan["windows"]:
        first, last = window["source_frame_range"]
        selected = select_frames(episode.time_edges, first, last, stride_seconds, max_images)
        if selected != window["source_evidence_frames"]:
            raise ValueError("Window planner and evidence preparer disagree on native-time sampling")
    binding = {"bundle_sha256": digest(bundle), "evidence_sha256": digest(verified),
               "candidate_sha256": digest(candidate), "validation_sha256": digest(validation),
               "delivery_sha256": digest(delivery), "source_receipt_sha256": digest(receipt),
               "settings_sha256": digest(settings)}
    plan["binding"] = binding
    selected_window = None
    if window_id is not None:
        selected_window = next((row for row in plan["windows"] if row["window_id"] == window_id), None)
        if selected_window is None:
            raise ValueError("window_id is not in the derived plan")
    if output.exists():
        if not resume:
            raise FileExistsError("Plan output exists; use --resume for exact completed artifacts")
        if _read_json(output / "window_plan.json") != plan:
            raise ValueError("Existing window plan has different inputs or parameters")
    else:
        output.mkdir(parents=True, exist_ok=False)
        _atomic_new(output / "window_plan.json", plan)
    if selected_window is None:
        return {"status": "planned", "output": str(output), "windows": len(plan["windows"]),
                "core_partition_verified": True, "api_calls_made": 0, "accepted_for_training": False}

    window = selected_window
    destination = output / "windows" / window_id
    first, last = window["source_frame_range"]
    local_first, local_last = window["context_frame_range"]
    expected_context = deepcopy(context)
    expected_context["source_identity"] = episode.source_identity | {
        "source_frame_range": [first, last],
        "frame_mapping": "native_frame = candidate_frame + source_frame_range[0]"}
    expected_context["time_edges"] = context["time_edges"][local_first:local_last + 1]
    expected_context["evidence_frames"] = window["window_evidence_frames"]
    expected_bundle = prepare_bundle(expected_context, policy_dir, mode=bundle.get("mode", "combined"))
    window_binding = {"window_plan_sha256": digest(plan), "window_id": window_id,
                      "expected_bundle_sha256": digest(expected_bundle)}

    def verify_prepared():
        actual_bundle, _, images = prepare_request(destination / "bundle.json", destination / "evidence.json", policy_dir)
        source_receipt = _read_json(destination / "source_receipt.json")
        if actual_bundle != expected_bundle:
            raise ValueError("Prepared window bundle differs from its exact native clock, registry or policy")
        expected_sampling = {"method": "native_time_stride_with_clip_endpoints", "stride_seconds": stride_seconds,
            "grid_origin_seconds": episode.time_edges[first], "max_images": max_images,
            "source_frames": window["source_evidence_frames"], "candidate_frames": window["window_evidence_frames"],
            "extra_source_frames": [], "extra_candidate_frames": [],
            "extra_frames_rule": "Sorted union with fixed grid; overlapping grid frames count once; no drops.",
            "tile_size": sampling["tile_size"]}
        if (source_receipt.get("source_identity") != expected_context["source_identity"]
                or source_receipt.get("clock") != episode.clock_receipt
                or source_receipt.get("sampling") != expected_sampling
                or source_receipt.get("annotation_columns_read") is not False
                or source_receipt.get("accepted_for_training") is not False):
            raise ValueError("Prepared window source receipt differs from its expected source/sampling")
        return images, source_receipt

    def completion_receipt(images, source_receipt):
        return {"schema_version": "grana.annotation_stage_window_receipt.v1", "binding": window_binding,
            "evidence_sha256": digest(images), "source_receipt_sha256": digest(source_receipt),
            "window_local_core_range": window["window_local_core_range"],
            "source_core_frame_range": window["source_core_frame_range"],
            "stage_hypothesis_enforced": False, "stage_hint_sent_to_api_context": False,
            "outcome_scope": window["outcome_scope"], "api_calls_made": 0,
            "human_annotations_read": False, "accepted_for_training": False}

    if destination.exists():
        if not resume:
            raise FileExistsError("Window output already exists")
        completed = _read_json(destination / "window_receipt.json")
        if completed.get("binding") != window_binding:
            raise ValueError("Existing window completion receipt has different inputs")
        images, source_receipt = verify_prepared()
        if completed != completion_receipt(images, source_receipt):
            raise ValueError("Existing window completion receipt differs from expected metadata or artifacts")
        resumed = True
    else:
        prepare(source, episode.episode_index, settings, destination, start_frame=first, end_frame=last,
                stride_seconds=stride_seconds, max_images=max_images, tile_size=sampling["tile_size"], policy_dir=policy_dir,
                mode=bundle.get("mode", "combined"))
        images, source_receipt = verify_prepared()
        _atomic_new(destination / "window_receipt.json", completion_receipt(images, source_receipt))
        resumed = False
    return {"status": "prepared", "output": str(destination), "window_id": window_id,
            "source_frame_range": [first, last], "source_core_frame_range": window["source_core_frame_range"],
            "images": len(images), "resumed": resumed, "api_calls_made": 0, "accepted_for_training": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline-dir", required=True, type=Path)
    parser.add_argument("--candidate-dir", required=True, type=Path)
    parser.add_argument("--settings", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--policy-dir", required=True, type=Path)
    parser.add_argument("--window-id", help="Prepare exactly this derived window; omit to write the plan only")
    parser.add_argument("--max-core-seconds", type=float, default=30.0)
    parser.add_argument("--context-seconds", type=float, default=2.0)
    parser.add_argument("--stride-seconds", type=float, default=0.5)
    parser.add_argument("--max-images", type=int, default=128)
    parser.add_argument("--resume", action="store_true")
    args = vars(parser.parse_args(argv))
    try:
        args["settings"] = _read_json(args["settings"])
        print(json.dumps(run(**args), indent=2))
    except (ValueError, TypeError, KeyError, OSError) as error:
        parser.exit(2, f"prepare_annotation_windows: {error}\n")


if __name__ == "__main__":
    main()
