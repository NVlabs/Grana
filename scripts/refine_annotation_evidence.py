#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare bounded extra evidence from one validated API candidate; no API calls."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grana.annotation.refinement import digest, plan_refinement
from grana.annotation.source import LeRobotSource
from scripts.annotation_skill import POLICY_DIR, _read_json, _write_new, prepare_bundle
from scripts.annotation_skill_api import prepare_request
from scripts.prepare_annotation_evidence import select_frames


def run(baseline_dir, candidate_dir, settings, output, *, policy_dir=POLICY_DIR,
        max_extra_frames=64, max_total_images=256, radius_seconds=1.0,
        dense_hz=10.0, seed="grana-refinement-v1", resume=False):
    """Verify source, baseline delivery and candidate, then copy grid/decode extras.

    ``settings`` is the same six-field object used by prepare_annotation_evidence.
    Existing completed output may only be reused with ``resume=True`` and exact
    input/parameter/artifact matches. Incomplete outputs are not overwritten.
    """
    baseline_dir = Path(baseline_dir).expanduser().resolve()
    candidate_dir = Path(candidate_dir).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    bundle, _, verified = prepare_request(baseline_dir / "bundle.json", baseline_dir / "evidence.json", policy_dir)
    candidate = _read_json(candidate_dir / "candidate.json")
    validation = _read_json(candidate_dir / "validation.json")
    delivery = _read_json(candidate_dir / "delivery.json")
    source_receipt = _read_json(baseline_dir / "source_receipt.json")
    plan = plan_refinement(bundle, candidate, validation, source_receipt, policy_dir=policy_dir,
                           max_extra_frames=max_extra_frames, max_total_images=max_total_images,
                           radius_seconds=radius_seconds, dense_hz=dense_hz, seed=seed)
    if (delivery.get("status") != "completed" or delivery.get("accepted_for_training") is not False
            or delivery.get("bundle_sha256") != digest(bundle)
            or delivery.get("context_sha256") != bundle["context_sha256"]
            or delivery.get("candidate_sha256") != digest(candidate)
            or delivery.get("frame_order") != plan["baseline_frames"]
            or delivery.get("images") != verified):
        raise ValueError("Completed baseline delivery does not bind the exact bundle, candidate and evidence")
    required = {"views", "task", "task_instruction", "stage_semantics", "task_policy", "registry"}
    context = bundle["context"]
    if (not isinstance(settings, dict) or set(settings) != required
            or any(settings[k] != context[k] for k in required - {"views"})):
        raise ValueError("Settings must retain the original task, registry, semantics and policy")
    identity = context["source_identity"]
    source = Path(identity["source_root"]).expanduser().resolve()
    if (output.is_relative_to(ROOT) or output.is_relative_to(source)
            or any((p / ".git").exists() for p in (output, *output.parents))
            or output == baseline_dir or output == candidate_dir):
        raise ValueError("Expanded evidence output must be outside code checkouts, raw source and original inputs")
    reader = LeRobotSource(source, settings["views"], allow_shared_videos=True)
    episode = reader.load_episode(identity["episode_index"])
    start, end = plan["source_frame_range"]
    expected_identity = episode.source_identity | {"source_frame_range": [start, end],
        "frame_mapping": "native_frame = candidate_frame + source_frame_range[0]"}
    if (identity != expected_identity or source_receipt.get("clock") != episode.clock_receipt
            or context["time_edges"] != list(episode.time_edges[start:end + 1])
            or any(row["view_names"] != list(episode.view_names) for row in verified)):
        raise ValueError("Raw source, views or native clock changed since baseline evidence preparation")
    sampling = source_receipt["sampling"]
    original_frames = select_frames(episode.time_edges, start, end, sampling["stride_seconds"],
                                    sampling["max_images"], extra_frames=sampling.get("extra_source_frames", []))
    if original_frames != sampling["source_frames"]:
        raise ValueError("Baseline source receipt does not reproduce its fixed grid and explicit extras")
    expanded_context = deepcopy(context)
    expanded_context["evidence_frames"] = plan["expanded_frames"]
    expanded_bundle = prepare_bundle(expanded_context, policy_dir, mode=bundle.get("mode", "combined"))
    binding = {"baseline_bundle_sha256": digest(bundle), "baseline_evidence_sha256": digest(verified),
               "baseline_delivery_sha256": digest(delivery), "plan_sha256": digest(plan),
               "settings_sha256": digest(settings), "expanded_bundle_sha256": digest(expanded_bundle)}

    def verify_output():
        frozen, _, images = prepare_request(output / "bundle.json", output / "evidence.json", policy_dir)
        if frozen != expanded_bundle:
            raise ValueError("Expanded bundle differs from the expected immutable context")
        original_hashes = {row["frame"]: row["sha256"] for row in verified}
        if any(row["sha256"] != original_hashes[row["frame"]] for row in images if row["frame"] in original_hashes):
            raise ValueError("Expanded evidence changed an original mosaic")
        return images

    if output.exists():
        if not resume:
            raise FileExistsError("Output exists; use --resume for exact completed outputs or a fresh directory")
        receipt = _read_json(output / "refinement_receipt.json")
        if receipt.get("binding") != binding or _read_json(output / "refinement_plan.json") != plan:
            raise ValueError("Existing refinement has different inputs or parameters")
        images = verify_output()
        if (receipt.get("expanded_evidence_sha256") != digest(images)
                or receipt.get("source_receipt_sha256") != digest(_read_json(output / "source_receipt.json"))):
            raise ValueError("Existing refinement artifacts differ from their completion receipt")
        return {"status": "prepared", "output": str(output), "resumed": True, **plan["budget"]}

    output.mkdir(parents=True, exist_ok=False)
    _write_new(output / "refinement_plan.json", plan)
    image_dir = output / "images"
    image_dir.mkdir()
    records = []
    for row in verified:
        original_path = Path(row["image_path"])
        destination = image_dir / f"baseline_frame_{row['frame']:06d}{original_path.suffix}"
        shutil.copyfile(original_path, destination)
        if hashlib.sha256(destination.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError("Original mosaic changed while copying")
        records.append({k: row[k] for k in ("frame", "timestamp", "sha256", "view_names")} |
                       {"image_path": str(destination)})
    if plan["extra_source_frames"]:
        records.extend(reader.make_evidence(episode, plan["extra_source_frames"], image_dir,
                                            sampling["tile_size"], frame_offset=start))
    records.sort(key=lambda row: row["frame"])
    _write_new(output / "bundle.json", expanded_bundle)
    manifest = {"schema_version": "grana.annotation_evidence.v1", "images": records}
    if expanded_bundle.get("mode"):
        manifest.update(schema_version="grana.annotation_evidence.v2", context_sha256=expanded_bundle["context_sha256"])
    _write_new(output / "evidence.json", manifest)
    expanded_source = deepcopy(source_receipt)
    expanded_source["sampling"].update(max_images=max_total_images,
        source_frames=[f + start for f in plan["expanded_frames"]], candidate_frames=plan["expanded_frames"],
        extra_source_frames=sorted(set(sampling.get("extra_source_frames", [])) | set(plan["extra_source_frames"])),
        extra_candidate_frames=sorted(set(sampling.get("extra_candidate_frames", [])) | set(plan["extra_frames"])))
    expanded_source["refinement"] = {"plan_sha256": digest(plan), "baseline_grid_and_bytes_preserved": True,
                                     "human_annotations_read": False}
    _write_new(output / "source_receipt.json", expanded_source)
    images = verify_output()
    _write_new(output / "refinement_receipt.json", {"schema_version": "grana.annotation_refinement_receipt.v1",
        "binding": binding, "expanded_evidence_sha256": digest(images),
        "source_receipt_sha256": digest(expanded_source), "original_images_preserved": len(verified),
        "extra_images": len(plan["extra_frames"]), "human_annotations_read": False,
        "api_calls_made": 0, "accepted_for_training": False})
    return {"status": "prepared", "output": str(output), "resumed": False, **plan["budget"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True, help="Original bundle/evidence/source_receipt directory")
    parser.add_argument("--candidate-dir", type=Path, required=True, help="Completed candidate/validation/delivery directory")
    parser.add_argument("--settings", type=Path, required=True, help="Original preparation settings JSON")
    parser.add_argument("--output", type=Path, required=True, help="Fresh private evidence directory outside checkouts/raw source")
    parser.add_argument("--policy-dir", type=Path, default=POLICY_DIR)
    parser.add_argument("--max-extra-frames", type=int, default=64)
    parser.add_argument("--max-total-images", type=int, default=256)
    parser.add_argument("--radius-seconds", type=float, default=1.0)
    parser.add_argument("--dense-hz", type=float, default=10.0)
    parser.add_argument("--seed", default="grana-refinement-v1")
    parser.add_argument("--resume", action="store_true", help="Reuse only an exact, fully verified completed preparation")
    args = vars(parser.parse_args(argv))
    try:
        args["settings"] = _read_json(args["settings"])
        print(json.dumps(run(**args), indent=2))
    except (ValueError, TypeError, KeyError, OSError) as error:
        parser.exit(2, f"refine_annotation_evidence: {error}\n")


if __name__ == "__main__":
    main()
