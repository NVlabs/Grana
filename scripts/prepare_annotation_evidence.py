#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare clock-checked LeRobot RGB evidence; default to first-round stages."""
from __future__ import annotations

import argparse
import bisect
from fractions import Fraction
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from annotation_skill import POLICY_DIR, _read_json, _write_new, prepare_bundle
from grana.annotation.source import LeRobotSource


def select_frames(time_edges, start, end, stride_seconds, max_images, *, extra_frames=None):
    """Native-time grid and clip endpoints, union explicit native frame IDs."""
    try:
        valid_clock = (isinstance(time_edges, (list, tuple)) and len(time_edges) >= 2
                       and all(not isinstance(value, bool) and isinstance(value, (int, float))
                               and math.isfinite(value) for value in time_edges)
                       and all(a < b for a, b in zip(time_edges, time_edges[1:])))
        valid_stride = (not isinstance(stride_seconds, bool) and isinstance(stride_seconds, (int, float))
                        and math.isfinite(stride_seconds) and stride_seconds > 0)
    except OverflowError:
        raise ValueError("Native clock and stride_seconds must contain finite numbers") from None
    if not valid_clock:
        raise ValueError("time_edges must contain N+1 finite strictly increasing native times")
    if (isinstance(start, bool) or not isinstance(start, int)
            or isinstance(end, bool) or not isinstance(end, int)
            or not 0 <= start < end <= len(time_edges) - 1):
        raise ValueError("Expected 0 <= start_frame < end_frame <= source frame count")
    if not valid_stride:
        raise ValueError("stride_seconds must be positive and finite")
    if isinstance(max_images, bool) or not isinstance(max_images, int) or max_images < 1:
        raise ValueError("max_images must be a positive integer")
    extras = [] if extra_frames is None else extra_frames
    if (not isinstance(extras, (list, tuple))
            or any(isinstance(frame, bool) or not isinstance(frame, int) for frame in extras)):
        raise ValueError("extra_frames must be a list of integer native frame IDs (not bool)")
    if len(set(extras)) != len(extras):
        raise ValueError("extra_frames must contain unique native frame IDs")
    if any(not start <= frame < end for frame in extras):
        raise ValueError("extra_frames must lie within [start_frame, end_frame)")
    selected = {start, end - 1, *extras}
    if len(selected) > max_images:
        raise ValueError("Evidence union exceeds max_images; no explicit frames were dropped")
    # Anchor every target to the clip start, not the previously selected frame:
    # native-grid rounding must not accumulate into sampling drift. Decimal
    # ratios avoid float overflow/underflow and targets such as 3 * .1 skipping
    # an actual .3-second frame. Skip targets already covered by the last frame,
    # so even a subnormal stride takes at most one iteration per selected frame.
    clock_value = lambda value: Fraction(str(value))
    origin, stride = clock_value(time_edges[start]), clock_value(stride_seconds)
    frame = start
    while frame < end - 1:
        tick = (clock_value(time_edges[frame]) - origin) // stride + 1
        target = origin + tick * stride
        frame = bisect.bisect_left(time_edges, target, lo=frame + 1, hi=end, key=clock_value)
        if frame < end:
            selected.add(frame)
        if len(selected) > max_images:
            raise ValueError("Evidence exceeds max_images; shorten the clip or increase stride_seconds")
    if len(selected) > max_images:
        raise ValueError("Evidence exceeds max_images")
    return sorted(selected)


def prepare(source, episode, settings, output, *, start_frame=0, end_frame=None,
            stride_seconds=1.0, max_images=96, tile_size=(320, 240), policy_dir=POLICY_DIR,
            extra_frames=None, mode="stages"):
    required = {"views", "task", "task_instruction", "stage_semantics", "task_policy", "registry"}
    if not isinstance(settings, dict) or set(settings) != required:
        raise ValueError(f"Settings must contain exactly {sorted(required)}")
    reader = LeRobotSource(Path(source), settings["views"], allow_shared_videos=True)
    ep = reader.load_episode(episode)
    end = ep.frame_count if end_frame is None else end_frame
    indices = select_frames(ep.time_edges, start_frame, end, stride_seconds, max_images,
                            extra_frames=extra_frames)
    extras = sorted(extra_frames) if extra_frames is not None else []
    identity = {**ep.source_identity, "source_frame_range": [start_frame, end],
                "frame_mapping": "native_frame = candidate_frame + source_frame_range[0]"}
    context = {key: settings[key] for key in required - {"views"}}
    context.update(schema_version="grana.annotation_context.v1", source_identity=identity,
                   time_edges=list(ep.time_edges[start_frame:end + 1]),
                   evidence_frames=[i - start_frame for i in indices])
    if mode not in ("stages", "combined"):
        raise ValueError("Evidence preparation mode must be stages or combined")
    bundle = prepare_bundle(context, policy_dir, mode=mode)
    output = Path(output).expanduser().resolve()
    source = Path(source).expanduser().resolve()
    if (output.is_relative_to(ROOT) or output.is_relative_to(source)
            or any((p / ".git").exists() for p in (output, *output.parents))):
        raise ValueError("Evidence output must be outside code checkouts and raw source")
    output.mkdir(parents=True, exist_ok=False)
    records = reader.make_evidence(ep, indices, output / "images", tile_size,
                                  frame_offset=start_frame)
    _write_new(output / "bundle.json", bundle)
    manifest = {"schema_version": "grana.annotation_evidence.v1", "images": records}
    if mode == "stages":
        manifest.update(schema_version="grana.annotation_evidence.v2", context_sha256=bundle["context_sha256"])
    _write_new(output / "evidence.json", manifest)
    _write_new(output / "source_receipt.json", {
        "source_identity": identity, "clock": ep.clock_receipt,
        "sampling": {"method": "native_time_stride_with_clip_endpoints", "stride_seconds": stride_seconds,
                     "grid_origin_seconds": ep.time_edges[start_frame], "max_images": max_images,
                     "source_frames": indices, "candidate_frames": context["evidence_frames"],
                     "extra_source_frames": extras,
                     "extra_candidate_frames": [frame - start_frame for frame in extras],
                     "extra_frames_rule": "Sorted union with fixed grid; overlapping grid frames count once; no drops.",
                     "tile_size": list(tile_size)},
        "annotation_columns_read": False, "accepted_for_training": False,
        "scope": "Only the declared source_frame_range; no evidence for the rest of the trajectory."})
    return {"status": "prepared", "output": str(output), "images": len(records),
            "source_frame_range": [start_frame, end], "context_sha256": bundle["context_sha256"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--settings", type=Path, required=True, help="JSON with task/policy/registry/views")
    parser.add_argument("--output", type=Path, required=True, help="Fresh private evidence directory outside checkouts")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int)
    parser.add_argument("--stride-seconds", type=float, default=1.0)
    parser.add_argument("--max-images", type=int, default=96)
    parser.add_argument("--extra-frames", type=Path,
                        help="JSON file of unique native frame IDs to union with the fixed time grid")
    parser.add_argument("--tile-size", type=int, nargs=2, default=(320, 240), metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--policy-dir", type=Path, default=POLICY_DIR)
    parser.add_argument("--mode", choices=("stages", "combined"), default="stages",
                        help="Default: stages only; combined is the legacy joint research workflow")
    args = parser.parse_args(argv)
    try:
        result = prepare(args.source, args.episode, _read_json(args.settings), args.output,
                         start_frame=args.start_frame, end_frame=args.end_frame,
                         stride_seconds=args.stride_seconds, max_images=args.max_images,
                         tile_size=args.tile_size, policy_dir=args.policy_dir,
                         extra_frames=_read_json(args.extra_frames) if args.extra_frames else None, mode=args.mode)
        print(json.dumps({key: value for key, value in result.items() if key != "context_sha256"}, indent=2))
    except (ValueError, OSError, TypeError) as error:
        parser.exit(2, f"prepare_annotation_evidence: {error}\n")


if __name__ == "__main__":
    main()
