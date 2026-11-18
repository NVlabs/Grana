#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded stage-first annotation. Planning is offline; --run submits API requests.

Stages alone are a complete result. --with-quality adds a second round inside
each frozen stage, split further into short clips. Reuse the same output and
arguments to resume completed requests or add AQ later. All requests share one
persisted attempt budget. No uncertain request is automatically retried.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grana.annotation.api import KEY_ENV, AnnotationAPIError
from grana.annotation.rounds import DEFAULTS, read, run_episode, split
from grana.annotation.source import LeRobotSource


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "settings", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--run", action="store_true", help="Explicitly submit; without this flag, plan offline")
    parser.add_argument("--with-quality", action="store_true", help="After stages complete, annotate AQ within each stage")
    parser.add_argument("--policy-dir", type=Path)
    parser.add_argument("--template", dest="template_path", type=Path, help="Optional human reference template")
    parser.add_argument("--api-dir", type=Path)
    parser.add_argument("--request-name", default="episode")
    parser.add_argument("--max-requests", type=int, default=24)
    parser.add_argument("--model", help="Provider model name; otherwise GRANA_ANNOTATION_MODEL")
    parser.add_argument("--base-url", help="Responses API base URL; otherwise GRANA_ANNOTATION_BASE_URL")
    parser.add_argument("--api-key-env", default=KEY_ENV)
    parser.add_argument("--reasoning-effort", choices=("high", "xhigh"), default="xhigh")
    parser.add_argument("--timeout-seconds", type=float, default=480)
    parser.add_argument("--max-output-tokens", type=int, default=16384)
    for key, default in DEFAULTS.items():
        parser.add_argument("--" + key.replace("_", "-"), type=type(default), default=default)
    args = vars(parser.parse_args(argv))
    run = args.pop("run")
    try:
        args["settings"] = read(args["settings"])
        if run:
            result = run_episode(**args)
        else:
            ep = LeRobotSource(args["source"], args["settings"]["views"], allow_shared_videos=True).load_episode(args["episode"])
            cores = split(ep.time_edges, 0, ep.frame_count, args["stage_seconds"], args["stage_stride"], args["max_images"])
            result = {"status": "planned", "http_requests": 0, "stage_requests": len(cores),
                      "stage_cores": cores, "with_quality": args["with_quality"],
                      "quality_requests": "depends on completed stages" if args["with_quality"] else 0,
                      "limits": {k: args[k] for k in DEFAULTS}, "max_http_attempts": args["max_requests"]}
        print(json.dumps(result, indent=2))
    except (ValueError, OSError, TypeError, KeyError, AnnotationAPIError) as error:
        parser.exit(2, f"annotate_episode: {error}\n")


if __name__ == "__main__":
    main()
