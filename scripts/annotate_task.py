#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare a same-task campaign offline; --run explicitly submits its frozen jobs.

Manifest: {"schema_version":"grana.annotation_targets.v1", "task":"task-id",
"targets":[{"name":"day2", "source":"../raw/day2", "episodes":[0,1]}]}.
Each target optionally supplies views (same ordered view names, new camera keys)
and source_task (an exact metadata task alias). Relative sources use the manifest
directory. --plan reads and validates all sources without writing or calling API.
Default preparation requires a fresh output. --run requires completed unchanged
preparation; it resumes identical completed requests without another API call.
Every request carries all 1 to 5 seeds of the template.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grana.annotation.api import KEY_ENV
from grana.annotation.templates import campaign_plan, prepare_campaign, run_campaign


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("template", "manifest", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan", action="store_true", help="Read-only source/task/image-count validation; no API")
    mode.add_argument("--run", action="store_true", help="Submit an already prepared identical campaign through the existing API batch runner")
    parser.add_argument("--stride-seconds", type=float, default=.5)
    parser.add_argument("--max-images", type=int, default=96)
    parser.add_argument("--tile-size", type=int, nargs=2, default=(320, 240), metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--policy-dir", type=Path, help="Defaults to the policy frozen inside the template")
    parser.add_argument("--max-requests", type=int, default=24, help="One persisted HTTP-attempt budget for the whole campaign, including retries")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--model", help="Provider model name; otherwise GRANA_ANNOTATION_MODEL")
    parser.add_argument("--base-url", help="Responses API base URL; otherwise GRANA_ANNOTATION_BASE_URL")
    parser.add_argument("--api-key-env", default=KEY_ENV, help="Environment variable NAME holding a credential; never the credential itself")
    parser.add_argument("--reasoning-effort", choices=("high", "xhigh"), default="xhigh")
    parser.add_argument("--max-output-tokens", type=int, default=32768)
    parser.add_argument("--timeout-seconds", type=float, default=480)
    parser.add_argument("--max-attempts", type=int, default=1)
    parser.add_argument("--retry-uncertain", action="store_true")
    args = vars(parser.parse_args(argv))
    plan_only, submit = args.pop("plan"), args.pop("run")
    args["template_path"], args["manifest_path"] = args.pop("template"), args.pop("manifest")
    api_names = ("max_requests", "workers", "model", "base_url", "api_key_env", "reasoning_effort",
                 "max_output_tokens", "timeout_seconds", "max_attempts", "retry_uncertain")
    api = {key: args.pop(key) for key in api_names}
    try:
        if plan_only:
            plan = campaign_plan(**args)
            result = {"status": "planned", "task": plan["task"], "template_id": plan["template_id"],
                      "output": plan["output"], "seeds": plan["seeds"],
                      "jobs": [{"name": job["name"], "target_images": len(job["evidence_frames"]),
                                "seed_images": plan["seed_images"],
                                "total_images": plan["seed_images"] + len(job["evidence_frames"])} for job in plan["jobs"]],
                      "planned_annotation_requests": len(plan["jobs"]), "http_requests": 0}
        elif submit:
            result = run_campaign(**args, **api)
        else:
            result = prepare_campaign(**args)
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    except (ValueError, TypeError, KeyError, OSError, subprocess.SubprocessError) as error:
        parser.exit(2, f"annotate_task: {error}\n")
    if result.get("status") == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
