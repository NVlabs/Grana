#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export 1 to 5 reviewed human examples as one template, or prepare a different same-task target (offline)."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grana.annotation.templates import export_template, prepare_target


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export", help="Freeze 1 to 5 submitted, review-confirmed canonical human labels and sampled RGB")
    export.add_argument("--config", type=Path, required=True, help="Active human-label YAML recipe")
    export.add_argument("--episode", dest="episodes", type=int, action="append", required=True,
                        help="A seed episode; repeat it for up to 5 seeds, sent in this order")
    export.add_argument("--criteria", type=Path, help="Optional declared JSON {version,rules}; no criteria are inferred by a model")
    prepare = commands.add_parser("prepare", help="Prepare target evidence from the template; never copies seed intervals")
    prepare.add_argument("--template", type=Path, required=True, help="Private template directory or template.json")
    prepare.add_argument("--source", type=Path, required=True)
    prepare.add_argument("--task", help="Explicit canonical task binding; required if source episode task metadata is absent")
    prepare.add_argument("--source-task", help="Exact source metadata task string explicitly declared as an alias of this same task")
    prepare.add_argument("--views", type=Path, help="JSON mapping of the same ordered template camera names to target feature keys")
    prepare.add_argument("--episode", type=int, required=True)
    prepare.add_argument("--start-frame", type=int, default=0)
    prepare.add_argument("--end-frame", type=int)
    for command, stride, limit, images in ((export, 2.0, 64, "Total over all seeds"), (prepare, .5, 96, None)):
        command.add_argument("--output", type=Path, required=True, help="Fresh private directory outside source and code checkouts")
        command.add_argument("--stride-seconds", type=float, default=stride)
        command.add_argument("--max-images", type=int, default=limit, help=images)
        command.add_argument("--tile-size", type=int, nargs=2, default=(320, 240), metavar=("WIDTH", "HEIGHT"))
        command.add_argument("--policy-dir", type=Path, help="Frozen policy; preparation defaults to the policy embedded in the template")
    args = vars(parser.parse_args(argv))
    command = args.pop("command")
    try:
        if command == "export":
            args["config_path"] = args.pop("config")
            if args["criteria"] is not None:
                args["criteria"] = json.loads(args["criteria"].read_text())
            result = export_template(**args)
        else:
            args["template_path"] = args.pop("template")
            if args["views"] is not None:
                args["views"] = json.loads(args["views"].read_text())
            result = prepare_target(**args)
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    except (ValueError, TypeError, KeyError, OSError, subprocess.SubprocessError) as error:
        parser.exit(2, f"annotation_template: {error}\n")


if __name__ == "__main__":
    main()
