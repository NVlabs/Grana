#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compile explicit task/stage cards into API-visible annotation settings."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grana.annotation.task_cards import compile_task_settings


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cards", type=Path, default=ROOT / "configs/annotation_task_cards.json")
    parser.add_argument("--profile", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--views", type=Path, required=True, help="JSON mapping view names to LeRobot camera features")
    parser.add_argument("--owner-policy", type=Path,
                        help="Optional explicit task-owner override; never inferred from a task name")
    parser.add_argument("--output", type=Path, required=True, help="Fresh settings JSON; existing files are never overwritten")
    args = parser.parse_args(argv)
    try:
        read = lambda path: json.loads(path.read_text())
        settings = compile_task_settings(read(args.cards), args.profile, args.task, read(args.views),
                                         owner_policy=read(args.owner_policy) if args.owner_policy else None)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            json.dump(settings, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
        print(json.dumps({"status": "compiled", "output": str(args.output),
                          "task_policy_version": settings["task_policy"]["version"]}))
    except (ValueError, TypeError, KeyError, OSError) as error:
        parser.exit(2, f"compile_annotation_settings: {error}\n")


if __name__ == "__main__":
    main()
