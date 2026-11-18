# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""An explicit local RUN_SPECS list: one or more --config files, in order."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from grana.config import load_config, physical_batch_size, supervision_budget
from grana.runtime import require_owned_environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, action="append", type=Path)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.dry_run:
        require_owned_environment()
    configs = [(path.resolve(), load_config(path, args.set)) for path in args.config]
    for path, cfg in configs:
        if args.dry_run:
            print(json.dumps({"config_file": str(path), "physical_batch_size": physical_batch_size(cfg),
                              "supervision_budget": supervision_budget(cfg), "config": cfg}, indent=2))
            continue
        command = [sys.executable, str(ROOT / "scripts/ray_migration/submit_local_ray_job.py"), "--config", str(path)]
        for value in args.set:
            command.extend(["--set", value])
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
