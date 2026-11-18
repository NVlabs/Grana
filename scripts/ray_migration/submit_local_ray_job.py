# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Launch an owned local Ray Train driver; no cloud or shared-cluster mutation."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from grana.config import load_config
from grana.runtime import require_owned_environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--set", action="append", default=[])
    args = parser.parse_args()
    require_owned_environment()
    config = load_config(args.config, args.set)
    output = Path(config["training"]["output_dir"]).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Use a fresh output directory; refusing to overwrite {output}")
    command = [sys.executable, str(ROOT / "scripts/ray_migration/train_ray.py"), "--config", str(args.config.resolve())]
    for value in args.set:
        command.extend(["--set", value])
    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
