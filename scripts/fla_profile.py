#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Select and record the checkout environment's FLA runtime profile.

FLA 0.5.2 refuses its Triton gated ``chunk_bwd_dqkwg`` backward on Hopper (sm_90)
with Triton 3.4-3.7.0 (FLA issue #640) and dispatches that kernel to TileLang
instead. The ``default`` profile keeps the qualified pure-Triton runtime
(``FLA_TILELANG=0``, backend dispatch disabled). The ``hopper-tilelang`` profile
installs TileLang plus a CUDA 13.0 compiler into the venv, enables FLA backend
dispatch and leaves ``FLA_TILELANG`` to FLA's own hardware default, which selects
TileLang only on Hopper. Standard library only: setup runs this before any
packages exist.
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

DEFAULT = "default"
HOPPER = "hopper-tilelang"
PROFILES = (DEFAULT, HOPPER)
MARKER = "grana-fla-profile.json"
# Relative to the venv; the nvidia-cuda-nvcc 13 wheel installs bin/nvcc here.
CUDA_HOME_RELATIVE = "lib/python3.12/site-packages/nvidia/cu13"


def parse_compute_capabilities(text: str) -> list[str]:
    """Parse ``nvidia-smi --query-gpu=compute_cap --format=csv,noheader`` output."""
    return [line.strip() for line in text.splitlines() if line.strip()]


def detect_compute_capabilities() -> list[str]:
    try:
        output = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                                check=True, capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return parse_compute_capabilities(output)


def is_hopper(capability: str) -> bool:
    return capability.split(".", 1)[0] == "9"


def select_profile(mode: str, capabilities: list[str]) -> str:
    """``auto`` selects TileLang when any local GPU is Hopper; ``1``/``0`` force it."""
    if mode == "1":
        return HOPPER
    if mode == "0":
        return DEFAULT
    if mode != "auto":
        raise ValueError("GRANA_FLA_TILELANG must be auto, 1 or 0")
    return HOPPER if any(is_hopper(c) for c in capabilities) else DEFAULT


def installation_plan(mode: str, capabilities: list[str], *, system: str, machine: str) -> dict:
    """Preview the same CPU wheel and GPU backend selection used during setup."""
    if system != "Linux" or machine not in ("x86_64", "aarch64"):
        raise ValueError(f"The full runtime supports Linux x86_64/aarch64; detected {system} {machine}")
    profile = select_profile(mode, capabilities)
    requirements = ["requirements/runtime.txt", "requirements/inference-fast.txt"]
    if profile == HOPPER:
        requirements.append("requirements/hopper-tilelang.txt")
    return {
        "platform_system": system,
        "platform_machine": machine,
        "native_wheel_platform": f"linux_{machine}",
        "detected_compute_capabilities": capabilities,
        "requested": mode,
        "profile": profile,
        "requirements": requirements,
    }


def runtime_defaults_source(profile: str, venv: Path) -> str:
    """Source of the venv's ``grana_runtime_defaults`` module, imported at interpreter start."""
    lines = ["import os",
             'os.environ.setdefault("USE_TF", "0")',
             'os.environ.setdefault("USE_FLAX", "0")']
    if profile == DEFAULT:
        lines += ['os.environ["FLA_TILELANG"] = "0"',
                  'os.environ["FLA_DISABLE_BACKEND_DISPATCH"] = "1"']
    elif profile == HOPPER:
        # FLA_TILELANG stays unset so FLA enables TileLang only on Hopper; the CUDA
        # 13.0 compiler wheel replaces an older container CUDA_HOME for its JIT.
        lines += ['os.environ["FLA_DISABLE_BACKEND_DISPATCH"] = "0"',
                  f'os.environ["CUDA_HOME"] = {str(venv.absolute() / CUDA_HOME_RELATIVE)!r}']
    else:
        raise ValueError(f"unknown FLA profile: {profile}")
    return "\n".join(lines) + "\n"


def read_profile(venv: Path) -> dict:
    """Recorded profile; environments created before profiles existed are ``default``."""
    path = venv / MARKER
    if not path.is_file():
        return {"profile": DEFAULT, "recorded": False}
    record = json.loads(path.read_text())
    if record.get("profile") not in PROFILES:
        raise ValueError(f"unknown FLA profile in {path}: {record.get('profile')}")
    return {**record, "recorded": True}


def write_runtime(venv: Path, profile: str, mode: str, capabilities: list[str]) -> None:
    site = venv / "lib/python3.12/site-packages"
    (site / "grana_runtime_defaults.py").write_text(runtime_defaults_source(profile, venv))
    (site / "grana_runtime_defaults.pth").write_text("import grana_runtime_defaults\n")
    record = {"profile": profile, "requested": mode, "detected_compute_capabilities": capabilities,
              "platform_system": platform.system(), "platform_machine": platform.machine()}
    if profile == HOPPER:
        record["cuda_home"] = str(venv.absolute() / CUDA_HOME_RELATIVE)
    (venv / MARKER).write_text(json.dumps(record, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    select = sub.add_parser("select", help="Print the profile for GRANA_FLA_TILELANG=auto|1|0")
    select.add_argument("--mode", default="auto")
    plan = sub.add_parser("plan", help="Preview CPU wheel and GPU backend selection without installing")
    plan.add_argument("--mode", default="auto")
    write = sub.add_parser("write", help="Write the venv runtime defaults and profile record")
    write.add_argument("--venv", type=Path, required=True)
    write.add_argument("--profile", choices=PROFILES, required=True)
    write.add_argument("--mode", default="auto")
    args = parser.parse_args()
    capabilities = detect_compute_capabilities()
    if args.command == "select":
        print(select_profile(args.mode, capabilities))
    elif args.command == "plan":
        try:
            result = installation_plan(args.mode, capabilities, system=platform.system(), machine=platform.machine())
        except ValueError as error:
            parser.error(str(error))
        print(json.dumps(result, indent=2))
    else:
        write_runtime(args.venv, args.profile, args.mode, capabilities)
        print(json.dumps(read_profile(args.venv)), file=sys.stderr)


if __name__ == "__main__":
    main()
