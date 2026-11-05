#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Create an isolated venv with an owned CPython runtime and no filesystem links."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def reject_links(event: str, args: tuple) -> None:
    if event in {"os.symlink", "os.link"}:
        raise RuntimeError(f"link creation is forbidden while creating this venv: {event}")


def create_copied_venv(source_python: Path, destination: Path, version: str) -> None:
    source_python = source_python.resolve(strict=True)
    destination = destination.absolute()
    if destination.exists() or destination.is_symlink():
        raise ValueError(f"copy-mode creation requires an absent destination: {destination}")
    info = json.loads(
        subprocess.check_output(
            [
                str(source_python),
                "-I",
                "-c",
                "import json,platform,sys; print(json.dumps([sys.prefix,sys.base_prefix,platform.python_version()]))",
            ],
            text=True,
        )
    )
    source_root = Path(info[1]).resolve(strict=True)
    if info[0] != info[1] or info[2] != version:
        raise ValueError(f"require a base CPython {version} interpreter, got {info}")
    parts = source_root.parts
    if not any(parts[i : i + 2] == ("uv", "python") for i in range(len(parts) - 1)):
        # uv permits a custom managed store. Prove ownership through uv's
        # read-only resolver instead of requiring its default directory name.
        managed_python = Path(subprocess.check_output(
            ["uv", "python", "find", "--system", "--managed-python", "--no-python-downloads", version],
            text=True,
        ).strip()).resolve(strict=True)
        if managed_python != source_python:
            raise ValueError(f"source must be the existing uv-managed CPython distribution: {source_root}")
    source_python.relative_to(source_root)
    for directory, dirs, files in os.walk(source_root, followlinks=False):
        for name in dirs + files:
            entry = Path(directory) / name
            if entry.is_symlink():
                entry.resolve(strict=True).relative_to(source_root)

    sys.addaudithook(reject_links)
    runtime = destination / ".python"
    # Dereference source aliases while reading; every destination is a new file.
    shutil.copytree(source_root, runtime, symlinks=False)
    copied_python = runtime / source_python.relative_to(source_root)
    (destination / "lib").mkdir()
    # stdlib venv otherwise creates this alias even with symlinks=False.
    (destination / "lib64").mkdir()
    # The copied executable's ELF RUNPATH is relative to its new bin directory.
    for library in (runtime / "lib").glob("libpython*.so*"):
        shutil.copy2(library, destination / "lib" / library.name)
    builder = """
import sys, venv
def reject_links(event, args):
    if event in {'os.symlink', 'os.link'}:
        raise RuntimeError('link creation is forbidden: ' + event)
sys.addaudithook(reject_links)
venv.EnvBuilder(symlinks=False, with_pip=True).create(sys.argv[1])
"""
    subprocess.run([str(copied_python), "-I", "-c", builder, str(destination)], check=True)
    receipt = {
        "schema": 1,
        "mode": "copy",
        "python_version": version,
        "source_python": str(source_python),
        "executable_sha256": hashlib.sha256(source_python.read_bytes()).hexdigest(),
    }
    (destination / "grana-python-copy.json").write_text(json.dumps(receipt, indent=2) + "\n")
    for directory, dirs, files in os.walk(destination, followlinks=False):
        for name in dirs + files:
            entry = Path(directory) / name
            if entry.is_symlink() or (entry.is_file() and entry.stat().st_nlink != 1):
                raise RuntimeError(f"unexpected filesystem link in copied venv: {entry}")
    print(f"Created isolated copy-only venv: {destination}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-python", required=True, type=Path)
    parser.add_argument("--venv", required=True, type=Path)
    parser.add_argument("--python-version", required=True)
    args = parser.parse_args()
    create_copied_venv(args.source_python, args.venv, args.python_version)
