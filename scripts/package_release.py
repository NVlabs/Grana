#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build a reproducible source archive from the reviewed release-files.txt.

No Git checkout or third-party Python packages are required. New public files
must be added to release-files.txt explicitly. Local Markdown references are
checked before any output is written. SOURCE_DATE_EPOCH sets the archive time;
the default is zero, making unchanged source trees byte-for-byte reproducible.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import tarfile

from check_release import read_release, validate_references

ROOT = Path(__file__).resolve().parents[1]


def build_release(root: Path, output: Path, *, epoch: int = 0) -> dict:
    """Validate and snapshot allowlisted bytes, then write archive and checksums."""
    if not 0 <= epoch <= 0xFFFFFFFF:
        raise ValueError("SOURCE_DATE_EPOCH must be an integer between 0 and 4294967295")
    sources = read_release(root)
    validate_references(sources)
    output = output.expanduser().absolute()
    receipt_path = output.with_suffix(output.suffix + ".manifest.json")
    for path in (output, receipt_path):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"Use a fresh artifact path: {path}")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Use exclusive creation: a preexisting archive/receipt is never replaced.
    with output.open("xb") as raw:
        try:
            # Suppress the output filename and host mtime in the gzip header.
            with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=epoch) as compressed:
                with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive:
                    for name, content in sources.items():
                        info = tarfile.TarInfo("grana/" + name)
                        info.size = len(content)
                        info.mtime = epoch
                        info.uid = info.gid = 0
                        info.uname = info.gname = ""
                        # Permissions derive from content, never local umask or chmod.
                        info.mode = 0o755 if content.startswith(b"#!") else 0o644
                        archive.addfile(info, io.BytesIO(content))
        except BaseException:
            output.unlink()
            raise
    receipt = {
        "format_version": 1,
        "archive": output.name,
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "source_date_epoch": epoch,
        "file_count": len(sources),
        "source_bytes": sum(map(len, sources.values())),
        "files": {name: hashlib.sha256(content).hexdigest() for name, content in sources.items()},
    }
    with receipt_path.open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist/grana-0.1.0.tar.gz")
    args = parser.parse_args()
    try:
        receipt = build_release(ROOT, args.output, epoch=int(os.environ.get("SOURCE_DATE_EPOCH", "0")))
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Source release failed: {exc}\n")
    print(json.dumps({key: value for key, value in receipt.items() if key != "files"}, indent=2))


if __name__ == "__main__":
    main()
