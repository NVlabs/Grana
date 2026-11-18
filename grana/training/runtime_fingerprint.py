# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Freeze installed training dependencies without importing CUDA or model code.

Wheel metadata alone misses edits made in site-packages without reinstalling a
package. We therefore also hash installed Python/kernel files. Large native
libraries retain their wheel hash plus their filesystem change identity instead
of rereading gigabytes at every campaign boundary. This is an operational drift
check, not a cryptographic attestation of an untrusted filesystem. Capture a new
baseline after copying an environment, and keep that environment unchanged.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import sys
import sysconfig
from typing import Iterable, Mapping


SCHEMA = "grana-runtime-fingerprint-v1"
DEFAULT_PACKAGES = (
    "torch", "transformers", "peft", "ray", "datasets", "wandb",
    "flash-linear-attention", "fla-core", "triton", "causal-conv1d",
    "accelerate", "safetensors", "numpy", "einops", "tokenizers",
    "huggingface-hub", "flash-attn", "mamba-ssm", "xformers",
    "liger-kernel", "unsloth", "unsloth-zoo",
    # Hopper FLA profile: TileLang replaces FLA's gated backward kernel.
    "tilelang",
)
_NATIVE_HASH_LIMIT = 64 * 1024 * 1024
_KERNEL_SUFFIXES = {".py", ".pyi", ".dll", ".dylib", ".ptx", ".cubin", ".fatbin", ".bc"}
_KERNEL_EXECUTABLES = {"ptxas", "cuobjdump", "nvdisasm", "nvcc"}


class RuntimeFingerprintMismatch(ValueError):
    """The installed runtime differs from a frozen campaign baseline."""


def _digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _identity(path: Path) -> tuple[int, int, int, int, int]:
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


@lru_cache(maxsize=32768)
def _file_digest(path: str, identity: tuple[int, int, int, int, int]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    if _identity(Path(path)) != identity:
        raise RuntimeError(f"Runtime file changed while fingerprinting: {Path(path).name}")
    return digest.hexdigest()


def _package_fingerprint(name: str) -> dict:
    try:
        distribution = importlib.metadata.distribution(name)
    except importlib.metadata.PackageNotFoundError:
        # Absence matters: installing an optional kernel can change the backend.
        return {"installed": False}

    metadata = {}
    for filename in ("METADATA", "RECORD", "WHEEL", "direct_url.json", "entry_points.txt"):
        content = distribution.read_text(filename)
        # direct_url.json may contain a private source URL. Publish only a hash.
        metadata[filename] = None if content is None else hashlib.sha256(content.encode()).hexdigest()
    files = distribution.files
    observed, large_native, missing = [], [], []
    for entry in sorted(files or (), key=str):
        relative = str(entry)
        name_path = Path(relative)
        if not (name_path.suffix in _KERNEL_SUFFIXES or ".so" in name_path.suffixes
                or ".so." in name_path.name or name_path.name in _KERNEL_EXECUTABLES):
            continue
        path = Path(distribution.locate_file(entry))
        try:
            identity = _identity(path)
        except FileNotFoundError:
            missing.append(relative)
            continue
        is_source = name_path.suffix in {".py", ".pyi"}
        if not is_source and identity[2] > _NATIVE_HASH_LIMIT:
            recorded_hash = entry.hash
            large_native.append({
                # Nanosecond timestamps exceed JavaScript's exact integer
                # range. Strings preserve the fingerprint through W&B JSON.
                "path": relative, "size": identity[2], "mtime_ns": str(identity[3]), "ctime_ns": str(identity[4]),
                "record_hash": (None if recorded_hash is None
                                else f"{recorded_hash.mode}={recorded_hash.value}"),
            })
        else:
            observed.append([relative, _file_digest(str(path), identity)])
    return {
        "installed": True,
        "version": distribution.version,
        "metadata_sha256": _digest(metadata),
        "record_sha256": metadata["RECORD"],
        "file_manifest_available": files is not None,
        "observed_files_sha256": _digest(observed),
        "observed_file_count": len(observed),
        "large_native_files": large_native,
        "missing_files": missing,
    }


def capture_runtime_fingerprint(*, package_names: Iterable[str] | None = None) -> dict:
    """Return a deterministic JSON payload; never inspect credentials or CUDA.

    Default capture includes optional kernels even when absent and all installed
    NVIDIA wheel distributions. ``package_names`` narrows capture for callers
    testing a dependency subset; campaigns should use the default.
    """
    if package_names is None:
        names = set(DEFAULT_PACKAGES)
        for distribution in importlib.metadata.distributions():
            name = distribution.metadata.get("Name", "").lower().replace("_", "-")
            if name.startswith("nvidia-"):
                names.add(name)
    else:
        names = {str(name).lower().replace("_", "-") for name in package_names}
    fingerprint = {
        "schema": SCHEMA,
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "cache_tag": sys.implementation.cache_tag,
            "abi": sysconfig.get_config_var("SOABI"),
            "platform": sysconfig.get_platform(),
        },
        "packages": {name: _package_fingerprint(name) for name in sorted(names)},
    }
    return {**fingerprint, "sha256": _digest(fingerprint)}


def _validate_payload(fingerprint: Mapping, *, label: str) -> None:
    if fingerprint.get("schema") != SCHEMA or not isinstance(fingerprint.get("packages"), dict):
        raise RuntimeFingerprintMismatch(f"Invalid {label} runtime fingerprint schema")
    if fingerprint.get("sha256") != _digest({key: value for key, value in fingerprint.items() if key != "sha256"}):
        raise RuntimeFingerprintMismatch(f"Invalid {label} runtime fingerprint digest")


def validate_runtime_fingerprint(expected: Mapping, current: Mapping | None = None) -> dict:
    """Return the current fingerprint or fail on any frozen dependency drift.

    Error messages name changed dependencies but do not include private package
    metadata or process environment values. The full captured payload is safe to
    persist in the run directory or upload as experiment provenance.
    """
    _validate_payload(expected, label="expected")
    actual = capture_runtime_fingerprint() if current is None else dict(current)
    _validate_payload(actual, label="current")
    if expected["sha256"] != actual["sha256"]:
        changes = []
        if expected.get("python") != actual.get("python"):
            changes.append("Python/platform")
        for name in sorted(expected["packages"].keys() | actual["packages"].keys()):
            if expected["packages"].get(name) != actual["packages"].get(name):
                changes.append(name)
        raise RuntimeFingerprintMismatch("Frozen runtime changed: " + ", ".join(changes or ["fingerprint metadata"]))
    return actual
