# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A campaign must reject dependency drift, including same-version kernel edits."""
import copy
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from grana.training import runtime_fingerprint as runtime


class Distribution:
    def __init__(self, root, *, version="1.0", metadata=None):
        self.root = root
        self.version = version
        self.metadata = {"Name": root.name}
        self.contents = {"METADATA": "Name: fixture\nVersion: 1.0\n", "RECORD": "original wheel record"}
        self.contents.update(metadata or {})
        self.files = []

    def write(self, relative, contents):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents)
        entry = importlib.metadata.PackagePath(relative)
        entry.hash = None
        self.files.append(entry)
        return path

    def read_text(self, filename):
        return self.contents.get(filename)

    def locate_file(self, entry):
        return self.root / entry


@pytest.fixture
def installed(tmp_path, monkeypatch):
    registry = {}

    def distribution(name):
        if name not in registry:
            raise importlib.metadata.PackageNotFoundError(name)
        return registry[name]

    monkeypatch.setattr(runtime.importlib.metadata, "distribution", distribution)
    monkeypatch.setattr(runtime.importlib.metadata, "distributions", lambda: list(registry.values()))
    runtime._file_digest.cache_clear()
    return registry


def rewrite_same_size_preserving_mtime(path, contents):
    before = path.stat()
    assert len(contents) == before.st_size
    deadline = time.monotonic() + 3
    # Metadata-based tracking cannot observe writes sharing one filesystem
    # timestamp. Wait for the ctime change this fixture is intended to exercise.
    while True:
        path.write_bytes(contents)
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
        after = path.stat()
        assert after.st_size == before.st_size
        assert after.st_mtime_ns == before.st_mtime_ns
        if after.st_ctime_ns != before.st_ctime_ns:
            break
        if time.monotonic() >= deadline:
            pytest.fail("Filesystem did not expose a ctime change within 3 seconds")
        time.sleep(.01)
    assert after.st_ctime_ns != before.st_ctime_ns


def test_missing_optional_kernel_becomes_installed(installed, tmp_path):
    baseline = runtime.capture_runtime_fingerprint(package_names=["causal-conv1d"])
    assert baseline["packages"]["causal-conv1d"] == {"installed": False}
    kernel = Distribution(tmp_path / "causal-conv1d")
    kernel.write("causal_conv1d/causal_conv1d_interface.py", b"backend = 'cuda'\n")
    installed["causal-conv1d"] = kernel
    current = runtime.capture_runtime_fingerprint(package_names=["causal-conv1d"])
    with pytest.raises(runtime.RuntimeFingerprintMismatch, match="causal-conv1d"):
        runtime.validate_runtime_fingerprint(baseline, current)


@pytest.mark.parametrize("relative", ["fla/ops/delta.py", "causal_conv1d_cuda.so"])
def test_same_version_same_size_kernel_edit_is_detected(installed, tmp_path, relative):
    kernel = Distribution(tmp_path / "fla-core")
    path = kernel.write(relative, b"original")
    installed["fla-core"] = kernel
    baseline = runtime.capture_runtime_fingerprint(package_names=["fla-core"])
    rewrite_same_size_preserving_mtime(path, b"modified")
    current = runtime.capture_runtime_fingerprint(package_names=["fla-core"])
    assert current["packages"]["fla-core"]["version"] == baseline["packages"]["fla-core"]["version"]
    assert current["packages"]["fla-core"]["record_sha256"] == baseline["packages"]["fla-core"]["record_sha256"]
    with pytest.raises(runtime.RuntimeFingerprintMismatch, match="fla-core"):
        runtime.validate_runtime_fingerprint(baseline, current)


def test_large_native_change_uses_identity_without_hashing_binary(installed, tmp_path, monkeypatch):
    kernel = Distribution(tmp_path / "torch")
    path = kernel.write("torch/lib/libtorch_cuda.so", b"original")
    installed["torch"] = kernel
    monkeypatch.setattr(runtime, "_NATIVE_HASH_LIMIT", 4)
    monkeypatch.setattr(runtime, "_file_digest", lambda *_: pytest.fail("Large binary was read"))
    baseline = runtime.capture_runtime_fingerprint(package_names=["torch"])
    native = baseline["packages"]["torch"]["large_native_files"][0]
    assert native["mtime_ns"] == str(path.stat().st_mtime_ns)
    assert native["ctime_ns"] == str(path.stat().st_ctime_ns)
    assert runtime.validate_runtime_fingerprint(baseline, json.loads(json.dumps(baseline))) == baseline
    rewrite_same_size_preserving_mtime(path, b"modified")
    current = runtime.capture_runtime_fingerprint(package_names=["torch"])
    with pytest.raises(runtime.RuntimeFingerprintMismatch, match="torch"):
        runtime.validate_runtime_fingerprint(baseline, current)


def test_metadata_reinstall_and_deleted_file_are_detected(installed, tmp_path):
    package = Distribution(tmp_path / "transformers")
    source = package.write("transformers/modeling_utils.py", b"x = 1\n")
    installed["transformers"] = package
    baseline = runtime.capture_runtime_fingerprint(package_names=["transformers"])
    package.contents["RECORD"] = "different build of same version"
    changed = runtime.capture_runtime_fingerprint(package_names=["transformers"])
    with pytest.raises(runtime.RuntimeFingerprintMismatch, match="transformers"):
        runtime.validate_runtime_fingerprint(baseline, changed)
    package.contents["RECORD"] = "original wheel record"
    source.unlink()
    changed = runtime.capture_runtime_fingerprint(package_names=["transformers"])
    assert changed["packages"]["transformers"]["missing_files"] == ["transformers/modeling_utils.py"]
    with pytest.raises(runtime.RuntimeFingerprintMismatch, match="transformers"):
        runtime.validate_runtime_fingerprint(baseline, changed)


def test_repeated_capture_ignores_bytecode_and_reuses_content_hashes(installed, tmp_path):
    package = Distribution(tmp_path / "fla-core")
    package.write("fla/ops/delta.py", b"x = 1\n")
    bytecode = package.write("fla/ops/__pycache__/delta.pyc", b"compiled1")
    installed["fla-core"] = package
    baseline = runtime.capture_runtime_fingerprint(package_names=["fla-core"])
    initial_misses = runtime._file_digest.cache_info().misses
    bytecode.write_bytes(b"compiled2")
    current = runtime.capture_runtime_fingerprint(package_names=["fla-core"])
    assert runtime.validate_runtime_fingerprint(baseline, current) == baseline
    assert runtime._file_digest.cache_info().misses == initial_misses
    assert runtime._file_digest.cache_info().hits >= 1


def test_default_capture_adds_nvidia_dependencies_and_never_exposes_secrets(installed, tmp_path, monkeypatch):
    secret = "secret-that-must-never-appear"
    monkeypatch.setenv("WANDB_API_KEY", secret)
    package = Distribution(tmp_path / "nvidia-cudnn-cu13", metadata={"direct_url.json": secret})
    package.write("nvidia/cudnn/lib/libcudnn.so.9", b"native library")
    installed["nvidia-cudnn-cu13"] = package
    snapshot = runtime.capture_runtime_fingerprint()
    assert snapshot["packages"]["nvidia-cudnn-cu13"]["installed"]
    assert snapshot["packages"]["causal-conv1d"] == {"installed": False}
    assert secret not in json.dumps(snapshot)
    assert "WANDB_API_KEY" not in json.dumps(snapshot)


def test_validate_captures_current_and_rejects_corrupt_baseline(installed, monkeypatch):
    baseline = runtime.capture_runtime_fingerprint(package_names=["causal-conv1d"])
    monkeypatch.setattr(runtime, "capture_runtime_fingerprint", lambda: copy.deepcopy(baseline))
    assert runtime.validate_runtime_fingerprint(baseline) == baseline
    altered = copy.deepcopy(baseline)
    altered["packages"]["causal-conv1d"]["installed"] = True
    with pytest.raises(runtime.RuntimeFingerprintMismatch, match="expected.*digest"):
        runtime.validate_runtime_fingerprint(altered)
    altered["schema"] = "unknown"
    with pytest.raises(runtime.RuntimeFingerprintMismatch, match="schema"):
        runtime.validate_runtime_fingerprint(altered)


def test_fingerprint_module_and_capture_do_not_import_torch_or_cuda():
    program = """
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'transformers', 'triton', 'fla', 'wandb'}:
        raise AssertionError('Imported model runtime: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from grana.training.runtime_fingerprint import capture_runtime_fingerprint
snapshot = capture_runtime_fingerprint(package_names=['intentionally-absent-fixture-package'])
assert snapshot['packages']['intentionally-absent-fixture-package']['installed'] is False
"""
    result = subprocess.run([sys.executable, "-c", program], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr


def test_environment_summary_includes_kernel_provenance(installed, monkeypatch):
    from grana.training.logging import environment_summary

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        version=SimpleNamespace(cuda="13.0"),
        cuda=SimpleNamespace(device_count=lambda: 0, get_device_name=lambda _: "unused"),
    ))
    summary = environment_summary()
    assert summary["packages"]["causal-conv1d"] is None
    assert "fla-core" in summary["packages"] and "triton" in summary["packages"]
    assert summary["runtime_fingerprint"]["schema"] == runtime.SCHEMA
    assert summary["gpu_names"] == []
