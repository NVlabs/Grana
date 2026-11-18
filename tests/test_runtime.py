# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
import sys

import pytest

from grana.runtime import configure_cuda_allocator_defaults


@pytest.mark.parametrize("overrides", [
    {},
    {"PYTORCH_ALLOC_CONF": "expandable_segments:False"},
    {"PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128"},
    {"PYTORCH_ALLOC_CONF": "backend:cudaMallocAsync", "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:64"},
    {"PYTORCH_ALLOC_CONF": ""},
    {"PYTORCH_CUDA_ALLOC_CONF": ""},
])
def test_allocator_default_respects_overrides_and_reaches_child_processes(monkeypatch, overrides):
    keys = ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF")
    for key in keys:
        # Track even originally absent keys so the helper's mutation is undone.
        monkeypatch.setenv(key, "")
        monkeypatch.delenv(key)
    for key, value in overrides.items():
        monkeypatch.setenv(key, value)
    expected = {key: overrides.get(key) for key in keys}
    if not overrides:
        expected[keys[0]] = "expandable_segments:True"
    assert configure_cuda_allocator_defaults() == expected
    assert configure_cuda_allocator_defaults() == expected
    code = "import os,json; print(json.dumps({k:os.environ.get(k) for k in " + repr(keys) + "}))"
    child = subprocess.check_output([sys.executable, "-c", code], text=True)
    assert json.loads(child) == expected


def test_allocator_setup_does_not_import_torch_or_ray():
    code = """
import sys
from grana.runtime import configure_cuda_allocator_defaults
configure_cuda_allocator_defaults()
assert 'torch' not in sys.modules
assert 'ray' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True)
