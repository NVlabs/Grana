# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Optional Node regression checks for the annotation page; no npm dependencies."""
from pathlib import Path
import shutil
import subprocess

import pytest


def test_label_ui_javascript():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is needed for the annotation JavaScript tests")
    version = int(subprocess.check_output([node, "-p", "process.versions.node.split('.')[0]"], text=True))
    if version < 18:
        pytest.skip("The JavaScript test runner needs Node.js 18+")
    subprocess.run([node, "--test", str(Path(__file__).with_suffix(".cjs"))], check=True)
