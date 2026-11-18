# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional fast-inference wheels stay hash-pinned to the runtime they were built for."""
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
WHEEL = re.compile(
    r"causal-conv1d @ https://github\.com/Dao-AILab/causal-conv1d/releases/download/v(?P<tag>[\d.]+)/"
    r"causal_conv1d-(?P<version>[\d.]+)%2Bcu(?P<cuda>\d+)torch(?P<torch>\d+\.\d+)cxx11abiTRUE-cp312-cp312-"
    r"linux_(?P<arch>\w+)\.whl#sha256=[0-9a-f]{64} ; platform_machine == \"(?P<marker>\w+)\"")


def test_inference_fast_wheels_match_pinned_torch_cuda_and_check_env():
    runtime = (ROOT / "requirements/runtime.txt").read_text()
    torch = re.search(r"^torch==(\d+\.\d+)\.\d+\+cu(\d+)$", runtime, re.MULTILINE)
    spec = importlib.util.spec_from_file_location("check_env", ROOT / "scripts/check_env.py")
    check_env = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(check_env)
    text = (ROOT / "requirements/inference-fast.txt").read_text()
    lines = [line for line in text.splitlines() if line.startswith("causal-conv1d")]
    wheels = [WHEEL.fullmatch(line) for line in lines]
    assert all(wheels), lines
    assert {wheel["arch"] for wheel in wheels} == {wheel["marker"] for wheel in wheels} == {"x86_64", "aarch64"}
    for wheel in wheels:
        assert wheel["tag"] == wheel["version"] == check_env.EXPECTED["causal-conv1d"]
        assert wheel["torch"] == torch[1] and torch[2].startswith(wheel["cuda"])
    # The wheel declares ninja; keep this dependency pinned by hash.
    assert re.search(r"^ninja==[\d.]+ \\\n(\s+--hash=sha256:[0-9a-f]{64}( \\)?\n){2}", text, re.MULTILINE)


def test_setup_rejects_unknown_arguments_before_changing_anything():
    for arguments in (["--inference-fast"], ["--no-inference-fast"], ["--extra"],
                      ["--plan", "--extra"], ["--check", "--extra"], ["--help", "--extra"]):
        result = subprocess.run(["bash", str(ROOT / "scripts/setup_env.sh"), *arguments],
                                capture_output=True, text=True, timeout=30)
        assert result.returncode == 2 and "Unknown arguments" in result.stderr, arguments
    usage = subprocess.run(["bash", str(ROOT / "scripts/setup_env.sh"), "--help"],
                           capture_output=True, text=True, timeout=30)
    assert usage.returncode == 0 and "causal-conv1d" in usage.stdout


def test_setup_plan_detects_hopper_without_creating_an_environment(tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    for name in ("setup_env.sh", "fla_profile.py"):
        shutil.copy2(ROOT / "scripts" / name, scripts / name)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    smi = bin_dir / "nvidia-smi"
    smi.write_text("#!/bin/sh\nprintf '9.0\\n9.0\\n'\n")
    smi.chmod(0o755)
    before = set(tmp_path.rglob("*"))
    # Host shell startup hooks can call nvidia-smi and recurse through this fake CLI.
    env = {key: value for key, value in os.environ.items() if key not in ("BASH_ENV", "ENV")}
    result = subprocess.run(["bash", str(scripts / "setup_env.sh"), "--plan"],
                            env={**env, "PATH": f"{bin_dir}:{os.environ['PATH']}",
                                 "SOURCE_PYTHON": sys.executable, "GRANA_FLA_TILELANG": "auto"},
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan["profile"] == "hopper-tilelang"
    assert plan["detected_compute_capabilities"] == ["9.0", "9.0"]
    assert set(tmp_path.rglob("*")) == before
