# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
import platform
import re
from pathlib import Path

import pytest
from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("fla_profile", ROOT / "scripts/fla_profile.py")
fla_profile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fla_profile)


def test_parse_compute_capabilities_ignores_blank_lines():
    assert fla_profile.parse_compute_capabilities("9.0\n 9.0 \n\n") == ["9.0", "9.0"]


@pytest.mark.parametrize("capabilities, expected", [
    (["9.0"] * 8, fla_profile.HOPPER),           # H100 / H200 / H20
    (["8.9", "9.0"], fla_profile.HOPPER),         # any Hopper GPU needs the TileLang backward
    (["10.3"], fla_profile.DEFAULT),              # GB300
    (["12.0"], fla_profile.DEFAULT),              # RTX 5090
    (["8.9"], fla_profile.DEFAULT),               # RTX 4090
    ([], fla_profile.DEFAULT),                    # no GPU or nvidia-smi
])
def test_auto_selects_tilelang_only_for_hopper(capabilities, expected):
    assert fla_profile.select_profile("auto", capabilities) == expected


def test_explicit_mode_overrides_detection():
    assert fla_profile.select_profile("1", []) == fla_profile.HOPPER
    assert fla_profile.select_profile("0", ["9.0"]) == fla_profile.DEFAULT
    with pytest.raises(ValueError, match="auto, 1 or 0"):
        fla_profile.select_profile("yes", ["9.0"])


@pytest.mark.parametrize("machine, capabilities, expected", [
    ("x86_64", ["9.0"], fla_profile.HOPPER),
    ("x86_64", ["8.9"], fla_profile.DEFAULT),
    ("x86_64", ["12.0"], fla_profile.DEFAULT),
    ("aarch64", ["10.3"], fla_profile.DEFAULT),
    ("x86_64", ["8.9", "9.0"], fla_profile.HOPPER),
])
def test_installation_plan_matches_native_wheel_markers(machine, capabilities, expected):
    plan = fla_profile.installation_plan("auto", capabilities, system="Linux", machine=machine)
    assert plan["platform_machine"] == machine
    assert plan["detected_compute_capabilities"] == capabilities
    assert plan["profile"] == expected
    assert ("requirements/hopper-tilelang.txt" in plan["requirements"]) == (expected == fla_profile.HOPPER)
    requirements = [Requirement(line) for line in (ROOT / "requirements/inference-fast.txt").read_text().splitlines()
                    if line.startswith("causal-conv1d")]
    selected = [r for r in requirements if r.marker.evaluate({"platform_machine": machine})]
    assert len(selected) == 1
    assert f'-{plan["native_wheel_platform"]}.whl#sha256=' in selected[0].url


@pytest.mark.parametrize("system, machine", [("Darwin", "arm64"), ("Linux", "ppc64le")])
def test_installation_plan_rejects_unsupported_native_wheel_platforms(system, machine):
    with pytest.raises(ValueError, match="supports Linux x86_64/aarch64"):
        fla_profile.installation_plan("auto", ["9.0"], system=system, machine=machine)


def run_defaults(source, environ):
    """Execute the generated startup module against a fake ``os.environ``."""
    assert source.startswith("import os\n")
    fake_os = type("FakeOs", (), {"environ": environ})()
    exec(source.removeprefix("import os\n"), {"os": fake_os})  # noqa: S102 - trusted generated source
    return environ


def test_default_profile_forces_qualified_triton_runtime(tmp_path):
    source = fla_profile.runtime_defaults_source(fla_profile.DEFAULT, tmp_path)
    environ = run_defaults(source, {"FLA_TILELANG": "1", "FLA_DISABLE_BACKEND_DISPATCH": "0"})
    assert environ["FLA_TILELANG"] == "0"
    assert environ["FLA_DISABLE_BACKEND_DISPATCH"] == "1"
    assert "CUDA_HOME" not in environ


def test_hopper_profile_enables_dispatch_and_leaves_tilelang_to_fla(tmp_path):
    source = fla_profile.runtime_defaults_source(fla_profile.HOPPER, tmp_path)
    environ = run_defaults(source, {"CUDA_HOME": "/usr/local/cuda", "USE_TF": "1"})
    assert environ["FLA_DISABLE_BACKEND_DISPATCH"] == "0"
    assert "FLA_TILELANG" not in environ
    assert environ["CUDA_HOME"] == str(tmp_path / fla_profile.CUDA_HOME_RELATIVE)
    assert environ["USE_TF"] == "1" and environ["USE_FLAX"] == "0"


def test_write_and_read_profile_round_trip(tmp_path):
    venv = tmp_path / "venv"
    (venv / "lib/python3.12/site-packages").mkdir(parents=True)
    assert fla_profile.read_profile(venv) == {"profile": fla_profile.DEFAULT, "recorded": False}
    fla_profile.write_runtime(venv, fla_profile.HOPPER, "auto", ["9.0"])
    site = venv / "lib/python3.12/site-packages"
    assert (site / "grana_runtime_defaults.pth").read_text() == "import grana_runtime_defaults\n"
    assert 'FLA_DISABLE_BACKEND_DISPATCH"] = "0"' in (site / "grana_runtime_defaults.py").read_text()
    record = fla_profile.read_profile(venv)
    assert record["profile"] == fla_profile.HOPPER and record["recorded"]
    assert record["detected_compute_capabilities"] == ["9.0"]
    assert record["platform_machine"] == platform.machine()
    assert record["cuda_home"] == str(venv / fla_profile.CUDA_HOME_RELATIVE)
    (venv / fla_profile.MARKER).write_text(json.dumps({"profile": "other"}))
    with pytest.raises(ValueError, match="unknown FLA profile"):
        fla_profile.read_profile(venv)


def test_hopper_requirements_match_check_env_pins():
    pins = dict(line.split("==") for line in (ROOT / "requirements/hopper-tilelang.txt").read_text().splitlines()
                if line and not line.startswith("#"))
    check_env = (ROOT / "scripts/check_env.py").read_text()
    for name in ("tilelang", "nvidia-cuda-nvcc", "nvidia-cuda-crt", "nvidia-nvvm", "nvidia-cuda-cccl"):
        assert f'"{name}": "{pins[name]}"' in check_env


def test_launchers_do_not_override_the_venv_fla_profile():
    # Ray workers inherit the driver environment; a hard-coded FLA_TILELANG=0 would
    # disable the Hopper TileLang backward that the venv profile selected.
    for path in ("scripts/setup_env.sh", "scripts/train.sh", "scripts/ray_migration/train_ray.py"):
        text = (ROOT / path).read_text()
        assert not re.search(r"(?<![A-Z_])FLA_TILELANG\s*=|[\"']FLA_TILELANG[\"']\s*:", text), path
