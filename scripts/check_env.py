#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify the installed Grana runtime without starting a training job."""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import sys

EXPECTED = {
    "torch": "2.10.0+cu130",
    "torchvision": "0.25.0+cu130",
    "triton": "3.6.0",
    "transformers": "5.6.0.dev0",
    "accelerate": "1.14.0",
    "peft": "0.18.1",
    "flash-linear-attention": "0.5.2",
    "fla-core": "0.5.2",
    "ray": "2.48.0",
    "datasets": "4.1.1",
    "wandb": "0.27.2",
    "causal-conv1d": "1.7.0",  # requirements/inference-fast.txt
}
TRANSFORMERS_COMMIT = "a29df2d916e3b820aecd19d3b5a877abc523ba3c"
HOPPER_EXPECTED = {
    "tilelang": "0.1.15",
    "nvidia-cuda-nvcc": "13.0.88",
    "nvidia-cuda-crt": "13.0.88",
    "nvidia-nvvm": "13.0.88",
    "nvidia-cuda-cccl": "13.0.85",
}
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fla_profile  # noqa: E402


def check_hopper_tilelang_backward(device: int) -> dict:
    """Run one Qwen3.5-shaped gated delta-rule fwd+bwd; require TileLang and fp32-reference agreement."""
    import logging
    import torch
    import torch.nn.functional as F
    import fla.ops.common.backends.tilelang.chunk_bwd as tilelang_bwd
    logging.getLogger("tilelang").setLevel(logging.WARNING)  # keep the JSON receipt alone on stdout
    from fla.ops.common.backends.tilelang import TileLangBackend
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule

    assert TileLangBackend.is_available(), "TileLang or its CUDA 13 nvcc is unavailable"
    assert TileLangBackend.is_enabled(), "FLA did not enable TileLang on this Hopper GPU (is FLA_TILELANG=0 set?)"
    calls = []
    original = tilelang_bwd.chunk_bwd_dqkwg_tilelang
    tilelang_bwd.chunk_bwd_dqkwg_tilelang = lambda *a, **k: calls.append(1) or original(*a, **k)
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        generator = torch.Generator(device=f"cuda:{device}").manual_seed(0)
        def randn(*shape):
            return torch.randn(*shape, generator=generator, device=f"cuda:{device}")
        B, T, H, D = 1, 1024, 4, 128
        inputs = {"q": F.silu(randn(B, T, H, D)).bfloat16(), "k": F.silu(randn(B, T, H, D)).bfloat16(),
                  "v": F.silu(randn(B, T, H, D)).bfloat16(), "beta": randn(B, T, H).sigmoid().bfloat16(),
                  # Qwen3.5-like decay: -exp(A_log) * softplus(a + dt_bias).
                  "g": -randn(H).exp() * F.softplus(randn(B, T, H))}
        grad_output = randn(B, T, H, D).bfloat16()
        def run(function, dtype=None):
            leaves = {n: (t.to(dtype) if dtype else t).detach().clone().requires_grad_(True) for n, t in inputs.items()}
            output, _ = function(leaves["q"], leaves["k"], leaves["v"], g=leaves["g"], beta=leaves["beta"],
                                 initial_state=None, output_final_state=False, use_qk_l2norm_in_kernel=True)
            output.backward(grad_output.to(output.dtype))
            return {"o": output.detach(), **{"d" + n: leaf.grad for n, leaf in leaves.items()}}
        reference = run(torch_chunk_gated_delta_rule, torch.float32)
        actual = run(chunk_gated_delta_rule)
    finally:
        tilelang_bwd.chunk_bwd_dqkwg_tilelang = original
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
    errors = {n: ((actual[n].double() - reference[n].double()).norm() / reference[n].double().norm()).item()
              for n in reference}
    assert calls, "FLA backward did not dispatch chunk_bwd_dqkwg to TileLang"
    # bf16 vs fp32: measured relative L2 errors are about 5e-3 for TileLang and torch alike.
    assert all(error < 2e-2 for error in errors.values()), errors
    return {"tilelang_calls": len(calls), "relative_l2_vs_fp32": errors}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-only", action="store_true", help="Skip the CUDA reduction check")
    parser.add_argument("--output", type=Path, help="Save the verification receipt")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    venv = root / ".venv-qwen35-unsloth"
    assert Path(sys.prefix).resolve() == venv, (sys.prefix, venv)
    assert Path(sys.base_prefix).resolve() == venv / ".python", sys.base_prefix
    assert platform.python_version() == "3.12.3", platform.python_version()
    assert (venv / "grana-python-copy.json").is_file()
    assert "include-system-site-packages = false" in (venv / "pyvenv.cfg").read_text()
    profile = fla_profile.read_profile(venv)
    hopper_profile = profile["profile"] == fla_profile.HOPPER
    versions = {name: metadata.version(name) for name in EXPECTED}
    assert versions == EXPECTED, versions
    if hopper_profile:
        hopper_versions = {name: metadata.version(name) for name in HOPPER_EXPECTED}
        assert hopper_versions == HOPPER_EXPECTED, hopper_versions
        versions.update(hopper_versions)
    for name in EXPECTED:
        Path(metadata.distribution(name).locate_file("")).resolve().relative_to(venv)
    direct_url = json.loads(metadata.distribution("transformers").read_text("direct_url.json"))
    assert direct_url["vcs_info"]["commit_id"] == TRANSFORMERS_COMMIT, direct_url
    for module in ("torch", "torchvision", "transformers", "peft", "ray.train", "datasets", "fla", "wandb",
                   "causal_conv1d"):
        loaded = importlib.import_module(module)
        Path(loaded.__file__).resolve().relative_to(venv)
    from transformers import Qwen3_5ForConditionalGeneration
    assert Qwen3_5ForConditionalGeneration is not None
    if hopper_profile:
        # FLA_TILELANG stays unset: FLA enables TileLang only on Hopper.
        assert os.environ.get("FLA_DISABLE_BACKEND_DISPATCH") == "0"
        assert os.environ.get("CUDA_HOME") == profile["cuda_home"], os.environ.get("CUDA_HOME")
        assert (Path(profile["cuda_home"]) / "bin/nvcc").is_file(), profile["cuda_home"]
        forbidden_packages = ("unsloth", "tensorflow")
    else:
        assert os.environ.get("FLA_TILELANG") == "0"
        assert os.environ.get("FLA_DISABLE_BACKEND_DISPATCH") == "1"
        forbidden_packages = ("tilelang", "unsloth", "tensorflow")
    for forbidden in forbidden_packages:
        try:
            metadata.distribution(forbidden)
        except metadata.PackageNotFoundError:
            pass
        else:
            raise AssertionError(f"Unexpected legacy/unsupported dependency: {forbidden}")
    import torch
    devices = []
    hopper_backward = None
    if not args.cpu_only:
        assert torch.version.cuda == "13.0", torch.version.cuda
        assert torch.cuda.is_available(), "CUDA is unavailable"
        for device in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(device)
            devices.append({"index": device, "name": props.name, "capability": list(torch.cuda.get_device_capability(device))})
        # Reduction exercises NVRTC compilation, which cu128 cannot do on GB300.
        value = torch.arange(1024, dtype=torch.float32, device="cuda").sum().item()
        assert value == 523776.0, value
        # The prebuilt causal-conv1d wheel must match this Torch/CUDA build and run on the default GPU.
        from causal_conv1d import causal_conv1d_fn
        from transformers.models.qwen3_5 import modeling_qwen3_5
        assert modeling_qwen3_5.causal_conv1d_fn is not None, "Transformers does not see causal-conv1d"
        generator = torch.Generator().manual_seed(0)
        x = torch.randn(1, 64, 256, generator=generator).to("cuda", torch.bfloat16)
        weight = torch.randn(64, 4, generator=generator).to("cuda", torch.bfloat16)
        reference = torch.nn.functional.conv1d(x.float(), weight.float().unsqueeze(1), padding=3, groups=64)
        reference = torch.nn.functional.silu(reference[..., :256])
        error = (causal_conv1d_fn(x, weight, None, activation="silu").float() - reference).abs().max().item()
        assert error < 0.1, error
        hopper = [d["index"] for d in devices if d["capability"][0] == 9]
        if hopper:
            assert hopper_profile, ("Hopper GPU detected, but FLA 0.5.2's Triton backward is refused on Hopper; "
                                    "rerun scripts/setup_env.sh (GRANA_FLA_TILELANG=auto) to add TileLang")
            hopper_backward = check_hopper_tilelang_backward(hopper[0])
    receipt = {
        "status": "passed", "python": platform.python_version(), "executable": sys.executable,
        "platform_system": platform.system(), "platform_machine": platform.machine(),
        "venv": str(venv), "versions": versions, "transformers_commit": TRANSFORMERS_COMMIT,
        "cuda_runtime": torch.version.cuda, "devices": devices, "cuda_check": not args.cpu_only,
        "fla_profile": profile["profile"], "hopper_tilelang_backward": hopper_backward,
        "training_launched": False,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
