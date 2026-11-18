# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Training must run in a checkout-owned environment and Ray Train worker."""
import os
from pathlib import Path
import sys

from ._paths import PROJECT_ROOT


def configure_cuda_allocator_defaults() -> dict:
    """Set the training allocator default before importing Ray/PyTorch.

    Prompt lengths vary between updates even at fixed physical batch size.
    Expandable segments let those allocations grow without leaving unusable
    slivers in many separate segments. Preserve either spelling of an explicit
    caller override, including an empty value; never merge allocator options.
    The local Ray driver propagates this environment to every worker.
    """
    keys = ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF")
    if not any(key in os.environ for key in keys):
        os.environ[keys[0]] = "expandable_segments:True"
    return {key: os.environ.get(key) for key in keys}


def require_owned_environment() -> None:
    root = PROJECT_ROOT
    expected = root / ".venv-qwen35-unsloth"
    if Path(sys.prefix).resolve() != expected:
        raise RuntimeError(f"Activate {expected}/bin/activate before training")


def require_ray_worker() -> None:
    require_owned_environment()
    from ray.train._internal.session import get_session
    if get_session() is None:
        raise RuntimeError("Training is allowed only inside the Grana Ray Train worker")
