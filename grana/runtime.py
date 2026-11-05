# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Training must run in a checkout-owned environment and Ray Train worker."""
from pathlib import Path
import sys

from ._paths import PROJECT_ROOT


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
