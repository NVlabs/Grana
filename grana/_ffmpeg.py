# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FFmpeg options shared by the exact-frame decoders."""
from functools import lru_cache
import subprocess


@lru_cache(maxsize=None)
def passthrough(executable: str) -> tuple[str, ...]:
    """Keep every decoded frame: FFmpeg 5.1 added -fps_mode and later releases removed -vsync."""
    options = subprocess.run([executable, "-hide_banner", "-h", "full"], capture_output=True, text=True).stdout
    return ("-fps_mode", "passthrough") if "-fps_mode" in options else ("-vsync", "0")
