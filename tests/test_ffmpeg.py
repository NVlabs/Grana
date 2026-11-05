# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from grana._ffmpeg import passthrough


@pytest.mark.parametrize(("help_text", "expected"), [
    ("-fps_mode[:<stream_spec>]  set framerate mode", ("-fps_mode", "passthrough")),
    ("-vsync  set video sync method globally", ("-vsync", "0")),
])
def test_passthrough_matches_installed_ffmpeg(tmp_path, help_text, expected):
    executable = tmp_path / "ffmpeg"
    executable.write_text(f"#!/bin/sh\necho '{help_text}'\n")
    executable.chmod(0o755)
    assert passthrough(str(executable)) == expected
