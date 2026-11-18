#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
GRANA_INFERENCE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
# Convenience launcher for the single supported CUDA inference runtime.
export PYTHONPATH="${GRANA_INFERENCE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
# -P excludes the caller's cwd from import precedence so this source root wins;
# relative CLI paths still resolve in the caller's cwd.
exec "${PYTHON_BIN:-${GRANA_INFERENCE_ROOT}/.venv-qwen35-unsloth/bin/python}" \
  -P -m grana.inference.server "$@"
