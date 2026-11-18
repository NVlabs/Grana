#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail
GRANA_SCRIPT_REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
GRANA_SCRIPT_PYTHON="${GRANA_PYTHON:-$GRANA_SCRIPT_REPO/.venv-qwen35-unsloth/bin/python}"
if [[ ! -x "$GRANA_SCRIPT_PYTHON" ]]; then
  printf '%s\n' "Missing Python environment: $GRANA_SCRIPT_PYTHON. Run bash scripts/setup_env.sh first." >&2
  exit 1
fi
cd "$GRANA_SCRIPT_REPO"
export USE_TF=0
export TRANSFORMERS_NO_TF=1
