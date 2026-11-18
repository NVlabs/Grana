#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
source .venv-qwen35-unsloth/bin/activate
# FLA_TILELANG/FLA_DISABLE_BACKEND_DISPATCH come from the venv FLA profile (scripts/fla_profile.py).
export TOKENIZERS_PARALLELISM=false
export USE_TF=0
export TRANSFORMERS_NO_TF=1
# Repeat --config to run a small, explicit sequence of independent recipes.
python scripts/ray_migration/run_specs_cli.py "$@"
