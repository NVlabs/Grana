#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"
exec "$GRANA_SCRIPT_PYTHON" -m grana.training.campaign_cli status "$@"
