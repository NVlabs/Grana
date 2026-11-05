# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared checkout location, independent of a workflow module's nesting."""
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
