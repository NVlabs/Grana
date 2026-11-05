# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint prediction, request loading and HTTP serving.

The public predictor types remain available directly from grana.inference.
Importing this package alone does not load PyTorch or a model.
"""
from importlib import import_module


__all__ = ["GranaPredictor", "InferenceRequest", "POI_SLOT", "VIEW_POLICIES",
           "compose_frame_indices", "render_prediction"]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = {"compose_frame_indices": ".window", "render_prediction": ".presentation"}.get(name, ".predictor")
    value = getattr(import_module(module, __name__), name)
    globals()[name] = value
    return value
