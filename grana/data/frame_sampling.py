# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Independent eight-observation selection and query-order sampling.

Canonical order means the selector's base occurrence order, not necessarily
chronological order. In particular, regular CTSP preserves its historical
context/local ordering at boundaries. Every occurrence remains supervised.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real
import random


DEFAULT_TRAIN_STRIDE_PAIRS = ((10, 2), (16, 4), (24, 6), (32, 8))
DEFAULT_EVAL_STRIDE_PAIRS = ((16, 4),)
FRAME_SELECTION_PROFILES = ("regular_ctsp", "irregular_stratified")
QUERY_ORDER_PROFILES = ("forward", "backward", "tail_rewind")
FRAME_SELECTION_KEY = "frame_selection_profile_weights"
QUERY_ORDER_KEY = "query_order_profile_weights"
IDENTITY_ORDER = tuple(range(8))
QUERY_ORDERS = {"forward": IDENTITY_ORDER, "backward": tuple(reversed(IDENTITY_ORDER)),
                "tail_rewind": (0, 1, 2, 7, 6, 5, 4, 3)}


def default_train_frame_sampling() -> dict:
    return {FRAME_SELECTION_KEY: dict(zip(FRAME_SELECTION_PROFILES, (0.6, 0.4))),
            QUERY_ORDER_KEY: dict(zip(QUERY_ORDER_PROFILES, (0.5, 0.2, 0.3)))}


def normalize_train_frame_sampling(config: dict) -> dict:
    """Resolve partial maps and normalize in a fixed, configuration-independent order."""
    prefix = "data.train_frame_sampling"
    if not isinstance(config, dict):
        raise ValueError(f"{prefix} must be a mapping")
    result = default_train_frame_sampling()
    if set(config) - set(result):
        raise ValueError(f"Unknown configuration key in {prefix}: {sorted(set(config) - set(result), key=str)}")
    for key, defaults in result.items():
        supplied = config.get(key, {})
        path = f"{prefix}.{key}"
        if not isinstance(supplied, dict):
            raise ValueError(f"{path} must be a mapping")
        if set(supplied) - set(defaults):
            raise ValueError(f"Unknown configuration key in {path}: {sorted(set(supplied) - set(defaults), key=str)}")
        weights = {name: supplied.get(name, value) for name, value in defaults.items()}
        for name, value in weights.items():
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError(f"{path}.{name} must be finite, nonnegative and nonboolean")
            try:
                finite = math.isfinite(value)
            except OverflowError:
                finite = False
            if not finite or value < 0:
                raise ValueError(f"{path}.{name} must be finite and nonnegative")
        maximum = max(weights.values())
        if maximum <= 0:
            raise ValueError(f"{path} must have positive total weight")
        # Scaling first avoids overflow when multiple finite weights are near 1e308.
        scaled = {name: value / maximum for name, value in weights.items()}
        total = math.fsum(scaled.values())
        result[key] = {name: value / total for name, value in scaled.items()}
    return result


def choose_profile(weights: dict[str, float], rng: random.Random) -> str:
    draw, cumulative = rng.random(), 0.0
    last_positive = None
    for name, weight in weights.items():
        if weight > 0:
            last_positive = name
            cumulative += weight
            if draw < cumulative:
                return name
    assert last_positive is not None
    return last_positive  # Floating-point normalization can end just below one.


def _validate_geometry(poi: int, num_frames: int, context_stride: int, tail_stride: int) -> None:
    if any(isinstance(x, bool) or not isinstance(x, Integral) or x <= 0
           for x in (num_frames, context_stride, tail_stride)):
        raise ValueError("Frame count and strides must be positive integers")
    if isinstance(poi, bool) or not isinstance(poi, Integral) or not 0 <= poi < num_frames:
        raise ValueError("POI must be a real source-frame index")


def regular_frame_indices(poi: int, num_frames: int, context_stride: int, tail_stride: int) -> tuple[int, ...]:
    """Original CTSP occurrence order, including supervised clipped duplicates."""
    _validate_geometry(poi, num_frames, context_stride, tail_stride)
    past = ((poi - 3 * context_stride, poi - 2 * context_stride, poi - context_stride)
            if poi >= 3 * context_stride else (0, poi // 3, 2 * poi // 3))
    tail = tuple(max(0, min(num_frames - 1, poi + offset * tail_stride)) for offset in (-2, -1, 0, 1, 2))
    return past + tail


@dataclass(frozen=True)
class FrameSamplingDecision:
    frame_indices: tuple[int, ...]
    poi_slot: int
    frame_weight_order: tuple[int, ...]
    frame_selection_profile: str
    query_order_profile: str


def sample_frame_indices(poi: int, num_frames: int, context_stride: int, tail_stride: int, *,
                         frame_selection_profile: str, query_order_profile: str,
                         rng: random.Random) -> FrameSamplingDecision:
    """Select eight base occurrences, then apply an exact permutation without redrawing."""
    _validate_geometry(poi, num_frames, context_stride, tail_stride)
    if frame_selection_profile not in FRAME_SELECTION_PROFILES:
        raise ValueError("Unknown frame_selection_profile")
    if query_order_profile not in QUERY_ORDER_PROFILES:
        raise ValueError("Unknown query_order_profile")
    c, t = context_stride, tail_stride
    span = max(3 * c, 2 * t + 3)
    if frame_selection_profile == "regular_ctsp":
        base, poi_position = regular_frame_indices(poi, num_frames, c, t), 5
    else:
        if poi >= span:
            bounds = tuple(poi - span + i * (span - 2 * t) // 3 for i in range(4))
            past = tuple(rng.randrange(bounds[i], bounds[i + 1]) for i in range(3))
        else:
            past = (0, poi // 3, 2 * poi // 3)
        def distances() -> tuple[int, int]:
            # A two-draw rank map handles even strides larger than sys.maxsize.
            first, second = rng.randrange(2 * t), rng.randrange(2 * t - 1)
            second += second >= first
            return tuple(sorted((first + 1, second + 1)))
        left_near, left_far = distances()
        right_near, right_far = distances()
        local = (poi - left_far, poi - left_near, poi, poi + right_near, poi + right_far)
        base = past + tuple(max(0, min(num_frames - 1, index)) for index in local)
        poi_position = 5
    order = QUERY_ORDERS[query_order_profile]
    return FrameSamplingDecision(tuple(base[index] for index in order), order.index(poi_position), order,
                                 frame_selection_profile, query_order_profile)
