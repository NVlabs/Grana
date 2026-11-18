# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compose inference observations with the training sampler's canonical geometry."""
from __future__ import annotations

from numbers import Integral

from grana.data.frame_sampling import DEFAULT_EVAL_STRIDE_PAIRS, regular_frame_indices


def compose_frame_indices(frame_index: int, num_frames: int, *,
                          prediction_frame: str = "poi") -> tuple[int, ...]:
    """Select eight occurrences using the current evaluation CTSP stride pair.

    ``poi`` queries canonical slot 5 and may observe later source frames.
    ``last`` queries slot 7, aligns it with ``frame_index`` and uses only frames
    already available at that time, even when the supplied array is longer.
    Preserve the training sampler's clipped duplicates and boundary ordering.
    Indices/strides refer to source array frames, not a wall-clock frame rate.
    """
    if (isinstance(num_frames, bool) or not isinstance(num_frames, Integral)
            or num_frames <= 0):
        raise ValueError("num_frames must be a positive integer")
    if (isinstance(frame_index, bool) or not isinstance(frame_index, Integral)
            or not 0 <= frame_index < num_frames):
        raise ValueError("frame_index must be an in-range source frame index")
    if prediction_frame not in ("poi", "last"):
        raise ValueError("prediction_frame must be poi or last for window composition")
    context_stride, tail_stride = DEFAULT_EVAL_STRIDE_PAIRS[0]
    if prediction_frame == "last":
        anchor = max(0, int(frame_index) - 2 * tail_stride)
        available = int(frame_index) + 1
    else:
        anchor, available = int(frame_index), int(num_frames)
    return regular_frame_indices(anchor, available, context_stride, tail_stride)
