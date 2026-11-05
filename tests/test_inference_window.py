# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from grana.data.sampling import frame_indices
from grana.inference.window import compose_frame_indices


def test_last_and_poi_query_the_same_training_window_at_different_positions():
    expected = (72, 88, 104, 112, 116, 120, 124, 128)
    assert compose_frame_indices(120, 500) == expected
    assert compose_frame_indices(120, 500, prediction_frame="poi") == expected
    assert compose_frame_indices(128, 500, prediction_frame="last") == expected
    assert expected == frame_indices(120, 500, 16, 4)


@pytest.mark.parametrize("index", [0, 1, 3, 4, 7, 8, 9, 15, 16, 47, 48, 127, 199])
def test_last_has_no_lookahead_including_start_boundary_and_recorded_future(index):
    offline_array = compose_frame_indices(index, 200, prediction_frame="last")
    streaming_buffer = compose_frame_indices(index, index + 1, prediction_frame="last")
    assert offline_array == streaming_buffer
    assert len(offline_array) == 8 and offline_array[-1] == index
    assert all(0 <= value <= index for value in offline_array)
    assert offline_array == frame_indices(max(0, index - 8), index + 1, 16, 4)


def test_poi_preserves_training_boundary_order_and_clipped_supervised_duplicates():
    assert compose_frame_indices(9, 20, prediction_frame="poi") == (0, 3, 6, 1, 5, 9, 13, 17)
    assert compose_frame_indices(19, 20, prediction_frame="poi")[-3:] == (19, 19, 19)


@pytest.mark.parametrize("index,count,mode", [
    (True, 20, "last"), (1.5, 20, "poi"), (-1, 20, "last"),
    (20, 20, "last"), (0, 0, "poi"), (0, True, "last"), (0, 20, "custom"),
])
def test_bad_window_geometry_fails(index, count, mode):
    with pytest.raises(ValueError):
        compose_frame_indices(index, count, prediction_frame=mode)
