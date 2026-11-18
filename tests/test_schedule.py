# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from grana.training.engine import learning_rate_for_step


def test_formal_schedule_warms_then_decays_without_skipping_final_update():
    training = {"learning_rate": 1e-5, "warmup_steps": 20,
                "max_steps": 400, "lr_schedule": "cosine"}
    rates = [learning_rate_for_step(training, step) for step in range(1, 401)]
    assert rates[0] == pytest.approx(5e-7)
    assert rates[19] == rates[20] == training["learning_rate"]
    assert all(a <= b for a, b in zip(rates[:19], rates[1:20]))
    assert all(a >= b for a, b in zip(rates[20:-1], rates[21:]))
    assert 0 < rates[-1] < rates[0]


def test_constant_schedule_preserves_short_recipe_behavior():
    training = {"learning_rate": 1e-5, "warmup_steps": 0,
                "max_steps": 2, "lr_schedule": "constant"}
    assert [learning_rate_for_step(training, step) for step in (1, 2)] == [1e-5, 1e-5]
