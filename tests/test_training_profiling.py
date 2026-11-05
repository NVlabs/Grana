# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from grana.training.profiling import UpdateProfiler


@pytest.mark.parametrize("enabled", [False, True])
def test_profiling_preserves_autograd_rng_and_exceptions(enabled):
    torch.manual_seed(42)
    initial_rng = torch.random.get_rng_state()
    parameter = torch.ones(16, requires_grad=True)
    expected = torch.nn.functional.dropout(parameter, p=.25).square().sum()
    expected.backward()
    expected_gradient = parameter.grad.clone()
    expected_rng = torch.random.get_rng_state()
    torch.random.set_rng_state(initial_rng)
    parameter.grad = None
    profiler = UpdateProfiler(enabled=enabled, device="cpu")
    with profiler.section("forward", cuda=True):
        loss = torch.nn.functional.dropout(parameter, p=.25).square().sum()
    with profiler.section("backward", cuda=True):
        loss.backward()
    with pytest.raises(RuntimeError, match="propagated"):
        with profiler.section("failure"):
            raise RuntimeError("propagated")
    torch.testing.assert_close(parameter.grad, expected_gradient, rtol=0, atol=0)
    assert torch.equal(expected_rng, torch.random.get_rng_state())
    metrics = profiler.metrics()
    if enabled:
        assert set(metrics) == {f"time/profile/{name}_host_seconds" for name in ("forward", "backward", "failure")}
        assert all(value >= 0 for value in metrics.values())
    else:
        assert metrics == {}


def test_disabled_profile_never_creates_cuda_events(monkeypatch):
    monkeypatch.setattr(torch.cuda, "Event", lambda **kwargs: pytest.fail("disabled profiling touched CUDA"))
    profiler = UpdateProfiler(enabled=False, device="cuda:0")
    with profiler.section("forward", cuda=True):
        pass
    assert profiler.metrics() == {}
