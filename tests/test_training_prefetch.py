# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import random
from threading import Event, enumerate as threads
from types import SimpleNamespace

import pytest
import torch

from grana.data.sampling import RoleSampler, _image_digest
from grana.training.prefetch import SamplePrefetcher
from test_roles import dataset


@pytest.mark.parametrize("roles,accumulation,order", [
    ({"A": 2, "B": 1, "C": 1, "D": 0}, 4, ("A", "A", "B", "C") * 4),
    ({"A": 2, "B": 4, "C": 3, "D": 3}, 4, ("A", "B", "B", "C_paired", "C_paired", "C_paired", "A", "B", "B") * 2),
    ({"A": 2, "B": 4, "C": 6, "D": 3}, 4,
     ("A", "B", "B", "C_paired", "C_paired", "C_paired", "C_independent", "C_independent",
      "A", "B", "B", "C_independent") * 2),
])
@pytest.mark.parametrize("enabled", [False, True])
def test_real_sampling_pixels_metadata_rng_and_budget_are_unchanged(dataset, roles, accumulation, order, enabled):
    settings = {"train_frame_sampling": {
        "frame_selection_profile_weights": {"regular_ctsp": .3, "irregular_stratified": .7},
        "query_order_profile_weights": {"forward": .2, "backward": .3, "tail_rewind": .5},
    }}
    reference = RoleSampler(dataset, settings, seed=827)
    candidate = RoleSampler(dataset, settings, seed=827)
    global_rng, torch_rng = random.getstate(), torch.random.get_rng_state().clone()
    with SamplePrefetcher(candidate, roles, accumulation, 6, enabled=enabled) as stream:
        for role in order * 6:
            expected, actual = reference.sample(role), stream.sample(role)
            assert {k: v for k, v in vars(actual).items() if k != "images"} == {
                k: v for k, v in vars(expected).items() if k != "images"}
            assert _image_digest(actual.images) == _image_digest(expected.images)
            if role == "C_paired":
                derived = candidate.make_d(actual, 1)
                assert derived.images is actual.images
                assert derived.targets == reference.make_d(expected, 1).targets
        with pytest.raises(RuntimeError, match="budget"):
            stream.sample(order[0])
    assert candidate.state_dict() == reference.state_dict()
    assert random.getstate() == global_rng
    assert torch.equal(torch.random.get_rng_state(), torch_rng)


def test_read_ahead_is_bounded_and_early_close_joins_the_producer():
    calls = []
    second_update_ready = Event()

    def sample(role):
        calls.append(role)
        if len(calls) == 6:
            second_update_ready.set()
        return SimpleNamespace(role=role)

    with SamplePrefetcher(SimpleNamespace(sample=sample), dict(A=1, B=1, C=1, D=0), 1, 100, enabled=True) as stream:
        stream.sample("A")
        assert second_update_ready.wait(5)
    assert calls == ["A", "B", "C"] * 2
    assert not any(thread.name.startswith("grana-sample-prefetch") for thread in threads())


def test_producer_failure_reaches_consumer_and_cleanup_still_runs():
    calls = []

    def sample(role):
        calls.append(role)
        if len(calls) == 4:
            raise OSError("missing training RGB")
        return role

    with pytest.raises(OSError, match="missing training RGB"):
        with SamplePrefetcher(SimpleNamespace(sample=sample), dict(A=1, B=1, C=1, D=0), 1, 10, enabled=True) as stream:
            assert [stream.sample(r) for r in "ABC"] == list("ABC")
            stream.sample("A")
    assert len(calls) == 4
    assert not any(thread.name.startswith("grana-sample-prefetch") for thread in threads())


@pytest.mark.parametrize("enabled", [False, True])
def test_order_mismatch_is_detected_before_consuming_a_row(enabled):
    with SamplePrefetcher(SimpleNamespace(sample=lambda r: r), dict(A=1, B=1, C=0, D=0), 2, 1, enabled=enabled) as stream:
        with pytest.raises(RuntimeError, match="expected A, got B"):
            stream.sample("B")
        assert [stream.sample(r) for r in "ABAB"] == list("ABAB")
    with pytest.raises(RuntimeError, match="budget"):
        stream.sample("A")
