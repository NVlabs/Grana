# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from copy import deepcopy

from grana.inference.resources import (autotune_batch_size, balanced_shards,
                                       compare_batch_predictions, is_cuda_oom, select_devices)


def measurement(size, *, throughput=None, peak=None):
    windows = 2 * size
    return {"peak_memory_bytes": size * 100 if peak is None else peak,
            "elapsed_seconds": windows / (throughput if throughput is not None else size * 10),
            "windows_measured": windows}


def test_all_visible_devices_default_and_explicit_user_override():
    assert select_devices("auto", 8, 24) == list(range(8))
    assert select_devices("6,2", 8, 24) == [6, 2]
    assert select_devices("auto", 8, 2) == [0, 1]
    with pytest.raises(RuntimeError, match="CUDA"):
        select_devices("auto", 0, 24)
    for spec in ("0,0", "8", "-1", "", "0,,2"):
        with pytest.raises(ValueError):
            select_devices(spec, 8, 24)


def test_trajectory_shards_use_every_device_without_losing_or_duplicating_work():
    tasks = [{"key": str(index), "cost": cost} for index, cost in enumerate([128] * 22 + [70, 77])]
    shards = balanced_shards(tasks, list(range(8)))
    assert set(shards) == set(range(8)) and all(shards.values())
    assert sorted(row["key"] for rows in shards.values() for row in rows) == sorted(row["key"] for row in tasks)
    loads = [sum(row["cost"] for row in rows) for rows in shards.values()]
    assert max(loads) - min(loads) <= max(row["cost"] for row in tasks)
    assert shards == balanced_shards(list(reversed(tasks)), list(range(8)))


def test_autotune_uses_real_memory_measurements_and_refines_failed_bound():
    attempts, cleanups = [], []
    def probe(size):
        attempts.append(size)
        return measurement(size)
    size, receipt = autotune_batch_size(probe, 32, 1000, cleanup=lambda: cleanups.append(True))
    assert size == 10 and receipt["selected_peak_memory_bytes"] == 1000
    assert attempts == [32, 16, 8, 12, 10]
    assert len(cleanups) == len(attempts)
    assert [row["outcome"] for row in receipt["probes"]].count("memory_budget") == 3
    assert receipt["selected_throughput_requests_per_second"] == 100
    assert receipt["stop_reason"] == "memory_budget"
    assert receipt["initial_probe_batch_size"] == 32
    assert receipt["selection_policy"]["objective"] == "memory_capacity"


def test_cuda_oom_is_recoverable_but_other_model_errors_are_not():
    def probe(size):
        if size > 3:
            raise RuntimeError("CUDA out of memory. Tried to allocate a tensor")
        return measurement(size)
    size, receipt = autotune_batch_size(probe, 8, 1000)
    assert size == 3 and any(row["outcome"] == "cuda_oom" for row in receipt["probes"])
    assert receipt["stop_reason"] == "cuda_oom"
    cleanups = []
    def wrong(_):
        raise ValueError("Different model output shape")
    with pytest.raises(ValueError, match="Different model"):
        autotune_batch_size(wrong, 8, 1000, cleanup=lambda: cleanups.append(True))
    assert cleanups == [True]


def test_wrapped_cuda_oom_does_not_hide_unrelated_errors():
    wrapped = RuntimeError("Compiler wrapper")
    wrapped.__cause__ = RuntimeError("CUDA error: out of memory")
    assert is_cuda_oom(wrapped)
    torch_oom = type("OutOfMemoryError", (RuntimeError,), {"__module__": "torch"})
    assert is_cuda_oom(torch_oom("allocation failed"))
    assert not is_cuda_oom(RuntimeError("CPU allocator failed"))
    assert not is_cuda_oom(ValueError("Invalid frame selection"))


def test_oom_context_suppressed_by_a_new_error_is_not_recoverable():
    try:
        try:
            raise RuntimeError("CUDA out of memory")
        except RuntimeError:
            raise ValueError("Invalid output schema") from None
    except ValueError as error:
        assert not is_cuda_oom(error)


def test_small_workload_bounds_probe_and_batch_one_failure_is_explicit():
    attempts = []
    def probe(size):
        attempts.append(size)
        return measurement(size, peak=100)
    selected, receipt = autotune_batch_size(probe, 7, 200)
    assert selected == 7 and receipt["stop_reason"] == "max_batch_size"
    assert attempts == [7]
    with pytest.raises(RuntimeError, match="batch size 1 failed"):
        autotune_batch_size(probe, 7, 50)


@pytest.mark.parametrize("invalid", [None,
    {"peak_memory_bytes": -1, "elapsed_seconds": 1, "windows_measured": 1},
    {"peak_memory_bytes": True, "elapsed_seconds": 1, "windows_measured": 1},
    *[{"peak_memory_bytes": 100, "elapsed_seconds": seconds, "windows_measured": 1}
      for seconds in (0, -1, True, float("nan"), float("inf"), 5e-324)],
    *[{"peak_memory_bytes": 100, "elapsed_seconds": 1, "windows_measured": windows}
      for windows in (0, -1, True, 1.5, None)],
    {"peak_memory_bytes": 100, "elapsed_seconds": 1}])
def test_invalid_measurements_do_not_produce_a_batch_choice(invalid):
    with pytest.raises(ValueError):
        autotune_batch_size(lambda size: invalid, 1, 1000)


def test_probe_counts_only_complete_measured_physical_batches():
    def probe(size):
        return {**measurement(size), "windows_measured": 1 if size == 1 else 3}
    with pytest.raises(ValueError, match="whole number"):
        autotune_batch_size(probe, 8, 1000)


def test_auto_ignores_prediction_comparison_fields():
    def probe(size):
        return {**measurement(size), "predictions_compatible": False, "max_abs_D_difference": .04}
    size, receipt = autotune_batch_size(probe, 32, 100_000)
    assert size == 32
    assert all(item["outcome"] == "ok" for item in receipt["probes"])
    assert receipt["selected_throughput_requests_per_second"] == 320


def test_capacity_growth_starts_at_32_and_does_not_stop_for_falling_throughput():
    rates = {32: 20, 64: 18, 128: 16, 256: 8}
    attempts = []
    def probe(size):
        attempts.append(size)
        return measurement(size, throughput=rates[size])
    selected, receipt = autotune_batch_size(probe, 256, 100_000)
    assert selected == 256
    assert attempts == [32, 64, 128, 256]
    assert receipt["selected_throughput_requests_per_second"] == 8
    assert receipt["maximum_measured_throughput_requests_per_second"] == 20
    assert receipt["largest_fitting_batch_size"] == 256
    assert receipt["stop_reason"] == "max_batch_size"
    assert receipt["selection_policy"]["throughput_role"] == "report_only"


def test_growth_and_bounded_refinement_choose_only_measured_fitting_capacity():
    attempts = []
    def probe(size):
        attempts.append(size)
        return measurement(size, throughput=10)
    selected, receipt = autotune_batch_size(probe, 256, 11_000)
    assert attempts == [32, 64, 128, 96, 112]
    assert selected == 96
    assert receipt["selected_peak_memory_bytes"] == 9600
    assert receipt["stop_reason"] == "memory_budget"


def test_initial_oom_backoff_finds_fit_then_refines_upward():
    attempts = []
    def probe(size):
        attempts.append(size)
        if size > 20:
            raise RuntimeError("CUDA out of memory")
        return measurement(size)
    selected, receipt = autotune_batch_size(probe, 128, 100_000)
    assert attempts == [32, 16, 24, 20]
    assert selected == 20 and receipt["stop_reason"] == "cuda_oom"


def test_fitting_32_is_never_reduced_for_timings_and_user_cap_is_respected():
    attempts = []
    def probe(size):
        attempts.append(size)
        return measurement(size, throughput=.01)
    selected, receipt = autotune_batch_size(probe, 32, 3200)
    assert selected == 32 and attempts == [32]
    assert receipt["initial_probe_batch_size"] == 32
    attempts.clear()
    selected, receipt = autotune_batch_size(probe, 19, 3200)
    assert selected == 19 and attempts == [19]
    assert receipt["initial_probe_batch_size"] == 19


def test_memory_refinement_chooses_larger_capacity_even_when_smaller_batch_was_faster():
    attempts = []
    def probe(size):
        attempts.append(size)
        return measurement(size, throughput=60 if size == 12 else size * 10)
    selected, receipt = autotune_batch_size(probe, 64, 1300)
    assert attempts == [32, 16, 8, 12, 14]
    assert selected == 12 and receipt["largest_fitting_batch_size"] == 12
    assert receipt["selected_peak_memory_bytes"] == 1200
    assert receipt["selected_throughput_requests_per_second"] == 60
    assert receipt["maximum_measured_throughput_requests_per_second"] == 80


def test_warmup_time_is_not_used_in_reported_warm_throughput():
    def probe(size):
        return {**measurement(size), "warmup_elapsed_seconds": size * 1000}
    selected, receipt = autotune_batch_size(probe, 4, 1000)
    assert selected == 4
    assert receipt["selected_throughput_requests_per_second"] == 40
    assert receipt["probes"][-1]["warmup_elapsed_seconds"] == 4000


def prediction():
    return {"stage": {"ids": [3] * 8}, "D_focus_stage": 3,
            "progress": {"D": {"values": [.5] * 8}}}


def test_numerical_qualification_checks_nonselected_slots_and_conditioned_focus():
    reference = [prediction(), prediction()]
    candidate = deepcopy(reference)
    candidate[0]["progress"]["D"]["values"][0] += .0005
    report = compare_batch_predictions(reference, candidate)
    assert report["predictions_compatible"] and report["windows_checked"] == 2
    candidate[1]["stage"]["ids"][0] = 6
    assert not compare_batch_predictions(reference, candidate)["predictions_compatible"]
    candidate = deepcopy(reference)
    candidate[1]["D_focus_stage"] = 6
    assert not compare_batch_predictions(reference, candidate)["matched_D_focus"]
    candidate = deepcopy(reference)
    candidate[0]["progress"]["D"]["values"][0] += .04
    report = compare_batch_predictions(reference, candidate)
    assert not report["predictions_compatible"]
    assert report["max_abs_D_difference"] == pytest.approx(.04)


def test_numerical_qualification_rejects_invalid_or_partial_outputs():
    with pytest.raises(ValueError, match="paired predictions"):
        compare_batch_predictions([], [])
    candidate = prediction()
    candidate["stage"]["ids"] = [3]
    with pytest.raises(ValueError, match="all eight"):
        compare_batch_predictions([prediction()], [candidate])
    candidate = prediction()
    candidate["progress"]["D"]["values"][0] = float("nan")
    with pytest.raises(ValueError, match="finite D"):
        compare_batch_predictions([prediction()], [candidate])
