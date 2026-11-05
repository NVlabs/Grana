# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from grana.inference.cli import load_request
from grana.inference.client import build_binary
from grana.inference.selection import select_prediction
from scripts.inference import evaluate_frames as offline
from scripts.inference import summarize_http_frames as summary


def fake_uuid(index):
    return f"GPU-{index:08x}-0000-0000-0000-000000000000"


class CudaUuidLike:
    def __str__(self):
        return "ABCDEF01-2345-6789-ABCD-EF0123456789"


@pytest.mark.parametrize("value,expected", [
    (CudaUuidLike(), "GPU-abcdef01-2345-6789-abcd-ef0123456789"),
    ("ABCDEF01-2345-6789-ABCD-EF0123456789", "GPU-abcdef01-2345-6789-abcd-ef0123456789"),
    ("GPU-ABCDEF01-2345-6789-ABCD-EF0123456789", "GPU-abcdef01-2345-6789-abcd-ef0123456789"),
    ("MIG-ABCDEF01-2345-6789-ABCD-EF0123456789", "MIG-abcdef01-2345-6789-abcd-ef0123456789"),
])
def test_cuda_uuid_supports_pytorch_objects_and_normalizes_complete_identifiers(value, expected):
    assert offline.cuda_uuid(SimpleNamespace(uuid=value)) == expected


def test_cuda_uuid_rejects_arbitrary_object_representations_and_duplicate_case_variants():
    with pytest.raises(RuntimeError, match="complete GPU/MIG UUID"):
        offline.cuda_uuid(SimpleNamespace(uuid=object()))
    variants = ["GPU-ABCDEF01-2345-6789-ABCD-EF0123456789", "abcdef01-2345-6789-abcd-ef0123456789"]
    with pytest.raises(ValueError, match="duplicate GPU UUID"):
        offline.isolated_device_mappings([0, 1], lambda index: SimpleNamespace(uuid=variants[index]))


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    def parent_properties(device):
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        selector = str(device) if visible is None else visible.split(",")[device]
        return SimpleNamespace(uuid=selector if selector.startswith("GPU-") else fake_uuid(int(selector)))

    # No test consults real CUDA hardware or initializes a CUDA context.
    monkeypatch.setattr(torch.cuda, "get_device_properties", parent_properties)
    source = tmp_path / "source"
    source.mkdir()
    plan = {"schema": "grana-http-frame-comparison-plan-v1", "seeds": [11, 29],
            "samples_per_trajectory": 3, "anchors_per_trajectory": 2, "stride_pair": [16, 4],
            "checkpoint": {"path": "/checkpoint", "weights_sha256": "a" * 64}, "trajectories": []}
    for index in range(8):
        key, frames = f"episode_{index}", 20
        image_path = source / f"{key}.npy"
        np.save(image_path, np.arange(frames * 2 * 3 * 3, dtype=np.uint8).reshape(frames, 2, 3, 3))
        request = source / f"{key}_request.json"
        request.write_text(json.dumps({"views": {"top": image_path.name}, "task": "Assemble",
                                       "stage_prompts": ["Connect", "Finish"]}))
        metadata = source / f"{key}_metadata.json"
        metadata.write_text(json.dumps({"stage_ids": [0] * frames, "within_stage_progress": [.8] * frames}))
        stat = image_path.stat()
        plan["trajectories"].append({"key": key, "trajectory_id": str(index),
            "group": "L1L2" if index < 4 else "L3L4", "num_frames": frames, "stage_count": 2,
            "source_fps": 10, "anchors": [0, 19], "targets": {"11": [0, 9, 19], "29": [0, 10, 19]},
            "request_path": request.name, "metadata_path": metadata.name,
            "request_sha256": offline.file_digest(request), "metadata_sha256": offline.file_digest(metadata),
            "sources": {"top": {"path": str(image_path), "shape": [frames, 2, 3, 3],
                                  "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}}})
    path = source / "plan.json"
    path.write_text(json.dumps(plan))
    args = SimpleNamespace(plan=path, output=tmp_path / "offline", devices="auto", batch_size="auto",
                           memory_fraction=.9, max_batch_size=None, seeds=None, model_name_or_path=None,
                           worker_device=0)
    return args, plan


class FakeCuda:
    def __init__(self, uuid=None):
        self.device, self.peak, self.cleanups = None, 100, 0
        self.uuid = fake_uuid(0) if uuid is None else uuid
        self.elapsed_seconds = 0.

    def device_count(self):
        return 1

    def get_device_properties(self, device):
        assert device == 0
        return SimpleNamespace(uuid=self.uuid)

    def set_device(self, device):
        self.device = device

    def synchronize(self, device):
        assert device == self.device

    def reset_peak_memory_stats(self, device):
        assert device == self.device
        self.peak = 100

    def max_memory_allocated(self, device):
        assert device == self.device
        return self.peak

    def memory_allocated(self, device):
        assert device == self.device
        return 100

    def mem_get_info(self, device):
        assert device == self.device
        return 1000, 2000

    def empty_cache(self):
        self.cleanups += 1


class FakePredictor:
    def __init__(self, cuda, *, maximum=8, failure=None, bytes_per_request=100, drift=None):
        self.cuda, self.maximum, self.failure, self.calls = cuda, maximum, failure, []
        self.bytes_per_request = bytes_per_request
        self.drift = drift
        self.checkpoint = {"weights_sha256": "a" * 64,
                           "inference_runtime": {"policy": "cuda_optimized_v1", "gpu_image_resize": True,
                                                 "cpu_threads": 8}}
        self.inference_runtime = deepcopy(self.checkpoint["inference_runtime"])

    def predict_batch(self, requests, roles, batch_size):
        assert roles == ("C", "D") and batch_size == len(requests)
        self.calls.append(len(requests))
        if self.failure:
            raise self.failure
        if batch_size > self.maximum:
            raise torch.OutOfMemoryError("CUDA out of memory in the synthetic test")
        # Each physical shape has an expensive first compile, followed by
        # deterministic warm timings whose throughput improves with batching.
        self.cuda.elapsed_seconds += (.01 + .001 * batch_size
                                      + (5. if self.calls.count(batch_size) == 1 else 0.))
        self.cuda.peak = 100 + self.bytes_per_request * len(requests)
        results = []
        for request in requests:
            assert not hasattr(request, "gt") and request.focus_stage is None
            assert all(len(images) == 8 for images in request.images.values())
            assert all(not image.flags.writeable for images in request.images.values() for image in images)
            ids = [frame % 2 for frame in request.frame_indices]
            focus = ids[request.poi_slot]
            values = [(frame % 10) / 10 for frame in request.frame_indices]
            if len(requests) > 1:
                if self.drift == "C":
                    ids[0] = (ids[0] + 1) % 2
                if self.drift == "D":
                    values[0] += .02
                if self.drift == "late_D" and request.frame_indices[request.poi_slot] == 19:
                    values[0] += .02
                if self.drift == "D_focus":
                    focus = (focus + 1) % 2
                if self.drift == "selected_C":
                    ids[request.poi_slot] = (ids[request.poi_slot] + 1) % 2
                    focus = ids[request.poi_slot]
            result = {"checkpoint": deepcopy(self.checkpoint), "roles": list(roles), "poi_slot": request.poi_slot,
                      "prediction_frame": request.prediction_frame, "frame_indices": list(request.frame_indices),
                      "stage": {"ids": ids}, "D_focus_stage": focus,
                      "progress": {"D": {"values": values, "focus_stage": focus}}}
            result["selected_prediction"] = select_prediction(result)
            results.append(result)
        return results


def execute_worker(args, monkeypatch, *, maximum=8, failure=None, bytes_per_request=100, drift=None):
    schedule = json.loads((args.output / "schedule.json").read_text())
    mapping = schedule["device_mappings"][str(args.worker_device)]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", mapping["worker_cuda_visible_devices"])
    cuda = FakeCuda(mapping["device_uuid"])
    predictor = FakePredictor(cuda, maximum=maximum, failure=failure, bytes_per_request=bytes_per_request, drift=drift)
    monkeypatch.setattr(torch, "set_num_threads", lambda count: None)
    monkeypatch.setattr(offline.time, "perf_counter", lambda: cuda.elapsed_seconds)

    def factory(path, model_name_or_path, device):
        assert path == "/checkpoint" and device == "cuda:0"
        assert model_name_or_path == args.model_name_or_path
        return predictor

    receipt = offline.run_worker(args, predictor_factory=factory, cuda=cuda)
    return receipt, predictor, cuda


def mock_sizing(monkeypatch, selected):
    monkeypatch.setattr(offline, "autotune_batch_size", lambda *a, **kw:
                        (selected, {"selected_batch_size": selected, "probes": []}))


@pytest.mark.parametrize("extra,expected", [([], 64), (["--batch-size", "auto"], "auto"),
                                          (["--batch-size", "32"], 32)])
def test_cli_defaults_to_per_gpu_64_and_keeps_explicit_batch_overrides(campaign, monkeypatch, extra, expected):
    args, _ = campaign
    monkeypatch.setattr(offline, "run", lambda parsed: parsed)
    parsed = offline.main(["--plan", str(args.plan), "--output", str(args.output), *extra])
    assert parsed.batch_size == expected and parsed.devices == "auto"


def test_default_fixed_64_skips_probing_and_records_per_gpu_capacity(campaign, monkeypatch):
    args, _ = campaign
    monkeypatch.setattr(offline, "run", lambda parsed: parsed)
    parsed = offline.main(["--plan", str(args.plan), "--output", str(args.output), "--devices", "0"])
    parsed.worker_device = 0
    plan, schedule, output = offline.prepare_campaign(parsed, device_count=8)

    def forbidden_probe(*args, **kwargs):
        pytest.fail("Default fixed batch 64 must not launch auto-sizing probes")

    monkeypatch.setattr(offline, "autotune_batch_size", forbidden_probe)
    receipt, predictor, _ = execute_worker(parsed, monkeypatch, maximum=64, bytes_per_request=10)
    assert predictor.calls == [64, 64, 32]
    assert receipt["sizing"]["probes"] == []
    assert receipt["initial_batch_size"] == receipt["final_batch_size"] == 64
    assert receipt["batch_size_scope"] == "per_gpu" and receipt["requested_batch_size_per_gpu"] == 64
    assert schedule["requested_batch_size_per_gpu"] == schedule["requested_aggregate_batch_capacity"] == 64
    merged = offline.merge_workers(plan, schedule, output, 1.)
    assert merged["final_per_gpu_batch_sizes"] == {"0": 64} and merged["final_aggregate_batch_capacity"] == 64
    assert merged["requests_written"] == 160


def test_default_balances_complete_trajectories_across_all_visible_gpus(campaign):
    args, original = campaign
    before = args.plan.read_bytes()
    plan, schedule, output = offline.prepare_campaign(args, device_count=8)
    assert schedule["devices"] == list(range(8))
    assert all(len(tasks) == 1 for tasks in schedule["shards"].values())
    assert 1 <= schedule["cpu_threads_per_worker"] <= 8
    tasks = [task for members in schedule["shards"].values() for task in members]
    assert {task["key"] for task in tasks} == {row["key"] for row in original["trajectories"]}
    assert sum(task["jobs"] for task in tasks) == len(offline.jobs_for(plan)) == 160
    assert args.plan.read_bytes() == before
    assert plan["execution"]["requested_batch_size"] == "auto"
    assert schedule["batch_size_scope"] == "per_gpu" and schedule["requested_batch_size_per_gpu"] == "auto"
    assert schedule["requested_aggregate_batch_capacity"] is None and schedule["workers_synchronized"] is False
    assert schedule["plan_sha256"] == offline.file_digest(output / "plan.json")
    assert all(Path(row["request_path"]).is_absolute() for row in plan["trajectories"])
    assert all(Path(row["metadata_path"]).is_absolute() for row in plan["trajectories"])
    with pytest.raises(FileExistsError, match="new offline"):
        offline.prepare_campaign(args, device_count=8)


def test_explicit_devices_batches_and_seed_subset_preserve_source_plan(campaign):
    args, original = campaign
    args.devices, args.batch_size, args.seeds = "3,1", 5, [29]
    plan, schedule, _ = offline.prepare_campaign(args, device_count=8)
    assert schedule["devices"] == [3, 1]
    assert plan["seeds"] == [29] and all(set(row["targets"]) == {"29"} for row in plan["trajectories"])
    assert len(offline.jobs_for(plan)) == 80
    assert json.loads(args.plan.read_text()) == original
    assert plan["execution"]["requested_batch_size"] == 5


@pytest.mark.parametrize("parent_mask,device_count,expected", [
    (None, 8, {"2": fake_uuid(2), "0": fake_uuid(0)}),
    ("7,2,5", 3, {"2": fake_uuid(5), "0": fake_uuid(7)}),
    (f"{fake_uuid(7)},{fake_uuid(2)},{fake_uuid(5)}", 3, {"2": fake_uuid(5), "0": fake_uuid(7)}),
])
def test_parent_logical_device_mapping_uses_cuda_properties(campaign, monkeypatch, parent_mask, device_count, expected):
    args, _ = campaign
    args.devices = "2,0"
    if parent_mask is not None:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", parent_mask)
    plan, schedule, _ = offline.prepare_campaign(args, device_count=device_count)
    assert schedule["devices"] == [2, 0]
    assert schedule["device_mappings"] == plan["execution"]["device_mappings"]
    for device, uuid in expected.items():
        assert schedule["device_mappings"][device] == {"parent_logical_device": int(device),
            "device_uuid": uuid, "parent_cuda_visible_devices": parent_mask,
            "worker_cuda_visible_devices": uuid, "worker_local_device": 0}
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == parent_mask


@pytest.mark.parametrize("uuid", [None, "0", "GPU-short", "GPU-a,GPU-b"])
def test_invalid_cuda_uuid_fails_instead_of_guessing_physical_device(campaign, uuid):
    args, _ = campaign
    with pytest.raises(RuntimeError, match="complete GPU/MIG UUID"):
        offline.prepare_campaign(args, device_count=8, get_device_properties=lambda index: SimpleNamespace(uuid=uuid))
    assert not args.output.exists()


def test_duplicate_cuda_uuid_is_rejected_before_output_creation(campaign):
    args, _ = campaign
    with pytest.raises(ValueError, match="duplicate GPU UUID"):
        offline.prepare_campaign(args, device_count=8,
                                 get_device_properties=lambda index: SimpleNamespace(uuid=fake_uuid(0)))
    assert not args.output.exists()


def test_nonzero_parent_device_uses_local_cuda_zero_and_keeps_audit_identity(campaign, monkeypatch):
    args, _ = campaign
    args.devices, args.worker_device, args.batch_size = "3", 3, 8
    plan, schedule, output = offline.prepare_campaign(args, device_count=8)
    receipt, _, cuda = execute_worker(args, monkeypatch)
    assert cuda.device == 0 and receipt["worker_device"] == 3
    assert receipt["device_mapping"] == {**schedule["device_mappings"]["3"],
                                          "actual_worker_device_uuid": fake_uuid(3)}
    records = [json.loads(line) for line in (output / "workers/gpu_3/predictions.jsonl").read_text().splitlines()]
    assert all(row["worker_device"] == 3 and row["worker_local_device"] == 0
               and row["worker_device_uuid"] == fake_uuid(3) for row in records)
    assert offline.merge_workers(plan, schedule, output, 1.)["device_mappings"] == schedule["device_mappings"]


@pytest.mark.parametrize("damage,match", [("mask", "visibility differs"), ("count", "exactly one"),
                                         ("uuid", "UUID differs")])
def test_worker_mapping_mismatch_fails_before_loading_model(campaign, monkeypatch, damage, match):
    args, _ = campaign
    _, schedule, _ = offline.prepare_campaign(args, device_count=8)
    uuid = schedule["device_mappings"]["0"]["device_uuid"]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", uuid if damage != "mask" else fake_uuid(7))
    cuda = FakeCuda(uuid if damage != "uuid" else fake_uuid(7))
    if damage == "count":
        cuda.device_count = lambda: 8

    def forbidden_model(*args, **kwargs):
        pytest.fail("A mapping mismatch must fail before model initialization")

    with pytest.raises(ValueError, match=match):
        offline.run_worker(args, predictor_factory=forbidden_model, cuda=cuda)
    assert not (args.output / "workers").exists()


@pytest.mark.parametrize("updates,match", [
    ({"memory_fraction": 0.}, "memory-fraction"),
    ({"memory_fraction": float("nan")}, "memory-fraction"),
    ({"max_batch_size": 0}, "max-batch-size"),
    ({"batch_size": 8, "max_batch_size": 4}, "exceeds"),
    ({"seeds": [11, 11]}, "distinct seeds"),
    ({"seeds": [99]}, "distinct seeds"),
])
def test_invalid_campaign_limits_fail_before_creating_output(campaign, updates, match):
    args, _ = campaign
    vars(args).update(updates)
    with pytest.raises(ValueError, match=match):
        offline.prepare_campaign(args, device_count=8)
    assert not args.output.exists()


def test_window_cache_reuses_mmaps_and_matches_public_binary_identity(campaign, monkeypatch):
    args, _ = campaign
    plan, _, _ = offline.prepare_campaign(args, device_count=8)
    row = plan["trajectories"][0]
    cache, calls, original_load = offline.WindowCache(plan), [], np.load

    def counted_load(*args, **kwargs):
        calls.append(args[0])
        return original_load(*args, **kwargs)

    monkeypatch.setattr(np, "load", counted_load)
    jobs = offline.jobs_for(plan, [row["key"]])
    for job in jobs:
        request = cache.request(job)
        assert request.frame_indices[request.poi_slot] == job["source_frame"]
        if job["mode"] == "last":
            assert max(request.frame_indices) <= job["source_frame"]
    assert len(calls) == 2  # One public validation, one persistent mmap.
    assert cache.labels == {}  # Request composition never reads labels.
    first = jobs[0]
    assert cache.request(first) is cache.request({**first, "seed": 999, "kind": "quality"})
    public = load_request(Path(row["request_path"]), frame_index=first["source_frame"], prediction_frame=first["mode"])
    assert cache.input_digest(first, cache.request(first)) == hashlib.sha256(
        build_binary(public, ("C", "D"), request_id="identity")).hexdigest()
    assert cache.ground_truth(first) == {"stage_id": 0, "within_stage_progress": .8}


def test_auto_worker_probes_real_batches_and_complete_merge_is_summarizable(campaign, monkeypatch):
    args, _ = campaign
    args.devices = "0"
    plan, schedule, output = offline.prepare_campaign(args, device_count=8)
    receipt, predictor, cuda = execute_worker(args, monkeypatch)
    assert receipt["initial_batch_size"] == receipt["final_batch_size"] == 8
    assert receipt["cuda_memory"]["memory_budget_bytes"] == 1000
    assert receipt["cuda_memory"]["maximum_evaluation_peak_bytes"] == 900
    assert receipt["requests_written"] == 160 and receipt["batch_histogram"] == {8: 20}
    assert receipt["sizing"]["probes"][0]["batch_size"] == 32
    assert any(probe["outcome"] == "cuda_oom" for probe in receipt["sizing"]["probes"])
    assert cuda.cleanups == 1 + len(receipt["sizing"]["probes"])
    assert max(predictor.calls) > receipt["final_batch_size"]
    assert receipt["sizing"]["probe_policy"]["representative_windows"] == 32
    assert receipt["sizing"]["probe_policy"]["representative_modes"] == {"poi": 16, "last": 16}
    first = next(probe for probe in receipt["sizing"]["probes"] if probe["outcome"] == "ok")
    assert first["warmup_elapsed_seconds"] == pytest.approx(5.018)
    assert first["elapsed_seconds"] == pytest.approx(.036)
    assert first["windows_measured"] == 16 and first["warmup_batches"] == 1 and first["timed_batches"] == 2
    assert receipt["sizing"]["selected_throughput_requests_per_second"] == pytest.approx(8 / .018)
    assert predictor.calls[:5] == [32, 16, 8, 8, 8]
    assert not (output / "workers/gpu_0/qualification_references.json").exists()
    rows = [json.loads(line) for line in (output / "workers/gpu_0/predictions.jsonl").read_text().splitlines()]
    assert len({offline.identity(row) for row in rows}) == 160
    assert all("client_total_ms" not in row and "server_handler_ms" not in row for row in rows)
    assert all(row["execution"] == "local_batched" and row["transport"] == "local"
               and row["input_identity_only"] is True for row in rows)
    assert all(row["inference_amortized_ms"] == pytest.approx(row["inference_batch_seconds"] * 1000 / row["batch_size"])
               for row in rows)
    result = offline.merge_workers(plan, schedule, output, elapsed_seconds=12.)
    assert result["all_expected_predictions_present"] is True and result["requests_written"] == 160
    report = summary.summarize(output, bootstrap_repetitions=10)
    assert report["schema"] == "grana-local-frame-comparison-summary-v1"
    assert report["sampling"]["quality_requests"] == 96
    assert report["sampling"]["anchor_requests"] == 64
    assert report["anchor_determinism"]["max_abs_D_difference"] == 0


def test_auto_runtime_oom_halves_before_writing_and_keeps_every_pair(campaign, monkeypatch):
    args, _ = campaign
    args.devices = "0"
    plan, schedule, output = offline.prepare_campaign(args, device_count=8)
    mock_sizing(monkeypatch, 8)
    receipt, predictor, cuda = execute_worker(args, monkeypatch, maximum=4)
    assert predictor.calls[:2] == [8, 4]
    assert receipt["initial_batch_size"] == 8 and receipt["final_batch_size"] == 4
    assert receipt["batch_histogram"] == {4: 40}
    assert receipt["runtime_oom_retries"][0]["offset"] == 0
    assert receipt["runtime_oom_retries"][0]["retry_batch_size"] == 4
    assert cuda.cleanups == 2
    merged = offline.merge_workers(plan, schedule, output, 1.)
    assert merged["requests_written"] == 160
    assert merged["initial_per_gpu_batch_sizes"] == {"0": 8}
    assert merged["final_per_gpu_batch_sizes"] == {"0": 4}
    assert merged["initial_aggregate_batch_capacity"] == 8 and merged["final_aggregate_batch_capacity"] == 4


@pytest.mark.parametrize("drift", ["C", "D", "selected_C", "late_D"])
def test_valid_output_drift_never_blocks_memory_capacity_selection(campaign, monkeypatch, drift):
    args, _ = campaign
    args.devices = "0"
    offline.prepare_campaign(args, device_count=8)
    receipt, _, _ = execute_worker(args, monkeypatch, drift=drift)
    assert receipt["initial_batch_size"] == receipt["final_batch_size"] == 8
    assert receipt["batch_histogram"] == {8: 20}
    assert "qualified_batch_sizes" not in receipt["sizing"]
    assert all(probe["outcome"] != "prediction_mismatch" and "predictions_compatible" not in probe
               for probe in receipt["sizing"]["probes"])


def test_probe_still_rejects_invalid_c_conditioned_d_request_semantics(campaign, monkeypatch):
    args, _ = campaign
    args.devices = "0"
    offline.prepare_campaign(args, device_count=8)
    with pytest.raises(ValueError, match="D focus does not match C"):
        execute_worker(args, monkeypatch, drift="D_focus")
    assert not (args.output / "workers/gpu_0/receipt.json").exists()


def test_probe_above_sample_count_keeps_actual_physical_size_and_user_cap(campaign, monkeypatch):
    args, _ = campaign
    args.devices, args.max_batch_size = "0", 64
    offline.prepare_campaign(args, device_count=8)
    receipt, predictor, _ = execute_worker(args, monkeypatch, maximum=64, bytes_per_request=10)
    assert receipt["final_batch_size"] == 64
    selected = receipt["sizing"]["probes"][-1]
    assert selected["batch_size"] == 64 and selected["windows_measured"] == 128
    assert max(predictor.calls) == 64 and receipt["batch_histogram"] == {64: 2, 32: 1}


def test_auto_selects_larger_fitting_capacity_even_when_its_throughput_is_lower(campaign, monkeypatch):
    args, _ = campaign
    args.devices, args.max_batch_size = "0", 64
    offline.prepare_campaign(args, device_count=8)
    original = FakePredictor.predict_batch

    def slow_large_batch(self, requests, roles, batch_size):
        results = original(self, requests, roles, batch_size)
        if batch_size == 64:
            self.cuda.elapsed_seconds += 1.
        return results

    monkeypatch.setattr(FakePredictor, "predict_batch", slow_large_batch)
    receipt, _, _ = execute_worker(args, monkeypatch, maximum=64, bytes_per_request=10)
    probes = {probe["batch_size"]: probe for probe in receipt["sizing"]["probes"]}
    assert probes[64]["throughput_requests_per_second"] < probes[32]["throughput_requests_per_second"]
    assert receipt["initial_batch_size"] == receipt["final_batch_size"] == 64
    assert receipt["batch_histogram"] == {64: 2, 32: 1}


@pytest.mark.parametrize("device_count,aggregate_capacity", [(2, 64), (6, 192)])
def test_explicit_batch_size_32_is_per_gpu_and_worker_capacities_add(campaign, monkeypatch, device_count, aggregate_capacity):
    args, original = campaign
    original["samples_per_trajectory"] = 6
    for row in original["trajectories"]:
        row["targets"] = {str(seed): [0, 4, 8, 12, 16, 19] for seed in original["seeds"]}
    args.plan.write_text(json.dumps(original))
    args.batch_size = 32
    plan, schedule, output = offline.prepare_campaign(args, device_count=device_count)
    assert schedule["requested_batch_size_per_gpu"] == 32
    assert schedule["requested_aggregate_batch_capacity"] == aggregate_capacity
    assert plan["execution"]["requested_aggregate_batch_capacity"] == aggregate_capacity
    assert schedule["workers_synchronized"] is False
    receipts = []
    for device in schedule["devices"]:
        args.worker_device = device
        receipt, _, _ = execute_worker(args, monkeypatch, maximum=32, bytes_per_request=10)
        receipts.append(receipt)
        assert receipt["sizing"]["requested_batch_size"] == 32
        assert receipt["batch_size_scope"] == "per_gpu" and receipt["requested_batch_size_per_gpu"] == 32
        assert receipt["initial_batch_size"] == receipt["final_batch_size"] == 32
        assert set(receipt["batch_histogram"]) == {32}
    assert sum(receipt["final_batch_size"] for receipt in receipts) == aggregate_capacity
    merged = offline.merge_workers(plan, schedule, output, 1.)
    assert merged["requests_written"] == 256
    assert merged["initial_per_gpu_batch_sizes"] == merged["final_per_gpu_batch_sizes"] == {
        str(device): 32 for device in schedule["devices"]}
    assert merged["initial_aggregate_batch_capacity"] == merged["final_aggregate_batch_capacity"] == aggregate_capacity
    assert merged["batch_size_scope"] == "per_gpu" and merged["workers_synchronized"] is False


@pytest.mark.parametrize("size,cap,expected", [(96, None, 48), (48, None, 32), (33, None, 32),
                                             (32, None, 16), (20, None, 10), (48, 16, 16), (12, 12, 6)])
def test_auto_shrinking_preserves_32_until_it_fails_and_respects_smaller_limits(size, cap, expected):
    assert offline.smaller_auto_batch(size, cap) == expected


@pytest.mark.parametrize("maximum,bytes_per_request,expected,reason", [
    (32, 10, 32, "cuda_oom"), (64, 28, 32, "memory_budget"),
    (16, 10, 16, "cuda_oom"), (64, 40, 16, "memory_budget"),
])
def test_runtime_failure_tries_32_before_shrinking_below_it(campaign, monkeypatch, maximum, bytes_per_request, expected, reason):
    args, _ = campaign
    args.devices = "0"
    plan, schedule, output = offline.prepare_campaign(args, device_count=8)
    mock_sizing(monkeypatch, 48)
    receipt, predictor, _ = execute_worker(args, monkeypatch, maximum=maximum, bytes_per_request=bytes_per_request)
    assert predictor.calls[:2] == [48, 32]
    if expected == 16:
        assert predictor.calls[2] == 16
    assert receipt["initial_batch_size"] == 48 and receipt["final_batch_size"] == expected
    assert receipt["runtime_batch_retries"][0]["retry_batch_size"] == 32
    assert all(retry["reason"] == reason for retry in receipt["runtime_batch_retries"])
    assert receipt["batch_histogram"] == {expected: 160 // expected}
    assert offline.merge_workers(plan, schedule, output, 1.)["requests_written"] == 160


def test_tail_processes_remaining_windows_as_one_short_batch(campaign, monkeypatch):
    args, _ = campaign
    args.devices = "0"
    offline.prepare_campaign(args, device_count=8)
    mock_sizing(monkeypatch, 9)
    receipt, _, _ = execute_worker(args, monkeypatch, maximum=9)
    assert receipt["batch_histogram"] == {9: 17, 7: 1}


def test_runtime_oom_halves_to_a_new_physical_shape_without_extra_gate(campaign, monkeypatch):
    args, _ = campaign
    args.devices = "0"
    offline.prepare_campaign(args, device_count=8)
    mock_sizing(monkeypatch, 6)
    receipt, predictor, _ = execute_worker(args, monkeypatch, maximum=4)
    assert predictor.calls[:2] == [6, 3]
    assert receipt["runtime_batch_retries"][0]["retry_batch_size"] == 3
    assert receipt["batch_histogram"] == {3: 53, 1: 1}


def test_single_job_probe_has_no_division_by_zero(campaign, monkeypatch):
    args, _ = campaign
    plan, _, _ = offline.prepare_campaign(args, device_count=8)
    one_job = offline.jobs_for(plan)[0]
    monkeypatch.setattr(offline, "jobs_for", lambda *args: [one_job])
    receipt, predictor, _ = execute_worker(args, monkeypatch)
    assert receipt["initial_batch_size"] == receipt["final_batch_size"] == 1
    assert receipt["requests_written"] == 1 and predictor.calls == [1, 1, 1, 1]


def test_runtime_memory_budget_shrinks_before_writing_even_without_oom(campaign, monkeypatch):
    args, _ = campaign
    args.devices = "0"
    plan, schedule, output = offline.prepare_campaign(args, device_count=8)
    # Simulate later inputs being more expensive than the successful probes.
    mock_sizing(monkeypatch, 8)
    receipt, predictor, _ = execute_worker(args, monkeypatch, bytes_per_request=200)
    assert predictor.calls[:2] == [8, 4]
    assert receipt["final_batch_size"] == 4 and receipt["batch_histogram"] == {4: 40}
    assert receipt["runtime_oom_retries"] == []
    assert receipt["runtime_batch_retries"] == [{"offset": 0, "failed_batch_size": 8,
        "retry_batch_size": 4, "reason": "memory_budget", "peak_memory_bytes": 1700, "memory_budget_bytes": 1000}]
    assert receipt["memory_budget_exceeded_batches"] == []
    assert offline.merge_workers(plan, schedule, output, 1.)["requests_written"] == 160
    rows = [json.loads(line) for line in (output / "workers/gpu_0/predictions.jsonl").read_text().splitlines()]
    assert all(row["batch_size"] == 4 for row in rows)


def test_singleton_runtime_memory_budget_failure_is_explicit(campaign, monkeypatch):
    args, _ = campaign
    args.devices = "0"
    offline.prepare_campaign(args, device_count=8)
    mock_sizing(monkeypatch, 1)
    with pytest.raises(RuntimeError, match="batch size 1 exceeded the GPU memory budget"):
        execute_worker(args, monkeypatch, bytes_per_request=1000)
    assert not (args.output / "workers/gpu_0/receipt.json").exists()
    assert (args.output / "workers/gpu_0/predictions.jsonl").read_text() == ""


def test_explicit_batch_records_budget_excess_without_changing_user_choice(campaign, monkeypatch):
    args, _ = campaign
    args.devices, args.batch_size = "0", 8
    offline.prepare_campaign(args, device_count=8)
    receipt, predictor, _ = execute_worker(args, monkeypatch, bytes_per_request=200)
    assert set(predictor.calls) == {8} and receipt["final_batch_size"] == 8
    assert receipt["runtime_batch_retries"] == []
    assert receipt["sizing"]["mode"] == "explicit" and receipt["sizing"]["probes"] == []
    assert len(receipt["memory_budget_exceeded_batches"]) == 20
    assert all(batch["accepted_explicit_batch_size"] is True and batch["peak_memory_bytes"] == 1700
               for batch in receipt["memory_budget_exceeded_batches"])
    assert receipt["final_source_revalidation"] == {"trajectories": 8, "request_hashes": 8,
                                                     "metadata_hashes": 8, "rgb_stats": 8}


@pytest.mark.parametrize("source", ["request", "metadata", "RGB"])
def test_sources_changed_during_worker_prevent_completion(campaign, monkeypatch, source):
    args, _ = campaign
    args.devices, args.batch_size = "0", 8
    offline.prepare_campaign(args, device_count=8)
    original = offline.WindowCache.revalidate_sources

    def change_before_final_check(cache):
        key = next(iter(cache.sources))
        row = cache.rows[key]
        path = Path(cache.sources[key][1]["top"].filename) if source == "RGB" else Path(row[f"{source}_path"])
        with path.open("ab") as stream:
            stream.write(b"\n")
        return original(cache)

    monkeypatch.setattr(offline.WindowCache, "revalidate_sources", change_before_final_check)
    with pytest.raises(ValueError, match=f"{source} source changed while"):
        execute_worker(args, monkeypatch)
    assert not (args.output / "workers/gpu_0/receipt.json").exists()
    assert not (args.output / "execution.json").exists()


def test_merge_failure_never_publishes_an_incomplete_seed_set(campaign, monkeypatch):
    args, _ = campaign
    args.devices, args.batch_size = "0", 8
    plan, schedule, output = offline.prepare_campaign(args, device_count=8)
    execute_worker(args, monkeypatch)
    original_write = offline.write_json

    def interrupted_write(path, value):
        if path.name == "completion.json" and path.parent.name == "seed_29":
            raise OSError("simulated write interruption")
        return original_write(path, value)

    monkeypatch.setattr(offline, "write_json", interrupted_write)
    with pytest.raises(OSError, match="write interruption"):
        offline.merge_workers(plan, schedule, output, 1.)
    assert not (output / "runs").exists()
    assert not (output / "execution.json").exists()


@pytest.mark.parametrize("batch_size,failure,error_type", [
    (8, None, torch.OutOfMemoryError),
    ("auto", ValueError("bad model output"), ValueError),
])
def test_explicit_oom_and_non_oom_errors_fail_without_completion(campaign, monkeypatch, batch_size, failure, error_type):
    args, _ = campaign
    args.devices, args.batch_size = "0", batch_size
    offline.prepare_campaign(args, device_count=8)
    with pytest.raises(error_type):
        execute_worker(args, monkeypatch, maximum=4, failure=failure)
    assert not (args.output / "workers/gpu_0/receipt.json").exists()
    assert not (args.output / "execution.json").exists()
    assert not (args.output / "runs").exists()


@pytest.mark.parametrize("damage,match", [("duplicate", "duplicate"), ("missing", "omitted"),
                                           ("wrong_device", "assigned"), ("changed_digest", "identity differs")])
def test_merge_rejects_corrupt_or_incomplete_workers(campaign, monkeypatch, damage, match):
    args, _ = campaign
    args.devices, args.batch_size = "0", 8
    plan, schedule, output = offline.prepare_campaign(args, device_count=8)
    execute_worker(args, monkeypatch)
    journal, receipt_path = output / "workers/gpu_0/predictions.jsonl", output / "workers/gpu_0/receipt.json"
    records = [json.loads(line) for line in journal.read_text().splitlines()]
    if damage == "duplicate":
        records.append(records[0])
    elif damage == "missing":
        records.pop()
    else:
        records[0]["worker_device"] = 7
    journal.write_text("".join(json.dumps(row) + "\n" for row in records))
    if damage != "changed_digest":
        receipt = json.loads(receipt_path.read_text())
        receipt.update(journal_sha256=offline.file_digest(journal), requests_written=len(records))
        receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match=match):
        offline.merge_workers(plan, schedule, output, 1.)
    assert not (output / "runs").exists()
    assert not (output / "execution.json").exists()


def test_parent_isolates_workers_without_changing_parent_visibility(campaign, monkeypatch):
    args, _ = campaign
    launched = []
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7,6,5,4,3,2,1,0")
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 8)

    class Process:
        def __init__(self, command, **kwargs):
            launched.append((command, kwargs))

        def poll(self):
            return 0

    monkeypatch.setattr(offline.subprocess, "Popen", Process)
    monkeypatch.setattr(offline, "merge_workers", lambda p, s, o, elapsed:
                        {"devices": s["devices"], "requests_written": len(offline.jobs_for(p)), "campaign_wall_seconds": elapsed,
                         "final_per_gpu_batch_sizes": {str(device): 8 for device in s["devices"]},
                         "final_aggregate_batch_capacity": 8 * len(s["devices"])})
    assert offline.run(args)["requests_written"] == 160
    assert len(launched) == 8
    for device, (command, options) in enumerate(launched):
        assert command[command.index("--worker-device") + 1] == str(device)
        assert command[command.index("--batch-size") + 1] == "auto"
        assert options["env"]["CUDA_VISIBLE_DEVICES"] == fake_uuid(7 - device)
        assert options["env"]["TORCHINDUCTOR_COMPILE_THREADS"] == "2"
        assert options["start_new_session"] is True
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "7,6,5,4,3,2,1,0"
