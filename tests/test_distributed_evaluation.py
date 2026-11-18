# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise native CPU evaluation, unequal shards, exact routes and merged metrics."""
from copy import deepcopy
from dataclasses import replace
from datetime import timedelta
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from grana.data.cache import Trajectory
from grana.training.evaluation import EvaluationPlan, EvaluationWindow, evaluate
from grana.modeling.inputs import preprocessing_spec
from grana.training.campaign_report import verify_evaluation
from grana.training.distributed_evaluation import split_plan, merge_results

FRAME = (.1, .2, .3, .5, 1., 2., 3., 4.)
ROLE = {"A": 1., "B": 1., "C": 10., "D": 1.}
LOSS = {"stage": {"cdf_weight": .7}, "progress": {"cdf_weight": 1.3, "ceiling_weight": 0.}}


class BatchModel(nn.Module):
    def forward(self, inputs, roles, stage_counts):
        examples = inputs["examples"]
        if roles[0] == "C":
            logits = torch.full((len(examples), 8, 3), -2.)
            for row, example in enumerate(examples):
                for slot, frame in enumerate(example.frame_indices):
                    logits[row, slot, (frame // 7) % 3] = 2.
            return {"stage_logits": logits}
        logits = torch.empty(len(examples), 8, 10)
        for row, example in enumerate(examples):
            for slot, frame in enumerate(example.frame_indices):
                logits[row, slot] = torch.arange(10) * ((frame % 13) - 6) / 3
        return {"progress_logits": logits}


def fake_vision(calls):
    return {"attention_modules": 2, "vision_encoders": 1, "evaluation_enabled": True, "training_enabled": False,
            "training": {"calls": 0, "hits": 0, "images": 0, "fallbacks": {}},
            "evaluation": {"calls": calls * 2, "hits": calls * 2, "images": calls * 64, "fallbacks": {}},
            "profile": {"enabled": True, "vision_calls": calls, "host_seconds": calls * .1,
                        "cuda_seconds": calls * .2, "pending_cuda_spans": 0}}


def make_plan(root):
    rows, windows, tasks = [], [], []
    for task, frames in (("first", 41), ("second", 28)):
        pixels = np.zeros((frames, 2, 3, 3), dtype=np.uint8)
        path = root / (task + ".npy")
        np.save(path, pixels)
        row = Trajectory(root / task, task, task, frames, (path,), pixels.shape, ("top",), ("B", "C"),
                         tuple(i / (frames - 1) for i in range(frames)),
                         tuple(min(2, i * 3 // frames) for i in range(frames)),
                         tuple((i % 7) / 7 for i in range(frames)), ("one", "two", "three"), "Complete task.")
        rows.append(row)
        windows.extend(EvaluationWindow(row, poi, (1, 1), ("B", "C", "D"), task) for poi in range(frames))
        tasks.append({"task": task, "selected_trajectories": 1})
    dataset = SimpleNamespace(records=tuple(rows))
    plan = EvaluationPlan(dataset, tuple(windows), tuple(tasks),
                          {"batch_size": 32, "preprocessing": preprocessing_spec(76800), "stride_pairs": [[1, 1]],
                           "seed": 1729, "purpose": "deploy_dense"})
    return plan


@pytest.fixture(scope="module")
def measured(tmp_path_factory):
    torch.set_num_threads(2)
    root = tmp_path_factory.mktemp("native-shard-evaluation")
    plan = make_plan(root)
    def run(selected):
        result = evaluate(BatchModel(), None, selected, device="cpu", max_pixels=76800,
                          frame_weights=FRAME, role_weights=ROLE, loss_config=LOSS,
                          collate_fn=lambda processor, examples, pixels, device: {"examples": examples})
        result["vision_sdpa"] = fake_vision(result["forward_calls"])
        return result
    full = run(plan)
    shards = {shard.rank: run(shard.plan) for shard in split_plan(plan)}
    return plan, full, shards, root


def merge(plan, shards):
    return merge_results(plan, shards, elapsed_seconds=100., frame_weights=FRAME, role_weights=ROLE, loss_config=LOSS)


def assert_equivalent(left, right):
    if isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_equivalent(left[key], right[key])
    elif isinstance(left, list):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            assert_equivalent(a, b)
    elif isinstance(left, (int, float)) and not isinstance(left, bool):
        assert math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-12)
    else:
        assert left == right


def test_native_full_evaluation_and_merged_unequal_shards_have_all_same_quality(measured):
    plan, full, shards, root = measured
    assert [len(shard.plan.windows) for shard in split_plan(plan)] == [64, 5]
    merged = merge(plan, shards)
    for key in ("metrics", "roles", "per_task", "diagnostics", "loss_config", "image_preprocessing"):
        assert_equivalent(merged[key], full[key])
    for actual, expected in zip(merged["records"], full["records"], strict=True):
        assert_equivalent({key: value for key, value in actual.items() if key != "elapsed_seconds"},
                          {key: value for key, value in expected.items() if key != "elapsed_seconds"})
    merged["optimizer_steps"] = 250
    path = root / "merged-evaluation.json"
    path.write_text(json.dumps(merged))
    verify_evaluation(path, 250)
    assert merged["window_batch_sizes"] == [32, 32, 5]
    assert merged["forward_batch_sizes"] == {"5": 3, "32": 6}
    assert merged["forward_calls"] == 9
    assert merged["elapsed_seconds"] == 100.
    assert merged["role_rows_per_second"] == 207 / 100
    assert merged["vision_sdpa"]["evaluation"]["hits"] == 18
    assert merged["vision_sdpa"]["profile"]["vision_calls"] == 9
    assert merged["cuda_memory"] is None and merged["cuda_memory_by_worker"] == {"0": None, "1": None}


def test_never_averages_shard_summaries_or_rank_means(measured):
    plan, full, shards, _ = measured
    changed = deepcopy(shards)
    for worker in changed.values():
        worker["metrics"] = {"wrong": 123456}
        worker["roles"] = worker["per_task"] = worker["diagnostics"] = {"wrong": 999}
    merged = merge(plan, changed)
    assert_equivalent(merged["metrics"], full["metrics"])
    key = "role_B_poi_rmse"
    assert not math.isclose(merged["metrics"][key], sum(worker["metrics"][key] for worker in shards.values()) / 2)


def test_original_6366_plan_keeps_all_199_native_batches_and_tail(measured):
    small, _, _, _ = measured
    row = replace(small.windows[0].trajectory, num_frames=6366)
    plan = replace(small, windows=tuple(EvaluationWindow(row, i, (1, 1), ("B", "C", "D")) for i in range(6366)))
    left, right = split_plan(plan)
    assert [(s.window_start, s.window_stop, s.chunk_start, s.chunk_stop) for s in (left, right)] == [
        (0, 3200, 0, 100), (3200, 6366, 100, 199)]
    assert [s.manifest()["model_calls"] for s in (left, right)] == [9600, 9498]
    assert left.plan.windows + right.plan.windows == plan.windows
    assert [s.plan.settings for s in (left, right)] == [plan.settings, plan.settings]


@pytest.mark.parametrize("failure", ["missing", "duplicate", "order", "route", "input", "loss", "loss_config",
                                     "plan_hash", "tail", "histogram", "forward_calls", "rank", "timing", "vision", "preprocessing"])
def test_rejects_incomplete_misbound_or_invalid_shard_results(measured, failure):
    plan, _, original, _ = measured
    shards = deepcopy(original)
    worker = shards[1]
    if failure == "missing": worker["records"].pop()
    elif failure == "duplicate": worker["records"][-1] = deepcopy(worker["records"][0])
    elif failure == "order": worker["records"][0], worker["records"][1] = worker["records"][1], worker["records"][0]
    elif failure == "route": worker["records"][2]["focus_stage"] = (worker["records"][2]["focus_stage"] + 1) % 3
    elif failure == "input": worker["records"][0]["frame_indices"][0] += 1
    elif failure == "loss": worker["records"][0]["slot_losses"][0] += .1
    elif failure == "loss_config": worker["loss_config"]["progress"]["ceiling_weight"] = 1
    elif failure == "plan_hash": worker["plan_sha256"] = "foreign"
    elif failure == "preprocessing": worker["image_preprocessing"]["resize_backend"] = "foreign"
    elif failure == "tail": worker["window_batch_sizes"] = [3, 2]
    elif failure == "histogram": worker["forward_batch_sizes"] = {"5": 4}
    elif failure == "forward_calls": worker["forward_calls"] += 1
    elif failure == "rank": shards[2] = shards.pop(1)
    elif failure == "timing": worker["elapsed_seconds"] = 101.
    elif failure == "vision": worker["vision_sdpa"]["profile"]["pending_cuda_spans"] = 1
    with pytest.raises(ValueError):
        merge(plan, shards)


def test_fewer_batches_than_ranks_preserves_batch_membership(measured):
    plan, _, _, _ = measured
    short = replace(plan, windows=plan.windows[:32])
    shards = split_plan(short, workers=4)
    assert [len(s.plan.windows) for s in shards] == [32, 0, 0, 0]
    assert all(s.window_start == s.window_stop == 32 for s in shards[1:])


def test_empty_ranks_do_not_change_merged_metrics(measured):
    plan, full, _, _ = measured
    results = {}
    for shard in split_plan(plan, workers=4):
        results[shard.rank] = evaluate(
            BatchModel(), None, shard.plan, device="cpu", max_pixels=76800,
            frame_weights=FRAME, role_weights=ROLE, loss_config=LOSS,
            collate_fn=lambda processor, examples, pixels, device: {"examples": examples}
        ) if shard.plan.windows else None
    actual = merge(plan, results)
    assert_equivalent(actual["metrics"], full["metrics"])
    assert actual["execution_world_size"] == 4
    assert actual["parallel_execution"]["shards"][-1]["forward_calls"] == 0


@pytest.mark.parametrize("batch_size", [1, 16, 32])
@pytest.mark.parametrize("roles", ["A", "mixed"])
def test_pretraining_and_mixed_role_plans_preserve_denominators(measured, batch_size, roles):
    plan, _, _, _ = measured
    rows = {row.trajectory_id: replace(row, roles=("A",)) if roles == "A" or row.task == "first" else row
            for row in plan.dataset.records}
    windows = tuple(replace(window, trajectory=rows[window.trajectory.trajectory_id], roles=("A",))
                    if rows[window.trajectory.trajectory_id].roles == ("A",) else window for window in plan.windows)
    plan = replace(plan, windows=windows, dataset=SimpleNamespace(records=tuple(rows.values())),
                   settings={k: v for k, v in plan.settings.items() if k != "purpose"})
    plan = replace(plan, settings={**plan.settings, "batch_size": batch_size})

    def run(selected):
        return evaluate(BatchModel(), None, selected, device="cpu", max_pixels=76800,
                        frame_weights=FRAME, role_weights=ROLE, loss_config=LOSS,
                        collate_fn=lambda processor, examples, pixels, device: {"examples": examples})

    full = run(plan)
    actual = merge(plan, {s.rank: run(s.plan) for s in split_plan(plan)})
    for key in ("metrics", "roles", "per_task", "diagnostics"):
        assert_equivalent(actual[key], full[key])


def _distributed_worker(rank, world_size, plan, root, scenario):
    import torch.distributed as dist
    from grana.training.distributed_evaluation import evaluate_distributed

    torch.set_num_threads(1)
    root = Path(root)
    dist.init_process_group("gloo", init_method=f"file://{root / 'rendezvous'}",
                            rank=rank, world_size=world_size, timeout=timedelta(seconds=45))
    calls = []
    try:
        if scenario == "plan_mismatch" and rank == 1:
            plan = replace(plan, settings={**plan.settings, "seed": 1234})
        elif scenario == "manifest_failure" and rank == 1:
            plan = replace(plan, settings={**plan.settings, "invalid_metadata": object()})
        elif scenario == "invalid_batch" and rank == 1:
            plan = replace(plan, settings={**plan.settings, "batch_size": 0})
        output_dir = root / "shards"
        if scenario == "invalid_output" and rank == 1:
            output_dir = None
        elif scenario == "persistence_failure" and rank == 1:
            output_dir = root / "not-a-directory"
            output_dir.write_text("injected rank-local result persistence failure")

        def run(shard):
            calls.append(len(shard.windows))
            if scenario == "worker_failure" and rank == 1:
                raise ValueError("injected worker failure")
            value = evaluate(BatchModel(), None, shard, device="cpu", max_pixels=76800,
                             frame_weights=FRAME, role_weights=ROLE, loss_config=LOSS,
                             collate_fn=lambda processor, examples, pixels, device: {"examples": examples})
            if scenario == "bad_record" and rank == 1:
                value["records"][0]["slot_losses"][0] += 1
            return value

        error = None
        try:
            result = evaluate_distributed(plan, run, output_dir=output_dir, frame_weights=FRAME,
                                          role_weights=ROLE, loss_config=LOSS)
            if rank == 0:
                (root / "merged.json").write_text(json.dumps(result))
            else:
                assert result is None
        except RuntimeError as caught:
            error = str(caught)
        (root / f"proof-{rank}.json").write_text(json.dumps({"calls": calls, "error": error}))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("scenario,workers", [("success", 4), ("empty_rank", 2),
                                             ("worker_failure", 2), ("bad_record", 2), ("plan_mismatch", 2),
                                             ("manifest_failure", 2), ("invalid_batch", 2),
                                             ("invalid_output", 2), ("persistence_failure", 2)])
def test_real_process_sharding_and_failure_propagation(tmp_path, scenario, workers):
    import torch.multiprocessing as mp
    plan = make_plan(tmp_path)
    if scenario == "success":
        plan = replace(plan, settings={**plan.settings, "batch_size": 16})
    elif scenario == "empty_rank":
        plan = replace(plan, windows=plan.windows[:32])
    mp.spawn(_distributed_worker, args=(workers, plan, str(tmp_path), scenario), nprocs=workers, join=True)
    proofs = [json.loads((tmp_path / f"proof-{rank}.json").read_text()) for rank in range(workers)]
    if scenario in ("success", "empty_rank"):
        assert all(p["error"] is None for p in proofs)
        expected = [len(s.plan.windows) for s in split_plan(plan, workers)]
        assert [sum(p["calls"]) for p in proofs] == expected
        full = evaluate(BatchModel(), None, plan, device="cpu", max_pixels=76800,
                        frame_weights=FRAME, role_weights=ROLE, loss_config=LOSS,
                        collate_fn=lambda processor, examples, pixels, device: {"examples": examples})
        merged = json.loads((tmp_path / "merged.json").read_text())
        assert_equivalent(merged["metrics"], full["metrics"])
        assert merged["execution_world_size"] == workers
        assert merged["parallel_execution"]["active_workers"] == sum(bool(n) for n in expected)
        assert merged["elapsed_seconds"] >= merged["parallel_execution"]["worker_elapsed_seconds_max"]
    else:
        expected = {"worker_failure": "injected worker failure", "bad_record": "raw slots",
                    "plan_mismatch": "different evaluation plans",
                    "manifest_failure": "preflight failed", "invalid_batch": "preflight failed",
                    "invalid_output": "preflight failed", "persistence_failure": "FileExistsError"}[scenario]
        assert all(expected in p["error"] for p in proofs)
        if scenario in ("manifest_failure", "invalid_batch", "invalid_output", "plan_mismatch"):
            assert all(not p["calls"] for p in proofs)
        assert not (tmp_path / "merged.json").exists()


class TrainEvalModel(BatchModel):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(.7))

    def forward(self, inputs, roles=None, stage_counts=None):
        if isinstance(inputs, torch.Tensor):
            return inputs * self.weight
        assert not self.training
        return super().forward(inputs, roles, stage_counts)


def _native_stages_worker(rank, world_size, plan, root, corrupt):
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    from grana.training.distributed_evaluation import evaluate_distributed

    torch.set_num_threads(1)
    root = Path(root)
    dist.init_process_group("gloo", init_method=f"file://{root / 'rendezvous'}",
                            rank=rank, world_size=world_size, timeout=timedelta(seconds=45))
    try:
        model = DistributedDataParallel(TrainEvalModel())
        optimizer = torch.optim.SGD(model.parameters(), lr=.05)
        weights = [float(model.module.weight.detach())]

        def train():
            optimizer.zero_grad()
            model(torch.tensor(float(rank + 1))).square().backward()
            assert model.module.weight.grad.isfinite().all()
            optimizer.step()
            weights.append(float(model.module.weight.detach()))

        train()
        observed = []

        def collate(_processor, examples, _max_pixels, _device):
            for example in examples:
                if example.role == "C":
                    assert example.targets == example.stage_ids and example.num_stages == 3
                observed.append([example.task, example.role, example.num_stages])
            return {"examples": examples}

        def run(shard):
            value = evaluate(model.module, None, shard, device="cpu", max_pixels=76800,
                             frame_weights=FRAME, role_weights=ROLE, loss_config=LOSS, collate_fn=collate)
            if corrupt and rank == 1:
                value["records"][0]["targets"][0] = -1.0
            return value

        error = None
        try:
            result = evaluate_distributed(plan, run, output_dir=root / "shards", frame_weights=FRAME,
                                          role_weights=ROLE, loss_config=LOSS)
            assert model.module.training
            train()
            if rank == 0:
                (root / "merged.json").write_text(json.dumps(result))
        except RuntimeError as caught:
            error = str(caught)
        (root / f"proof-{rank}.json").write_text(json.dumps({"error": error, "weights": weights,
                                                              "observed": observed}))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("workers,corrupt", [(2, False), (4, False), (2, True)])
def test_real_process_native_stages_and_train_eval_train(tmp_path, workers, corrupt):
    import torch.multiprocessing as mp

    plan = make_plan(tmp_path)
    plan = replace(plan, settings={**{k: v for k, v in plan.settings.items() if k != "purpose"},
                                   "batch_size": 16})
    mp.spawn(_native_stages_worker, args=(workers, plan, str(tmp_path), corrupt), nprocs=workers, join=True)
    proofs = [json.loads((tmp_path / f"proof-{rank}.json").read_text()) for rank in range(workers)]
    if corrupt:
        assert all("targets differ" in proof["error"] for proof in proofs)
        assert not (tmp_path / "merged.json").exists()
        return
    assert all(proof["error"] is None and len(proof["weights"]) == 3 for proof in proofs)
    assert all(proof["weights"] == proofs[0]["weights"] for proof in proofs)
    assert proofs[0]["weights"][0] != proofs[0]["weights"][1] != proofs[0]["weights"][2]
    assert all(proof["observed"] for proof in proofs)
    result = json.loads((tmp_path / "merged.json").read_text())
    assert result["execution_world_size"] == result["parallel_execution"]["active_workers"] == workers
    assert result["per_task"]["first"]["D"]["windows"] == 41
    assert result["per_task"]["second"]["D"]["windows"] == 28
    full = evaluate(BatchModel(), None, plan, device="cpu", max_pixels=76800,
                    frame_weights=FRAME, role_weights=ROLE, loss_config=LOSS,
                    collate_fn=lambda processor, examples, pixels, device: {"examples": examples})
    for key in ("metrics", "roles", "per_task", "diagnostics"):
        assert_equivalent(result[key], full[key])
