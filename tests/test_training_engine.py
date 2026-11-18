# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the actual worker loop with CPU tensors and fake Ray/CUDA I/O."""
from contextlib import nullcontext
from dataclasses import replace
from datetime import timedelta
import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import pytest
import torch
from torch import nn
import yaml

from grana.config import load_config
from grana.data.sampling import RoleExample, RoleSampler, _image_digest
from grana.training import engine
from test_training_initialization import TinyModel


class RuntimeModel(TinyModel):
    def forward(self, inputs, roles, stage_counts):
        hidden = self.temporal_head(self.backbone.adapter(torch.ones(len(roles), 8, 4)))
        progress_indices = [i for i, role in enumerate(roles) if role != "C"]
        stage_indices = [i for i, role in enumerate(roles) if role == "C"]
        return {"progress_logits": self.progress_head(hidden[progress_indices]) if progress_indices else None,
                "stage_logits": nn.functional.linear(hidden[stage_indices], self.backbone.output.weight[:3]) if stage_indices else None,
                "progress_indices": progress_indices, "stage_indices": stage_indices}


class Dataset:
    def __init__(self, *args, **kwargs):
        self.by_role = {"A": {}, "B": {"task": [None]}, "C": {"task": [None]}}

    def summary(self):
        return {"trajectories": 1}


class Sampler:
    make_d = RoleSampler.make_d

    def __init__(self, *args, **kwargs):
        self.images = (Image.new("RGB", (1, 1)),) * 8
        self.draw = 0

    def validate_role_budget(self, roles):
        pass


    def sample(self, role):
        role = "C" if role in ("C_paired", "C_independent") else role
        self.draw += 1
        return RoleExample(
            role=role, images=self.images, frame_indices=(0,) * 8,
            targets=(1,) * 8 if role == "C" else (.25,) * 8,
            num_stages=3, task="task", trajectory_id=f"draw-{self.draw}", prompt="prompt",
            view_names=("top",), stage_prompts=("first", "second", "third"),
            stage_ids=(1,) * 8, within_stage_progress=(.25,) * 8, stride_pair=(10, 1),
            focus_stage=1 if role == "B" else None,
            observation_digest=_image_digest(self.images) if role == "C" else None,
        )


class Logger:
    def __init__(self):
        self.evaluations = []
        self.dense_bundles = []
        self.history_calls = []

    def start(self, **kwargs):
        pass

    def log_checkpoint(self, *args, **kwargs):
        pass


    def log_step(self, *args, **kwargs):
        self.history_calls.append(("training", kwargs["step"]))

    def log_evaluation(self, metrics, **kwargs):
        self.evaluations.append((metrics, kwargs))
        self.history_calls.append(("evaluation", kwargs["step"]))

    def log_evaluation_renders(self, bundle, **kwargs):
        pass

    def log_deploy_dense(self, bundle, **kwargs):
        self.dense_bundles.append(bundle)
        self.history_calls.append(("media", kwargs["step"]))

    def commit_step(self, *, step):
        self.history_calls.append(("commit", step))

    def finish(self, **kwargs):
        pass


def test_detached_stage_choices_follow_compact_rows_and_designated_duplicate_occurrence():
    examples = [SimpleNamespace(role="B"), SimpleNamespace(role="C", poi_slot=2, frame_indices=(0,) * 8),
                SimpleNamespace(role="A"), SimpleNamespace(role="C", poi_slot=0, frame_indices=(0,) * 8)]
    logits = torch.zeros(2, 8, 3)
    logits[:, 5, 0] = 10  # Literal slot 5 would select the wrong stage for both.
    logits[0, 0, 2] = 10
    logits[1, 2, 1] = 10
    logits.requires_grad_()
    output = {"stage_indices": [3, 1], "stage_logits": logits}
    assert engine.detached_stage_choices(output, examples) == [2, 1]
    assert logits.grad is None and logits.requires_grad
    assert engine.detached_stage_choices({"stage_indices": [], "stage_logits": None}, [examples[0]]) is None


@pytest.mark.parametrize("slot", [-1, 8, True, 2.0])
def test_stage_choices_reject_invalid_query_slot(slot):
    with pytest.raises(ValueError, match="poi_slot"):
        engine.detached_stage_choices({"stage_indices": [0], "stage_logits": torch.zeros(1, 8, 3)},
                                      [SimpleNamespace(role="C", poi_slot=slot)])


@pytest.mark.parametrize("indices", [[], [0, 0], [1], [True]])
def test_stage_choices_require_exact_c_row_coverage(indices):
    with pytest.raises(ValueError, match="every C example"):
        engine.detached_stage_choices({"stage_indices": indices, "stage_logits": torch.zeros(len(indices), 8, 3)},
                                      [SimpleNamespace(role="C", poi_slot=2), SimpleNamespace(role="B")])


@pytest.mark.parametrize("field,value", [("poi_slot", 2), ("frame_weight_order", tuple(reversed(range(8)))),
                                         ("frame_selection_profile", "irregular_stratified"),
                                         ("query_order_profile", "backward")])
def test_c_d_pair_checks_sampling_metadata_even_for_identical_images(field, value):
    sampler = Sampler()
    source = sampler.sample("C")
    derived = sampler.make_d(source, 1)
    engine._validate_c_d_pair(source, derived)
    mutated = SimpleNamespace(**{**vars(derived), field: value})
    with pytest.raises(RuntimeError, match="sampling metadata"):
        engine._validate_c_d_pair(source, mutated)


@pytest.mark.parametrize("batch,accumulation", [(64, 2), (32, 4)])
@pytest.mark.parametrize("reordered", [False, True])
@pytest.mark.parametrize("gpu_image_resize", [False, True])
@pytest.mark.parametrize("prefetch_samples", [False, True])
def test_worker_supervises_exact_bcd_budget_and_reuses_dense_predictions(
        tmp_path, monkeypatch, batch, accumulation, reordered, gpu_image_resize, prefetch_samples, mixed=False):
    import ray.train
    import ray.train.torch
    from grana.data import cache, sampling
    from grana.modeling import model as modeling
    from grana.training import deploy_dense, evaluation

    context = SimpleNamespace(get_world_rank=lambda: 0, get_world_size=lambda: 1)
    monkeypatch.setattr(engine, "require_ray_worker", lambda: None)
    monkeypatch.setattr(ray.train, "get_context", lambda: context)
    monkeypatch.setattr(ray.train, "report", lambda *args, **kwargs: None)
    monkeypatch.setattr(ray.train.torch, "get_device", lambda: torch.device("cuda:0"))
    monkeypatch.setattr(ray.train.torch, "prepare_model", lambda model, **kwargs: model)
    monkeypatch.setattr(cache, "CachedTrajectories", Dataset)
    class ReorderedSampler(Sampler):
        def sample(self, role):
            example = super().sample(role)
            # All eight physical indices are zero. Only the designated
            # occurrence identifies the query after changing presentation order.
            reverse = self.draw % 2 == 0
            return replace(example, poi_slot=2 if reverse else 5,
                           frame_weight_order=tuple(reversed(range(8))) if reverse else (0, 1, 2, 7, 6, 5, 4, 3),
                           frame_selection_profile="irregular_stratified",
                           query_order_profile="backward" if reverse else "tail_rewind")

        def make_d(self, source, focus):
            assert focus == source.poi_slot // 3
            return super().make_d(source, focus)

    class SlotModel(RuntimeModel):
        def forward(self, inputs, roles, stage_counts):
            result = super().forward(inputs, roles, stage_counts)
            if result["stage_logits"] is not None:
                slot_bias = nn.functional.one_hot(torch.arange(8) // 3, 3).float() * 20
                result["stage_logits"] = (result["stage_logits"] + slot_bias).flip(0)
                result["stage_indices"] = list(reversed(result["stage_indices"]))
            return result

    monkeypatch.setattr(sampling, "RoleSampler", ReorderedSampler if reordered else Sampler)
    monkeypatch.setattr(modeling, "build_model", lambda config: ((SlotModel() if reordered else RuntimeModel()), None))
    configured_processor = object()
    preprocessing_calls, collated_processors = [], []

    def configure(processor, device, *, resize_on_device):
        preprocessing_calls.append((processor, str(device), resize_on_device))
        return configured_processor

    def capture_collate(processor, *args, **kwargs):
        collated_processors.append(processor)
        return {}

    monkeypatch.setattr(engine, "configure_image_preprocessing", configure)
    monkeypatch.setattr(engine, "collate_examples", capture_collate)
    monkeypatch.setattr(torch, "autocast", lambda *args, **kwargs: nullcontext())
    for name in ("set_device", "synchronize", "reset_peak_memory_stats", "manual_seed_all"):
        monkeypatch.setattr(torch.cuda, name, lambda *args, **kwargs: None)
    for name in ("memory_allocated", "memory_reserved", "max_memory_allocated", "max_memory_reserved"):
        monkeypatch.setattr(torch.cuda, name, lambda *args, **kwargs: 1024)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *args: "cpu-test")
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *args: (2**30, 2**31))
    for name in ("tensor", "zeros"):
        original = getattr(torch, name)

        def on_cpu(*args, _original=original, **kwargs):
            if str(kwargs.get("device", "")).startswith("cuda"):
                kwargs["device"] = "cpu"
            return _original(*args, **kwargs)

        monkeypatch.setattr(torch, name, on_cpu)

    selection_plan = SimpleNamespace(manifest=lambda: {"selection": "all"})
    dense_plan = SimpleNamespace(manifest=lambda: {"selection": "dense"})
    monkeypatch.setattr(evaluation, "build_evaluation_plan", lambda *args, **kwargs: selection_plan)
    monkeypatch.setattr(deploy_dense, "require_encoder", lambda: None)
    monkeypatch.setattr(deploy_dense, "build_dense_plan", lambda *args, **kwargs: dense_plan)
    evaluations = []
    rendered_results = []

    def evaluate(*args, **kwargs):
        assert args[1] is configured_processor
        assert args[2] is dense_plan
        result = {"plan_sha256": "dense", "elapsed_seconds": .01, "model_calls": 3,
                  "task_count": 1, "metrics": {"C_poi_accuracy": .5},
                  "diagnostics": {"support": 1}}
        evaluations.append(result)
        return result

    def render(result, *args, **kwargs):
        rendered_results.append(result)
        return {"rendering_seconds": 0., "items": []}

    monkeypatch.setattr(evaluation, "evaluate", evaluate)
    monkeypatch.setattr(deploy_dense, "render_dense_evaluation", render)
    path = tmp_path / "recipe.yaml"
    factor = batch // 16
    path.write_text(yaml.safe_dump({
        "roles": {"A": 0, "B": 8 * factor, "C": (14 if mixed else 12) * factor,
                  "D": (10 if mixed else 12) * factor},
        "data": {"caches": ["/unused/train"], "gpu_image_resize": gpu_image_resize},
        "training": {"stage": "posttrain", "max_steps": 2, "warmup_steps": 0,
                     "prefetch_samples": prefetch_samples,
                     "global_batch_size": 128, "gradient_accumulation_steps": accumulation,
                     "save_steps": 2, "output_dir": str(tmp_path / "run")},
        "evaluation": {"enabled": True, "mode": "dense", "caches": ["/unused/eval"],
                       "every_steps": 2, "max_trajectories_per_task": None,
                       "deploy_dense": {"enabled": True, "max_trajectories_per_task": None}},
    }))
    logger = Logger()
    engine._train_worker(load_config(path), logger)
    output = tmp_path / "run"
    assert preprocessing_calls == [(None, "cuda:0", gpu_image_resize)]
    assert collated_processors and all(item is configured_processor for item in collated_processors)
    runtime = json.loads((output / "runtime-000.json").read_text())
    assert runtime["gpu_image_resize"] is gpu_image_resize
    assert runtime["prefetch_samples"] is prefetch_samples
    assert runtime["image_preprocessing"]["version"] == (2 if gpu_image_resize else 1)
    accounting = json.loads((output / "batch_accounting.json").read_text())
    expected = {"A": 0, "B": 32, "C": 56 if mixed else 48, "D": 40 if mixed else 48}
    assert accounting["actual_rows_per_update_global"] == expected
    assert accounting["derived_C_D_pairs_per_update_global"] == expected["D"]
    assert len(accounting["forward_role_rows_per_worker"]) == accumulation
    assert all(sum(phase.values()) == batch for phase in accounting["forward_role_rows_per_worker"])
    metrics = [json.loads(line) for line in (output / "training_metrics.jsonl").read_text().splitlines()]
    assert len(metrics) == 2
    assert {role: metrics[-1][f"role_{role}_rows_cumulative_global"] for role in expected} == {
        role: count * 2 for role, count in expected.items()}
    assert len(evaluations) == len(rendered_results) == 1
    assert evaluations[0] is rendered_results[0]
    assert evaluations[0]["provenance"] == {"mode": "during_training", "optimizer_steps": 2}
    assert json.loads((output / "evaluation-000002.json").read_text()) == json.loads(
        (output / "deploy-dense-evaluation-000002.json").read_text())
    assert logger.evaluations[0][1]["diagnostics"] == {"support": 1}
    # The logger receives the full reusable predictions and the completed
    # timing event, not an early snapshot missing rendering/persistence time.
    assert logger.evaluations[0][1]["evaluation"] is evaluations[0]
    assert "evaluation_and_persistence_seconds" in logger.evaluations[0][0]
    assert logger.evaluations[0][0]["deploy_dense_reused_primary_predictions"] is True
    assert logger.history_calls == [("training", 1), ("commit", 1),
                                    ("training", 2), ("media", 2), ("evaluation", 2), ("commit", 2)]
    result = json.loads((output / "result.json").read_text())
    assert result["checkpoint_reload_equal"] and result["checkpoint_prediction_parity"]
    assert result["runtime_summary"]["actual_role_rows_over_run_global"] == {
        role: count * 2 for role, count in expected.items()}
    coverage = json.loads((output / "sampling_coverage.json").read_text())
    assert coverage["derived_D_rows"] == expected["D"] * 2
    family = "irregular_stratified" if reordered else "regular_ctsp"
    orders = ("backward", "tail_rewind") if reordered else ("forward",)
    for role in "ABC":
        counts = coverage["independent_profile_order_by_role"][role][family]
        assert sum(counts[order] for order in orders) == expected[role] * 2
        assert sum(coverage["independent_poi_slots_by_role"][role].values()) == expected[role] * 2
    if reordered:
        assert all(coverage["independent_poi_slots_by_role"]["C"][str(slot)] > 0 for slot in (2, 5))
        assert all((row["query_order_profile"], row["poi_slot"]) in (("backward", 2), ("tail_rewind", 5))
                   for row in result["verification_examples"])


@pytest.mark.parametrize("prefetch", [False, True])
def test_worker_runs_paired_and_independent_c_in_both_phases(tmp_path, monkeypatch, prefetch):
    test_worker_supervises_exact_bcd_budget_and_reuses_dense_predictions(
        tmp_path, monkeypatch, 32, 4, False, False, prefetch, mixed=True)


def _gloo_worker(rank, recipe, rendezvous, plan, failure_mode=None):
    """A real two-process collective/DDP path; only CUDA and external I/O fake."""
    import ray.train
    import ray.train.torch
    import torch.distributed as distributed
    from grana.data import cache, sampling
    from grana.modeling import model as modeling
    from grana.training import deploy_dense, evaluation, render_plan, rendering

    torch.set_num_threads(1)
    distributed.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2,
                                   timeout=timedelta(seconds=60))
    patch = pytest.MonkeyPatch()
    evaluations = []

    class IndependentSampler(Sampler):
        def sample(self, role):
            example = super().sample(role)
            return replace(example, targets=(rank + 1,) * 8 if example.role == "C" else (.25 + .5 * rank,) * 8,
                           stage_ids=(rank + 1,) * 8, within_stage_progress=(.25 + .5 * rank,) * 8,
                           poi_slot=2 if rank == 0 else 5,
                           frame_weight_order=tuple(reversed(range(8))) if rank == 0 else tuple(range(8)),
                           frame_selection_profile="irregular_stratified" if rank == 0 else "regular_ctsp",
                           query_order_profile="backward" if rank == 0 else "forward")

    native_evaluate = evaluation.evaluate
    full_metrics = []

    def evaluate(*args, **kwargs):
        # Unequal rank-local forwards must bypass DDP and restore training/RNG.
        assert not isinstance(args[0], nn.parallel.DistributedDataParallel)
        assert args[0].training
        kwargs.update(device="cpu", collate_fn=lambda *args: {})
        state = torch.random.get_rng_state().clone()
        if rank == 0:
            full_metrics.append(native_evaluate(args[0], args[1], plan, **kwargs)["metrics"])
        value = native_evaluate(*args, **kwargs)
        assert args[0].training and torch.equal(state, torch.random.get_rng_state())
        evaluations.append(value)
        return value

    try:
        context = SimpleNamespace(get_world_rank=lambda: rank, get_world_size=lambda: 2)
        patch.setattr(engine, "require_ray_worker", lambda: None)
        patch.setattr(ray.train, "get_context", lambda: context)
        patch.setattr(ray.train, "report", lambda *args, **kwargs: None)
        patch.setattr(ray.train.torch, "get_device", lambda: torch.device("cuda:0"))
        patch.setattr(ray.train.torch, "prepare_model", lambda model, **kwargs: nn.parallel.DistributedDataParallel(
            model, find_unused_parameters=True))
        patch.setattr(cache, "CachedTrajectories", Dataset)
        patch.setattr(sampling, "RoleSampler", IndependentSampler)
        patch.setattr(modeling, "build_model", lambda config: (RuntimeModel(), None))
        patch.setattr(engine, "collate_examples", lambda *args, **kwargs: {})
        patch.setattr(torch, "autocast", lambda *args, **kwargs: nullcontext())
        for name in ("set_device", "synchronize", "reset_peak_memory_stats", "manual_seed_all"):
            patch.setattr(torch.cuda, name, lambda *args, **kwargs: None)
        for name in ("memory_allocated", "memory_reserved", "max_memory_allocated", "max_memory_reserved"):
            patch.setattr(torch.cuda, name, lambda *args, **kwargs: 1024 * (rank + 1))
        patch.setattr(torch.cuda, "get_device_name", lambda *args: "cpu-gloo-test")
        patch.setattr(torch.cuda, "mem_get_info", lambda *args: (2**30, 2**31))
        for name in ("tensor", "zeros"):
            original = getattr(torch, name)

            def on_cpu(*args, _original=original, **kwargs):
                if str(kwargs.get("device", "")).startswith("cuda"):
                    kwargs["device"] = "cpu"
                return _original(*args, **kwargs)

            patch.setattr(torch, name, on_cpu)
        broadcast = distributed.broadcast_object_list
        patch.setattr(distributed, "broadcast_object_list", lambda objects, **kwargs: broadcast(
            objects, **{**kwargs, "device": torch.device("cpu")}))
        def build_plan(*args, **kwargs):
            if failure_mode == "plan_failure" and rank == 1:
                raise ValueError("injected evaluation plan failure")
            return plan

        def require_encoder():
            if failure_mode == "encoder_failure":
                raise ValueError("injected evaluation encoder failure")

        patch.setattr(evaluation, "build_evaluation_plan", build_plan)
        patch.setattr(deploy_dense, "require_encoder", require_encoder)
        patch.setattr(deploy_dense, "build_dense_plan", lambda *args, **kwargs: plan)
        patch.setattr(render_plan, "build_render_plan", lambda *args, **kwargs: plan)
        patch.setattr(rendering, "render_evaluation", lambda *args, **kwargs: {
            "rendering_seconds": 0., "rendered_trajectories": 2})
        patch.setattr(evaluation, "evaluate", evaluate)
        patch.setattr(deploy_dense, "render_dense_evaluation", lambda *args, **kwargs: {
            "rendering_seconds": 0., "items": []})
        logger = Logger()
        if failure_mode == "media_failure":
            def fail(*args, **kwargs):
                raise ValueError("injected media persistence failure")
            logger.log_deploy_dense = fail
        if failure_mode == "commit_failure":
            def fail_commit(*, step):
                raise ValueError("injected history commit failure")
            logger.commit_step = fail_commit
        cfg = load_config(recipe)
        try:
            engine._train_worker(cfg, logger)
        except RuntimeError as error:
            failures = {"media_failure": "injected media persistence failure",
                        "commit_failure": "injected history commit failure",
                        "plan_failure": "injected evaluation plan failure",
                        "encoder_failure": "injected evaluation encoder failure"}
            if failure_mode not in failures:
                raise
            assert failures[failure_mode] in str(error)
            (Path(recipe).parent / f"gloo-proof-{rank}.json").write_text(json.dumps({"error": str(error)}))
            return
        from test_distributed_evaluation import assert_equivalent
        prefixes = ["evaluation"] if cfg["evaluation"]["mode"] == "dense" else [
            "evaluation", "render-evaluation", "deploy-dense-evaluation"]
        paths = [Path(cfg["training"]["output_dir"]) / f"{prefix}-{step:06d}.json"
                 for step in (1, 2) for prefix in prefixes]
        for path, metrics in zip(paths, full_metrics, strict=rank == 0):
            assert_equivalent(json.loads(path.read_text())["metrics"], metrics)
        (Path(recipe).parent / f"gloo-proof-{rank}.json").write_text(json.dumps({
            "evaluation_calls": len(evaluations), "status": "completed",
            "all_metrics_match_single_rank": True}))
    finally:
        patch.undo()
        distributed.destroy_process_group()


@pytest.mark.parametrize("evaluation_mode", ["dense", "sampled", "media_failure", "commit_failure", "plan_failure", "encoder_failure"])
def test_two_workers_preserve_common_initialization_observed_budget_and_eval_barriers(tmp_path, evaluation_mode):
    import torch.multiprocessing as multiprocessing

    recipe = tmp_path / "two-workers.yaml"
    output = tmp_path / "run"
    recipe.write_text(yaml.safe_dump({
        "roles": {"A": 0, "B": 16, "C": 24, "D": 24},
        "data": {"caches": ["/unused/train"], "max_pixels": 76800},
        "training": {"stage": "posttrain", "max_steps": 2, "warmup_steps": 0,
                     "num_workers": 2, "global_batch_size": 128, "gradient_accumulation_steps": 2,
                     "save_steps": 2, "output_dir": str(output)},
        "evaluation": {"enabled": True, "mode": "sampled" if evaluation_mode == "sampled" else "dense",
                       "caches": ["/unused/eval"], "renders": {"enabled": evaluation_mode == "sampled"},
                       "every_steps": 1, "max_trajectories_per_task": None,
                       "deploy_dense": {"enabled": True, "max_trajectories_per_task": None}},
    }))
    from test_distributed_evaluation import make_plan
    plan = make_plan(tmp_path)
    multiprocessing.spawn(_gloo_worker, args=(str(recipe), f"file://{tmp_path / 'rendezvous'}", plan, evaluation_mode),
                          nprocs=2, join=True)
    if evaluation_mode in ("plan_failure", "encoder_failure"):
        expected = ("injected evaluation plan failure" if evaluation_mode == "plan_failure"
                    else "injected evaluation encoder failure")
        for rank in (0, 1):
            assert expected in json.loads((tmp_path / f"gloo-proof-{rank}.json").read_text())["error"]
        assert not (output / "training_metrics.jsonl").exists()
        assert not (output / "result.json").exists()
        return
    if evaluation_mode in ("media_failure", "commit_failure"):
        expected = ("injected media persistence failure" if evaluation_mode == "media_failure"
                    else "injected history commit failure")
        for rank in (0, 1):
            assert expected in json.loads((tmp_path / f"gloo-proof-{rank}.json").read_text())["error"]
        assert len((output / "training_metrics.jsonl").read_text().splitlines()) == 1
        assert not (output / "result.json").exists()
        return
    receipts = [json.loads((output / f"initialization-{rank:03d}.json").read_text()) for rank in range(2)]
    assert receipts[0]["all_weights_sha256"] == receipts[1]["all_weights_sha256"]
    assert all(receipt["all_workers_initial_weights_equal"] for receipt in receipts)
    assert [receipt["effective_sampler_seed"] for receipt in receipts] == [42, 43]
    for rank in range(2):
        accounting = json.loads((output / f"batch-accounting-{rank:03d}.json").read_text())
        assert accounting["actual_rows_per_update_local"] == {"A": 0, "B": 16, "C": 24, "D": 24}
        assert accounting["actual_rows_per_update_global"] == {"A": 0, "B": 32, "C": 48, "D": 48}
        assert accounting["forward_role_rows_per_worker"] == [
            {"A": 0, "B": 8, "C": 24, "D": 0}, {"A": 0, "B": 8, "C": 0, "D": 24}]
        proof = json.loads((tmp_path / f"gloo-proof-{rank}.json").read_text())
        assert proof["status"] == "completed"
        assert proof["evaluation_calls"] == (6 if evaluation_mode == "sampled" else 2)
        assert proof["all_metrics_match_single_rank"]
    result = json.loads((output / "result.json").read_text())
    assert result["checkpoint_reload_equal"] and result["checkpoint_prediction_parity"]
    assert len(result["evaluation_events"]) == 2
    assert all(event["execution_world_size"] == 2 for event in result["evaluation_events"])
    for step in (1, 2):
        merged = json.loads((output / f"evaluation-{step:06d}.json").read_text())
        assert [s["window_count"] for s in merged["parallel_execution"]["shards"]] == [64, 5]
        assert merged["model_calls"] == 207
        assert merged["parallel_execution"]["active_workers"] == 2
        assert len(list((output / "evaluation_shards" / f"step-{step:06d}" / "primary").glob("rank-*.json"))) == 2
    assert result["runtime_summary"]["actual_role_rows_over_run_global"] == {
        "A": 0, "B": 64, "C": 96, "D": 96}
    coverage = result["sampling_coverage"]
    assert coverage["derived_D_rows"] == 96
    for role, per_rank in (("B", 32), ("C", 48)):
        assert coverage["independent_profile_order_by_role"][role]["irregular_stratified"]["backward"] == per_rank
        assert coverage["independent_profile_order_by_role"][role]["regular_ctsp"]["forward"] == per_rank
        assert coverage["independent_poi_slots_by_role"][role]["2"] == per_rank
        assert coverage["independent_poi_slots_by_role"][role]["5"] == per_rank
