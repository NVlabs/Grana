# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from grana.data.cache import Trajectory
from grana.training.evaluation import build_evaluation_plan, evaluate, record_diagnostics
from grana.training.losses import FRAME_WEIGHTS
from grana.data.sampling import CTSP9, RoleSampler, frame_indices
from grana.config import load_config
from grana.training.deploy_dense import build_dense_plan, dense_series
from grana.training.render_plan import build_render_plan
from grana.training.rendering import _validated_records


DATA = {"views": ["top"], "max_pixels": 76800, "stride_pairs": [[1, 1], [2, 1]]}
EVAL = {"max_trajectories_per_task": 2, "windows_per_trajectory": 2, "seed": 1729,
        "stride_pairs": [[1, 1], [2, 1]]}


def dataset(rows):
    return SimpleNamespace(records=tuple(rows), by_role={
        role: {task: tuple(row for row in rows if row.task == task and role in row.roles)
               for task in sorted({row.task for row in rows if role in row.roles})}
        for role in "ABC"})


@pytest.fixture
def inventories(tmp_path):
    def make_row(identity, task="task", stages=3, roles=("A", "B", "C"), frames=9):
        path = tmp_path / f"{identity}.npy"
        pixels = np.broadcast_to(np.arange(frames, dtype=np.uint8)[:, None, None, None], (frames, 2, 3, 3)).copy()
        np.save(path, pixels)
        return Trajectory(tmp_path, identity, task, frames, (path,), pixels.shape, ("top",), roles,
                          tuple(i / max(frames - 1, 1) for i in range(frames)), (0,) * frames,
                          (.2,) * frames, tuple(f"Operation {i}" for i in range(stages)), "Perform the entire task.")
    return make_row


def collate(_processor, examples, _max_pixels, _device):
    return {"examples": examples}


class FixedPredictions(nn.Module):
    """Predict stage 2 only at literal slot 5; progress is always 0.5."""
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.seen = []

    def forward(self, inputs, roles, stage_counts):
        assert not self.training and not torch.is_grad_enabled()
        torch.rand(1)  # Evaluation must preserve training's torch RNG even here.
        example = inputs["examples"][0]
        self.seen.append(example)
        if roles == ["C"]:
            assert example.focus_stage is None and "Focus stage:" not in example.prompt
            logits = torch.full((1, 8, max(3, stage_counts[0])), -3.)
            logits[:, :, 0] = 3.
            logits[:, 5, 0] = -3.
            logits[:, 5, min(2, stage_counts[0] - 1)] = 3.
            logits[:, :, stage_counts[0]:] = -torch.inf
            return {"stage_logits": logits, "progress_logits": None, "stage_indices": [0], "progress_indices": []}
        return {"progress_logits": torch.zeros(1, 8, 10), "stage_logits": None,
                "progress_indices": [0], "stage_indices": []}


def test_plan_covers_tasks_and_exact_calls_without_images_or_rng(inventories, monkeypatch):
    rows = [inventories(f"{task}{i}", task=task) for task in ("first", "second") for i in range(3)]
    heldout, train = dataset(rows), dataset([inventories("train")])
    sampler = RoleSampler(train, DATA, seed=41)
    before = {role: generator.getstate() for role, generator in sampler._rng.items()}
    monkeypatch.setattr(Trajectory, "images", lambda *args: pytest.fail("plan must not load RGB"))
    plan = build_evaluation_plan(heldout, EVAL, DATA, training_dataset=train)
    manifest = plan.manifest()
    assert manifest["task_count"] == 2 and manifest["trajectory_count"] == 4
    assert manifest["window_count"] == 8 and manifest["model_calls"] == 32
    assert manifest["role_window_counts"] == dict.fromkeys("ABCD", 8)
    assert {window.poi for window in plan.windows} == {3, 6}
    assert all(window.manifest()["frame_indices"][5] == window.poi for window in plan.windows)
    assert before == {role: generator.getstate() for role, generator in sampler._rng.items()}
    reversed_plan = build_evaluation_plan(dataset(rows[::-1]), EVAL, DATA, training_dataset=train)
    assert reversed_plan.manifest() == manifest


def test_default_eval_uses_fixed_c16_t4_while_training_draws_four_pairs(inventories):
    data = {"views": ["top"], "max_pixels": 76800}
    heldout, train = dataset([inventories("heldout", frames=200)]), dataset([inventories("train")])
    sampler = RoleSampler(train, data)
    eval_config = {key: value for key, value in EVAL.items() if key != "stride_pairs"}
    base = build_evaluation_plan(heldout, eval_config, data, training_dataset=train)
    dense, render = build_dense_plan(base), build_render_plan(base)
    assert sampler.stride_pairs == ((10, 2), (16, 4), (24, 6), (32, 8))
    for plan in (base, dense, render):
        assert plan.settings["stride_pairs"] == [[16, 4]]
        assert all(window.stride_pair == (16, 4) for window in plan.windows)
        assert all(window.manifest()["frame_indices"] == list(frame_indices(window.poi, 200, 16, 4)) for window in plan.windows)


def test_training_stride_changes_cannot_change_eval_but_explicit_eval_override_can(inventories):
    heldout, train = dataset([inventories("heldout", frames=200)]), dataset([inventories("train")])
    eval_config = {key: value for key, value in EVAL.items() if key != "stride_pairs"}
    manifests = [build_evaluation_plan(heldout, eval_config, {**DATA, "stride_pairs": pairs},
                                      training_dataset=train).manifest()
                 for pairs in ([[10, 2]], [[40, 4]], [[24, 6], [32, 8]])]
    assert manifests[0] == manifests[1] == manifests[2]
    changed = build_evaluation_plan(heldout, {**eval_config, "stride_pairs": [[24, 6]]}, DATA,
                                    training_dataset=train)
    assert changed.settings["stride_pairs"] == [[24, 6]]
    assert all(window.stride_pair == (24, 6) for window in changed.windows)
    assert changed.manifest()["plan_sha256"] != manifests[0]["plan_sha256"]


def test_explicit_historical_plan_keeps_its_stride_support_and_hash(inventories):
    data = {**DATA, "stride_pairs": [list(pair) for pair in CTSP9]}
    heldout, train = dataset([inventories("heldout", frames=200)]), dataset([inventories("train")])
    frozen = build_evaluation_plan(heldout, {**EVAL, "stride_pairs": [list(pair) for pair in CTSP9]}, data,
                                   training_dataset=train)
    before = frozen.manifest()
    current = build_evaluation_plan(heldout, {key: value for key, value in EVAL.items() if key != "stride_pairs"},
                                    {"views": ["top"], "max_pixels": 76800}, training_dataset=train)
    assert current.settings["stride_pairs"] == [[16, 4]]
    for plan in (frozen, build_dense_plan(frozen), build_render_plan(frozen)):
        assert plan.settings["stride_pairs"] == [list(pair) for pair in CTSP9]
        assert all(window.stride_pair in CTSP9 for window in plan.windows)
    assert frozen.manifest() == before


def test_plan_hash_tracks_effective_camera_inventory_and_order(inventories):
    first, second = inventories("one"), inventories("two")
    two_views = replace(second, view_names=("wrist", "left"),
                        frame_paths=(second.frame_paths[0], inventories("other_camera").frame_paths[0]))
    swapped = replace(two_views, view_names=two_views.view_names[::-1], frame_paths=two_views.frame_paths[::-1])
    plans = [build_evaluation_plan(dataset([first, row]), EVAL, DATA, training_dataset=dataset([])).manifest()
             for row in (second, two_views, swapped)]
    assert len({plan["plan_sha256"] for plan in plans}) == 3
    assert all(plan["settings"]["views"] == ["top"] for plan in plans)
    mixed = plans[1]
    assert mixed["settings"]["view_sets"] == [["top"], ["wrist", "left"]]
    assert {tuple(w["view_names"]) for w in mixed["windows"]} == {("top",), ("wrist", "left")}
    assert all(plan["model_calls"] == 16 for plan in plans)


def test_plan_hash_and_execution_track_preprocessing_contract(inventories):
    heldout = dataset([inventories("heldout", roles=("A",))])
    plans = [build_evaluation_plan(heldout, EVAL, {**DATA, "max_pixels": budget}, training_dataset=dataset([]))
             for budget in (None, 76800)]
    assert plans[0].manifest()["plan_sha256"] != plans[1].manifest()["plan_sha256"]
    assert plans[0].settings["preprocessing"] == {"version": 1, "policy": "processor_default", "max_pixels": None}
    assert plans[1].settings["preprocessing"] == {"version": 1, "policy": "pixel_budget", "max_pixels": 76800}
    model = FixedPredictions()
    with pytest.raises(ValueError, match="fixed plan's preprocessing"):
        evaluate(model, None, plans[0], device="cpu", max_pixels=76800, collate_fn=collate)
    assert model.seen == []
    seen_budgets = []

    def capture(processor, examples, max_pixels, device):
        seen_budgets.append(max_pixels)
        return collate(processor, examples, max_pixels, device)

    result = evaluate(model, None, plans[0], device="cpu", max_pixels=None, collate_fn=capture)
    assert result["model_calls"] == len(seen_budgets) == 2
    assert seen_budgets == [None, None]


def test_gpu_resize_is_frozen_in_plan_and_used_for_every_evaluation_batch(inventories, monkeypatch):
    from grana.training import evaluation

    heldout = dataset([inventories("heldout", roles=("B", "C"))])
    cpu = build_evaluation_plan(heldout, EVAL, DATA, training_dataset=dataset([]))
    gpu = build_evaluation_plan(heldout, EVAL, {**DATA, "gpu_image_resize": True},
                                training_dataset=dataset([]))
    assert "gpu_image_resize" not in cpu.settings  # Preserve historical plan hashes.
    assert gpu.settings["gpu_image_resize"] is True
    assert gpu.settings["preprocessing"]["version"] == 2
    assert gpu.settings["preprocessing"]["resize_backend"] == "torchvision_cuda"
    assert gpu.manifest()["plan_sha256"] != cpu.manifest()["plan_sha256"]
    original, configured = object(), object()
    calls, processors = [], []

    def configure(processor, device, *, resize_on_device):
        calls.append((processor, str(device), resize_on_device))
        return configured

    def capture(processor, examples, budget, device):
        processors.append(processor)
        return collate(processor, examples, budget, device)

    monkeypatch.setattr(evaluation, "configure_image_preprocessing", configure)
    result = evaluate(FixedPredictions(), original, gpu, device="cpu", max_pixels=76800, collate_fn=capture)
    assert calls == [(original, "cpu", True)]
    assert len(processors) == result["forward_calls"]
    assert all(processor is configured for processor in processors)
    assert result["image_preprocessing"] == gpu.settings["preprocessing"]
    damaged = replace(gpu, settings={**gpu.settings, "gpu_image_resize": False})
    with pytest.raises(ValueError, match="fixed plan's preprocessing"):
        evaluate(FixedPredictions(), original, damaged, device="cpu", max_pixels=76800, collate_fn=capture)


def test_gpu_resize_evaluation_rejects_cpu_execution_before_collation(inventories):
    plan = build_evaluation_plan(dataset([inventories("heldout", roles=("A",))]), EVAL,
                                 {**DATA, "gpu_image_resize": True}, training_dataset=dataset([]))
    with pytest.raises(ValueError, match="CUDA"):
        evaluate(FixedPredictions(), None, plan, device="cpu", max_pixels=76800,
                 collate_fn=lambda *args: pytest.fail("CPU fallback must not execute"))


@pytest.mark.parametrize("same_identity", [True, False])
def test_plan_rejects_train_identity_or_payload_overlap(inventories, same_identity):
    train_row = inventories("train")
    heldout_row = inventories("eval")
    if same_identity:
        heldout_row = replace(heldout_row, trajectory_id="train")
    else:
        heldout_row = replace(heldout_row, frame_paths=train_row.frame_paths)
    with pytest.raises(ValueError, match="overlaps training"):
        build_evaluation_plan(dataset([heldout_row]), EVAL, DATA, training_dataset=dataset([train_row]))


def test_plan_all_trajectories_and_short_windows_have_explicit_counts(inventories):
    rows = [inventories("one", frames=1), inventories("two", frames=2)]
    plan = build_evaluation_plan(dataset(rows), {**EVAL, "max_trajectories_per_task": None, "windows_per_trajectory": 5},
                                 DATA, training_dataset=dataset([]))
    assert plan.manifest()["trajectory_count"] == 2
    assert plan.manifest()["window_count"] == 3 and plan.manifest()["model_calls"] == 12
    assert len({(w.trajectory.trajectory_id, w.poi) for w in plan.windows}) == 3


def test_evaluate_exact_poi_metrics_loss_mass_and_predicted_d_media(inventories):
    plan = build_evaluation_plan(dataset([inventories("eval")]), EVAL, DATA, training_dataset=dataset([]))
    model = FixedPredictions().train()
    rng_before = torch.random.get_rng_state().clone()
    result = evaluate(model, None, plan, device="cpu", max_pixels=76800, collate_fn=collate,
                      role_weights={"A": 1., "B": 1., "C": .25, "D": 1.})
    assert model.training and torch.equal(rng_before, torch.random.get_rng_state())
    assert result["model_calls"] == 8 and result["task_count"] == 1
    assert result["metrics"]["role_A_poi_rmse"] == pytest.approx(math.sqrt(((.5 - .375) ** 2 + (.5 - .75) ** 2) / 2), abs=1e-7)
    assert result["metrics"]["role_B_poi_rmse"] == pytest.approx(.3, abs=1e-7)
    assert result["metrics"]["role_D_poi_rmse"] == pytest.approx(.5, abs=1e-7)
    assert result["metrics"]["role_C_poi_accuracy"] == 0
    assert result["roles"]["C"]["dense_accuracy"] == 7 / 8
    assert result["metrics"]["role_C_poi_mae"] == 2
    assert result["metrics"]["role_C_dense_mae"] == 2 / 8
    assert result["metrics"]["role_C_poi_far_error_rate"] == 1
    assert result["metrics"]["role_C_dense_far_error_rate"] == 1 / 8
    assert result["metrics"]["task/task/role_C_poi_mae"] == 2
    assert result["metrics"]["role_C_task_macro_poi_mae"] == 2
    assert result["roles"]["C"]["dense_slots"] == 16
    assert result["roles"]["C"]["dense_weight_mass"] == pytest.approx(2 * sum(FRAME_WEIGHTS))
    denominator = math.exp(3) + 2 * math.exp(-3)
    small, large = math.exp(-3) / denominator, math.exp(3) / denominator
    correct_loss = -math.log(large) + ((2 * small) ** 2 + small ** 2) / 2
    wrong_loss = -math.log(small) + ((1 - small) ** 2 + (1 - 2 * small) ** 2) / 2
    expected_loss = (correct_loss * (sum(FRAME_WEIGHTS) - 2) + wrong_loss * 2) / sum(FRAME_WEIGHTS)
    assert result["roles"]["C"]["dense_weighted_loss"] == pytest.approx(expected_loss, rel=1e-6)
    for i in range(0, len(model.seen), 4):
        a, b, c, d = model.seen[i:i + 4]
        assert [x.role for x in (a, b, c, d)] == list("ABCD")
        assert b.focus_stage == 0 and d.focus_stage == 2
        assert d.images is c.images and d.frame_indices is c.frame_indices
        assert d.targets == (0.,) * 8
    assert all(math.isfinite(value) for value in result["metrics"].values())
    assert all(record["elapsed_seconds"] > 0 for record in result["records"])


@pytest.mark.parametrize("stage_cdf,progress_cdf,ceiling", [(2.5, 1.7, 4.0), (0.0, 0.0, 0.0)])
def test_evaluation_uses_independent_ce_cdf_weights(inventories, stage_cdf, progress_cdf, ceiling):
    plan = build_evaluation_plan(dataset([inventories("eval")]), EVAL, DATA,
                                 training_dataset=dataset([]), roles=("B", "C", "D"))
    settings = {"stage": {"cdf_weight": stage_cdf},
                "progress": {"cdf_weight": progress_cdf, "ceiling_weight": ceiling}}
    result = evaluate(FixedPredictions(), None, plan, device="cpu", max_pixels=76800,
                      loss_config=settings, collate_fn=collate)
    denominator = math.exp(3) + 2 * math.exp(-3)
    small, large = math.exp(-3) / denominator, math.exp(3) / denominator
    correct_loss = -math.log(large) + stage_cdf * ((2 * small) ** 2 + small ** 2) / 2
    wrong_loss = -math.log(small) + stage_cdf * ((1 - small) ** 2 + (1 - 2 * small) ** 2) / 2
    # D's native stage-2 focus is ahead of ground truth at every slot: q=e_0.
    expected_progress = math.log(10) + progress_cdf * sum((i / 10 - 1) ** 2 for i in range(1, 10)) / 9
    expected_progress += ceiling * (.5 - 1 / 9)
    expected_stage = (correct_loss * (sum(FRAME_WEIGHTS) - FRAME_WEIGHTS[5])
                      + wrong_loss * FRAME_WEIGHTS[5]) / sum(FRAME_WEIGHTS)
    assert result["roles"]["C"]["dense_weighted_loss"] == pytest.approx(expected_stage, rel=1e-6)
    assert result["roles"]["D"]["dense_weighted_loss"] == pytest.approx(expected_progress, rel=1e-6)
    assert result["loss_config"]["stage"] == settings["stage"]
    assert result["loss_config"]["progress"] == settings["progress"]
    assert result["loss_config"]["frame_weights"] == list(FRAME_WEIGHTS)
    # Configuration changes the objective; C still routes its own prediction to D.
    c_records = [record for record in result["records"] if record["role"] == "C"]
    d_records = [record for record in result["records"] if record["role"] == "D"]
    assert all(record["predictions"][5] == 2 for record in c_records)
    assert all(record["focus_stage"] == 2 and record["targets"] == [0.] * 8 for record in d_records)


def test_stage_error_macro_averages_tasks_instead_of_windows(inventories):
    plan = build_evaluation_plan(dataset([inventories("short", task="one", stages=2, frames=1),
                                          inventories("long", task="two", stages=3, frames=9)]),
                                 EVAL, DATA, training_dataset=dataset([]), roles=("C", "D"))
    result = evaluate(FixedPredictions(), None, plan, device="cpu", max_pixels=76800,
                      collate_fn=collate)
    assert result["metrics"]["role_C_poi_mae"] == pytest.approx(5 / 3)
    assert result["metrics"]["role_C_task_macro_poi_mae"] == 1.5
    assert result["metrics"]["role_C_poi_far_error_rate"] == pytest.approx(2 / 3)
    assert result["metrics"]["role_C_task_macro_poi_far_error_rate"] == .5
    assert result["per_task"]["one"]["C"]["poi_far_error_rate"] == 0
    assert result["per_task"]["two"]["C"]["poi_far_error_rate"] == 1


def test_a_only_holdout_does_not_invent_unassessed_role_metrics(inventories):
    plan = build_evaluation_plan(dataset([inventories("atomic", stages=1, roles=("A",))]), EVAL, DATA,
                                 training_dataset=dataset([]), roles={"A": 2, "B": 2, "C": 1, "D": 1})
    assert plan.manifest()["role_window_counts"] == {"A": 2, "B": 0, "C": 0, "D": 0}
    result = evaluate(FixedPredictions(), None, plan, device="cpu", max_pixels=76800, collate_fn=collate)
    assert result["model_calls"] == 2
    assert all(result["roles"][role]["status"] == "unassessed" for role in "BCD")
    assert not any(f"role_{role}_" in key for key in result["metrics"] for role in "BCD")


def test_single_stage_native_c_has_zero_loss_and_d_uses_only_stage(inventories):
    plan = build_evaluation_plan(dataset([inventories("single", stages=1)]), EVAL, DATA,
                                 training_dataset=dataset([]), roles=("C", "D"))
    model = FixedPredictions()
    result = evaluate(model, None, plan, device="cpu", max_pixels=76800, collate_fn=collate)
    assert result["roles"]["C"]["dense_weighted_loss"] == 0
    assert result["metrics"]["role_C_poi_accuracy"] == 1
    assert result["metrics"]["role_D_poi_rmse"] == pytest.approx(.3, abs=1e-7)
    for c, d in zip(model.seen[::2], model.seen[1::2], strict=True):
        assert "A: Stage 0: Operation 0" in c.prompt and "Focus stage:" not in c.prompt
        assert d.focus_stage == 0 and d.images is c.images and d.targets == (.2,) * 8


def test_eval_failure_restores_mode_and_rng(inventories):
    plan = build_evaluation_plan(dataset([inventories("eval")]), EVAL, DATA, training_dataset=dataset([]))
    model = FixedPredictions().train()
    rng_before = torch.random.get_rng_state().clone()

    def fail(*args):
        torch.rand(1)
        raise RuntimeError("test collation failure")

    with pytest.raises(RuntimeError, match="collation failure"):
        evaluate(model, None, plan, device="cpu", max_pixels=76800, collate_fn=fail)
    assert model.training and torch.equal(rng_before, torch.random.get_rng_state())


def test_support_aware_stage_and_progress_diagnostics_include_wrong_focus_and_baselines():
    records = []
    for poi, (true, predicted, b_target, b_prediction, d_target, d_prediction) in enumerate([
        (0, 0, 0., .1, 0., .2), (0, 2, .5, .6, 0., .1), (1, 1, .8, .9, .8, .4),
    ]):
        for role, target, prediction in [("C", true, predicted), ("B", b_target, b_prediction), ("D", d_target, d_prediction)]:
            records.append({"cache": "/exact/cache", "task": "unchanged task", "trajectory_id": "heldout", "poi": poi,
                            "role": role, "group": "L1L2", "num_stages": 4,
                            "targets": [target] * 8, "predictions": [prediction] * 8})
    diagnostics, metrics = record_diagnostics(records)
    overall = diagnostics["overall"]
    assert overall["C"]["poi_accuracy"] == pytest.approx(2 / 3)
    assert overall["C"]["stage_macro_accuracy"] == .75
    assert overall["C"]["stage_support"] == [2, 1, 0, 0]
    assert overall["C"]["stage_accuracy"] == [.5, 1., None, None]
    assert overall["C"]["confusion_matrix"] == [[1, 0, 1, 0], [0, 1, 0, 0], [0] * 4, [0] * 4]
    assert overall["C"]["stage_mae"] == pytest.approx(2 / 3)
    assert overall["C"]["abs_error_ge_2_rate"] == pytest.approx(1 / 3)
    assert overall["B"]["mae"] == pytest.approx(.1)
    assert overall["B"]["accuracy_within_0.1"] == 1.
    assert overall["B"]["constant_0.5_rmse"] == pytest.approx(math.sqrt(.34 / 3))
    assert overall["D"]["correct_C"]["support"] == 2
    assert overall["D"]["correct_C"]["rmse"] == pytest.approx(math.sqrt(.1))
    assert overall["D"]["incorrect_C"]["support"] == 1
    assert overall["D"]["incorrect_C"]["accuracy_within_0.1"] == 1.
    assert overall["D"]["gt_lt_0.05"]["support"] == 2
    assert overall["D"]["gt_lt_0.05"]["rmse"] == pytest.approx(math.sqrt(.025))
    assert overall["A"]["status"] == "unassessed" and overall["A"]["rmse"] is None
    assert diagnostics["per_group"]["L1L2"] == overall
    assert metrics["diagnostics/per_group/L1L2/C/stage_2/support"] == 0
    assert "diagnostics/per_group/L1L2/C/stage_2/accuracy" not in metrics
    assert all(math.isfinite(value) for value in metrics.values())


class VaryingBatchPredictions(nn.Module):
    """Outputs depend on each observation/focus, never its batch position."""
    def __init__(self):
        super().__init__()
        self.seen = []
        self.batch_sizes = []
        self.c_examples = {}

    def forward(self, inputs, roles, stage_counts):
        assert not self.training and not torch.is_grad_enabled()
        torch.rand(1)
        examples = inputs['examples']
        assert len(examples) == len(roles) == len(stage_counts)
        self.seen.extend(examples)
        self.batch_sizes.append(len(examples))
        if roles[0] == 'C':
            logits = torch.full((len(examples), 8, max(stage_counts)), -torch.inf)
            for i, (example, count) in enumerate(zip(examples, stage_counts, strict=True)):
                assert example.focus_stage is None
                logits[i, :, :count] = -3.
                for slot in range(8):
                    logits[i, slot, (example.poi + slot) % count] = 3.
                self.c_examples[example.task, example.trajectory_id, example.poi] = example
            return {'stage_logits': logits}
        logits = torch.zeros(len(examples), 8, 10)
        for i, example in enumerate(examples):
            if example.role == 'D':
                paired = self.c_examples[example.task, example.trajectory_id, example.poi]
                assert example.images is paired.images
                assert example.frame_indices is paired.frame_indices
                assert example.view_names is paired.view_names
                assert example.focus_stage == (example.poi + 5) % example.num_stages
            for slot in range(8):
                logits[i, slot, (example.poi + slot + (example.focus_stage or 0)) % 10] = 2.
        return {'progress_logits': logits}


def without_record_times(result):
    return [{key: value for key, value in record.items() if key != 'elapsed_seconds'}
            for record in result['records']]


def assert_same_assessment(left, right):
    assert without_record_times(left) == without_record_times(right)
    for key in ('metrics', 'roles', 'per_task', 'diagnostics', 'loss_config', 'model_calls'):
        assert left[key] == right[key]


@pytest.mark.parametrize('count', [1, 63, 64, 65, 129])
@pytest.mark.parametrize('batch_size', [1, 32, 64])
def test_batching_preserves_every_window_and_per_window_c_to_d_pair(inventories, count, batch_size):
    row = inventories('eval', frames=count)
    baseline_plan = build_evaluation_plan(dataset([row]), {**EVAL, 'windows_per_trajectory': count}, DATA,
                                          training_dataset=dataset([]), roles=('B', 'C', 'D'))
    plan = build_evaluation_plan(dataset([row]), {**EVAL, 'windows_per_trajectory': count,
                                                'batch_size': batch_size}, DATA,
                                training_dataset=dataset([]), roles=('B', 'C', 'D'))
    assert plan.manifest()['windows'] == baseline_plan.manifest()['windows']
    baseline = evaluate(VaryingBatchPredictions(), None, baseline_plan, device='cpu',
                        max_pixels=76800, collate_fn=collate)
    callbacks, model = [], VaryingBatchPredictions().train()
    rng_before = torch.random.get_rng_state().clone()
    result = evaluate(model, None, plan, device='cpu', max_pixels=76800, collate_fn=collate,
                      progress_callback=lambda done, total: callbacks.append((done, total)))
    assert model.training and torch.equal(rng_before, torch.random.get_rng_state())
    assert_same_assessment(baseline, result)
    chunks = [min(batch_size, count - offset) for offset in range(0, count, batch_size)]
    expected_forwards = [size for size in chunks for _ in range(3)]
    assert model.batch_sizes == expected_forwards
    assert result['window_batch_sizes'] == chunks
    assert result['forward_calls'] == len(chunks) * 3
    assert result['model_calls'] == 3 * count
    assert sum(int(size) * calls for size, calls in result['forward_batch_sizes'].items()) == 3 * count
    assert result['max_forward_batch_size'] == min(count, batch_size)
    assert callbacks == [(done, 3 * count) for done in range(1, 3 * count + 1)]
    assert result['cuda_memory'] is None
    assert all(value >= 0 for value in result['timing_seconds'].values())
    assert result['windows_per_second'] > 0 and result['role_rows_per_second'] > 0
    assert all(len(record['slot_losses']) == 8 for record in result['records'])
    assert [(r['poi'], r['role']) for r in result['records']] == [
        (poi, role) for poi in range(count) for role in ('B', 'C', 'D')]


@pytest.mark.parametrize('roles', [('A',), ('B', 'C', 'D'), ('A', 'B', 'C', 'D')])
def test_mixed_stages_tasks_roles_and_prompts_have_batch_invariant_metrics(inventories, roles):
    rows = [inventories('one', task='single', stages=1, frames=1),
            inventories('nine', task='longer instruction', stages=9, frames=8),
            inventories('three', task='short instruction', stages=3, frames=9)]
    if 'A' in roles:
        rows.append(inventories('atomic', task='atomic', stages=0, roles=('A',), frames=7))
    evaluation = {**EVAL, 'windows_per_trajectory': 9, 'max_trajectories_per_task': None}
    results = []
    for batch_size in (1, 32, 64):
        plan = build_evaluation_plan(dataset(rows), {**evaluation, 'batch_size': batch_size}, DATA,
                                     training_dataset=dataset([]), roles=roles)
        result = evaluate(VaryingBatchPredictions(), None, plan, device='cpu', max_pixels=76800,
                          role_weights={'A': 2., 'B': 1., 'C': 10., 'D': 1.},
                          frame_weights=(.1, .2, .3, .5, 1., 2., 3., 4.),
                          loss_config={'stage': {'cdf_weight': .7},
                                       'progress': {'cdf_weight': 1.3, 'ceiling_weight': 2.}},
                          collate_fn=collate)
        results.append(result)
        if 'C' in roles:
            assert result['per_task']['single']['C']['dense_weighted_loss'] == 0
        if 'A' in roles:
            assert result['per_task']['atomic']['C']['status'] == 'unassessed'
    for result in results[1:]:
        assert_same_assessment(results[0], result)


@pytest.mark.parametrize('failure', ['collate', 'forward', 'callback'])
@pytest.mark.parametrize('training', [True, False])
def test_batch_failure_restores_rng_and_original_mode(inventories, failure, training):
    plan = build_evaluation_plan(dataset([inventories('eval', frames=65)]),
                                 {**EVAL, 'windows_per_trajectory': 65, 'batch_size': 64}, DATA,
                                 training_dataset=dataset([]), roles=('B', 'C', 'D'))
    model = VaryingBatchPredictions().train(training)
    rng_before = torch.random.get_rng_state().clone()
    def fail(*args):
        torch.rand(1)
        raise RuntimeError('intentional batch failure')
    if failure == 'forward':
        model.forward = fail
    with pytest.raises(RuntimeError, match='intentional batch failure'):
        evaluate(model, None, plan, device='cpu', max_pixels=76800,
                 collate_fn=fail if failure == 'collate' else collate,
                 progress_callback=fail if failure == 'callback' else None)
    assert model.training is training
    assert torch.equal(rng_before, torch.random.get_rng_state())


def test_frozen_legacy_plan_strides_explicitly_reconstruct_literal_pre_batching_hashes(tmp_path):
    # Fixture generated using git-show a05e497 source, never new-code round trips.
    fixture = json.loads((Path(__file__).parent / 'fixtures/evaluation_legacy_a05e497.json').read_text())
    assert fixture['source_commit'] == 'a05e497a07cbf3cc89cbb43ef77bd1216acc9647'
    assert 'batch_size' not in fixture['config']['evaluation']
    assert 'batched_vision_sdpa' not in fixture['config']['evaluation']
    path = tmp_path / 'resolved_config.json'
    path.write_text(json.dumps(fixture['config']))
    cfg = load_config(path)
    assert cfg['evaluation']['batch_size'] == 1
    # New configs never inherit evaluation strides from data. An explicit frozen
    # plan remains authoritative when replaying historical evaluation inputs.
    cfg['evaluation']['stride_pairs'] = fixture['sampled_manifest']['settings']['stride_pairs']
    row = Trajectory(Path('/legacy/heldout'), 'episode-7', 'assemble', 9,
                     (Path('/legacy/heldout/rgb.npy'),), (9, 2, 3, 3), ('top',), ('A', 'B', 'C'),
                     tuple(i / 8 for i in range(9)), (0,) * 9, (.2,) * 9,
                     ('Reach', 'Insert', 'Finish'), 'Assemble the item.')
    base = build_evaluation_plan(dataset([row]), cfg['evaluation'], cfg['data'], training_dataset=dataset([]))
    assert base.manifest() == fixture['sampled_manifest']
    assert build_dense_plan(base, frame_stride=3, max_trajectories_per_task=None).manifest() == fixture['dense_manifest']
    assert build_render_plan(base, points_per_trajectory=3, max_trajectories_per_task=None).manifest() == fixture['render_manifest']


@pytest.mark.parametrize('plan_kind', ['dense', 'render'])
def test_batched_records_feed_dense_and_render_without_new_model_calls(inventories, plan_kind):
    rows = [inventories('native', frames=65), inventories('atomic', task='atomic', roles=('A',), frames=5)]
    comparisons = []
    for batch_size in (1, 32, 64):
        base = build_evaluation_plan(dataset(rows), {**EVAL, 'batch_size': batch_size}, DATA,
                                     training_dataset=dataset([]))
        if plan_kind == 'dense':
            plan = build_dense_plan(base, frame_stride=1, max_trajectories_per_task=None)
        else:
            plan = build_render_plan(base, points_per_trajectory=65, max_trajectories_per_task=None)
        model = VaryingBatchPredictions()
        result = evaluate(model, None, plan, device='cpu', max_pixels=76800, collate_fn=collate)
        calls_before = len(model.batch_sizes)
        records = _validated_records(result, plan)
        assert sum(len(by_role) for sample in records.values() for by_role in sample.values()) == result['model_calls']
        series = dense_series(result, plan) if plan_kind == 'dense' else None
        assert len(model.batch_sizes) == calls_before
        comparisons.append((result, series))
    for result, series in comparisons[1:]:
        assert_same_assessment(comparisons[0][0], result)
        assert comparisons[0][1] == series


def stage_policy_data(*, count=10, enabled=True):
    return {**DATA, "train_stage_augmentation": {
        "enabled": enabled, "contiguous_shift": {"enabled": True, "candidate_count": count},
        "task_overrides": {"first": {"enabled": True}, "second": {"enabled": False}}}}


def test_stage_policy_filters_only_its_task_d_and_keeps_canonical_eval(inventories, monkeypatch):
    heldout = dataset([inventories("first-eval", task="first"), inventories("second-eval", task="second")])
    train = dataset([inventories("first-train", task="first"), inventories("second-train", task="second")])
    plan = build_evaluation_plan(heldout, EVAL, stage_policy_data(), training_dataset=train)
    assert {w.trajectory.task: w.roles for w in plan.windows} == {
        "first": ("A", "B", "C"), "second": ("A", "B", "C", "D")}
    routing = plan.settings["stage_routing"]
    assert routing["augmentation_enabled"] is False and routing["execution_mode"] == "eval"
    assert routing["task_policies"] == {"first": {"stage_augmentation_enabled": True, "d_allowed": False},
                                        "second": {"stage_augmentation_enabled": False, "d_allowed": True}}
    first_report = next(t for t in plan.tasks if t["task"] == "first")["stage_routing"]
    assert first_report["excluded_roles"] == {"D": "task_stage_augmentation_enabled"}
    assert first_report["role_window_counts"]["D"] == 0
    original, calls = RoleSampler.make_d, []

    def guarded(self, example, focus):
        calls.append(example.task)
        assert example.task == "second"
        return original(self, example, focus)

    monkeypatch.setattr(RoleSampler, "make_d", guarded)
    model = FixedPredictions()
    result = evaluate(model, None, plan, device="cpu", max_pixels=76800, collate_fn=collate)
    assert calls == ["second", "second"]
    assert result["per_task"]["first"]["D"]["status"] == "unassessed"
    assert "task/first/role_D_poi_rmse" not in result["metrics"]
    assert result["stage_augmentation"]["augmented_rows"] == 0
    assert result["stage_augmentation"]["candidate_count_histogram"] == {"3": 4}
    assert all(example.stage_choices is None for example in model.seen)
    for example in model.seen:
        if example.role == "C":
            assert tuple(example.targets) == tuple(example.stage_ids)
            assert "J:" not in example.prompt and example.num_stages == 3


def test_train_candidate_count_seed_and_donor_inventory_do_not_change_eval(inventories):
    heldout = dataset([inventories("first-eval", task="first"), inventories("second-eval", task="second")])
    donor = inventories("donor", task="donor")
    modified_donor = replace(donor, stage_prompts=("Different", "Training", "Descriptions"))
    plans = [build_evaluation_plan(heldout, EVAL, {**stage_policy_data(count=count), "seed": seed},
                                   training_dataset=dataset([row]))
             for count, seed, row in ((3, 1, donor), (10, 42, donor), (26, 999, modified_donor))]
    assert plans[0].manifest() == plans[1].manifest() == plans[2].manifest()
    # Exact eval input prompts/targets also remain canonical under capacity 3/10/26.
    seen = []
    for plan in plans:
        model = FixedPredictions()
        evaluate(model, None, plan, device="cpu", max_pixels=76800, collate_fn=collate)
        seen.append([(x.task, x.role, x.prompt, x.targets, x.frame_indices) for x in model.seen])
    assert seen[0] == seen[1] == seen[2]


def test_disabled_stage_augmentation_preserves_legacy_plan_identity(inventories):
    heldout = dataset([inventories("first-eval", task="first"), inventories("second-eval", task="second")])
    old = build_evaluation_plan(heldout, EVAL, DATA, training_dataset=dataset([]))
    explicit_off = build_evaluation_plan(heldout, EVAL, stage_policy_data(enabled=False), training_dataset=dataset([]))
    frozen = old.manifest()
    assert explicit_off.manifest() == frozen and "stage_routing" not in frozen["settings"]
    build_evaluation_plan(heldout, EVAL, stage_policy_data(), training_dataset=dataset([]))
    assert old.manifest() == frozen


def test_dense_and_render_preserve_task_d_filter_and_route_metric_names(inventories):
    from grana.training.deploy_dense import dense_metrics
    from grana.training.evaluation_reporting import compact_evaluation_metrics

    heldout = dataset([inventories("first-eval", task="first"), inventories("second-eval", task="second")])
    base = build_evaluation_plan(heldout, EVAL, stage_policy_data(), training_dataset=dataset([]))
    render = build_render_plan(base, points_per_trajectory=3)
    assert all("D" not in w.roles for w in render.windows if w.trajectory.task == "first")
    dense = build_dense_plan(base, frame_stride=4, group_by_cache={str(heldout.records[0].cache_root): "combined"})
    assert {w.trajectory.task: w.roles for w in dense.windows} == {
        "first": ("A", "C"), "second": ("B", "C", "D")}
    assert next(t for t in dense.tasks if t["task"] == "first")["stage_routing"]["role_window_counts"] == {
        "A": 3, "B": 0, "C": 3, "D": 0}
    result = evaluate(FixedPredictions(), None, dense, device="cpu", max_pixels=76800, collate_fn=collate)
    assert result["per_task"]["first"]["C"]["status"] == "assessed"
    series = dense_series(result, dense)
    metrics = dense_metrics(series)
    assert "eval_quality/first/predicted_stage_route_accumulated_progress_raw_rmse" not in metrics
    assert "eval_quality/second/predicted_stage_route_accumulated_progress_raw_rmse" in metrics
    assert "overlay_acc/combined_mixed_route_accumulated_progress_raw_rmse" in metrics
    assert "overlay_acc/combined_predicted_stage_route_accumulated_progress_raw_rmse" not in metrics
    compact = compact_evaluation_metrics(metrics, evaluation=result)
    assert "task/first/compressed_whole_task_progress_poi_micro_raw_rmse" in compact
    assert "task/second/predicted_stage_route_accumulated_progress_poi_micro_raw_rmse" in compact


def test_stage_only_augmented_task_dense_fails_without_silently_restoring_d(inventories):
    heldout = dataset([inventories("first-eval", task="first", roles=("B", "C"))])
    base = build_evaluation_plan(heldout, EVAL, stage_policy_data(), training_dataset=dataset([]))
    assert {w.roles for w in base.windows} == {("B", "C")}
    with pytest.raises(ValueError, match="ordinary stage evaluation"):
        build_dense_plan(base)


def test_new_plan_rejects_forged_d_route_before_images(inventories, monkeypatch):
    heldout = dataset([inventories("first-eval", task="first")])
    base = build_evaluation_plan(heldout, EVAL, stage_policy_data(), training_dataset=dataset([]))
    forged = replace(base, windows=tuple(replace(w, roles=(*w.roles, "D")) for w in base.windows))
    monkeypatch.setattr(Trajectory, "images", lambda *args: pytest.fail("forged D route decoded RGB"))
    for action in (lambda: forged.manifest(), lambda: build_render_plan(forged), lambda: build_dense_plan(forged),
                   lambda: evaluate(FixedPredictions(), None, forged, device="cpu", max_pixels=76800, collate_fn=collate)):
        with pytest.raises(ValueError, match="Role D is forbidden"):
            action()


@pytest.mark.parametrize("field,value", [("stage_augmentation_applied", True), ("stage_answer_count", 10)])
def test_render_rejects_augmented_eval_record_metadata(inventories, field, value):
    plan = build_evaluation_plan(dataset([inventories("eval")]), EVAL, DATA, training_dataset=dataset([]), roles=("C",))
    result = evaluate(FixedPredictions(), None, plan, device="cpu", max_pixels=76800, collate_fn=collate)
    result["records"][0][field] = value
    with pytest.raises(ValueError, match="augmentation|answer count"):
        _validated_records(result, plan)
