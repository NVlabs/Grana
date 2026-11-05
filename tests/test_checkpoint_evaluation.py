# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Saved checkpoint evaluation is a new run, without trainer-state restoration."""
from copy import deepcopy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from grana.config import DEFAULTS
from grana.data.cache import Trajectory
from grana.modeling.checkpoint import save_weights, tensor_digest, trainable_state
from grana.training.checkpoint_evaluation import prepare_checkpoint_evaluation, run
from grana.training.deploy_dense import build_dense_plan
from grana.training.evaluation import build_evaluation_plan


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(12, 4).requires_grad_(False)
        self.output = nn.Linear(4, 12, bias=False).requires_grad_(False)
        self.adapter = nn.Linear(4, 4)

    def get_input_embeddings(self):
        return self.embedding

    def get_output_embeddings(self):
        return self.output


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = TinyBackbone()
        self.progress_head = nn.Linear(4, 10)
        self.temporal_head = nn.Linear(4, 4)
        self.token_ids = {name: index for index, name in enumerate(("reward", "sim", "prog", "stage"), 8)}
        self.model_spec = {"name_or_path": "tiny", "revision": "fixed", "token_ids": self.token_ids,
                           "lora_rank": 2, "lora_alpha": 2}

    def forward(self, inputs, roles, stage_counts):
        assert not self.training and not torch.is_grad_enabled()
        count = len(roles)
        if roles[0] == "C":
            logits = torch.zeros(count, 8, 3)
            logits[:, :, 1] = 2
            return {"stage_logits": logits}
        return {"progress_logits": self.progress_head(torch.ones(count, 8, 4))}


def write(path, value):
    path.write_text(json.dumps(value) + "\n")


def inventory(rows):
    return SimpleNamespace(records=tuple(rows), by_role={
        role: {task: tuple(row for row in rows if row.task == task and role in row.roles)
               for task in sorted({row.task for row in rows if role in row.roles})} for role in "ABC"})


@pytest.fixture
def saved_run(tmp_path, monkeypatch, request):
    previous_threads = torch.get_num_threads()
    source = tmp_path / "unfinished-training"
    source.mkdir()
    pixels = tmp_path / "heldout.npy"
    np.save(pixels, np.zeros((5, 2, 3, 3), dtype=np.uint8))
    row = Trajectory(tmp_path / "heldout", "episode", "assemble", 5, (pixels,), (5, 2, 3, 3),
                     ("top",), ("B", "C"), (), (0, 0, 1, 1, 2), (.2,) * 5,
                     ("Reach", "Insert", "Finish"), None)
    train, heldout = inventory([]), inventory([row])
    monkeypatch.setattr("grana.data.cache.CachedTrajectories",
                        lambda caches, **kwargs: heldout if caches == ["/heldout"] else train)
    config = deepcopy(DEFAULTS)
    config["model"].update(name_or_path="tiny", revision="fixed", lora_rank=2, lora_alpha=2)
    config["data"].update(caches=["/train"], views=["top"], max_pixels=76800)
    config["roles"] = {"A": 0, "B": 16, "C": 24, "D": 24}
    config["training"].update(stage="posttrain", max_steps=500, save_steps=250,
                               output_dir=str(source), cpus_per_worker=1)
    config["evaluation"].update(enabled=True, caches=["/heldout"], mode="dense", max_trajectories_per_task=None)
    config["evaluation"]["deploy_dense"].update(enabled=True, frame_stride=1, max_trajectories_per_task=None)
    # Historical configs predate all new optimization fields.
    config["evaluation"].pop("batch_size")
    config["evaluation"].pop("batched_vision_sdpa")
    config["training"].pop("batched_vision_sdpa")
    historical_pairs = getattr(request, "param", None)
    if historical_pairs is not None:
        config["evaluation"].pop("stride_pairs")
        config["data"]["stride_pairs"] = historical_pairs
    write(source / "resolved_config.json", config)
    model = TinyModel()
    checkpoint = source / "checkpoint-000250"
    save_weights(model, checkpoint, config, 250)
    metric = {"optimizer_steps": 250, "checkpoint_path": str(checkpoint), "elapsed_seconds": 7100.5}
    write(source / "training_metrics.jsonl", metric)
    plan_settings = {**config["evaluation"]}
    if historical_pairs is not None:
        plan_settings["stride_pairs"] = historical_pairs
    base = build_evaluation_plan(heldout, plan_settings, config["data"], training_dataset=train, roles=config["roles"])
    plan = build_dense_plan(base, frame_stride=1, max_trajectories_per_task=None)
    write(source / "evaluation_plan.json", plan.manifest())
    write(source / "wandb_run.json", {"mode": "online", "entity": "team", "project": "grana", "id": "old-run", "status": "failed"})
    # Old archived records need not have the newly introduced slot_losses.
    write(source / "evaluation-000010.json", {"records": [{"role": "B", "predictions": [.5] * 8}]})
    yield SimpleNamespace(source=source, config=config, checkpoint=checkpoint, model=model, plan=plan, base=base,
                          output=tmp_path / "new-evaluation")
    torch.set_num_threads(previous_threads)


def prepare(saved, **kwargs):
    return prepare_checkpoint_evaluation(saved.source, 250, batch_size=kwargs.get("batch_size", 64),
                                         batched_vision_sdpa=kwargs.get("vision", True))


def snapshot(directory):
    return {str(path.relative_to(directory)): path.read_bytes() for path in directory.rglob("*") if path.is_file()}


@pytest.mark.parametrize("batch_size", [1, 32, 64])
def test_unfinished_source_retains_exact_windows_and_legacy_manifest(saved_run, batch_size):
    saved = saved_run
    before = snapshot(saved.source)
    assert "loss" not in json.loads((saved.checkpoint / "metadata.json").read_text())["config"]
    config, checkpoint, plan, provenance = prepare(saved, batch_size=batch_size)
    assert not (saved.source / "result.json").exists()
    assert config == saved.config and checkpoint == saved.checkpoint
    assert plan.manifest()["windows"] == saved.plan.manifest()["windows"]
    assert provenance["source_plan_sha256"] == saved.plan.manifest()["plan_sha256"]
    assert provenance["new_optimizer_updates"] == 0
    assert provenance["mode"] == "checkpoint_replay"
    assert provenance["training_elapsed_seconds"] == 7100.5
    if batch_size == 1:
        assert plan.manifest() == saved.plan.manifest()
    else:
        assert plan.manifest()["batch_size"] == batch_size
    assert snapshot(saved.source) == before


def test_checkpoint_replay_keeps_saved_task_routes_but_never_training_mapping(saved_run):
    from grana.training.evaluation import evaluate

    saved = saved_run
    config = deepcopy(saved.config)
    config["data"]["train_stage_augmentation"] = {
        "enabled": True, "contiguous_shift": {"enabled": True, "candidate_count": 10},
        "task_overrides": {"assemble": {"enabled": True}}}
    config["evaluation"]["mode"] = "sampled"
    config["evaluation"]["deploy_dense"]["enabled"] = False
    write(saved.source / "resolved_config.json", config)
    metadata = json.loads((saved.checkpoint / "metadata.json").read_text())
    metadata["config"]["data"] = config["data"]
    metadata["config"]["evaluation"] = config["evaluation"]
    write(saved.checkpoint / "metadata.json", metadata)
    plan = build_evaluation_plan(saved.base.dataset, config["evaluation"], config["data"],
                                 training_dataset=inventory([]), roles=config["roles"])
    write(saved.source / "evaluation_plan.json", plan.manifest())
    before = snapshot(saved.source)
    _, _, replay_plan, _ = prepare(saved, batch_size=1)
    assert {window.roles for window in replay_plan.windows} == {("B", "C")}
    assert replay_plan.manifest() == plan.manifest()
    seen = []

    def collate(processor, examples, pixels, device):
        seen.extend(examples)
        return {}

    result = evaluate(saved.model, None, replay_plan, device="cpu", max_pixels=76800, collate_fn=collate)
    assert all(example.stage_choices is None and example.num_stages == 3 for example in seen)
    assert result["roles"]["D"]["status"] == "unassessed"
    assert result["stage_augmentation"]["augmented_rows"] == 0
    assert snapshot(saved.source) == before
    # A separate explicitly configured job derives eligibility from its own
    # policy; loading these transferable weights does not permanently ban D.
    fresh = build_evaluation_plan(saved.base.dataset, saved.config["evaluation"], saved.config["data"],
                                  training_dataset=inventory([]), roles=config["roles"])
    assert {window.roles for window in fresh.windows} == {("B", "C", "D")}


@pytest.mark.parametrize("saved_run", [[[40, 4]], [[20, 2], [40, 4]]], indirect=True)
def test_legacy_checkpoint_replay_keeps_frozen_strides(saved_run):
    before = snapshot(saved_run.source)
    config, _, plan, provenance = prepare(saved_run, batch_size=1)
    assert "stride_pairs" not in config["evaluation"]
    assert plan.settings["stride_pairs"] == config["data"]["stride_pairs"]
    assert plan.manifest() == saved_run.plan.manifest()
    assert provenance["source_plan_sha256"] == saved_run.plan.manifest()["plan_sha256"]
    assert snapshot(saved_run.source) == before


@pytest.mark.parametrize("saved_run", [[[40, 4]]], indirect=True)
def test_legacy_checkpoint_evaluation_rejects_valid_plan_with_different_strides(saved_run):
    wrong_base = build_evaluation_plan(
        saved_run.base.dataset, {**saved_run.config["evaluation"], "stride_pairs": [[16, 4]]},
        saved_run.config["data"], training_dataset=inventory([]), roles=saved_run.config["roles"])
    wrong_plan = build_dense_plan(wrong_base, frame_stride=1, max_trajectories_per_task=None)
    # This plan is internally consistent, including its digest, but belongs to
    # different sampling settings than the saved config and checkpoint.
    write(saved_run.source / "evaluation_plan.json", wrong_plan.manifest())
    before = snapshot(saved_run.source)
    with pytest.raises(ValueError, match="differs from"):
        prepare(saved_run)
    assert snapshot(saved_run.source) == before


def test_explicit_eval_strides_must_match_frozen_plan(saved_run):
    config = deepcopy(saved_run.config)
    config["evaluation"]["stride_pairs"] = [[40, 4]]
    write(saved_run.source / "resolved_config.json", config)
    path = saved_run.checkpoint / "metadata.json"
    metadata = json.loads(path.read_text())
    metadata["config"]["evaluation"] = config["evaluation"]
    write(path, metadata)
    before = snapshot(saved_run.source)
    with pytest.raises(ValueError, match="differs from the saved plan"):
        prepare(saved_run)
    assert snapshot(saved_run.source) == before


@pytest.mark.parametrize("damage", ["step", "stage", "model", "config", "plan", "missing_receipt", "duplicate_receipt", "wrong_path", "elapsed", "weights"])
def test_reject_inconsistent_saved_provenance(saved_run, damage):
    saved = saved_run
    path = saved.checkpoint / "metadata.json"
    metadata = json.loads(path.read_text())
    if damage == "step":
        metadata["source_step"] = 249
    elif damage == "stage":
        metadata["source_stage"] = "pretrain"
    elif damage == "model":
        metadata["model"]["revision"] = "different"
    elif damage == "config":
        metadata["config"]["roles"]["C"] = 25
    elif damage == "plan":
        plan = saved.plan.manifest()
        plan["windows"][0]["frame_indices"][0] = 4
        write(saved.source / "evaluation_plan.json", plan)
    elif damage in ("missing_receipt", "duplicate_receipt", "wrong_path", "elapsed"):
        metric_path = saved.source / "training_metrics.jsonl"
        metric = json.loads(metric_path.read_text())
        if damage == "missing_receipt":
            metric_path.write_text("")
        elif damage == "duplicate_receipt":
            metric_path.write_text(metric_path.read_text() * 2)
        else:
            metric["checkpoint_path" if damage == "wrong_path" else "elapsed_seconds"] = "/different/checkpoint" if damage == "wrong_path" else float("nan")
            write(metric_path, metric)
    else:
        (saved.checkpoint / "weights.safetensors").unlink()
    write(path, metadata)
    before = snapshot(saved.source)
    with pytest.raises((ValueError, FileNotFoundError)):
        prepare(saved)
    assert snapshot(saved.source) == before


def test_reject_legacy_saved_loss_drift_before_reconstructing_replay(saved_run, monkeypatch):
    metadata_path = saved_run.checkpoint / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["config"]["loss"] = deepcopy(saved_run.config["loss"])
    write(metadata_path, metadata)
    changed = deepcopy(saved_run.config)
    changed["loss"]["role_weights"]["C"] += 1
    write(saved_run.source / "resolved_config.json", changed)
    monkeypatch.setattr("grana.data.cache.CachedTrajectories",
                        lambda *a, **kw: pytest.fail("mismatched objective reached replay preparation"))
    before = snapshot(saved_run.source)
    with pytest.raises(ValueError, match="Checkpoint saved loss configuration differs"):
        prepare(saved_run)
    assert snapshot(saved_run.source) == before
    assert not saved_run.output.exists()


@pytest.mark.parametrize("step,batch,vision", [(True, 64, True), (0, 64, True), (501, 64, True), (250, True, True), (250, 0, True), (250, 64, 1)])
def test_reject_invalid_execution_arguments(saved_run, step, batch, vision):
    with pytest.raises(ValueError):
        prepare_checkpoint_evaluation(saved_run.source, step, batch_size=batch, batched_vision_sdpa=vision)


def arguments(saved, **overrides):
    return SimpleNamespace(**({"source_run": saved.source, "step": 250, "output": saved.output,
                              "batch_size": 32, "batched_vision_sdpa": True, "device": "cpu",
                              "logging_mode": "disabled", "dry_run": False} | overrides))


@pytest.mark.parametrize("saved_run", [None, [[40, 4]], [[20, 2], [40, 4]]], indirect=True)
def test_dry_run_never_builds_model_creates_output_or_changes_source(saved_run, monkeypatch, capsys):
    monkeypatch.setattr("grana.runtime.require_owned_environment", lambda: None)
    monkeypatch.setattr("grana.modeling.model.build_model", lambda *_: pytest.fail("dry run loaded a model"))
    before = snapshot(saved_run.source)
    run(arguments(saved_run, dry_run=True))
    assert json.loads(capsys.readouterr().out)["new_optimizer_updates"] == 0
    assert not saved_run.output.exists() and snapshot(saved_run.source) == before


@pytest.mark.parametrize("nested", [False, True])
def test_reject_output_inside_source(saved_run, monkeypatch, nested):
    monkeypatch.setattr("grana.runtime.require_owned_environment", lambda: None)
    output = saved_run.source / "nested" if nested else saved_run.source
    with pytest.raises(ValueError, match="outside the immutable source"):
        run(arguments(saved_run, output=output, dry_run=True))


def test_real_cpu_checkpoint_load_and_eval_create_only_new_artifacts(saved_run, monkeypatch):
    saved = saved_run
    target, builds, bundles = TinyModel(), [], []
    monkeypatch.setattr("grana.runtime.require_owned_environment", lambda: None)
    monkeypatch.setattr("grana.modeling.model.build_model", lambda cfg: (builds.append(deepcopy(cfg)) or target, None))
    monkeypatch.setattr("grana.modeling.vision_sdpa.configure_batched_vision_sdpa", lambda *a, **kw: None)
    monkeypatch.setattr("grana.training.evaluation.collate_examples", lambda processor, examples, max_pixels, device: {"examples": examples})
    monkeypatch.setattr("grana.training.deploy_dense.require_encoder", lambda: "fake-encoder")
    monkeypatch.setattr(torch.optim.AdamW, "step", lambda *a, **kw: pytest.fail("evaluation performed an optimizer update"))
    for key in ("WANDB_SERVICE", "_WANDB_SERVICE", "WANDB_RUN_ID"):
        monkeypatch.setenv(key, "old-training-binding")
    def render(result, plan, **kwargs):
        bundles.append((deepcopy(result), kwargs))
        return {"manifest": str(saved.output / "manifest.json"), "items": [{"image": "plot.png", "video": "replay.mp4"}]}
    monkeypatch.setattr("grana.training.deploy_dense.render_dense_evaluation", render)
    before = snapshot(saved.source)
    run(arguments(saved))
    report = json.loads((saved.output / "checkpoint_evaluation.json").read_text())
    result = json.loads((saved.output / "evaluation-000250.json").read_text())
    assert report["status"] == "completed" and report["new_optimizer_updates"] == 0
    assert report["provenance"]["source_run"] == "team/grana/old-run"
    assert report["checkpoint_load"]["adapter_tensor_count"] == 2
    assert all(row["outcome"] == "loaded" for row in report["checkpoint_load"]["heads"].values())
    assert result["model_calls"] == 15 and result["forward_calls"] == 3
    assert all(len(row["slot_losses"]) == 8 for row in result["records"])
    assert bundles[0][1]["provenance"]["mode"] == "checkpoint_replay"
    assert builds[0]["training"]["output_dir"] == str(saved.output)
    assert builds[0]["evaluation"]["batch_size"] == 32
    assert builds[0]["logging"]["job_type"] == "checkpoint-evaluation"
    assert tensor_digest(trainable_state(target)) == tensor_digest(trainable_state(saved.model))
    assert all(key not in os.environ for key in ("WANDB_SERVICE", "_WANDB_SERVICE", "WANDB_RUN_ID"))
    assert snapshot(saved.source) == before


def test_corrupted_weights_fail_before_forward_and_leave_failed_receipt(saved_run, monkeypatch):
    from safetensors.torch import load_file, save_file

    path = saved_run.checkpoint / "weights.safetensors"
    weights = load_file(str(path))
    weights[next(iter(weights))].add_(1)
    save_file(weights, str(path))
    monkeypatch.setattr("grana.runtime.require_owned_environment", lambda: None)
    monkeypatch.setattr("grana.modeling.model.build_model", lambda cfg: (TinyModel(), None))
    monkeypatch.setattr("grana.training.deploy_dense.require_encoder", lambda: "fake-encoder")
    monkeypatch.setattr("grana.training.evaluation.evaluate", lambda *a, **kw: pytest.fail("invalid checkpoint reached evaluation"))
    with pytest.raises(ValueError, match="tensor digest mismatch"):
        run(arguments(saved_run))
    report = json.loads((saved_run.output / "checkpoint_evaluation.json").read_text())
    assert report["status"] == "failed"
