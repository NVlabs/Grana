# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Phase supervision, resource scaling, composition and retained-run semantics."""
import json
from pathlib import Path

import pytest
import yaml

from grana.config import config_sources, load_config, supervision_budget
from grana.training.losses import window_role_counts

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("phase,physical,workers,expected", [
    *(("pretraining", 32, workers, {"A": 40, "B": 40, "C": 48, "D": 0}) for workers in (1, 2)),
    *(("posttraining", 8, workers, {"A": 0, "B": 80, "C": 24, "D": 24}) for workers in (1, 2, 4, 8)),
])
def test_global_supervision_is_preserved_when_worker_count_changes(phase, physical, workers, expected):
    cfg = load_config(ROOT / f"configs/{phase}/custom.yaml", [f"training.num_workers={workers}"])
    budget = supervision_budget(cfg)
    assert budget["role_rows_per_update_global"] == expected
    assert budget["global_batch_size"] == 128
    assert budget["physical_batch_size_per_worker"] == physical
    assert budget["gradient_accumulation_steps"] == 128 // (physical * workers)
    runtime = window_role_counts(cfg["roles"], cfg["training"]["gradient_accumulation_steps"])
    assert {role: count * workers for role, count in runtime.items()} == expected
    assert budget["stage_label_slots_per_update"] == expected["C"] * 8


def test_phase_selection_owns_defaults_and_explicit_overrides_win(tmp_path):
    path = tmp_path / "recipe.yaml"
    path.write_text("data:\n  caches: [/train]\n")
    pre = load_config(path)
    post = load_config(path, ["training.stage=posttrain", "training.init_checkpoint=/same-size/pretrain"])
    assert pre["roles"] == {"A": 10, "B": 10, "C": 12, "D": 0}
    assert post["roles"] == {"A": 0, "B": 10, "C": 3, "D": 3}
    assert pre["training"]["max_steps"] == 5000 and post["training"]["max_steps"] == 200
    assert pre["loss"]["role_weights"]["C"] == .25 and post["loss"]["role_weights"]["C"] == 10
    assert pre["training"]["learning_rate"] == post["training"]["learning_rate"] == 2e-5
    explicit = load_config(path, ["training={stage: posttrain, init_checkpoint: /pre, learning_rate: 1.0e-5}",
                                 "roles={A: 0, B: 2, C: 1, D: 1}", "loss.role_weights.C=4"])
    assert explicit["roles"]["C"] == 1 and explicit["training"]["learning_rate"] == 1e-5
    assert explicit["loss"]["role_weights"]["C"] == 4


@pytest.mark.parametrize("setting", [
    "training.global_batch_size=127", "training.global_batch_size=8", "training.global_batch_size=true",
    "training.global_batch_size=128.0", "training.num_workers=3", "training.num_workers=4",
    "training.num_workers=8", "training.num_workers=16",
    "training.gradient_accumulation_steps=2",
])
def test_unrepresentable_or_conflicting_budget_fails_before_launch(setting):
    with pytest.raises(ValueError, match="global_batch_size"):
        load_config(ROOT / "configs/pretraining/custom.yaml", [setting])


def test_explicit_small_count_mode_and_old_resolved_configs_keep_their_budget(tmp_path):
    cfg = load_config(ROOT / "configs/pretraining/custom.yaml", ["training.global_batch_size=null",
                      "training.gradient_accumulation_steps=2"])
    assert supervision_budget(cfg)["global_batch_size"] == 64
    historical = {
        "data": {"caches": ["/train"]}, "roles": {"A": 2, "B": 2, "C": 1, "D": 1},
        "loss": {"role_weights": {"A": 1., "B": 1., "C": 2.5, "D": 1.}},
        "training": {"stage": "posttrain", "init_checkpoint": "/old/pretrain/checkpoint-000400",
                     "max_steps": 200, "num_workers": 1, "gradient_accumulation_steps": 2,
                     "learning_rate": 1e-6, "warmup_steps": 10},
    }
    path = tmp_path / "historical.json"
    path.write_text(json.dumps(historical))
    loaded = load_config(path)
    assert loaded["training"]["global_batch_size"] is None
    assert loaded["training"]["learning_rate"] == 1e-6
    assert loaded["roles"] == historical["roles"]
    assert supervision_budget(loaded)["role_rows_per_update_global"] == historical["roles"]


def test_inheritance_keeps_manifest_owner_and_model_overlays_independent(tmp_path):
    phase, leaf = tmp_path / "phase", tmp_path / "models"
    phase.mkdir(); leaf.mkdir()
    (phase / "data.yaml").write_text("caches:\n- path: payload/train\n  roles: [B, C]\n")
    (phase / "base.yaml").write_text("training: {stage: posttrain, init_checkpoint: /pre}\ndata: {manifest: data.yaml}\n")
    (leaf / "model.yaml").write_text("model: {name_or_path: Qwen/Qwen3.5-4B}\n")
    path = leaf / "task.yaml"
    path.write_text("extends: [../phase/base.yaml, model.yaml]\nloss: {role_weights: {C: 6}}\n")
    cfg = load_config(path)
    assert cfg["data"]["caches"][0]["path"] == str(phase / "payload/train")
    assert cfg["data"]["caches"][0]["roles"] == ["B", "C"]
    assert cfg["model"]["name_or_path"] == "Qwen/Qwen3.5-4B"
    assert cfg["roles"]["A"] == 0 and cfg["loss"]["role_weights"]["C"] == 6
    assert set(config_sources(path)) == {path, phase / "base.yaml", leaf / "model.yaml"}
    (leaf / "other.yaml").write_text("caches: [payload/other]\n")
    changed = load_config(path, ["data.manifest=other.yaml"])
    assert changed["data"]["caches"] == [str(leaf / "payload/other")]


def test_inherited_unknown_keys_and_cycles_are_rejected(tmp_path):
    parent, child = tmp_path / "parent.yaml", tmp_path / "child.yaml"
    child.write_text("extends: parent.yaml\ndata: {caches: [/train]}\n")
    parent.write_text("model: {typo: true}\n")
    with pytest.raises(ValueError, match="model.typo"):
        load_config(child)
    parent.write_text("extends: child.yaml\n")
    with pytest.raises(ValueError, match="inheritance cycle"):
        load_config(child)


@pytest.mark.parametrize("size", ["0.8B", "2B", "4B", "9B", "27B"])
@pytest.mark.parametrize("phase", ["pretraining", "posttraining"])
def test_model_overlays_compose_with_user_owned_phase_recipes(size, phase, tmp_path):
    recipe = tmp_path / "project.yaml"
    recipe.write_text(yaml.safe_dump({
        "extends": [str(ROOT / f"configs/{phase}/default.yaml"),
                    str(ROOT / f"configs/models/qwen35/{size.lower()}.yaml")],
        "data": {"caches": [str(tmp_path / "train")]},
        "training": {"init_checkpoint": None if phase == "pretraining" else str(tmp_path / "checkpoint")},
        "evaluation": {"enabled": True, "caches": [str(tmp_path / "eval")],
                       "deploy_dense": {"enabled": True}}}))
    cfg = load_config(recipe)
    pretrain = phase == "pretraining"
    assert cfg["model"]["name_or_path"] == f"Qwen/Qwen3.5-{size}"
    assert cfg["model"]["delta_rule_backend"] == ("torch" if size == "27B" else "auto")
    assert cfg["model"]["gradient_checkpointing"] is True
    assert cfg["roles"]["D"] == (0 if pretrain else 3)
    assert cfg["roles"]["A"] == (10 if pretrain else 0)
    assert cfg["training"]["max_steps"] == (5000 if pretrain else 200)
    assert cfg["training"]["learning_rate"] == 2e-5
    assert cfg["training"]["profile_steps"] == 0
    assert cfg["loss"]["stage"] == {"cdf_weight": 1.0}
    assert cfg["loss"]["progress"] == {"cdf_weight": 1.0, "ceiling_weight": 0.0}
    assert cfg["evaluation"]["deploy_dense"]["enabled"]
    assert supervision_budget(cfg)["global_batch_size"] == 128


def test_posttraining_accepts_multiple_task_caches_without_enabling_A(tmp_path):
    path = tmp_path / "multiple.yaml"
    path.write_text(yaml.safe_dump({"training": {"stage": "posttrain", "init_checkpoint": "/pre"},
                                   "data": {"caches": [{"path": "/task1", "roles": ["B", "C"]},
                                                       {"path": "/task2", "roles": ["B", "C"]}]}}))
    cfg = load_config(path)
    assert len(cfg["data"]["caches"]) == 2 and cfg["roles"]["A"] == 0


@pytest.mark.parametrize("roles,physical,accumulation", [
    ({"A": 0, "B": 32, "C": 48, "D": 48}, 64, 2),
    ({"A": 0, "B": 16, "C": 24, "D": 24}, 32, 4),
    ({"A": 0, "B": 8, "C": 12, "D": 12}, 16, 8),
])
def test_raw_backbone_posttraining_preserves_explicit_supervision(tmp_path, roles, physical, accumulation):
    path = tmp_path / "raw.yaml"
    path.write_text(yaml.safe_dump({"training": {"stage": "posttrain", "init_checkpoint": None,
                                               "global_batch_size": 128, "max_steps": 500},
                                   "model": {"name_or_path": "Qwen/Qwen3.5-4B"},
                                   "data": {"caches": ["/unused/train"]}, "roles": roles}))
    cfg = load_config(path)
    budget = supervision_budget(cfg)
    expected = {"A": 0, "B": 32, "C": 48, "D": 48}
    assert cfg["training"]["init_checkpoint"] is None
    assert cfg["loss"]["role_weights"] == {"A": 1., "B": 1., "C": 10., "D": 1.}
    assert cfg["loss"]["progress"] == {"cdf_weight": 1., "ceiling_weight": 0.}
    assert budget["role_rows_per_update_global"] == expected
    assert budget["physical_batch_size_per_worker"] == physical
    assert budget["gradient_accumulation_steps"] == accumulation
    assert budget["stage_label_slots_per_update"] == 384
    assert budget["role_rows_over_run_global"] == {role: count * 500 for role, count in expected.items()}
    assert window_role_counts(cfg["roles"], accumulation) == expected
