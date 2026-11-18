# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import json

import pytest
import torch
from torch import nn

from grana.modeling.checkpoint import load_weights, plan_head_load, save_weights, tensor_digest


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(12, 4)
        self.output = nn.Linear(4, 12, bias=False)
        self.embedding.weight.requires_grad_(False)
        self.output.weight.requires_grad_(False)
        self.adapter = nn.Linear(4, 4)

    def get_input_embeddings(self):
        return self.embedding

    def get_output_embeddings(self):
        return self.output


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = Backbone()
        self.progress_head = nn.Sequential(nn.Linear(4, 3), nn.LayerNorm(3))
        self.temporal_head = nn.Linear(4, 4)
        self.token_ids = {"reward": 8, "sim": 9, "prog": 10, "stage": 11}
        self.model_spec = {"name_or_path": "tiny", "token_ids": self.token_ids, "lora_rank": 2, "lora_alpha": 2}


@pytest.mark.parametrize("head", ["progress_head", "temporal_head"])
@pytest.mark.parametrize("damage", ["absent", "incomplete", "shape"])
def test_whole_head_fallback_is_explicit_and_never_partial(head, damage):
    model = TinyModel()
    state = copy.deepcopy(model.state_dict())
    keys = [key for key in state if key.startswith(head + ".")]
    if damage == "absent":
        for key in keys:
            del state[key]
    elif damage == "incomplete":
        del state[keys[0]]
    else:
        state[keys[0]] = torch.zeros(1)
    before = tensor_digest(model.state_dict())
    with pytest.raises(ValueError):
        plan_head_load(model, state)
    selected, report = plan_head_load(model, state, [head])
    assert report[head]["outcome"] == "fresh"
    assert not any(key.startswith(head + ".") for key in selected)
    assert tensor_digest(model.state_dict()) == before


def test_complete_heads_load_even_if_fresh_allowed():
    model = TinyModel()
    selected, report = plan_head_load(model, model.state_dict(), ["progress_head", "temporal_head"])
    assert all(item["outcome"] == "loaded" for item in report.values())
    assert set(selected) == {name for name in model.state_dict() if name.startswith(("progress_head.", "temporal_head."))}


def test_sampling_provenance_changes_config_without_changing_transferable_tensors(tmp_path):
    source = TinyModel()
    data = {"train_frame_sampling": {
        "frame_selection_profile_weights": {"regular_ctsp": .5, "irregular_stratified": .5},
        "query_order_profile_weights": {"forward": .5, "tail_rewind": .25, "backward": .25}}}
    baseline = save_weights(source, tmp_path / "baseline", {"training": {"stage": "posttrain"}}, 7)
    sampled = save_weights(source, tmp_path / "sampled", {"training": {"stage": "posttrain"}, "data": data}, 7)
    assert sampled["weights_sha256"] == baseline["weights_sha256"]
    metadata = json.loads((tmp_path / "sampled" / "metadata.json").read_text())
    assert metadata["config"]["data"] == data
    assert load_weights(TinyModel(), sampled["path"])["weights_sha256"] == baseline["weights_sha256"]


def test_save_reload_cross_objective_is_weights_only(tmp_path):
    source = TinyModel()
    cfg = {"training": {"stage": "pretrain"}, "loss": {"stage": {"profile": "arbitrary-source-objective"}}}
    saved = save_weights(source, tmp_path / "checkpoint", cfg, 7)
    target = TinyModel()
    report = load_weights(target, saved["path"])
    assert report["initialization"] == "weights_only"
    assert report["source_step"] == 7
    for name, param in target.named_parameters():
        if param.requires_grad:
            torch.testing.assert_close(param, dict(source.named_parameters())[name])
    ids = torch.tensor(list(source.token_ids.values()))
    torch.testing.assert_close(target.backbone.embedding.weight[ids], source.backbone.embedding.weight[ids])
    assert set(p.name for p in (tmp_path / "checkpoint").iterdir()) == {"weights.safetensors", "metadata.json"}
    metadata = json.loads((tmp_path / "checkpoint" / "metadata.json").read_text())
    assert "loss" not in metadata["config"]
    assert cfg["loss"]["stage"]["profile"] == "arbitrary-source-objective"
    metadata["source_stage"] = "different-objective"
    # A source artifact may contain anything (or nothing) about its objective;
    # none of that is consulted when transferring its weights.
    metadata["config"] = {"loss": {"stage": {"profile": "unknown-to-this-code"}}}
    (tmp_path / "checkpoint" / "metadata.json").write_text(json.dumps(metadata))
    assert load_weights(TinyModel(), saved["path"])["heads"]["progress_head"]["outcome"] == "loaded"


@pytest.mark.parametrize("source_backend", [None, "auto", "torch"])
def test_legacy_training_backend_and_loss_do_not_constrain_weight_transfer(tmp_path, source_backend):
    source, target = TinyModel(), TinyModel()
    config = {"training": {"stage": "pretrain"}, "model": {"gradient_checkpointing": True}}
    if source_backend is not None:
        config["model"]["delta_rule_backend"] = source_backend
        source.model_spec["delta_rule_backend"] = {"requested": source_backend, "resolved": "torch"}
    target.model_spec["delta_rule_backend"] = {"requested": "auto", "resolved": "fla"}
    path = tmp_path / "checkpoint"
    save_weights(source, path, config, 400)
    metadata_path = path / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    # Older pretrain199 exports kept training objectives in the metadata. They
    # remain weights-only inputs even when deployment uses a different kernel.
    metadata["config"]["loss"] = {"stage": {"profile": "retired-objective"}}
    metadata_path.write_text(json.dumps(metadata))
    original = metadata_path.read_bytes()

    report = load_weights(target, path)

    assert report["adapter_tensor_count"] == 2
    assert all(head["outcome"] == "loaded" for head in report["heads"].values())
    for name, parameter in target.named_parameters():
        if parameter.requires_grad:
            torch.testing.assert_close(parameter, dict(source.named_parameters())[name], rtol=0, atol=0)
    assert metadata_path.read_bytes() == original


@pytest.mark.parametrize("max_pixels", [None, 1024, 76800])
def test_new_checkpoint_records_explicit_preprocessing_semantics(tmp_path, max_pixels):
    path = tmp_path / "checkpoint"
    config = {"training": {"stage": "pretrain"}, "data": {"max_pixels": max_pixels}}
    save_weights(TinyModel(), path, config, 1)
    metadata = json.loads((path / "metadata.json").read_text())
    assert metadata["format"] == "grana-weights-v1"
    assert metadata["image_preprocessing"] == {
        "version": 1, "policy": "processor_default" if max_pixels is None else "pixel_budget",
        "max_pixels": max_pixels}
    assert metadata["config"] == config
    # Loading weights does not inherit or require the source resize policy.
    metadata["image_preprocessing"] = {"version": "not-used-by-weights-only-loader"}
    (path / "metadata.json").write_text(json.dumps(metadata))
    assert load_weights(TinyModel(), path)["initialization"] == "weights_only"


@pytest.mark.parametrize("max_pixels", [True, 0, 1023, 76800.0])
def test_invalid_preprocessing_does_not_create_checkpoint(tmp_path, max_pixels):
    path = tmp_path / "checkpoint"
    with pytest.raises(ValueError, match="max_pixels"):
        save_weights(TinyModel(), path, {"training": {"stage": "pretrain"},
                                       "data": {"max_pixels": max_pixels}}, 1)
    assert not path.exists()


def test_gpu_resize_checkpoint_records_backend_for_inference(tmp_path):
    path = tmp_path / "checkpoint"
    config = {"training": {"stage": "pretrain"}, "data": {"max_pixels": 76800, "gpu_image_resize": True}}
    save_weights(TinyModel(), path, config, 1)
    metadata = json.loads((path / "metadata.json").read_text())
    assert metadata["image_preprocessing"] == {
        "version": 2, "policy": "pixel_budget", "max_pixels": 76800, "resize_backend": "torchvision_cuda"}
    assert metadata["config"]["data"]["gpu_image_resize"] is True
    assert load_weights(TinyModel(), path)["initialization"] == "weights_only"


def test_invalid_resize_policy_does_not_create_checkpoint(tmp_path):
    path = tmp_path / "checkpoint"
    with pytest.raises(ValueError, match="gpu_image_resize"):
        save_weights(TinyModel(), path, {"training": {"stage": "pretrain"},
                                        "data": {"gpu_image_resize": "true"}}, 1)
    assert not path.exists()


def test_invalid_adapter_plan_does_not_mutate_heads(tmp_path):
    from safetensors.torch import load_file, save_file
    source, target = TinyModel(), TinyModel()
    path = tmp_path / "checkpoint"
    save_weights(source, path, {"training": {"stage": "pretrain"}}, 1)
    state = load_file(str(path / "weights.safetensors"))
    del state["backbone.adapter.weight"]
    save_file(state, str(path / "weights.safetensors"))
    metadata = json.loads((path / "metadata.json").read_text())
    metadata["weights_sha256"] = tensor_digest(state)
    (path / "metadata.json").write_text(json.dumps(metadata))
    before = tensor_digest(target.state_dict())
    with pytest.raises(ValueError, match="adapter"):
        load_weights(target, path)
    assert before == tensor_digest(target.state_dict())


def test_different_lora_scaling_fails_before_any_tensor_mutation(tmp_path):
    source, target = TinyModel(), TinyModel()
    path = tmp_path / "checkpoint"
    save_weights(source, path, {"training": {"stage": "pretrain"}}, 1)
    target.model_spec["lora_alpha"] = 4
    before = tensor_digest(target.state_dict())
    with pytest.raises(ValueError, match="lora_alpha"):
        load_weights(target, path)
    assert before == tensor_digest(target.state_dict())


def test_legacy_off_checkpoint_metadata_remains_weights_only(tmp_path):
    source = TinyModel()
    cfg = {"training": {"stage": "pretrain"}, "data": {
        "train_stage_augmentation": {"enabled": False, "contiguous_shift": {"enabled": True},
                                     "task_overrides": {"old-task": {"enabled": True}}}}}
    saved = save_weights(source, tmp_path / "historical-off", cfg, 1000)
    metadata_path = tmp_path / "historical-off" / "metadata.json"
    before = metadata_path.read_bytes()
    target = TinyModel()
    report = load_weights(target, saved["path"])
    assert report["weights_sha256"] == saved["weights_sha256"]
    assert report["source_step"] == 1000
    assert all(head["outcome"] == "loaded" for head in report["heads"].values())
    for name, param in target.named_parameters():
        if param.requires_grad:
            torch.testing.assert_close(param, dict(source.named_parameters())[name])
    assert metadata_path.read_bytes() == before
