# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
import torch
from torch import nn

from grana.modeling.checkpoint import save_weights, tensor_digest
from grana.modeling.checkpoint_verification import compare_checkpoint_predictions, verify_checkpoint_reload


def test_bf16_prediction_tolerance_is_explicit_and_bounded():
    before = {"progress_logits": torch.tensor([[[0.0, 0.125]]], dtype=torch.bfloat16),
              "stage_logits": torch.tensor([[[1.0, -torch.inf]]])}
    after = {key: value.clone() for key, value in before.items()}
    after["progress_logits"][0, 0, 0] = 0.025390625
    with pytest.raises(AssertionError):
        compare_checkpoint_predictions(before, after)
    report = compare_checkpoint_predictions(before, after, rtol=0.016, atol=0.05)
    assert report["prediction_max_abs_delta"] == 0.025390625
    assert report["rtol"] == 0.016 and report["atol"] == 0.05
    after["progress_logits"][0, 0, 0] = 0.5
    with pytest.raises(AssertionError):
        compare_checkpoint_predictions(before, after, rtol=0.016, atol=0.05)


@pytest.mark.parametrize("key,value", [("progress_logits", float("nan")),
                                      ("progress_logits", -float("inf")),
                                      ("stage_logits", float("inf"))])
def test_prediction_tolerance_never_accepts_invalid_logits(key, value):
    invalid = {key: torch.tensor([[[0.0, value]]])}
    with pytest.raises(FloatingPointError):
        compare_checkpoint_predictions(invalid, invalid, rtol=0.016, atol=0.05)


def test_prediction_comparison_rejects_changed_stage_padding():
    with pytest.raises(AssertionError):
        compare_checkpoint_predictions({"stage_logits": torch.tensor([[[1.0, -torch.inf]]])},
                                       {"stage_logits": torch.tensor([[[1.0, 0.0]]])},
                                       rtol=0.016, atol=0.05)


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(12, 4).requires_grad_(False)
        self.output = nn.Linear(4, 12, bias=False).requires_grad_(False)
        self.adapter = nn.Linear(4, 4)

    def get_input_embeddings(self):
        return self.embedding

    def get_output_embeddings(self):
        return self.output


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = Backbone()
        self.progress_head = nn.Linear(4, 10)
        self.temporal_head = nn.Linear(4, 4)
        self.token_ids = {"reward": 8, "sim": 9, "prog": 10, "stage": 11}
        self.model_spec = {"token_ids": self.token_ids, "lora_rank": 2, "lora_alpha": 2}
        self.calls = []
        self.invalid = None

    def forward(self, inputs, roles, counts):
        assert not self.training and not torch.is_grad_enabled()
        self.calls.append((inputs, tuple(roles), tuple(counts)))
        hidden = self.backbone.adapter(self.backbone.embedding(inputs["input_ids"]))
        progress = [i for i, role in enumerate(roles) if role != "C"]
        stage = [i for i, role in enumerate(roles) if role == "C"]
        p = self.progress_head(self.temporal_head(hidden[progress])) if progress else None
        s = self.backbone.output(hidden[stage]) if stage else None
        if s is not None:
            mask = torch.arange(12)[None, None, :] >= torch.tensor([counts[i] for i in stage])[:, None, None]
            s = s.masked_fill(mask, -torch.inf)
        if self.invalid is not None:
            value = p if p is not None else s
            value[0, 0, 0] = self.invalid
        return {"progress_logits": p, "stage_logits": s}


@pytest.fixture
def saved(tmp_path):
    model = Model().eval()
    checkpoint = tmp_path / "checkpoint-001000"
    receipt = save_weights(model, checkpoint, {"training": {"stage": "pretrain"}}, 1000)
    return model, checkpoint, receipt


@pytest.mark.parametrize("roles,counts", [(["A"] * 32, [0] * 32), (["A", "C", "C"], [0, 1, 4])])
def test_real_reload_uses_same_batch_and_restores_complete_checkpoint(saved, roles, counts):
    model, checkpoint, receipt = saved
    inputs = {"input_ids": torch.full((len(roles), 8), 10, dtype=torch.long)}
    before = tensor_digest(model.state_dict())
    rng = torch.get_rng_state().clone()
    report = verify_checkpoint_reload(model, checkpoint, inputs, roles, counts)
    assert report["mode"] == "same_batch_reloaded_checkpoint_validation"
    assert report["source_optimizer_step"] == 1000
    assert report["physical_batch_size"] == len(roles)
    assert report["checkpoint_reload_equal"] and report["prediction_parity"]
    assert report["weights_sha256"] == report["restored_weights_sha256"] == receipt["weights_sha256"]
    assert report["perturbed_weights_sha256"] != receipt["weights_sha256"]
    assert report["prediction_max_abs_delta"] == 0
    assert report["forward_calls"] == len(model.calls) == 2
    assert report["new_optimizer_updates"] == 0
    assert all(call[0] is inputs and call[1:] == (tuple(roles), tuple(counts)) for call in model.calls)
    assert tensor_digest(model.state_dict()) == before
    assert torch.equal(torch.get_rng_state(), rng)
    assert not any(parameter.grad is not None for parameter in model.parameters())
    assert all(value["outcome"] == "loaded" for value in report["checkpoint_load"]["heads"].values())
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("target", ["adapter", "head", "added_embedding"])
def test_incomplete_initial_load_is_rejected_before_prediction_or_mutation(saved, target):
    model, checkpoint, _ = saved
    with torch.no_grad():
        value = {"adapter": model.backbone.adapter.weight, "head": model.temporal_head.weight,
                 "added_embedding": model.backbone.embedding.weight}[target]
        value[10 if target == "added_embedding" else 0, 0].add_(1)
    before = tensor_digest(model.state_dict())
    with pytest.raises(ValueError, match="every saved"):
        verify_checkpoint_reload(model, checkpoint, {"input_ids": torch.ones(2, 8, dtype=torch.long)}, ["A"] * 2, [0] * 2)
    assert not model.calls and tensor_digest(model.state_dict()) == before


def test_training_submodule_is_rejected(saved):
    model, checkpoint, _ = saved
    model.temporal_head.train()
    with pytest.raises(ValueError, match="eval mode"):
        verify_checkpoint_reload(model, checkpoint, {"input_ids": torch.ones(2, 8, dtype=torch.long)}, ["A"] * 2, [0] * 2)
    assert not model.calls


@pytest.mark.parametrize("role,invalid", [("A", float("nan")), ("A", -float("inf")), ("C", -float("inf"))])
def test_invalid_real_predictions_fail_without_mutation(saved, role, invalid):
    model, checkpoint, _ = saved
    model.invalid = invalid
    before = tensor_digest(model.state_dict())
    with pytest.raises(FloatingPointError):
        verify_checkpoint_reload(model, checkpoint, {"input_ids": torch.ones(2, 8, dtype=torch.long)}, [role] * 2, [2] * 2)
    assert len(model.calls) == 1 and tensor_digest(model.state_dict()) == before


def test_failed_load_restores_deliberate_mutation(saved, monkeypatch):
    model, checkpoint, _ = saved
    before = tensor_digest(model.state_dict())

    def fail(*args, **kwargs):
        raise ValueError("corrupt checkpoint")

    monkeypatch.setattr("grana.modeling.checkpoint_verification.load_weights", fail)
    with pytest.raises(ValueError, match="corrupt checkpoint"):
        verify_checkpoint_reload(model, checkpoint, {"input_ids": torch.ones(2, 8, dtype=torch.long)}, ["A"] * 2, [0] * 2)
    assert len(model.calls) == 1 and tensor_digest(model.state_dict()) == before


def test_loader_that_does_not_restore_is_rejected(saved, monkeypatch):
    model, checkpoint, receipt = saved
    before = tensor_digest(model.state_dict())
    monkeypatch.setattr("grana.modeling.checkpoint_verification.load_weights",
                        lambda *args, **kwargs: {"weights_sha256": receipt["weights_sha256"]})
    with pytest.raises(RuntimeError, match="restore all transferable"):
        verify_checkpoint_reload(model, checkpoint, {"input_ids": torch.ones(2, 8, dtype=torch.long)}, ["A"] * 2, [0] * 2)
    assert len(model.calls) == 1 and tensor_digest(model.state_dict()) == before


def test_same_batch_prediction_change_is_detected_after_successful_weight_restore(saved, monkeypatch):
    model, checkpoint, _ = saved
    before = tensor_digest(model.state_dict())
    original = model.forward

    def changed(inputs, roles, counts):
        result = original(inputs, roles, counts)
        if len(model.calls) == 2:
            result["progress_logits"] += 1
        return result

    monkeypatch.setattr(model, "forward", changed)
    with pytest.raises(AssertionError):
        verify_checkpoint_reload(model, checkpoint, {"input_ids": torch.ones(2, 8, dtype=torch.long)}, ["A"] * 2, [0] * 2)
    assert len(model.calls) == 2 and tensor_digest(model.state_dict()) == before
