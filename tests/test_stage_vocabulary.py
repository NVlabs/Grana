# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native letter projection respects each task's ordered stage vocabulary."""
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from grana.modeling.model import GranaModel, SPECIAL_TOKENS
from grana.training.losses import FRAME_WEIGHTS, role_loss_components, stage_loss_components


class Tokenizer:
    def convert_tokens_to_ids(self, token):
        return SPECIAL_TOKENS.index(token)

    def encode(self, letter, add_special_tokens=False):
        return [10 + ord(letter) - ord("A")]


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(text_config=SimpleNamespace(hidden_size=8))
        self.embeddings = nn.Embedding(48, 8)
        self.lm_head = nn.Linear(8, 48)
        self.calls = 0

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, input_ids, **kwargs):
        assert kwargs == {"output_hidden_states": True, "logits_to_keep": 1,
                          "use_cache": False, "return_dict": True}
        self.calls += 1
        return SimpleNamespace(hidden_states=(self.embeddings(input_ids),))


@pytest.fixture
def model():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(128)
        result = GranaModel(Backbone(), Tokenizer())
        # Keep the production native-stage projection; progress-head details
        # have separate qualification tests and are irrelevant to this mask.
        result.temporal_head = nn.Identity()
        result.progress_head = nn.Sequential(nn.Linear(8, 10))
    return result


def inputs(roles):
    return {"input_ids": torch.tensor([[3 if role == "C" else 2] * 8 for role in roles])}


def test_mixed_task_stage_counts_use_each_rows_native_prefix(model):
    roles = ["A", "C", "B", "C", "D", "C"]
    counts = [0, 10, 3, 3, 8, 8]
    batch = inputs(roles)
    output = model(batch, roles, num_stages=counts)
    assert output["stage_indices"] == [1, 3, 5]
    assert output["progress_indices"] == [0, 2, 4]
    assert output["progress_logits"].shape == (3, 8, 10)
    assert output["stage_logits"].shape == (3, 8, 26)
    raw_hidden = model.backbone.embeddings(batch["input_ids"])[output["stage_indices"]]
    raw_logits = F.linear(raw_hidden, model.backbone.lm_head.weight[model.answer_ids],
                          model.backbone.lm_head.bias[model.answer_ids])
    for row, count in enumerate((10, 3, 8)):
        torch.testing.assert_close(output["stage_logits"][row, :, :count], raw_logits[row, :, :count])
        assert torch.isneginf(output["stage_logits"][row, :, count:]).all()
        torch.testing.assert_close(output["stage_logits"][row].softmax(-1).sum(-1), torch.ones(8))
    examples = [SimpleNamespace(role=role, num_stages=count, targets=(.5,) * 8)
                for role, count in zip(roles, counts, strict=True)]
    examples[1] = SimpleNamespace(role="C", num_stages=10, stage_ids=(4,) * 8,
                                  targets=(4,) * 8)
    examples[3].targets = (1,) * 8
    examples[5].targets = (7,) * 8
    parts = role_loss_components(output, examples)
    sum(value["total"] for value in parts.values()).backward()
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters()
               if parameter.grad is not None)
    native_grad = model.backbone.lm_head.weight.grad[model.answer_ids]
    assert (native_grad[:10].abs().sum(-1) > 0).all()
    assert torch.count_nonzero(native_grad[10:]) == 0


@pytest.mark.parametrize("count", [1, 3, 8, 10, 26])
def test_native_answer_projection_handles_full_capacity_and_padding(model, count):
    output = model(inputs(["C"]), ["C"], [count])
    logits = output["stage_logits"]
    assert torch.isfinite(logits[..., :count]).all()
    assert torch.isneginf(logits[..., count:]).all()
    parts = stage_loss_components(logits, torch.full((1, 8), count - 1, dtype=torch.long))
    assert torch.isfinite(parts["total"]).all()
    parts["total"].sum().backward()
    assert torch.isfinite(model.backbone.lm_head.weight.grad).all()
    if count == 1:
        assert parts["total"].sum() == 0
        assert torch.count_nonzero(model.backbone.lm_head.weight.grad) == 0


@pytest.mark.parametrize("count", [0, 27, True, False, 10.0, "10", None])
def test_invalid_c_answer_counts_fail_before_backbone(model, count):
    with pytest.raises(ValueError, match="Stage vocabulary"):
        model(inputs(["C"]), ["C"], [count])
    assert model.backbone.calls == 0


@pytest.mark.parametrize("roles,counts", [(["C"], []), (["C"], [10, 3]), (["C", "B"], [10, 3])])
def test_answer_counts_and_roles_must_align_with_batch_rows(model, roles, counts):
    with pytest.raises(ValueError, match="align with input rows"):
        model(inputs(["C"]), roles, counts)
    assert model.backbone.calls == 0
