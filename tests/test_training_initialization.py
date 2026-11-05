# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Initial receipts must cover frozen base weights and checkpointed markers."""
import torch
from torch import nn

from grana.modeling.checkpoint import save_weights, tensor_digest
from grana.training.engine import initialization_receipt


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(12, 4)
        self.output = nn.Linear(4, 12, bias=False)
        self.requires_grad_(False)
        self.adapter = nn.Linear(4, 4)

    def get_input_embeddings(self):
        return self.embedding

    def get_output_embeddings(self):
        return self.output


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = Backbone()
        self.progress_head = nn.Linear(4, 10)
        self.temporal_head = nn.Linear(4, 4)
        self.token_ids = {"reward": 8, "sim": 9, "prog": 10, "stage": 11}
        self.model_spec = {"name_or_path": "tiny", "token_ids": self.token_ids,
                           "lora_rank": 2, "lora_alpha": 2, "revision": "pinned"}


def config():
    return {"training": {"stage": "posttrain", "init_checkpoint": None, "seed": 42},
            "data": {"seed": 42}}


def test_initial_receipt_covers_exact_checkpoint_state_without_mutation(tmp_path):
    model = TinyModel()
    before = tensor_digest(model.state_dict())
    rng = torch.random.get_rng_state().clone()
    receipt = initialization_receipt(model, config())
    checkpoint = save_weights(model, tmp_path / "checkpoint", config(), 0)
    assert receipt["initialization"] == "pretrained_backbone"
    assert receipt["model"]["revision"] == "pinned"
    assert receipt["all_weights_sha256"] == before == tensor_digest(model.state_dict())
    assert receipt["transferable_weights_sha256"] == checkpoint["weights_sha256"]
    assert receipt["input_embedding_requires_grad"] is False
    assert receipt["output_embedding_requires_grad"] is False
    assert torch.equal(rng, torch.random.get_rng_state())


def test_receipt_distinguishes_frozen_base_marker_and_adaptation_changes():
    model = TinyModel()
    first = initialization_receipt(model, config())
    with torch.no_grad():
        model.backbone.embedding.weight[0, 0].add_(1)
    second = initialization_receipt(model, config())
    assert first["all_weights_sha256"] != second["all_weights_sha256"]
    assert first["trainable_weights_sha256"] == second["trainable_weights_sha256"]
    assert first["added_embeddings_sha256"] == second["added_embeddings_sha256"]
    with torch.no_grad():
        model.backbone.output.weight[11, 0].add_(1)
    third = initialization_receipt(model, config())
    assert second["added_embeddings_sha256"] != third["added_embeddings_sha256"]
    assert second["transferable_weights_sha256"] != third["transferable_weights_sha256"]
    with torch.no_grad():
        model.progress_head.bias[0].add_(1)
    fourth = initialization_receipt(model, config(), {"source_step": 7})
    assert fourth["initialization"] == "weights_only"
    assert fourth["report"] == {"source_step": 7}
    assert third["trainable_weights_sha256"] != fourth["trainable_weights_sha256"]
    assert third["added_embeddings_sha256"] == fourth["added_embeddings_sha256"]
