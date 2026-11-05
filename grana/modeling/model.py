# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native Qwen3.5 with language LoRA, progress and temporal heads."""
from __future__ import annotations

import re
import string

import torch
from torch import nn
from torch.nn import functional as F

from .backends import configure_delta_rule_backend

SPECIAL_TOKENS = ("<|reward_token|>", "<|sim_token|>", "<|prog_token|>", "<|stage_token|>")
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
                "in_proj_a", "in_proj_b", "in_proj_qkv", "in_proj_z", "out_proj")


class TemporalHead(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.down = nn.Linear(hidden_size, 512)
        self.positions = nn.Embedding(8, 512)
        layer = nn.TransformerEncoderLayer(512, 8, 1024, dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
        self.up = nn.Linear(512, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(8, device=hidden.device)
        encoded = self.encoder(self.down(hidden) + self.positions(positions))
        return self.norm(hidden + self.up(encoded))


class GranaModel(nn.Module):
    def __init__(self, backbone: nn.Module, tokenizer):
        super().__init__()
        self.backbone = backbone
        config = backbone.config
        self.hidden_size = config.text_config.hidden_size
        self.token_ids = {token: tokenizer.convert_tokens_to_ids(token) for token in SPECIAL_TOKENS}
        answer_ids = [tokenizer.encode(letter, add_special_tokens=False) for letter in string.ascii_uppercase]
        if any(len(ids) != 1 for ids in answer_ids):
            raise ValueError("Native A–Z answer labels must each encode to one original token")
        self.register_buffer("answer_ids", torch.tensor([ids[0] for ids in answer_ids]), persistent=False)
        self.progress_head = nn.Sequential(
            nn.Linear(self.hidden_size, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(512, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, 10),
        )
        self.temporal_head = TemporalHead(self.hidden_size)

    def forward(self, inputs: dict[str, torch.Tensor], roles: list[str], stage_answer_counts: list[int]) -> dict:
        """Mask each C row by its answer count, which may exceed true stages.

        Canonical C uses K true stages; augmented C uses N full candidates.
        Counts for progress roles do not affect their ten-bin progress head.
        """
        if len(roles) != inputs["input_ids"].shape[0] or len(stage_answer_counts) != len(roles):
            raise ValueError("Roles and stage answer counts must align with input rows")
        if any(type(stage_answer_counts[i]) is not int or not 1 <= stage_answer_counts[i] <= 26
               for i, role in enumerate(roles) if role == "C"):
            raise ValueError("Stage answer vocabulary must contain 1–26 candidates")
        output = self.backbone(**inputs, output_hidden_states=True, logits_to_keep=1, use_cache=False, return_dict=True)
        hidden = output.hidden_states[-1]
        anchors = []
        for index, role in enumerate(roles):
            marker = "<|stage_token|>" if role == "C" else "<|prog_token|>"
            positions = (inputs["input_ids"][index] == self.token_ids[marker]).nonzero(as_tuple=True)[0]
            if positions.numel() != 8:
                raise ValueError(f"Role {role} requires exactly eight {marker} markers; found {positions.numel()}")
            anchors.append(hidden[index, positions])
        anchors = torch.stack(anchors)
        progress_indices = [i for i, role in enumerate(roles) if role != "C"]
        stage_indices = [i for i, role in enumerate(roles) if role == "C"]
        progress_logits = stage_logits = None
        if progress_indices:
            progress_hidden = anchors[progress_indices].to(self.progress_head[0].weight.dtype)
            progress_logits = self.progress_head(self.temporal_head(progress_hidden))
        if stage_indices:
            # Native original-vocabulary logits read RAW marker hidden states.
            lm_head = self.backbone.get_output_embeddings()
            weights = lm_head.weight.index_select(0, self.answer_ids)
            bias = None if lm_head.bias is None else lm_head.bias.index_select(0, self.answer_ids)
            stage_logits = F.linear(anchors[stage_indices], weights, bias).float()
            counts = torch.tensor([stage_answer_counts[i] for i in stage_indices], device=stage_logits.device)
            mask = torch.arange(26, device=stage_logits.device)[None, :] >= counts[:, None]
            stage_logits = stage_logits.masked_fill(mask[:, None, :], -torch.inf)
        return {"progress_logits": progress_logits, "progress_indices": progress_indices,
                "stage_logits": stage_logits, "stage_indices": stage_indices}


def build_model(config: dict):
    """Load each Qwen3.5 size with its own weights and configured delta rule."""
    from peft import LoraConfig, get_peft_model
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    model_config = config["model"]
    # Legacy checkpoints predate this optional backend selector.
    delta_rule_backend = model_config.get("delta_rule_backend", "auto")
    if delta_rule_backend not in ("auto", "torch"):
        raise ValueError("model.delta_rule_backend must be auto or torch")
    revision = model_config.get("revision")
    pretrained_options = {"revision": revision} if revision is not None else {}
    processor = AutoProcessor.from_pretrained(model_config["name_or_path"], **pretrained_options)
    tokenizer = processor.tokenizer
    tokenizer.padding_side = "left"
    tokenizer.add_special_tokens({"additional_special_tokens": list(SPECIAL_TOKENS)})
    base = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_config["name_or_path"], dtype=torch.bfloat16,
        attn_implementation=model_config["attn_implementation"],
        **pretrained_options,
    )
    base.resize_token_embeddings(len(tokenizer))
    delta_rule_spec = configure_delta_rule_backend(base, delta_rule_backend)
    # Use fully qualified module paths: suffix-only targets would also adapt vision.
    targets = [name for name, module in base.named_modules()
               if "language_model" in name and name.rsplit(".", 1)[-1] in LORA_TARGETS and isinstance(module, nn.Linear)]
    if not targets:
        raise ValueError("No Qwen3.5 language projection targets were found")
    backbone = get_peft_model(base, LoraConfig(
        r=model_config["lora_rank"], lora_alpha=model_config["lora_alpha"],
        lora_dropout=model_config["lora_dropout"], target_modules="(?:" + "|".join(re.escape(name) for name in targets) + ")",
        bias="none", task_type="CAUSAL_LM",
    ))
    if model_config["gradient_checkpointing"]:
        backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    result = GranaModel(backbone, tokenizer)
    evaluation_vision_sdpa = config.get("evaluation", {}).get("batched_vision_sdpa", False)
    training_vision_sdpa = config.get("training", {}).get("batched_vision_sdpa", False)
    if evaluation_vision_sdpa or training_vision_sdpa:
        from .vision_sdpa import configure_batched_vision_sdpa
        configure_batched_vision_sdpa(result, enabled=evaluation_vision_sdpa,
                                     training_enabled=training_vision_sdpa)
    # Keep small heads FP32 and let the training autocast choose BF16 compute.
    result.model_spec = {"name_or_path": model_config["name_or_path"], "revision": revision,
                         "hidden_size": result.hidden_size,
                         "lora_rank": model_config["lora_rank"], "lora_alpha": model_config["lora_alpha"],
                         "lora_dropout": model_config["lora_dropout"], "token_ids": result.token_ids,
                         "delta_rule_backend": delta_rule_spec}
    return result, processor
