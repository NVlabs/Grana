# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU correctness only: these checks do not qualify GPU BF16 or throughput."""
import copy

import pytest
import torch
from peft import LoraConfig, get_peft_model
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config, Qwen3_5VisionConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5ForConditionalGeneration, Qwen3_5VisionAttention, Qwen3_5VisionModel,
)

from grana.modeling.model import GranaModel, SPECIAL_TOKENS
from grana.modeling.vision_sdpa import batched_vision_sdpa_stats, configure_batched_vision_sdpa


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def vision_case(images=4):
    torch.manual_seed(42)
    config = Qwen3_5VisionConfig(hidden_size=32, num_heads=4, intermediate_size=64,
                               depth=2, out_hidden_size=32, patch_size=2,
                               spatial_merge_size=2, num_position_embeddings=16)
    config._attn_implementation = "sdpa"
    original = Qwen3_5VisionModel(config).train().requires_grad_(False)
    original.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    assert original.is_gradient_checkpointing
    optimized = copy.deepcopy(original)
    configure_batched_vision_sdpa(optimized, enabled=True, training_enabled=True)
    pixels = torch.randn(images * 16, 3 * 2 * 2**2)
    grid = torch.tensor([[1, 4, 4]] * images)
    return original, optimized, pixels, grid


@pytest.mark.parametrize("images", [1, 4, 64])
def test_frozen_encoder_training_and_eval_parity_and_independent_images(images):
    original, optimized, pixels, grid = vision_case(images)
    assert original.state_dict().keys() == optimized.state_dict().keys()
    for training in (True, False, True):
        original.train(training)
        optimized.train(training)
        # Normal global autograd remains enabled for training; the frozen visual
        # output itself needs no graph and language adapters remain unaffected.
        with torch.set_grad_enabled(training):
            expected = original(pixels, grid).pooler_output
            actual = optimized(pixels, grid).pooler_output
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        assert not actual.requires_grad
    stats = batched_vision_sdpa_stats(optimized)
    assert stats["training"] == {"calls": 4, "hits": 4, "images": images * 4, "fallbacks": {}}
    assert stats["evaluation"] == {"calls": 2, "hits": 2, "images": images * 2, "fallbacks": {}}
    if images > 1:
        changed = pixels.clone()
        changed[16:32] += 10
        actual = optimized(changed, grid).pooler_output
        torch.testing.assert_close(actual[:4], expected[:4], rtol=0, atol=0)
        torch.testing.assert_close(actual[8:], expected[8:], rtol=0, atol=0)
        assert not torch.equal(actual[4:8], expected[4:8])


@pytest.mark.parametrize("condition,reason", [
    ("trainable_merger", "trainable_encoder"),
    ("trainable_patch", "trainable_encoder"),
    ("grad_pixels", "input_requires_grad"),
    ("dropout", "attention_dropout"),
    ("backend", "non_sdpa_backend"),
    ("variable_images", "unequal_image_lengths"),
    ("disabled", "disabled"),
])
def test_training_fallbacks_preserve_outputs_and_gradients(condition, reason):
    original, optimized, pixels, grid = vision_case(2)
    if condition == "trainable_merger":
        next(original.merger.parameters()).requires_grad_(True)
        next(optimized.merger.parameters()).requires_grad_(True)
    elif condition == "trainable_patch":
        next(original.patch_embed.parameters()).requires_grad_(True)
        next(optimized.patch_embed.parameters()).requires_grad_(True)
    elif condition == "grad_pixels":
        pixels.requires_grad_(True)
    elif condition == "dropout":
        for model in (original, optimized):
            for block in model.blocks:
                block.attn.attention_dropout = 0.1
    elif condition == "backend":
        original.config._attn_implementation = optimized.config._attn_implementation = "eager"
    elif condition == "variable_images":
        grid = torch.tensor([[1, 4, 4], [1, 2, 2]])
        pixels = pixels[:20]
    elif condition == "disabled":
        configure_batched_vision_sdpa(optimized, enabled=True, training_enabled=False)
    left = pixels.detach().clone().requires_grad_(pixels.requires_grad)
    right = pixels.detach().clone().requires_grad_(pixels.requires_grad)
    rng = torch.get_rng_state()
    expected = original(left, grid).pooler_output
    torch.set_rng_state(rng)
    actual = optimized(right, grid).pooler_output
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    if expected.requires_grad:
        expected.square().mean().backward()
        actual.square().mean().backward()
        if left.requires_grad:
            torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)
        for a, b in zip(original.parameters(), optimized.parameters(), strict=True):
            assert (a.grad is None) == (b.grad is None)
            if a.grad is not None:
                torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
    stats = batched_vision_sdpa_stats(optimized)["training"]
    assert stats["hits"] == 0
    assert stats["fallbacks"][reason] == 2
    # Non-reentrant checkpointing may replay trainable blocks outside the
    # encoder scope. Those remain original too, rather than trusting stale state.
    assert set(stats["fallbacks"]) <= {reason, "unverified_encoder"}


def test_training_requires_complete_encoder_scope_and_restores_scope_on_error(monkeypatch):
    original, optimized, pixels, grid = vision_case(1)
    attn = optimized.blocks[0].attn
    hidden, angles = torch.randn(16, 32), torch.randn(16, 8)
    positions = angles.cos(), angles.sin()
    boundaries = torch.tensor([0, 16], dtype=torch.int32)
    attn(hidden, boundaries, position_embeddings=positions)
    assert batched_vision_sdpa_stats(optimized)["training"]["fallbacks"] == {"unverified_encoder": 1}
    with monkeypatch.context() as patch:
        patch.setattr(optimized, "_grana_original_encoder_forward", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("probe")))
        with pytest.raises(RuntimeError, match="probe"):
            optimized(pixels, grid)
    attn(hidden, boundaries, position_embeddings=positions)
    stats = batched_vision_sdpa_stats(optimized)["training"]
    assert stats["hits"] == 0 and stats["fallbacks"] == {"unverified_encoder": 2}


def test_counters_reset_toggle_and_profile_do_not_change_checkpoint_or_mode():
    original, optimized, pixels, grid = vision_case(2)
    weights = {name: value.clone() for name, value in optimized.state_dict().items()}
    configure_batched_vision_sdpa(optimized, enabled=False, training_enabled=True, profile=True)
    optimized(pixels, grid)
    optimized.eval()
    with torch.no_grad():
        optimized(pixels, grid)
    stats = batched_vision_sdpa_stats(optimized, reset=True, synchronize=True)
    assert stats["training"]["hits"] == 2
    assert stats["evaluation"]["fallbacks"] == {"disabled": 2}
    assert stats["profile"]["vision_calls"] == 2
    assert stats["profile"]["host_seconds"] > 0
    assert stats["profile"]["cuda_seconds"] == stats["profile"]["pending_cuda_spans"] == 0
    assert not optimized.training and not any(p.requires_grad for p in optimized.parameters())
    assert batched_vision_sdpa_stats(optimized)["training"]["calls"] == 0
    assert batched_vision_sdpa_stats(optimized)["profile"]["vision_calls"] == 0
    for name, weight in optimized.state_dict().items():
        torch.testing.assert_close(weight, weights[name], rtol=0, atol=0)
    assert batched_vision_sdpa_stats(original)["attention_modules"] == 0
    original.load_state_dict(optimized.state_dict(), strict=True)


class Tokenizer:
    def convert_tokens_to_ids(self, token):
        return 40 + SPECIAL_TOKENS.index(token)

    def encode(self, letter, add_special_tokens=False):
        return [10 + ord(letter) - ord("A")]


def language_lora_case():
    torch.manual_seed(812)
    config = Qwen3_5Config(
        image_token_id=3, vision_start_token_id=1, vision_end_token_id=2,
        text_config={"vocab_size": 128, "hidden_size": 32, "intermediate_size": 64,
                     "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2,
                     "head_dim": 8, "layer_types": ["full_attention", "full_attention"]},
        vision_config={"hidden_size": 32, "num_heads": 4, "intermediate_size": 64,
                       "depth": 2, "out_hidden_size": 32, "patch_size": 2,
                       "spatial_merge_size": 2, "num_position_embeddings": 16})
    config._attn_implementation = "sdpa"
    base = Qwen3_5ForConditionalGeneration(config)
    backbone = get_peft_model(base, LoraConfig(r=2, lora_alpha=4, lora_dropout=0.1,
        target_modules=r".*language_model.*\.(q_proj|v_proj)", bias="none", task_type="CAUSAL_LM"))
    backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    assert backbone.is_gradient_checkpointing
    model = GranaModel(backbone, Tokenizer()).train()
    assert all(not p.requires_grad for p in base.model.visual.parameters())
    # Both LoRA matrices receive useful gradients; production starts B at zero,
    # but a nonzero A/B pair also checks backward at later training steps.
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_B" in name:
                parameter.normal_(std=0.02)
    inputs = {"input_ids": torch.tensor([[1, 3, 3, 3, 3, 2] + [marker] * 8 for marker in (42, 43, 42)]),
              "attention_mask": torch.ones(3, 14, dtype=torch.long),
              "mm_token_type_ids": torch.tensor([[0, 1, 1, 1, 1, 0] + [0] * 8] * 3),
              "pixel_values": torch.randn(3 * 16, 3 * 2 * 2**2),
              "image_grid_thw": torch.tensor([[1, 4, 4]] * 3)}
    return model, inputs


def test_frozen_training_path_preserves_lora_heads_gradients_and_optimizer_update_across_eval():
    original, inputs = language_lora_case()
    optimized = copy.deepcopy(original)
    configure_batched_vision_sdpa(optimized, enabled=True, training_enabled=True)
    optimizers = [torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
                  for model in (original, optimized)]
    initial_keys = set(original.state_dict())
    for step in range(2):
        rng = torch.get_rng_state()
        outputs = []
        for model, optimizer in zip((original, optimized), optimizers, strict=True):
            torch.set_rng_state(rng)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            output = model(inputs, ["B", "C", "D"], [9, 9, 9])
            loss = output["progress_logits"].square().mean() + output["stage_logits"][..., :9].square().mean()
            loss.backward()
            outputs.append((output, loss))
        for key in ("progress_logits", "stage_logits"):
            torch.testing.assert_close(outputs[0][0][key], outputs[1][0][key], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(outputs[0][1], outputs[1][1], rtol=1e-5, atol=1e-6)
        assert torch.equal(outputs[0][0]["stage_logits"][:, 5].argmax(-1), outputs[1][0]["stage_logits"][:, 5].argmax(-1))
        saw_lora = saw_progress = saw_temporal = False
        for (name, left), (right_name, right) in zip(original.named_parameters(), optimized.named_parameters(), strict=True):
            assert name == right_name
            assert (left.grad is None) == (right.grad is None)
            if left.grad is not None:
                torch.testing.assert_close(left.grad, right.grad, rtol=1e-4, atol=1e-5)
                saw_lora |= "lora_" in name and bool(left.grad.abs().sum())
                saw_progress |= "progress_head" in name and bool(left.grad.abs().sum())
                saw_temporal |= "temporal_head" in name and bool(left.grad.abs().sum())
        assert saw_lora and saw_progress and saw_temporal
        for optimizer in optimizers:
            optimizer.step()
        for left, right in zip(original.parameters(), optimized.parameters(), strict=True):
            torch.testing.assert_close(left, right, rtol=1e-5, atol=1e-6)
        if step == 0:
            for model in (original, optimized):
                model.eval()
            with torch.no_grad():
                left = original(inputs, ["B", "C", "D"], [9, 9, 9])
                right = optimized(inputs, ["B", "C", "D"], [9, 9, 9])
            torch.testing.assert_close(left["progress_logits"], right["progress_logits"], rtol=1e-5, atol=1e-6)
    stats = batched_vision_sdpa_stats(optimized)
    assert stats["training"]["hits"] == 4 and not stats["training"]["fallbacks"]
    assert stats["evaluation"]["hits"] == 2
    assert set(optimized.state_dict()) == initial_keys
    assert optimized.training and optimized.backbone.is_gradient_checkpointing
    assert all(p.grad is None for p in optimized.backbone.base_model.model.model.visual.parameters())
