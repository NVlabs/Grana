# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from contextvars import ContextVar
import copy
from types import SimpleNamespace

import pytest
import torch
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config, Qwen3_5VisionConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model, Qwen3_5VisionAttention

from grana.inference.optimization import RequestImages, configure_request_image_reuse
from grana.modeling.vision_sdpa import configure_batched_vision_sdpa


def attention_case(lengths):
    torch.manual_seed(42)
    config = Qwen3_5VisionConfig(hidden_size=32, num_heads=4, intermediate_size=64, depth=1)
    config._attn_implementation = "sdpa"
    original = Qwen3_5VisionAttention(config).eval()
    optimized = copy.deepcopy(original)
    configure_batched_vision_sdpa(optimized)
    hidden = torch.randn(sum(lengths), 32)
    angles = torch.randn(sum(lengths), 8)
    boundaries = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32)
    return original, optimized, hidden, boundaries, (angles.cos(), angles.sin())


@pytest.mark.parametrize("lengths", [[8], [8, 8, 8, 8], [3, 5, 2]])
def test_attention_parity_for_equal_images_and_variable_length_fallback(lengths):
    original, optimized, hidden, boundaries, positions = attention_case(lengths)
    assert original.state_dict().keys() == optimized.state_dict().keys()
    with torch.no_grad():
        torch.testing.assert_close(optimized(hidden, boundaries, position_embeddings=positions),
                                   original(hidden, boundaries, position_embeddings=positions), rtol=1e-5, atol=1e-6)


def test_attention_has_no_cross_image_leakage_and_can_be_disabled():
    original, optimized, hidden, boundaries, positions = attention_case([8, 8, 8])
    with torch.no_grad():
        expected = optimized(hidden, boundaries, position_embeddings=positions)
        changed = hidden.clone()
        changed[8:16] += 10
        actual = optimized(changed, boundaries, position_embeddings=positions)
        torch.testing.assert_close(actual[:8], expected[:8], rtol=0, atol=0)
        torch.testing.assert_close(actual[16:], expected[16:], rtol=0, atol=0)
        assert not torch.equal(actual[8:16], expected[8:16])
        configure_batched_vision_sdpa(optimized, enabled=False)
        torch.testing.assert_close(optimized(hidden, boundaries, position_embeddings=positions),
                                   original(hidden, boundaries, position_embeddings=positions), rtol=0, atol=0)


@pytest.mark.parametrize("training", [True, False])
def test_autograd_and_training_preserve_original_path(training):
    original, optimized, hidden, boundaries, positions = attention_case([8, 8])
    original.train(training)
    optimized.train(training)
    left, right = hidden.clone().requires_grad_(), hidden.clone().requires_grad_()
    original(left, boundaries, position_embeddings=positions).square().sum().backward()
    optimized(right, boundaries, position_embeddings=positions).square().sum().backward()
    torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)
    for a, b in zip(original.parameters(), optimized.parameters()):
        torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)


class ImageProcessor:
    def __init__(self):
        self.calls = 0

    def __call__(self, images, **kwargs):
        self.calls += 1
        return {"pixel_values": torch.full((4, 3 * 2 * 16**2), float(images[0])),
                "image_grid_thw": torch.tensor([[1, 2, 2]])}


def tiny_vision_model():
    config = Qwen3_5Config(
        text_config={"vocab_size": 32, "hidden_size": 32, "intermediate_size": 64,
                     "num_hidden_layers": 1, "num_attention_heads": 4, "num_key_value_heads": 2,
                     "head_dim": 8, "layer_types": ["full_attention"]},
        vision_config={"hidden_size": 32, "num_heads": 4, "intermediate_size": 64,
                       "depth": 1, "out_hidden_size": 32, "patch_size": 16,
                       "spatial_merge_size": 2, "num_position_embeddings": 16})
    return Qwen3_5Model(config).eval()


def test_request_reuse_is_exact_and_new_request_recomputes_images():
    model = tiny_vision_model()
    context = ContextVar("test-image-reuse", default=None)
    configure_request_image_reuse(model, context)
    processor = SimpleNamespace(image_processor=ImageProcessor())

    def score(color):
        state = RequestImages(processor)
        token = context.set(state)
        try:
            cpu = state.processor.image_processor(images=[color], return_tensors="pt")
            inputs = state.to_device(cpu, "cpu")
            with torch.inference_mode():
                first = model.get_image_features(inputs["pixel_values"], inputs["image_grid_thw"], return_dict=True)
                second_cpu = state.processor.image_processor(images=[color], return_tensors="pt")
                second_inputs = state.to_device(second_cpu, "cpu")
                assert second_inputs["pixel_values"] is inputs["pixel_values"]
                second = model.get_image_features(second_inputs["pixel_values"], second_inputs["image_grid_thw"], return_dict=True)
                assert first is second
                return first.pooler_output[0].clone(), state.stats()
        finally:
            context.reset(token)

    first, stats = score(1)
    second, other = score(2)
    assert not torch.equal(first, second)
    assert stats == other == {"scope": "one_prediction", "image_processor_calls": 1,
                              "vision_calls": 1, "vision_cache_hits": 1}
    assert processor.image_processor.calls == 2 and context.get() is None


def test_image_reuse_rejects_changed_images_or_preprocessing():
    state = RequestImages(SimpleNamespace(image_processor=ImageProcessor()))
    state.processor.image_processor(images=[1], return_tensors="pt")
    with pytest.raises(ValueError, match="identical observations"):
        state.processor.image_processor(images=[2], return_tensors="pt")
    with pytest.raises(ValueError, match="identical observations"):
        state.processor.image_processor(images=[1], return_tensors="pt", max_pixels=1024)
