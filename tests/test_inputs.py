# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch

from grana.modeling.inputs import collate_examples, preprocessing_spec, serialize_messages, validate_max_pixels


def example(height=240, width=320):
    pixels = np.arange(height * width * 3, dtype=np.uint8).reshape(height, width, 3)
    images = tuple(Image.fromarray(pixels) for _ in range(8))
    return SimpleNamespace(images=images, messages=lambda: serialize_messages("Perform the task.", "A", images, ("top",)))


class RecordingProcessor:
    def __init__(self, grid=(1, 16, 20), patch_size=16):
        self.image_processor = SimpleNamespace(size={"shortest_edge": 65536, "longest_edge": 16777216},
                                               patch_size=patch_size)
        self.grid = grid
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        return "template"

    def __call__(self, *, text, images, padding, return_tensors, **kwargs):
        self.calls.append({"images": images, "kwargs": kwargs})
        return {"input_ids": torch.zeros(len(text), 8, dtype=torch.long),
                "image_grid_thw": torch.tensor([self.grid] * len(images))}


@pytest.mark.parametrize("value", [True, False, 0, -1, 1023, 76800.0, "76800", {}, [], float("nan")])
def test_invalid_budget_rejected_before_serialization(value):
    with pytest.raises(ValueError, match="max_pixels"):
        validate_max_pixels(value)
    with pytest.raises(ValueError, match="max_pixels"):
        preprocessing_spec(value)
    with pytest.raises(ValueError, match="max_pixels"):
        collate_examples(None, [object()], value, "cpu")


def test_default_has_no_resize_override_and_budget_uses_actual_patch_size():
    sample = example()
    processor = RecordingProcessor(grid=(1, 14, 20), patch_size=14)
    assert preprocessing_spec(None) == {"version": 1, "policy": "processor_default", "max_pixels": None}
    assert preprocessing_spec(60000) == {"version": 1, "policy": "pixel_budget", "max_pixels": 60000}
    collate_examples(processor, [sample], None, "cpu")
    collate_examples(processor, [sample], 60000, "cpu")
    assert processor.calls[0]["kwargs"] == {}
    assert processor.calls[1]["kwargs"] == {"min_pixels": 60000, "max_pixels": 60000}
    assert all(got is expected for got, expected in zip(processor.calls[1]["images"], sample.images, strict=True))


def test_explicit_budget_cannot_be_verified_without_processor_grid():
    processor = RecordingProcessor()
    processor.image_processor.patch_size = None
    with pytest.raises(ValueError, match="Cannot verify max_pixels"):
        collate_examples(processor, [example()], 76800, "cpu")


@pytest.mark.parametrize("grid", [(1, float("nan"), 20), (1, float("inf"), 20),
                                  (1, 14.0, 20.0), (1, -1, 20), (1, 0, 20), (0, 14, 20)])
def test_explicit_budget_rejects_invalid_processor_grid(grid):
    with pytest.raises(ValueError, match="Cannot verify max_pixels"):
        collate_examples(RecordingProcessor(grid=grid), [example()], 76800, "cpu")


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_explicit_budget_cannot_overflow_processor_grid_arithmetic(dtype):
    class OversizedGrid(RecordingProcessor):
        def __call__(self, **kwargs):
            inputs = super().__call__(**kwargs)
            edge = 65536 if dtype == torch.int32 else 2**32
            inputs["image_grid_thw"] = torch.tensor([[1, edge, edge]] * 8, dtype=dtype)
            return inputs

    with pytest.raises(ValueError, match="Processor resize exceeded max_pixels"):
        collate_examples(OversizedGrid(), [example()], 76800, "cpu")


@pytest.fixture
def real_processor():
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor

    class Adapter(RecordingProcessor):
        def __call__(self, *, text, images, padding, return_tensors, **kwargs):
            self.calls.append({"images": images, "kwargs": kwargs})
            batch = self.image_processor(images=images, return_tensors=return_tensors, **kwargs)
            return {"input_ids": torch.zeros(len(text), 8, dtype=torch.long), **batch}

    old_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    processor = Adapter()
    processor.image_processor = Qwen2VLImageProcessor(
        size={"shortest_edge": 65536, "longest_edge": 16777216}, patch_size=16,
        temporal_patch_size=2, merge_size=2,
    )
    yield processor
    torch.set_num_threads(old_threads)


def test_native_processor_default_matches_historical_max_only_tensors(real_processor):
    sample = example()
    original_bytes = [image.tobytes() for image in sample.images]
    old = real_processor(text=["template"], images=list(sample.images), padding=True,
                         return_tensors="pt", max_pixels=76800)
    new = collate_examples(real_processor, [sample], None, "cpu")
    assert set(new) == set(old)
    assert all(torch.equal(new[key], old[key]) for key in new)
    assert new["image_grid_thw"].tolist() == [[1, 16, 20]] * 8
    assert [image.tobytes() for image in sample.images] == original_bytes


def test_native_processor_numeric_budget_changes_geometry_and_respects_cap(real_processor):
    sample = example()
    batch = collate_examples(real_processor, [sample], 76800, "cpu")
    assert batch["image_grid_thw"].tolist() == [[1, 14, 20]] * 8
    assert (batch["image_grid_thw"][:, 1:].prod(1) * 16**2 <= 76800).all()
    assert all(image.size == (320, 240) for image in sample.images)


def test_native_rounding_over_budget_is_rejected_and_legacy_small_cap_reproduces(real_processor):
    sample = example(64, 160)
    with pytest.raises(ValueError, match="max_pixels=76800: got 79872"):
        collate_examples(real_processor, [sample], 76800, "cpu")
    # The old small-cap path set both min/max and could itself exceed the cap.
    sample = example(1, 200)
    old = real_processor(text=["template"], images=list(sample.images), padding=True,
                         return_tensors="pt", min_pixels=1024, max_pixels=1024)
    with pytest.raises(ValueError, match="max_pixels=1024: got 15360"):
        collate_examples(real_processor, [sample], 1024, "cpu")
    compatible = collate_examples(real_processor, [sample], 1024, "cpu", enforce_pixel_budget=False)
    assert all(torch.equal(compatible[key], old[key]) for key in old)
