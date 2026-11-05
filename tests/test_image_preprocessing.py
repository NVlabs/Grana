# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared training/serving image preprocessing and stateless batch handling."""
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch

from grana.modeling import image_preprocessing as shared
from grana.inference.optimization import RequestImages, device_image_processor

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="shared image preprocessing needs CUDA")


class RecordingImageProcessor:
    size = {"shortest_edge": 1024, "longest_edge": 81920}
    patch_size = 16
    merge_size = 2

    def __init__(self):
        self.calls = []

    def rescale_and_normalize(self, images, *args, **kwargs):
        return images

    def __call__(self, images=None, **kwargs):
        self.calls.append((images, kwargs))
        return {"pixel_values": torch.stack(images), "image_grid_thw": torch.tensor([[1, 2, 2]] * len(images))}


class Processor:
    def __init__(self):
        self.image_processor = RecordingImageProcessor()
        self.tokenizer = object()

    def __call__(self, text=None, images=None, **kwargs):
        return {"input_ids": torch.tensor([[len(text)]]), **self.image_processor(images=images, **kwargs)}


def test_wrapper_prepares_each_batch_and_preserves_options_and_metadata(monkeypatch):
    transfers = []

    def transfer(images, device):
        transfers.append((images, device))
        return [torch.from_numpy(np.asarray(frame).copy()).permute(2, 0, 1) for frame in images]

    monkeypatch.setattr(shared, "_device_frames", transfer)
    original = Processor()
    configured = shared.configure_image_preprocessing(original, "cuda:1", resize_on_device=True)
    assert configured is not original and configured.tokenizer is original.tokenizer
    assert configured.image_processor.size is original.image_processor.size
    assert configured.image_processor.patch_size == 16
    assert configured.image_processor.merge_size == 2
    assert configured.image_preprocessing == {"device": "cuda:1"}
    assert not hasattr(original, "image_preprocessing")
    frames = [Image.fromarray(np.full((4, 4, 3), value, dtype=np.uint8)) for value in (7, 29)]
    options = {"return_tensors": "pt", "min_pixels": 1024, "max_pixels": 81920}
    first = configured(text="stage", images=frames, **options)
    second = configured(text="progress", images=frames[::-1], **options)
    assert len(transfers) == len(original.image_processor.calls) == 2
    assert torch.equal(first["pixel_values"], second["pixel_values"].flip(0))
    assert not torch.equal(first["pixel_values"], second["pixel_values"])
    assert all(kwargs == {"device": "cuda:1", **options} for _, kwargs in original.image_processor.calls)


def test_configuration_is_idempotent_and_switches_from_original_processor():
    original = Processor()
    configured = shared.configure_image_preprocessing(original, "cuda:0", resize_on_device=True)
    assert shared.configure_image_preprocessing(configured, "cuda:0", resize_on_device=True) is configured
    moved = shared.configure_image_preprocessing(configured, "cuda:1", resize_on_device=True)
    assert moved is not configured
    assert moved.image_processor.original is original.image_processor
    assert moved.image_preprocessing["device"] == "cuda:1"
    restored = shared.configure_image_preprocessing(moved, "cpu", resize_on_device=False)
    assert restored.image_processor is original.image_processor
    assert not hasattr(restored, "image_preprocessing")
    assert configured.image_preprocessing["device"] == "cuda:0"
    assert shared.configure_image_preprocessing(original, "cpu", resize_on_device=False) is original
    dummy = object()
    assert shared.configure_image_preprocessing(dummy, "cpu", resize_on_device=False) is dummy


@pytest.mark.parametrize("device", ["cpu", "meta"])
def test_gpu_resize_rejects_non_cuda_devices(device):
    with pytest.raises(ValueError, match="requires a CUDA device"):
        shared.configure_image_preprocessing(Processor(), device, resize_on_device=True)


@pytest.mark.parametrize("flag", [None, 1, "true"])
def test_resize_configuration_requires_boolean(flag):
    with pytest.raises(ValueError, match="must be a boolean"):
        shared.configure_image_preprocessing(Processor(), "cpu", resize_on_device=flag)


@pytest.mark.parametrize("images", [[], [np.zeros((3, 4, 3), dtype=np.float32)],
                                     [np.zeros((3, 4), dtype=np.uint8)],
                                     [np.zeros((3, 4, 4), dtype=np.uint8)]])
def test_gpu_frames_reject_invalid_inputs_before_transfer(images):
    with pytest.raises(ValueError, match="RGB observations?"):
        shared._device_frames(images, torch.device("cuda"))


def test_device_override_cannot_silently_change_the_configured_backend():
    process, _ = shared.device_image_processor(RecordingImageProcessor(), "cuda:0")
    with pytest.raises(ValueError, match="must match"):
        process(images=[], device="cpu")


def real_image_processor():
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor

    return Qwen2VLImageProcessor(size={"shortest_edge": 1024, "longest_edge": 81920}, patch_size=16,
                                 temporal_patch_size=2, merge_size=2, image_mean=[0.5] * 3, image_std=[0.5] * 3)


@needs_cuda
def test_training_and_request_processing_match_on_cuda_across_shapes_and_batches():
    rng = np.random.default_rng(17)
    frames = [rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
              for height, width in ((240, 320), (320, 240), (240, 320))]
    processor = SimpleNamespace(image_processor=real_image_processor())
    configured = shared.configure_image_preprocessing(processor, "cuda", resize_on_device=True)
    options = {"return_tensors": "pt", "min_pixels": 1024, "max_pixels": 65536}
    for observations in (frames, [Image.fromarray(frame) for frame in frames[::-1]]):
        training = configured.image_processor(images=observations, **options)
        state = RequestImages(processor, device_image_processor(processor.image_processor, torch.device("cuda")))
        inference = state.processor.image_processor(images=observations, **options)
        host = processor.image_processor(images=observations, **options)
        assert training["pixel_values"].device.type == "cuda"
        assert torch.equal(training["pixel_values"], inference["pixel_values"])
        assert torch.equal(training["image_grid_thw"], inference["image_grid_thw"])
        assert torch.equal(training["image_grid_thw"].cpu(), host["image_grid_thw"])
        assert not torch.equal(training["pixel_values"].cpu(), host["pixel_values"])
        assert torch.all(training["image_grid_thw"][:, 1:].prod(-1) * 16 ** 2 <= 65536)
