# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host-side serving pieces: exact text-token reuse and device image preprocessing."""
from contextvars import ContextVar
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch

from grana.inference import predictor as inference
from grana.inference.optimization import RequestImages, TextInputCache, device_image_processor

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="device preprocessing parity needs CUDA")


class ImageProcessor:
    """Grid depends on the image value, so cache keys must include it."""

    def __init__(self):
        self.calls = 0

    def __call__(self, images=None, **kwargs):
        self.calls += 1
        value = float(images[0])
        return {"pixel_values": torch.full((len(images), 4), value),
                "image_grid_thw": torch.tensor([[1, 2, 2 * int(value)]] * len(images))}


class Processor:
    """Minimal HF-processor contract: text tokens depend on text and grid; images via image_processor."""

    def __init__(self):
        self.image_processor = ImageProcessor()
        # A shared list, not an int: RequestImages calls a shallow copy of this processor.
        self.text_requests = []

    @property
    def text_calls(self):
        return len(self.text_requests)

    def __call__(self, text=None, images=None, padding=True, return_tensors="pt", **kwargs):
        self.text_requests.append(text)
        # Like HF processors, pass only the image options (not text padding) to the image processor.
        image_inputs = self.image_processor(images=images, **kwargs)
        width = int(image_inputs["image_grid_thw"][:, 1:].prod(-1).sum())
        ids = torch.tensor([[len(item) + width + index for index in range(3)] for item in text])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids),
                "mm_token_type_ids": torch.zeros_like(ids), **image_inputs}


def call(state, text, image=1.0, **kwargs):
    return state.processor(text=[text], images=[image], padding=True, return_tensors="pt", **kwargs)


def test_text_cache_reuses_exact_tokens_across_requests_and_keys_on_text_grid_and_options():
    processor, cache = Processor(), TextInputCache(max_entries=8)
    first = RequestImages(processor, text_cache=cache)
    expected = call(first, "stage prompt")
    again = call(first, "stage prompt")  # same request: tokens cached, images reused
    assert processor.text_calls == 1 and processor.image_processor.calls == 1
    assert all(torch.equal(again[key], expected[key]) for key in expected)
    assert again["input_ids"] is not expected["input_ids"]
    again["input_ids"].fill_(-1)  # callers cannot corrupt the cache
    second = RequestImages(processor, text_cache=cache)
    hit = call(second, "stage prompt")
    assert torch.equal(hit["input_ids"], expected["input_ids"])
    assert hit["pixel_values"] is second.image_processor.output["pixel_values"]
    assert processor.text_calls == 1 and processor.image_processor.calls == 2
    assert second.stats()["text_cache_hits"] == 1 and second.stats()["text_cache_misses"] == 0
    call(second, "focus prompt")  # new text
    third = RequestImages(processor, text_cache=cache)
    changed_grid = call(third, "stage prompt", image=2.0)  # same text, different image grid
    assert processor.text_calls == 3
    assert not torch.equal(changed_grid["input_ids"], expected["input_ids"])
    fourth = RequestImages(processor, text_cache=cache)
    call(fourth, "stage prompt", max_pixels=1024)  # different processor options
    assert processor.text_calls == 4
    # Every processor run is a miss, including the first call with unseen options.
    assert cache.stats() == {"entries": 4, "max_entries": 8, "hits": 2, "misses": 4}


def test_text_cache_is_bounded_lru_and_validates_size():
    processor, cache = Processor(), TextInputCache(max_entries=2)
    for text in ("a", "b", "a", "c"):
        call(RequestImages(processor, text_cache=cache), text)
    assert cache.stats()["entries"] == 2
    calls = processor.text_calls
    call(RequestImages(processor, text_cache=cache), "a")  # retained: most recently used before c
    assert processor.text_calls == calls
    call(RequestImages(processor, text_cache=cache), "b")  # evicted
    assert processor.text_calls == calls + 1
    for bad in (0, -1, 1.5, True):
        with pytest.raises(ValueError, match="positive integer"):
            TextInputCache(max_entries=bad)


def test_stats_are_unchanged_without_host_pieces():
    state = RequestImages(Processor())
    call(state, "prompt")
    assert state.stats() == {"scope": "one_prediction", "image_processor_calls": 1,
                             "vision_calls": 0, "vision_cache_hits": 0}


IMAGE_STD = 0.5


def real_image_processor():
    """The released Qwen3.5 preprocessor_config.json (mean/std 0.5, not the CLIP defaults)."""
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
    return Qwen2VLImageProcessor(size={"shortest_edge": 65536, "longest_edge": 16777216}, patch_size=16,
                                 temporal_patch_size=2, merge_size=2, image_mean=[0.5] * 3, image_std=[IMAGE_STD] * 3)


def frames(count=4, seed=0):
    rng = np.random.default_rng(seed)
    return [Image.fromarray(rng.integers(0, 256, (240, 320, 3), dtype=np.uint8)) for _ in range(count)]


def test_device_processor_on_cpu_matches_host_processor_exactly():
    original = real_image_processor()
    images = frames()
    expected = original(images=images, return_tensors="pt", max_pixels=81920)
    process, description = device_image_processor(original, torch.device("cpu"))
    actual = process(images=images, return_tensors="pt", max_pixels=81920)
    assert description == {"device": "cpu"}
    assert torch.equal(actual["pixel_values"], expected["pixel_values"])
    assert torch.equal(actual["image_grid_thw"], expected["image_grid_thw"])


def test_device_processor_rejects_processors_without_the_torch_backend():
    with pytest.raises(ValueError, match="torch-backend image processor"):
        device_image_processor(SimpleNamespace(), torch.device("cpu"))


@needs_cuda
def test_cuda_preprocessing_matches_host_grid_and_stays_within_one_level():
    original = real_image_processor()
    images = frames(8, seed=3)
    expected = original(images=images, return_tensors="pt", max_pixels=81920)
    process, description = device_image_processor(original, torch.device("cuda"))
    approximate = process(images=images, return_tensors="pt", max_pixels=81920)
    assert description == {"device": "cuda"} and approximate["pixel_values"].device.type == "cuda"
    assert torch.equal(approximate["image_grid_thw"].cpu(), expected["image_grid_thw"])
    # Device interpolation rounds differently from the host uint8 kernel. 240x320 resizes only the
    # height (to 256), so the difference is at most one uint8 level. Resizing both axes can differ more
    # (the host kernel rounds/clamps to uint8 between passes): up to 23 levels on noise.
    levels = (approximate["pixel_values"].cpu() - expected["pixel_values"]).abs() * 255 * IMAGE_STD
    assert levels.max() <= 1 + 1e-4  # one RGB level at most; equality would also be acceptable


@needs_cuda
def test_request_images_with_device_processor_keeps_reuse_identity():
    original = real_image_processor()
    processor = SimpleNamespace(image_processor=original)
    state = RequestImages(processor, device_image_processor(original, torch.device("cuda")))
    images = frames(2)
    first = state.processor.image_processor(images=images, return_tensors="pt")
    second = state.processor.image_processor(images=images, return_tensors="pt")
    assert first is second and first["pixel_values"].device.type == "cuda"
    moved = state.to_device(dict(first), torch.device("cuda"))
    assert moved["pixel_values"] is first["pixel_values"]
    assert state.stats()["image_preprocessing"] == {"device": "cuda"}


def stub_predictor(device="cpu", reuse=True):
    predictor = inference.GranaPredictor.__new__(inference.GranaPredictor)
    predictor.device, predictor.reuse_images = torch.device(device), reuse
    predictor.processor = SimpleNamespace(image_processor=real_image_processor())
    predictor._request_images = ContextVar("host-pipeline-test", default=None)
    return predictor


def test_host_pipeline_requires_cuda_and_enables_both_serving_pieces():
    with pytest.raises(ValueError, match="requires a CUDA device"):
        stub_predictor()._configure_host_pipeline()
    predictor = stub_predictor("cuda")
    assert predictor._configure_host_pipeline() == {
        "text_input_cache": {"enabled": True, "max_entries": 64}, "gpu_image_preprocessing": True}
    assert predictor.text_cache_stats() == {"entries": 0, "max_entries": 64, "hits": 0, "misses": 0}
    assert predictor._device_images is not None


def test_text_cache_bypasses_pretokenized_and_positional_calls():
    processor, cache = Processor(), TextInputCache()
    state = RequestImages(processor, text_cache=cache)
    # Pre-tokenized (unhashable) text runs the plain processor instead of failing on the cache key.
    state.processor(text=[["tok", "ens"]], images=[1.0], padding=True, return_tensors="pt")
    assert cache.stats()["entries"] == 0 and processor.text_calls == 1


def test_missing_device_processor_fails_instead_of_selecting_host_preprocessing():
    predictor = stub_predictor("cuda")
    predictor.processor = SimpleNamespace(image_processor=SimpleNamespace())
    with pytest.raises(ValueError, match="torch-backend image processor"):
        predictor._configure_host_pipeline()


BUNDLED_PROCESSOR = Path(os.environ.get("GRANA_QWEN35_PROCESSOR",
                                        "/root/grana_packages/qwen35_inference_examples_20261102/models/0.8B/base"))


def _local_processor_available() -> bool:
    try:  # An unreadable parent (e.g. /root for a non-root user) raises instead of returning False.
        return BUNDLED_PROCESSOR.is_dir()
    except OSError:
        return False


@pytest.mark.skipif(not _local_processor_available(), reason="needs a local Qwen3.5 processor (GRANA_QWEN35_PROCESSOR)")
def test_real_processor_cache_matches_plain_processor_across_roles_and_requests():
    """Request 1: C and D miss; request 2: C hits, D misses with a new focus; request 3: all hit."""
    from transformers import AutoProcessor
    from grana.inference.predictor import _Input
    from grana.modeling.inputs import collate_examples
    from grana.modeling.prompts import stage_progress_prompt, stage_prompt, with_views

    processor = AutoProcessor.from_pretrained(BUNDLED_PROCESSOR)
    views, stages = ("top", "left"), ["reach the cup", "grasp the cup", "place the cup"]
    images = tuple(frames(16, seed=7))

    def example(role, prompt):
        return _Input(role, images, with_views(prompt, views), views, len(stages))

    c = example("C", stage_prompt("Put the cup away", stages))
    d = {focus: example("D", stage_progress_prompt("Put the cup away", stages, focus)) for focus in (0, 2)}
    cache = TextInputCache()
    for examples in ((c, d[0]), (c, d[2]), (c, d[0])):
        state = RequestImages(processor, text_cache=cache)
        for item in examples:
            actual = collate_examples(state.processor, [item], 76800, None)
            expected = collate_examples(processor, [item], 76800, "cpu")
            assert set(actual) == set(expected)
            assert all(torch.equal(actual[key].cpu(), expected[key]) for key in expected), item.role
        assert state.image_processor.output is not None
    assert cache.stats()["hits"] == 3 and cache.stats()["misses"] == 3
