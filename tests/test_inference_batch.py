# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from contextvars import ContextVar
from dataclasses import replace
import json
import re
from types import MethodType

import numpy as np
import pytest
import torch

from grana.inference.predictor import GranaPredictor, InferenceRequest
from grana.modeling.inputs import collate_examples


class Images:
    size = {"shortest_edge": 65536, "longest_edge": 16777216}

    def __call__(self, images, **kwargs):
        return {"pixel_values": torch.tensor([float(np.asarray(image)[0, 0, 0]) for image in images]),
                "image_grid_thw": torch.tensor([[1, 2, 2]] * len(images))}


class Processor:
    def __init__(self):
        self.image_processor = Images()

    def apply_chat_template(self, messages, **kwargs):
        return messages[0]["content"][0]["text"]

    def __call__(self, *, text, images, **kwargs):
        return {"input_ids": torch.zeros((len(text), 8), dtype=torch.long),
                **self.image_processor(images=images, return_tensors="pt")}


@pytest.fixture
def predictor():
    result = GranaPredictor.__new__(GranaPredictor)
    result.processor, result._device_images, result._text_cache = Processor(), None, None
    result._request_images = ContextVar("batch-test-images", default=None)
    result.device = torch.device("cpu")
    result.view_names, result.view_policy = ("top", "left"), "checkpoint"
    result.max_pixels, result.enforce_pixel_budget = None, True
    result.checkpoint = {"weights_sha256": "a" * 64}
    result.image_preprocessing = {"resize_backend": "torchvision_cuda"}
    result.calls, result.fail_role = [], None

    def forward(self, examples):
        cached = self._request_images.get()
        inputs = collate_examples(cached.processor, examples, None, None)
        moved = cached.to_device(inputs, "cpu")
        self.calls.append({"roles": [e.role for e in examples], "examples": examples,
                           "pixels": moved["pixel_values"], "cache": cached})
        if self.fail_role == examples[0].role:
            raise torch.OutOfMemoryError("synthetic OOM")
        if examples[0].role == "C":
            logits = torch.full((len(examples), 8, 26), -torch.inf)
            for i, example in enumerate(examples):
                identity = int(np.asarray(example.images[0])[0, 0, 0])
                logits[i, :, :example.num_stages] = -4.
                for slot in range(8):
                    logits[i, slot, (identity + slot) % example.num_stages] = 4.
            return {"stage_logits": logits, "progress_logits": None}
        logits = torch.zeros((len(examples), 8, 10))
        for i, example in enumerate(examples):
            focus = re.search(r"Focus stage: Stage (\d+):", example.prompt)
            value = int(focus.group(1)) if focus else 0
            logits[i, :, min(value + 2, 9)] = 3.
        return {"progress_logits": logits, "stage_logits": None}

    result._forward = MethodType(forward, result)
    return result


def request(identity, mode="poi", stages=3):
    return InferenceRequest(
        images={view: [np.full((2, 3, 3), identity + 8 * camera + slot, dtype=np.uint8)
                       for slot in range(8)] for camera, view in enumerate(("top", "left"))},
        task="Assemble", compressed_task_prompt="Complete assembly",
        stage_prompts=[f"Stage description {i}" for i in range(stages)], focus_stage=0,
        prediction_frame=mode, frame_indices=tuple(range(20, 28)))


def numerical(result):
    return {key: value for key, value in result.items() if key != "image_reuse"}


def test_batch_really_batches_roles_reuses_identical_images_and_routes_each_focus(predictor):
    requests = [request(0, "poi"), request(1, "last"), request(2, "poi", stages=4)]
    scored = predictor.predict_batch(requests, roles=("A", "B", "C", "D"))
    assert [call["roles"] for call in predictor.calls] == [[role] * 3 for role in ("A", "B", "C", "D")]
    assert all(call["pixels"] is predictor.calls[0]["pixels"] for call in predictor.calls)
    assert all(call["cache"] is predictor.calls[0]["cache"] for call in predictor.calls)
    for index, (sample, result) in enumerate(zip(requests, scored, strict=True)):
        expected_focus = (index + sample.poi_slot) % len(sample.stage_prompts)
        assert result["D_focus_stage"] == result["stage"]["ids"][sample.poi_slot] == expected_focus
        assert result["selected_prediction"]["focus_stage"]["D"] == expected_focus
        assert result["selected_prediction"]["frame_index"] == sample.frame_indices[sample.poi_slot]
        assert result["progress"]["D"]["focus_stage"] == expected_focus
        assert len(result["stage"]["ids"]) == len(result["progress"]["D"]["values"]) == 8
        assert result["image_reuse"]["scope"] == "one_prediction_batch"
        assert result["image_reuse"]["requests"] == 3 and result["image_reuse"]["batch_index"] == index
        assert result["image_reuse"]["image_processor_calls"] == 1
        assert f"Focus stage: Stage {expected_focus}:" in predictor.calls[-1]["examples"][index].prompt
    assert predictor._request_images.get() is None


@pytest.mark.parametrize("roles", [("A",), ("B",), ("C",), ("D",), ("C", "D"), ("A", "B", "C", "D")])
def test_batch_results_equal_individual_reference_and_are_isolated(predictor, roles):
    requests = [request(0, "poi"), request(1, "last"), request(2, "poi")]
    expected = [predictor.predict(sample, roles=roles) for sample in requests]
    actual = predictor.predict_batch(requests, roles=roles)
    assert [numerical(row) for row in actual] == [numerical(row) for row in expected]
    actual[0]["checkpoint"]["weights_sha256"] = "changed"
    assert actual[1]["checkpoint"]["weights_sha256"] == "a" * 64
    assert predictor.checkpoint["weights_sha256"] == "a" * 64


def test_batch_limit_chunks_and_single_remainder_uses_existing_predict(predictor):
    samples = [request(i, "poi" if i % 2 else "last") for i in range(5)]
    results = predictor.predict_batch(samples, batch_size=2)
    assert [len(call["roles"]) for call in predictor.calls] == [2, 2, 2, 2, 1, 1]
    assert [result["poi_slot"] for result in results] == [sample.poi_slot for sample in samples]
    assert results[-1]["image_reuse"]["scope"] == "one_prediction"
    predictor.calls.clear()
    results = predictor.predict_batch(samples, batch_size=1)
    assert len(predictor.calls) == 10 and all(len(call["roles"]) == 1 for call in predictor.calls)
    assert all(row["image_reuse"]["scope"] == "one_prediction" for row in results)


def test_batch_override_preserves_inputs_and_original_eight_frames(predictor):
    samples = [request(0), request(1)]
    outputs = predictor.predict_batch(samples, prediction_frame="last")
    assert all(sample.poi_slot == 5 for sample in samples)
    assert all(output["poi_slot"] == 7 for output in outputs)
    assert all(output["frame_indices"] == list(range(20, 28)) for output in outputs)


@pytest.mark.parametrize("batch_size", [0, -1, True, 1.0, "2"])
def test_invalid_batch_size_fails_before_forward(predictor, batch_size):
    with pytest.raises(ValueError, match="batch_size"):
        predictor.predict_batch([request(0)], batch_size=batch_size)
    assert not predictor.calls


@pytest.mark.parametrize("samples", [[], [object()], "requests", iter([])])
def test_invalid_requests_fail_before_forward(predictor, samples):
    with pytest.raises(ValueError, match="requests"):
        predictor.predict_batch(samples)
    assert not predictor.calls


def test_invalid_later_row_does_not_score_earlier_rows_in_physical_batch(predictor):
    with pytest.raises(ValueError, match="task"):
        predictor.predict_batch([request(0), replace(request(1), task=None)])
    assert not predictor.calls and predictor._request_images.get() is None


def test_oom_releases_context_and_propagates_for_caller_backoff(predictor):
    predictor.fail_role = "D"
    with pytest.raises(torch.OutOfMemoryError):
        predictor.predict_batch([request(0), request(1)])
    assert predictor._request_images.get() is None
    predictor.fail_role = None
    outputs = predictor.predict_batch([request(0), request(1)], batch_size=1)
    assert len(outputs) == 2
    json.dumps(outputs, allow_nan=False)
