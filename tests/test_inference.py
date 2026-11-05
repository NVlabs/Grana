# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
from dataclasses import replace
import json
import math
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch
from torch import nn

from grana.inference import predictor as inference
from grana.modeling.checkpoint import save_weights
from grana.config import DEFAULTS
from grana.modeling.prompts import PROGRESS_MARKER, STAGE_MARKER, progress_prompt, stage_progress_prompt, stage_prompt, with_views
from grana.data.sampling import RoleExample


VIEWS = ("top", "left", "right", "bottom")
STAGES = ("Open the drawer", "Place the spatula", "Close the drawer")


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(16, 4)
        self.output = nn.Linear(4, 16, bias=False)
        self.embedding.requires_grad_(False)
        self.output.requires_grad_(False)
        self.adapter = nn.Parameter(torch.full((4,), -3.0))

    def get_input_embeddings(self):
        return self.embedding

    def get_output_embeddings(self):
        return self.output


class TinyPredictorModel(nn.Module):
    """A real tiny checkpoint, with diagnostic predictions at different slots."""
    def __init__(self):
        super().__init__()
        self.backbone = TinyBackbone()
        self.progress_head = nn.Linear(4, 10)
        self.temporal_head = nn.Linear(4, 4)
        self.token_ids = {token: i + 8 for i, token in enumerate(
            ("<|reward_token|>", "<|sim_token|>", PROGRESS_MARKER, STAGE_MARKER))}
        self.model_spec = {"name_or_path": "fixture/base", "hidden_size": 4,
                           "lora_rank": 2, "lora_alpha": 4, "lora_dropout": .05,
                           "token_ids": self.token_ids}
        self.calls = []
        self.nonfinite_role = None

    def forward(self, inputs, roles, stage_counts):
        assert not self.training and not torch.is_grad_enabled()
        self.calls.append({"roles": tuple(roles), "stage_counts": tuple(stage_counts)})
        progress_indices = [i for i, role in enumerate(roles) if role != "C"]
        stage_indices = [i for i, role in enumerate(roles) if role == "C"]
        stage_logits = progress_logits = None
        if stage_indices:
            stage_logits = torch.full((len(stage_indices), 8, 26), -torch.inf)
            for index, row in enumerate(stage_indices):
                count = stage_counts[row]
                stage_logits[index, :, :count] = -3.
                stage_logits[index, :, 0] = 3.
                stage_logits[index, 5, 0] = -3.
                stage_logits[index, 5, min(2, count - 1)] = 3.
            if self.nonfinite_role == "C":
                stage_logits[:, 5, 0] = torch.nan
        if progress_indices:
            progress_logits = torch.zeros(len(progress_indices), 8, 10)
            progress_logits[:, 7, 9] = math.log(9)
            for index, row in enumerate(progress_indices):
                if self.nonfinite_role == roles[row]:
                    progress_logits[index, 5, 0] = torch.nan
        return {"progress_logits": progress_logits, "stage_logits": stage_logits,
                "progress_indices": progress_indices, "stage_indices": stage_indices}


class SpyProcessor:
    def __init__(self):
        self.image_processor = SimpleNamespace(
            size={"shortest_edge": 65536, "longest_edge": 16777216}, patch_size=16)
        self.messages = []
        self.batches = []
        self.grid_side = 2

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is False and add_generation_prompt is True
        self.messages.append(messages)
        return f"request-{len(self.messages)}"

    def __call__(self, *, text, images, padding, return_tensors, **kwargs):
        assert padding is True and return_tensors == "pt"
        self.batches.append({"images": images, "image_kwargs": kwargs})
        return {"input_ids": torch.zeros(len(text), 8, dtype=torch.long),
                "image_grid_thw": torch.tensor([[1, self.grid_side, self.grid_side]] * len(images))}


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    config = copy.deepcopy(DEFAULTS)
    config["model"].update(name_or_path="fixture/base", lora_rank=2, lora_alpha=4)
    config["data"]["views"] = list(VIEWS)
    config["data"]["caches"] = ["/unavailable/training/cache"]
    config["data"]["manifest"] = "/unavailable/training/manifest.yaml"
    config["training"].update(stage="posttrain", init_checkpoint="/unavailable/old/checkpoint")
    source = TinyPredictorModel()
    with torch.no_grad():
        for parameter in source.parameters():
            parameter.fill_(7. if parameter.requires_grad else 11.)
    checkpoint = tmp_path / "checkpoint-000012"
    save_weights(source, checkpoint, config, 12)
    made = []
    state = SimpleNamespace(checkpoint=checkpoint, source=source, made=made, backend="fla")

    def build_model(cfg):
        model, processor = TinyPredictorModel(), SpyProcessor()
        model.model_spec["delta_rule_backend"] = {"requested": "auto", "resolved": state.backend}
        processor.image_processor.rescale_and_normalize = lambda *args, **kwargs: None
        monkeypatch.setattr(model, "to", lambda *args, **kwargs: model)
        made.append(SimpleNamespace(model=model, processor=processor, config=copy.deepcopy(cfg)))
        return model, processor

    def cpu_forward(self, examples):
        # Keep real collation/prompt and checkpoint logic on CPU. CUDA kernels
        # and full-stack setup have their own targeted tests and GPU receipts.
        inputs = inference.collate_examples(self.processor, examples, self.max_pixels, "cpu",
                                           enforce_pixel_budget=self.enforce_pixel_budget)
        with torch.inference_mode():
            return self.model(inputs, [e.role for e in examples], [e.num_stages for e in examples])

    def fixed_stack(self):
        return {"batched_vision_sdpa": True, "reuse_images": True, "merge_lora": True,
                "cpu_position_ids": True, "compile_decoder": True, "compile_vision": True,
                "causal_conv1d": {"active": True}, "fla_output_guard": {"applied": True}}

    previous_threads = torch.get_num_threads()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(inference, "build_model", build_model)
    monkeypatch.setattr(inference.GranaPredictor, "_configure_inference_stack", fixed_stack)
    monkeypatch.setattr(inference.GranaPredictor, "_forward", cpu_forward)
    yield state
    torch.set_num_threads(previous_threads)


def request(**changes):
    images = {view: [np.full((2, 3, 3), 16 * camera + slot, dtype=np.uint8)
                     for slot in range(8)] for camera, view in enumerate(VIEWS)}
    return inference.InferenceRequest(images=images, task="put_spatula_in_drawer",
                                      compressed_task_prompt="Put the spatula in the drawer.",
                                      stage_prompts=STAGES, focus_stage=0, **changes)


def make_predictor(runtime, **kwargs):
    predictor = inference.GranaPredictor(runtime.checkpoint, device="cuda:0", **kwargs)
    return predictor, runtime.made[-1]


@pytest.mark.parametrize("slot", [0, 2, 7])
def test_explicit_poi_routes_c_to_d_and_progress_without_reordering(runtime, slot):
    predictor, made = make_predictor(runtime)
    result = predictor.predict(request(poi_slot=slot), roles=("A", "B", "C", "D"))
    assert result["poi_slot"] == slot
    assert result["prediction_frame"] == ("last" if slot == 7 else "custom")
    assert result["selected_prediction"]["slot"] == slot
    assert result["stage"]["poi_stage_id"] == result["stage"]["ids"][slot] == 0
    assert result["D_focus_stage"] == result["progress"]["D"]["focus_stage"] == 0
    for role in ("A", "B", "D"):
        assert result["progress"][role]["poi_score"] == result["progress"][role]["values"][slot]
    batches = made.processor.batches
    assert all(batch["images"] == batches[0]["images"] for batch in batches)


@pytest.mark.parametrize("slot", [True, False, -1, 8, 2.0, "2", None])
def test_invalid_query_slot_is_rejected_before_model_allocation(slot):
    with pytest.raises(ValueError, match="poi_slot"):
        request(poi_slot=slot)


@pytest.mark.parametrize("mode", ["first", "custom", "", False, 7, [], {}])
def test_invalid_named_frame_is_rejected_before_model_allocation(mode):
    with pytest.raises(ValueError, match="prediction_frame"):
        request(prediction_frame=mode)


@pytest.mark.parametrize("mode,slot", [("poi", 7), ("last", 5), ("last", 2)])
def test_named_frame_and_explicit_slot_must_agree(mode, slot):
    with pytest.raises(ValueError, match="conflicts"):
        request(prediction_frame=mode, poi_slot=slot)


@pytest.mark.parametrize("indices", [[0] * 7, [0] * 9, [True] * 8, [-1] * 8, [1.0] * 8, "01234567"])
def test_invalid_source_frame_indices_are_rejected(indices):
    with pytest.raises(ValueError, match="frame_indices"):
        request(frame_indices=indices)


@pytest.mark.parametrize("roles,forwards", [(("A",), [("A",)]), (("C",), [("C",)]),
                                          (("C", "D"), [("C",), ("D",)])])
@pytest.mark.parametrize("mode,slot,focus", [("poi", 5, 2), ("last", 7, 0)])
def test_role_and_frame_selection_preserve_all_eight_outputs(runtime, roles, forwards, mode, slot, focus):
    predictor, made = make_predictor(runtime)
    indices = (10, 11, 12, 1, 5, 9, 13, 13)
    sample = request(prediction_frame=mode, frame_indices=indices)
    result = predictor.predict(sample, roles=roles)
    assert [call["roles"] for call in made.model.calls] == forwards
    assert result["poi_slot"] == sample.poi_slot == slot
    assert result["prediction_frame"] == mode
    assert result["frame_indices"] == list(indices)
    selected = result["selected_prediction"]
    assert selected["slot"] == slot and selected["frame_index"] == indices[slot]
    for role, output in result["progress"].items():
        assert len(output["values"]) == len(output["bin_probabilities"]) == 8
        assert selected["progress"][role] == output["values"][slot] == output["poi_score"]
    if "C" in roles:
        assert all(len(result["stage"][key]) == 8 for key in ("ids", "logits", "probabilities"))
        assert selected["stage_id"] == result["stage"]["ids"][slot] == focus
    else:
        assert "stage" not in result and "stage_id" not in selected
    if "D" in roles:
        assert result["D_focus_stage"] == result["progress"]["D"]["focus_stage"] == focus
        assert selected["focus_stage"]["D"] == focus
        d_prompt = made.processor.messages[-1][0]["content"][0]["text"]
        assert f"Focus stage: Stage {focus}:" in d_prompt
    # Selection changes output routing only, never slot/camera composition.
    for batch in made.processor.batches:
        assert [int(np.asarray(image)[0, 0, 0]) for image in batch["images"]] == [
            16 * camera + index for index in range(8) for camera in range(len(VIEWS))]


def test_predict_named_override_does_not_mutate_the_request(runtime):
    predictor, _ = make_predictor(runtime)
    sample = request(prediction_frame="poi")
    result = predictor.predict(sample, prediction_frame="last")
    assert sample.poi_slot == 5 and sample.prediction_frame == "poi"
    assert result["poi_slot"] == 7 and result["D_focus_stage"] == 0
    assert result["stage"]["ids"][5] == 2


@pytest.mark.parametrize("backend", [None, "auto", "torch"])
def test_inference_requires_fla_sdpa_without_rewriting_training_backend(runtime, backend):
    path = runtime.checkpoint / "metadata.json"
    metadata = json.loads(path.read_text())
    if backend is None:
        metadata["config"]["model"].pop("delta_rule_backend", None)
    else:
        metadata["config"]["model"]["delta_rule_backend"] = backend
    path.write_text(json.dumps(metadata))
    original = path.read_bytes()
    predictor, made = make_predictor(runtime)
    assert made.config["model"]["delta_rule_backend"] == "auto"
    assert made.config["model"]["attn_implementation"] == "sdpa"
    assert predictor.inference_runtime["delta_rule_backend"]["resolved"] == "fla"
    assert made.config["model"]["gradient_checkpointing"] is False
    assert predictor.predict(request(), roles=("A",))["checkpoint"]["source_step"] == 12
    assert path.read_bytes() == original


REMOVED_RUNTIME_OPTIONS = (
    "optimize", "batched_vision_sdpa", "reuse_images", "merge_lora", "cpu_position_ids",
    "compile_decoder", "compile_vision", "causal_conv1d", "fla_output_guard",
    "delta_rule_backend", "attn_implementation", "text_input_cache", "gpu_image_preprocessing",
)


@pytest.mark.parametrize("name", REMOVED_RUNTIME_OPTIONS)
@pytest.mark.parametrize("value", [False, True, None])
def test_removed_runtime_options_fail_before_model_load(runtime, name, value):
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        make_predictor(runtime, **{name: value})
    assert not runtime.made


@pytest.mark.parametrize("device", ["cpu", "mps", "meta"])
def test_non_cuda_inference_fails_before_loading(runtime, device):
    with pytest.raises(ValueError, match="CUDA device"):
        inference.GranaPredictor(runtime.checkpoint, device=device)
    assert not runtime.made


def test_cuda_unavailable_never_selects_cpu(runtime, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="requires CUDA"):
        inference.GranaPredictor(runtime.checkpoint)
    assert not runtime.made


@pytest.mark.parametrize("backend", ["torch", "mixed", "none"])
def test_missing_fla_fails_instead_of_running_torch(runtime, backend):
    runtime.backend = backend
    with pytest.raises(RuntimeError, match="requires FLA"):
        make_predictor(runtime)
    assert len(runtime.made) == 1 and not runtime.made[0].model.calls


def test_runtime_records_one_required_stack_and_fixed_cpu_budget(runtime):
    torch.set_num_threads(2)
    predictor, _ = make_predictor(runtime)
    settings = predictor.inference_runtime
    assert settings["policy"] == "cuda_optimized_v1"
    assert settings["cpu_threads"] == torch.get_num_threads() == 8
    assert settings["fallback_policy"] == "error"
    assert settings["attn_implementation"] == "sdpa"
    for name in ("batched_vision_sdpa", "reuse_images", "merge_lora", "cpu_position_ids",
                 "compile_decoder", "compile_vision", "gpu_image_preprocessing", "gpu_image_resize"):
        assert settings[name] is True
    assert settings["text_input_cache"] == {"enabled": True, "max_entries": 64}
    assert settings["causal_conv1d"]["active"] is True
    assert settings["fla_output_guard"]["applied"] is True
    result = predictor.predict(request(), roles=("A",))
    assert result["checkpoint"]["inference_runtime"] == settings
    json.dumps(settings, allow_nan=False)


def rewrite_preprocessing(runtime, budget, *, legacy=False, spec=None):
    path = runtime.checkpoint / "metadata.json"
    metadata = json.loads(path.read_text())
    metadata["config"]["data"]["max_pixels"] = budget
    if legacy:
        metadata.pop("image_preprocessing", None)
    else:
        metadata["image_preprocessing"] = spec if spec is not None else {
            "version": 1, "policy": "processor_default" if budget is None else "pixel_budget",
            "max_pixels": budget}
    path.write_text(json.dumps(metadata))
    return path.read_bytes()


@pytest.mark.parametrize("budget", [None, 16384, 76800])
def test_versioned_preprocessing_inherits_default_or_enforces_numeric_budget(runtime, budget):
    original = rewrite_preprocessing(runtime, budget)
    predictor, made = make_predictor(runtime)
    result = predictor.predict(request(), roles=("A",))
    spec = result["image_preprocessing"]
    assert predictor.max_pixels == result["max_pixels"] == budget
    assert spec["source"] == "checkpoint" and spec["legacy_checkpoint"] is False
    assert spec["legacy_requested_max_pixels"] is None
    assert spec["enforce_pixel_budget"] is (budget is not None)
    expected = {} if budget is None else {"min_pixels": min(65536, budget), "max_pixels": budget}
    assert made.processor.batches[-1]["image_kwargs"] == expected
    assert spec["effective_size"] == {"shortest_edge": 65536 if budget is None else min(65536, budget),
                                       "longest_edge": 16777216 if budget is None else budget}
    assert (runtime.checkpoint / "metadata.json").read_bytes() == original


@pytest.mark.parametrize("budget", [65536, 76800, 1048576])
def test_legacy_at_or_above_default_minimum_keeps_processor_defaults(runtime, budget):
    original = rewrite_preprocessing(runtime, budget, legacy=True)
    predictor, made = make_predictor(runtime)
    result = predictor.predict(request(), roles=("A",))
    assert predictor.max_pixels is None and result["max_pixels"] is None
    assert made.processor.batches[-1]["image_kwargs"] == {}
    assert result["image_preprocessing"] == {
        "version": 2, "policy": "processor_default", "max_pixels": None,
        "resize_backend": "torchvision_cuda", "effective_size": {"shortest_edge": 65536, "longest_edge": 16777216},
        "source": "legacy_checkpoint", "legacy_checkpoint": True,
        "legacy_requested_max_pixels": budget, "enforce_pixel_budget": False}
    assert (runtime.checkpoint / "metadata.json").read_bytes() == original


def test_legacy_smallcap_retains_old_rounding_but_explicit_override_enforces(runtime):
    original = rewrite_preprocessing(runtime, 16384, legacy=True)
    predictor, made = make_predictor(runtime)
    made.processor.grid_side = 16  # Historical rounding may exceed the requested small cap.
    result = predictor.predict(request(), roles=("A",))
    assert made.processor.batches[-1]["image_kwargs"] == {"min_pixels": 16384, "max_pixels": 16384}
    assert result["image_preprocessing"]["policy"] == "legacy_smallcap"
    assert result["image_preprocessing"]["enforce_pixel_budget"] is False
    assert predictor.max_pixels == 16384
    explicit, made_explicit = make_predictor(runtime, max_pixels=16384)
    made_explicit.processor.grid_side = 16
    assert explicit.image_preprocessing["source"] == "explicit_override"
    assert explicit.image_preprocessing["policy"] == "pixel_budget"
    assert explicit.image_preprocessing["enforce_pixel_budget"] is True
    with pytest.raises(ValueError, match="pixel"):
        explicit.predict(request(), roles=("A",))
    assert not made_explicit.model.calls
    assert (runtime.checkpoint / "metadata.json").read_bytes() == original


def test_explicit_numeric_budget_overrides_legacy_ignored_max(runtime):
    rewrite_preprocessing(runtime, 76800, legacy=True)
    predictor, made = make_predictor(runtime, max_pixels=np.int64(76800))
    result = predictor.predict(request(), roles=("A",))
    assert predictor.max_pixels == 76800
    assert made.processor.batches[-1]["image_kwargs"] == {"min_pixels": 65536, "max_pixels": 76800}
    assert result["image_preprocessing"]["source"] == "explicit_override"
    assert result["image_preprocessing"]["legacy_requested_max_pixels"] == 76800


@pytest.mark.parametrize("budget", [0, True, False, 1023, 1024.0, "76800"])
def test_invalid_explicit_budget_rejected_before_model_load(runtime, budget):
    with pytest.raises(ValueError, match="max_pixels"):
        make_predictor(runtime, max_pixels=budget)
    assert not runtime.made


@pytest.mark.parametrize("spec", [
    None, {}, {"version": True, "policy": "processor_default", "max_pixels": None},
    {"version": 2, "policy": "processor_default", "max_pixels": None},
    {"version": 1, "policy": "unknown", "max_pixels": None},
    {"version": 1, "policy": "processor_default", "max_pixels": None, "extra": 1},
    {"version": 1, "policy": "pixel_budget", "max_pixels": True},
    {"version": 1, "policy": "pixel_budget", "max_pixels": 0},
    {"version": 1, "policy": "pixel_budget", "max_pixels": 76800},
])
def test_invalid_or_contradictory_descriptor_rejected_even_with_explicit_override(runtime, spec):
    path = runtime.checkpoint / "metadata.json"
    metadata = json.loads(path.read_text())
    metadata["config"]["data"]["max_pixels"] = None
    metadata["image_preprocessing"] = spec
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        make_predictor(runtime, max_pixels=76800)
    assert not runtime.made


def gpu_resize_checkpoint(runtime):
    path = runtime.checkpoint / "metadata.json"
    metadata = json.loads(path.read_text())
    metadata["config"]["data"]["gpu_image_resize"] = True
    metadata["image_preprocessing"] = inference.preprocessing_spec(None, gpu_image_resize=True)
    path.write_text(json.dumps(metadata))
    return path.read_bytes()


@pytest.mark.parametrize("legacy", [False, True])
def test_host_trained_checkpoint_still_uses_required_cuda_resize(runtime, legacy):
    original = rewrite_preprocessing(runtime, 81920, legacy=legacy)
    predictor, _ = make_predictor(runtime)
    assert predictor.inference_runtime["gpu_image_preprocessing"] is True
    assert predictor.inference_runtime["gpu_image_resize"] is True
    assert predictor.inference_runtime["checkpoint_gpu_image_resize"] is False
    assert predictor.inference_runtime["resize_matches_checkpoint"] is False
    assert predictor.inference_runtime["gpu_image_resize_source"] == "required_runtime"
    assert predictor.image_preprocessing["resize_backend"] == "torchvision_cuda"
    assert predictor.image_preprocessing["legacy_checkpoint"] is legacy
    assert (runtime.checkpoint / "metadata.json").read_bytes() == original


def test_gpu_trained_checkpoint_preprocessing_is_aligned(runtime):
    original = gpu_resize_checkpoint(runtime)
    predictor, _ = make_predictor(runtime)
    assert predictor.inference_runtime["gpu_image_resize"] is True
    assert predictor.inference_runtime["checkpoint_gpu_image_resize"] is True
    assert predictor.inference_runtime["resize_matches_checkpoint"] is True
    assert predictor.image_preprocessing["resize_backend"] == "torchvision_cuda"
    assert (runtime.checkpoint / "metadata.json").read_bytes() == original


@pytest.mark.parametrize("change", ["wrong_backend", "wrong_config", "missing_descriptor"])
def test_gpu_resize_descriptor_contradictions_fail_before_loading(runtime, change):
    gpu_resize_checkpoint(runtime)
    path = runtime.checkpoint / "metadata.json"
    metadata = json.loads(path.read_text())
    if change == "wrong_backend":
        metadata["image_preprocessing"]["resize_backend"] = "cpu"
    elif change == "wrong_config":
        metadata["config"]["data"]["gpu_image_resize"] = False
    else:
        del metadata["image_preprocessing"]
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="image_preprocessing"):
        make_predictor(runtime)
    assert not runtime.made


@pytest.mark.parametrize("gpu_resize", [False, True])
def test_descriptor_budget_type_is_validated_even_when_numerically_equal(runtime, gpu_resize):
    path = runtime.checkpoint / "metadata.json"
    metadata = json.loads(path.read_text())
    metadata["config"]["data"].update(max_pixels=81920, gpu_image_resize=gpu_resize)
    metadata["image_preprocessing"] = inference.preprocessing_spec(81920, gpu_image_resize=gpu_resize)
    metadata["image_preprocessing"]["max_pixels"] = 81920.0
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="max_pixels"):
        make_predictor(runtime)
    assert not runtime.made


def text_and_pixels(messages):
    """Compare serialized content independently of PIL object identity."""
    return [(item["type"], item["text"] if item["type"] == "text"
             else (item["image"].mode, item["image"].size, item["image"].tobytes()))
            for item in messages[0]["content"]]


def test_real_checkpoint_load_precedes_freeze_and_needs_no_training_files(runtime):
    predictor, made = make_predictor(runtime, model_name_or_path="relocated/base")
    assert made.config["model"]["name_or_path"] == "relocated/base"
    assert made.config["model"]["gradient_checkpointing"] is False
    assert not made.model.training
    # The real loader selects adapters by requires_grad: a premature freeze
    # would leave this adapter at its constructor value (-3), not saved value7.
    torch.testing.assert_close(made.model.backbone.adapter, runtime.source.backbone.adapter)
    for head in ("progress_head", "temporal_head"):
        for key, value in getattr(runtime.source, head).state_dict().items():
            torch.testing.assert_close(getattr(made.model, head).state_dict()[key], value)
    ids = torch.tensor(list(runtime.source.token_ids.values()))
    for accessor in ("get_input_embeddings", "get_output_embeddings"):
        torch.testing.assert_close(getattr(made.model.backbone, accessor)().weight[ids],
                                   getattr(runtime.source.backbone, accessor)().weight[ids])
    result = predictor.predict(replace(request(), focus_stage=None), roles=("A",))
    assert result["checkpoint"]["load_report"]["adapter_tensor_count"] == 1
    assert all(value["outcome"] == "loaded" for value in result["checkpoint"]["load_report"]["heads"].values())


def test_label_free_abcd_matches_training_prompts_and_serialization(runtime):
    predictor, made = make_predictor(runtime)
    sample = request()
    result = predictor.predict(sample, roles=("A", "B", "C", "D"))
    assert [role for call in made.model.calls for role in call["roles"]] == list("ABCD")
    assert result["poi_slot"] == 5 and result["D_focus_stage"] == 2
    images = tuple(Image.fromarray(sample.images[view][slot]) for slot in range(8) for view in VIEWS)
    for role, observed in zip("ABCD", made.processor.messages, strict=True):
        focus = 0 if role == "B" else 2 if role == "D" else None
        prompt = progress_prompt(sample.compressed_task_prompt) if role == "A" else (
            stage_prompt(sample.task, STAGES) if role == "C" else stage_progress_prompt(sample.task, STAGES, focus))
        training_example = RoleExample(role=role, images=images, frame_indices=tuple(range(8)),
            targets=(.1,) * 8, num_stages=3, task=sample.task, trajectory_id="training-reference",
            prompt=with_views(prompt, VIEWS), view_names=VIEWS, stage_prompts=STAGES,
            stage_ids=(1,) * 8, within_stage_progress=(.7,) * 8, stride_pair=(1, 1), focus_stage=focus)
        assert text_and_pixels(observed) == text_and_pixels(training_example.messages())
        marker = STAGE_MARKER if role == "C" else PROGRESS_MARKER
        groups = observed[0]["content"][1:]
        assert len(groups) == 8 * 9
        for slot in range(8):
            group = groups[slot * 9:(slot + 1) * 9]
            assert group[-1] == {"type": "text", "text": marker}
            assert sum(item["type"] == "image" for item in group) == 4
    assert "Focus stage:" not in made.processor.messages[2][0]["content"][0]["text"]
    c_images, d_images = ([item["image"] for item in made.processor.messages[i][0]["content"]
                          if item["type"] == "image"] for i in (2, 3))
    assert all(c is d for c, d in zip(c_images, d_images, strict=True))
    assert len(c_images) == 32
    assert [np.asarray(image)[0, 0, 0] for image in c_images] == [16 * camera + slot for slot in range(8) for camera in range(4)]


def test_poi_readout_is_slot_five_and_progress_is_distribution_expectation(runtime):
    predictor, _ = make_predictor(runtime)
    result = predictor.predict(replace(request(), focus_stage=None), roles=("A", "C", "D"))
    for role in ("A", "D"):
        assert result["progress"][role]["poi_score"] == pytest.approx(.5)
        assert result["progress"][role]["values"][5] == pytest.approx(.5)
        assert result["progress"][role]["values"][7] > .7
    assert result["stage"]["ids"][5] == 2 and result["stage"]["ids"][7] == 0
    assert result["stage"]["poi_stage_id"] == result["D_focus_stage"] == 2
    assert np.asarray(result["stage"]["probabilities"]).shape == (8, 3)
    assert np.asarray(result["stage"]["logits"]).shape == (8, 3)
    np.testing.assert_allclose(np.asarray(result["stage"]["probabilities"]).sum(-1), 1, atol=1e-6)
    json.dumps(result, allow_nan=False)


def test_d_only_computes_native_c_without_gt_or_supplied_focus(runtime):
    predictor, made = make_predictor(runtime)
    sample = replace(request(), compressed_task_prompt=None, focus_stage=None)
    result = predictor.predict(sample, roles=("D",))
    assert [call["roles"] for call in made.model.calls] == [("C",), ("D",)]
    assert set(result["progress"]) == {"D"}
    assert result["D_focus_stage"] == 2


def test_default_prediction_uses_only_stage_and_predicted_focus_forwards(runtime):
    predictor, made = make_predictor(runtime)
    sample = replace(request(), compressed_task_prompt=None, focus_stage=None)
    result = predictor.predict(sample)
    assert result["roles"] == ["C", "D"]
    assert [call["roles"] for call in made.model.calls] == [("C",), ("D",)]
    assert set(result["progress"]) == {"D"}
    assert result["stage"]["poi_stage_id"] == result["D_focus_stage"]


def test_a_only_needs_no_task_stages_focus_or_ground_truth(runtime):
    predictor, made = make_predictor(runtime)
    sample = inference.InferenceRequest(images=request().images,
                                        compressed_task_prompt="Put the spatula away.")
    result = predictor.predict(sample, roles=("A",))
    assert [call["roles"] for call in made.model.calls] == [("A",)]
    assert set(result["progress"]) == {"A"} and "stage" not in result


def test_b_only_uses_explicit_focus_without_a_stage_prediction_call(runtime):
    predictor, made = make_predictor(runtime)
    sample = replace(request(), compressed_task_prompt=None, focus_stage=1)
    result = predictor.predict(sample, roles=("B",))
    assert [call["roles"] for call in made.model.calls] == [("B",)]
    assert result["progress"]["B"]["focus_stage"] == 1 and "stage" not in result
    assert "Focus stage: Stage 1: Place the spatula" in made.processor.messages[0][0]["content"][0]["text"]


def test_singleton_stage_returns_finite_probability_one_and_routes_d_to_zero(runtime):
    predictor, _ = make_predictor(runtime)
    result = predictor.predict(replace(request(), stage_prompts=("Only operation",), focus_stage=None), roles=("C", "D"))
    assert result["stage"]["ids"] == [0] * 8 and result["D_focus_stage"] == 0
    assert result["stage"]["probabilities"] == [[1.]] * 8
    assert np.asarray(result["stage"]["logits"]).shape == (8, 1)
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("role", ["A", "C"])
def test_nonfinite_predictions_fail_instead_of_writing_json_nan(runtime, role):
    predictor, made = make_predictor(runtime)
    made.model.nonfinite_role = role
    with pytest.raises((ValueError, FloatingPointError), match="[Ff]inite|NaN"):
        predictor.predict(replace(request(), focus_stage=None), roles=(role,))


@pytest.mark.parametrize("changes,roles", [
    ({"focus_stage": None}, ("B",)),
    ({"focus_stage": True}, ("B",)),
    ({"focus_stage": 3}, ("B",)),
    ({"compressed_task_prompt": None}, ("A",)),
    ({"task": None}, ("C",)),
    ({"stage_prompts": ()}, ("D",)),
])
def test_missing_semantic_inputs_are_not_inferred_from_gt_or_task_id(runtime, changes, roles):
    predictor, made = make_predictor(runtime)
    with pytest.raises(ValueError):
        predictor.predict(replace(request(), **changes), roles=roles)
    assert not made.model.calls


@pytest.mark.parametrize("damage", ["missing_view", "view_order", "seven_frames", "float_pixels"])
def test_wrong_geometry_is_rejected_before_model_forward(runtime, damage):
    predictor, made = make_predictor(runtime)
    sample = request()
    images = dict(sample.images)
    if damage == "missing_view":
        images.pop("bottom")
    elif damage == "view_order":
        images = dict(reversed(list(images.items())))
    elif damage == "seven_frames":
        images["top"] = images["top"][:7]
    else:
        images["top"] = [frame.astype(np.float32) for frame in images["top"]]
    with pytest.raises(ValueError):
        predictor.predict(replace(sample, images=images), roles=("A",))
    assert not made.model.calls


def test_request_view_policy_changes_camera_count_without_reloading_or_mutating_checkpoint(runtime):
    predictor, made = make_predictor(runtime, view_policy="request")
    checkpoint_bytes = (runtime.checkpoint / "metadata.json").read_bytes()
    for count in (1, 2, 3, 4, 5):
        views = tuple(f"camera_{index}" for index in reversed(range(count)))
        images = {view: tuple(np.full((camera + 2, camera + 3, 3), camera * 16 + slot, dtype=np.uint8)
                              for slot in range(8)) for camera, view in enumerate(views)}
        result = predictor.predict(replace(request(), images=images), roles=tuple("ABCD"))
        assert result["view_names"] == list(views)
        assert result["view_policy"] == "request" and result["checkpoint_view_names"] == list(VIEWS)
        assert predictor.view_names == VIEWS and result["poi_slot"] == 5 and result["D_focus_stage"] == 2
        for role, messages in zip("ABCD", made.processor.messages[-4:], strict=True):
            content = messages[0]["content"]
            pixels = [np.asarray(item["image"])[0, 0, 0] for item in content if item["type"] == "image"]
            assert pixels == [16 * camera + slot for slot in range(8) for camera in range(count)]
            marker = STAGE_MARKER if role == "C" else PROGRESS_MARKER
            assert sum(item.get("text") == marker for item in content) == 8
        c_images, d_images = ([item["image"] for item in messages[0]["content"] if item["type"] == "image"]
                              for messages in made.processor.messages[-2:])
        assert all(c is d for c, d in zip(c_images, d_images, strict=True))
    assert len(runtime.made) == 1
    assert (runtime.checkpoint / "metadata.json").read_bytes() == checkpoint_bytes


@pytest.mark.parametrize("name", ["", "   ", " left", "left ", "\tleft\n", "<|stage_token|>", "x" * 129, 1])
def test_request_view_policy_rejects_invalid_camera_names_before_forward(runtime, name):
    predictor, made = make_predictor(runtime, view_policy="request")
    with pytest.raises(ValueError):
        predictor.predict(replace(request(), images={name: request().images["top"]}), roles=("A",))
    assert not made.model.calls


def test_request_view_policy_rejects_whitespace_aliases_before_forward(runtime):
    predictor, made = make_predictor(runtime, view_policy="request")
    frames = request().images["top"]
    with pytest.raises(ValueError, match="whitespace"):
        predictor.predict(replace(request(), images={"left": frames, " left ": frames}), roles=("A",))
    assert not made.model.calls


def test_checkpoint_camera_names_are_not_silently_trimmed(runtime):
    path = runtime.checkpoint / "metadata.json"
    metadata = json.loads(path.read_text())
    metadata["config"]["data"]["views"] = [" top", "bottom"]
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="whitespace"):
        make_predictor(runtime)
    assert not runtime.made


def test_unknown_view_policy_is_rejected_before_model_load(runtime):
    with pytest.raises(ValueError, match="view_policy"):
        make_predictor(runtime, view_policy="automatic")
    assert not runtime.made
