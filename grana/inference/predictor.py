# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Label-free scoring of eight observed RGB frames from a Grana checkpoint."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass, field, replace
import json
from numbers import Integral
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from grana.modeling.checkpoint import FORMAT, load_weights
from grana.modeling.inputs import collate_examples, preprocessing_spec, serialize_messages, validate_max_pixels
from grana.modeling.model import build_model
from grana.modeling.prompts import progress_prompt, stage_progress_prompt, stage_prompt, with_views
from .selection import (_UNSET, prediction_frame_for_slot, resolve_prediction_slot,
                        select_prediction, validate_frame_indices, validate_slot as _poi_slot)

POI_SLOT = 5
VIEW_POLICIES = ("checkpoint", "request")
INFERENCE_CPU_THREADS = 8
INFERENCE_POLICY = "cuda_optimized_v1"


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty text")
    if "<|" in value or "|>" in value:
        raise ValueError(f"{name} cannot contain reserved model markers")
    return value.strip()


@dataclass(frozen=True)
class InferenceRequest:
    """Eight RGB observations per named view, in the supplied camera order.

    A needs its own explicit compressed task prompt. B/C/D need the task and
    ordered stage descriptions; B additionally needs a requested focus stage.
    Images are observations only: this interface accepts no progress/stage GT.
    The predictor's default policy requires checkpoint camera names and order;
    explicit request policy uses this mapping's names and order for every role.
    ``prediction_frame`` selects POI (slot 5) or last (slot 7); omission retains
    slot 5. ``frame_indices`` records source-frame provenance in supplied order.
    Neither selection changes the eight input observations or output arrays.
    """

    images: Mapping[str, Sequence[Image.Image | np.ndarray]]
    task: str | None = None
    compressed_task_prompt: str | None = None
    stage_prompts: Sequence[str] = ()
    focus_stage: int | None = None
    # Distinguish omission from an invalid explicit None and from named last.
    poi_slot: int = field(default=_UNSET)
    prediction_frame: str | None = None
    frame_indices: Sequence[int] | None = None

    def __post_init__(self):
        object.__setattr__(self, "poi_slot", resolve_prediction_slot(self.prediction_frame, self.poi_slot))
        object.__setattr__(self, "frame_indices", validate_frame_indices(self.frame_indices))


@dataclass(frozen=True)
class _Input:
    role: str
    images: tuple[Image.Image, ...]
    prompt: str
    view_names: tuple[str, ...]
    num_stages: int

    def messages(self) -> list[dict]:
        return serialize_messages(self.prompt, self.role, self.images, self.view_names)


def _validate_view_names(views: tuple[str, ...]) -> None:
    if not views:
        raise ValueError("At least one named camera view is required")
    for view in views:
        if _text(view, "view name") != view:
            raise ValueError("View names cannot contain leading or trailing whitespace")
        if len(view) > 128:
            raise ValueError("View names must contain at most 128 characters")
    if len(set(views)) != len(views):
        raise ValueError("Camera view names must be unique")


def _checkpoint_pixel_budget(metadata: dict, data: dict) -> tuple[int | None, bool]:
    """Validate new descriptors before model loading; absence denotes legacy behavior."""
    gpu_resize = data.get("gpu_image_resize", False)
    if type(gpu_resize) is not bool:
        raise ValueError("Checkpoint data.gpu_image_resize must be boolean")
    if "image_preprocessing" not in metadata:
        if gpu_resize:
            raise ValueError("GPU resize checkpoints must record image_preprocessing")
        legacy_budget = data.get("max_pixels", 76800)
        validate_max_pixels(legacy_budget)
        if legacy_budget is None:
            raise ValueError("Legacy checkpoint max_pixels must be an integer")
        return legacy_budget, True
    spec = metadata["image_preprocessing"]
    if (not isinstance(spec, dict) or type(spec.get("version")) is not int
            or spec["version"] not in (1, 2)):
        raise ValueError("Invalid checkpoint image_preprocessing descriptor or version")
    validate_max_pixels(spec.get("max_pixels"))
    expected = preprocessing_spec(data.get("max_pixels"), gpu_image_resize=gpu_resize)
    if spec != expected:
        raise ValueError("Checkpoint image_preprocessing contradicts its config or has an unknown policy")
    return spec["max_pixels"], False


def _observations(request: InferenceRequest, expected_views: tuple[str, ...] | None) -> tuple[Image.Image, ...]:
    if not isinstance(request.images, Mapping) or not request.images:
        raise ValueError("images must map ordered camera names to eight RGB observations")
    views = tuple(request.images)
    _validate_view_names(views)
    if expected_views is not None and views != expected_views:
        raise ValueError(f"Image view names/order must match the checkpoint: {expected_views}; got {views}")
    converted = {}
    for view, frames in request.images.items():
        if not isinstance(frames, (Sequence, np.ndarray)) or isinstance(frames, (str, bytes)) or len(frames) != 8:
            raise ValueError(f"View {view!r} must supply exactly eight RGB observations")
        images = []
        for frame in frames:
            if isinstance(frame, Image.Image):
                if frame.format == "TIFF":
                    raise ValueError("TIFF image inputs are not supported")
                if frame.mode != "RGB" or min(frame.size) <= 0:
                    raise ValueError("PIL observations must have RGB mode and nonzero dimensions")
                images.append(frame.copy())
            elif isinstance(frame, np.ndarray):
                if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[-1] != 3 or min(frame.shape[:2]) <= 0:
                    raise ValueError("Array observations must be nonempty uint8 [height, width, 3] RGB")
                images.append(Image.fromarray(np.ascontiguousarray(frame)).copy())
            else:
                raise ValueError("Observations must be RGB PIL images or uint8 RGB arrays")
        converted[view] = tuple(images)
    # Materialize once. Every role, especially paired C and D, reuses these
    # exact PIL objects in the same timestep-major, camera-minor order.
    return tuple(converted[view][slot] for slot in range(8) for view in views)


class GranaPredictor:
    """Load only inference-relevant checkpoint configuration and model weights."""

    def __init__(self, checkpoint: str | Path, model_name_or_path: str | None = None,
                 device: str | torch.device | None = None, max_pixels: int | None = None,
                 view_policy: str = "checkpoint"):
        """Load the single supported CUDA inference runtime.

        GPU resize, FLA/SDPA, causal-conv1d, LoRA merge, image/text reuse,
        host position IDs and compiled decoder/vision are required. Missing
        capabilities fail instead of selecting a slower implementation.
        Compilation happens on the first request; warm representative inputs
        before measuring latency. Checkpoint files are never changed.
        """
        if view_policy not in VIEW_POLICIES:
            raise ValueError(f"view_policy must be one of {VIEW_POLICIES}")
        self.view_policy = view_policy
        self.checkpoint_path = Path(checkpoint).expanduser().resolve()
        metadata = json.loads((self.checkpoint_path / "metadata.json").read_text())
        if metadata.get("format") != FORMAT:
            raise ValueError("Inference requires a Grana weights-v1 checkpoint")
        # Training cache paths and evaluation/training options may be absent on
        # an inference machine. Never run training config or cache validation.
        config = deepcopy(metadata["config"])
        if model_name_or_path is not None:
            config["model"]["name_or_path"] = _text(model_name_or_path, "model_name_or_path")
        config["model"]["gradient_checkpointing"] = False
        # Training backend choices do not select a different serving runtime.
        config["model"]["delta_rule_backend"] = "auto"
        config["model"]["attn_implementation"] = "sdpa"
        data = config.get("data", {})
        saved_views = data.get("views")
        if not isinstance(saved_views, (list, tuple)) or not saved_views:
            raise ValueError("Checkpoint must record its ordered data.views")
        self.view_names = tuple(saved_views)
        _validate_view_names(self.view_names)
        saved_budget, legacy = _checkpoint_pixel_budget(metadata, data)
        if isinstance(max_pixels, Integral) and not isinstance(max_pixels, bool):
            max_pixels = int(max_pixels)
        validate_max_pixels(max_pixels)
        self.device = torch.device("cuda" if device is None else device)
        if self.device.type != "cuda":
            raise ValueError("Inference requires a CUDA device; CPU inference is not supported")
        if not torch.cuda.is_available():
            raise RuntimeError("Inference requires CUDA; run bash scripts/setup_env.sh on a supported NVIDIA GPU")
        # Match the qualified serving budget on every public entry point,
        # including callers that did not set OMP_NUM_THREADS before importing Torch.
        torch.set_num_threads(INFERENCE_CPU_THREADS)
        saved_gpu_resize = data.get("gpu_image_resize", False)
        self.model, self.processor = build_model(config)
        resolved_backend = self.model.model_spec.get("delta_rule_backend")
        if not resolved_backend or resolved_backend["resolved"] != "fla":
            raise RuntimeError("Inference requires FLA kernels; run bash scripts/setup_env.sh "
                               "instead of using the Torch delta-rule fallback")
        default_size = self.processor.image_processor.size
        if any(type(default_size.get(key)) is not int or default_size[key] <= 0
               for key in ("shortest_edge", "longest_edge")):
            raise ValueError("Processor must expose positive integer shortest_edge and longest_edge pixel bounds")
        default_size = {key: default_size[key] for key in ("shortest_edge", "longest_edge")}
        self.enforce_pixel_budget = True
        if max_pixels is not None:
            self.max_pixels, source = max_pixels, "explicit_override"
        elif legacy:
            source = "legacy_checkpoint"
            # Historical max-only kwargs were ignored at/above this minimum.
            # Below it both bounds were supplied; retain even old patch rounding.
            self.max_pixels = saved_budget if saved_budget < default_size["shortest_edge"] else None
            self.enforce_pixel_budget = self.max_pixels is None
        else:
            self.max_pixels, source = saved_budget, "checkpoint"
        effective_size = default_size if self.max_pixels is None else {
            "shortest_edge": min(default_size["shortest_edge"], self.max_pixels),
            "longest_edge": self.max_pixels}
        # The strict loader finds adapters by requires_grad. Freeze only after
        # all adapter/head coverage and tensor identities have been verified.
        load_report = load_weights(self.model, self.checkpoint_path)
        self.model.to(self.device)
        self.model.requires_grad_(False)
        self.model.eval()
        self._request_images = ContextVar(f"grana_images_{id(self)}", default=None)
        self._counters = {}
        stack = self._configure_inference_stack()
        self.reuse_images = True
        host = self._configure_host_pipeline()
        self.image_preprocessing = {
            **preprocessing_spec(self.max_pixels, gpu_image_resize=True), "effective_size": effective_size,
            "source": source, "legacy_checkpoint": legacy,
            "legacy_requested_max_pixels": saved_budget if legacy else None,
            "enforce_pixel_budget": self.enforce_pixel_budget and self.max_pixels is not None}
        if not self.enforce_pixel_budget:
            self.image_preprocessing["policy"] = "legacy_smallcap"
        self.inference_runtime = {**stack, **host,
                                  "policy": INFERENCE_POLICY,
                                  "cpu_threads": torch.get_num_threads(),
                                  "fallback_policy": "error",
                                  "gpu_image_resize": True,
                                  "gpu_image_resize_source": "required_runtime",
                                  "checkpoint_gpu_image_resize": saved_gpu_resize,
                                  "resize_matches_checkpoint": saved_gpu_resize,
                                  "delta_rule_backend": deepcopy(resolved_backend),
                                  "attn_implementation": "sdpa"}
        self.checkpoint = {"path": str(self.checkpoint_path), "format": metadata["format"],
                           "weights_sha256": metadata["weights_sha256"],
                           "source_step": metadata.get("source_step"), "source_stage": metadata.get("source_stage"),
                           "model_name_or_path": config["model"]["name_or_path"], "load_report": load_report,
                           "inference_runtime": deepcopy(self.inference_runtime)}

    def _configure_inference_stack(self) -> dict:
        """Require the complete inference stack on this in-memory model."""
        from . import acceleration
        from grana.modeling.backends import configure_delta_rule_backend
        from grana.modeling.vision_sdpa import configure_batched_vision_sdpa
        from .optimization import configure_request_image_reuse

        model = self.model
        fla_guard = acceleration.configure_fla_output_kernel_guard(True)
        if not fla_guard["applied"]:
            raise RuntimeError(f"Inference requires the FLA output-kernel guard: {fla_guard['reason']}; "
                               "run bash scripts/setup_env.sh")
        lora_merge = acceleration.merge_lora(model)
        live = configure_delta_rule_backend(model.backbone, "auto")
        if live["resolved"] != "fla":
            raise RuntimeError("LoRA merge changed the required FLA delta-rule implementation")
        conv = acceleration.configure_causal_conv1d(model, True, self.device)
        vision_modules = configure_batched_vision_sdpa(model)
        if not vision_modules:
            raise RuntimeError("Inference requires Qwen3.5 batched vision SDPA")
        configure_request_image_reuse(model, self._request_images)
        self._counters["cpu_position_ids"] = acceleration.configure_host_position_ids(model, strict=True)
        self._counters["decoder_compile"] = acceleration.configure_decoder_compile(model, strict=True, allow_padding=True)
        self._counters["vision_compile"] = acceleration.configure_vision_compile(model, strict=True)
        compile_spec = {"mode": acceleration.COMPILE_MODE, "dynamic": False, "cuda_graphs": False,
                        "lazy": True, "recompile_limit": acceleration.RECOMPILE_LIMIT,
                        "vision_variant": acceleration.VISION_COMPILE_VARIANT}
        return {"batched_vision_sdpa": True, "batched_vision_modules": vision_modules,
                "reuse_images": True, "image_cache_scope": "one_prediction",
                "merge_lora": True, "lora_merge": lora_merge, "cpu_position_ids": True,
                "compile_decoder": True, "compile_vision": True, "compile": compile_spec,
                "causal_conv1d": conv, "fla_output_guard": fla_guard}

    def _configure_host_pipeline(self) -> dict:
        """Require device image preprocessing and exact text-token reuse."""
        from .optimization import TextInputCache, device_image_processor, supports_device_preprocessing

        if self.device.type != "cuda":
            raise ValueError("GPU image preprocessing requires a CUDA device")
        if not supports_device_preprocessing(self.processor.image_processor):
            raise ValueError("Inference requires a torch-backend image processor for GPU resize")
        self._text_cache = TextInputCache()
        self._device_images = device_image_processor(self.processor.image_processor, self.device)
        return {"text_input_cache": {"enabled": True, "max_entries": self._text_cache.max_entries},
                "gpu_image_preprocessing": True}

    def text_cache_stats(self) -> dict:
        return self._text_cache.stats()

    def runtime_counters(self) -> dict:
        """Live per-piece counters (fast-path calls, fallbacks) for runtime verification."""
        return deepcopy(self._counters)

    def _forward(self, examples: list[_Input]) -> dict:
        cached = self._request_images.get()
        inputs = collate_examples(cached.processor, examples, self.max_pixels, None,
                                  enforce_pixel_budget=self.enforce_pixel_budget)
        inputs = cached.to_device(inputs, self.device)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            return self.model(inputs, [example.role for example in examples], [example.num_stages for example in examples])

    @staticmethod
    def _progress(logits: torch.Tensor, poi_slot: int = POI_SLOT) -> dict:
        if logits.shape != (8, 10) or not torch.isfinite(logits).all():
            raise FloatingPointError("Expected eight finite ten-bin progress predictions")
        probabilities = logits.detach().float().softmax(-1)
        centers = torch.arange(10, device=probabilities.device, dtype=torch.float32) / 9
        values = (probabilities * centers).sum(-1)
        return {"values": values.cpu().tolist(), "poi_score": float(values[poi_slot]),
                "bin_probabilities": probabilities.cpu().tolist()}

    def predict(self, request: InferenceRequest, roles: Sequence[str] = ("C", "D"), *,
                prediction_frame: str | None = None) -> dict:
        """Score eight slots, optionally selecting POI or last for scalar output and D focus."""
        # Share only this request's image work. Always release cached CPU/GPU
        # tensors, including on validation/forward errors and concurrent callers.
        from .optimization import RequestImages
        if prediction_frame is not None:
            slot = resolve_prediction_slot(prediction_frame)
            request = replace(request, poi_slot=slot, prediction_frame=prediction_frame)
        cached = RequestImages(self.processor, self._device_images, self._text_cache)
        token = self._request_images.set(cached)
        try:
            result = self._predict(request, roles)
            result["image_reuse"] = cached.stats()
            return result
        finally:
            self._request_images.reset(token)

    def predict_batch(self, requests: Sequence[InferenceRequest], roles: Sequence[str] = ("C", "D"), *,
                      batch_size: int | None = None, prediction_frame: str | None = None) -> list[dict]:
        """Score multiple independent windows using actual batched model forwards.

        ``batch_size`` caps the physical window batch; None uses the supplied
        request count. Offline resource planners can probe a suitable cap before
        calling this method using measured throughput and available memory.
        A one-window batch always uses ``predict``
        unchanged, including the original unpadded compiled decoder graph.
        Larger batches retain each request's eight outputs and selected slot,
        and D uses that row's selected C prediction. Every physical batch shares
        one image/vision computation across roles, then releases its tensors.
        OOM propagates to the caller so its resource policy can retry a smaller
        batch. No CPU/eager fallback or language-state reuse is introduced.
        """
        if (not isinstance(requests, Sequence) or isinstance(requests, (str, bytes))
                or not requests or any(not isinstance(request, InferenceRequest) for request in requests)):
            raise ValueError("requests must be a nonempty sequence of InferenceRequest objects")
        if batch_size is not None and (type(batch_size) is not int or batch_size <= 0):
            raise ValueError("batch_size must be a positive integer or None")
        roles = tuple(roles)
        if not roles or len(set(roles)) != len(roles) or any(role not in ("A", "B", "C", "D") for role in roles):
            raise ValueError("roles must be a nonempty unique selection from A, B, C, D")
        if prediction_frame is not None:
            slot = resolve_prediction_slot(prediction_frame)
            requests = [replace(request, poi_slot=slot, prediction_frame=prediction_frame) for request in requests]
        batch_size = len(requests) if batch_size is None else batch_size
        results = []
        for offset in range(0, len(requests), batch_size):
            batch = requests[offset:offset + batch_size]
            if len(batch) == 1:
                results.append(self.predict(batch[0], roles))
                continue
            from .optimization import RequestImages
            cached = RequestImages(self.processor, self._device_images, self._text_cache)
            token = self._request_images.set(cached)
            try:
                scored = self._predict_batch(batch, roles)
                shared = {**cached.stats(), "scope": "one_prediction_batch", "requests": len(batch)}
                for row, result in enumerate(scored):
                    result["image_reuse"] = {**deepcopy(shared), "batch_index": row}
                results.extend(scored)
            finally:
                self._request_images.reset(token)
        return results

    def _predict_batch(self, requests: Sequence[InferenceRequest], roles: tuple[str, ...]) -> list[dict]:
        prepared, results = [], []
        # Validate every row before starting a forward. Keep observation object
        # order unchanged in all role batches so RequestImages can reuse vision.
        for request in requests:
            slot = _poi_slot(request.poi_slot)
            images = _observations(request, self.view_names if self.view_policy == "checkpoint" else None)
            views, task, stages = tuple(request.images), None, ()
            if any(role in roles for role in ("B", "C", "D")):
                task = _text(request.task, "task")
                if isinstance(request.stage_prompts, (str, bytes)) or not isinstance(request.stage_prompts, Sequence):
                    raise ValueError("stage_prompts must contain 1..26 ordered stage descriptions")
                stages = tuple(_text(stage, "stage prompt") for stage in request.stage_prompts)
                if not 1 <= len(stages) <= 26:
                    raise ValueError("stage_prompts must contain 1..26 ordered stage descriptions")
            focus = request.focus_stage
            if "B" in roles and (isinstance(focus, bool) or not isinstance(focus, Integral) or not 0 <= focus < len(stages)):
                raise ValueError("Role B needs an integer focus_stage in the task's stage range")
            prompts = {}
            if "A" in roles:
                prompts["A"] = progress_prompt(_text(request.compressed_task_prompt, "compressed_task_prompt"))
            if "B" in roles:
                prompts["B"] = stage_progress_prompt(task, stages, int(focus))
            if "C" in roles or "D" in roles:
                prompts["C"] = stage_prompt(task, stages)
            prepared.append({"images": images, "views": views, "task": task, "stages": stages,
                             "focus": focus, "slot": slot, "prompts": prompts})
            result = {"checkpoint": deepcopy(self.checkpoint), "roles": list(roles), "poi_slot": slot,
                      "prediction_frame": prediction_frame_for_slot(slot), "view_names": list(views),
                      "view_policy": self.view_policy, "checkpoint_view_names": list(self.view_names),
                      "max_pixels": self.max_pixels, "image_preprocessing": deepcopy(self.image_preprocessing),
                      "progress": {}}
            if request.frame_indices is not None:
                result["frame_indices"] = list(request.frame_indices)
            results.append(result)
        for role in prepared[0]["prompts"]:
            examples = [_Input(role, row["images"], with_views(row["prompts"][role], row["views"]),
                               row["views"], len(row["stages"]) or 1) for row in prepared]
            output = self._forward(examples)
            for index, (row, result) in enumerate(zip(prepared, results, strict=True)):
                if role == "C":
                    logits = output["stage_logits"][index, :, :len(row["stages"])].detach().float()
                    if logits.shape != (8, len(row["stages"])) or not torch.isfinite(logits).all():
                        raise FloatingPointError("Expected eight finite task-local stage predictions")
                    ids = logits.argmax(-1)
                    result["stage"] = {"ids": ids.cpu().tolist(), "probabilities": logits.softmax(-1).cpu().tolist(),
                                       "logits": logits.cpu().tolist(), "poi_stage_id": int(ids[row["slot"]])}
                else:
                    prediction = self._progress(output["progress_logits"][index], row["slot"])
                    if role == "B":
                        prediction["focus_stage"] = int(row["focus"])
                    result["progress"][role] = prediction
        if "D" in roles:
            examples = []
            for row, result in zip(prepared, results, strict=True):
                focus = result["stage"]["ids"][row["slot"]]
                prompt = with_views(stage_progress_prompt(row["task"], row["stages"], focus), row["views"])
                examples.append(_Input("D", row["images"], prompt, row["views"], len(row["stages"])))
                result["D_focus_stage"] = focus
            output = self._forward(examples)
            for index, (row, result) in enumerate(zip(prepared, results, strict=True)):
                result["progress"]["D"] = {**self._progress(output["progress_logits"][index], row["slot"]),
                                            "focus_stage": result["D_focus_stage"]}
        for result in results:
            result["selected_prediction"] = select_prediction(result)
            json.dumps(result, allow_nan=False)
        return results

    def _predict(self, request: InferenceRequest, roles: Sequence[str]) -> dict:
        if not isinstance(request, InferenceRequest):
            raise TypeError("request must be an InferenceRequest")
        poi_slot = _poi_slot(request.poi_slot)
        if isinstance(roles, str):
            roles = tuple(roles)
        else:
            roles = tuple(roles)
        if not roles or len(set(roles)) != len(roles) or any(role not in ("A", "B", "C", "D") for role in roles):
            raise ValueError("roles must be a nonempty unique selection from A, B, C, D")
        images = _observations(request, self.view_names if self.view_policy == "checkpoint" else None)
        views = tuple(request.images)
        stages: tuple[str, ...] = ()
        task = None
        if any(role in roles for role in ("B", "C", "D")):
            task = _text(request.task, "task")
            if isinstance(request.stage_prompts, (str, bytes)) or not isinstance(request.stage_prompts, Sequence):
                raise ValueError("stage_prompts must contain 1..26 ordered stage descriptions")
            stages = tuple(_text(stage, "stage prompt") for stage in request.stage_prompts)
            if not 1 <= len(stages) <= 26:
                raise ValueError("stage_prompts must contain 1..26 ordered stage descriptions")
        focus = request.focus_stage
        if "B" in roles and (isinstance(focus, bool) or not isinstance(focus, Integral) or not 0 <= focus < len(stages)):
            raise ValueError("Role B needs an integer focus_stage in the task's stage range")
        prompts = {}
        if "A" in roles:
            prompts["A"] = progress_prompt(_text(request.compressed_task_prompt, "compressed_task_prompt"))
        if "B" in roles:
            prompts["B"] = stage_progress_prompt(task, stages, int(focus))
        if "C" in roles or "D" in roles:
            prompts["C"] = stage_prompt(task, stages)
        examples = [_Input(role, images, with_views(prompt, views), views, len(stages) or 1)
                    for role, prompt in prompts.items()]
        result = {"checkpoint": deepcopy(self.checkpoint), "roles": list(roles), "poi_slot": poi_slot,
                  "prediction_frame": prediction_frame_for_slot(poi_slot),
                  "view_names": list(views), "view_policy": self.view_policy,
                  "checkpoint_view_names": list(self.view_names), "max_pixels": self.max_pixels,
                  "image_preprocessing": deepcopy(self.image_preprocessing), "progress": {}}
        if request.frame_indices is not None:
            result["frame_indices"] = list(request.frame_indices)
        stage_output = None
        # Preserve the checkpoint evaluation path's physical batch size of one.
        # This also bounds memory independently of how many roles were selected.
        for example in examples:
            output = self._forward([example])
            role = example.role
            if role == "C":
                stage_output = output
                continue
            prediction = self._progress(output["progress_logits"][0], poi_slot)
            if role == "B":
                prediction["focus_stage"] = int(focus)
            result["progress"][role] = prediction
        if "C" in prompts:
            logits = stage_output["stage_logits"][0, :, :len(stages)].detach().float()
            if logits.shape != (8, len(stages)) or not torch.isfinite(logits).all():
                raise FloatingPointError("Expected eight finite task-local stage predictions")
            probabilities = logits.softmax(-1)
            stage_ids = logits.argmax(-1)
            result["stage"] = {"ids": stage_ids.cpu().tolist(), "probabilities": probabilities.cpu().tolist(),
                               "logits": logits.cpu().tolist(), "poi_stage_id": int(stage_ids[poi_slot])}
            if "D" in roles:
                predicted_focus = int(stage_ids[poi_slot])
                d_prompt = with_views(stage_progress_prompt(task, stages, predicted_focus), views)
                d_example = _Input("D", images, d_prompt, views, len(stages))
                d_output = self._forward([d_example])
                result["progress"]["D"] = {**self._progress(d_output["progress_logits"][0], poi_slot), "focus_stage": predicted_focus}
                result["D_focus_stage"] = predicted_focus
        result["selected_prediction"] = select_prediction(result)
        # Reject NaN/Inf at the public boundary, including checkpoint metadata.
        json.dumps(result, allow_nan=False)
        return result
