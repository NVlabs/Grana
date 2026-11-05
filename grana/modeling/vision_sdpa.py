# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Batch independent equal-length image attention in Qwen3.5.

Evaluation and frozen-vision training are independently opt-in. The training
path certifies the entire encoder on each forward, without changing module
modes, autograd, checkpointing, or state-dict keys. Unsupported inputs retain
Transformers' original attention implementation.
"""
from collections import Counter
from contextvars import ContextVar
from time import perf_counter
from types import MethodType

import torch
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    ALL_ATTENTION_FUNCTIONS,
    Qwen3_5VisionAttention,
    Qwen3_5VisionModel,
    apply_rotary_pos_emb_vision,
    eager_attention_forward,
)

# A scope rather than a cached boolean prevents standalone attention calls or
# checkpoint recomputation outside a certified encoder forward using the path.
_VISION_SCOPE = ContextVar("grana_frozen_vision_scope", default=None)


def _new_counts():
    return {"calls": 0, "hits": 0, "images": 0, "fallbacks": Counter()}


def _vision_forward(self, hidden_states, grid_thw, **kwargs):
    training_enabled = self._grana_training_vision_sdpa
    scope = {
        "key": self._grana_vision_scope_key,
        "frozen": self.training and training_enabled and not any(
            parameter.requires_grad for parameter in self.parameters()),
        "input_requires_grad": hidden_states.requires_grad,
    }
    token = _VISION_SCOPE.set(scope)
    profile = self._grana_vision_profile
    start = perf_counter() if profile else None
    events = None
    stream = None
    if profile and hidden_states.is_cuda:
        events = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        stream = torch.cuda.current_stream(hidden_states.device)
        events[0].record(stream)
    try:
        return self._grana_original_encoder_forward(hidden_states, grid_thw, **kwargs)
    finally:
        _VISION_SCOPE.reset(token)
        if profile:
            if events is not None:
                events[1].record(stream)
                self._grana_vision_cuda_events.append(events)
            self._grana_vision_profile_calls += 1
            self._grana_vision_host_seconds += perf_counter() - start


def _batched_forward(self, hidden_states, cu_seqlens, rotary_pos_emb=None,
                     position_embeddings=None, **kwargs):
    phase = "training" if self.training else "evaluation"
    counts = self._grana_vision_counts[phase]
    counts["calls"] += 1

    def original(reason):
        counts["fallbacks"][reason] += 1
        return self._grana_original_vision_forward(
            hidden_states, cu_seqlens, rotary_pos_emb=rotary_pos_emb,
            position_embeddings=position_embeddings, **kwargs)

    enabled = self._grana_training_vision_sdpa if self.training else self._grana_batched_vision_sdpa
    if not enabled:
        return original("disabled")
    if self.config._attn_implementation != "sdpa":
        return original("non_sdpa_backend")
    if position_embeddings is None:
        return original("missing_position_embeddings")
    if self.training:
        scope = _VISION_SCOPE.get()
        if scope is None or scope["key"] is not getattr(self, "_grana_vision_scope_key", None):
            return original("unverified_encoder")
        if not scope["frozen"]:
            return original("trainable_encoder")
        if scope["input_requires_grad"] or hidden_states.requires_grad or any(
                tensor.requires_grad for tensor in position_embeddings):
            return original("input_requires_grad")
        if self.attention_dropout != 0:
            return original("attention_dropout")
    elif torch.is_grad_enabled():
        # Preserve the existing inference contract for callers that use eval()
        # while differentiating through images or parameters.
        return original("autograd_enabled")
    boundaries = cu_seqlens.tolist()
    lengths = [end - start for start, end in zip(boundaries, boundaries[1:])]
    total = hidden_states.shape[0]
    if (not lengths or lengths[0] <= 0 or boundaries[0] != 0 or boundaries[-1] != total):
        return original("invalid_boundaries")
    if any(length != lengths[0] for length in lengths):
        return original("unequal_image_lengths")
    counts["hits"] += 1
    counts["images"] += len(lengths)
    return equal_length_attention(self, hidden_states, position_embeddings, len(lengths), **kwargs)


def equal_length_attention(self, hidden_states, position_embeddings, sequences: int, **kwargs):
    """SDPA over independent contiguous equal-length images, never across images."""
    total = hidden_states.shape[0]
    query, key, value = (self.qkv(hidden_states).reshape(total, 3, self.num_heads, -1)
                         .permute(1, 0, 2, 3).unbind(0))
    query, key = apply_rotary_pos_emb_vision(query, key, *position_embeddings)
    query, key, value = (tensor.reshape(sequences, total // sequences, self.num_heads, -1).transpose(1, 2)
                         for tensor in (query, key, value))
    attention = ALL_ATTENTION_FUNCTIONS.get_interface("sdpa", eager_attention_forward)
    output, _ = attention(self, query, key, value, attention_mask=None,
                          scaling=self.scaling, dropout=0.0, is_causal=False, **kwargs)
    return self.proj(output.reshape(total, -1).contiguous())


def configure_batched_vision_sdpa(model, *, enabled=True, training_enabled=False, profile=False):
    """Independently toggle eval and frozen-encoder training; preserve state dict.

    ``enabled`` retains the historical inference-only meaning. Training hits
    require a recognized Qwen3.5 vision encoder, verified frozen on every call.
    ``profile`` optionally measures full encoder spans (including patch/merge),
    with asynchronous CUDA events; no per-forward synchronization is introduced.
    This does not reset counters or change model mode or requires_grad flags.
    """
    for name, value in (("enabled", enabled), ("training_enabled", training_enabled), ("profile", profile)):
        if type(value) is not bool:
            raise ValueError(f"{name} must be a bool")
    modules = [module for module in model.modules() if type(module) is Qwen3_5VisionAttention]
    if not modules:
        raise ValueError("Batched vision SDPA requires Qwen3.5 vision attention modules")
    encoders = [module for module in model.modules() if type(module) is Qwen3_5VisionModel]
    for encoder in encoders:
        if not hasattr(encoder, "_grana_original_encoder_forward"):
            encoder._grana_original_encoder_forward = encoder.forward
            encoder.forward = MethodType(_vision_forward, encoder)
            encoder._grana_vision_scope_key = object()
            encoder._grana_vision_profile_calls = 0
            encoder._grana_vision_host_seconds = 0.0
            encoder._grana_vision_cuda_seconds = 0.0
            encoder._grana_vision_cuda_events = []
        encoder._grana_training_vision_sdpa = training_enabled
        encoder._grana_vision_profile = profile
        for module in encoder.modules():
            if type(module) is Qwen3_5VisionAttention:
                module._grana_vision_scope_key = encoder._grana_vision_scope_key
    for module in modules:
        if not hasattr(module, "_grana_original_vision_forward"):
            module._grana_original_vision_forward = module.forward
            module.forward = MethodType(_batched_forward, module)
            module._grana_vision_counts = {"training": _new_counts(), "evaluation": _new_counts()}
        module._grana_batched_vision_sdpa = enabled
        module._grana_training_vision_sdpa = training_enabled
    return len(modules)


def batched_vision_sdpa_stats(model, *, reset=False, synchronize=False):
    """Snapshot counters; optionally return-and-reset or resolve CUDA profiling.

    Counters count attention-layer calls; images counts sequences on optimized
    calls and thus includes each layer. Profile spans count whole encoder calls.
    Without ``synchronize``, unfinished CUDA spans are reported as pending and
    excluded from cuda_seconds. Reset discards any still-pending spans, so timing
    consumers should synchronize first at their existing measurement boundary.
    An unconfigured model yields empty/zero counts.
    """
    modules = [module for module in model.modules() if hasattr(module, "_grana_vision_counts")]
    encoders = [module for module in model.modules() if hasattr(module, "_grana_vision_cuda_events")]
    result = {"attention_modules": len(modules), "vision_encoders": len(encoders),
              "evaluation_enabled": bool(modules) and all(module._grana_batched_vision_sdpa for module in modules),
              "training_enabled": bool(modules) and all(module._grana_training_vision_sdpa for module in modules),
              "training": _new_counts(), "evaluation": _new_counts(),
              "profile": {"enabled": any(module._grana_vision_profile for module in encoders),
                          "vision_calls": 0, "host_seconds": 0.0, "cuda_seconds": 0.0,
                          "pending_cuda_spans": 0}}
    for module in modules:
        for phase in ("training", "evaluation"):
            for key in ("calls", "hits", "images"):
                result[phase][key] += module._grana_vision_counts[phase][key]
            result[phase]["fallbacks"].update(module._grana_vision_counts[phase]["fallbacks"])
        if reset:
            module._grana_vision_counts = {"training": _new_counts(), "evaluation": _new_counts()}
    for encoder in encoders:
        pending = []
        for start, end in encoder._grana_vision_cuda_events:
            if synchronize:
                end.synchronize()
            if end.query():
                encoder._grana_vision_cuda_seconds += start.elapsed_time(end) / 1000
            else:
                pending.append((start, end))
        encoder._grana_vision_cuda_events = pending
        result["profile"]["vision_calls"] += encoder._grana_vision_profile_calls
        result["profile"]["host_seconds"] += encoder._grana_vision_host_seconds
        result["profile"]["cuda_seconds"] += encoder._grana_vision_cuda_seconds
        result["profile"]["pending_cuda_spans"] += len(pending)
        if reset:
            encoder._grana_vision_profile_calls = 0
            encoder._grana_vision_host_seconds = 0.0
            encoder._grana_vision_cuda_seconds = 0.0
            encoder._grana_vision_cuda_events = []
    for phase in ("training", "evaluation"):
        result[phase]["fallbacks"] = dict(result[phase]["fallbacks"])
    return result
