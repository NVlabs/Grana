# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inference-only latency stack for a loaded, frozen Grana predictor model.

Model patches affect only the predictor's in-memory instance. The FLA output
guard is the exception: it configures a shared autotuner process-wide, including
other FLA users in that process. Checkpoint files and state-dict formats are not
changed. ``GranaPredictor`` requires every piece; non-strict helper behavior is
retained for internal reference comparisons only:

* ``merge_lora``: fold language LoRA into the base projections once at load:
  ``W <- bf16(fp32(W) + (alpha/r) * fp32(B) @ fp32(A))`` with TF32 off, then
  PEFT ``unload()``. Vision and the Grana heads have no LoRA and are untouched.
* ``configure_host_position_ids``: compute HF's multimodal RoPE position IDs
  with exact host arithmetic and pass them in, so HF skips its GPU
  ``get_rope_index``. The strict predictor rejects unsupported inputs (video,
  cached decoding, ...); non-strict reference helpers can use HF.
* ``configure_decoder_compile``: ``torch.compile`` the decoder layers plus the
  final norm as one function (default mode, ``dynamic=False``, no CUDA graphs).
  Embedding, position IDs, masks and rotary tables stay eager.
* ``configure_vision_compile``: per-grid cache of vision position embeddings and
  rotary tables, plus one compiled graph for patch embedding, all blocks and the
  merger. Attention uses the batched equal-length SDPA math with the frame count
  passed as a Python int, avoiding a per-block ``cu_seqlens.tolist()`` sync.
* ``configure_causal_conv1d``: require the CUDA causal-conv1d kernel for the
  predictor; reference helpers can inspect or disable it.

C and D remain independent forwards: nothing here caches language-model
prefixes, KV or recurrent state. The vision cache holds only grid-derived
position tables, never image or text data.
"""
from __future__ import annotations

import os
import threading
from types import MethodType
import warnings

import numpy as np
import torch

COMPILE_MODE = "default"
VISION_COMPILE_VARIANT = "poscache,cblocks:default"
RECOMPILE_LIMIT = 64
POSITION_CACHE_LIMIT = 64


def _single(model, class_name: str):
    modules = [module for module in model.modules() if type(module).__name__ == class_name]
    return modules[0] if len(modules) == 1 else None


# --------------------------------------------------------------------------- LoRA merge

def lora_layers(module) -> list:
    from peft.tuners.lora import LoraLayer
    return [(name, layer) for name, layer in module.named_modules() if isinstance(layer, LoraLayer)]


def _check_lora_layout(name: str, layer) -> None:
    if (list(layer.lora_A) != ["default"] or list(layer.lora_B) != ["default"]
            or layer.use_dora.get("default") or getattr(layer, "fan_in_fan_out", False)):
        raise ValueError(f"Unsupported LoRA layout for merging at {name}")


def lora_merge_supported(model) -> bool:
    """True when the backbone has mergeable single-adapter LoRA layers."""
    backbone = getattr(model, "backbone", None)
    if backbone is None or not hasattr(backbone, "unload"):
        return False
    try:
        layers = lora_layers(backbone)
        for name, layer in layers:
            _check_lora_layout(name, layer)
    except (ImportError, ValueError):
        return False
    return bool(layers)


def _set_cuda_matmul_tf32(enabled):
    """Set cuBLAS fp32 matmul TF32 through the API the host already uses; returns a restore function.

    Some Torch builds reject legacy ``allow_tf32`` reads after the host chose
    TF32 with ``fp32_precision``. Use the new API in that case; other builds
    allow legacy access and restore the same effective precision through it.
    """
    matmul = torch.backends.cuda.matmul
    try:
        previous = matmul.allow_tf32
    except RuntimeError:
        previous = matmul.fp32_precision
        matmul.fp32_precision = "tf32" if enabled else "ieee"
        return lambda: setattr(matmul, "fp32_precision", previous)
    matmul.allow_tf32 = enabled
    return lambda: setattr(matmul, "allow_tf32", previous)


@torch.no_grad()
def merged_lora_weight(layer) -> torch.Tensor:
    """Merged base weight for one PEFT LoRA layer, rounded once; does not modify it."""
    weight = layer.get_base_layer().weight
    lora_a, lora_b = layer.lora_A["default"].weight, layer.lora_B["default"].weight
    restore = _set_cuda_matmul_tf32(False)
    try:
        return (weight.float() + (lora_b.float() @ lora_a.float()) * layer.scaling["default"]).to(weight.dtype)
    finally:
        restore()


@torch.no_grad()
def merge_lora(model) -> dict:
    """Merge and unload the backbone's LoRA in place; returns a JSON-safe summary."""
    layers = lora_layers(model.backbone)
    if not layers:
        raise ValueError("LoRA merge requires PEFT LoRA layers in the backbone")
    for name, layer in layers:
        _check_lora_layout(name, layer)
    scalings, dtypes = set(), set()
    for _, layer in layers:
        base = layer.get_base_layer()
        scalings.add(float(layer.scaling["default"]))
        dtypes.add(str(base.weight.dtype).removeprefix("torch."))
        base.weight.copy_(merged_lora_weight(layer))
    model.backbone = model.backbone.unload()
    remaining = lora_layers(model)
    parameters = [name for name, _ in model.named_parameters() if "lora_" in name]
    if remaining or parameters:
        raise RuntimeError(f"LoRA merge incomplete: {[name for name, _ in remaining[:3]]} {parameters[:3]}")
    device = next(model.parameters()).device
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {"method": "fp32_round_once", "tf32": False, "merged_layers": len(layers),
            "scaling": sorted(scalings), "weight_dtypes": sorted(dtypes)}


# --------------------------------------------------------------------------- host M-RoPE position IDs

def host_rope_index(input_ids, mm_token_type_ids, image_grid_thw, attention_mask, spatial_merge_size):
    """Exact host re-implementation of HF ``Qwen3_5Model.get_rope_index`` for text and images.

    Returns ``(position_ids, rope_deltas)`` on ``input_ids.device``, or ``None``
    for anything this path does not cover (video tokens, unknown token types,
    inconsistent image runs, unusual shapes); callers then use HF's own path.
    """
    if (input_ids.ndim != 2 or mm_token_type_ids.shape != input_ids.shape
            or image_grid_thw.ndim != 2 or image_grid_thw.shape[1] != 3
            or (attention_mask is not None and attention_mask.shape != input_ids.shape)
            or type(spatial_merge_size) is not int or spatial_merge_size <= 0):
        return None
    types = mm_token_type_ids.cpu().numpy()
    grids = image_grid_thw.cpu().numpy().astype(np.int64)
    mask = None if attention_mask is None else attention_mask.cpu().numpy().astype(bool)
    batch, length = input_ids.shape
    out = np.zeros((3, batch, length), dtype=np.int64)
    deltas = []
    grid_index = 0
    for row in range(batch):
        token_types = types[row] if mask is None else types[row][mask[row]]
        count = token_types.shape[0]
        if count == 0:
            return None
        # Run-length groups of equal token type, as itertools.groupby in HF.
        change = np.flatnonzero(np.diff(token_types)) + 1
        starts = np.concatenate(([0], change))
        ends = np.concatenate((change, [count]))
        positions = np.empty((3, count), dtype=np.int64)
        current = 0
        for kind, start, end in zip(token_types[starts].tolist(), starts.tolist(), ends.tolist()):
            if kind == 0:
                positions[:, start:end] = np.arange(end - start, dtype=np.int64) + current
                current += end - start
            elif kind == 1:
                if grid_index >= len(grids):
                    return None
                frames, height, width = grids[grid_index].tolist()
                grid_index += 1
                h, w = height // spatial_merge_size, width // spatial_merge_size
                if end - start != frames * h * w:
                    return None
                positions[0, start:end] = np.repeat(np.arange(frames, dtype=np.int64), h * w) + current
                positions[1, start:end] = np.tile(np.repeat(np.arange(h, dtype=np.int64) + current, w), frames)
                positions[2, start:end] = np.tile(np.arange(w, dtype=np.int64) + current, h * frames)
                current += max(height, width) // spatial_merge_size
            else:
                return None
        if mask is None:
            out[:, row] = positions
        else:
            out[:, row, mask[row]] = positions
        deltas.append(int(positions.max()) + 1 - count)
    device = input_ids.device
    position_ids = torch.from_numpy(out).to(input_ids.dtype)
    rope_deltas = torch.tensor(deltas, dtype=torch.int64).unsqueeze(1)
    if device.type == "cuda":
        position_ids = position_ids.pin_memory().to(device, non_blocking=True)
        rope_deltas = rope_deltas.pin_memory().to(device, non_blocking=True)
    elif device.type != "cpu":
        position_ids, rope_deltas = position_ids.to(device), rope_deltas.to(device)
    return position_ids, rope_deltas


def configure_host_position_ids(model, *, verify_calls: int = 2, strict: bool = False) -> dict:
    """Pass host-computed position IDs into the multimodal backbone's forward.

    The first ``verify_calls`` fast-path calls also run HF ``get_rope_index`` and
    compare exactly. A mismatch permanently disables this path for the model
    (with a warning) and that call and later calls use HF's own computation.
    The public inference runtime uses ``strict=True``: unsupported inputs or a
    verification mismatch raise instead of selecting another implementation.
    Returns a live counter dict.
    """
    module = _single(model, "Qwen3_5Model")
    if module is None:
        raise ValueError("Host position IDs require one Qwen3.5 multimodal backbone")
    original = module.forward
    state = {"calls": 0, "fallbacks": 0, "verified": 0, "mismatches": 0, "enabled": True}

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None,
                inputs_embeds=None, pixel_values=None, pixel_values_videos=None, image_grid_thw=None,
                video_grid_thw=None, mm_token_type_ids=None, **kwargs):
        computed = None
        if (state["enabled"] and position_ids is None and input_ids is not None
                and mm_token_type_ids is not None and image_grid_thw is not None
                and video_grid_thw is None and pixel_values_videos is None
                and (past_key_values is None or past_key_values.get_seq_length() == 0)):
            computed = host_rope_index(input_ids, mm_token_type_ids, image_grid_thw, attention_mask,
                                       self.config.vision_config.spatial_merge_size)
        if computed is not None and state["verified"] < verify_calls:
            reference, reference_deltas = self.get_rope_index(
                input_ids, image_grid_thw=image_grid_thw, video_grid_thw=None,
                attention_mask=attention_mask, mm_token_type_ids=mm_token_type_ids)
            state["verified"] += 1
            if not (reference.dtype == computed[0].dtype and reference.shape == computed[0].shape
                    and torch.equal(reference, computed[0]) and reference_deltas.shape == computed[1].shape
                    and torch.equal(reference_deltas.to(computed[1].dtype), computed[1])):
                state["mismatches"] += 1
                state["enabled"] = False
                computed = None
                if strict:
                    raise RuntimeError("Inference position-ID verification failed; host/HF positions differ")
                warnings.warn("Host position IDs differed from HF get_rope_index; using HF's path",
                              RuntimeWarning, stacklevel=2)
        if computed is None:
            if strict:
                raise ValueError("Inference requires supported image/text position IDs without a generation cache")
            state["fallbacks"] += 1
        else:
            position_ids, self.rope_deltas = computed
            state["calls"] += 1
        return original(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
                        past_key_values=past_key_values, inputs_embeds=inputs_embeds, pixel_values=pixel_values,
                        pixel_values_videos=pixel_values_videos, image_grid_thw=image_grid_thw,
                        video_grid_thw=video_grid_thw, mm_token_type_ids=mm_token_type_ids, **kwargs)

    module.forward = MethodType(forward, module)
    return state


# --------------------------------------------------------------------------- causal-conv1d

def configure_causal_conv1d(model, enabled: bool | None = None, device=None) -> dict:
    """Report whether Qwen3.5 linear-attention layers use the causal-conv1d kernel.

    ``None`` keeps Transformers' import-time choice (active iff the optional
    ``causal_conv1d`` package imports) on CUDA devices; on any other device the
    CUDA-only kernel is replaced by the native torch convolution. ``False``
    forces the native torch convolution on this model. ``True`` requires the
    kernel on every layer.
    """
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen

    layers = [module for module in model.modules() if isinstance(module, qwen.Qwen3_5GatedDeltaNet)]
    off_cuda = device is not None and torch.device(device).type != "cuda"
    if enabled is True and off_cuda:
        raise RuntimeError("The causal-conv1d fast path requires a CUDA device")
    if enabled is False or (enabled is None and off_cuda):
        for layer in layers:
            layer.causal_conv1d_fn = None
            layer.causal_conv1d_update = qwen.torch_causal_conv1d_update
    active = bool(layers) and all(layer.causal_conv1d_fn is not None for layer in layers)
    if enabled is True and not active:
        raise RuntimeError("Inference requires the causal-conv1d CUDA kernel; "
                           "run bash scripts/setup_env.sh for the pinned environment")
    return {"requested": "auto" if enabled is None else enabled,
            "package_available": qwen.causal_conv1d_fn is not None,
            "active": active, "linear_attention_layers": len(layers)}


# --------------------------------------------------------------------------- FLA output-kernel guard

FLA_OUTPUT_MAX_WARPS = 4
_FLA_OUTPUT_GUARD_LOCK = threading.Lock()


def configure_fla_output_kernel_guard(enabled: bool = True) -> dict:
    """Remove 8-warp autotune configs of FLA's ``chunk_fwd_kernel_o`` (process-wide).

    Triton 3.6/3.7 miscompiles the 8-warp variant on Blackwell (triton#10590,
    FLA #1228/#1327; fixed in FLA 0.6.0), which can make repeated prefill
    forwards disagree. The kernel is about 1% of a forward, so the remaining
    2- and 4-warp configs cost little. Configs loaded from an FLA config
    directory (``FLA_CONFIG_DIR``) are filtered the same way.

    The first call fixes the process policy, including an explicit opt-out.
    Later conflicting calls fail instead of silently changing existing
    predictors. Use separate processes for guarded and unguarded inference.
    """
    report = {"requested": enabled, "scope": "process", "applied": False}
    try:
        import fla
        from fla.ops.common import chunk_o
    except ImportError:
        return {**report, "reason": "flash-linear-attention is not installed"}
    report["fla_version"] = getattr(fla, "__version__", "unknown")
    tuner = getattr(chunk_o.chunk_fwd_kernel_o, "fn", chunk_o.chunk_fwd_kernel_o)
    with _FLA_OUTPUT_GUARD_LOCK:
        previous = getattr(tuner, "_grana_fla_output_guard", None)
        if previous is not None:
            if previous["requested"] != enabled:
                raise RuntimeError(
                    "FLA output-kernel guard is process-wide and already configured with "
                    f"fla_output_guard={previous['requested']}; cannot change it to {enabled}. "
                    "Use separate processes for guarded and unguarded predictors.")
            return dict(previous)
        configs = getattr(tuner, "configs", None)
        if not isinstance(configs, list) or not isinstance(getattr(tuner, "cache", None), dict):
            return {**report, "reason": "unexpected FLA autotuner layout"}
        if not enabled:
            report["reason"] = "disabled by process policy"
            tuner._grana_fla_output_guard = dict(report)
            return report
        kept = [config for config in configs if config.num_warps <= FLA_OUTPUT_MAX_WARPS]
        if not kept:
            return {**report, "reason": "no config at or below the warp limit"}
        tuner.configs = kept
        for key in [key for key, config in tuner.cache.items() if config.num_warps > FLA_OUTPUT_MAX_WARPS]:
            del tuner.cache[key]
        if hasattr(tuner, "maybe_load_cached_config"):
            load = tuner.maybe_load_cached_config

            def guarded_load(key, *args, _load=load, _tuner=tuner, **kwargs):
                # FLA 0.6 adds nargs/runtime_kwargs for cached-config pruning.
                _load(key, *args, **kwargs)
                config = _tuner.cache.get(key.autotune_key)
                if config is not None and config.num_warps > FLA_OUTPUT_MAX_WARPS:
                    del _tuner.cache[key.autotune_key]

            tuner.maybe_load_cached_config = guarded_load
        report.update(applied=True, kernel="chunk_fwd_kernel_o",
                      max_num_warps=FLA_OUTPUT_MAX_WARPS, removed_configs=len(configs) - len(kept))
        tuner._grana_fla_output_guard = dict(report)
        return report


# --------------------------------------------------------------------------- torch.compile

def _raise_recompile_limit() -> None:
    import torch._dynamo
    config = torch._dynamo.config
    name = "recompile_limit" if hasattr(config, "recompile_limit") else "cache_size_limit"
    setattr(config, name, max(getattr(config, name), RECOMPILE_LIMIT))


def _compile(function):
    # Default Inductor mode only: CUDA-graph modes measured no gain here.
    return torch.compile(function, dynamic=False, fullgraph=False)


def require_compiled_inference() -> dict:
    """Keep compile errors and shape-cache exhaustion from silently selecting eager execution."""
    import torch._dynamo
    config = torch._dynamo.config
    disabled_by = [name for name in ("TORCH_COMPILE_DISABLE", "TORCHDYNAMO_DISABLE")
                   if os.environ.get(name) == "1"]
    if config.disable or disabled_by:
        reason = ", ".join(disabled_by) or "torch._dynamo.config.disable"
        raise RuntimeError(f"Inference requires torch.compile, but {reason} disables it; "
                           "remove the override and restart the process")
    config.suppress_errors = False
    config.fail_on_recompile_limit_hit = True
    return {"suppress_errors": False, "fail_on_recompile_limit_hit": True}


def configure_decoder_compile(model, *, strict: bool = False, allow_padding: bool = False) -> dict:
    """Compile the decoder and final norm; optionally compile HF's padded-batch path.

    The original unpadded function remains the latency path. ``allow_padding``
    adds a separate compiled function that passes the unmodified HF masks to
    both linear and full-attention layers; it does not enable an eager fallback.
    """
    from transformers.masking_utils import create_causal_mask
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ModelOutputWithPast

    text = _single(model, "Qwen3_5TextModel")
    if text is None:
        raise ValueError("Decoder compile requires one Qwen3.5 text model")
    _raise_recompile_limit()
    if strict:
        require_compiled_inference()
    count = text.config.num_hidden_layers
    layer_types = list(text.config.layer_types)
    layers = list(text.layers[:count])

    def run_layers(hidden, cos, sin, causal_mask, text_positions):
        embeddings = (cos, sin)
        for index, layer in enumerate(layers):
            mask = None if layer_types[index] == "linear_attention" else causal_mask
            hidden = layer(hidden, position_embeddings=embeddings, attention_mask=mask,
                           position_ids=text_positions, past_key_values=None, use_cache=False)
        return text.norm(hidden)

    compiled = _compile(run_layers)
    state = {"compiled_calls": 0, "fallbacks": 0}
    # Keep the qualified unpadded graph unchanged. Offline batches can contain
    # different predicted-focus prompts, so they additionally need HF's exact
    # linear-attention padding mask rather than an eager decoder fallback.
    def run_padded_layers(hidden, cos, sin, causal_mask, text_positions, linear_mask):
        embeddings = (cos, sin)
        for index, layer in enumerate(layers):
            mask = linear_mask if layer_types[index] == "linear_attention" else causal_mask
            hidden = layer(hidden, position_embeddings=embeddings, attention_mask=mask,
                           position_ids=text_positions, past_key_values=None, use_cache=False)
        return text.norm(hidden)

    compiled_padded = _compile(run_padded_layers) if allow_padding else None
    if allow_padding:
        state["padded_compiled_calls"] = 0
    original = text.forward

    def forward(input_ids=None, attention_mask=None, position_ids=None, past_key_values=None,
                inputs_embeds=None, use_cache=None, **kwargs):
        def fallback(**changes):
            if strict:
                layout = "an image/text request" if allow_padding else "an unpadded image/text request"
                raise ValueError("Inference requires the compiled decoder: eval mode, no gradients, "
                                 f"no generation cache and {layout}")
            state["fallbacks"] += 1
            arguments = dict(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
                             past_key_values=past_key_values, inputs_embeds=inputs_embeds, use_cache=use_cache)
            arguments.update(changes)
            return original(**arguments, **kwargs)

        if past_key_values is not None or use_cache or text.training or torch.is_grad_enabled():
            return fallback()
        if (input_ids is None) == (inputs_embeds is None):
            return fallback()
        if inputs_embeds is None:
            inputs_embeds = text.embed_tokens(input_ids)
        positions = position_ids
        if positions is None:
            positions = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)
            positions = positions.view(1, 1, -1).expand(4, inputs_embeds.shape[0], -1)
        elif positions.ndim == 2:
            positions = positions[None, ...].expand(4, positions.shape[0], -1)
        if positions.ndim == 3 and positions.shape[0] == 4:
            text_positions, positions = positions[0], positions[1:]
        else:
            text_positions = None
        causal_mask = create_causal_mask(config=text.config, inputs_embeds=inputs_embeds,
                                         attention_mask=attention_mask, past_key_values=None,
                                         position_ids=text_positions)
        linear_mask = text._update_linear_attn_mask(attention_mask, None)
        if ((linear_mask is not None and not allow_padding)
                or (causal_mask is not None and not torch.is_tensor(causal_mask))):
            return fallback(input_ids=None, inputs_embeds=inputs_embeds)
        cos, sin = text.rotary_emb(inputs_embeds, positions)
        if linear_mask is None:
            hidden = compiled(inputs_embeds, cos, sin, causal_mask, text_positions).clone()
        else:
            hidden = compiled_padded(inputs_embeds, cos, sin, causal_mask, text_positions, linear_mask).clone()
            state["padded_compiled_calls"] += 1
        state["compiled_calls"] += 1
        return Qwen3_5ModelOutputWithPast(last_hidden_state=hidden, past_key_values=None, hidden_states=(hidden,))

    text.forward = forward
    return state


def _static_vision_attention(self, hidden_states, cu_seqlens, rotary_pos_emb=None, position_embeddings=None,
                             **kwargs):
    # Same math as batched vision SDPA; the sequence count is a Python int, not cu_seqlens.
    from grana.modeling.vision_sdpa import equal_length_attention
    return equal_length_attention(self, hidden_states, position_embeddings, self._grana_static_images, **kwargs)


def configure_vision_compile(model, *, strict: bool = False) -> dict:
    """Compile the vision tower for equal-length image batches; returns live counters.

    Requires batched vision SDPA to be configured first. Unequal image sizes,
    non-SDPA attention, training and enabled autograd use the existing path.
    The public runtime uses ``strict=True`` and rejects those inputs instead.
    """
    from transformers.modeling_outputs import BaseModelOutputWithPooling

    vision = _single(model, "Qwen3_5VisionModel")
    if vision is None:
        raise ValueError("Vision compile requires one Qwen3.5 vision tower")
    attentions = [block.attn for block in vision.blocks]
    if not attentions or any(not hasattr(attention, "_grana_batched_vision_sdpa") for attention in attentions):
        raise ValueError("Vision compile requires batched vision SDPA")
    _raise_recompile_limit()
    if strict:
        require_compiled_inference()
    static_forwards = [MethodType(_static_vision_attention, attention) for attention in attentions]
    original = vision.forward

    def body(pixels, positions, cos, sin):
        hidden = vision.patch_embed(pixels) + positions
        for block in vision.blocks:
            hidden = block(hidden, cu_seqlens=None, position_embeddings=(cos, sin))
        return hidden, vision.merger(hidden)

    compiled = _compile(body)
    tables: dict = {}
    # Attention forwards are swapped during the compiled call; serialize this
    # tower so concurrent callers never observe another call's swap.
    lock = threading.Lock()
    state = {"compiled_calls": 0, "fallbacks": 0}

    def forward(hidden_states, grid_thw=None, **kwargs):
        with lock:
            grid = grid_thw.tolist() if torch.is_tensor(grid_thw) and grid_thw.ndim == 2 else []
            # HF attends within each temporal frame: cu_seqlens = repeat_interleave(h * w, t).
            lengths = [h * w for t, h, w in grid for _ in range(t)]
            if (torch.is_grad_enabled() or vision.training or not lengths or lengths[0] <= 0
                    or any(length != lengths[0] for length in lengths)
                    or hidden_states.shape[0] != sum(lengths)
                    or vision.config._attn_implementation != "sdpa"
                    or not all(attention._grana_batched_vision_sdpa for attention in attentions)):
                if strict:
                    raise ValueError("Inference requires the compiled vision tower with equal resized "
                                     "image token counts, SDPA, eval mode and no gradients")
                state["fallbacks"] += 1
                return original(hidden_states, grid_thw=grid_thw, **kwargs)
            key = tuple(map(tuple, grid))
            if key in tables:
                positions, cos, sin = tables[key]
            else:
                positions = vision.fast_pos_embed_interpolate(grid_thw)
                rotary = vision.rot_pos_emb(grid_thw).reshape(hidden_states.shape[0], -1)
                embedding = torch.cat((rotary, rotary), dim=-1)
                cos, sin = embedding.cos(), embedding.sin()
                if len(tables) >= POSITION_CACHE_LIMIT:
                    tables.clear()
                tables[key] = (positions, cos, sin)
            saved = [attention.forward for attention in attentions]
            for attention, static in zip(attentions, static_forwards):
                attention._grana_static_images = len(lengths)
                attention.forward = static
            try:
                last, pooled = compiled(hidden_states, positions, cos, sin)
            finally:
                for attention, previous in zip(attentions, saved):
                    attention.forward = previous
            state["compiled_calls"] += 1
            return BaseModelOutputWithPooling(last_hidden_state=last, pooler_output=pooled)

    vision.forward = forward
    return state
