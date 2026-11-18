# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configure SDPA safety and instance-local Qwen3.5 delta-rule functions."""
from __future__ import annotations

from contextlib import contextmanager


@contextmanager
def checkpoint_verification_delta_rule(model):
    """Use a repeatable recurrence for the two checkpoint-integrity forwards.

    FLA's BF16 forward can vary for unchanged weights on GB300/27B. Comparing
    two reference forwards permits a strict restore check without relaxing
    tolerances or changing training/evaluation kernels. Restore every callable
    even when validation fails; do not change weights, configs or global HF
    functions. This context is only for checkpoint integrity, never scoring.
    """
    saved = [(module, module.chunk_gated_delta_rule, module.recurrent_gated_delta_rule)
             for module in model.modules()
             if hasattr(module, "chunk_gated_delta_rule") and hasattr(module, "recurrent_gated_delta_rule")]
    uses_fla = any(getattr(fn, "__module__", "").split(".")[0] == "fla"
                   for _, chunk, recurrent in saved for fn in (chunk, recurrent))
    if not uses_fla:
        yield {"mode": "native", "scope": "checkpoint_integrity_only", "switched_layers": 0}
        return
    if any(module.training for module in model.modules()):
        raise ValueError("Checkpoint verification reference backend requires eval mode")
    try:
        spec = configure_delta_rule_backend(model, "torch")
        yield {"mode": "torch_reference", "scope": "checkpoint_integrity_only",
               "switched_layers": len(spec["layers"])}
    finally:
        for module, chunk, recurrent in saved:
            module.chunk_gated_delta_rule = chunk
            module.recurrent_gated_delta_rule = recurrent


def configure_sdpa_backends() -> dict:
    """Exclude cuDNN SDPA for this model process, including GC recomputation.

    H100 BF16 training with left-padded batches reproduced finite forward
    values but NaNs in ScaledDotProductCudnnAttentionBackward0 (torch 2.10).
    Keep the other backend flags unchanged so PyTorch can select a supported
    Flash, memory-efficient or math implementation. A forward-only context
    would end before non-reentrant gradient checkpointing recomputes layers.

    Apply the same policy in training and inference on every platform rather
    than re-enabling an unqualified backend on a different GPU. These are
    eligibility flags, not a claim about which kernel any given call selects.
    """
    from torch.backends import cuda

    cuda.enable_cudnn_sdp(False)
    return {"policy": "exclude_cudnn", "scope": "process",
            "enabled": {"cudnn": cuda.cudnn_sdp_enabled(),
                        "flash": cuda.flash_sdp_enabled(),
                        "memory_efficient": cuda.mem_efficient_sdp_enabled(),
                        "math": cuda.math_sdp_enabled()}}


def configure_delta_rule_backend(base, backend: str = "auto") -> dict:
    """Keep HF's selected functions or use its native torch recurrence per layer.

    Call once before training or inference. Convolution, gated normalization,
    and full attention retain their independently selected implementations.
    The returned paths describe the actual chunk and recurrent callables.
    """
    if not isinstance(backend, str) or backend not in ("auto", "torch"):
        raise ValueError("model.delta_rule_backend must be 'auto' or 'torch'")

    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5GatedDeltaNet,
        torch_chunk_gated_delta_rule,
        torch_recurrent_gated_delta_rule,
    )

    targets = [(name or "<root>", module) for name, module in base.named_modules()
               if isinstance(module, Qwen3_5GatedDeltaNet)]
    if backend == "torch" and not targets:
        raise ValueError("model.delta_rule_backend='torch' requires Qwen3_5GatedDeltaNet layers")

    implementations, resolved = {}, set()
    native = (torch_chunk_gated_delta_rule, torch_recurrent_gated_delta_rule)
    for name, module in targets:
        if backend == "torch":
            # Instance attributes are plain callables; binding a method would
            # insert an extra self argument into HF's existing call sites.
            module.chunk_gated_delta_rule = torch_chunk_gated_delta_rule
            module.recurrent_gated_delta_rule = torch_recurrent_gated_delta_rule
        implementations[name] = {}
        for kind, function in (("chunk", module.chunk_gated_delta_rule),
                               ("recurrent", module.recurrent_gated_delta_rule)):
            origin = getattr(function, "__module__", type(function).__module__)
            qualname = getattr(function, "__qualname__", type(function).__qualname__)
            implementations[name][kind] = f"{origin}.{qualname}"
            if any(function is candidate for candidate in native):
                resolved.add("torch")
            elif origin == "fla" or origin.startswith("fla."):
                resolved.add("fla")
            else:
                resolved.add(origin)
    actual = next(iter(resolved)) if len(resolved) == 1 else "mixed" if resolved else "none"
    return {"requested": backend, "resolved": actual,
            "layers": [name for name, _ in targets], "implementations": implementations}
