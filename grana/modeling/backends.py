# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Select Qwen3.5 delta-rule functions without changing other model instances."""
from __future__ import annotations


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
