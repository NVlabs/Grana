# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Independent checkpoint restore validation on one unchanged physical batch."""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from .checkpoint import load_weights, tensor_digest, trainable_state
from .backends import checkpoint_verification_delta_rule


def compare_checkpoint_predictions(before, after, *, rtol=1e-5, atol=1e-5):
    """Compare identical-input logits with explicit, recorded numeric tolerances.

    This does not replace the separate exact checkpoint tensor digest check.
    BF16 execution can differ even between repeated forwards of unchanged
    weights, so callers must choose tolerances for their compute precision.
    Native stage padding may contain matching negative infinities only.
    """
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0
           for v in (rtol, atol)):
        raise ValueError("Prediction tolerances must be finite and nonnegative")
    if not before or set(before) != set(after):
        raise ValueError("Checkpoint reload changed prediction outputs")
    maximum = 0.0
    for key, expected in before.items():
        actual = after[key]
        for value in (expected, actual):
            valid = torch.isfinite(value)
            if key == "stage_logits":
                valid = valid | torch.isneginf(value)
            if not bool(valid.all()) or not bool(torch.isfinite(value).any()):
                raise FloatingPointError(f"Invalid checkpoint verification predictions: {key}")
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
        finite = torch.isfinite(expected)
        maximum = max(maximum, float((actual[finite].double() - expected[finite].double()).abs().max()))
    return {"prediction_parity": True, "prediction_max_abs_delta": maximum,
            "rtol": rtol, "atol": atol, "comparison": "Same inputs before and after exact weight restore"}


def _transferable_digest(model) -> str:
    """Match save_weights, including the frozen special-token embedding rows."""
    state = trainable_state(model)
    embeddings = (model.backbone.get_input_embeddings(), model.backbone.get_output_embeddings())
    for name, embedding in zip(("added_input_embeddings", "added_output_embeddings"), embeddings, strict=True):
        ids = torch.tensor(list(model.token_ids.values()), device=embedding.weight.device)
        state[name] = embedding.weight.index_select(0, ids).detach().cpu().contiguous()
    return tensor_digest(state)


def _prediction_snapshot(model, inputs, roles, num_stages, device) -> dict:
    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                               enabled=device.type == "cuda"):
        output = model(inputs, roles, num_stages)
    values = {key: value.detach().cpu().clone() for key, value in output.items()
              if key.endswith("logits") and value is not None}
    if not values:
        raise ValueError("Checkpoint verification requires prediction logits")
    for key, value in values.items():
        if key == "stage_logits":
            counts = [count for role, count in zip(roles, num_stages, strict=True) if role == "C"]
            if value.ndim != 3 or value.shape[:2] != (len(counts), 8):
                raise ValueError("Stage verification logits must identify eight slots per C row")
            valid = (torch.arange(value.shape[-1])[None, None, :] < torch.tensor(counts)[:, None, None])
            valid = valid.expand_as(value)
            finite = torch.isfinite(value[valid]).all() and torch.isneginf(value[~valid]).all()
        else:
            finite = torch.isfinite(value).all()
        if not finite:
            raise FloatingPointError(f"Invalid checkpoint verification predictions: {key}")
    return values


def verify_checkpoint_reload(model, checkpoint, inputs, roles, num_stages, *,
                             rtol: float = 1e-5, atol: float = 1e-5) -> dict:
    """Perturb and strictly restore a loaded model, comparing the same batch.

    The caller owns the physical batch and must already have loaded the saved
    checkpoint and set every model module to evaluation mode. Both forwards
    reuse exactly these inputs; there is no singleton or cross-batch comparison,
    optimizer update, sampler access, or source-run receipt mutation. This is an
    independent recovery validation, not evidence that a previous trainer ran
    its final verification. Only small CPU prediction copies survive a forward.
    """
    if any(module.training for module in model.modules()):
        raise ValueError("Checkpoint verification requires an already loaded model in eval mode")
    if any(isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value)
           or value < 0 for value in (rtol, atol)):
        raise ValueError("Checkpoint verification tolerances must be finite and nonnegative")
    batch_size = inputs["input_ids"].shape[0]
    if batch_size < 1 or len(roles) != batch_size or len(num_stages) != batch_size:
        raise ValueError("Checkpoint verification must preserve the complete physical batch")
    if any(role not in ("A", "B", "C", "D") for role in roles):
        raise ValueError("Checkpoint verification roles must be A/B/C/D")
    if any(type(count) is not int or not 1 <= count <= 26
           for role, count in zip(roles, num_stages, strict=True) if role == "C"):
        raise ValueError("C verification rows need 1..26 stage answer choices")
    checkpoint = Path(checkpoint)
    metadata = json.loads((checkpoint / "metadata.json").read_text())
    expected = metadata["weights_sha256"]
    before_digest = _transferable_digest(model)
    if before_digest != expected:
        raise ValueError("Model must contain every saved checkpoint tensor before reload verification")
    name, parameter = next((name, parameter) for name, parameter in model.named_parameters()
                           if parameter.requires_grad and parameter.numel())
    index = (0,) * parameter.ndim
    original_scalar = parameter[index].detach().clone()
    device = inputs["input_ids"].device
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices), checkpoint_verification_delta_rule(model) as verification_backend:
        before = _prediction_snapshot(model, inputs, roles, num_stages, device)
        try:
            with torch.no_grad():
                parameter[index].add_(1)
            perturbed_digest = _transferable_digest(model)
            if perturbed_digest == before_digest:
                raise RuntimeError("Checkpoint verification did not alter the saved trainable tensor")
            loaded = load_weights(model, checkpoint)
            after_digest = _transferable_digest(model)
            if after_digest != before_digest or loaded["weights_sha256"] != expected:
                raise RuntimeError("Checkpoint reload did not restore all transferable tensors")
            after = _prediction_snapshot(model, inputs, roles, num_stages, device)
            comparison = compare_checkpoint_predictions(before, after, rtol=rtol, atol=atol)
            maximum = comparison["prediction_max_abs_delta"]
        finally:
            # Even a failed load must not leave our deliberate scalar mutation.
            with torch.no_grad():
                parameter[index].copy_(original_scalar)
    return {
        "mode": "same_batch_reloaded_checkpoint_validation",
        "source_optimizer_step": loaded["source_step"], "weights_sha256": expected,
        "initial_loaded_weights_sha256": before_digest, "restored_weights_sha256": after_digest,
        "perturbed_weights_sha256": perturbed_digest, "perturbed_tensor": name,
        "checkpoint_reload_equal": True, "prediction_parity": True,
        "prediction_max_abs_delta": maximum, "prediction_logits": sorted(before),
        "physical_batch_size": batch_size, "roles": list(roles), "forward_calls": 2,
        "new_optimizer_updates": 0, "checkpoint_load": loaded, "rtol": rtol, "atol": atol,
        "verification_backend": verification_backend,
        "comparison": "The same physical batch before and after restoring the saved checkpoint; independent of original training completion.",
    }
