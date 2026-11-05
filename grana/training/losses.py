# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dense eight-slot ABCD objectives, normalized separately by role."""
from __future__ import annotations

import math
from numbers import Integral

import torch
from torch.nn import functional as F

FRAME_WEIGHTS = (0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 2.0, 2.0)
ROLES = ("A", "B", "C", "D")
LOSS_COMPONENTS = ("total", "ce", "cdf", "ceiling")


def continuous_to_c51(targets: torch.Tensor) -> torch.Tensor:
    """Project onto ten centers using the source's float64 interpolation."""
    if not targets.is_floating_point() or not torch.isfinite(targets).all():
        raise ValueError("Progress targets must be finite floating-point values")
    if ((targets < 0) | (targets > 1)).any():
        raise ValueError("Progress targets must be in [0, 1]")
    position = targets.detach().double() * 9
    lower, upper = position.floor().long(), position.ceil().long()
    lower_mass = torch.where(lower == upper, 1.0, upper - position)
    upper_mass = torch.where(lower == upper, 0.0, position - lower)
    result = torch.zeros((*targets.shape, 10), device=targets.device, dtype=torch.float64)
    result.scatter_add_(-1, lower.unsqueeze(-1), lower_mass.unsqueeze(-1))
    result.scatter_add_(-1, upper.unsqueeze(-1), upper_mass.unsqueeze(-1))
    return result.float()


def _validate_weights(**weights: float) -> None:
    for name, weight in weights.items():
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight < 0:
            raise ValueError(f"{name} must be a finite nonnegative number")


def progress_loss_components(logits: torch.Tensor, targets: torch.Tensor, *,
                             cdf_weight: float = 1.0, ceiling_weight: float = 0.0) -> dict[str, torch.Tensor]:
    """Per-slot C51 soft CE and normalized squared CDF contributions.

    The CDF term averages the nine internal boundaries of ten progress bins.
    The optional low-progress ceiling defaults to zero.
    """
    _validate_weights(cdf_weight=cdf_weight, ceiling_weight=ceiling_weight)
    if logits.shape != (*targets.shape, 10):
        raise ValueError("Progress logits must align with targets and have ten bins")
    distribution = continuous_to_c51(targets)
    logits = logits.float()
    probabilities = logits.softmax(-1)
    bins = torch.arange(10, device=logits.device, dtype=torch.float32)
    # Sparse predictions can exactly match C51 targets; avoid 0 * -inf.
    log_probabilities = logits.log_softmax(-1).masked_fill(distribution == 0, 0)
    ce = -(distribution * log_probabilities).sum(-1)
    cdf = cdf_weight * (probabilities.cumsum(-1)[..., :-1] - distribution.cumsum(-1)[..., :-1]).square().mean(-1)
    expected = (probabilities * (bins / 9)).sum(-1)
    target = (distribution * (bins / 9)).sum(-1)
    ceiling = ceiling_weight * F.relu(expected - target - 1 / 9) * (target < 0.05)
    return {"total": ce + cdf + ceiling, "ce": ce, "cdf": cdf, "ceiling": ceiling}


def progress_loss(logits: torch.Tensor, targets: torch.Tensor, *,
                  cdf_weight: float = 1.0, ceiling_weight: float = 0.0) -> torch.Tensor:
    """Per-slot progress objective; components use the same reduction as CE."""
    return progress_loss_components(logits, targets, cdf_weight=cdf_weight, ceiling_weight=ceiling_weight)["total"]


def stage_loss_components(logits: torch.Tensor, targets: torch.Tensor, *,
                          cdf_weight: float = 1.0) -> dict[str, torch.Tensor]:
    """Hard-label CE plus squared CDF over all valid ordered answer candidates.

    Native answer logits may be padded with negative infinity. Only the first
    N valid classes count, so the CDF term divides by N - 1, not the padded
    vocabulary size. A one-candidate answer has zero CE and zero CDF gradient.
    An augmented one-stage task still uses all N candidates, including donors.
    """
    _validate_weights(cdf_weight=cdf_weight)
    if logits.shape[:-1] != targets.shape or targets.dtype != torch.long:
        raise ValueError("Stage targets must be int64 aligned with logits")
    if logits.shape[-1] < 1:
        raise ValueError("Stage logits need at least one answer class")
    logits = logits.float()
    bins = torch.arange(logits.shape[-1], device=logits.device)
    valid = torch.isfinite(logits)
    counts = valid.sum(-1)
    invalid_mask = (valid != (bins < counts.unsqueeze(-1))) | torch.isnan(logits) | torch.isposinf(logits)
    invalid_target = (targets < 0) | (targets >= counts)
    if invalid_mask.any() | invalid_target.any():
        raise ValueError("Stage logits need a finite prefix with -inf padding, and targets inside that prefix")
    ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.flatten(), reduction="none").view_as(targets)
    probabilities = logits.softmax(-1)
    target_cdf = (bins >= targets.unsqueeze(-1)).to(probabilities.dtype)
    boundaries = bins < (counts - 1).unsqueeze(-1)
    squared_error = (probabilities.cumsum(-1) - target_cdf).square()
    cdf = cdf_weight * (squared_error * boundaries).sum(-1) / (counts - 1).clamp_min(1)
    ceiling = torch.zeros_like(ce)
    return {"total": ce + cdf, "ce": ce, "cdf": cdf, "ceiling": ceiling}


def stage_loss(logits: torch.Tensor, targets: torch.Tensor, *,
               cdf_weight: float = 1.0) -> torch.Tensor:
    """Per-slot stage objective, normalized over valid ordered classes."""
    return stage_loss_components(logits, targets, cdf_weight=cdf_weight)["total"]


def window_role_counts(role_counts: dict[str, int], accumulation_steps: int) -> dict[str, int]:
    """Counts over an optimizer window; D consumes complete two-pass pairs."""
    if role_counts["D"]:
        if accumulation_steps % 2 or role_counts["A"] % 2 or role_counts["B"] % 2 or role_counts["C"] < role_counts["D"]:
            raise ValueError("D needs even A/B and accumulation, with C>=D")
        multiplier = accumulation_steps // 2
    else:
        multiplier = accumulation_steps
    return {role: role_counts[role] * multiplier for role in ROLES}


def role_loss_components(output: dict, examples: list, frame_weights=FRAME_WEIGHTS,
                         loss_config: dict | None = None) -> dict[str, dict[str, torch.Tensor]]:
    """Weight each selected occurrence, retaining its weight after presentation reordering."""
    exemplar = next(value for key, value in output.items() if key.endswith("logits") and value is not None)
    weights = torch.as_tensor(frame_weights, device=exemplar.device, dtype=torch.float32)
    if weights.shape != (8,):
        raise ValueError("Frame weights must describe eight selected occurrences")
    result = {role: {name: exemplar.new_zeros((), dtype=torch.float32) for name in LOSS_COMPONENTS} for role in ROLES}
    loss_config = loss_config or {}
    for kind, loss_fn, dtype in (("progress", progress_loss_components, torch.float32), ("stage", stage_loss_components, torch.long)):
        indices = output[kind + "_indices"]
        logits = output[kind + "_logits"]
        if not indices:
            continue
        orders = [getattr(examples[i], "frame_weight_order", tuple(range(8))) for i in indices]
        if any(not isinstance(order, (list, tuple)) or len(order) != 8
               or any(isinstance(slot, bool) or not isinstance(slot, Integral) for slot in order)
               or set(order) != set(range(8)) for order in orders):
            raise ValueError("frame_weight_order must be a permutation of eight occurrence indices")
        row_weights = weights[torch.tensor(orders, device=logits.device, dtype=torch.long)]
        targets = torch.tensor([examples[i].targets for i in indices], device=logits.device, dtype=dtype)
        components = loss_fn(logits, targets, **loss_config.get(kind, {}))
        losses = {name: (values * row_weights).sum(-1) for name, values in components.items()}
        for local, index in enumerate(indices):
            for name, values in losses.items():
                result[examples[index].role][name] = result[examples[index].role][name] + values[local]
    return result


def role_numerators(output: dict, examples: list, frame_weights=FRAME_WEIGHTS,
                    loss_config: dict | None = None) -> dict[str, torch.Tensor]:
    """Sum weighted anchors without normalizing a role inside a microbatch."""
    return {role: components["total"] for role, components in role_loss_components(output, examples, frame_weights, loss_config).items()}


def normalized_loss(numerators: dict[str, torch.Tensor], global_denominators: dict[str, float], role_weights: dict[str, float], world_size: int = 1) -> torch.Tensor:
    """Compensate DDP's gradient mean after global optimizer-window normalization."""
    terms = [value * (world_size * role_weights[role] / global_denominators[role])
             for role, value in numerators.items() if global_denominators[role] > 0]
    if not terms:
        raise ValueError("The optimizer window has no supervised rows")
    return sum(terms)
