# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Selection of an output frame; this never reorders the eight input slots."""
from __future__ import annotations

from numbers import Integral

PREDICTION_FRAME_SLOTS = {"poi": 5, "last": 7}
_UNSET = object()


def validate_slot(value) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or not 0 <= value < 8:
        raise ValueError("poi_slot must be an integer in 0..7")
    return int(value)


def resolve_prediction_slot(prediction_frame: str | None = None, poi_slot=_UNSET, *, default: str = "poi") -> int:
    """Resolve a named frame or advanced slot, rejecting contradictory choices."""
    if prediction_frame is not None and (not isinstance(prediction_frame, str)
                                         or prediction_frame not in PREDICTION_FRAME_SLOTS):
        raise ValueError("prediction_frame must be 'poi' or 'last'")
    if not isinstance(default, str) or default not in PREDICTION_FRAME_SLOTS:
        raise ValueError("default prediction_frame must be 'poi' or 'last'")
    slot = validate_slot(poi_slot) if poi_slot is not _UNSET else None
    if prediction_frame is not None:
        named_slot = PREDICTION_FRAME_SLOTS[prediction_frame]
        if slot is not None and slot != named_slot:
            raise ValueError(f"prediction_frame={prediction_frame!r} conflicts with poi_slot={slot}")
        return named_slot
    return PREDICTION_FRAME_SLOTS[default] if slot is None else slot


def prediction_frame_for_slot(slot: int) -> str:
    """Return the canonical output mode, including advanced custom slots."""
    slot = validate_slot(slot)
    return next((name for name, index in PREDICTION_FRAME_SLOTS.items() if index == slot), "custom")


def validate_frame_indices(values) -> tuple[int, ...] | None:
    """Validate optional source-frame provenance without sorting or deduplication."""
    if values is None:
        return None
    if (not isinstance(values, (list, tuple)) or len(values) != 8
            or any(isinstance(value, bool) or not isinstance(value, Integral) or value < 0 for value in values)):
        raise ValueError("frame_indices must contain eight nonnegative integers in the supplied slot order")
    return tuple(int(value) for value in values)


def select_prediction(result: dict) -> dict:
    """Summarize one slot while leaving all per-frame output arrays intact."""
    slot = validate_slot(result["poi_slot"])
    selected = {"slot": slot, "progress": {}, "focus_stage": {}}
    for role, output in result.get("progress", {}).items():
        selected["progress"][role] = output["values"][slot]
        if "focus_stage" in output:
            selected["focus_stage"][role] = output["focus_stage"]
    if "stage" in result:
        selected["stage_id"] = result["stage"]["ids"][slot]
    if result.get("frame_indices") is not None:
        selected["frame_index"] = result["frame_indices"][slot]
    return selected


def add_prediction_arguments(parser) -> None:
    """Add the same output-frame options to local and HTTP client CLIs."""
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--prediction-frame", choices=tuple(PREDICTION_FRAME_SLOTS),
                       help="Output frame: poi=slot 5, last=slot 7 (default: request selection, otherwise poi)")
    group.add_argument("--poi-slot", type=int, choices=range(8),
                       help="Advanced output-slot override; mutually exclusive with --prediction-frame")


ROLES_HELP = ("A: task progress only; C: stage only; C D: stage then progress focused on the selected "
              "frame's predicted stage (default). B: progress for an explicit focus stage")
