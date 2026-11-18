# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Identical label-free image serialization for training and inference."""
from __future__ import annotations

from .prompts import PROGRESS_MARKER, STAGE_MARKER


class PixelBudgetError(ValueError):
    """The processor's rounded image geometry exceeds an explicit budget."""


def validate_max_pixels(value: int | None) -> None:
    if value is not None and (type(value) is not int or value < 1024):
        raise ValueError("max_pixels must be null (processor defaults) or an integer >= 1024")


def preprocessing_spec(max_pixels: int | None, *, gpu_image_resize: bool = False) -> dict:
    validate_max_pixels(max_pixels)
    if type(gpu_image_resize) is not bool:
        raise ValueError("gpu_image_resize must be a boolean")
    spec = {"version": 1, "policy": "processor_default" if max_pixels is None else "pixel_budget",
            "max_pixels": max_pixels}
    if gpu_image_resize:
        # Preserve existing host-resize descriptors and plan fingerprints.
        # A GPU-trained checkpoint must retain its resize semantics at serving.
        spec.update(version=2, resize_backend="torchvision_cuda")
    return spec


def serialize_messages(prompt: str, role: str, images, view_names) -> list[dict]:
    """Put one marker after each of eight complete ordered view groups."""
    if role not in ("A", "B", "C", "D"):
        raise ValueError("role must be A, B, C, or D")
    if not view_names or len(images) != 8 * len(view_names):
        raise ValueError("Each input must have eight complete view groups")
    marker = STAGE_MARKER if role == "C" else PROGRESS_MARKER
    content = [{"type": "text", "text": prompt}]
    for slot in range(8):
        for view_index, view in enumerate(view_names):
            if len(view_names) > 1:
                content.append({"type": "text", "text": f"T{slot + 1} {view[0].upper()}{view[1:]} view: "})
            content.append({"type": "image", "image": images[slot * len(view_names) + view_index]})
        content.append({"type": "text", "text": marker})
    return [{"role": "user", "content": content}]


def collate_examples(processor, examples: list, max_pixels: int | None, device, *,
                     enforce_pixel_budget: bool = True) -> dict:
    """Serialize original RGB, then verify an explicit per-image pixel budget.

    None preserves the processor's defaults. The compatibility exception is
    exclusively for reproducing old checkpoints whose small budgets also set
    the resize minimum; their patch rounding could exceed the stated number.
    """
    import torch

    validate_max_pixels(max_pixels)
    if type(enforce_pixel_budget) is not bool:
        raise ValueError("enforce_pixel_budget must be a boolean")
    texts = [processor.apply_chat_template(example.messages(), tokenize=False, add_generation_prompt=True) for example in examples]
    images = [image for example in examples for image in example.images]
    image_kwargs = {}
    if max_pixels is not None:
        default_minimum = processor.image_processor.size["shortest_edge"]
        image_kwargs = {"min_pixels": min(default_minimum, max_pixels), "max_pixels": max_pixels}
    inputs = processor(text=texts, images=images, padding=True, return_tensors="pt", **image_kwargs)
    if max_pixels is not None and enforce_pixel_budget:
        grid = inputs.get("image_grid_thw")
        patch_size = getattr(processor.image_processor, "patch_size", None)
        if (not torch.is_tensor(grid) or grid.ndim != 2 or grid.shape != (len(images), 3)
                or grid.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
                or type(patch_size) is not int or patch_size <= 0):
            raise ValueError("Cannot verify max_pixels: processor must return image_grid_thw and its spatial patch_size")
        dimensions = grid.detach().cpu().tolist()
        if any(dimension <= 0 for row in dimensions for dimension in row):
            raise ValueError("Cannot verify max_pixels: image_grid_thw dimensions must be positive integers")
        # Python integer multiplication cannot wrap an oversized processor grid.
        spatial_pixels = [row[1] * row[2] * patch_size ** 2 for row in dimensions]
        largest = max(spatial_pixels, default=0)
        if largest > max_pixels:
            raise PixelBudgetError(f"Processor resize exceeded max_pixels={max_pixels}: got {largest} spatial pixels per image; "
                                   "minimum size and patch rounding cannot satisfy this budget")
    if device is None:  # caller places tensors (e.g. request-local image reuse)
        return dict(inputs)
    return {name: value.to(device) if torch.is_tensor(value) else value for name, value in inputs.items()}
