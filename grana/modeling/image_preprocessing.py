# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared device image preprocessing for training, evaluation and serving.

The image processor owns resize geometry, interpolation options, normalization
and patch ordering. Passing the same uint8 frames and options through this
module therefore selects the same device kernels in every execution path.
GPU resizing can round differently from the legacy CPU uint8 resize.
"""
from copy import copy

import numpy as np
import torch


def _device_frames(images, device: torch.device) -> list[torch.Tensor]:
    """Transfer flattened RGB uint8 PIL/NumPy frames without converting to float."""
    arrays = [np.asarray(image) for image in images]
    if not arrays:
        raise ValueError("GPU image preprocessing requires at least one RGB observation")
    if any(a.dtype != np.uint8 or a.ndim != 3 or a.shape[-1] != 3 for a in arrays):
        raise ValueError("GPU image preprocessing requires uint8 [height, width, 3] RGB observations")
    if len({a.shape for a in arrays}) == 1:
        batch = _to_device(torch.from_numpy(np.stack(arrays)), device)
        return list(batch.permute(0, 3, 1, 2))
    return [_to_device(torch.from_numpy(np.array(a)), device).permute(2, 0, 1) for a in arrays]


def _to_device(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    if device.type == "cuda":
        return tensor.pin_memory().to(device, non_blocking=True)
    return tensor.to(device)


def supports_device_preprocessing(image_processor) -> bool:
    """Device preprocessing needs the torch image-processor backend."""
    return callable(getattr(image_processor, "rescale_and_normalize", None))


def device_image_processor(image_processor, device: torch.device):
    """Run resize, rescale, normalization and patchification on ``device``.

    Original uint8 frames move once, through pinned memory on CUDA. The
    processor retains its resize geometry and options, but device interpolation
    may round differently from the host uint8 kernel. The CPU device remains
    available for reference checks. The original processor is never modified.
    """
    device = torch.device(device)
    if not supports_device_preprocessing(image_processor):
        raise ValueError("Device image preprocessing requires a torch-backend image processor")

    def process(images=None, **kwargs):
        requested_device = kwargs.pop("device", device)
        if torch.device(requested_device) != device:
            raise ValueError("Image processor device must match the configured preprocessing device")
        return image_processor(images=_device_frames(images, device), device=str(device), **kwargs)

    return process, {"device": str(device)}


class _StatelessImageProcessor:
    """Delegate metadata to the source processor and prepare every batch anew."""

    def __init__(self, original, process, description):
        self.original = original
        self.process = process
        self.description = description

    def __getattr__(self, name):
        return getattr(self.original, name)

    def __call__(self, images=None, **kwargs):
        return self.process(images=images, **kwargs)


def configure_image_preprocessing(processor, device, *, resize_on_device: bool):
    """Copy a multimodal processor and configure stateless image preprocessing.

    Unlike inference's request-local image cache, this wrapper accepts different
    observations on every call. Metadata such as ``size``, ``patch_size`` and
    ``merge_size`` remains available to collation and multimodal tokenization.
    False keeps native host preprocessing. Repeated configuration with the same
    device is a no-op; changing modes starts from the original image processor,
    so training and nested evaluation cannot apply preprocessing twice.
    """
    if type(resize_on_device) is not bool:
        raise ValueError("resize_on_device must be a boolean")
    image_processor = getattr(processor, "image_processor", None)
    wrapped = isinstance(image_processor, _StatelessImageProcessor)
    if not resize_on_device:
        if not wrapped:
            return processor
        configured = copy(processor)
        configured.image_processor = image_processor.original
        del configured.image_preprocessing
        return configured
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("GPU image resizing requires a CUDA device")
    if wrapped:
        if image_processor.description == {"device": str(device)}:
            return processor
        image_processor = image_processor.original
    process, description = device_image_processor(image_processor, device)
    configured = copy(processor)
    configured.image_processor = _StatelessImageProcessor(image_processor, process, description)
    configured.image_preprocessing = description
    return configured
