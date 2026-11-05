# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Request-local image preprocessing, transfer and vision-feature reuse.

The reference scorer's Tier A/B caches motivate this smaller serving cache.
Only images are shared between roles; stage focus and multimodal position IDs
continue through the HF forward on every call. Two optional serving pieces
remove host work around the forward:

* Device image preprocessing moves the original uint8 RGB frames once through
  pinned memory and runs the processor's resize/rescale/normalize/patchify on
  the model device instead of building and transferring fp32 patches on the
  CPU. Device interpolation rounds differently from the host uint8 kernel, so
  it is close to, but not bitwise identical with, host preprocessing.
* A predictor-lifetime text cache reuses the processor's token outputs for an
  exact (chat-template texts, image grid, call options) key. Misses run the
  unmodified processor, so cached token IDs are identical by construction.
"""
from collections import OrderedDict
from contextvars import ContextVar
from copy import copy, deepcopy
import threading
from types import MethodType

import torch

# Keep these imports available for existing inference callers.
from grana.modeling.image_preprocessing import (
    _device_frames, _to_device, device_image_processor, supports_device_preprocessing,
)

TEXT_CACHE_ENTRIES = 64


class _ImageProcessor:
    def __init__(self, original, device_processor=None):
        self.original = original
        self.device_processor, self.device = (None, None) if device_processor is None else device_processor
        self.identity = self.kwargs = self.output = None

    def __getattr__(self, name):
        return getattr(self.original, name)

    def __call__(self, images=None, **kwargs):
        identity = tuple(id(image) for image in images)
        if self.output is None:
            self.identity, self.kwargs = identity, deepcopy(kwargs)
            process = self.original if self.device_processor is None else self.device_processor
            self.output = process(images=images, **kwargs)
        elif identity != self.identity or kwargs != self.kwargs:
            raise ValueError("Image reuse requires identical observations and preprocessing within one prediction")
        return self.output


def _options_key(kwargs: dict) -> str:
    return repr(sorted(kwargs.items()))


class TextInputCache:
    """Thread-safe LRU of processor text outputs (token IDs, masks, multimodal token types)."""

    def __init__(self, max_entries: int = TEXT_CACHE_ENTRIES):
        if type(max_entries) is not int or max_entries <= 0:
            raise ValueError("max_entries must be a positive integer")
        self.max_entries = max_entries
        self._entries = OrderedDict()
        self._images_kwargs = {}
        self._lock = threading.Lock()
        self.hits = self.misses = 0

    def lookup(self, key):
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self.misses += 1
                return None
            self._entries.move_to_end(key)
            self.hits += 1
            return entry

    def record_miss(self) -> None:
        """Count a request whose call options were never seen, so it skipped ``lookup``."""
        with self._lock:
            self.misses += 1

    def store(self, key, entry: dict, options: str, images_kwargs: dict) -> None:
        with self._lock:
            self._images_kwargs[options] = deepcopy(images_kwargs)
            self._entries[key] = entry
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)

    def images_kwargs(self, options: str):
        with self._lock:
            value = self._images_kwargs.get(options)
            return None if value is None else deepcopy(value)

    def stats(self) -> dict:
        with self._lock:
            return {"entries": len(self._entries), "max_entries": self.max_entries,
                    "hits": self.hits, "misses": self.misses}


class _CachedTextProcessor:
    """Processor view that reuses exact text outputs; images still come from this request."""

    def __init__(self, processor, image_processor: _ImageProcessor, cache: TextInputCache):
        self._processor, self._image_processor, self._cache = processor, image_processor, cache
        self.hits = self.misses = 0

    def __getattr__(self, name):
        return getattr(self._processor, name)

    def __call__(self, *args, text=None, images=None, **kwargs):
        from transformers import BatchFeature

        texts = tuple(text) if isinstance(text, (list, tuple)) else (text,)
        # Only keyword string-text calls with images are cacheable; everything else is the plain processor.
        if args or images is None or "videos" in kwargs or not all(isinstance(item, str) for item in texts):
            return self._processor(*args, text=text, images=images, **kwargs)
        options = _options_key(kwargs)
        images_kwargs = self._cache.images_kwargs(options)
        if images_kwargs is not None:
            image_inputs = self._image_processor(images=images, **images_kwargs)
            grid = image_inputs["image_grid_thw"]
            key = (texts, tuple(map(tuple, grid.tolist())), options)
            entry = self._cache.lookup(key)
            if entry is not None:
                self.hits += 1
                # Clones keep the cached tensors immutable for later requests.
                return BatchFeature(data={**{name: value.clone() for name, value in entry.items()}, **image_inputs})
        else:
            self._cache.record_miss()
        self.misses += 1
        output = self._processor(text=text, images=images, **kwargs)
        image_inputs = self._image_processor.output
        key = (texts, tuple(map(tuple, output["image_grid_thw"].tolist())), options)
        entry = {name: value.detach().clone() for name, value in output.items()
                 if name not in image_inputs and torch.is_tensor(value) and value.device.type == "cpu"}
        if set(entry) | set(image_inputs) == set(output):
            self._cache.store(key, entry, options, self._image_processor.kwargs)
        return output


class RequestImages:
    def __init__(self, processor, device_processor=None, text_cache: TextInputCache | None = None):
        """``device_processor`` is a ``device_image_processor`` result; None keeps host preprocessing."""
        self.processor = copy(processor)
        self.image_processor = _ImageProcessor(processor.image_processor, device_processor)
        self.processor.image_processor = self.image_processor
        self.text = None
        if text_cache is not None:
            self.text = _CachedTextProcessor(self.processor, self.image_processor, text_cache)
            self.processor = self.text
        self.device_inputs = {}
        self.features = None
        self.feature_key = None
        self.vision_calls = self.vision_cache_hits = 0

    def to_device(self, inputs, device):
        moved = {}
        for name, value in inputs.items():
            if name in ("pixel_values", "image_grid_thw"):
                if value is not self.image_processor.output[name]:
                    raise ValueError("Processor changed a cached image tensor")
                if name not in self.device_inputs:
                    self.device_inputs[name] = value.to(device)
                moved[name] = self.device_inputs[name]
            else:
                moved[name] = value.to(device) if torch.is_tensor(value) else value
        return moved

    def stats(self):
        stats = {"scope": "one_prediction", "image_processor_calls": int(self.image_processor.output is not None),
                 "vision_calls": self.vision_calls, "vision_cache_hits": self.vision_cache_hits}
        if self.image_processor.device is not None:
            stats["image_preprocessing"] = dict(self.image_processor.device)
        if self.text is not None:
            stats["text_cache_hits"], stats["text_cache_misses"] = self.text.hits, self.text.misses
        return stats


def configure_request_image_reuse(model, context: ContextVar):
    """Reuse HF image features only for this context's exact prepared tensors.

    Keeping HF's own multimodal merge and position-ID computation avoids
    reusing any language state between the stage and predicted-focus prompts.
    ContextVars isolate direct concurrent callers as well as HTTP requests.
    """
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model

    modules = [module for module in model.modules() if type(module) is Qwen3_5Model]
    if len(modules) != 1:
        raise ValueError("Image feature reuse requires one Qwen3.5 multimodal backbone")
    module = modules[0]
    original = module.get_image_features

    def image_features(self, pixel_values, image_grid_thw=None, **kwargs):
        state = context.get()
        if (state is None or self.training or torch.is_grad_enabled()
                or pixel_values is not state.device_inputs.get("pixel_values")
                or image_grid_thw is not state.device_inputs.get("image_grid_thw")
                or kwargs != {"return_dict": True}):
            return original(pixel_values, image_grid_thw, **kwargs)
        device_type = pixel_values.device.type
        key = (id(pixel_values), id(image_grid_thw), torch.is_autocast_enabled(device_type),
               torch.get_autocast_dtype(device_type))
        if state.features is not None and state.feature_key == key:
            state.vision_cache_hits += 1
            return state.features
        state.features = original(pixel_values, image_grid_thw, **kwargs)
        state.feature_key = key
        state.vision_calls += 1
        return state.features

    module.get_image_features = MethodType(image_features, module)
