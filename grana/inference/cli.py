# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""File input for the label-free predictor; paths are relative to the request JSON."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from .selection import (_UNSET, ROLES_HELP, add_prediction_arguments, prediction_frame_for_slot,
                        resolve_prediction_slot, validate_frame_indices, validate_slot)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path, help="JSON request with ordered views and task prompts")
    parser.add_argument("--roles", nargs="+", choices=list("ABCD"), default=["C", "D"],
                        help=ROLES_HELP)
    parser.add_argument("--output", type=Path, help="Write a new JSON file; otherwise print to stdout")
    parser.add_argument("--plot", type=Path, help="Write a new PNG labeling the requested roles and selected frame")
    parser.add_argument("--model-name-or-path", help="Equivalent base model identifier or local snapshot")
    parser.add_argument("--device", help="Torch device, e.g. cuda:0; defaults to CUDA; CPU inference is unsupported")
    parser.add_argument("--max-pixels", type=int,
                        help="Enforce a per-image pixel budget (>=1024); omitted inherits checkpoint preprocessing")
    parser.add_argument("--view-policy", choices=("checkpoint", "request"), default="checkpoint",
                        help="checkpoint requires saved camera names/order; request uses supplied named views")
    add_prediction_arguments(parser)
    parser.add_argument("--frame-index", type=int,
                        help="Compose the standard eight-frame window at this source frame from .npy camera arrays")


def load_request(path: Path, *, prediction_frame: str | None = None, poi_slot: int | None = None,
                 default_prediction_frame: str = "poi", frame_index: int | None = None):
    """Read exactly eight RGB frames per view, without labels or cache dependencies."""
    import numpy as np
    from PIL import Image
    from .client import json_loads
    from .predictor import InferenceRequest, _validate_view_names

    path = path.expanduser().resolve()
    raw = json_loads(path.read_text())
    allowed = {"views", "frame_indices", "task", "compressed_task_prompt", "stage_prompts", "focus_stage",
               "poi_slot", "prediction_frame"}
    if not isinstance(raw, dict) or set(raw) - allowed:
        raise ValueError("Request must be an object containing only views, frame_indices, task, "
                         "compressed_task_prompt, stage_prompts, focus_stage, poi_slot, prediction_frame")
    if "prediction_frame" in raw and not isinstance(raw["prediction_frame"], str):
        raise ValueError("prediction_frame must be 'poi' or 'last'")
    raw_slot = validate_slot(raw["poi_slot"]) if "poi_slot" in raw else _UNSET
    selected_slot = resolve_prediction_slot(raw.get("prediction_frame"), raw_slot, default=default_prediction_frame)
    if prediction_frame is not None or poi_slot is not None:
        selected_slot = resolve_prediction_slot(prediction_frame, poi_slot if poi_slot is not None else _UNSET)
    selected_mode = prediction_frame if prediction_frame is not None else (
        None if poi_slot is not None else raw.get("prediction_frame"))
    views = raw.get("views")
    if not isinstance(views, dict) or not views:
        raise ValueError("views must be an ordered mapping of camera names to .npy paths or eight image paths")
    _validate_view_names(tuple(views))
    indices = validate_frame_indices(raw.get("frame_indices"))
    if frame_index is not None:
        if type(frame_index) is not int or frame_index < 0:
            raise ValueError("frame_index must be a nonnegative integer")
        if "frame_indices" in raw:
            raise ValueError("--frame-index cannot be combined with request frame_indices")
        if prediction_frame_for_slot(selected_slot) == "custom":
            raise ValueError("--frame-index requires prediction_frame poi/last or poi_slot 5/7")
        if any(not isinstance(source, str) for source in views.values()):
            raise ValueError("--frame-index requires .npy camera arrays")

    def resolve(value: str) -> Path:
        item = Path(value).expanduser()
        return item if item.is_absolute() else path.parent / item

    images = {}
    array_lengths = set()
    for view, source in views.items():
        if isinstance(source, str):
            filename = resolve(source)
            if filename.suffix.lower() != ".npy":
                raise ValueError(f"{view}: use a .npy array or a list of eight RGB image paths")
            frames = np.load(filename, mmap_mode="r", allow_pickle=False)
            if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[-1] != 3:
                raise ValueError(f"{view}: expected RGB uint8 [T,H,W,3]")
            array_lengths.add(len(frames))
            if frame_index is not None:
                from .window import compose_frame_indices
                indices = compose_frame_indices(frame_index, len(frames),
                                                prediction_frame=prediction_frame_for_slot(selected_slot))
            selected = indices if indices is not None else list(range(len(frames)))
            if len(selected) != 8 or max(selected) >= len(frames):
                raise ValueError(f"{view}: supply exactly eight in-range frame_indices, or a length-eight array")
            images[view] = tuple(np.array(frames[i], copy=True) for i in selected)
        elif isinstance(source, list) and len(source) == 8 and all(isinstance(item, str) for item in source):
            if indices is not None:
                raise ValueError("frame_indices applies only to .npy arrays; image lists already specify eight frames")
            rows = []
            for filename in source:
                with Image.open(resolve(filename)) as image:
                    if image.mode != "RGB":
                        raise ValueError(f"{view}: image must already be RGB, got {image.mode}")
                    rows.append(image.copy())
            images[view] = tuple(rows)
        else:
            raise ValueError(f"{view}: expected a .npy path or eight image paths")
    if len(array_lengths) > 1:
        raise ValueError("All camera arrays must have the same frame count")
    return InferenceRequest(images=images, task=raw.get("task"),
                            compressed_task_prompt=raw.get("compressed_task_prompt"),
                            stage_prompts=raw.get("stage_prompts", ()), focus_stage=raw.get("focus_stage"),
                            poi_slot=selected_slot, prediction_frame=selected_mode, frame_indices=indices)


def run(args: argparse.Namespace) -> None:
    from .predictor import GranaPredictor
    from .presentation import render_prediction, validate_plot_path

    # Validate file input before allocating a model or GPU.
    plot = validate_plot_path(getattr(args, "plot", None), args.output)
    request = load_request(args.input, prediction_frame=getattr(args, "prediction_frame", None),
                           poi_slot=getattr(args, "poi_slot", None),
                           frame_index=getattr(args, "frame_index", None))
    if args.output and args.output.expanduser().exists():
        raise FileExistsError(f"Use a fresh inference output: {args.output}")
    predictor = GranaPredictor(args.checkpoint, model_name_or_path=args.model_name_or_path,
                               device=args.device, max_pixels=args.max_pixels, view_policy=args.view_policy)
    result = predictor.predict(request, roles=args.roles)
    if plot is not None:
        render_prediction(result, request, plot)
    payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        output = args.output.expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x") as stream:
            stream.write(payload)
    else:
        print(payload, end="")
