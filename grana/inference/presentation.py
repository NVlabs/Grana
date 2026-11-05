# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Client-side prediction charts with explicit task and frame selection.

Rendering consumes the returned eight-slot arrays; it never reruns the model
or changes the D focus. Pillow is imported only when an image is requested.
"""
from __future__ import annotations

import json
import math
from numbers import Integral, Real
from pathlib import Path


def _values(values, label):
    if (not isinstance(values, (list, tuple)) or len(values) != 8
            or any(isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v) for v in values)):
        raise ValueError(f"{label} must contain eight finite predictions")
    return list(values)


def prediction_chart_data(result, request) -> dict:
    """Validate the selection and describe exactly which predictions are drawn."""
    from .selection import prediction_frame_for_slot, validate_frame_indices

    slot = result["poi_slot"]
    mode = prediction_frame_for_slot(slot)
    if result.get("prediction_frame", mode) != mode:
        raise ValueError("Prediction frame metadata disagrees with the selected slot")
    # A Python predict(..., prediction_frame=...) override intentionally leaves
    # the caller's original request object unchanged. The response owns the
    # selection; use it for both the highlighted image and D-focus validation.
    request_indices = validate_frame_indices(getattr(request, "frame_indices", None))
    result_indices = validate_frame_indices(result.get("frame_indices"))
    if request_indices is not None and result_indices is not None and request_indices != result_indices:
        raise ValueError("Request and response source frame indices differ")
    indices = result_indices if result_indices is not None else request_indices
    roles = result["roles"]
    if not roles or len(set(roles)) != len(roles) or any(r not in ("A", "B", "C", "D") for r in roles):
        raise ValueError("Prediction chart requires the requested roles")
    series = []
    for role in roles:
        if role == "C":
            values = _values(result["stage"]["ids"], "C stage IDs")
            if any(isinstance(v, bool) or not isinstance(v, Integral) or v < 0 for v in values):
                raise ValueError("Stage IDs must be nonnegative integers")
            count = len(request.stage_prompts)
            upper = max(1, count - 1)
            if max(values) >= count:
                raise ValueError("Stage ID exceeds the supplied stage descriptions")
            title = "C | stage ID (zero-based)"
        else:
            progress = result["progress"][role]
            values = _values(progress["values"], f"{role} progress")
            if any(not 0 <= v <= 1 for v in values):
                raise ValueError("Progress must be in [0, 1]")
            upper = 1
            if role == "A":
                title = "A | whole-task progress [0, 1]"
            else:
                focus = progress["focus_stage"]
                title = f"{role} | progress relative to stage {focus} [0, 1]"
                if role == "D":
                    if (focus != result["D_focus_stage"]
                            or focus != result["stage"]["ids"][slot]):
                        raise ValueError("D focus does not match C at the selected frame; rerun C,D")
        series.append({"role": role, "title": title, "values": values,
                       "selected_value": values[slot], "upper": upper})
    return {"roles": list(roles), "prediction_frame": mode, "slot": slot,
            "frame_indices": list(indices) if indices is not None else None,
            "frame_index": indices[slot] if indices is not None else None,
            "series": series, "weights_sha256": result.get("checkpoint", {}).get("weights_sha256"),
            "D_focus_stage": result.get("D_focus_stage"),
            "x_axis": "query slot in supplied input order (not elapsed time)",
            "D_condition": "All eight D values use the same focus selected by C at the query slot."}


def validate_plot_path(destination: str | Path | None, output: str | Path | None = None) -> Path | None:
    """Check output paths before an expensive inference call."""
    if destination is None:
        return None
    destination = Path(destination).expanduser()
    if destination.suffix.lower() != ".png":
        raise ValueError("Prediction plots must use a .png filename")
    if destination.exists():
        raise FileExistsError(f"Use a fresh prediction plot: {destination}")
    if output is not None and destination.resolve() == Path(output).expanduser().resolve():
        raise ValueError("Prediction plot and JSON output must use different paths")
    return destination


def render_prediction(result, request, destination: str | Path) -> dict:
    """Write a new PNG with selected observations, eight-slot charts and provenance."""
    from PIL import Image, ImageDraw, ImageFont, PngImagePlugin

    destination = validate_plot_path(destination)
    data = prediction_chart_data(result, request)
    slot = data["slot"]
    labels = {"poi": "POI / third from last", "last": "LAST / last frame", "custom": "CUSTOM"}
    selection = f"{labels[data['prediction_frame']]} | slot {slot} (zero-based), frame {slot + 1} of 8"
    source = f"source array frame {data['frame_index']}" if data["frame_index"] is not None else "source frame index not supplied"
    width, panel_height = 1280, 230
    views = list(request.images)
    if not views or any(len(request.images[v]) != 8 for v in views):
        raise ValueError("Prediction plot requires eight observations per camera")
    camera_rows = math.ceil(len(views) / 4)
    plot_top = 135 + camera_rows * 210
    height = plot_top + panel_height * len(data["series"]) + 88
    canvas = Image.new("RGB", (width, height), "#eef3f8")
    draw = ImageDraw.Draw(canvas)
    fonts = {}

    def font(size, bold=False):
        key = size, bold
        if key not in fonts:
            name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
            try:
                fonts[key] = ImageFont.truetype(name, size)
            except OSError:
                fonts[key] = ImageFont.load_default(size=size)
        return fonts[key]

    def text(xy, value, size=18, fill="#182d45", bold=False, max_width=1232):
        value = str(value)
        chosen = font(size, bold)
        if draw.textlength(value, font=chosen) > max_width:
            while value and draw.textlength(value + "...", font=chosen) > max_width:
                value = value[:-1]
            value += "..."
        draw.text(xy, value, font=chosen, fill=fill)

    text((24, 15), "GRANA | " + " + ".join(data["roles"]) + " prediction", 26, bold=True)
    text((24, 53), selection + " | " + source, 21, "#075e7d", True)
    task = request.compressed_task_prompt if data["roles"] == ["A"] else request.task
    text((24, 87), task or "Eight observed frames per camera", 17)
    text((24, 111), "Cameras below show the selected frame. Charts retain all eight returned predictions.", 15, "#53647a")
    columns = min(4, len(views))
    cell_width = (width - 48) // columns
    for index, view in enumerate(views):
        row, col = divmod(index, 4)
        x, y = 24 + col * cell_width, 138 + row * 210
        image = request.images[view][slot]
        image = image.copy() if isinstance(image, Image.Image) else Image.fromarray(image)
        image.thumbnail((cell_width - 12, 166))
        draw.rounded_rectangle((x, y, x + cell_width - 10, y + 199), radius=8, fill="#14273e")
        canvas.paste(image, (x + (cell_width - 10 - image.width) // 2, y + 25 + (166 - image.height) // 2))
        text((x + 9, y + 3), f"{view} | slot {slot}", 15, "#ffffff", max_width=cell_width - 25)
    for panel, series in enumerate(data["series"]):
        top = plot_top + panel * panel_height
        draw.rounded_rectangle((24, top, width - 24, top + panel_height - 10), radius=10, fill="white")
        text((42, top + 9), series["title"], 20, bold=True, max_width=900)
        selected = series["selected_value"]
        formatted = str(selected) if series["role"] == "C" else f"{selected:.4f}"
        text((970, top + 10), f"{data['prediction_frame'].upper()} = {formatted}", 19, "#075e7d", True, 250)
        x0, x1, y0, y1 = 85, width - 65, top + 48, top + 164
        upper = series["upper"]
        ticks = ((0, upper // 2, upper) if upper > 8 else range(upper + 1)) if series["role"] == "C" else (0, .5, 1)
        for tick in ticks:
            y = y1 - tick / upper * (y1 - y0)
            draw.line((x0, y, x1, y), fill="#dbe3ed", width=1)
            text((39, y - 9), f"{tick:g}", 14)
        points = [(x0 + i / 7 * (x1 - x0), y1 - value / upper * (y1 - y0))
                  for i, value in enumerate(series["values"])]
        marker = points[slot][0]
        draw.line((marker, y0 - 3, marker, y1 + 4), fill="#ed9b21", width=3)
        if series["role"] == "C":
            for left, right in zip(points, points[1:]):
                draw.line((left, (right[0], left[1]), right), fill="#136fa8", width=3)
        else:
            draw.line(points, fill="#136fa8", width=3)
        for index, (x, y) in enumerate(points):
            radius = 6 if index == slot else 3
            draw.ellipse((x - radius, y - radius, x + radius, y + radius),
                         fill="#ed9b21" if index == slot else "#136fa8")
            text((x - 5, y1 + 7), str(index), 14)
            if data["frame_indices"] is not None:
                text((x - 23, y1 + 25), f"f{data['frame_indices'][index]}", 12, "#53647a", max_width=120)
        text((42, top + 201), "Query slot in input order; gold marker = " + data["prediction_frame"].upper(), 12, "#53647a")
    footer = height - 77
    note = (f"D uses stage {data['D_focus_stage']} from C at slot {slot} for every point; values at other slots share that focus."
            if "D" in data["roles"] else "Each plot is labeled with its requested prediction task and selected frame.")
    text((24, footer), note, 15, "#53647a")
    text((24, footer + 24), "Slot positions are not timestamps. Last-frame output is online only if every input frame is already observed.", 15, "#53647a")
    text((24, footer + 48), "Checkpoint weights SHA256: " + (data["weights_sha256"] or "not supplied"), 13, "#53647a")
    png = PngImagePlugin.PngInfo()
    png.add_text("grana_prediction", json.dumps(data, ensure_ascii=False, allow_nan=False))
    png.add_text("Description", selection + " | " + source + " | roles " + ",".join(data["roles"]))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as stream:
        canvas.save(stream, format="PNG", pnginfo=png)
    return data
