#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render frozen HotWheels binary HTTP predictions for A/C/C+D and POI/LAST.

This CPU-only renderer reads recorded HTTP envelopes and cached RGB arrays. It
never starts an inference server. Each declared case produces an H.264 video,
a poster, and an entry in a receipt that hashes and verifies all artifacts.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
from functools import lru_cache
import hashlib
import html
import json
import math
from pathlib import Path
import re
import shutil
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont

WIDTH, HEIGHT = 1440, 1080
BG, PANEL, INK, MUTED = "#f3f6fa", "#ffffff", "#142338", "#5c6c7f"
BLUE, ORANGE, GREEN = "#2563eb", "#e77724", "#079178"
BORDER = "#d9e2ed"
FRAME_SLOTS = {"poi": 5, "last": 7}
SUPPORTED_ROLES = (("A",), ("C",), ("C", "D"))


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    with Path(path).open("x") as output:
        json.dump(value, output, indent=2, ensure_ascii=False, allow_nan=False)
        output.write("\n")


def finite(value, name, *, lower=0.0, upper=None):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if value < lower or (upper is not None and value > upper):
        raise ValueError(f"{name} is out of range")
    return float(value)


def integer(value, name, *, lower=0, upper=None):
    if type(value) is not int or value < lower or (upper is not None and value > upper):
        raise ValueError(f"{name} must be an in-range integer")
    return value


def same_number(actual, expected, name):
    if not math.isclose(finite(actual, name), expected, rel_tol=0, abs_tol=1e-7):
        raise ValueError(f"{name} differs from the selected raw output")


def validate_record(manifest, case, record):
    """Select from the eight raw outputs only after checking frame/focus identity."""
    episode = manifest["episode"]
    roles = case["roles"]
    mode, slot = case["prediction_frame"], case["slot"]
    frame = integer(record["frame_index"], "frame_index", upper=episode["num_frames"] - 1)
    indices = record["frame_indices"]
    if not isinstance(indices, list) or len(indices) != 8:
        raise ValueError("frame_indices must contain exactly eight source frames")
    for value in indices:
        integer(value, "source frame index", upper=episode["num_frames"] - 1)
    if type(record.get("slot")) is not int or record["slot"] != slot or indices[slot] != frame:
        raise ValueError("Selected source frame / slot differs from the recorded request")
    if mode == "last" and any(value > frame for value in indices):
        raise ValueError("LAST input window contains future frames")
    envelope = record["response"]
    request_id = record["request_id"]
    if not isinstance(request_id, str) or not request_id or envelope.get("request_id") != request_id:
        raise ValueError("HTTP request/response identity mismatch")
    if envelope.get("error"):
        raise ValueError("Cannot render a failed HTTP response")
    prediction = envelope["result"]
    if prediction["checkpoint"]["weights_sha256"] != manifest["checkpoint"]["weights_sha256"]:
        raise ValueError("HTTP checkpoint digest differs from the manifest")
    if prediction.get("roles") != roles:
        raise ValueError("HTTP result roles differ from the case")
    if prediction.get("poi_slot") != slot or prediction.get("prediction_frame") != mode:
        raise ValueError("HTTP result prediction frame / slot differs from the case")
    if prediction.get("frame_indices") != indices:
        raise ValueError("HTTP result source frame indices differ from the request")
    stage_count = len(episode["stage_prompts"])
    raw = {}
    for role in roles:
        values = prediction.get("stage", {}).get("ids") if role == "C" else prediction.get("progress", {}).get(role, {}).get("values")
        if not isinstance(values, list) or len(values) != 8:
            raise ValueError(f"Expected eight raw {role} outputs")
        raw[role] = [integer(v, "C stage ID", upper=stage_count - 1) if role == "C"
                     else finite(v, f"{role} progress", upper=1.0) for v in values]
        summary = prediction.get("stage", {}).get("poi_stage_id") if role == "C" else prediction["progress"][role].get("poi_score")
        if summary is not None:
            same_number(summary, raw[role][slot], f"{role} POI summary")
    selected = {role: values[slot] for role, values in raw.items()}
    gt = record["gt"]
    normalized = {"frame_index": frame, "frame_indices": list(indices), "slot": slot,
                  "request_id": request_id, "client_total_ms": finite(record["client_total_ms"], "client_total_ms"),
                  "values": selected,
                  "gt": {"whole_progress": finite(gt["whole_progress"], "GT whole progress", upper=1.0),
                         "stage_id": integer(gt["stage_id"], "GT stage ID", upper=stage_count - 1),
                         "within_stage_progress": finite(gt["within_stage_progress"], "GT stage progress", upper=1.0)}}
    if "D" in roles:
        focus = integer(prediction.get("D_focus_stage"), "D focus", upper=stage_count - 1)
        if focus != selected["C"] or prediction["progress"]["D"].get("focus_stage") != focus:
            raise ValueError("D focus must equal C at the selected frame")
        normalized["D_focus_stage"] = focus
        normalized["D_focus_target"] = (1.0 if focus < gt["stage_id"] else 0.0 if focus > gt["stage_id"]
                                        else normalized["gt"]["within_stage_progress"])
    summary = prediction.get("selected_prediction")
    if summary is not None:
        if summary.get("slot") != slot or summary.get("frame_index", frame) != frame:
            raise ValueError("Selected prediction summary refers to a different frame")
        for role in roles:
            value = summary.get("stage_id") if role == "C" else summary.get("progress", {}).get(role)
            same_number(value, selected[role], f"{role} selected prediction summary")
        if "D" in roles and summary.get("focus_stage", {}).get("D") != normalized["D_focus_stage"]:
            raise ValueError("Selected D summary has a different focus")
    return normalized


def validate_case(manifest, case, payload):
    roles = case.get("roles")
    if not isinstance(roles, list) or tuple(roles) not in SUPPORTED_ROLES:
        raise ValueError("Replay roles must be A, C, or C D")
    mode = case.get("prediction_frame")
    if mode not in FRAME_SLOTS or type(case.get("slot")) is not int or case["slot"] != FRAME_SLOTS[mode]:
        raise ValueError("Case prediction frame and slot must agree: POI=5, LAST=7")
    if payload.get("roles") != roles or payload.get("prediction_frame") != mode:
        raise ValueError("Case file roles / prediction frame differ from the manifest")
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("Case must contain recorded HTTP predictions")
    normalized = [validate_record(manifest, case, record) for record in records]
    frames = [r["frame_index"] for r in normalized]
    if frames != sorted(set(frames)):
        raise ValueError("Selected source frames must be distinct and increasing")
    targets = manifest["sampling"].get("target_frames")
    if targets is not None and frames != targets:
        raise ValueError("Case sample coverage differs from the frozen target_frames")
    if len({r["request_id"] for r in normalized}) != len(normalized):
        raise ValueError("HTTP request IDs must be unique within a case")
    return normalized


def load_inputs(manifest_path):
    manifest = read_json(manifest_path)
    episode = manifest["episode"]
    integer(episode["num_frames"], "num_frames", lower=1)
    stages = episode["stage_prompts"]
    if not isinstance(stages, list) or not 1 <= len(stages) <= 26 or any(not isinstance(s, str) or not s.strip() for s in stages):
        raise ValueError("Expected 1..26 stage descriptions")
    views = episode["view_names"]
    if not isinstance(views, list) or len(views) != 4 or len(set(views)) != 4:
        raise ValueError("This replay layout requires four distinct ordered camera views")
    digest = manifest["checkpoint"]["weights_sha256"]
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("Expected a checkpoint SHA256")
    finite(manifest["sampling"]["source_fps"], "source_fps", lower=1e-9)
    if "A_accuracy_note" in manifest and not isinstance(manifest["A_accuracy_note"], str):
        raise ValueError("A_accuracy_note must be text when supplied")
    cases = []
    seen = set()
    for case in manifest["cases"]:
        name = case["name"]
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name) or name in seen:
            raise ValueError("Case names must be unique safe artifact names")
        seen.add(name)
        path = Path(case["records_path"]).expanduser()
        if not path.is_absolute():
            raise ValueError("Case records_path must be absolute")
        cases.append((case, validate_case(manifest, case, read_json(path))))
    if not cases:
        raise ValueError("Manifest must declare at least one case")
    frame_sets = {tuple(record["frame_index"] for record in records) for _, records in cases}
    if len(frame_sets) != 1:
        raise ValueError("All replay cases must cover the same selected source frames")
    arrays = {}
    for view in views:
        frames = np.load(episode["frame_paths"][view], mmap_mode="r", allow_pickle=False)
        if (frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[0] != episode["num_frames"]
                or min(frames.shape[1:3]) < 1 or frames.shape[-1] != 3):
            raise ValueError(f"Invalid cached uint8 RGB array for {view}")
        expected_shape = episode.get("frame_shapes", {}).get(view)
        if expected_shape is not None and list(frames.shape) != expected_shape:
            raise ValueError(f"Cached RGB shape differs from the manifest for {view}")
        arrays[view] = frames
    return manifest, cases, arrays


@lru_cache(maxsize=32)
def font(size, bold=False):
    filename = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(str(Path("/usr/share/fonts/truetype/dejavu") / filename), size)


def text(draw, xy, value, size=20, fill=INK, bold=False):
    draw.text(xy, str(value), font=font(size, bold), fill=fill)


def wrapped(draw, xy, value, width, size=16, max_lines=3, fill=INK, bold=False):
    face = font(size, bold)
    words, lines, line = str(value).split(), [], ""
    for word in words:
        candidate = (line + " " + word).strip()
        if line and draw.textlength(candidate, font=face) > width:
            lines.append(line)
            line = word
        else:
            line = candidate
    if line:
        lines.append(line)
    for index, line in enumerate(lines[:max_lines]):
        if index == max_lines - 1 and len(lines) > max_lines:
            while line and draw.textlength(line + "…", font=face) > width:
                line = line[:-1]
            line += "…"
        text(draw, (xy[0], xy[1] + index * (size + 6)), line, size, fill, bold)


def curve_values(record, roles):
    if roles == ["A"]:
        return record["gt"]["whole_progress"], record["values"]["A"]
    if roles == ["C"]:
        return record["gt"]["stage_id"], record["values"]["C"]
    return (record["gt"]["stage_id"] + record["gt"]["within_stage_progress"],
            record["values"]["C"] + record["values"]["D"])


def draw_curve(draw, records, index, roles, stage_count):
    left, top, right, bottom = 1036, 613, 1387, 783
    maximum = 1.0 if roles == ["A"] else max(1, stage_count - 1) if roles == ["C"] else stage_count
    draw.rectangle((left, top, right, bottom), fill="#f8fafc", outline=BORDER)
    for value in (0, maximum / 2, maximum):
        y = bottom - (bottom - top) * value / maximum
        draw.line((left, y, right, y), fill="#e6edf4")
        text(draw, (993, y - 9), f"{value:g}", 14, MUTED)
    first, last = records[0]["frame_index"], records[-1]["frame_index"]
    truth, predicted = [], []
    for sample in records[:index + 1]:
        x = left + (right - left) * (sample["frame_index"] - first) / max(1, last - first)
        gt, model = curve_values(sample, roles)
        truth.append((x, bottom - (bottom - top) * gt / maximum))
        predicted.append((x, bottom - (bottom - top) * model / maximum))
    # Straight segments connect only already-revealed samples; no interpolation
    # creates new observations, and no smoothing or future values are displayed.
    if len(truth) > 1:
        draw.line(truth, fill=BLUE, width=3)
        draw.line(predicted, fill=ORANGE, width=3)
    for point, color in ((truth[-1], BLUE), (predicted[-1], ORANGE)):
        draw.ellipse((point[0] - 4, point[1] - 4, point[0] + 4, point[1] + 4), fill=color)
    for frame, x in ((first, left), (last, right)):
        label = str(frame)
        text(draw, (x - draw.textlength(label, font=font(13)) / 2, bottom + 6), label, 13, MUTED)


def render_frame(manifest, case, records, arrays, index, playback_fps=2):
    episode, record = manifest["episode"], records[index]
    roles, mode, slot = case["roles"], case["prediction_frame"], case["slot"]
    frame, gt, values = record["frame_index"], record["gt"], record["values"]
    count = len(episode["stage_prompts"])
    canvas = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, WIDTH, 98), fill=INK)
    text(draw, (27, 13), "HOTWHEELS  /  BINARY HTTP REPLAY", 28, "white", True)
    wrapped(draw, (29, 55), f"{manifest['model_label']} · checkpoint step {manifest['checkpoint']['step']}",
            1100, 17, 1, "#c2d4ec")
    text(draw, (1160, 56), f"SAMPLE {index + 1:02d} / {len(records)}", 16, "#c2d4ec", True)
    text(draw, (28, 111), f"Roles {' + '.join(roles)}  ·  {mode.upper()}  ·  slot {slot}  ·  source frame {frame} / {episode['num_frames'] - 1}", 23, INK, True)
    for view, (x, y) in zip(episode["view_names"], ((25, 155), (500, 155), (25, 514), (500, 514)), strict=True):
        image = Image.fromarray(np.asarray(arrays[view][frame]))
        image.thumbnail((450, 337), Image.Resampling.LANCZOS)
        draw.rounded_rectangle((x, y, x + 450, y + 337), radius=8, fill="#111d2d")
        canvas.paste(image, (x + (450 - image.width) // 2, y + (337 - image.height) // 2))
        label_width = int(draw.textlength(view.upper(), font=font(16, True))) + 24
        draw.rounded_rectangle((x + 8, y + 8, x + label_width + 8, y + 38), radius=5, fill=INK)
        text(draw, (x + 20, y + 12), view.upper(), 16, "white", True)
    draw.rounded_rectangle((975, 155, 1415, 851), radius=14, fill=PANEL, outline=BORDER, width=2)
    text(draw, (996, 174), "ACTUAL HTTP PREDICTION", 16, MUTED, True)
    title = "A · whole-task progress" if roles == ["A"] else "C · predicted stage" if roles == ["C"] else "C → D · stage + progress"
    text(draw, (996, 204), title, 23, INK, True)
    text(draw, (996, 247), "GROUND TRUTH", 14, MUTED, True)
    text(draw, (1210, 247), "MODEL", 14, MUTED, True)
    if roles == ["A"]:
        text(draw, (996, 273), f"{gt['whole_progress']:.3f}", 32, BLUE, True)
        text(draw, (1210, 273), f"{values['A']:.3f}", 32, ORANGE, True)
        text(draw, (996, 335), f"Absolute error {abs(values['A'] - gt['whole_progress']):.3f}", 19, MUTED)
        text(draw, (996, 385), f"GT stage {gt['stage_id']} / {count - 1}", 19, BLUE, True)
        wrapped(draw, (996, 421), episode["stage_prompts"][gt["stage_id"]], 392, 16, 3)
        plot_title, axis_label = "WHOLE-TASK PROGRESS", "Progress 0–1 · source frame index"
    else:
        text(draw, (996, 273), f"{gt['stage_id']:02d} / {count - 1}", 32, BLUE, True)
        color = GREEN if values["C"] == gt["stage_id"] else ORANGE
        text(draw, (1210, 273), f"{values['C']:02d} / {count - 1}", 32, color, True)
        if "D" in roles:
            text(draw, (996, 328), f"D · focus stage {record['D_focus_stage']}", 16, MUTED, True)
            text(draw, (996, 354), f"{values['D']:.3f}", 30, ORANGE, True)
            text(draw, (1153, 364), f"focus GT {record['D_focus_target']:.3f}", 18, BLUE)
            text(draw, (996, 404), f"GT stage progress {gt['within_stage_progress']:.3f}", 16, MUTED)
            text(draw, (996, 439), "PREDICTED STAGE ACTION", 14, MUTED, True)
            wrapped(draw, (996, 465), episode["stage_prompts"][values["C"]], 392, 16, 2)
            plot_title, axis_label = "ACCUMULATED C + D", "Stage units · source frame index"
        else:
            text(draw, (996, 336), "MATCH" if values["C"] == gt["stage_id"] else "DIFFERENT STAGE", 20, color, True)
            text(draw, (996, 385), "PREDICTED STAGE ACTION", 14, MUTED, True)
            wrapped(draw, (996, 416), episode["stage_prompts"][values["C"]], 392, 16, 4)
            plot_title, axis_label = "STAGE ID", "Zero-based stage ID · source frame index"
    draw.line((996, 531, 1392, 531), fill=BORDER, width=2)
    text(draw, (996, 547), plot_title, 17, MUTED, True)
    text(draw, (996, 579), "● GT", 15, BLUE, True)
    text(draw, (1102, 579), "● HTTP prediction", 15, ORANGE, True)
    draw_curve(draw, records, index, roles, count)
    text(draw, (996, 821), axis_label, 14, MUTED)
    draw.line((25, 873, 1415, 873), fill=BORDER, width=2)
    text(draw, (28, 886), f"Input slots 0–7: {record['frame_indices']}  ·  selected slot {slot} → frame {frame}", 18, INK, True)
    if mode == "last":
        window_note = f"LAST: no future input frames; maximum observed frame {max(record['frame_indices'])} ≤ selected frame {frame}."
    else:
        future = sum(value > frame for value in record["frame_indices"])
        window_note = f"POI: offline slot 5 convention; this window contains {future} later observations (up to frame {max(record['frame_indices'])})."
    text(draw, (28, 918), window_note, 17, MUTED)
    text(draw, (28, 948), f"Sampled offline replay at {playback_fps:g} fps · {len(records)} samples / {len(records) / playback_fps:g} s. Playback rate is not model throughput.", 17, MUTED)
    text(draw, (28, 978), f"Client request → response {record['client_total_ms']:.1f} ms · source {manifest['sampling']['source_fps']:g} fps · GT is display-only.", 17, MUTED)
    note = manifest.get("A_accuracy_note", "A reports whole-task progress at the selected frame.") if roles == ["A"] else "Stage IDs are zero-based. D target is 1 / within-stage GT / 0 when focus is before / at / after the GT stage." if "D" in roles else "Stage IDs are zero-based. This C-only request does not compute a progress head."
    wrapped(draw, (28, 1008), note, 1384, 16, 1, MUTED)
    wrapped(draw, (28, 1040), f"Task: {episode['task']} · trajectory {episode['trajectory_id']}", 1384, 15, 1, MUTED)
    return canvas


def executable(name):
    path = shutil.which(name)
    if path is None:
        raise RuntimeError(f"{name} is required")
    return path


def verify_video(path, frames, playback_fps, ffmpeg, ffprobe):
    info = json.loads(subprocess.check_output([ffprobe, "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)]))
    streams = [stream for stream in info["streams"] if stream["codec_type"] == "video"]
    if len(streams) != 1:
        raise ValueError("Expected exactly one video stream")
    stream = streams[0]
    duration = float(info["format"]["duration"])
    if (stream["codec_name"] != "h264" or stream["pix_fmt"] != "yuv420p"
            or (int(stream["width"]), int(stream["height"])) != (WIDTH, HEIGHT)
            or int(stream["nb_frames"]) != frames
            or not math.isclose(float(Fraction(stream["avg_frame_rate"])), playback_fps, abs_tol=1e-6)
            or not math.isclose(duration, frames / playback_fps, abs_tol=0.01)):
        raise ValueError("Encoded video failed format, dimensions, frame-count, FPS or duration verification")
    subprocess.run([ffmpeg, "-v", "error", "-xerror", "-i", str(path), "-f", "null", "-"], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    return {"codec": stream["codec_name"], "pixel_format": stream["pix_fmt"], "frames": frames,
            "width": WIDTH, "height": HEIGHT, "fps": stream["avg_frame_rate"], "duration_seconds": duration,
            "full_decode": "passed"}


def render_case(manifest, case, records, arrays, output, playback_fps, ffmpeg, ffprobe):
    video, poster = output / f"{case['name']}.mp4", output / f"{case['name']}.png"
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-n", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{WIDTH}x{HEIGHT}", "-r", str(playback_fps), "-i", "-", "-an", "-c:v", "libx264",
               "-threads", "4", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(video)]
    with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE) as process:
        try:
            for index in range(len(records)):
                frame = render_frame(manifest, case, records, arrays, index, playback_fps)
                if index == len(records) // 2:
                    frame.save(poster)
                process.stdin.write(frame.tobytes())
            process.stdin.close()
            error = process.stderr.read().decode("utf-8", errors="replace")
            if process.wait() != 0:
                raise RuntimeError(f"ffmpeg encoding failed: {error}")
        except BaseException:
            process.kill()
            process.wait()
            raise
    verification = verify_video(video, len(records), playback_fps, ffmpeg, ffprobe)
    return {"name": case["name"], "roles": case["roles"], "prediction_frame": case["prediction_frame"], "slot": case["slot"],
            "video": str(video), "poster": str(poster), "video_sha256": sha256_file(video), "poster_sha256": sha256_file(poster),
            "records_path": str(case["records_path"]), "records_sha256": sha256_file(case["records_path"]),
            "validation": {"checkpoint_and_request_ids": "passed", "selected_frame_and_raw8": "passed",
                           "last_has_no_future_frames": "passed" if case["prediction_frame"] == "last" else "not applicable",
                           "D_focus_matches_selected_C": "passed" if "D" in case["roles"] else "not applicable"},
            **verification}


def write_index(output, manifest, artifacts, playback_fps):
    cards = []
    for item in artifacts:
        title = html.escape(f"{' + '.join(item['roles'])} · {item['prediction_frame'].upper()} · slot {item['slot']}")
        video, poster = html.escape(Path(item["video"]).name, quote=True), html.escape(Path(item["poster"]).name, quote=True)
        note = manifest.get("A_accuracy_note", "Whole-task progress at the selected frame.") if item["roles"] == ["A"] else "D uses C at the selected frame." if "D" in item["roles"] else "Stage prediction only."
        cards.append(f'<article><h2>{title}</h2><video controls preload="metadata" poster="{poster}" src="{video}"></video><p>{html.escape(note)}</p><a href="{video}">Download MP4</a></article>')
    content = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>HotWheels binary HTTP replay</title><style>body{{margin:32px;background:#f3f6fa;color:#142338;font:16px/1.6 system-ui}}main{{max-width:1500px;margin:auto}}section{{display:grid;grid-template-columns:repeat(auto-fit,minmax(480px,1fr));gap:24px}}article{{padding:18px;background:white;border-radius:14px}}video{{width:100%}}a{{color:#2563eb}}@media(max-width:600px){{body{{margin:14px}}section{{grid-template-columns:1fr}}}}</style>
<main><h1>{html.escape(manifest['model_label'])} · binary HTTP replay</h1><p>Sampled offline playback at {playback_fps:g} fps; playback rate is not model throughput. LAST windows contain no future frames. POI uses slot 5 and may include later observations. Ground truth is display-only. Curves reveal raw samples without smoothing.</p><section>{''.join(cards)}</section><p><a href="render_receipt.json">Hashes and full-decode verification</a></p></main></html>'''
    (output / "index.html").write_text(content)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New output directory; existing artifacts are never replaced")
    parser.add_argument("--playback-fps", type=float, default=2.0)
    args = parser.parse_args(argv)
    playback_fps = finite(args.playback_fps, "playback_fps", lower=0.01, upper=60)
    manifest_path, output = args.manifest.expanduser().resolve(), args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    manifest, cases, arrays = load_inputs(manifest_path)
    ffmpeg, ffprobe = executable("ffmpeg"), executable("ffprobe")
    output.mkdir(parents=True, exist_ok=False)
    artifacts = []
    for case, records in cases:
        artifact = render_case(manifest, case, records, arrays, output, playback_fps, ffmpeg, ffprobe)
        artifacts.append(artifact)
        print(json.dumps({"rendered": artifact["video"], "frames": artifact["frames"], "full_decode": "passed"}), flush=True)
    write_index(output, manifest, artifacts, playback_fps)
    receipt = {"schema": "grana-http-replay-render-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
               "manifest": str(manifest_path), "manifest_sha256": sha256_file(manifest_path),
               "renderer_sha256": sha256_file(__file__), "checkpoint": manifest["checkpoint"],
               "source_frame_files": {view: {"path": path, "sha256": sha256_file(path)}
                                      for view, path in manifest["episode"]["frame_paths"].items()},
               "playback_fps": playback_fps, "curve_policy": "raw selected outputs, reveal through current sample; no smoothing",
               "transport": "recorded binary HTTP requests", "A_accuracy_note": manifest.get("A_accuracy_note"),
               "index_sha256": sha256_file(output / "index.html"), "cases": artifacts}
    write_json(output / "render_receipt.json", receipt)
    print(json.dumps({"receipt": str(output / "render_receipt.json"), "videos": len(artifacts)}), flush=True)
    return receipt


if __name__ == "__main__":
    main()
