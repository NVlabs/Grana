#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render paired POI/LAST error investigations from frozen evaluation records.

The ranking uses the original repeated-sampling trajectory RMSE. Curves use
the union of unique measured targets, with identical repeat predictions checked
before deduplication. Camera tiles and bright cursors identify the same target
in both modes. No model calls, smoothing, or additional predictions occur.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import html
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image, ImageDraw

from scripts.inference import summarize_http_frames as summary
from scripts.inference.render_http_replay import font, sha256_file

WIDTH, HEIGHT = 1920, 1280
BG, INK, MUTED = "#f2f5f9", "#172638", "#5b6878"
POI, LAST, GT, CURSOR = "#8054b3", "#187ba2", "#283c4a", "#e3aa16"
MODES = ("poi", "last")
PLOT = (1052, 182, 1870, 441)
LOCAL_PLOT = (1052, 510, 1870, 683)


def dump_new(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def text(draw, xy, value, size=20, color=INK, bold=False):
    draw.text(xy, str(value), fill=color, font=font(size, bold))


def wrap(draw, xy, value, width, size=16, lines=3, color=INK):
    face, row, rendered = font(size), "", []
    for word in value.split():
        candidate = f"{row} {word}".strip()
        if row and draw.textlength(candidate, font=face) > width:
            rendered.append(row)
            row = word
        else:
            row = candidate
    if row:
        rendered.append(row)
    for index, line in enumerate(rendered[:lines]):
        if index == lines - 1 and len(rendered) > lines:
            while line and draw.textlength(line + "…", font=face) > width:
                line = line[:-1]
            line += "…"
        text(draw, (xy[0], xy[1] + index * (size + 5)), line, size, color)


def select_cases(plan, records, report, top):
    """Rank original RMSE gaps and retain exact unique paired measurements."""
    if type(top) is not int or top < 1:
        raise ValueError("top must be a positive integer")
    if report["checkpoint"]["weights_sha256"] != plan["checkpoint"]["weights_sha256"]:
        raise ValueError("Summary and experiment checkpoints differ")
    ranked = sorted(report["trajectories"],
                    key=lambda row: (-row["delta_last_minus_poi"]["normalized_rmse"], row["trajectory_id"]))
    ranked = [row for row in ranked if row["delta_last_minus_poi"]["normalized_rmse"] > 0][:top]
    if len(ranked) != top:
        raise ValueError("Not enough trajectories with a positive measured POI advantage")
    planned = {row["key"]: row for row in plan["trajectories"]}
    cases = []
    for rank, row in enumerate(ranked, 1):
        trajectory = planned[row["key"]]
        grouped = defaultdict(list)
        for (seed, key, kind, frame, mode), record in records.items():
            if key == row["key"] and kind == "quality":
                grouped[frame, mode].append(record)
        frames = sorted(frame for frame, mode in grouped if mode == "poi")
        if set(frames) != {frame for frame, mode in grouped if mode == "last"}:
            raise ValueError("POI and LAST must cover the same physical targets")
        points = []
        for frame in frames:
            by_mode, multiplicities = {}, []
            for mode in MODES:
                observations = grouped[frame, mode]
                if len({r["prediction_sha256"] for r in observations}) != 1:
                    raise ValueError("Cannot deduplicate nonidentical repeated predictions")
                if len({r["input_sha256"] for r in observations}) != 1:
                    raise ValueError("Repeated target inputs differ")
                measured = min(observations, key=lambda r: plan["seeds"].index(r["seed"]))
                slot = 5 if mode == "poi" else 7
                c, d = measured["C_ids"][slot], measured["D_values"][slot]
                by_mode[mode] = {"C": c, "D": d, "value": c + d, "slot": slot,
                                 "frame_indices": measured["frame_indices"],
                                 "request_ids": [r["request_id"] for r in observations],
                                 "input_sha256": measured["input_sha256"],
                                 "prediction_sha256": measured["prediction_sha256"]}
                multiplicities.append(len(observations))
            if multiplicities[0] != multiplicities[1]:
                raise ValueError("POI and LAST repeat counts differ")
            gt = grouped[frame, "poi"][0]["gt"]
            target = gt["stage_id"] + gt["within_stage_progress"]
            for mode in MODES:
                by_mode[mode]["error"] = by_mode[mode]["value"] - target
                by_mode[mode]["squared_error"] = by_mode[mode]["error"] ** 2
            gap = by_mode["last"]["squared_error"] - by_mode["poi"]["squared_error"]
            points.append({"frame": frame, "seconds": frame / trajectory["source_fps"],
                           "gt": gt, "target": target, "repeat_count": multiplicities[0],
                           "squared_error_gap": gap,
                           "weighted_squared_error_gap": gap * multiplicities[0], **by_mode})
        count, scale = sum(p["repeat_count"] for p in points), trajectory["stage_count"]
        original, unique = {}, {}
        for mode in MODES:
            original[mode] = math.sqrt(math.fsum(p[mode]["squared_error"] * p["repeat_count"] for p in points) / count) / scale
            unique[mode] = math.sqrt(math.fsum(p[mode]["squared_error"] for p in points) / len(points)) / scale
            if not math.isclose(original[mode], row["metrics"][mode]["normalized_rmse"], rel_tol=0, abs_tol=1e-12):
                raise ValueError("Ranking RMSE does not reproduce the measured repeated-sampling result")
        total_gap = math.fsum(p["weighted_squared_error_gap"] for p in points)
        events = sorted(points, key=lambda p: p["weighted_squared_error_gap"], reverse=True)[:3]
        event_rows = []
        for event_rank, point in enumerate(events, 1):
            after = {}
            for mode in MODES:
                kept = [p for p in points if p["frame"] != point["frame"]]
                after[mode] = math.sqrt(math.fsum(p[mode]["squared_error"] * p["repeat_count"] for p in kept)
                                        / sum(p["repeat_count"] for p in kept)) / scale
            event_rows.append({"rank": event_rank, "frame": point["frame"], "seconds": point["seconds"],
                "gt_stage": point["gt"]["stage_id"], "gt_value": point["target"],
                "poi_C": point["poi"]["C"], "poi_D": point["poi"]["D"],
                "last_C": point["last"]["C"], "last_D": point["last"]["D"],
                "repeat_count": point["repeat_count"], "squared_error_gap_stage_units": point["squared_error_gap"],
                "weighted_gap_stage_units": point["weighted_squared_error_gap"],
                "share_of_net_squared_error_gap": point["weighted_squared_error_gap"] / total_gap,
                "diagnostic_rmse_if_this_target_removed": after})
        same_stage_gap = math.fsum(p["weighted_squared_error_gap"] for p in points if p["poi"]["C"] == p["last"]["C"])
        cases.append({"rank": rank, "trajectory": trajectory, "points": points,
                      "original_normalized_rmse": original, "original_samples_per_mode": count,
                      "unique_normalized_rmse": unique, "unique_targets": len(points),
                      "gap_analysis": {"net_weighted_squared_error_gap_stage_units": total_gap,
                          "gap_at_same_C_stage": same_stage_gap, "gap_at_different_C_stages": total_gap - same_stage_gap,
                          "note": "A point can exceed 100% of the net gap because other points favor LAST. Removing points is diagnostic only; primary metrics keep every sample.",
                          "top_points": event_rows}})
    return cases


def map_xy(frame, value, bounds, low, high, xmax):
    left, top, right, bottom = bounds
    return (left + (right - left) * frame / max(1, xmax),
            bottom - (bottom - top) * (value - low) / max(1e-9, high - low))


def plot(draw, bounds, points, gt_values, first, last, *, stage_count, current=None, source_fps=10):
    left, top, right, bottom = bounds
    draw.rectangle(bounds, fill="#fbfcfe", outline="#cad4e0")
    if current is None:
        low, high = 0, stage_count
    else:
        visible = [p for p in points if first <= p["frame"] <= last]
        values = [v for p in visible for v in (p["target"], p["poi"]["value"], p["last"]["value"])]
        values += gt_values[first:last + 1]
        low, high = max(0, math.floor(min(values) * 2) / 2 - .25), min(stage_count, math.ceil(max(values) * 2) / 2 + .25)
        if high - low < .5:
            high = min(stage_count, low + .5)
    def xy(frame, value):
        x, y = map_xy(frame - first, value, bounds, low, high, last - first)
        return int(round(x)), int(round(y))
    for fraction in (0, .5, 1):
        value = low + (high - low) * fraction
        y = xy(first, value)[1]
        draw.line((left, y, right, y), fill="#e4eaf1")
        text(draw, (left - 48, y - 9), f"{value:g}", 15, MUTED)
    gt_line = [xy(frame, gt_values[frame]) for frame in range(first, last + 1)]
    draw.line(gt_line, fill=GT, width=2)
    for mode, color in (("poi", POI), ("last", LAST)):
        coordinates = [xy(p["frame"], p[mode]["value"]) for p in points if first <= p["frame"] <= last]
        if len(coordinates) > 1:
            draw.line(coordinates, fill=color, width=3)
        for x, y in coordinates:
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=color)
    for fraction in (0, .5, 1):
        frame = round(first + (last - first) * fraction)
        x = xy(frame, low)[0]
        label = f"{frame / source_fps:.1f}s"
        text(draw, (min(right - 55, max(left, x - 28)), bottom + 7), label, 15, MUTED)
    return xy


def cursor(draw, bounds, x, point, xy):
    left, top, right, bottom = bounds
    draw.rectangle((x - 4, top, x + 4, bottom), fill="#f5e6b9")
    draw.line((x, top, x, bottom), fill=CURSOR, width=3)
    draw.polygon([(x - 7, top - 9), (x + 7, top - 9), (x, top)], fill=CURSOR)
    for role, value, color in (("gt", point["target"], GT), ("poi", point["poi"]["value"], POI),
                               ("last", point["last"]["value"], LAST)):
        _, y = xy(point["frame"], value)
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color, outline="white", width=1)


def base_canvas(case, metadata, request):
    canvas = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(canvas)
    trajectory = case["trajectory"]
    text(draw, (24, 18), f"POI vs LAST · HotWheels D100 4B · step 1000 · largest POI advantage #{case['rank']}", 27, INK, True)
    text(draw, (24, 55), f"Trajectory {trajectory['trajectory_id']} · C→D · CTSP(16,4) · binary HTTP + GPU resize", 19, MUTED)
    rmse = case["original_normalized_rmse"]
    text(draw, (24, 88), f"Original 3-round normalized RMSE ({case['original_samples_per_mode']} samples): POI {rmse['poi']:.6f}  |  LAST {rmse['last']:.6f}", 20, INK, True)
    text(draw, (1000, 132), "FULL TRAJECTORY · C + D (stage units)", 20, INK, True)
    for x, label, color in ((1462, "GT", GT), (1555, "POI [5]", POI), (1708, "LAST [7]", LAST)):
        draw.line((x, 145, x + 22, 145), fill=color, width=4)
        text(draw, (x + 30, 132), label, 18, color, True)
    gt_values = [c + d for c, d in zip(metadata["stage_ids"], metadata["within_stage_progress"], strict=True)]
    plot(draw, PLOT, case["points"], gt_values, 0, trajectory["num_frames"] - 1,
         stage_count=trajectory["stage_count"], source_fps=trajectory["source_fps"])
    for x, y, view in ((24, 166, "top"), (492, 166, "left"), (24, 546, "right"), (492, 546, "bottom")):
        text(draw, (x, y - 28), view.upper(), 17, MUTED, True)
    text(draw, (24, 902), "POI LOOKAHEAD · actual TOP-camera frames", 19, INK, True)
    text(draw, (24, 1231), "Extra POI images are shown for offline inspection; LAST receives no future frames.", 17, MUTED)
    return canvas, gt_values


def render_frame(case, index, arrays, metadata, request, base, gt_values, fps, *, hold=False, hold_seconds=2):
    canvas = base.copy()
    draw = ImageDraw.Draw(canvas)
    point, trajectory = case["points"][index], case["trajectory"]
    frame, scale = point["frame"], trajectory["stage_count"]
    stamp = f"SOURCE {point['seconds']:.1f}s · FRAME {frame} / {trajectory['num_frames'] - 1}"
    text(draw, (24, 119), stamp, 18, INK, True)
    for view, x, y in (("top", 24, 166), ("left", 492, 166), ("right", 24, 546), ("bottom", 492, 546)):
        picture = Image.fromarray(arrays[view][frame]).resize((456, 342), Image.Resampling.LANCZOS)
        canvas.paste(picture, (x, y))
    xy = lambda f, v: tuple(map(round, map_xy(f, v, PLOT, 0, scale, trajectory["num_frames"] - 1)))
    cursor(draw, PLOT, xy(frame, 0)[0], point, xy)
    text(draw, (1000, 476), f"LOCAL DETAIL · ±12 seconds around {point['seconds']:.1f}s", 19, INK, True)
    radius = round(12 * trajectory["source_fps"])
    first, last = max(0, frame - radius), min(trajectory["num_frames"] - 1, frame + radius)
    local_xy = plot(draw, LOCAL_PLOT, case["points"], gt_values, first, last,
                    stage_count=scale, current=frame, source_fps=trajectory["source_fps"])
    cursor(draw, LOCAL_PLOT, local_xy(frame, point["target"])[0], point, local_xy)
    for x, title, color, stage, progress, value, error in (
            (1000, "GT", GT, point["gt"]["stage_id"], point["gt"]["within_stage_progress"], point["target"], 0),
            (1300, "POI · SLOT 5", POI, point["poi"]["C"], point["poi"]["D"], point["poi"]["value"], point["poi"]["error"]),
            (1600, "LAST · SLOT 7", LAST, point["last"]["C"], point["last"]["D"], point["last"]["value"], point["last"]["error"])):
        draw.rounded_rectangle((x, 730, x + 280, 864), radius=10, fill="white", outline="#d7e0eb")
        text(draw, (x + 12, 740), title, 18, color, True)
        text(draw, (x + 12, 770), f"C {stage}  +  D {progress:.3f}", 21, color, True)
        text(draw, (x + 12, 801), f"Progress {value:.3f} / {scale}", 17, INK)
        text(draw, (x + 12, 830), f"|error| {abs(error) / scale:.5f} of task", 16, MUTED)
    for x, label, color, stage in ((1000, "GT action", GT, point["gt"]["stage_id"]),
                                  (1300, "POI action", POI, point["poi"]["C"]),
                                  (1600, "LAST action", LAST, point["last"]["C"])):
        text(draw, (x, 880), f"{label} · stage {stage}", 16, color, True)
        wrap(draw, (x, 907), request["stage_prompts"][stage], 278, 15, 4)
    for column, slot in enumerate((5, 6, 7)):
        source = point["poi"]["frame_indices"][slot]
        x, y = 24 + column * 312, 962
        text(draw, (x, 934), f"{(source - frame) / trajectory['source_fps']:+.1f}s · frame {source}", 17,
             POI if slot > 5 else INK, True)
        canvas.paste(Image.fromarray(arrays["top"][source]).resize((300, 225), Image.Resampling.LANCZOS), (x, y))
    text(draw, (1000, 1016), f"POI [5] window: {point['poi']['frame_indices']}", 16, POI, True)
    text(draw, (1000, 1043), f"LAST [7] window: {point['last']['frame_indices']}", 16, LAST, True)
    gap = point["squared_error_gap"]
    text(draw, (1000, 1083), f"Point squared-error gap (LAST − POI): {gap:+.4f} stage²", 17, INK, True)
    event = next((event for event in case["gap_analysis"]["top_points"] if event["frame"] == frame), None)
    if event:
        draw.rounded_rectangle((1000, 1115, 1880, 1179), radius=8, fill="#fff3cf", outline="#dba426", width=2)
        text(draw, (1013, 1124), f"ERROR HOTSPOT #{event['rank']} · " + (f"{hold_seconds:g}-second inspection pause" if hold else "recorded target"), 20, INK, True)
        text(draw, (1013, 1151), f"C-stage choice: GT {point['gt']['stage_id']} | POI {point['poi']['C']} | LAST {point['last']['C']}", 17, INK)
    else:
        wrap(draw, (1000, 1122), "Inspect stage selection as well as D progress. A large jump in C can dominate RMSE even when most samples agree.", 870, 17, 2, MUTED)
    text(draw, (1000, 1200), f"Sampled replay · {fps:g} fps · {case['unique_targets']} unique measured targets", 17, MUTED)
    text(draw, (1000, 1227), "Markers are measured; lines join samples. Full curves are offline analysis.", 16, MUTED)
    text(draw, (24, 1256), "POI vs LAST changes the input window and selected slot together; this comparison does not isolate the causal effect of lookahead.", 15, MUTED)
    return canvas


def verify_video(path, count, fps):
    info = json.loads(subprocess.check_output([shutil.which("ffprobe"), "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)]))
    streams = [row for row in info["streams"] if row["codec_type"] == "video"]
    if len(streams) != 1:
        raise ValueError("Expected one video stream")
    stream = streams[0]
    if (stream["codec_name"] != "h264" or stream["pix_fmt"] != "yuv420p"
            or (stream["width"], stream["height"]) != (WIDTH, HEIGHT)
            or int(stream["nb_frames"]) != count or float(Fraction(stream["avg_frame_rate"])) != fps
            or not math.isclose(float(info["format"]["duration"]), count / fps, abs_tol=.01)):
        raise ValueError("Video dimensions/count/FPS/format verification failed")
    subprocess.run([shutil.which("ffmpeg"), "-v", "error", "-xerror", "-i", str(path), "-f", "null", "-"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)
    return {"width": WIDTH, "height": HEIGHT, "frames": count, "fps": fps, "duration_seconds": count / fps,
            "codec": "h264", "pixel_format": "yuv420p", "full_decode": "passed"}


def render_case(case, output, fps, hold_seconds):
    trajectory = case["trajectory"]
    metadata = json.loads(Path(trajectory["metadata_path"]).read_text())
    request = json.loads(Path(trajectory["request_path"]).read_text())
    arrays = {view: np.load(path, mmap_mode="r", allow_pickle=False) for view, path in request["views"].items()}
    if set(arrays) != {"top", "left", "right", "bottom"}:
        raise ValueError("Paired layout requires top/left/right/bottom cameras")
    for view, array in arrays.items():
        if array.dtype != np.uint8 or list(array.shape) != trajectory["sources"][view]["shape"]:
            raise ValueError("Unexpected cached RGB source")
        stat = Path(request["views"][view]).stat()
        source = trajectory["sources"][view]
        if (stat.st_size, stat.st_mtime_ns) != (source["size_bytes"], source["mtime_ns"]):
            raise ValueError("RGB source changed after evaluation")
    stem = f"poi_advantage_{case['rank']:02d}_{trajectory['trajectory_id']}"
    video, poster = output / (stem + ".mp4"), output / (stem + ".png")
    base, gt_values = base_canvas(case, metadata, request)
    events = {row["frame"]: row for row in case["gap_analysis"]["top_points"]}
    pauses = round(hold_seconds * fps)
    command = [shutil.which("ffmpeg"), "-hide_banner", "-loglevel", "error", "-n", "-f", "rawvideo",
               "-pix_fmt", "rgb24", "-s", f"{WIDTH}x{HEIGHT}", "-r", str(fps), "-i", "-", "-an",
               "-c:v", "libx264", "-threads", "4", "-preset", "medium", "-crf", "20",
               "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(video)]
    count, positions = 0, []
    with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE) as process:
        try:
            for index, point in enumerate(case["points"]):
                event = events.get(point["frame"])
                frame = render_frame(case, index, arrays, metadata, request, base, gt_values, fps,
                                     hold=bool(event and pauses), hold_seconds=hold_seconds)
                if event:
                    positions.append({"event_rank": event["rank"], "source_frame": point["frame"],
                                      "video_seconds": count / fps})
                    if event["rank"] == 1:
                        frame.save(poster)
                repeats = 1 + (pauses if event else 0)
                raw = frame.tobytes()
                for _ in range(repeats):
                    process.stdin.write(raw)
                count += repeats
            process.stdin.close()
            error = process.stderr.read().decode(errors="replace")
            if process.wait() != 0:
                raise RuntimeError(error)
        except BaseException:
            process.kill()
            process.wait()
            raise
    verification = verify_video(video, count, fps)
    result = {"rank": case["rank"], "trajectory_id": trajectory["trajectory_id"],
              "group": trajectory["group"], "video": str(video), "poster": str(poster),
              "video_sha256": sha256_file(video), "poster_sha256": sha256_file(poster),
              "original_normalized_rmse": case["original_normalized_rmse"],
              "original_samples_per_mode": case["original_samples_per_mode"],
              "displayed_unique_target_rmse": case["unique_normalized_rmse"],
              "displayed_unique_targets": case["unique_targets"], "hotspot_video_positions": positions,
              "gap_analysis": case["gap_analysis"], **verification}
    dump_new(output / (stem + ".json"), {**result, "points": case["points"]})
    print(json.dumps({"rendered": str(video), "duration_seconds": verification["duration_seconds"]}), flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top", type=int, default=3)
    parser.add_argument("--playback-fps", type=float, default=8)
    parser.add_argument("--hotspot-hold-seconds", type=float, default=2)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args(argv)
    if not 0 < args.playback_fps <= 60 or not 0 <= args.hotspot_hold_seconds <= 10 or args.workers < 1:
        raise ValueError("Invalid FPS, hotspot hold, or worker count")
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise RuntimeError("ffmpeg and ffprobe are required")
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    experiment = args.experiment.expanduser().resolve()
    plan, records, sources = summary.load_experiment(experiment)
    report = json.loads((experiment / "summary" / "summary.json").read_text())
    if report["sources"]["plan_sha256"] != sources["plan_sha256"]:
        raise ValueError("Summary came from another experiment plan")
    cases = select_cases(plan, records, report, args.top)
    output.mkdir(parents=True)
    with ThreadPoolExecutor(max_workers=min(args.workers, len(cases))) as pool:
        artifacts = list(pool.map(lambda case: render_case(case, output, args.playback_fps, args.hotspot_hold_seconds), cases))
    receipt = {"schema": "grana-paired-frame-comparison-render-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
               "checkpoint": plan["checkpoint"], "sources": sources, "renderer_sha256": sha256_file(__file__),
               "selection": "Top positive LAST-minus-POI normalized RMSE gaps from original repeated-sampling trajectory metrics",
               "curves": "Unique recorded quality targets; equal repeated predictions validated before deduplication. No smoothing or model calls.",
               "playback": "One display frame per unique target, plus fixed inspection holds at the three largest weighted squared-error gaps. Not source-speed playback.",
               "causal_limit": "POI and LAST differ in window composition and selected slot; observed differences do not isolate the causal effect of future frames.",
               "artifacts": artifacts}
    dump_new(output / "render_receipt.json", receipt)
    cards = []
    for artifact in artifacts:
        title = html.escape(artifact["trajectory_id"])
        video, poster = Path(artifact["video"]).name, Path(artifact["poster"]).name
        cards.append(f'<article><h2>#{artifact["rank"]}: {title}</h2><video controls preload="metadata" src="{video}" poster="{poster}"></video></article>')
    (output / "index.html").write_text('<!doctype html><meta charset="utf-8"><title>POI versus LAST</title>'
        '<style>body{background:#f2f5f9;color:#172638;font:16px system-ui;margin:30px}main{max-width:1500px;margin:auto}'
        'article{background:white;border-radius:12px;padding:20px;margin-bottom:24px}video{width:100%}</style>'
        '<main><h1>Three largest POI RMSE advantages</h1><p>Both recorded prediction curves and GT share one time axis. '
        'The bright cursor identifies the camera target. Error hotspots pause for inspection. Playback is sampled, not original speed.</p>'
        + ''.join(cards) + '<a href="render_receipt.json">Measured error contributions and video verification</a></main>')
    print(json.dumps({"receipt": str(output / "render_receipt.json"), "videos": len(artifacts)}), flush=True)
    return receipt


if __name__ == "__main__":
    main()
