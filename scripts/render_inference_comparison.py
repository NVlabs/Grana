#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render sampled, cached-RGB inference comparisons without running a model.

Inputs are the frozen comparison manifest and per-arm/per-episode JSON results.
Only presentation artifacts are written: H.264 videos, posters, an HTML index,
and a verification receipt. Original NPY camera arrays remain memory-mapped.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from hashlib import sha256
import html
import json
import math
from pathlib import Path
import shutil
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont

WIDTH, HEIGHT = 1440, 1080
BACKGROUND = "#edf2f8"
INK, MUTED, BORDER = "#17243b", "#52637c", "#d5dfed"
COLORS = ("#2563eb", "#008d83", "#8b4bd5", "#db7418")
FONT_ROOT = Path("/usr/share/fonts/truetype/dejavu")


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


class Painter:
    def __init__(self, font_path=None):
        self.regular = Path(font_path) if font_path else FONT_ROOT / "DejaVuSans.ttf"
        self.bold = self.regular if font_path else FONT_ROOT / "DejaVuSans-Bold.ttf"
        self.fonts = {}

    def font(self, size, bold=False):
        key = size, bold
        if key not in self.fonts:
            self.fonts[key] = ImageFont.truetype(str(self.bold if bold else self.regular), size)
        return self.fonts[key]

    def text(self, draw, xy, value, size=20, fill=INK, bold=False):
        draw.text(xy, str(value), font=self.font(size, bold), fill=fill)

    def wrapped(self, draw, xy, value, width, size=20, lines=3, fill=INK, bold=False):
        font = self.font(size, bold)
        words = str(value).split()
        output, line = [], ""
        for word in words:
            candidate = f"{line} {word}".strip()
            if line and draw.textlength(candidate, font=font) > width:
                output.append(line)
                line = word
            else:
                line = candidate
        if line:
            output.append(line)
        for index, line in enumerate(output[:lines]):
            if index == lines - 1 and len(output) > lines:
                while line and draw.textlength(line + "…", font=font) > width:
                    line = line[:-1]
                line += "…"
            self.text(draw, (xy[0], xy[1] + index * (size + 5)), line, size, fill, bold)


def find_encoder(explicit=None):
    candidates = [explicit] if explicit else ["/usr/bin/ffmpeg", shutil.which("ffmpeg")]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            result = subprocess.run([candidate, "-hide_banner", "-encoders"], capture_output=True, text=True, check=True)
            if "libx264 " in result.stdout:
                return candidate
    raise RuntimeError("A CPU ffmpeg build with libx264 is required; pass --ffmpeg")


def arm_label(arm, short=False):
    if "label" in arm:
        step = f" · step {arm['_checkpoint_step']}" if not short and arm.get("_checkpoint_step") is not None else ""
        return arm["label"] + step
    size = str(arm["size"])
    if not size.endswith("B"):
        size += "B"
    phase = "pre" if str(arm["phase"]).startswith("pre") else "post"
    step = f" {arm['_checkpoint_step']}" if arm.get("_checkpoint_step") is not None else ""
    return f"{size} {phase}" if short else f"Qwen3.5-{size} · {arm['phase']}{step}"


def task_arms(manifest, episode):
    task = episode["task"]
    arms = [arm for arm in manifest["arms"]
            if (episode.get("task_alias") in arm["cohorts"] if "cohorts" in arm
                else task in arm.get("tasks", [task]))]
    if not 1 <= len(arms) <= 4:
        raise ValueError(f"The comparison layout supports one to four checkpoint arms per cohort: {task!r}")
    return arms


def identifier(value):
    if not isinstance(value, str) or not value or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-" for char in value):
        raise ValueError(f"Unsafe artifact identifier: {value!r}")
    return value


def load_results(manifest, manifest_hash, results_dir):
    results = {}
    for episode in manifest["episodes"]:
        identifier(episode["id"])
        if len(episode["windows"]) < 2:
            raise ValueError("Comparison needs at least two declared sampled windows")
        for arm in task_arms(manifest, episode):
            identifier(arm["id"])
            result = read_json(results_dir / arm["id"] / f"{episode['id']}.json")
            if result["manifest_sha256"] != manifest_hash or result["episode_id"] != episode["id"]:
                raise ValueError("Inference result does not belong to this frozen manifest/episode")
            recorded_arm = result["arm"]["id"] if isinstance(result["arm"], dict) else result["arm"]
            if recorded_arm != arm["id"] or len(result["records"]) != len(episode["windows"]):
                raise ValueError("Inference arm or sampled-window coverage differs from manifest")
            step = result.get("checkpoint", {}).get("source_step")
            if "_checkpoint_step" in arm and arm["_checkpoint_step"] != step:
                raise ValueError("One comparison arm cannot contain different checkpoint steps")
            arm["_checkpoint_step"] = step
            for record, window in zip(result["records"], episode["windows"], strict=True):
                if record["poi"] != window["poi"] or record["frame_indices"] != window["frame_indices"] or record["frame_indices"][5] != record["poi"]:
                    raise ValueError("Results must retain exact window ordering and literal POI slot 5")
                gt, focus = record["gt"], record["D_focus"]
                if record["C"] != focus or not 0 <= focus < len(episode["stage_prompts"]):
                    raise ValueError("D must use C's task-local predicted focus")
                if gt["stage_id"] != window["gt"]["stage_id"] or not math.isclose(gt["A"], window["gt"]["whole_progress"], abs_tol=1e-7) or not math.isclose(gt["within_stage_progress"], window["gt"]["within_stage_progress"], abs_tol=1e-7):
                    raise ValueError("Result labels differ from the frozen sampled GT")
                expected_d = 1.0 if focus < gt["stage_id"] else float(gt["within_stage_progress"]) if focus == gt["stage_id"] else 0.0
                if not math.isclose(float(gt["D"]), expected_d, abs_tol=1e-7):
                    raise ValueError("D target must be 1/WSP/0 relative to this model's own predicted focus")
                for value in (record["A"], record["D"], gt["A"], gt["D"]):
                    if not math.isfinite(value) or not 0 <= value <= 1 + 1e-6:
                        raise ValueError("Progress values must be finite unit-range numbers")
            results[arm["id"], episode["id"]] = result
    return results


def metrics(records):
    count = len(records)
    return {"windows": count,
            "A_rmse": math.sqrt(sum((r["A"] - r["gt"]["A"]) ** 2 for r in records) / count),
            "C_accuracy": sum(r["C"] == r["gt"]["stage_id"] for r in records) / count,
            "C_stage_mae": sum(abs(r["C"] - r["gt"]["stage_id"]) for r in records) / count,
            "D_relative_rmse": math.sqrt(sum((r["D"] - r["gt"]["D"]) ** 2 for r in records) / count)}


def display_camera(canvas, array, poi, box, name, painter):
    left, top, right, bottom = box
    image = Image.fromarray(np.asarray(array[poi]))
    image.thumbnail((right - left, bottom - top), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle(box, radius=10, fill="#101827")
    canvas.paste(image, (left + (right - left - image.width) // 2, top + (bottom - top - image.height) // 2))
    draw = ImageDraw.Draw(canvas)
    label_width = int(draw.textlength(name.upper(), font=painter.font(18, True))) + 20
    draw.rounded_rectangle((left + 9, top + 8, left + label_width + 9, top + 38), radius=5, fill="#14233b")
    painter.text(draw, (left + 19, top + 10), name.upper(), 18, "white", True)


def plot_points(canvas, box, episode, arms, results, current, kind, painter):
    draw = ImageDraw.Draw(canvas)
    left, top, right, bottom = box
    draw.rounded_rectangle(box, radius=14, fill="white", outline=BORDER)
    title = "A · POI slot 5 · whole-task progress" if kind == "A" else "C · POI slot 5 · first unfinished stage"
    painter.text(draw, (left + 18, top + 12), title, 22, bold=True)
    legend = [("GT", "#17243b")] + [(arm_label(arm, True), COLORS[i]) for i, arm in enumerate(arms)]
    legend_x = left + 20
    for name, color in legend:
        draw.ellipse((legend_x, top + 48, legend_x + 9, top + 57), fill=color)
        painter.text(draw, (legend_x + 14, top + 40), name, 18, color)
        legend_x += int(draw.textlength(name, font=painter.font(18))) + 37
    x0, x1, y0, y1 = left + 56, right - 24, top + 79, bottom - 52
    pois = [window["poi"] for window in episode["windows"]]
    maximum = 1.0 if kind == "A" else max(1, len(episode["stage_prompts"]) - 1)
    ticks = (0, 0.5, 1) if kind == "A" else range(int(maximum) + 1)
    def xy(index, value):
        return x0 + (pois[index] - pois[0]) / max(1, pois[-1] - pois[0]) * (x1 - x0), y1 - value / maximum * (y1 - y0)
    for tick in ticks:
        y = y1 - tick / maximum * (y1 - y0)
        draw.line((x0, y, x1, y), fill="#e2e8f1", width=1)
        painter.text(draw, (left + 12, y - 12), f"{tick:.1f}" if kind == "A" else str(tick), 18, MUTED)
    for index in (0, len(pois) // 2, len(pois) - 1):
        x, _ = xy(index, 0)
        label = str(pois[index])
        painter.text(draw, (x - draw.textlength(label, font=painter.font(18)) / 2, y1 + 5), label, 18, MUTED)
    cursor, _ = xy(current, 0)
    draw.line((cursor, y0 - 4, cursor, y1 + 3), fill="#9aa9bf", width=2)
    for arm_index, arm in enumerate(arms):
        records = results[arm["id"], episode["id"]]["records"]
        for index, record in enumerate(records):
            x, y = xy(index, record[kind])
            radius = 3 + arm_index * 0.8
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=COLORS[arm_index], width=2)
    for index, window in enumerate(episode["windows"]):
        value = window["gt"]["whole_progress" if kind == "A" else "stage_id"]
        x, y = xy(index, value)
        draw.line((x - 3, y - 3, x + 3, y + 3), fill=INK, width=2)
        draw.line((x - 3, y + 3, x + 3, y - 3), fill=INK, width=2)
    painter.text(draw, (left + 18, bottom - 24), f"Source frame index · cursor {pois[current]} · sampled points only", 17, MUTED)


def render_frame(episode, arms, results, cameras, index, painter, replay_fps):
    canvas = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    window = episode["windows"][index]
    poi, gt = window["poi"], window["gt"]
    painter.wrapped(draw, (28, 16), episode["task"], 1380, size=30, lines=1, bold=True)
    total = len(episode["windows"])
    nominal = f" · nominal cache time {poi / episode['cached_fps']:.1f}s ({episode['cached_fps']:g} fps)" if episode.get("cached_fps") else ""
    painter.text(draw, (28, 60), f"A/C/D · POI slot 5 · source frame {poi} / {episode['num_frames'] - 1}{nominal}", 20, MUTED)
    painter.text(draw, (28, 87), f"OFFLINE POI REPLAY · may include later input frames · sample {index + 1}/{total} · {replay_fps:g} samples/s, not original speed", 18, MUTED)
    views = tuple(cameras)
    if len(views) != 4:
        raise ValueError("This comparison layout expects exactly four ordered camera views")
    camera_boxes = ((28, 123, 432, 361), (444, 123, 848, 361), (28, 373, 432, 611), (444, 373, 848, 611))
    for view, box in zip(views, camera_boxes, strict=True):
        display_camera(canvas, cameras[view], poi, box, view, painter)
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle((28, 625, 848, 780), radius=14, fill="white", outline=BORDER)
    painter.text(draw, (48, 638), f"GT stage {gt['stage_id']}  ·  whole-task progress {gt['whole_progress']:.4f}", 23, bold=True)
    painter.wrapped(draw, (48, 674), episode["stage_prompts"][gt["stage_id"]], 780, 20, lines=2)
    indices = ", ".join(str(value) for value in window["frame_indices"])
    painter.text(draw, (48, 744), f"8 inputs (POI slot 5): {indices}", 18, MUTED)
    for arm_index, arm in enumerate(arms):
        record = results[arm["id"], episode["id"]]["records"][index]
        left, top, right, bottom = 870, 123 + arm_index * 166, 1412, 279 + arm_index * 166
        color = COLORS[arm_index]
        draw.rounded_rectangle((left, top, right, bottom), radius=14, fill="white", outline=BORDER)
        draw.rounded_rectangle((left, top, left + 7, bottom), radius=3, fill=color)
        label = arm_label(arm)
        painter.text(draw, (left + 19, top + 10), label, 21, color, True)
        matched = record["C"] == gt["stage_id"]
        painter.text(draw, (left + 19, top + 43), f"C[5] stage {record['C']}  ·  GT {gt['stage_id']}  ·  {'match' if matched else 'different'}", 22, "#176b4d" if matched else "#a83d37", True)
        painter.text(draw, (left + 19, top + 76), f"A[5] whole-task: {record['A']:.4f}  |  GT {record['gt']['A']:.4f}", 20)
        painter.text(draw, (left + 19, top + 106), f"D[5] focus-relative: {record['D']:.4f}  |  GT {record['gt']['D']:.4f}", 20)
        painter.text(draw, (left + 19, top + 132), f"D focus = predicted stage {record['D_focus']} (C slot 5)", 18, MUTED)
    plot_points(canvas, (28, 801, 716, 1060), episode, arms, results, index, "A", painter)
    plot_points(canvas, (732, 801, 1412, 1060), episode, arms, results, index, "C", painter)
    return canvas


def verify_video(path, ffmpeg):
    sibling = str(Path(ffmpeg).with_name("ffprobe"))
    ffprobe = sibling if Path(sibling).is_file() else shutil.which("ffprobe")
    probe = json.loads(subprocess.run([ffprobe, "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)], capture_output=True, text=True, check=True).stdout)
    video = next(stream for stream in probe["streams"] if stream["codec_type"] == "video")
    if video["codec_name"] != "h264" or video["pix_fmt"] != "yuv420p" or (video["width"], video["height"]) != (WIDTH, HEIGHT):
        raise RuntimeError("Output is not the requested browser-playable H.264/yuv420p video")
    subprocess.run([ffmpeg, "-v", "error", "-xerror", "-i", str(path), "-f", "null", "-"], capture_output=True, check=True)
    # ISO-BMFF top-level box order directly checks that faststart moved moov.
    boxes = []
    with path.open("rb") as stream:
        while header := stream.read(8):
            if len(header) != 8:
                raise RuntimeError("Truncated MP4 box header")
            size, kind = int.from_bytes(header[:4], "big"), header[4:].decode("ascii", "replace")
            header_size = 8
            if size == 1:
                size = int.from_bytes(stream.read(8), "big")
                header_size = 16
            boxes.append(kind)
            if size == 0:
                break
            if size < header_size:
                raise RuntimeError("Invalid MP4 box size")
            stream.seek(size - header_size, 1)
    if "moov" not in boxes or "mdat" not in boxes or boxes.index("moov") > boxes.index("mdat"):
        raise RuntimeError("MP4 faststart verification failed")
    return {"codec": video["codec_name"], "pixel_format": video["pix_fmt"], "width": video["width"], "height": video["height"],
            "frames": int(video["nb_frames"]), "duration_seconds": float(probe["format"]["duration"]),
            "frame_rate": video["avg_frame_rate"], "full_decode": "passed", "faststart": True,
            "bytes": path.stat().st_size, "sha256": sha256(path.read_bytes()).hexdigest()}


def table(arms, arm_metrics):
    rows = []
    for arm in arms:
        item = arm_metrics[arm["id"]]
        rows.append(f"<tr><th>{html.escape(arm_label(arm))}</th><td>{item['A_rmse']:.4f}</td><td>{item['C_accuracy']:.1%}</td><td>{item['C_stage_mae']:.4f}</td><td>{item['D_relative_rmse']:.4f}</td></tr>")
    return "<div class='scroll'><table><thead><tr><th>Checkpoint</th><th>A RMSE ↓</th><th>C accuracy ↑</th><th>C stage MAE ↓</th><th>D relative RMSE*</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"


def write_index(output, manifest, results, artifacts, manifest_hash, replay_fps):
    task_records = defaultdict(lambda: defaultdict(list))
    cards = []
    for episode in manifest["episodes"]:
        arms = task_arms(manifest, episode)
        per_arm = {}
        for arm in arms:
            records = results[arm["id"], episode["id"]]["records"]
            per_arm[arm["id"]] = metrics(records)
            task_records[(episode.get("task_alias", ""), episode["task"])][arm["id"]].extend(records)
        title = html.escape(episode["task"])
        name = html.escape(episode["id"])
        coverage = "Sampled GT stages: " + ", ".join(map(str, episode.get("sampled_gt_stages", sorted({window["gt"]["stage_id"] for window in episode["windows"]}))))
        if episode.get("unsampled_gt_stages"):
            coverage += ". Present in the source but missed by these fixed samples: " + ", ".join(map(str, episode["unsampled_gt_stages"]))
        if episode.get("gt_stage_backtracks"):
            coverage += ". Source GT stage regressions are preserved"
        cards.append(f"""<section class='card'><h2>{title}</h2><p class='muted'>Episode {html.escape(str(episode['trajectory_id']))} · {len(episode['windows'])} identical sampled windows per checkpoint</p>
<video controls playsinline preload='metadata' poster='{name}.png'><source src='{name}.mp4' type='video/mp4'></video>
<p><a href='{name}.mp4'>Download MP4</a> · <a href='{name}.png'>Open poster</a></p><p class='muted'>{html.escape(coverage)}.</p>{table(arms, per_arm)}</section>""")
    summaries = []
    for (cohort, task), grouped in task_records.items():
        arms = task_arms(manifest, {"task": task, "task_alias": cohort})
        title = f"{cohort}: {task}" if cohort else task
        summaries.append(f"<section class='card'><h2>{html.escape(title)} — sampled-window summary</h2><p class='muted'>{len(next(iter(grouped.values())))} fixed windows per checkpoint; metrics pooled over these episodes.</p>{table(arms, {arm_id: metrics(records) for arm_id, records in grouped.items()})}</section>")
    document = """<!doctype html><html lang='en'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>Grana · inference video comparisons</title><style>
*{box-sizing:border-box}body{margin:0;background:#edf2f8;color:#17243b;font:17px/1.55 system-ui,sans-serif}main{max-width:1480px;margin:auto;padding:32px}h1{font-size:36px;line-height:1.15;margin:0 0 18px}h2{font-size:24px;margin:0 0 8px}.card,.intro{background:white;border:1px solid #d5dfed;border-radius:16px;padding:26px;margin:24px 0}.intro{border-top:5px solid #2563eb}.muted{color:#52637c}video{display:block;width:100%;background:#101827;border-radius:10px}a{color:#1856ca}table{border-collapse:collapse;width:100%;font-size:16px}th,td{padding:12px 16px;text-align:right;border-bottom:1px solid #d5dfed;white-space:nowrap}th:first-child{text-align:left}thead{background:#f2f5fa}.scroll{overflow:auto}.badge{display:inline-block;padding:4px 12px;border-radius:20px;background:#e8f0ff;margin:3px 5px 3px 0}code{overflow-wrap:anywhere}footer{color:#52637c;font-size:14px;margin:30px 0}@media(max-width:700px){main{padding:12px}.card,.intro{padding:15px}h1{font-size:29px}}
</style><main><h1>Grana inference comparisons</h1><section class='intro'>
<p>The selected checkpoints see the same eight RGB observations at every sampled position. Each video shows the four cameras at the scored source frame, cached GT labels, and actual A/C/D inference outputs at POI input slot 5. These saved comparisons remain offline POI replays; windows may include later observations. A last-frame D prediction requires a new forward with its focus selected from C at slot 7.</p>
"""
    document += f"<p><span class='badge'>{len(manifest['episodes'])} episodes</span><span class='badge'>Matched windows within each cohort</span><span class='badge'>{replay_fps:g} sampled positions / playback second</span><span class='badge'>No interpolated predictions</span></p>"
    document += """<p><strong>A</strong> is whole-task progress. <strong>C</strong> predicts the first unfinished stage. <strong>D</strong> predicts progress relative to that model’s own C-predicted stage at literal input slot 5. D’s target is 1 for a focus earlier than the observed GT stage, within-stage progress for the same stage, and 0 for a later focus. <strong>D values and D RMSE use model-dependent targets; they are not whole-task progress or a shared-target ranking.</strong></p>
<p class='muted'>This is a sampled trajectory replay, not continuous inference or original-speed video. Plots contain only the declared sampled points; fixed uniform samples may miss brief stages. Display resizing preserves aspect ratio; model inputs came from the unchanged cached RGB. Nominal cache seconds use the cache’s declared frame rate, not original timestamps. These few episodes are a qualitative comparison, not a model-quality benchmark. D error can be low when C selects the wrong stage, so C accuracy is shown beside it.</p></section>"""
    document += "".join(summaries) + "".join(cards)
    document += f"<footer><p>*D relative RMSE is against each model’s own focus-dependent target. A RMSE and C metrics share the same sampled GT.</p><p>Frozen manifest SHA256: <code>{manifest_hash}</code></p><p><a href='render_receipt.json'>Video verification receipt</a> · MP4: H.264, yuv420p, faststart; full decode verified. No external scripts or CDNs.</p></footer></main></html>"
    (output / "index.html").write_text(document)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fps", type=float, default=2.0, help="Sampled positions per replay second; not the cache FPS")
    parser.add_argument("--font", type=Path, help="Optional Unicode TTF/OTF for non-Latin stage descriptions")
    parser.add_argument("--ffmpeg", help="CPU ffmpeg executable with libx264")
    args = parser.parse_args()
    if not math.isfinite(args.fps) or args.fps <= 0:
        parser.error("--fps must be finite and positive")
    manifest_path = args.manifest.expanduser().resolve()
    manifest = read_json(manifest_path)
    manifest_hash = sha256(manifest_path.read_bytes()).hexdigest()
    results_dir = (args.results or manifest_path.parent / "results").expanduser().resolve()
    output = (args.output or manifest_path.parent / "media").expanduser().resolve()
    results = load_results(manifest, manifest_hash, results_dir)
    ffmpeg = find_encoder(args.ffmpeg)
    painter = Painter(args.font)
    output.mkdir(parents=True, exist_ok=True)
    artifacts = []
    for episode in manifest["episodes"]:
        arms = task_arms(manifest, episode)
        views = episode.get("view_names", manifest.get("view_names", tuple(episode["frame_paths"])))
        cameras = {view: np.load(episode["frame_paths"][view], mmap_mode="r", allow_pickle=False) for view in views}
        for array in cameras.values():
            if array.dtype != np.uint8 or array.ndim != 4 or array.shape[-1] != 3 or array.shape[0] != episode["num_frames"]:
                raise ValueError("Camera payload must be unchanged uint8 [frames,height,width,3] NPY")
        path = output / f"{episode['id']}.mp4"
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite a completed comparison: {path}")
        temporary = output / f".{episode['id']}.rendering.mp4"
        command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", f"{WIDTH}x{HEIGHT}",
                   "-framerate", str(args.fps), "-i", "pipe:0", "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-threads", "4",
                   "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(temporary)]
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            for index in range(len(episode["windows"])):
                frame = render_frame(episode, arms, results, cameras, index, painter, args.fps)
                if index == len(episode["windows"]) // 2:
                    frame.save(output / f"{episode['id']}.png")
                process.stdin.write(frame.tobytes())
            process.stdin.close()
            error = process.stderr.read().decode()
            if process.wait() != 0:
                raise RuntimeError(f"ffmpeg failed: {error}")
        except BaseException:
            if process.poll() is None:
                process.kill()
                process.wait()
            raise
        verification = verify_video(temporary, ffmpeg)
        if verification["frames"] != len(episode["windows"]):
            raise RuntimeError("Encoded video frame count differs from sampled inference coverage")
        temporary.rename(path)
        artifacts.append({"episode_id": episode["id"], "video": path.name, "poster": f"{episode['id']}.png", "poster_sample_index": len(episode["windows"]) // 2,
                          "prediction_roles": ["A", "C", "D"], "prediction_frame": "poi", "poi_slot": 5,
                          "source_frames": [window["frame_indices"][5] for window in episode["windows"]],
                          "verification": verification})
        print(json.dumps(artifacts[-1]), flush=True)
    write_json(output / "render_receipt.json", {"schema_version": 1, "manifest_sha256": manifest_hash, "sampled_replay_fps": args.fps,
                                               "source_timestamps": "unknown; cached_fps only gives nominal cache time", "ffmpeg": ffmpeg, "artifacts": artifacts})
    write_index(output, manifest, results, artifacts, manifest_hash, args.fps)
    print(f"Comparison page: {output / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
