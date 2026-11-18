#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render every source frame with machine stage candidates; uniform clocks only."""
import argparse
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import uuid

from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grana.annotation.review import ReviewStore


def uniform_fps(edges):
    if (len(edges) < 3 or any(type(t) not in (int, float) or not math.isfinite(t) for t in edges)
            or any(b <= a for a, b in zip(edges, edges[1:]))):
        raise ValueError("A finite increasing source clock with at least two frames is required")
    fps = Fraction((len(edges) - 1) / (edges[-1] - edges[0])).limit_denominator(1000000)
    if max(abs(t - edges[0] - i / float(fps)) for i, t in enumerate(edges)) > 2e-6:
        raise ValueError("Nonuniform source clock: this renderer will not invent a constant FPS")
    return fps


def probe(path, ffprobe, entries):
    return json.loads(subprocess.run([ffprobe, "-v", "error", "-select_streams", "v:0", "-count_frames",
                                     "-show_entries", entries, "-of", "json", str(path)],
                                    check=True, capture_output=True, text=True).stdout)


def render(run_dir, episode_index, output, ffmpeg=None, ffprobe="ffprobe"):
    ffmpeg = ffmpeg or ("/usr/bin/ffmpeg" if Path("/usr/bin/ffmpeg").is_file() else "ffmpeg")
    store = ReviewStore(run_dir)
    record = store.episode(episode_index)
    episode, candidate = record["episode"], record["candidate"]
    count, edges = episode["frame_count"], episode["time_edges"]
    fps, views = uniform_fps(edges), list(episode["video_urls"])
    if not 1 <= len(views) <= 4:
        raise ValueError("This compact renderer supports one to four original views")
    output = Path(output).expanduser().absolute()
    if any(p.is_symlink() for p in (output, *output.parents)) or output.exists():
        raise ValueError("Output must be a fresh path without symlinks")
    output = output.resolve()
    source = Path(episode["source_identity"]["source_root"]).resolve()
    if output.suffix.lower() != ".mp4" or any(output.is_relative_to(p) for p in (ROOT, source, store.root)):
        raise ValueError("Output must be a new MP4 outside source, annotation run and code checkout")
    if any((p / ".git").exists() for p in (output.parent, *output.parents)):
        raise ValueError("Output must be outside Git checkouts")
    paths, shapes = [], []
    for view in views:
        path = store.media[episode["video_urls"][view]][0]
        data = probe(path, ffprobe, "stream=width,height:frame=best_effort_timestamp_time")
        times = [float(frame["best_effort_timestamp_time"]) for frame in data["frames"]]
        offset = episode["clock_receipt"]["cameras"][view]["source_minus_video_offset_seconds"]
        if len(times) != count or any(not math.isfinite(t) or abs(t + offset - edges[i]) > 2e-6 for i, t in enumerate(times)):
            raise ValueError(f"Video PTS does not match the uniform source clock: {view}")
        shape = data["streams"][0]
        paths.append(path)
        shapes.append((shape["width"], shape["height"]))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.{uuid.uuid4().hex}.mp4")
    font, small = ImageFont.load_default(size=19), ImageFont.load_default(size=15)
    width, height, segment_index = max(960, 320 * len(views)), 450, 0
    processes = []
    with tempfile.TemporaryFile() as errors:
        try:
            for path in paths:
                processes.append(subprocess.Popen([ffmpeg, "-v", "error", "-nostdin", "-threads", "1", "-noautorotate",
                    "-i", str(path), "-map", "0:v:0", "-an", "-sn", "-dn", "-vsync", "0",
                    "-threads", "1", "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"], stdout=subprocess.PIPE, stderr=errors))
            encoder = subprocess.Popen([ffmpeg, "-v", "error", "-nostdin", "-n", "-f", "rawvideo", "-pix_fmt", "rgb24",
                "-s", f"{width}x{height}", "-framerate", str(fps), "-i", "pipe:0", "-an", "-c:v", "libx264",
                "-threads", "1", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                str(temporary)], stdin=subprocess.PIPE, stderr=errors)
            processes.append(encoder)
            for frame in range(count):
                canvas = Image.new("RGB", (width, height), "#142831")
                draw = ImageDraw.Draw(canvas)
                for column, (decoder, shape, view) in enumerate(zip(processes[:len(views)], shapes, views)):
                    pixels = decoder.stdout.read(shape[0] * shape[1] * 3)
                    if len(pixels) != shape[0] * shape[1] * 3:
                        raise ValueError(f"Video decode ended before source frame {frame}: {view}")
                    image = ImageOps.pad(Image.frombytes("RGB", shape, pixels), (320, 240))
                    canvas.paste(image, (column * 320, 105))
                    draw.text((column * 320 + 8, 82), view, font=small, fill="white")
                while frame >= candidate["segments"][segment_index]["end_frame"]:
                    segment_index += 1
                segment = candidate["segments"][segment_index]
                draw.text((12, 8), "MACHINE CANDIDATE - PENDING HUMAN REVIEW", font=font, fill="#ffce70")
                draw.text((12, 34), f"Episode {episode_index} | frame {frame}/{count - 1} | source {edges[frame]:.6f}s", font=small, fill="white")
                stage = segment["stage_id"] if segment["stage_id"] is not None else "UNKNOWN"
                draw.text((12, 57), f"Stage {stage} | {segment['identity_status']} | [{segment['start_frame']}, {segment['end_frame']})", font=small, fill="white")
                active = [f"{b['frame']} in [{b['lower_frame']}, {b['upper_frame']}]" for b in candidate["boundaries"] if b["lower_frame"] <= frame <= b["upper_frame"]]
                draw.text((12, 354), "Boundary bracket: " + ("; ".join(active) or "none at this frame"), font=small, fill="#ffce70")
                invalid = any(s["start_frame"] <= frame < s["end_frame"] for s in candidate["invalid_intervals"])
                draw.text((12, 377), "SOURCE INVALID" if invalid else "Stage identity does not imply action quality or task outcome.", font=small, fill="#f3b6ae" if invalid else "white")
                for s in candidate["segments"]:
                    color = "#63747a" if s["stage_id"] is None else ("#6eb2a0" if s["identity_status"] == "resolved" else "#c7a768")
                    draw.rectangle((12 + (width - 24) * s["start_frame"] / count, 408, 12 + (width - 24) * s["end_frame"] / count, 430), fill=color, outline="#142831")
                x = 12 + (width - 24) * frame / count
                draw.line((x, 402, x, 437), fill="white", width=3)
                encoder.stdin.write(canvas.tobytes())
            for decoder in processes[:len(views)]:
                if decoder.stdout.read(1) or decoder.wait() != 0:
                    raise ValueError("Video contains additional frames or failed to decode")
            encoder.stdin.close()
            if encoder.wait() != 0:
                raise RuntimeError("CPU video encoding failed")
            check = probe(temporary, ffprobe, "stream=nb_read_frames,avg_frame_rate,width,height")["streams"][0]
            if int(check["nb_read_frames"]) != count or Fraction(check["avg_frame_rate"]) != fps:
                raise ValueError("Rendered frame count or source cadence differs")
            os.link(temporary, output)
        finally:
            for process in processes:
                if process.poll() is None:
                    process.kill()
                process.wait()
                for stream in (process.stdin, process.stdout):
                    if stream is not None and not stream.closed:
                        stream.close()
            temporary.unlink(missing_ok=True)
    return {"output": str(output), "frames": count, "fps": str(fps), "source_seconds": edges[-1] - edges[0],
            "candidate_sha256": record["candidate_sha256"], "source_identity": episode["source_identity"],
            "human_review": "pending", "video_sha256": hashlib.sha256(output.read_bytes()).hexdigest()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ffmpeg", help="CPU FFmpeg with libx264; prefers /usr/bin/ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    args = parser.parse_args()
    print(json.dumps(render(args.run_dir, args.episode, args.output, args.ffmpeg, args.ffprobe), indent=2))
