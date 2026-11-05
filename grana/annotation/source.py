# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read-only LeRobot v2.x/v3.0 RGB/clock evidence, with no annotation or action input.

v2 stores each episode in its own parquet and video files. v3 packs many episodes into
shared files: an episode is its rows of the shared parquet and a time range of each
shared video, and only those enter its identity. Evidence decodes a v3 episode's frames
straight from the shared videos, each matched by its presentation time.
"""
from array import array
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import threading

from PIL import Image, ImageDraw

from grana._ffmpeg import passthrough
from grana._paths import PROJECT_ROOT
from grana.data.lerobot import LeRobotDataset, template_path


CLOCK_COLUMNS = ("frame_index", "episode_index", "timestamp")
CLOCK_TOLERANCE_SECONDS = 1e-4
PROBE_THREADS = 4
SEGMENT_MARGIN_FRAMES = 64   # packets read around a v3 episode: more than any decoder reorders
MATCH_FRAMES = 0.25   # a decoded frame is the probed packet within a quarter frame of its presentation time
UNHASHED_H264_UNITS = "7|8|9"   # sequence and picture parameter sets, access unit delimiters


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _integer(value, name):
    if isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _tool(name):
    executable = shutil.which(name)
    if not executable:
        raise RuntimeError(f"{name} is required for annotation evidence")
    return executable


class _FileCache:
    """Per-file results, reused while the file keeps its device, inode, size, mtime and ctime.

    ``part`` keeps separate results for parts of one file, such as the episodes of a shared v3 file.
    Any change recomputes. Least recently used entries go first once the total weight exceeds ``limit``.
    """

    def __init__(self, limit, weight=lambda value: 1):
        self.limit, self.weight, self.total = limit, weight, 0
        self.entries, self.lock = OrderedDict(), threading.Lock()

    def get(self, path, compute, part=None):
        path = Path(path).resolve()
        stat = path.stat()   # before computing: a file that changes meanwhile is recomputed next time
        version = stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns
        key = path, part
        with self.lock:
            cached = self.entries.get(key)
            if cached is not None and cached[0] == version:
                self.entries.move_to_end(key)
                return cached[1]
        value = compute(path)
        with self.lock:
            if key in self.entries:
                self.total -= self.weight(self.entries.pop(key)[1])
            self.entries[key] = version, value
            self.total += self.weight(value)
            while self.total > self.limit and len(self.entries) > 1:
                self.total -= self.weight(self.entries.popitem(last=False)[1][1])
        return value


_HASHES = _FileCache(4096)   # files, or episodes of a shared parquet
_PROBES = _FileCache(2**22, weight=lambda probe: len(probe[2]))   # frame timestamps, 8 bytes each: 32 MiB


def _file_sha(path):
    """sha256 of a source file, hashed again only after the file changes."""
    return _HASHES.get(path, _sha)


def _ffprobe(video):
    probe = subprocess.run([_tool("ffprobe"), "-v", "error", "-select_streams", "v:0",
                            "-show_entries", "stream=width,height:frame=best_effort_timestamp_time",
                            "-of", "json", str(video)], capture_output=True, text=True, check=True)
    result = json.loads(probe.stdout)
    streams = result.get("streams", [])
    if len(streams) != 1:
        raise ValueError("Expected one selected video stream")
    width, height = streams[0]["width"], streams[0]["height"]
    pts = array("d", (float(row.get("best_effort_timestamp_time", "nan")) for row in result.get("frames", [])))
    return width, height, pts


def _probe(video):
    """Width, height and frame PTS of the first video stream; ffprobe decodes every frame, so only once per file version."""
    return _PROBES.get(video, _ffprobe)


def _ffprobe_segment(video, start, end, fps):
    """Width, height, PTS and packet hash of the frames in [start, end) of a shared video, with half-frame bounds.

    Reads the packets around the range without decoding them and puts them in presentation order. H.264 packets are
    hashed without their parameter sets: appending an episode with FFmpeg's concat demuxer, as LeRobot does, copies
    them into the keyframes already in the file and leaves every picture as it was.
    """
    half, margin = 0.5 / fps, SEGMENT_MARGIN_FRAMES / fps
    probe = subprocess.run([_tool("ffprobe"), "-v", "error", "-select_streams", "v:0",
                            "-show_entries", "stream=codec_name,width,height", "-of", "json", str(video)],
                           capture_output=True, text=True, check=True)
    streams = json.loads(probe.stdout).get("streams", [])
    if len(streams) != 1:
        raise ValueError("Expected one selected video stream")
    first = max(0.0, start - margin)
    units = ["-bsf:v", f"filter_units=remove_types={UNHASHED_H264_UNITS}"] if streams[0].get("codec_name") == "h264" else []
    hashes = subprocess.run([_tool("ffmpeg"), "-v", "error", "-nostdin", "-copyts", "-ss", f"{first:.6f}",
                             "-t", f"{end + margin - first:.6f}", "-i", str(video), "-map", "0:v:0", "-c", "copy", *units,
                             "-f", "framehash", "-hash", "sha256", "pipe:1"], capture_output=True, text=True, check=True)
    lines = hashes.stdout.splitlines()
    time_base = next((line.removeprefix("#tb 0:").split("/") for line in lines if line.startswith("#tb 0:")), None)
    if time_base is None:
        raise ValueError(f"Could not read the video packets of {Path(video).name}")
    tick = int(time_base[0]) / int(time_base[1])
    packets = []
    for line in lines:
        if line.strip() and not line.startswith("#"):
            _, _, pts, _, _, data, *_ = (field.strip() for field in line.split(","))
            packets.append((float(f"{int(pts) * tick:f}"), data))   # seconds, rounded as ffprobe prints them
    packets.sort()
    kept = [(pts, data) for pts, data in packets if start - half <= pts < end - half]
    digest = hashlib.sha256(json.dumps([data for _, data in kept]).encode()).hexdigest()
    return streams[0]["width"], streams[0]["height"], array("d", (pts for pts, _ in kept)), digest


def _segment_probe(segment, fps):
    """_ffprobe_segment, once per file version and time range."""
    return _PROBES.get(segment.path, lambda path: _ffprobe_segment(path, segment.start, segment.end, fps),
                       part=(segment.start, segment.end, fps))


def episode_frame_times(segment, fps, packets_sha256):
    """Presentation times in the shared video of a v3 episode's frames, the ones its packet hash covers.

    These are the frames _ffprobe_segment keeps. Refuses if the video no longer holds those packets.
    """
    *_, pts, packets = _segment_probe(segment, fps)   # probed again only if the file changed
    if packets != packets_sha256:
        raise ValueError("Source video changed after clock validation")
    return pts


def read_around(times, fps):
    """FFmpeg input options that read the packets around ``times`` (increasing) of a shared video, as _ffprobe_segment does.

    Timestamps are kept, so a decoded frame's ``t`` is its presentation time in the file, comparable with ``times``.
    """
    margin = SEGMENT_MARGIN_FRAMES / fps
    first = max(0.0, times[0] - margin)
    return ["-copyts", "-ss", f"{first:.6f}", "-t", f"{times[-1] + margin - first:.6f}"]


def _rows_sha(dataset, ep):
    """Bind the episode's native clock without parsing supervision, state or action columns.

    RGB is bound separately by the video packet hashes and evidence image hashes.
    Updating labels or appending another episode must not change this RGB-only input.
    """
    rows = dataset.rows(ep, list(CLOCK_COLUMNS))
    return hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class Episode:
    episode_index: int
    frame_count: int
    timestamps: tuple[float, ...]
    time_edges: tuple[float, ...]
    video_paths: dict[str, Path]
    view_names: tuple[str, ...]
    source_identity: dict
    clock_receipt: dict
    video_segments: dict = field(default_factory=dict)   # v3: view -> VideoSegment of its shared file
    video_features: dict[str, str] = field(default_factory=dict)   # named views for source revalidation


# The episode's own part of a source identity. The rest records where the dataset was read (source_root,
# parquet_path, video_segments) and its dataset-wide metadata (metadata_sha256), which moving, copying or
# appending to it changes.
EPISODE_IDENTITY_KEYS = ("episode_index", "trajectory_id", "timestamps_sha256", "parquet_sha256", "video_sha256")
# v3 files hold other episodes too; bind this episode's clock rows and video packets.
# Use a distinct key from the experimental all-column fingerprint: they are not interchangeable.
SHARED_EPISODE_IDENTITY_KEYS = ("episode_index", "trajectory_id", "timestamps_sha256", "parquet_clock_sha256",
                                "video_packets_sha256")


def same_episode(identity, other) -> bool:
    """Whether two source identities name the same episode evidence, wherever the dataset lives.

    A LeRobot v2 file holds one episode, so its hash changes only with the episode; a v3 identity hashes the
    episode's clock and video packets instead.
    """
    if not isinstance(identity, dict) or not isinstance(other, dict):
        return False
    keys = SHARED_EPISODE_IDENTITY_KEYS if "parquet_clock_sha256" in identity else EPISODE_IDENTITY_KEYS
    return all(key in identity and key in other and identity[key] == other[key] for key in keys)


class LeRobotSource:
    def __init__(self, source: Path, views: dict, *, allow_shared_videos=False):
        # Legacy reviewers serve video_paths directly and assume an episode per file.
        # Evidence consumers opt in only when they decode each episode's segment.
        self.source = Path(source).expanduser().resolve()
        if (not isinstance(views, dict) or not views
                or any(not isinstance(k, str) or not k.strip() or k != k.strip()
                       or not isinstance(v, str) or not v.strip() for k, v in views.items())
                or len(set(views.values())) != len(views)):
            raise ValueError("views must map distinct trimmed camera names to distinct feature keys")
        self.views = dict(views)
        info_path, episodes_path = (self.source / "meta" / name for name in ("info.json", "episodes.jsonl"))
        self.info = json.loads(info_path.read_text())
        version = str(self.info.get("codebase_version", ""))
        if version.startswith("v3."):
            if not allow_shared_videos:
                raise ValueError("This video consumer supports LeRobot v2 only; shared-video evidence needs an episode-aware decoder")
            # meta/episodes/ locates each episode's rows and video time ranges inside shared files.
            self.shared = LeRobotDataset(self.source)
            self.episodes = self.shared.episodes
            metadata = self.shared.metadata_files()
        elif version.startswith("v2."):
            self.shared = None
            self.chunks_size = _integer(self.info.get("chunks_size"), "chunks_size")
            if self.chunks_size < 1:
                raise ValueError("chunks_size must be positive")
            self.episodes = {}
            for line in episodes_path.read_text().splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                index = _integer(row.get("episode_index"), "episode_index")
                if index < 0 or index in self.episodes:
                    raise ValueError("Episode metadata needs unique nonnegative IDs")
                self.episodes[index] = row
            metadata = [info_path, episodes_path]
        else:
            raise ValueError("Annotation source supports native LeRobot v2.x and v3.0 only")
        self.metadata_hashes = {str(path.relative_to(self.source / "meta")): _file_sha(path) for path in metadata}
        self._loaded = {}

    def _path(self, key, **values):
        path = (self.source / template_path(self.info, key, **values)).resolve()
        if not path.is_relative_to(self.source) or not path.is_file():
            raise ValueError("Source path must resolve to an existing file inside its source root")
        return path

    def load_episode(self, ep: int) -> Episode:
        import pyarrow.parquet as parquet

        ep = _integer(ep, "episode_index")
        if ep in self._loaded:
            return self._loaded[ep]
        if ep not in self.episodes:
            raise ValueError("Episode is absent from source metadata")
        metadata = self.episodes[ep]
        count = _integer(metadata.get("length"), "episode length")
        if count < 2:
            raise ValueError("At least two source frames are required to establish the terminal time edge")
        if self.shared is None:
            fields = {"episode_index": ep, "episode_chunk": ep // self.chunks_size}
            path = self._path("data_path", **fields)
            # Do not read annotation.*, rl_info.*, observation.state or action.
            table = parquet.read_table(path, columns=list(CLOCK_COLUMNS)).to_pydict()
        else:
            # Read only this episode's clock rows. Supervision, state and action columns stay unopened.
            path = self.shared.data_path(ep)
            table = self.shared.rows(ep, list(CLOCK_COLUMNS))
        frames = [_integer(x, "frame_index") for x in table["frame_index"]]
        episode_ids = [_integer(x, "episode_index") for x in table["episode_index"]]
        if frames != list(range(count)) or len(episode_ids) != count or set(episode_ids) != {ep}:
            raise ValueError("Source frame_index/episode_index must match contiguous metadata rows")
        times = []
        for raw in table["timestamp"]:
            value = raw[0] if isinstance(raw, (list, tuple)) and len(raw) == 1 else raw
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError("Source timestamps must be finite real values")
            times.append(float(value))
        if len(times) != count or any(b <= a for a, b in zip(times, times[1:])):
            raise ValueError("Source timestamps must be strictly increasing and frame-aligned")
        terminal = times[-1] + (times[-1] - times[-2])
        if not math.isfinite(terminal) or terminal <= times[-1]:
            raise ValueError("Terminal source time edge must be finite and increasing")
        videos, segments, camera_receipts, cameras = {}, {}, {}, {}
        # Probe and hash the cameras side by side, then check them in view order.
        with ThreadPoolExecutor(max_workers=min(len(self.views), PROBE_THREADS)) as pool:
            for view, key in self.views.items():
                feature = self.info.get("features", {}).get(key, {})
                if feature.get("dtype") != "video":
                    raise ValueError(f"{key} must be a declared video feature")
                if self.shared is None:
                    video = self._path("video_path", video_key=key, **fields)
                    job = pool.submit(lambda path: (*_probe(path), _file_sha(path)), video)
                else:
                    segment = segments[view] = self.shared.video(ep, key)
                    video, job = segment.path, pool.submit(_segment_probe, segment, self.shared.fps)
                cameras[view] = feature, video, job
        for view, (feature, video, job) in cameras.items():
            width, height, pts, video_sha = job.result()
            if feature.get("shape") != [height, width, 3]:
                raise ValueError(f"Camera shape differs from metadata: {view}")
            # A v3 episode's video time starts at its from_timestamp in the shared file.
            pts = [p - segments[view].start for p in pts] if view in segments else list(pts)
            if len(pts) != count or any(not math.isfinite(x) for x in pts) or any(b <= a for a, b in zip(pts, pts[1:])):
                raise ValueError(f"Camera frame count/PTS is invalid: {view}")
            offset = times[0] - pts[0]
            error = max(abs(t - (p + offset)) for t, p in zip(times, pts, strict=True))
            if error > CLOCK_TOLERANCE_SECONDS:
                raise ValueError(f"Camera PTS does not align with source timestamps: {view}, max error {error}")
            videos[view] = video
            camera_receipts[view] = {"frames": count, "shape": [height, width, 3],
                                     "source_minus_video_offset_seconds": offset,
                                     "maximum_timestamp_error_seconds": error,
                                     "pts_sha256": hashlib.sha256(json.dumps(pts).encode()).hexdigest(),
                                     ("video_packets_sha256" if view in segments else "video_sha256"): video_sha}
        identity = metadata.get("trajectory_id") or f"{self.source.name}__episode_{ep:06d}"
        if not isinstance(identity, str) or not identity.strip():
            raise ValueError("Source trajectory identity must be nonempty text")
        receipt = {"clock_columns": list(CLOCK_COLUMNS), "source_frames": count,
                   "first_timestamp": times[0], "last_timestamp": times[-1], "terminal_timestamp": terminal,
                   "terminal_timestamp_policy": "last_timestamp_plus_last_observed_positive_delta",
                   "alignment_policy": "source_row_rank_to_video_frame_rank_with_explicit_first_timestamp_offset",
                   "alignment_limit": "Verifies recorded cadence and row mapping, not unrecorded physical camera synchronization.",
                   "tolerance_seconds": CLOCK_TOLERANCE_SECONDS, "cameras": camera_receipts}
        source_identity = {"source_root": str(self.source), "episode_index": ep, "trajectory_id": identity,
                           "metadata_sha256": dict(self.metadata_hashes), "parquet_path": str(path)}
        if self.shared is None:
            source_identity.update(parquet_sha256=_file_sha(path),
                                   video_sha256={view: value["video_sha256"] for view, value in camera_receipts.items()})
        else:
            source_identity.update(
                parquet_clock_sha256=_HASHES.get(path, lambda _: _rows_sha(self.shared, ep), part=ep),
                video_segments={view: {"path": str(segment.path), "from_timestamp": segment.start,
                                       "to_timestamp": segment.end} for view, segment in segments.items()},
                video_packets_sha256={view: value["video_packets_sha256"] for view, value in camera_receipts.items()})
        source_identity["timestamps_sha256"] = hashlib.sha256(json.dumps(times).encode()).hexdigest()
        episode = Episode(ep, count, tuple(times), (*times, terminal), videos, tuple(self.views), source_identity, receipt,
                          segments, dict(self.views))
        self._loaded[ep] = episode
        return episode

    def make_evidence(self, ep, indices, directory, tile_size=(320, 240), *, frame_offset=0):
        episode = ep if isinstance(ep, Episode) else self.load_episode(ep)
        if episode.source_identity["source_root"] != str(self.source):
            raise ValueError("Episode belongs to a different source")
        return make_evidence(episode, indices, directory, tile_size, frame_offset=frame_offset)


def _image(raw, width, height, size):
    """An rgb24 frame as an image of ``size``, resized with LANCZOS only if its size differs."""
    image = Image.frombytes("RGB", (width, height), raw)
    return image if image.size == size else image.resize(size, Image.Resampling.LANCZOS)


def _frame(video, index, shape, size):
    """Native frame ``index`` of a one-episode (v2) video, decoded from its start."""
    ffmpeg = _tool("ffmpeg")
    command = [ffmpeg, "-v", "error", "-nostdin", "-threads", "1", "-noautorotate",
               "-i", str(video), "-map", "0:v:0", "-an", "-sn", "-dn",
               "-vf", f"select=eq(n\\,{index})", "-frames:v", "1", *passthrough(ffmpeg), "-threads", "1",
               "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"]
    raw = subprocess.run(command, capture_output=True, check=True).stdout
    if len(raw) != shape[0] * shape[1] * 3:
        raise ValueError("Exact source-frame decode returned an invalid RGB payload")
    return _image(raw, shape[1], shape[0], size)


def _shared_frames(ep, view, indices, size, fps):
    """Native frames ``indices`` (increasing) of one camera of a v3 episode, read from its shared video in one run.

    FFmpeg reads the packets around them, as _ffprobe_segment does, and keeps a frame only within a quarter frame of
    a requested frame's presentation time: each image is the frame the episode's packet hash covers, never one
    counted from a seek point. Exactly one frame must come out per requested frame.
    """
    segment = ep.video_segments[view]
    pts = episode_frame_times(segment, fps, ep.source_identity["video_packets_sha256"][view])
    times = [pts[i] for i in indices]
    keep = [f"lt(abs(t-{time:.6f})\\,{MATCH_FRAMES / fps:.6f})" for time in times]
    while len(keep) > 1:   # summed as a balanced tree: FFmpeg's parser refuses a flat sum of 99 such terms
        keep = ["(" + "+".join(keep[i:i + 2]) + ")" for i in range(0, len(keep), 2)]
    ffmpeg = _tool("ffmpeg")
    command = [ffmpeg, "-v", "error", "-nostdin", "-threads", "1", "-noautorotate", *read_around(times, fps),
               "-i", str(segment.path), "-map", "0:v:0", "-an", "-sn", "-dn",
               "-vf", f"select={keep[0]}", *passthrough(ffmpeg), "-threads", "1",
               "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"]
    height, width, _ = ep.clock_receipt["cameras"][view]["shape"]
    images, frame_bytes = [], width * height * 3
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=errors)
        try:
            while len(raw := process.stdout.read(frame_bytes)) == frame_bytes and len(images) < len(times):
                images.append(_image(raw, width, height, size))
            exact = not raw and process.wait() == 0 and len(images) == len(times)   # no partial or extra frame
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            process.stdout.close()
        if not exact:
            errors.seek(0)
            lines = [line.strip() for line in errors.read().decode(errors="replace").splitlines() if line.strip()]
            # FFmpeg's last line gives the reason, after the parts it names: "[dec:h264 @ 0x…] Error submitting packet…"
            reason = re.sub(r"^(\[[^\]]*\] *)+", "", lines[-1]) if lines else ""
            raise ValueError(f"Could not decode exactly the requested frames of {view} from {segment.path.name}"
                             + (f": {reason}" if reason else ""))
    return images


def make_evidence(ep: Episode, indices, directory, tile_size=(320, 240), *, frame_offset=0):
    """Decode native ranks, optionally labeling a clip from frame zero.

    ``indices`` always addresses the original source. ``frame_offset`` changes
    only the displayed/returned frame IDs; timestamps remain native source time.
    Callers must retain that offset in the clip's source identity. A v3 episode's
    frames are all decoded, from its shared videos, before anything is written.
    """
    if not isinstance(indices, (list, tuple)) or not indices:
        raise ValueError("Evidence needs explicit source-frame indices")
    selected = [_integer(i, "evidence index") for i in indices]
    if len(set(selected)) != len(selected) or any(not 0 <= i < ep.frame_count for i in selected):
        raise ValueError("Evidence indices must be unique real source frames")
    frame_offset = _integer(frame_offset, "frame_offset")
    if not 0 <= frame_offset <= min(selected):
        raise ValueError("frame_offset must be nonnegative and not exceed any selected frame")
    if (not isinstance(tile_size, (list, tuple)) or len(tile_size) != 2
            or any(_integer(x, "tile dimension") <= 0 for x in tile_size)):
        raise ValueError("tile_size must be [width,height] positive integers")
    destination = Path(directory).expanduser().resolve()
    project, source = PROJECT_ROOT, Path(ep.source_identity["source_root"])
    if destination.is_relative_to(project) or destination.is_relative_to(source) or any((p / ".git").exists() for p in (destination, *destination.parents)):
        raise ValueError("Evidence payload must be outside code checkouts and the raw source")
    width, height = tile_size
    if ep.video_segments:
        # A shared file may gain unrelated episodes. Revalidate this episode's own rows, pictures and clock,
        # and decode from its current segment rather than trusting a stale metadata/file snapshot.
        reader = LeRobotSource(source, ep.video_features, allow_shared_videos=True)
        current = reader.load_episode(ep.episode_index)
        if not same_episode(ep.source_identity, current.source_identity) or current.clock_receipt != ep.clock_receipt:
            raise ValueError("Source episode changed after clock validation")
        ep, fps = current, reader.shared.fps
        order = sorted(selected)
        with ThreadPoolExecutor(max_workers=min(len(ep.view_names), PROBE_THREADS)) as pool:   # one run per camera
            jobs = {view: pool.submit(_shared_frames, ep, view, order, (width, height), fps) for view in ep.view_names}
        decoded = {view: dict(zip(order, job.result(), strict=True)) for view, job in jobs.items()}
    else:
        decoded = {}
        for view, path in ep.video_paths.items():
            if _file_sha(path) != ep.source_identity["video_sha256"][view]:
                raise ValueError("Source video changed after clock validation")
    destination.mkdir(parents=True, exist_ok=True)
    columns = min(2, len(ep.view_names))
    rows, label_height = math.ceil(len(ep.view_names) / columns), 24
    records = []
    for index in selected:
        output = destination / f"episode_{ep.episode_index:06d}_frame_{index:06d}.png"
        if output.exists():
            raise FileExistsError(f"Evidence already exists: {output}")
        mosaic = Image.new("RGB", (columns * width, rows * (height + label_height)), "black")
        draw = ImageDraw.Draw(mosaic)
        for ordinal, view in enumerate(ep.view_names):
            shape = ep.clock_receipt["cameras"][view]["shape"]
            image = decoded[view][index] if decoded else _frame(ep.video_paths[view], index, shape, (width, height))
            x, y = (ordinal % columns) * width, (ordinal // columns) * (height + label_height)
            mosaic.paste(image, (x, y + label_height))
            draw.text((x + 3, y + 3), f"{view} | f={index - frame_offset}", fill="white")
        with output.open("xb") as stream:
            mosaic.save(stream, format="PNG")
        records.append({"frame": index - frame_offset, "timestamp": ep.timestamps[index], "image_path": str(output),
                        "sha256": _sha(output), "view_names": list(ep.view_names)})
    return records
