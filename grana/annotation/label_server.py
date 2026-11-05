# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``grana label``: a local page for marking subtasks, outcomes and action quality.

The page is the Annotation Lab editor (``label_ui/``). It reads the LeRobot v2.x
or v3.0 episodes named by a cache recipe and writes one label file per episode
into the recipe's ``labels`` directory. A v3 episode shares its video files with
other episodes, so the page plays an H.264 clip of its frames, cut once into
``.video-cache/`` next to the labels directory and shared by its stage versions
(if the folder holding the labels directory is read-only, inside the
``.stage-versions/`` there, or else inside the labels directory).
``grana build-cache`` with the same recipe turns finished files into a training
cache. The target preview uses the same function as cache construction. An
11-key file from the first ``grana label`` opens as it is; its next save
rewrites it in the current format after copying the original to
``<labels>/.v1-backup/``.
"""
from __future__ import annotations

from contextlib import contextmanager, nullcontext, suppress
from copy import deepcopy
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import ipaddress
import json
import os
from pathlib import Path
import select
import shutil
import signal
import socket
import subprocess
import threading
import time
from urllib.parse import parse_qs, urlsplit
import uuid

from .labels import (completion_problems, dense_labels, episode_lock, from_legacy, is_legacy, is_legacy_file,
                     label_path, legacy_revision, new_labels, validate_binding, validate_labels,
                     validate_legacy_binding, write_labels)
from .source import CLOCK_TOLERANCE_SECONDS, MATCH_FRAMES, LeRobotSource, episode_frame_times, read_around, same_episode
from .stage_settings import StageConflict, StageSettings, revision
from grana._loopback import local_request
from grana.data.builder import episode_targets, validate_build_config, validate_label_clock

UI = Path(__file__).with_name("label_ui")
CONTENT_TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
                 ".css": "text/css; charset=utf-8", ".svg": "image/svg+xml"}
MAX_BODY = 4 * 2**20
PREVIEW_POINTS = 1200
VIDEO_CHUNK = 8 * 2**20
CUT_POLL_SECONDS = 0.2   # how often a running or waiting clip cut checks that a browser still wants it
STALE_PARTIAL_SECONDS = 3600


def _asset(path: str):
    """A file directly inside label_ui/ with a served suffix, or None."""
    name = "index.html" if path == "/" else path[1:]
    if not name or "/" in name or name.startswith(".") or Path(name).suffix not in CONTENT_TYPES:
        return None
    return name if (UI / name).is_file() else None


class CutAbandoned(Exception):
    """Every request for a clip has gone, e.g. because the browser left the episode."""


def _cut(segment, times, fps, clip: Path, stop=lambda: False) -> None:
    """Cut one v3 episode out of its shared video as an H.264 clip of exactly its frames, starting at 0 s.

    ``times`` are the presentation times of the episode's frames in the shared video (episode_frame_times). The cut keeps
    the frames from a quarter frame before the first to a quarter frame after the last, matched by presentation time as
    evidence frames are, never counted from a seek point: an episode's first frame may lie up to half a frame before its
    from_timestamp. Any other frame in that span changes the count, and the cut is refused.
    ``stop`` is polled while FFmpeg runs; once it returns true, the cut ends with CutAbandoned.
    """
    clip.parent.mkdir(parents=True, exist_ok=True)
    # Each cut writes its own file; a process cutting the same clip at the same time renames a complete copy too.
    partial = clip.with_name(f".{clip.stem}.{uuid.uuid4().hex}.partial.mp4")
    frames, match = len(times), MATCH_FRAMES / fps
    # The clip restarts at 0 s with the episode's frames.
    keep = f"select=between(t\\,{times[0] - match:.6f}\\,{times[-1] + match:.6f}),setpts=N/({fps}*TB)"
    try:
        process = subprocess.Popen(["ffmpeg", "-v", "error", "-nostdin", "-y", *read_around(times, fps), "-i", str(segment.path),
                                    "-map", "0:v:0", "-an", "-vf", keep,
                                    "-r", str(fps), "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
                                    "-movflags", "+faststart", str(partial)],
                                   stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        try:
            while True:
                try:
                    errors = process.communicate(timeout=CUT_POLL_SECONDS)[1]
                    break
                except subprocess.TimeoutExpired:
                    if stop():
                        raise CutAbandoned from None
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate()
        if process.returncode:
            raise ValueError(f"Could not cut an episode clip from {segment.path.name}: {errors[-500:].strip()}")
        probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "packet=pts_time",
                                "-of", "json", str(partial)], capture_output=True, text=True, check=True)
        pts = sorted(float(row.get("pts_time", "nan")) for row in json.loads(probe.stdout).get("packets", []))
        # The browser maps time t to frame round(t * fps): exactly the episode's frames, at i / fps.
        if len(pts) != frames or not all(abs(t - i / fps) <= CLOCK_TOLERANCE_SECONDS for i, t in enumerate(pts)):
            raise ValueError(f"Could not cut the episode's {frames} frames from {segment.path.name}")
        os.replace(partial, clip)
    except subprocess.CalledProcessError as error:
        raise ValueError(f"Could not cut an episode clip from {segment.path.name}: {error.stderr[-500:].strip()}") from error
    finally:
        partial.unlink(missing_ok=True)


class _CutSlots:
    """At most ``size`` clip cuts at once. A free slot goes to the newest request, so cuts for episodes the browser
    has left cannot hold up the one on screen; a request stops waiting once ``stop`` returns true."""

    def __init__(self, size: int):
        self.size, self.running, self.waiting = size, 0, []
        self.condition = threading.Condition()

    @contextmanager
    def slot(self, stop):
        ticket = object()
        with self.condition:
            self.waiting.append(ticket)
            try:
                while True:
                    if stop():
                        raise CutAbandoned
                    if self.running < self.size and self.waiting[-1] is ticket:
                        break
                    self.condition.wait(CUT_POLL_SECONDS)
                self.running += 1
            finally:
                self.waiting.remove(ticket)
                self.condition.notify_all()
        try:
            yield
        finally:
            with self.condition:
                self.running -= 1
                self.condition.notify_all()

    def wait_idle(self, timeout: float) -> None:
        with self.condition:
            self.condition.wait_for(lambda: not self.running, timeout)


class SaveConflict(Exception):
    """Another save reached this episode after the sender loaded it."""

    def __init__(self, index: int, current: dict, *, anonymous: bool = False):
        self.annotator, self.updated_at = current["annotator"] or "someone", current["updated_at"]
        self.revision, self.anonymous = current["revision"], anonymous
        # Nobody to name: the file was removed, or it is an 11-key file, which records who wrote it with the
        # first grana label rather than a save made after this page loaded it.
        super().__init__(f"The saved labels for episode {index} changed after this page loaded them." if anonymous
                         else f"Episode {index} was saved by {self.annotator} at {self.updated_at} after you opened it.")


class LabelProject:
    """Episodes, clocks and label files for one cache recipe."""

    def __init__(self, cfg: dict, config_path=None):
        cfg = validate_build_config(cfg)
        if "labels" not in cfg:
            raise ValueError("Add `labels: <directory>` to the recipe; grana label saves one file per episode there")
        self.cfg = cfg
        self.source, self.labels = Path(cfg["source"]), Path(cfg["labels"])
        self.dataset = LeRobotSource(self.source, cfg["views"], allow_shared_videos=True)
        self.fps = self.dataset.info.get("fps")
        if isinstance(self.fps, bool) or not isinstance(self.fps, (int, float)) or not 0 < self.fps < float("inf"):
            raise ValueError("Source fps must be finite and positive")
        if cfg["fps"] > self.fps:
            raise ValueError("Output fps cannot exceed source fps")
        self.episodes = sorted(self.dataset.episodes) if cfg["episodes"] == "all" else list(cfg["episodes"])
        missing = set(self.episodes) - self.dataset.episodes.keys()
        if missing:
            raise ValueError(f"Episodes absent from the dataset: {sorted(missing)}")
        if not shutil.which("ffprobe"):
            raise FileNotFoundError("ffprobe (part of FFmpeg) must be on PATH to check the episode videos")
        # v3 episodes are checked by their packets and played as clips, both with ffmpeg.
        if self.dataset.shared is not None and not shutil.which("ffmpeg"):
            raise FileNotFoundError("ffmpeg (with libx264) must be on PATH to play LeRobot v3 episodes")
        self.stages = [stage["label"] for stage in cfg["stages"]]
        self.settings = StageSettings(cfg, config_path)
        # Clips are named after their packets, so every stage version of these labels can play the same ones. They go
        # next to the labels directory or, if the folder holding it is read-only, into the .stage-versions there, which
        # saves need anyway; with neither, into the labels directory this server started with.
        history = self.settings.history
        self._clip_folders = (history.parent / ".video-cache", history / ".video-cache", self.labels / ".video-cache")
        self._clip_requests, self._clip_lock = {}, threading.Lock()   # clip -> (lock for its cut, requests waiting)
        self._cut_slots = _CutSlots(len(cfg["views"]))   # one episode's cameras at once
        self._closing = threading.Event()   # set when the server stops
        # Old partial clips were left by a server stopped mid-cut; a newer one may be another server's cut in progress.
        for partial in (path for folder in self._clip_folders for path in folder.glob(".*.partial.mp4")):
            with suppress(OSError):
                if time.time() - partial.stat().st_mtime > STALE_PARTIAL_SECONDS:
                    partial.unlink()

    def close(self, timeout: float = 5.0) -> None:
        """Stop the clip cuts still running, as the server stops: each one ends FFmpeg and deletes its partial file."""
        self._closing.set()
        self._cut_slots.wait_idle(timeout)

    def _clip_folder(self) -> Path:
        """The first clip folder that exists or can be made, and is writable."""
        beside, history, started = self._clip_folders
        for folder, parents in ((beside, True), (history, False)):   # never creates .stage-versions itself
            try:
                folder.mkdir(parents=parents, exist_ok=True)
                if os.access(folder, os.W_OK | os.X_OK):
                    return folder
            except OSError:
                pass
        return started

    def update_stages(self, body):
        with self.settings.lock:
            cfg, version, reopened = self.settings.update(self.cfg, body, self._saved_documents)
            self.cfg, self.labels = cfg, Path(cfg["labels"])
            self.stages = [stage["label"] for stage in cfg["stages"]]
            return {"project": self.project(), "changed": version is not None, "reopened": reopened,
                    "previous_recipe": str(version / "previous-recipe.yaml") if version else None}

    def _saved_documents(self):
        """Validate all saved episodes before carrying them to a new recipe."""
        for path in sorted(self.labels.glob("episode_*.json")):
            index = int(path.stem.removeprefix("episode_"))
            if path != label_path(self.labels, index):
                raise ValueError(f"Unexpected label filename: {path.name}")
            # Include saved episodes outside the current selection; their labels
            # must remain usable when the recipe selects them again.
            yield self._document(index, refresh=True)

    def clock(self, index: int) -> list[float]:
        episode = self.dataset.load_episode(index)
        validate_label_clock(episode, self.fps)
        return list(episode.timestamps)

    def status(self, index: int) -> str:
        path = label_path(self.labels, index)
        if not path.is_file():
            return "new"
        try:
            doc = json.loads(path.read_text())
            return "done" if isinstance(doc, dict) and doc.get("done") is True else "draft"
        except (OSError, ValueError):
            return "draft"

    def project(self) -> dict:
        return {"task": self.cfg["task"], "compressed_task_prompt": self.cfg["compressed_task_prompt"],
                "stages": self.cfg["stages"], "views": list(self.cfg["views"]), "fps": self.fps,
                "revision": revision(self.cfg), "stages_editable": self.settings.path is not None,
                "labels": str(self.labels), "episodes": [{"index": i, "length": self.dataset.episodes[i]["length"],
                                                           "status": self.status(i)} for i in self.episodes]}

    def _document(self, index: int, *, refresh: bool = False) -> dict:
        # Saving rechecks the source, not only the startup snapshot: a file whose inode, size or times
        # changed is hashed and probed again (unchanged files reuse their results).
        reader = LeRobotSource(self.source, self.cfg["views"], allow_shared_videos=True) if refresh else self.dataset
        episode = reader.load_episode(index)
        validate_label_clock(episode, self.fps)
        blank = new_labels(index, episode.source_identity["trajectory_id"], episode.timestamps, self.stages,
                           source_identity=episode.source_identity, stage_registry=self.cfg["stages"])
        path = label_path(self.labels, index)
        if not path.is_file():
            return blank
        raw = path.read_bytes()
        doc = json.loads(raw)
        if is_legacy(doc):
            validate_legacy_binding(doc, index, episode.source_identity["trajectory_id"], episode.timestamps)
            doc = from_legacy(doc, source_identity=episode.source_identity, stage_registry=self.cfg["stages"],
                              revision=legacy_revision(raw))
        doc = validate_labels(doc, frame_count=episode.frame_count, stages=self.stages)
        validate_binding(doc, episode, self.cfg["stages"])
        # Bound to the episode, not to where the dataset lives: carry the current location and metadata,
        # which the next save writes.
        return {**doc, "source_identity": deepcopy(episode.source_identity)}

    def _keep_legacy(self, index: int) -> None:
        """Copy an 11-key file aside before its first save in the current format (never overwrites a copy)."""
        path = label_path(self.labels, index)
        try:
            raw = path.read_bytes()
        except FileNotFoundError:
            return
        if is_legacy(json.loads(raw)):
            backup = self.labels / ".v1-backup" / f"{path.stem}.{hashlib.sha256(raw).hexdigest()[:16]}.json"
            backup.parent.mkdir(exist_ok=True)
            try:
                with open(os.open(backup, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as stream:
                    stream.write(raw)
            except FileExistsError:
                pass

    def episode(self, index: int) -> dict:
        self._known(index)
        doc = self._document(index)
        return {"index": index, "frame_count": doc["frame_count"], "fps": self.fps, "trajectory_id": doc["trajectory_id"],
                "project_revision": revision(self.cfg),
                "views": {view: f"/video?episode={index}&view={view}" for view in self.cfg["views"]}, "labels": doc}

    def video(self, index: int, view: str, gone=lambda: False) -> Path:
        """The episode's video for the browser: the original file (v2) or a cached H.264 clip of its frames (v3).

        ``gone`` tells whether this request was dropped; a clip's cut stops once every request for it has gone.
        """
        self._known(index)
        if view not in self.cfg["views"]:
            raise KeyError(view)
        self.clock(index)
        episode = self.dataset.load_episode(index)
        if view not in episode.video_segments:
            return episode.video_paths[view]
        # Picture packets can be remuxed at another frame rate without changing their bytes. Bind the clip
        # to its playback clock and geometry as well; stage versions still share this unchanged contract.
        camera = episode.clock_receipt["cameras"][view]
        # Version 3 matches frames by presentation time; a version 2 clip could start one frame late.
        contract = {"version": 3, "packets": episode.source_identity["video_packets_sha256"][view],
                    "fps": float(self.fps), "frames": episode.frame_count,
                    "pts": camera["pts_sha256"], "shape": camera["shape"]}
        digest = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
        clip = self._clip_folder() / f"episode_{index:06d}_{view}_{digest[:16]}.mp4"
        with self._clip_lock:
            lock, requests = self._clip_requests.setdefault(clip, (threading.Lock(), []))
            requests.append(gone)

        def stop():   # the server is stopping, or every request for this clip has gone
            with self._clip_lock:
                waiting = list(requests)
            return self._closing.is_set() or all(left() for left in waiting)

        try:
            with lock:
                if not clip.is_file():
                    current = LeRobotSource(self.source, self.cfg["views"], allow_shared_videos=True).load_episode(index)
                    if not same_episode(current.source_identity, episode.source_identity):
                        raise ValueError("Source changed while the label server was running; restart and review the new source")
                    segment = current.video_segments[view]
                    times = episode_frame_times(segment, self.fps, current.source_identity["video_packets_sha256"][view])
                    with self._cut_slots.slot(stop):
                        _cut(segment, times, self.fps, clip, stop)
        finally:
            with self._clip_lock:
                requests.remove(gone)
                if not requests:
                    del self._clip_requests[clip]
        return clip

    def _merge(self, index: int, body: dict, current=None) -> dict:
        if not isinstance(body, dict) or set(body) - {"subtasks", "quality", "annotator", "done", "review_confirmed"}:
            raise ValueError("Send subtasks, quality, annotator, done and review_confirmed")
        if {"subtasks", "quality"} & body.keys() and "review_confirmed" not in body:
            body = {**body, "review_confirmed": False}
        doc = {**(current if current is not None else self._document(index)), **body}
        return validate_labels(doc, frame_count=doc["frame_count"], stages=self.stages)

    def save(self, index: int, body: dict) -> dict:
        """Compare an opaque revision and save atomically without losing another editor's work."""
        self._known(index)
        if not isinstance(body, dict) or "base_revision" not in body:
            raise ValueError("A save requires base_revision from the loaded label document")
        body = dict(body)
        base = body.pop("base_revision")
        with self.settings.writing(), episode_lock(self.labels, index):
            body = self.settings.check(self.cfg, body)
            snapshot = self.dataset.load_episode(index)
            current = self._document(index, refresh=True)
            if not same_episode(current["source_identity"], snapshot.source_identity):
                raise ValueError("Source changed while the label server was running; restart and review the new source")
            if current["revision"] != base:
                raise SaveConflict(index, current, anonymous=current["revision"] is None or is_legacy_file(self.labels, index))
            doc = self._merge(index, body, current)
            if doc["done"]:
                labels, scores, quality = dense_labels(doc)
                targets = episode_targets(self.cfg, labels, scores, quality, list(snapshot.timestamps), self.fps)
                if not targets["source_indices"]:
                    raise ValueError("These labels produce no training frames under the recipe's target policy")
            self._keep_legacy(index)
            write_labels(self.labels, doc)
            doc = self._document(index)
        return {"status": "done" if doc["done"] else "draft", "problems": completion_problems(doc),
                "updated_at": doc["updated_at"], "annotator": doc["annotator"], "revision": doc["revision"], "labels": doc}

    def preview(self, index: int, body: dict) -> dict:
        """The cache target for the current (unsaved) labels, or why it cannot be computed yet."""
        self._known(index)
        if not isinstance(body, dict):
            raise ValueError("Preview body must be a JSON object")
        body = self.settings.check(self.cfg, body)
        doc = self._merge(index, {**body, "done": False, "review_confirmed": True})
        problems = [p for p in completion_problems(doc) if "annotator" not in p]
        if problems:
            return {"problems": problems}
        labels, scores, quality = dense_labels(doc)
        times = self.clock(index)
        try:
            targets = episode_targets(self.cfg, labels, scores, quality, times, self.fps)
        except ValueError as error:
            return {"problems": [str(error)]}
        frames, progress = targets["source_indices"], targets["progress"]
        step = max(1, len(frames) // PREVIEW_POINTS)
        return {"problems": [], "frames": frames[::step] + frames[-1:], "progress": progress[::step] + progress[-1:]}

    def _known(self, index: int) -> None:
        if index not in self.episodes:
            raise KeyError(index)


class LabelHandler(BaseHTTPRequestHandler):
    server_version = "GranaLabel"

    def log_message(self, *args):
        pass

    def _send(self, status, body: bytes, content_type: str, extra=()):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for name, value in extra:
            self.send_header(name, value)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, value, status=200):
        self._send(status, json.dumps(value, allow_nan=False).encode(), "application/json")

    def _episode(self, query) -> int:
        values = query.get("episode", [])
        if len(values) != 1 or not values[0].isdigit():
            raise KeyError("episode")
        return int(values[0])

    def _guard(self, handler, *, lock=True):
        try:
            with self.server.project.settings.lock if lock else nullcontext():
                handler()
        except KeyError:
            self._json({"error": "not found"}, 404)
        except CutAbandoned:   # the browser has left; nobody is waiting for an answer
            pass
        except SaveConflict as error:
            fields = {"anonymous": True} if error.anonymous else {"annotator": error.annotator, "updated_at": error.updated_at}
            self._json({"error": str(error), **fields, "revision": error.revision}, 409)
        except StageConflict as error:
            self._json({"error": str(error), "project_conflict": True}, 409)
        except (ValueError, FileNotFoundError) as error:
            self._json({"error": str(error)}, 400)
        except subprocess.SubprocessError:
            self._json({"error": "Could not validate the source video with ffprobe"}, 400)
        except OSError as error:   # e.g. a full disk or a labels directory without write permission
            self._json({"error": f"Could not read or write labels: {error.strerror or error} ({error.filename or self.server.project.labels})"}, 500)

    def _gone(self) -> bool:
        """Whether the browser closed this request's connection, e.g. because it left the episode."""
        try:
            poll = select.poll()
            poll.register(self.connection, select.POLLIN)
            return bool(poll.poll(0)) and not self.connection.recv(1, socket.MSG_PEEK)
        except (OSError, ValueError):   # reset, or closed already
            return True

    def _refuse_remote(self) -> bool:
        """Answer 403 unless Host and Origin name this loopback server (DNS-rebinding protection).

        Browsers send no Origin on no-cors GETs such as a cross-site <video src>, so a request the browser marks
        as cross-site (Sec-Fetch-Site) is refused too; it could otherwise start clip cuts.
        """
        if local_request(self.headers) and self.headers.get("Sec-Fetch-Site") in (None, "same-origin", "none"):
            return False
        self._json({"error": "grana label accepts requests for 127.0.0.1 or localhost only"}, 403)
        return True

    def do_GET(self):
        if self._refuse_remote():
            return
        url = urlsplit(self.path)
        query = parse_qs(url.query)
        project = self.server.project
        name = _asset(url.path)
        if name:
            return self._send(200, (UI / name).read_bytes(), CONTENT_TYPES[Path(name).suffix])
        routes = {"/api/project": lambda: self._json(project.project()),
                  "/api/episode": lambda: self._json(project.episode(self._episode(query))),
                  "/video": lambda: self._video(project.video(self._episode(query), query.get("view", [""])[0], self._gone))}
        if url.path not in routes:
            return self._json({"error": "not found"}, 404)
        # Stage edits never change the source videos. Streaming them must not
        # block other camera streams or hold up label/configuration saves.
        self._guard(routes[url.path], lock=url.path != "/video")

    do_HEAD = do_GET

    def do_POST(self):
        if self._refuse_remote():
            return
        url = urlsplit(self.path)
        project = self.server.project
        if url.path not in ("/api/labels", "/api/preview", "/api/stages"):
            return self._json({"error": "not found"}, 404)
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            return self._json({"error": "Content-Length must be an integer"}, 400)
        if not 0 < length <= MAX_BODY or not self.headers.get("Content-Type", "").startswith("application/json"):
            return self._json({"error": "send a JSON body up to 4 MiB"}, 400)
        body = self.rfile.read(length)

        def respond():
            payload = json.loads(body)
            if url.path == "/api/stages":
                return self._json(project.update_stages(payload))
            index = self._episode(parse_qs(url.query))
            self._json(project.save(index, payload) if url.path == "/api/labels" else project.preview(index, payload))
        self._guard(respond)

    def _video(self, path: Path):
        """Serve a video file with byte ranges, streaming at most 8 MiB per open-ended range."""
        size = path.stat().st_size
        start, end, status = 0, size - 1, 200
        header = self.headers.get("Range")
        if header:
            try:
                unit, _, spec = header.partition("=")
                first, _, last = spec.partition("-")
                if unit.strip() != "bytes" or "," in spec:
                    raise ValueError
                if first:
                    start = int(first)
                    end = min(int(last), size - 1) if last else min(start + VIDEO_CHUNK - 1, size - 1)
                else:
                    start = max(0, size - int(last))
                if not 0 <= start <= end:
                    raise ValueError
            except ValueError:
                return self._send(416, b"", "text/plain", [("Content-Range", f"bytes */{size}")])
            status = 206
        self.send_response(status)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        if self.command == "HEAD":
            return
        remaining = end - start + 1
        try:
            with path.open("rb") as stream:
                stream.seek(start)
                while remaining:
                    block = stream.read(min(remaining, 2**20))
                    if not block:
                        break
                    self.wfile.write(block)
                    remaining -= len(block)
        except (BrokenPipeError, ConnectionResetError):
            pass


def create_label_server(cfg: dict, host: str = "127.0.0.1", port: int = 8790, *, config_path=None) -> ThreadingHTTPServer:
    if not ipaddress.ip_address("127.0.0.1" if host == "localhost" else host).is_loopback:
        raise ValueError("grana label binds a loopback address; forward the port to reach it remotely")
    project = LabelProject(cfg, config_path)
    server = ThreadingHTTPServer((host, port), LabelHandler)
    server.project = project
    return server


def _interrupt(signum, frame):
    raise KeyboardInterrupt


def serve_labels(cfg: dict, host: str = "127.0.0.1", port: int = 8790, *, config_path=None) -> None:
    server = create_label_server(cfg, host, port, config_path=config_path)
    print(f"Labeling {len(server.project.episodes)} episodes; labels go to {server.project.labels}")
    print(f"Open http://{host}:{server.server_address[1]}/  (Ctrl+C to stop)")
    # systemd and kill stop the server with SIGTERM: end as on Ctrl+C, so running clip cuts end too.
    main = threading.current_thread() is threading.main_thread()
    previous = signal.signal(signal.SIGTERM, _interrupt) if main else None
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if main:   # a second SIGTERM while closing ends the process at once
            signal.signal(signal.SIGTERM, signal.SIG_DFL if previous is None else previous)
        server.project.close()
        server.server_close()
