# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only, loopback video review of immutable stage-annotation candidates."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import ipaddress
import json
import math
import os
from pathlib import Path
import re
import stat
import threading
from urllib.parse import parse_qs, urlsplit
import uuid

from .schema import coverage, validate_candidate

MAX_BODY_BYTES = 2 * 1024 * 1024


def _json_bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _json_loads(value):
    def pairs(items):
        result = {}
        for key, item in items:
            if key in result:
                raise ValueError("Duplicate JSON keys are not accepted")
            result[key] = item
        return result

    def constant(_):
        raise ValueError("Nonfinite JSON numbers are not accepted")

    return json.loads(value, object_pairs_hook=pairs, parse_constant=constant)


def _no_symlinks(path):
    if not path.is_absolute() or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("Review paths must be absolute and contain no symlinks")
    return path


def _read_file(path):
    _no_symlinks(path)
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError("Review input must be a regular file")
        return stream.read()


def _video_identity(path, expected):
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError("source_identity.video_sha256 must bind every original camera")
    _no_symlinks(path)
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("Allowlisted video must be a regular file")
        identity = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    current = path.stat()
    if digest != expected or identity != (current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns):
        raise ValueError("Original video SHA256 differs from the manifest source identity")
    return identity


def _relative_file(root, value):
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("Candidate paths must be relative files within the run")
    relative = Path(value)
    if relative.is_absolute() or any(part in (".", "..") for part in value.split("/")):
        raise ValueError("Candidate paths must remain within the run")
    return _no_symlinks(root / relative)


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be explicitly supplied nonempty text")
    return value


def validate_review(value):
    """Validate explicit self-declared review metadata without assigning a decision."""
    keys = {"reviewer", "kind", "decision", "notes", "evidence_inspected"}
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError("review requires reviewer, kind, decision, notes and evidence_inspected")
    _text(value["reviewer"], "reviewer")
    if value["kind"] not in ("human", "agent"):
        raise ValueError("review.kind must explicitly be human or agent")
    if value["decision"] not in ("accepted", "needs_changes"):
        raise ValueError("review.decision must explicitly be accepted or needs_changes")
    _text(value["notes"], "review.notes")
    evidence = value["evidence_inspected"]
    if not isinstance(evidence, list) or not evidence:
        raise ValueError("evidence_inspected must list the inspected evidence")
    for item in evidence:
        _text(item, "evidence_inspected entry")
    if len({item.strip() for item in evidence}) != len(evidence):
        raise ValueError("evidence_inspected entries must be distinct")
    return deepcopy(value)


class ReviewStore:
    """Freeze manifest and original candidates; write only new revision files."""

    def __init__(self, run_dir):
        self.root = _no_symlinks(Path(run_dir).expanduser().absolute())
        self.manifest = _json_loads(_read_file(self.root / "manifest.json"))
        manifest = self.manifest
        if not isinstance(manifest, dict) or type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1:
            raise ValueError("Annotation manifest schema_version must be 1")
        _text(manifest.get("task"), "task")
        stages = manifest.get("stages")
        if not isinstance(stages, list) or not 1 <= len(stages) <= 26:
            raise ValueError("Manifest requires 1..26 ordered stages")
        for stage in stages:
            if not isinstance(stage, dict) or set(stage) != {"id", "label", "prompt"}:
                raise ValueError("Each stage requires exactly id, label and prompt")
            for key in stage:
                _text(stage[key], f"stage {key}")
        self.stage_ids = [stage["id"] for stage in stages]
        if len(set(self.stage_ids)) != len(self.stage_ids):
            raise ValueError("Stage IDs must be distinct")
        episodes = manifest.get("episodes")
        if not isinstance(episodes, list) or not episodes:
            raise ValueError("Manifest must list episodes")
        self.episodes, self.originals, self.media = {}, {}, {}
        self.lock = threading.RLock()
        for episode in episodes:
            if not isinstance(episode, dict):
                raise ValueError("Episode must be an object")
            index, count = episode.get("episode_index"), episode.get("frame_count")
            if type(index) is not int or index < 0 or index in self.episodes:
                raise ValueError("Episode indices must be distinct nonnegative integers")
            if type(count) is not int or count < 1:
                raise ValueError("frame_count must be positive")
            candidate = validate_candidate(_json_loads(_read_file(_relative_file(self.root, episode.get("candidate_file")))), self.stage_ids, count)
            times = episode.get("time_edges")
            if not isinstance(times, list) or len(times) != count + 1:
                raise ValueError("Episode time_edges must contain frame_count + 1 values")
            coverage(candidate, times)
            if not isinstance(episode.get("source_identity"), dict) or not episode["source_identity"]:
                raise ValueError("Episode source_identity must be a nonempty object")
            clock = episode.get("clock_receipt")
            if not isinstance(clock, dict) or not isinstance(clock.get("cameras"), dict):
                raise ValueError("clock_receipt must include per-camera source/video offsets")
            videos = episode.get("video_paths")
            if not isinstance(videos, dict) or not videos:
                raise ValueError("Episode must list original camera video paths")
            normalized = deepcopy(episode)
            normalized.pop("video_paths")
            normalized["video_urls"] = {}
            for view, filename in videos.items():
                if not isinstance(view, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", view) or not isinstance(filename, str):
                    raise ValueError("Video views must have simple names and explicit absolute paths")
                path = _no_symlinks(Path(filename))
                expected_hashes = episode["source_identity"].get("video_sha256", {})
                if not isinstance(expected_hashes, dict):
                    raise ValueError("source_identity.video_sha256 must bind every original camera")
                identity = _video_identity(path, expected_hashes.get(view))
                camera = clock["cameras"].get(view, {})
                offset = camera.get("source_minus_video_offset_seconds")
                if isinstance(offset, bool) or not isinstance(offset, (int, float)) or not math.isfinite(offset):
                    raise ValueError("Each camera needs its finite source_minus_video_offset_seconds")
                url = f"/media/{index}/{view}"
                self.media[url] = (path, identity)
                normalized["video_urls"][view] = url
            self.episodes[index], self.originals[index] = normalized, candidate

    def public_manifest(self):
        return {"schema_version": 1, "task": self.manifest["task"], "stages": deepcopy(self.manifest["stages"]),
                "episodes": [deepcopy(item) for item in self.episodes.values()],
                "review_notice": "Machine candidates are not human-accepted annotations. Review identity is self-declared."}

    def _revision_dir(self, index):
        return _no_symlinks(self.root / "revisions" / f"episode_{index:06d}")

    def episode(self, index, revision=None):
        if index not in self.episodes:
            raise KeyError("Unknown episode")
        with self.lock:
            metadata = deepcopy(self.episodes[index])
            revision_dir = self._revision_dir(index)
            names = sorted(p.name for p in revision_dir.glob("*.json")) if revision_dir.is_dir() else []
            candidate, wrapper = deepcopy(self.originals[index]), None
            if revision is not None:
                if not re.fullmatch(r"[0-9T]+Z_[0-9a-f]{32}\.json", revision) or revision not in names:
                    raise KeyError("Unknown revision")
                wrapper = _json_loads(_read_file(revision_dir / revision))
                candidate = validate_candidate(wrapper["candidate"], self.stage_ids, metadata["frame_count"])
                validate_review(wrapper["review"])
                if (wrapper["source_identity"] != metadata["source_identity"] or wrapper["time_edges"] != metadata["time_edges"]
                        or wrapper["stages"] != self.manifest["stages"]
                        or wrapper["candidate_sha256"] != hashlib.sha256(_json_bytes(candidate)).hexdigest()):
                    raise ValueError("Revision identity or candidate hash does not match the run")
            return {"episode": metadata, "candidate": candidate, "candidate_sha256": hashlib.sha256(_json_bytes(candidate)).hexdigest(),
                    "revision": revision, "review": wrapper["review"] if wrapper else None,
                    "revisions": names, "coverage": coverage(candidate, metadata["time_edges"])}

    def save(self, payload):
        if not isinstance(payload, dict) or set(payload) != {"episode_index", "candidate", "review"}:
            raise ValueError("Revision request requires exactly episode_index, candidate and review")
        index = payload["episode_index"]
        if type(index) is not int or index not in self.episodes:
            raise ValueError("Unknown episode_index")
        episode = self.episodes[index]
        candidate = validate_candidate(payload["candidate"], self.stage_ids, episode["frame_count"])
        review = validate_review(payload["review"])
        now = datetime.now(timezone.utc)
        wrapper = {"candidate": candidate, "candidate_sha256": hashlib.sha256(_json_bytes(candidate)).hexdigest(),
                   "review": review, "source_identity": deepcopy(episode["source_identity"]),
                   "stages": deepcopy(self.manifest["stages"]), "time_edges": deepcopy(episode["time_edges"]),
                   "created_at": now.isoformat()}
        filename = now.strftime("%Y%m%dT%H%M%S%fZ") + "_" + uuid.uuid4().hex + ".json"
        with self.lock:
            directory = self._revision_dir(index)
            directory.mkdir(parents=True, exist_ok=True)
            _no_symlinks(directory)
            temporary = directory / ("." + filename + ".tmp")
            final = directory / filename
            created = False
            try:
                descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
                created = True
                with os.fdopen(descriptor, "wb") as stream:
                    stream.write(_json_bytes(wrapper) + b"\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                # Atomic publication that cannot overwrite a prior revision.
                os.link(temporary, final)
            finally:
                if created:
                    temporary.unlink(missing_ok=True)
        return {"revision": filename, "relative_path": final.relative_to(self.root).as_posix(),
                "candidate_sha256": wrapper["candidate_sha256"], "review": review, "created_at": wrapper["created_at"]}


def _byte_range(value, size):
    if value is None:
        return 0, size - 1, 200
    match = re.fullmatch(r"bytes=(\d*)-(\d*)", value)
    if not match or not size or not any(match.groups()):
        raise ValueError("Invalid byte range")
    first, last = match.groups()
    if first:
        start, end = int(first), min(int(last), size - 1) if last else size - 1
        if start >= size or end < start:
            raise ValueError("Unsatisfiable byte range")
    else:
        length = int(last)
        if length < 1:
            raise ValueError("Unsatisfiable byte range")
        start, end = max(0, size - length), size - 1
    return start, end, 206


class ReviewServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, store):
        self.store = store
        super().__init__(address, ReviewHandler)


class ReviewHandler(BaseHTTPRequestHandler):
    server_version = "GranaAnnotationReview/1"
    sys_version = ""

    def log_message(self, *_):
        pass

    def _send(self, status, body, content_type="application/json; charset=utf-8", headers=None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-store")
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _error(self, status, message, headers=None):
        self._send(status, _json_bytes({"error": message}), headers=headers)

    def _host_allowed(self):
        if len(self.headers.get_all("Host", [])) != 1:
            return False
        try:
            parsed = urlsplit("http://" + self.headers.get("Host", ""))
            return (parsed.hostname == "localhost" or ipaddress.ip_address(parsed.hostname).is_loopback) and not (
                parsed.username or parsed.password or parsed.path or parsed.query or parsed.fragment)
        except (ValueError, TypeError):
            return False

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        if not self._host_allowed():
            self._error(403, "Host must identify loopback")
            return
        try:
            if self.path == "/":
                self._send(200, HTML.encode(), "text/html; charset=utf-8")
            elif self.path == "/api/manifest":
                self._send(200, _json_bytes(self.server.store.public_manifest()))
            elif self.path in self.server.store.media:
                self._media()
            else:
                parsed = urlsplit(self.path)
                if parsed.path not in ("/api/episode", "/api/candidate") or parsed.fragment:
                    raise KeyError("URL is not allowlisted")
                query = parse_qs(parsed.query, keep_blank_values=True, strict_parsing=True)
                if set(query) - {"episode", "revision"} or "episode" not in query or any(len(v) != 1 for v in query.values()):
                    raise ValueError("Supply one episode and optional revision")
                if not re.fullmatch(r"[0-9]+", query["episode"][0]):
                    raise ValueError("episode must be a nonnegative integer")
                index = int(query["episode"][0])
                result = self.server.store.episode(index, query.get("revision", [None])[0])
                if parsed.path == "/api/candidate":
                    self._send(200, _json_bytes(result["candidate"]), headers={"Content-Disposition": f'attachment; filename="episode_{index:06d}_stages.json"'})
                else:
                    self._send(200, _json_bytes(result))
        except KeyError:
            self._error(404, "Episode, revision or URL is not allowlisted")
        except (ValueError, TypeError, UnicodeError):
            self._error(400, "Invalid review query or stored revision")
        except (OSError, RuntimeError):
            self._error(404, "Allowlisted review input is unavailable")

    def _media(self):
        path, identity = self.server.store.media[self.path]
        _no_symlinks(path)
        with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), "rb") as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns) != identity:
                raise OSError("Allowlisted original video changed")
            try:
                if len(self.headers.get_all("Range", [])) > 1:
                    raise ValueError("Only one range is supported")
                start, end, status = _byte_range(self.headers.get("Range"), info.st_size)
            except ValueError:
                self._error(416, "Only one satisfiable byte range is supported", {"Content-Range": f"bytes */{info.st_size}"})
                return
            self.send_response(status)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Content-Length", str(max(0, end - start + 1)))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("X-Content-Type-Options", "nosniff")
            if status == 206:
                self.send_header("Content-Range", f"bytes {start}-{end}/{info.st_size}")
            self.end_headers()
            if self.command == "HEAD":
                return
            stream.seek(start)
            remaining = end - start + 1
            try:
                while remaining > 0:
                    chunk = stream.read(min(64 * 1024, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
            except (BrokenPipeError, ConnectionResetError):
                pass

    def do_POST(self):
        origin = self.headers.get("Origin")
        if (not self._host_allowed() or len(self.headers.get_all("Origin", [])) != 1
                or origin != "http://" + self.headers.get("Host", "")):
            self._error(403, "Writes require an explicit same-origin loopback Origin")
            return
        if self.path != "/api/revision":
            self._error(404, "Unknown write endpoint")
            return
        if self.headers.get_content_type() != "application/json":
            self._error(415, "Content-Type must be application/json")
            return
        lengths = self.headers.get_all("Content-Length", [])
        if self.headers.get("Transfer-Encoding") or len(lengths) != 1 or not re.fullmatch(r"[0-9]+", lengths[0]):
            self._error(400, "Supply one Content-Length; transfer encoding is unsupported")
            return
        if len(lengths[0]) > 20:
            self._error(413, "Revision JSON exceeds 2 MiB")
            return
        length = int(lengths[0])
        if length > MAX_BODY_BYTES:
            self._error(413, "Revision JSON exceeds 2 MiB")
            return
        try:
            self.connection.settimeout(15)
            body = self.rfile.read(length)
            if len(body) != length:
                raise ValueError("Truncated JSON body")
            saved = self.server.store.save(_json_loads(body))
            self._send(201, _json_bytes(saved))
        except (ValueError, TypeError, UnicodeError, TimeoutError, RecursionError) as error:
            self._error(400, str(error))
        except OSError:
            self._error(500, "Revision could not be saved; no original candidate was changed")

    def _unsupported(self):
        self._error(405, "Use GET/HEAD to read and POST /api/revision to create a revision")

    do_PUT = do_PATCH = do_DELETE = do_OPTIONS = _unsupported


def create_review_server(run_dir, host="127.0.0.1", port=8788):
    """Construct a CPU-only testable server without starting its event loop."""
    if not ipaddress.ip_address(host).is_loopback or ":" in host:
        raise ValueError("Annotation review must bind IPv4 loopback")
    return ReviewServer((host, port), ReviewStore(run_dir))


def serve_review(run_dir, host="127.0.0.1", port=8788):
    """Serve until Ctrl-C; no model, API client or training runtime is loaded."""
    server = create_review_server(run_dir, host, port)
    print(json.dumps({"url": f"http://{host}:{server.server_port}/", "run_dir": str(server.store.root)}), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


HTML = r'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Grana · stage annotation review</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#f3f5f4;color:#182d32;font:16px/1.5 system-ui,sans-serif}main{max-width:1500px;margin:auto;padding:28px}h1{letter-spacing:-.03em}section{background:white;border:1px solid #d3dfdc;border-radius:12px;padding:20px;margin:18px 0}.notice{background:#fff4d8;border-left:4px solid #b08224;padding:14px}.muted{color:#51696c}button,select,input,textarea{font:inherit;padding:8px;border:1px solid #a9bfb9;border-radius:6px}button{cursor:pointer;background:#e5f3ed}button:disabled{opacity:.5;cursor:wait}label{display:block;margin:8px 0}.controls{display:flex;gap:10px;align-items:center;flex-wrap:wrap}.videos{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}video{width:100%;display:block;background:#111}figure{margin:0}figcaption{font-weight:650;padding:6px}#scrub{width:100%}#timeline{display:flex;min-height:46px;margin:12px 0}#timeline button{border-radius:0;min-width:3px;overflow:hidden;padding:5px;color:#102b28}.uncertain{outline:3px dashed #9c6a1f;outline-offset:-4px}.unresolved{background:repeating-linear-gradient(45deg,#ddd,#ddd 7px,#999 7px,#999 14px)!important}table{width:100%;border-collapse:collapse;font-size:14px}th,td{text-align:left;padding:8px;border-bottom:1px solid #ddd}textarea{width:100%;min-height:100px}#candidate{font:13px/1.5 ui-monospace,monospace;min-height:380px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}a{color:#166954}#message{white-space:pre-wrap}.table-wrap{overflow:auto}@media(max-width:700px){main{padding:12px}.videos{grid-template-columns:1fr}}
</style><main>
<h1>Stage annotation review</h1><p id="task"></p>
<p class="notice"><strong>Machine candidate ≠ human accepted.</strong> Stage identity, boundary uncertainty and invalid source intervals remain separate. A revision records a self-declared reviewer and an explicit decision; agent acceptance is not human acceptance. Stage-only review supplies no action-quality or outcome labels.</p>
<section><div class="controls"><label>Episode <select id="episode"></select></label><label>Version <select id="revision"></select></label><a id="download">Download saved stage JSON</a></div><p id="review-state"></p><details><summary>Task stage registry</summary><pre id="registry"></pre></details></section>
<section><h2>Original source videos</h2><div class="videos" id="videos"></div><div class="controls"><button id="play">Play all</button><button id="pause">Pause all</button><button id="previous">Previous frame</button><button id="next">Next frame</button><label>Source frame <input id="frame" type="number" min="0" step="1" value="0"></label><button id="seek">Seek all views</button></div><input id="scrub" type="range" min="0" value="0" step="1"><p id="clock" class="muted"></p><p class="muted">Frame boundaries are source-row ranks; intervals are [start, end). Seeking uses each camera's recorded source/video time offset. Browser playback is approximate; the source clock and bracket remain authoritative.</p></section>
<section><h2>Stage timeline</h2><p id="preview-state"></p><div id="timeline"></div><div class="table-wrap"><table><thead><tr><th>Frames</th><th>Stage / identity</th><th>Reason / evidence</th></tr></thead><tbody id="segments"></tbody></table></div><h3>Boundary brackets</h3><div class="table-wrap"><table><thead><tr><th>Nominal / bracket</th><th>Source times</th><th>Reason / evidence</th></tr></thead><tbody id="boundaries"></tbody></table></div><details><summary>Separate uncertainty coverage, omissions and invalid intervals</summary><pre id="coverage"></pre></details></section>
<section><h2>Edit and save a revision</h2><p>The original candidate stays unchanged. The server validates exact coverage, stage identities, boundary brackets and evidence frame indices before saving a new file.</p><textarea id="candidate" spellcheck="false" aria-label="Complete candidate JSON"></textarea><button id="preview">Preview unsaved JSON</button><form id="review-form"><label>Reviewer name or agent identity <input id="reviewer" required autocomplete="off"></label><div class="controls"><label>Reviewer kind <select id="kind" required><option value="">Choose explicitly</option><option value="human">Human</option><option value="agent">Agent</option></select></label><label>Decision <select id="decision" required><option value="">Choose explicitly</option><option value="needs_changes">Needs changes</option><option value="accepted">Accepted</option></select></label></div><label>Evidence inspected (one concrete video / frame / bracket description per line)<textarea id="evidence" required></textarea></label><label>Notes<textarea id="notes" required></textarea></label><button id="save" type="submit">Validate and save new revision</button></form><p id="message" role="status" aria-live="polite"></p></section>
<section><details><summary>Recording details</summary><pre id="identity"></pre></details></section>
</main><script>
'use strict';
const $=id=>document.getElementById(id), colors=['#c6e2d7','#cddcf0','#eedbbd','#e1d2eb','#d8e6bc','#eccdd1'];
let manifest, loaded, videos=[], frame=0, loading=false;
async function get(url,options){const response=await fetch(url,options);const data=await response.json();if(!response.ok)throw new Error(data.error||`HTTP ${response.status}`);return data;}
function episodeURL(route='/api/episode',revision=$('revision').value){return route+'?'+new URLSearchParams({episode:$('episode').value,...(revision?{revision}: {})});}
function stop(){videos.forEach(item=>item.video.pause());}
function setFrame(value,seek=true){if(!loaded)return;frame=Math.max(0,Math.min(loaded.episode.frame_count-1,Math.round(Number(value)||0)));$('frame').value=frame;$('scrub').value=frame;const t=loaded.episode.time_edges[frame];$('clock').textContent=`Source frame ${frame} / ${loaded.episode.frame_count-1} · source time ${t.toFixed(6)} s · interval end ${loaded.episode.time_edges[frame+1].toFixed(6)} s`;if(seek){stop();videos.forEach(item=>{if(item.video.readyState>=1)item.video.currentTime=Math.max(0,t-item.offset);});}}
function sourceFrame(time){const edges=loaded.episode.time_edges;let lo=0,hi=edges.length-1;while(lo+1<hi){const mid=(lo+hi)>>1;if(edges[mid]<=time)lo=mid;else hi=mid;}return Math.min(lo,loaded.episode.frame_count-1);}
function tick(){if(loaded&&videos.length&&!videos[0].video.paused){const sourceTime=videos[0].video.currentTime+videos[0].offset;setFrame(sourceFrame(sourceTime),false);for(const item of videos.slice(1)){const target=sourceTime-item.offset;if(Math.abs(item.video.currentTime-target)>.12)item.video.currentTime=Math.max(0,target);}}requestAnimationFrame(tick);}
function cell(row,text){const td=document.createElement('td');td.textContent=text;row.append(td);return td;}
function jumpCell(row,text,target){const td=document.createElement('td'),button=document.createElement('button');button.textContent=text;button.onclick=()=>setFrame(target);td.append(button);row.append(td);}
function render(candidate,unsaved=false){$('timeline').replaceChildren();$('segments').replaceChildren();$('boundaries').replaceChildren();$('preview-state').textContent=unsaved?'Unsaved preview · not yet validated by server':'Saved validated candidate';const n=loaded.episode.frame_count;for(const s of candidate.segments){const index=manifest.stages.findIndex(stage=>stage.id===s.stage_id),button=document.createElement('button');button.style.width=`${100*(s.end_frame-s.start_frame)/n}%`;button.style.background=colors[Math.max(index,0)%colors.length];button.className=s.identity_status;button.textContent=s.stage_id??'?';button.title=`[${s.start_frame}, ${s.end_frame}) · ${s.identity_status}`;button.onclick=()=>setFrame(s.start_frame);$('timeline').append(button);const row=document.createElement('tr');jumpCell(row,`[${s.start_frame}, ${s.end_frame})`,s.start_frame);cell(row,`${s.stage_id??'unknown'} · ${s.identity_status}`);cell(row,`${s.reason}\nEvidence: ${s.evidence_frames.join(', ')}`);$('segments').append(row);}for(const b of candidate.boundaries){const row=document.createElement('tr');jumpCell(row,`${b.frame} · [${b.lower_frame}, ${b.upper_frame}]`,b.frame);cell(row,`${loaded.episode.time_edges[b.lower_frame]} → ${loaded.episode.time_edges[b.upper_frame]} s`);cell(row,`${b.reason}\nEvidence: ${b.evidence_frames.join(', ')}`);$('boundaries').append(row);}$('coverage').textContent=JSON.stringify({coverage:unsaved?'Recomputed on save':loaded.coverage,omitted_stages:candidate.omitted_stages,invalid_intervals:candidate.invalid_intervals},null,2);}
async function loadEpisode(revision=''){if(loading)return;loading=true;stop();loaded=null;videos=[];$('videos').replaceChildren();for(const id of ['timeline','segments','boundaries'])$(id).replaceChildren();$('coverage').textContent='';$('identity').textContent='';$('candidate').value='';$('review-state').textContent='Loading candidate…';$('download').removeAttribute('href');for(const id of ['episode','revision','save'])$(id).disabled=true;$('message').textContent='';try{loaded=await get(episodeURL('/api/episode',revision));$('revision').replaceChildren(new Option('Original machine candidate',''));for(const name of loaded.revisions)$('revision').add(new Option(name,name));$('revision').value=revision;const r=loaded.review;$('review-state').textContent=r?`${r.kind==='human'?'Human':'Agent'} review record: ${r.decision} · ${r.reviewer}${r.kind==='agent'?' · Human acceptance remains pending.':''}`:'Original machine candidate · Human acceptance pending.';$('download').href=episodeURL('/api/candidate',revision);$('registry').textContent=JSON.stringify(manifest.stages,null,2);$('identity').textContent=`Episode ${loaded.episode.episode_index} · ${loaded.episode.frame_count} frames\nCameras: ${Object.keys(loaded.episode.video_urls).join(', ')}\nSource time: ${loaded.episode.time_edges[0]} → ${loaded.episode.time_edges.at(-1)} s`;$('candidate').value=JSON.stringify(loaded.candidate,null,2);$('kind').value='';$('decision').value='';$('evidence').value='';$('notes').value='';$('videos').replaceChildren();videos=[];for(const [view,url]of Object.entries(loaded.episode.video_urls)){const figure=document.createElement('figure'),caption=document.createElement('figcaption'),video=document.createElement('video');caption.textContent=view;video.src=url;video.preload='metadata';video.muted=true;video.playsInline=true;const offset=loaded.episode.clock_receipt.cameras[view].source_minus_video_offset_seconds;const episodeAtLoad=loaded.episode;video.addEventListener('loadedmetadata',()=>{if(loaded?.episode.episode_index===episodeAtLoad.episode_index)video.currentTime=Math.max(0,episodeAtLoad.time_edges[frame]-offset);});video.addEventListener('ended',stop);video.addEventListener('error',()=>{$('message').textContent=`Video ${view} could not be decoded; inspect source before accepting.`;stop();});figure.append(caption,video);$('videos').append(figure);videos.push({video,offset});}$('frame').max=loaded.episode.frame_count-1;$('scrub').max=loaded.episode.frame_count-1;setFrame(0);render(loaded.candidate);}catch(error){$('review-state').textContent='No candidate loaded; saving is disabled.';$('message').textContent=error.message;}finally{loading=false;for(const id of ['episode','revision'])$(id).disabled=false;$('save').disabled=!loaded;}}
$('episode').onchange=()=>loadEpisode();$('revision').onchange=()=>loadEpisode($('revision').value);$('pause').onclick=stop;$('play').onclick=async()=>{try{await Promise.all(videos.map(item=>item.video.play()));}catch(error){stop();$('message').textContent=error.message;}};$('previous').onclick=()=>setFrame(frame-1);$('next').onclick=()=>setFrame(frame+1);$('seek').onclick=()=>setFrame($('frame').value);$('scrub').oninput=()=>setFrame($('scrub').value);
$('preview').onclick=()=>{try{render(JSON.parse($('candidate').value),true);$('message').textContent='Preview only. Saving performs strict server validation.';}catch(error){$('message').textContent='Cannot preview JSON: '+error.message;}};
$('review-form').onsubmit=async event=>{event.preventDefault();if(!loaded||loading)return;if(Number($('episode').value)!==loaded.episode.episode_index){$('message').textContent='Episode selection differs from loaded evidence; reload before saving.';return;}$('save').disabled=true;try{const payload={episode_index:loaded.episode.episode_index,candidate:JSON.parse($('candidate').value),review:{reviewer:$('reviewer').value,kind:$('kind').value,decision:$('decision').value,notes:$('notes').value,evidence_inspected:$('evidence').value.split('\n').map(x=>x.trim()).filter(Boolean)}};const result=await get('/api/revision',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});await loadEpisode(result.revision);$('message').textContent=`Saved immutable revision ${result.relative_path}. Original candidate unchanged.`;}catch(error){$('message').textContent='Not saved: '+error.message;}finally{$('save').disabled=!loaded;}};
(async()=>{try{manifest=await get('/api/manifest');$('task').textContent=manifest.task;for(const episode of manifest.episodes)$('episode').add(new Option(`Episode ${episode.episode_index} · ${episode.frame_count} frames`,episode.episode_index));await loadEpisode();requestAnimationFrame(tick);}catch(error){$('message').textContent=error.message;}})();
</script></html>'''


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8788)
    args = parser.parse_args(argv)
    serve_review(args.run_dir, args.host, args.port)


if __name__ == "__main__":
    main()
