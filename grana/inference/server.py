# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Loopback HTTP inference and explicitly allowlisted review media."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import ipaddress
import json
import mimetypes
import os
from pathlib import Path
import re
import stat
import threading
import time
from urllib.parse import urlsplit

import numpy as np

from .client import (BINARY_CONTENT_TYPE, BINARY_MAGIC, BINARY_PREAMBLE, BINARY_VERSION,
                     MAX_BINARY_BODY_BYTES, MAX_BINARY_HEADER_BYTES, MAX_IMAGE_DIMENSION,
                     MAX_RGB_BYTES, SCHEMA, WIRE_BINARY, WIRE_FORMATS, _request_id, _roles,
                     json_loads)
from .predictor import GranaPredictor, InferenceRequest, VIEW_POLICIES, _validate_view_names
from .selection import _UNSET, PREDICTION_FRAME_SLOTS, resolve_prediction_slot, validate_frame_indices
from grana.modeling.inputs import PixelBudgetError


BINARY_BODY_LIMIT = "Binary body limit is a 1 MiB header plus 48 MiB of RGB"
RGB_BUDGET = "RGB pixel bytes exceed the 48 MiB per-request budget"


def split_binary(body: bytes) -> tuple[dict, memoryview]:
    """Split a grana-binary-v1 body into its JSON header and raw RGB section (no pixel copy)."""
    if len(body) > MAX_BINARY_BODY_BYTES:
        raise ValueError(BINARY_BODY_LIMIT)
    if len(body) < BINARY_PREAMBLE.size:
        raise ValueError("Binary request is truncated before its preamble")
    magic, version, length = BINARY_PREAMBLE.unpack_from(body)
    if magic != BINARY_MAGIC or version != BINARY_VERSION:
        raise ValueError(f"Binary request must start with magic {BINARY_MAGIC!r} and version {BINARY_VERSION}")
    if not 2 <= length <= MAX_BINARY_HEADER_BYTES:
        raise ValueError(f"Binary header length must be 2..{MAX_BINARY_HEADER_BYTES} bytes")
    end = BINARY_PREAMBLE.size + length
    if end > len(body):
        raise ValueError("Binary request is truncated inside its header")
    data = memoryview(body)
    return json_loads(str(data[BINARY_PREAMBLE.size:end], "utf-8")), data[end:]


def decode_binary(body: bytes, view_names: tuple[str, ...], *,
                  view_policy: str = "checkpoint") -> tuple[InferenceRequest, list[str], str]:
    """Decode a grana-binary-v1 request with zero-copy RGB views."""
    header, raw = split_binary(body)
    return decode_payload(header, view_names, view_policy=view_policy, raw=raw)


def _binary_view(name: str, item: dict, expected_bytes: int, raw: memoryview, cursor: int) -> np.ndarray:
    offset, nbytes = item["offset"], item["nbytes"]
    if type(offset) is not int or type(nbytes) is not int:
        raise ValueError(f"{name}: offset and nbytes must be integers")
    if nbytes != expected_bytes:
        raise ValueError(f"{name}: nbytes does not match shape")
    if offset < cursor:
        raise ValueError(f"{name}: RGB bytes overlap an earlier view")
    if offset > cursor:
        raise ValueError(f"{name}: RGB bytes must follow the previous view without a gap")
    if offset + nbytes > len(raw):
        raise ValueError(f"{name}: RGB bytes extend past the end of the request")
    return np.frombuffer(raw, dtype=np.uint8, count=nbytes, offset=offset)


def decode_payload(payload: dict, view_names: tuple[str, ...], *,
                   view_policy: str = "checkpoint", raw: memoryview) -> tuple[InferenceRequest, list[str], str]:
    """Validate the binary header and raw RGB section before any model call."""
    if not isinstance(raw, memoryview):
        raise TypeError("Binary payload requires a raw RGB memoryview")
    allowed = {"schema", "request_id", "roles", "views", "task", "compressed_task_prompt", "stage_prompts",
               "focus_stage", "poi_slot", "prediction_frame", "frame_indices"}
    if not isinstance(payload, dict) or set(payload) - allowed:
        raise ValueError("Request contains unknown fields; labels, paths and ground truth are not accepted")
    if payload.get("schema") != SCHEMA:
        raise ValueError(f"schema must be {SCHEMA!r}")
    request_id = _request_id(payload.get("request_id"))
    if not isinstance(payload.get("roles"), list):
        raise ValueError("roles must be a JSON list")
    roles = _roles(payload["roles"])
    if "prediction_frame" in payload and not isinstance(payload["prediction_frame"], str):
        raise ValueError("prediction_frame must be 'poi' or 'last'")
    poi_slot = resolve_prediction_slot(payload.get("prediction_frame"), payload.get("poi_slot", _UNSET))
    frame_indices = validate_frame_indices(payload.get("frame_indices"))
    views = payload.get("views")
    if view_policy not in VIEW_POLICIES:
        raise ValueError(f"view_policy must be one of {VIEW_POLICIES}")
    if not isinstance(views, dict) or not views:
        raise ValueError("views must be a nonempty ordered mapping of named cameras")
    _validate_view_names(tuple(views))
    if view_policy == "checkpoint" and tuple(views) != view_names:
        raise ValueError(f"views names and order must match the checkpoint: {list(view_names)}")
    images = {}
    raw_bytes = 0
    fields, supply = {"shape", "offset", "nbytes"}, "shape, offset and nbytes"
    for name, item in views.items():
        if not isinstance(item, dict) or set(item) != fields:
            raise ValueError(f"{name}: supply only {supply}")
        shape = item["shape"]
        if (not isinstance(shape, list) or len(shape) != 4 or any(type(size) is not int for size in shape)
                or shape[0] != 8 or shape[3] != 3
                or not all(1 <= size <= MAX_IMAGE_DIMENSION for size in shape[1:3])):
            raise ValueError(f"{name}: shape must be [8, height, width, 3], dimensions 1..{MAX_IMAGE_DIMENSION}")
        expected_bytes = 8 * shape[1] * shape[2] * 3
        if raw_bytes + expected_bytes > MAX_RGB_BYTES:
            raise ValueError(f"{name}: {RGB_BUDGET}")
        array = _binary_view(name, item, expected_bytes, raw, raw_bytes).reshape(shape)
        raw_bytes += expected_bytes
        images[name] = tuple(array[slot] for slot in range(8))
    if raw_bytes != len(raw):
        raise ValueError("Binary RGB section must be exactly covered by the views, without trailing bytes")
    for field in ("task", "compressed_task_prompt"):
        if field in payload:
            value = payload[field]
            if not isinstance(value, str) or not value.strip() or "<|" in value or "|>" in value:
                raise ValueError(f"{field} must be nonempty text without reserved markers")
    stages = payload.get("stage_prompts", [])
    if (not isinstance(stages, list) or len(stages) > 26
            or any(not isinstance(item, str) or not item.strip() or "<|" in item or "|>" in item for item in stages)):
        raise ValueError("stage_prompts must contain at most 26 nonempty descriptions without reserved markers")
    focus = payload.get("focus_stage")
    if "focus_stage" in payload and (type(focus) is not int or not 0 <= focus < len(stages)):
        raise ValueError("focus_stage must be an integer in the supplied stage range")
    if "A" in roles and "compressed_task_prompt" not in payload:
        raise ValueError("Role A requires compressed_task_prompt")
    if any(role in roles for role in ("B", "C", "D")) and ("task" not in payload or not stages):
        raise ValueError("Roles B/C/D require task and 1..26 stage_prompts")
    if "B" in roles and focus is None:
        raise ValueError("Role B requires focus_stage")
    request = InferenceRequest(images, payload.get("task"), payload.get("compressed_task_prompt"), stages, focus,
                               poi_slot, prediction_frame=payload.get("prediction_frame"), frame_indices=frame_indices)
    return request, roles, request_id


def _regular_path(path: Path) -> Path:
    if not path.is_absolute() or any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError("Media allowlist paths must be absolute and contain no symlinks")
    if not path.is_file():
        raise ValueError(f"Media allowlist entry is not a regular file: {path}")
    return path


def _media_files(manifest: dict) -> dict[str, Path]:
    if not isinstance(manifest, dict):
        raise ValueError("Media manifest must map exact URL paths to absolute files")
    files = {}
    for url, filename in manifest.items():
        if (not isinstance(url, str) or not url.startswith("/") or url.startswith("//")
                or any(segment in (".", "..") for segment in url.split("/"))
                or any(char in url for char in ("?", "#", "\\", "%"))
                or url in ("/healthz", "/api/predict") or not isinstance(filename, str)):
            raise ValueError("Media manifest keys must be literal absolute URL paths without traversal or reserved endpoints")
        files[url] = _regular_path(Path(filename))
    return files


def _byte_range(value: str | None, size: int) -> tuple[int, int, int]:
    if value is None:
        return 0, size - 1, 200
    match = re.fullmatch(r"bytes=(\d*)-(\d*)", value)
    if not match or size == 0 or not any(match.groups()):
        raise ValueError("Unsatisfiable single byte range")
    first, last = match.groups()
    if first:
        start = int(first)
        end = min(int(last), size - 1) if last else size - 1
        if start >= size or end < start:
            raise ValueError("Unsatisfiable single byte range")
    else:
        suffix = int(last)
        if suffix == 0:
            raise ValueError("Unsatisfiable single byte range")
        start, end = max(0, size - suffix), size - 1
    return start, end, 206


class InferenceServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, predictor, media_manifest, audit_log):
        self.predictor = predictor
        self.media_files = _media_files(media_manifest)
        self.prediction_lock = threading.Lock()
        self._prediction_executor = None
        self.audit_lock = threading.Lock()
        self.audit_path = Path(audit_log).expanduser().absolute() if audit_log is not None else None
        if self.audit_path:
            self.audit_path.parent.mkdir(parents=True, exist_ok=True)
            with self.audit_path.open("a"):
                pass
        super().__init__(address, Handler)
        # HTTP connections have short-lived handler threads. Keep model calls
        # on one thread so warmed CUDA handles and cuDNN execution-plan caches
        # survive between requests. A lock alone serializes work but loses
        # these thread-local caches on every new connection.
        self._prediction_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="grana-inference")

    def server_close(self):
        try:
            super().server_close()
        finally:
            if self._prediction_executor is not None:
                self._prediction_executor.shutdown(wait=True)

    def audit(self, event):
        if self.audit_path:
            entry = {"time_utc": datetime.now(timezone.utc).isoformat(),
                     "checkpoint": {key: self.predictor.checkpoint.get(key)
                                    for key in ("weights_sha256", "source_step", "source_stage")}, **event}
            with self.audit_lock, self.audit_path.open("a") as stream:
                stream.write(json.dumps(entry, allow_nan=False, separators=(",", ":")) + "\n")


class Handler(BaseHTTPRequestHandler):
    server_version = "GranaHTTP/1"
    sys_version = ""

    def log_message(self, format, *args):
        # Requests are recorded without bodies or prompts in the JSONL audit.
        pass

    def _json(self, status, value, extra_headers=None):
        body = json.dumps(value, allow_nan=False, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-store")
        for key, value in (extra_headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _error(self, status, code, message, request_id=None, extra_headers=None):
        self._json(status, {"schema": SCHEMA, "request_id": request_id,
                            "error": {"code": code, "message": message}}, extra_headers)

    def do_HEAD(self):
        self.do_GET()

    def _method_not_allowed(self):
        self._error(405, "method_not_allowed", "Use GET/HEAD for media and POST for prediction")

    do_PUT = do_DELETE = do_PATCH = do_OPTIONS = _method_not_allowed

    def do_GET(self):
        if self.path == "/healthz":
            checkpoint = self.server.predictor.checkpoint
            self._json(200, {"schema": SCHEMA, "ready": True, "checkpoint": checkpoint,
                             "views": list(self.server.predictor.view_names), "poi_slot": 5,
                             "poi_slot_policy": "request_selected; API and CLI default 5; integer 0..7",
                             "prediction_frames": PREDICTION_FRAME_SLOTS,
                             "view_policy": getattr(self.server.predictor, "view_policy", "checkpoint"),
                             "checkpoint_view_names": list(self.server.predictor.view_names),
                             "image_preprocessing": getattr(self.server.predictor, "image_preprocessing", None),
                             "loaded_heads": checkpoint.get("load_report", {}).get("heads", {}),
                             "wire_formats": list(WIRE_FORMATS)})
            return
        filename = self.server.media_files.get(self.path)
        if filename is None:
            self._error(404, "not_found", "URL is not in the media allowlist")
            return
        try:
            _regular_path(filename)
            descriptor = os.open(filename, os.O_RDONLY | os.O_NOFOLLOW)
            with os.fdopen(descriptor, "rb") as stream:
                info = os.fstat(stream.fileno())
                if not stat.S_ISREG(info.st_mode):
                    raise ValueError("Media is not a regular file")
                try:
                    if len(self.headers.get_all("Range", [])) > 1:
                        raise ValueError("Only one Range header is supported")
                    start, end, status = _byte_range(self.headers.get("Range"), info.st_size)
                except ValueError:
                    self._error(416, "invalid_range", "Only one satisfiable byte range is supported",
                                extra_headers={"Content-Range": f"bytes */{info.st_size}", "Accept-Ranges": "bytes"})
                    return
                self.send_response(status)
                mime = {".js": "text/javascript", ".mp4": "video/mp4", ".json": "application/json"}.get(
                    filename.suffix.lower(), mimetypes.guess_type(str(filename))[0] or "application/octet-stream")
                self.send_header("Content-Type", mime)
                self.send_header("Content-Length", str(max(0, end - start + 1)))
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("Accept-Ranges", "bytes")
                if status == 206:
                    self.send_header("Content-Range", f"bytes {start}-{end}/{info.st_size}")
                self.end_headers()
                if self.command == "HEAD":
                    return
                stream.seek(start)
                remaining = end - start + 1
                while remaining > 0:
                    chunk = stream.read(min(64 * 1024, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            return
        except (FileNotFoundError, PermissionError, ValueError, OSError):
            # A removed/replaced allowlisted file is never resolved elsewhere.
            self._error(404, "not_found", "Allowlisted file is unavailable")

    def do_POST(self):
        started = time.monotonic()
        request_id = None
        roles = None
        shapes = None
        status = 500
        predicting = False
        wire_format = None
        try:
            if self.path != "/api/predict":
                status = 404
                self._error(status, "not_found", "Unknown prediction endpoint")
                return
            origin = self.headers.get("Origin")
            if origin is not None:
                parsed = urlsplit(origin)
                if parsed.scheme != "http" or parsed.netloc != self.headers.get("Host") or parsed.path or parsed.query or parsed.fragment:
                    status = 403
                    self._error(status, "invalid_origin", "Origin must match this server's origin")
                    return
            if self.headers.get_content_type() != BINARY_CONTENT_TYPE:
                status = 415
                self._error(status, "unsupported_media_type",
                            f"Content-Type must be {BINARY_CONTENT_TYPE}")
                return
            wire_format = WIRE_BINARY
            lengths = self.headers.get_all("Content-Length", [])
            if self.headers.get("Transfer-Encoding") or len(lengths) != 1 or not re.fullmatch(r"\d+", lengths[0]):
                status = 400
                self._error(status, "invalid_length", "Supply one valid Content-Length; transfer encoding is unsupported")
                return
            length = int(lengths[0])
            if length > MAX_BINARY_BODY_BYTES:
                status = 413
                self._error(status, "body_too_large", BINARY_BODY_LIMIT)
                return
            self.connection.settimeout(30)
            body = self.rfile.read(length)
            if len(body) != length:
                raise ValueError("Request body length differs from Content-Length")
            payload, raw = split_binary(body)
            if isinstance(payload, dict):
                try:
                    request_id = _request_id(payload.get("request_id"))
                except ValueError:
                    pass
            request, roles, request_id = decode_payload(payload, tuple(self.server.predictor.view_names),
                view_policy=getattr(self.server.predictor, "view_policy", "checkpoint"), raw=raw)
            shapes = {name: item["shape"] for name, item in payload["views"].items()}
            predicting = True
            with self.server.prediction_lock:
                result = self.server._prediction_executor.submit(
                    self.server.predictor.predict, request, roles=roles).result()
            elapsed = time.monotonic() - started
            response = {"schema": SCHEMA, "request_id": request_id, "elapsed_seconds": elapsed, "result": result}
            # Validate finite output before writing response headers.
            json.dumps(response, allow_nan=False)
            status = 200
            self._json(status, response)
        except PixelBudgetError as error:
            status = 400
            self._error(status, "image_pixel_budget_exceeded", str(error), request_id)
        except (ValueError, TypeError, UnicodeError, TimeoutError) as error:
            status = 500 if predicting else 400
            self._error(status, "prediction_failed" if predicting else "invalid_request",
                        "Prediction failed; inspect the server environment and checkpoint" if predicting else str(error), request_id)
        except Exception:
            status = 500
            self._error(status, "prediction_failed", "Prediction failed; inspect the server environment and checkpoint", request_id)
        finally:
            self.server.audit({"request_id": request_id, "status": status, "roles": roles, "wire_format": wire_format,
                               "view_policy": getattr(self.server.predictor, "view_policy", "checkpoint"),
                               "image_preprocessing": getattr(self.server.predictor, "image_preprocessing", None),
                               "view_shapes": shapes, "elapsed_seconds": time.monotonic() - started})


def create_server(predictor, host: str = "127.0.0.1", port: int = 0,
                  media_manifest: dict | None = None, audit_log: str | Path | None = None) -> InferenceServer:
    """Bind only after a caller has successfully constructed its predictor."""
    if not ipaddress.ip_address(host).is_loopback or ":" in host:
        raise ValueError("The review server must bind an IPv4 loopback address")
    if getattr(predictor, "view_policy", "checkpoint") not in VIEW_POLICIES:
        raise ValueError(f"view_policy must be one of {VIEW_POLICIES}")
    return InferenceServer((host, port), predictor, media_manifest or {}, audit_log)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--media-manifest", type=Path, help="Exact URL-to-absolute-file JSON allowlist")
    parser.add_argument("--audit-log", required=True, type=Path, help="Append metadata-only prediction events as JSONL")
    parser.add_argument("--model-name-or-path")
    parser.add_argument("--device", help="CUDA device, e.g. cuda:0; CPU inference is unsupported")
    parser.add_argument("--max-pixels", type=int,
                        help="Enforce a per-image pixel budget (>=1024); omitted inherits checkpoint preprocessing")
    parser.add_argument("--view-policy", choices=VIEW_POLICIES, default="checkpoint",
                        help="checkpoint requires saved camera names/order; request accepts supplied named views")
    args = parser.parse_args(argv)
    manifest = json_loads(args.media_manifest.read_bytes()) if args.media_manifest else {}
    _media_files(manifest)
    predictor = GranaPredictor(args.checkpoint, model_name_or_path=args.model_name_or_path,
                               device=args.device, max_pixels=args.max_pixels, view_policy=args.view_policy)
    server = create_server(predictor, args.host, args.port, manifest, args.audit_log)
    print(json.dumps({"ready": True, "url": f"http://{args.host}:{server.server_port}",
                      "view_policy": predictor.view_policy, "checkpoint_view_names": list(predictor.view_names),
                      "image_preprocessing": predictor.image_preprocessing,
                      "checkpoint": predictor.checkpoint}, allow_nan=False), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
