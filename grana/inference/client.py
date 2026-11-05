# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lossless RGB transport to a running Grana inference server."""
from __future__ import annotations

import argparse
from collections.abc import Mapping
from functools import lru_cache
from http.client import HTTPConnection, HTTPSConnection
import json
from pathlib import Path
import struct
import time
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import HTTPHandler, HTTPSHandler, Request, build_opener, urlopen
import uuid

import numpy as np
from PIL import Image
from .selection import (ROLES_HELP, _UNSET, add_prediction_arguments, prediction_frame_for_slot, resolve_prediction_slot,
                        validate_frame_indices, validate_slot as _poi_slot)

SCHEMA = "grana.inference.v1"
MAX_RESPONSE_BYTES = 64 * 1024 * 1024
MAX_IMAGE_DIMENSION = 4096
# Requests use one format: a JSON header followed by raw RGB bytes.
WIRE_BINARY = "grana-binary-v1"
WIRE_FORMATS = (WIRE_BINARY,)
BINARY_CONTENT_TYPE = "application/vnd.grana.binary-v1"
# Binary preamble: 4-byte magic, uint32 LE version, uint32 LE header length.
BINARY_MAGIC, BINARY_VERSION = b"GRNB", 1
BINARY_PREAMBLE = struct.Struct("<4sII")
MAX_BINARY_HEADER_BYTES = 1024 * 1024
# Raw RGB budget per request; independent of the response and header limits.
MAX_RGB_BYTES = 48 * 1024 * 1024
MAX_BINARY_BODY_BYTES = BINARY_PREAMBLE.size + MAX_BINARY_HEADER_BYTES + MAX_RGB_BYTES


class BinaryHeaderTooLarge(ValueError):
    """The grana-binary-v1 JSON header exceeds MAX_BINARY_HEADER_BYTES."""


def json_loads(data: bytes | str):
    """Preserve view order and reject duplicate keys and nonfinite JSON numbers."""
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def constant(value):
        raise ValueError(f"Nonfinite JSON number: {value}")

    return json.loads(data, object_pairs_hook=pairs, parse_constant=constant)


def _roles(roles) -> list[str]:
    if isinstance(roles, str):
        roles = list(roles)
    if not isinstance(roles, (list, tuple)) or not roles or any(role not in ("A", "B", "C", "D") for role in roles):
        raise ValueError("roles must be a nonempty selection from A, B, C, D")
    if len(set(roles)) != len(roles):
        raise ValueError("roles must be unique")
    return list(roles)


def _request_id(value) -> str:
    if not isinstance(value, str) or not 1 <= len(value) <= 128 or not value.isprintable() or not value.strip():
        raise ValueError("request_id must contain 1..128 printable characters")
    return value


def _view_arrays(name, frames) -> list[np.ndarray]:
    if not isinstance(name, str) or not name.strip() or len(name) > 128:
        raise ValueError("View names must be nonempty text of at most 128 characters")
    if not isinstance(frames, (list, tuple, np.ndarray)) or len(frames) != 8:
        raise ValueError(f"{name}: exactly eight RGB frames are required")
    arrays = []
    for frame in frames:
        if isinstance(frame, Image.Image):
            if frame.mode != "RGB":
                raise ValueError(f"{name}: PIL frames must have RGB mode")
            frame = np.asarray(frame)
        if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[-1] != 3:
            raise ValueError(f"{name}: frames must be uint8 [height, width, 3] RGB")
        if not all(1 <= size <= MAX_IMAGE_DIMENSION for size in frame.shape[:2]):
            raise ValueError(f"{name}: image dimensions must be 1..{MAX_IMAGE_DIMENSION}")
        arrays.append(frame)
    if any(array.shape != arrays[0].shape for array in arrays):
        raise ValueError(f"{name}: all eight frames must have the same shape")
    return arrays


def _header(request, roles, request_id: str | None, prediction_frame: str | None = None) -> dict:
    if not isinstance(request.images, Mapping) or not request.images:
        raise ValueError("images must be an ordered mapping of camera names to eight RGB frames")
    payload = {"schema": SCHEMA, "request_id": _request_id(request_id if request_id is not None else uuid.uuid4().hex),
               "roles": _roles(roles), "views": {}}
    if prediction_frame is not None:
        slot = resolve_prediction_slot(prediction_frame)
    else:
        prediction_frame = getattr(request, "prediction_frame", None)
        slot = resolve_prediction_slot(prediction_frame, getattr(request, "poi_slot", _UNSET))
    if slot != 5:
        payload["poi_slot"] = slot
    mode = prediction_frame_for_slot(slot)
    if mode != "custom":
        payload["prediction_frame"] = mode
    indices = validate_frame_indices(getattr(request, "frame_indices", None))
    if indices is not None:
        payload["frame_indices"] = list(indices)
    return payload


def _text_fields(payload: dict, request) -> dict:
    for field in ("task", "compressed_task_prompt", "focus_stage"):
        value = getattr(request, field)
        if value is not None:
            payload[field] = value
    if request.stage_prompts:
        payload["stage_prompts"] = list(request.stage_prompts)
    return payload


def build_binary(request, roles=("C", "D"), request_id: str | None = None, *,
                 prediction_frame: str | None = None) -> bytes:
    """Encode grana-binary-v1: preamble, UTF-8 JSON header, then raw RGB of every view in order."""
    header = _header(request, roles, request_id, prediction_frame)
    chunks, offset = [], 0
    for name, frames in request.images.items():
        arrays = [np.ascontiguousarray(array) for array in _view_arrays(name, frames)]
        nbytes = sum(array.nbytes for array in arrays)
        if offset + nbytes > MAX_RGB_BYTES:
            raise ValueError("RGB payload exceeds the 48 MiB per-request pixel budget")
        header["views"][name] = {"shape": [8, *arrays[0].shape], "offset": offset, "nbytes": nbytes}
        chunks.extend(arrays)
        offset += nbytes
    encoded = json.dumps(_text_fields(header, request), allow_nan=False, separators=(",", ":")).encode("utf-8")
    if len(encoded) > MAX_BINARY_HEADER_BYTES:
        raise BinaryHeaderTooLarge("Binary request header exceeds 1 MiB")
    return b"".join([BINARY_PREAMBLE.pack(BINARY_MAGIC, BINARY_VERSION, len(encoded)), encoded, *chunks])


def encode_request(request, roles=("C", "D"), request_id: str | None = None, *,
                   prediction_frame: str | None = None) -> tuple[bytes, str]:
    """Return the binary body and content type for POST /api/predict."""
    return build_binary(request, roles, request_id, prediction_frame=prediction_frame), BINARY_CONTENT_TYPE


class PredictionError(RuntimeError):
    """A structured error returned by the inference server."""

    def __init__(self, status: int, response: dict):
        self.status, self.response = status, response
        super().__init__(f"HTTP {status}: {response.get('error', {}).get('message', 'inference request failed')}")


def _base_url(server: str) -> str:
    parsed = urlsplit(server)
    if parsed.scheme not in ("http", "https") or not parsed.netloc or parsed.query or parsed.fragment or parsed.username:
        raise ValueError("server must be an HTTP(S) base URL without credentials, query or fragment")
    return server.rstrip("/")


class _SendTimedSocket:
    """Observe actual request writes on an already connected socket."""

    def __init__(self, sock, clock):
        self._sock, self._clock = sock, clock

    def __getattr__(self, name):
        return getattr(self._sock, name)

    def sendall(self, data, *args, **kwargs):
        if "started_ns" not in self._clock:
            self._clock["started_ns"] = time.perf_counter_ns()
        result = self._sock.sendall(data, *args, **kwargs)
        self._clock["sent_ns"] = time.perf_counter_ns()
        return result


@lru_cache(maxsize=1)
def _timing_https_context():
    # Match urlopen's reusable default TLS context rather than reloading trust
    # stores for every measured request.
    return HTTPSHandler()._context


def _open_with_send_timing(wire, timeout, clock):
    # Per-request handlers preserve urllib transport/error behavior without
    # globally patching sockets. Wrap only after connect (including TLS and
    # any proxy tunnel) so those setup operations cannot start the timer.
    def connection(base):
        class TimedConnection(base):
            def connect(self):
                super().connect()
                self.sock = _SendTimedSocket(self.sock, clock)
        return TimedConnection

    class TimedHTTPHandler(HTTPHandler):
        def http_open(self, req):
            return self.do_open(connection(HTTPConnection), req)

    class TimedHTTPSHandler(HTTPSHandler):
        def https_open(self, req):
            return self.do_open(connection(HTTPSConnection), req, context=self._context)

    opener = build_opener(TimedHTTPHandler(), TimedHTTPSHandler(context=_timing_https_context()))
    clock["http_started_ns"] = time.perf_counter_ns()
    return opener.open(wire, timeout=timeout)


def _clear_timing(timing):
    if timing is not None:
        for key in ("http_round_trip_ms", "client_send_to_receive_ms", "client_send_ms",
                    "client_send_complete_to_receive_ms"):
            timing.pop(key, None)


def post_encoded(server: str, body: bytes, content_type: str, request_id: str, timeout: float = 300, *,
                 timing: dict[str, float] | None = None) -> dict:
    """POST encoded RGB; optionally time send through full receipt before JSON parsing.

    The interval starts at the first request socket.sendall entry, after
    connection setup, and ends when response.read returns. Request headers
    and the complete body are sent inside the interval. Packing, connection
    setup and response JSON parsing are excluded. Send completion means the
    final sendall returned, not an acknowledgement from the server.

    ``http_round_trip_ms`` separately retains the HTTP-open-through-read
    interval, including connection setup. Owned timing fields are cleared
    first and filled only after a successful complete HTTP body read, before
    parsing. Other caller-owned dictionary entries and the envelope are unchanged.
    """
    _clear_timing(timing)
    if content_type != BINARY_CONTENT_TYPE:
        raise ValueError(f"Inference requests require {BINARY_CONTENT_TYPE}")
    wire = Request(_base_url(server) + "/api/predict", data=body,
                   headers={"Content-Type": content_type, "Accept": "application/json"}, method="POST")
    clock = {}
    try:
        response = (urlopen(wire, timeout=timeout) if timing is None else
                    _open_with_send_timing(wire, timeout, clock))
        with response:
            raw = response.read(MAX_RESPONSE_BYTES + 1)
            received = time.perf_counter_ns() if timing is not None else None
            if len(raw) > MAX_RESPONSE_BYTES:
                raise ValueError("Server response exceeds the response byte limit")
            if timing is not None:
                timing.update(http_round_trip_ms=(received - clock["http_started_ns"]) / 1_000_000,
                              client_send_to_receive_ms=(received - clock["started_ns"]) / 1_000_000,
                              client_send_ms=(clock["sent_ns"] - clock["started_ns"]) / 1_000_000,
                              client_send_complete_to_receive_ms=(received - clock["sent_ns"]) / 1_000_000)
            result = json_loads(raw)
    except HTTPError as error:
        try:
            result = json_loads(error.read(MAX_RESPONSE_BYTES + 1))
        except (ValueError, UnicodeError):
            result = {"error": {"code": "invalid_error_response", "message": "Server returned a non-JSON error"}}
        if not isinstance(result, dict):
            result = {"error": {"code": "invalid_error_response", "message": "Server returned a non-object error"}}
        raise PredictionError(error.code, result) from error
    if not isinstance(result, dict) or result.get("schema") != SCHEMA or result.get("request_id") != request_id:
        raise ValueError("Server response schema or request_id does not match the request")
    if not isinstance(result.get("result"), dict):
        raise ValueError("Server response does not contain a prediction result")
    return result


def predict(server: str, request, roles=("C", "D"), request_id: str | None = None,
            timeout: float = 300, *, prediction_frame: str | None = None,
            timing: dict[str, float] | None = None) -> dict:
    """Return the server envelope using binary RGB requests, without negotiation or retries.

    Optional ``timing`` records HTTP and socket-send intervals separately
    from the prediction; see ``post_encoded``. Binary packing is excluded.
    """
    _clear_timing(timing)
    base = _base_url(server)
    request_id = _request_id(request_id if request_id is not None else uuid.uuid4().hex)
    body, content_type = encode_request(request, roles, request_id, prediction_frame=prediction_frame)
    return post_encoded(base, body, content_type, request_id, timeout, timing=timing)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", required=True, help="Running server base URL, e.g. http://127.0.0.1:8787")
    parser.add_argument("--input", type=Path, required=True, help="Existing label-free inference request JSON")
    parser.add_argument("--roles", nargs="+", choices=list("ABCD"), default=["C", "D"], help=ROLES_HELP)
    parser.add_argument("--output", type=Path, required=True, help="New JSON output file")
    parser.add_argument("--plot", type=Path, help="Write a new PNG labeling the requested roles and selected frame")
    parser.add_argument("--request-id")
    add_prediction_arguments(parser)
    parser.add_argument("--frame-index", type=int,
                        help="Compose the standard eight-frame window at this source frame from .npy camera arrays")
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args(argv)
    output = args.output.expanduser()
    if output.exists():
        raise FileExistsError(f"Use a fresh client output: {output}")
    from .presentation import render_prediction, validate_plot_path
    plot = validate_plot_path(args.plot, output)
    from .cli import load_request
    request = load_request(args.input, prediction_frame=args.prediction_frame, poi_slot=args.poi_slot,
                           frame_index=args.frame_index)
    result = predict(args.server, request, roles=args.roles,
                     request_id=args.request_id, timeout=args.timeout)
    if plot is not None:
        render_prediction(result["result"], request, plot)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
