# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Binary RGB is the only inference request transport; errors never trigger fallback."""
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import time
from urllib.error import URLError

import numpy as np
from PIL import Image
import pytest

from grana.inference import client
from grana.inference.client import (BINARY_CONTENT_TYPE, BINARY_MAGIC, BINARY_PREAMBLE,
                                    MAX_BINARY_BODY_BYTES, MAX_BINARY_HEADER_BYTES, MAX_RGB_BYTES,
                                    SCHEMA, WIRE_BINARY, BinaryHeaderTooLarge, PredictionError,
                                    build_binary, encode_request, predict)
from grana.inference.predictor import InferenceRequest
from grana.inference.server import decode_binary, split_binary
from test_serving import VIEWS, http, observations, running  # noqa: F401 (shared fixture)


def mixed_request():
    request = observations()
    request.images["top"] = tuple(Image.fromarray(item) for item in request.images["top"])
    request.images["left"] = tuple(item[:, ::-1] for item in request.images["left"])  # noncontiguous
    return replace(request, poi_slot=2)


def body(header, raw):
    encoded = json.dumps(header).encode()
    return BINARY_PREAMBLE.pack(BINARY_MAGIC, 1, len(encoded)) + encoded + raw


def parts(request=None, **kwargs):
    header, raw = split_binary(build_binary(request or observations(), request_id="binary", **kwargs))
    return header, bytes(raw)


def assert_same_request(actual, expected):
    for field in ("task", "compressed_task_prompt", "stage_prompts", "focus_stage", "poi_slot"):
        left, right = getattr(actual, field), getattr(expected, field)
        assert (tuple(left) == tuple(right)) if field == "stage_prompts" else left == right
    assert tuple(actual.images) == tuple(expected.images)
    for view in expected.images:
        assert len(actual.images[view]) == len(expected.images[view]) == 8
        for left, right in zip(actual.images[view], expected.images[view], strict=True):
            left, right = np.asarray(left), np.asarray(right)
            assert left.dtype == right.dtype == np.uint8 and left.shape == right.shape
            assert left.tobytes() == right.tobytes()

def test_binary_decodes_the_exact_request_and_readonly_pixels():
    request = mixed_request()
    encoded, content_type = encode_request(request, ["A", "C", "D"], "same")
    decoded, roles, request_id = decode_binary(encoded, VIEWS)
    assert content_type == BINARY_CONTENT_TYPE and roles == ["A", "C", "D"] and request_id == "same"
    assert_same_request(decoded, request)
    frame = decoded.images["top"][0]
    assert not frame.flags.writeable and frame.base is not None


def test_binary_layout_is_preamble_header_then_views_in_order():
    request = observations()
    encoded = build_binary(request, request_id="layout")
    magic, version, length = BINARY_PREAMBLE.unpack_from(encoded)
    header = json.loads(encoded[BINARY_PREAMBLE.size:BINARY_PREAMBLE.size + length])
    assert (magic, version, BINARY_PREAMBLE.size) == (b"GRNB", 1, 12)
    assert header["schema"] == SCHEMA and "poi_slot" not in header
    assert header["roles"] == ["C", "D"]
    assert list(header["views"]) == list(VIEWS)
    assert [item["offset"] for item in header["views"].values()] == [0, 144, 288, 432]
    assert all(item == {"shape": [8, 2, 3, 3], "offset": item["offset"], "nbytes": 144} for item in header["views"].values())
    assert encoded[BINARY_PREAMBLE.size + length:] == b"".join(np.stack(frames).tobytes() for frames in request.images.values())
    assert encode_request(request, request_id="layout") == (encoded, BINARY_CONTENT_TYPE)


@pytest.mark.parametrize("mode,slot", [("poi", 5), ("last", 7)])
def test_binary_preserves_named_selection_source_indices_and_all_pixels(mode, slot):
    indices = (6, 7, 8, 1, 5, 9, 13, 13)
    request = replace(observations(), poi_slot=slot, prediction_frame=mode, frame_indices=indices)
    encoded = build_binary(request, request_id="selection")
    header, raw = split_binary(encoded)
    assert header["prediction_frame"] == mode and header["frame_indices"] == list(indices)
    decoded, roles, _ = decode_binary(encoded, VIEWS)
    assert decoded.poi_slot == slot and decoded.prediction_frame == mode and decoded.frame_indices == indices
    assert roles == ["C", "D"]
    assert_same_request(decoded, request)
    assert bytes(raw) == parts()[1]


def test_binary_named_override_wins_without_mutating_the_python_request():
    request = replace(observations(), prediction_frame="poi")
    encoded, _ = encode_request(request, prediction_frame="last")
    decoded, _, _ = decode_binary(encoded, VIEWS)
    assert decoded.poi_slot == 7 and decoded.prediction_frame == "last"
    assert request.poi_slot == 5 and request.prediction_frame == "poi"


def test_binary_legacy_omission_and_named_only_selection():
    header, raw = parts()
    del header["prediction_frame"]
    assert decode_binary(body(header, raw), VIEWS)[0].poi_slot == 5
    header["prediction_frame"] = "last"
    assert "poi_slot" not in header
    assert decode_binary(body(header, raw), VIEWS)[0].poi_slot == 7


@pytest.mark.parametrize("update,match", [
    ({"prediction_frame": "last", "poi_slot": 5}, "conflicts"),
    ({"prediction_frame": "poi", "poi_slot": 7}, "conflicts"),
    ({"prediction_frame": "custom"}, "prediction_frame"),
    ({"prediction_frame": None}, "prediction_frame"),
    ({"prediction_frame": []}, "prediction_frame"),
    ({"frame_indices": [True] * 8}, "frame_indices"),
    ({"frame_indices": [-1] * 8}, "frame_indices"),
    ({"frame_indices": [0] * 7}, "frame_indices"),
])
def test_binary_rejects_invalid_selection_and_source_indices(update, match):
    header, raw = parts()
    header.update(update)
    with pytest.raises(ValueError, match=match):
        decode_binary(body(header, raw), VIEWS)


def mutate_views(update):
    def apply(header, raw):
        update(header["views"])
        return body(header, raw)
    return apply


@pytest.mark.parametrize("make,match", [
    (lambda h, r: b"GRN", "truncated before its preamble"),
    (lambda h, r: b"XXXX" + body(h, r)[4:], "magic"),
    (lambda h, r: BINARY_PREAMBLE.pack(BINARY_MAGIC, 2, 2) + b"{}", "version"),
    (lambda h, r: BINARY_PREAMBLE.pack(BINARY_MAGIC, 1, 0), "header length"),
    (lambda h, r: BINARY_PREAMBLE.pack(BINARY_MAGIC, 1, MAX_BINARY_HEADER_BYTES + 1) + b"{}", "header length"),
    (lambda h, r: BINARY_PREAMBLE.pack(BINARY_MAGIC, 1, 100) + b"{}", "truncated inside its header"),
    (lambda h, r: BINARY_PREAMBLE.pack(BINARY_MAGIC, 1, 2) + b"\xff\xfe" + r, "utf-8"),
    (lambda h, r: BINARY_PREAMBLE.pack(BINARY_MAGIC, 1, 4) + b"[1]," + r, "Extra data"),
    (lambda h, r: (lambda e: BINARY_PREAMBLE.pack(BINARY_MAGIC, 1, len(e)) + e + r)(
        b'{"schema":"a","schema":"b"}'), "Duplicate JSON key"),
    (lambda h, r: (lambda e: BINARY_PREAMBLE.pack(BINARY_MAGIC, 1, len(e)) + e + r)(
        b'{"focus_stage":NaN}'), "Nonfinite"),
    (lambda h, r: body({**h, "schema": "grana.inference.v0"}, r), "schema"),
    (lambda h, r: body({**h, "gt": 1}, r), "unknown fields"),
    (lambda h, r: body({**h, "views": dict(reversed(list(h["views"].items())))}, r), "order must match"),
    (lambda h, r: body(h, r + b"\0"), "trailing bytes"),
    (lambda h, r: body(h, r[:-1]), "past the end"),
    (mutate_views(lambda v: v["left"].update(offset=100)), "overlap"),
    (mutate_views(lambda v: v["left"].update(offset=0)), "overlap"),
    (mutate_views(lambda v: v["left"].update(offset=145)), "gap"),
    (mutate_views(lambda v: v["top"].update(offset=-1)), "overlap"),
    (mutate_views(lambda v: v["top"].update(offset=1.0)), "integers"),
    (mutate_views(lambda v: v["top"].update(nbytes=True)), "integers"),
    (mutate_views(lambda v: v["top"].update(nbytes=143)), "nbytes does not match"),
    (mutate_views(lambda v: v["top"].update(shape=[8, 4097, 3, 3])), "dimensions"),
    (mutate_views(lambda v: v["top"].update(shape=[8, 0, 3, 3])), "dimensions"),
    (mutate_views(lambda v: v["top"].update(shape=[7, 2, 3, 3])), "dimensions"),
    (mutate_views(lambda v: v["top"].update(rgb_base64="")), "top: supply only shape, offset and nbytes$"),
    (mutate_views(lambda v: v["top"].pop("offset")), "supply only"),
    (lambda h, r: b"\0" * (MAX_BINARY_BODY_BYTES + 1), "Binary body limit is a 1 MiB header plus 48 MiB of RGB"),
    # The pixel budget is checked from the header, before any RGB bytes are touched.
    (mutate_views(lambda v: v["top"].update(shape=[8, 1025, 2048, 3], nbytes=8 * 1025 * 2048 * 3)),
     "top: RGB pixel bytes exceed the 48 MiB per-request budget"),
])
def test_binary_validation_failures(make, match):
    header, raw = parts()
    with pytest.raises((ValueError, UnicodeError), match=match):
        decode_binary(make(header, raw), VIEWS)

def test_binary_respects_checkpoint_and_request_view_policies():
    header, raw = parts()
    header["views"] = {"wrist": header["views"]["top"]}
    with pytest.raises(ValueError, match="order must match"):
        decode_binary(body(header, raw[:144]), VIEWS)
    decoded, _, _ = decode_binary(body(header, raw[:144]), VIEWS, view_policy="request")
    assert tuple(decoded.images) == ("wrist",)
    with pytest.raises(ValueError, match="view_policy"):
        decode_binary(body(header, raw[:144]), VIEWS, view_policy="automatic")
    header["views"] = {" wrist": header["views"]["wrist"]}
    with pytest.raises(ValueError, match="whitespace"):
        decode_binary(body(header, raw[:144]), VIEWS, view_policy="request")


def post_binary(server, data, headers=None):
    return http(server, "POST", "/api/predict", data, {"Content-Type": BINARY_CONTENT_TYPE, **(headers or {})})

def test_health_advertises_only_binary(running):
    status, _, data = http(running[0], "GET", "/healthz")
    assert status == 200 and json.loads(data)["wire_formats"] == [WIRE_BINARY]


def test_loopback_roundtrip_uses_binary(running):
    server, model, _, _ = running
    request = mixed_request()
    response = predict(f"http://127.0.0.1:{server.server_port}", request, roles=["A", "C", "D"],
                       request_id="roundtrip")
    assert response["schema"] == SCHEMA and response["request_id"] == "roundtrip"
    assert response["result"]["poi_slot"] == 2
    received, roles = model.requests[-1]
    assert roles == ["A", "C", "D"]
    assert_same_request(received, request)


def test_http_binary_rejections_before_model(running):
    server, model, _, _ = running
    good = build_binary(observations(), request_id="reject")
    header, raw = split_binary(good)
    header, raw = dict(header), bytes(raw)
    bad_bodies = [b"XXXX" + good[4:], good[:20], good + b"\0", good[:-1],
                  body({**header, "schema": "unknown"}, raw), body({**header, "roles": "D"}, raw)]
    for data in bad_bodies:
        status, _, response = post_binary(server, data)
        error = json.loads(response)
        assert status == 400 and error["schema"] == SCHEMA and error["error"]["code"] == "invalid_request"
    # A valid header's request_id is echoed even when its pixel section is invalid.
    assert json.loads(post_binary(server, good + b"\0")[2])["request_id"] == "reject"
    status, _, response = post_binary(server, good, {"Content-Type": "application/octet-stream"})
    assert status == 415 and BINARY_CONTENT_TYPE in json.loads(response)["error"]["message"]
    # The binary body cap is enforced before reading the request.
    for length in (MAX_BINARY_BODY_BYTES + 1, MAX_BINARY_BODY_BYTES + 1024):
        status, _, response = http(server, "POST", "/api/predict", b"",
                                   {"Content-Type": BINARY_CONTENT_TYPE, "Content-Length": str(length)})
        error = json.loads(response)["error"]
        assert status == 413 and error["code"] == "body_too_large" and "48 MiB of RGB" in error["message"]
    status, _, response = http(server, "POST", "/api/predict", b"",
                               {"Content-Type": "application/json", "Content-Length": str(MAX_BINARY_BODY_BYTES + 1)})
    assert status == 415 and BINARY_CONTENT_TYPE in json.loads(response)["error"]["message"]
    # JSON body under the binary type, and binary body under the JSON type, are both rejected.
    assert post_binary(server, json.dumps(header).encode())[0] == 400
    assert http(server, "POST", "/api/predict", good, {"Content-Type": "application/json"})[0] == 415
    assert not model.requests
    assert post_binary(server, good, {"Content-Type": BINARY_CONTENT_TYPE + "; version=1"})[0] == 200

def large_request():
    """Four 8x240x320 views: about 7 MiB of RGB, far more than a socket buffer."""
    request = observations()
    images = {name: tuple(np.full((240, 320, 3), 16 * camera + slot, dtype=np.uint8) for slot in range(8))
              for camera, name in enumerate(VIEWS)}
    return replace(request, images=images)

class OldServer(ThreadingHTTPServer):
    """An incompatible server that rejects binary before reading the upload."""

    daemon_threads = True

    def __init__(self):
        self.content_types, self.health_calls = [], 0
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def reply(self, status, value):
                data = json.dumps(value).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                server.health_calls += 1
                self.reply(200, {"schema": SCHEMA, "ready": True})

            def do_POST(self):
                server.content_types.append(self.headers.get_content_type())
                self.reply(415, {"schema": SCHEMA, "error": {"code": "unsupported_media_type", "message": "legacy server"}})
                self.close_connection = True

        super().__init__(("127.0.0.1", 0), Handler)


def serve(server):
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


@pytest.fixture
def old_server():
    yield from serve(OldServer())

def record_posts(monkeypatch):
    seen = []
    original = client.post_encoded
    monkeypatch.setattr(client, "post_encoded", lambda *args, **kwargs: seen.append(args[2]) or original(*args, **kwargs))
    return seen


def test_default_uses_binary_without_capability_probe(running, monkeypatch):
    server, model, _, _ = running
    url = f"http://127.0.0.1:{server.server_port}"
    seen = record_posts(monkeypatch)
    calls = []
    original = client.urlopen
    monkeypatch.setattr(client, "urlopen", lambda wire, **kwargs: calls.append(wire.full_url) or original(wire, **kwargs))
    response = predict(url, observations(), request_id="default")
    assert response["request_id"] == "default" and len(model.requests) == 1
    assert model.requests[0][1] == ["C", "D"]
    assert seen == [BINARY_CONTENT_TYPE] and calls == [url + "/api/predict"]


def test_default_binary_does_not_fall_back_on_old_server(old_server):
    url = f"http://127.0.0.1:{old_server.server_port}"
    with pytest.raises(PredictionError) as error:
        predict(url, observations(), roles=["A"], request_id="default-old")
    assert error.value.status == 415
    assert old_server.content_types == [BINARY_CONTENT_TYPE] and old_server.health_calls == 0

@pytest.mark.parametrize("failure", [
    lambda: PredictionError(415, {"error": {"message": "unsupported"}}),
    lambda: PredictionError(400, {"error": {"message": "bad request"}}),
    lambda: URLError(BrokenPipeError()),
    lambda: ConnectionResetError(),
    lambda: URLError(TimeoutError()),
    lambda: TimeoutError(),
])
def test_send_failure_never_retries_or_negotiates(old_server, monkeypatch, failure):
    calls = []

    def fail(*args, **kwargs):
        calls.append(args[2])
        raise failure()

    monkeypatch.setattr(client, "post_encoded", fail)
    with pytest.raises((PredictionError, OSError)):
        predict(f"http://127.0.0.1:{old_server.server_port}", observations(), roles=["A"])
    assert calls == [BINARY_CONTENT_TYPE] and old_server.health_calls == 0


def test_large_upload_to_incompatible_server_does_not_fall_back(old_server):
    with pytest.raises((PredictionError, URLError, ConnectionError)):
        predict(f"http://127.0.0.1:{old_server.server_port}", large_request(), roles=["A"])
    assert old_server.content_types == [BINARY_CONTENT_TYPE] and old_server.health_calls == 0


@pytest.mark.parametrize("wire_format", ["auto", "json", "binary"])
def test_removed_format_selector_is_rejected_before_network(wire_format):
    with pytest.raises(TypeError, match="wire_format"):
        predict("http://127.0.0.1:1", observations(), wire_format=wire_format)
    with pytest.raises(TypeError, match="wire_format"):
        encode_request(observations(), wire_format=wire_format)


def test_low_level_post_cannot_send_a_different_content_type():
    with pytest.raises(ValueError, match="require"):
        client.post_encoded("http://127.0.0.1:1", b"{}", "application/json", "unsupported")


def test_client_cli_uses_binary_and_rejects_removed_switch(running, tmp_path, monkeypatch):
    server, model, _, _ = running
    views = {}
    for name, frames in observations().images.items():
        np.save(tmp_path / f"{name}.npy", np.stack(frames))
        views[name] = f"{name}.npy"
    request_file = tmp_path / "request.json"
    request_file.write_text(json.dumps({"views": views, "task": observations().task,
                                       "stage_prompts": observations().stage_prompts}))
    seen = record_posts(monkeypatch)
    output = tmp_path / "default.json"
    args = ["--server", f"http://127.0.0.1:{server.server_port}", "--input", str(request_file)]
    client.main([*args, "--output", str(output)])
    assert json.loads(output.read_text())["result"]["poi_slot"] == 5
    for wire_format in ("binary", "json", "auto"):
        with pytest.raises(SystemExit):
            client.main([*args, "--output", str(tmp_path / f"{wire_format}.json"), "--wire-format", wire_format])
    assert seen == [BINARY_CONTENT_TYPE] and len(model.requests) == 1
    assert model.requests[0][1] == ["C", "D"]


def test_binary_encoder_enforces_limits():
    request = InferenceRequest({"top": tuple(np.zeros((2, 3, 3), dtype=np.uint8) for _ in range(7))})
    with pytest.raises(ValueError, match="eight"):
        build_binary(request)
    big = InferenceRequest({f"v{n}": (np.zeros((4096, 4096, 3), dtype=np.uint8),) * 8 for n in range(2)})
    with pytest.raises(ValueError, match="48 MiB per-request pixel budget"):
        build_binary(big)
    with pytest.raises(BinaryHeaderTooLarge, match="1 MiB"):
        build_binary(replace(observations(), stage_prompts=("x" * 50_000,) * 26), roles=["C"])

def test_binary_rgb_budget_is_enforced_on_client_and_server():
    assert MAX_RGB_BYTES == 48 * 1024 * 1024
    at_cap = (np.zeros((1024, 2048, 3), dtype=np.uint8),) * 8  # exactly MAX_RGB_BYTES
    over = (np.zeros((1025, 2048, 3), dtype=np.uint8),) * 8
    tiny = (np.zeros((1, 1, 3), dtype=np.uint8),) * 8
    encoded = build_binary(InferenceRequest({"top": at_cap}, compressed_task_prompt="Move"), roles=["A"], request_id="cap")
    assert len(encoded) <= MAX_BINARY_BODY_BYTES
    decoded, _, _ = decode_binary(encoded, ("top",), view_policy="request")
    assert decoded.images["top"][0].shape == (1024, 2048, 3)
    for images, match in (({"top": over}, "top"), ({"top": at_cap, "left": tiny}, None)):
        with pytest.raises(ValueError, match="48 MiB per-request pixel budget"):
            build_binary(InferenceRequest(images), roles=["A"])
    # The server enforces the budget across views even when a client skips the encoder check.
    header, raw = split_binary(encoded)
    header = dict(header, views={**header["views"], "left": {"shape": [8, 1, 1, 3], "offset": MAX_RGB_BYTES, "nbytes": 24}})
    with pytest.raises(ValueError, match="left: RGB pixel bytes exceed the 48 MiB per-request budget"):
        decode_binary(body(header, bytes(raw) + b"\0" * 24), ("top", "left"), view_policy="request")

def test_oversized_header_fails_before_upload_without_fallback(running, monkeypatch):
    server, model, _, _ = running
    request = replace(observations(), stage_prompts=("x" * 50_000,) * 26)
    seen = record_posts(monkeypatch)
    with pytest.raises(BinaryHeaderTooLarge):
        predict(f"http://127.0.0.1:{server.server_port}", request, roles=["C"], request_id="long")
    assert seen == [] and model.requests == []


def test_audit_records_binary_wire_format(running):
    server, _, audit, _ = running
    predict(f"http://127.0.0.1:{server.server_port}", observations(), roles=["A"], request_id="audit-binary")
    for _ in range(200):
        if audit.read_text().count("\n"):
            break
        time.sleep(0.005)
    events = {event["request_id"]: event for event in map(json.loads, audit.read_text().splitlines())}
    assert events["audit-binary"]["wire_format"] == WIRE_BINARY
