# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The HTTP boundary preserves observations and cannot supply labels or paths."""
from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
from http.client import HTTPConnection
import json
import threading
import time

import numpy as np
from PIL import Image
import pytest

from grana.inference.client import (BINARY_CONTENT_TYPE, BINARY_MAGIC, BINARY_PREAMBLE, MAX_BINARY_BODY_BYTES,
                                    SCHEMA, PredictionError, build_binary, predict)
from grana.inference.predictor import InferenceRequest
from grana.modeling.inputs import PixelBudgetError
from grana.inference.server import create_server, decode_binary, split_binary


VIEWS = ("top", "left", "right", "bottom")


def observations():
    return InferenceRequest(
        {view: tuple(np.full((2, 3, 3), 16 * camera + slot, dtype=np.uint8) for slot in range(8))
         for camera, view in enumerate(VIEWS)},
        task="Put a spatula in the drawer", compressed_task_prompt="Put spatula away",
        stage_prompts=("Open drawer", "Put spatula in drawer", "Close drawer"), focus_stage=1)


class Predictor:
    view_names = VIEWS
    image_preprocessing = {"version": 1, "policy": "processor_default", "max_pixels": None,
                           "effective_size": {"shortest_edge": 65536, "longest_edge": 16777216},
                           "source": "legacy_checkpoint", "legacy_checkpoint": True,
                           "legacy_requested_max_pixels": 76800, "enforce_pixel_budget": False}
    checkpoint = {"weights_sha256": "a" * 64, "source_step": 200, "source_stage": "posttrain",
                  "load_report": {"heads": {"progress_head": {"outcome": "loaded"},
                                            "temporal_head": {"outcome": "loaded"}}}}

    def __init__(self):
        self.requests = []
        self.active = 0
        self.maximum_active = 0
        self.failure = None

    def predict(self, request, roles):
        self.active += 1
        self.maximum_active = max(self.maximum_active, self.active)
        try:
            self.requests.append((request, roles))
            time.sleep(0.01)
            if self.failure:
                raise self.failure
            return {"checkpoint": self.checkpoint, "poi_slot": request.poi_slot, "roles": roles,
                    "pixels": {view: [int(image[0, 0, 0]) for image in frames]
                               for view, frames in request.images.items()}, "D_focus_stage": 2}
        finally:
            self.active -= 1


@pytest.fixture
def running(tmp_path):
    predictor = Predictor()
    media = tmp_path / "clip.mp4"
    media.write_bytes(bytes(range(256)) * 1024)
    page = tmp_path / "index.html"
    page.write_text("<!doctype html><title>Grana</title>")
    audit = tmp_path / "audit.jsonl"
    server = create_server(predictor, media_manifest={"/": str(page), "/clip.mp4": str(media)}, audit_log=audit)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server, predictor, audit, media
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


def http(server, method, path, body=None, headers=None):
    connection = HTTPConnection("127.0.0.1", server.server_port, timeout=5)
    connection.request(method, path, body=body, headers=headers or {})
    response = connection.getresponse()
    result = response.status, dict(response.getheaders()), response.read()
    connection.close()
    return result


class BinaryHeader(dict):
    """Mutable test header with the original binary RGB section kept separately."""

    def __init__(self, header, raw):
        super().__init__(header)
        self.raw = raw


def payload_for(request, *args, **kwargs):
    return BinaryHeader(*split_binary(build_binary(request, *args, **kwargs)))


def binary_body(payload):
    header = json.dumps(payload).encode()
    return BINARY_PREAMBLE.pack(BINARY_MAGIC, 1, len(header)) + header + payload.raw


def post(server, payload, headers=None):
    return http(server, "POST", "/api/predict", binary_body(payload),
                {"Content-Type": BINARY_CONTENT_TYPE, **(headers or {})})


def test_lossless_roundtrip_preserves_slot_view_order_and_checkpoint(running):
    server, model, audit, _ = running
    request = observations()
    # PIL and a noncontiguous array must serialize to the same supplied pixels.
    request.images["top"] = tuple(Image.fromarray(item) for item in request.images["top"])
    request.images["left"] = tuple(item[:, ::-1] for item in request.images["left"])
    result = predict(f"http://127.0.0.1:{server.server_port}", request, roles=["A", "C", "D"], request_id="pixel-test")
    assert result["schema"] == SCHEMA and result["request_id"] == "pixel-test"
    assert result["result"]["checkpoint"] == model.checkpoint
    assert result["result"]["poi_slot"] == 5
    received, roles = model.requests[0]
    assert tuple(received.images) == VIEWS and roles == ["A", "C", "D"]
    for view in VIEWS:
        for actual, expected in zip(received.images[view], request.images[view], strict=True):
            np.testing.assert_array_equal(actual, np.asarray(expected))
            assert actual.shape == (2, 3, 3) and actual.dtype == np.uint8
    for _ in range(100):
        if audit.stat().st_size:
            break
        time.sleep(0.005)
    event = json.loads(audit.read_text())
    assert event["request_id"] == "pixel-test" and event["status"] == 200
    assert event["checkpoint"]["weights_sha256"] == "a" * 64
    assert event["image_preprocessing"] == model.image_preprocessing
    assert event["view_shapes"] == {view: [8, 2, 3, 3] for view in VIEWS}
    assert all(key not in audit.read_text() for key in ("rgb_base64", "Put spatula", "stage_prompts"))


def test_health_exposes_loaded_checkpoint_and_heads(running):
    server, model, _, _ = running
    status, headers, body = http(server, "GET", "/healthz")
    data = json.loads(body)
    assert status == 200 and data["ready"] is True
    assert data["checkpoint"] == model.checkpoint
    assert data["views"] == list(VIEWS) and data["poi_slot"] == 5
    assert data["prediction_frames"] == {"poi": 5, "last": 7}
    assert data["loaded_heads"] == model.checkpoint["load_report"]["heads"]
    assert data["image_preprocessing"] == model.image_preprocessing
    assert headers["X-Content-Type-Options"] == "nosniff"


def test_query_slot_survives_http_and_default_wire_stays_compatible(running):
    server, model, _, _ = running
    request = observations()
    assert "poi_slot" not in payload_for(request)
    selected = replace(request, poi_slot=2)
    response = predict(f"http://127.0.0.1:{server.server_port}", selected, roles="D")
    assert response["result"]["poi_slot"] == 2
    assert model.requests[-1][0].poi_slot == 2
    assert payload_for(selected)["poi_slot"] == 2


@pytest.mark.parametrize("slot", [True, -1, 8, 2.0, "2", None])
def test_http_rejects_invalid_query_slot_before_model_call(running, slot):
    server, model, _, _ = running
    payload = payload_for(observations())
    payload["poi_slot"] = slot
    status, _, _ = post(server, payload)
    assert status == 400 and not model.requests


def test_http_named_override_and_source_indices_reach_the_model(running):
    server, model, _, _ = running
    indices = (6, 7, 8, 1, 5, 9, 13, 13)
    original = replace(observations(), frame_indices=indices, prediction_frame="poi")
    response = predict(f"http://127.0.0.1:{server.server_port}", original, roles=["C", "D"], prediction_frame="last")
    received, roles = model.requests[-1]
    assert received.poi_slot == response["result"]["poi_slot"] == 7
    assert received.prediction_frame == "last" and received.frame_indices == indices
    assert roles == ["C", "D"] and original.poi_slot == 5
    for view in VIEWS:
        assert len(received.images[view]) == 8
        for actual, expected in zip(received.images[view], original.images[view], strict=True):
            np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("selection", [{"prediction_frame": "last", "poi_slot": 5},
                                      {"prediction_frame": "poi", "poi_slot": 7},
                                      {"prediction_frame": "first"}, {"frame_indices": [True] * 8}])
def test_http_rejects_conflicting_or_invalid_selection_before_model(running, selection):
    server, model, _, _ = running
    payload = payload_for(observations())
    payload.update(selection)
    status, _, data = post(server, payload)
    assert status == 400 and not model.requests
    assert json.loads(data)["error"]["code"] == "invalid_request"


@pytest.mark.parametrize("range_header,expected_status,start,end", [
    (None, 200, 0, 262143), ("bytes=10-19", 206, 10, 19),
    ("bytes=262140-", 206, 262140, 262143), ("bytes=-4", 206, 262140, 262143),
    ("bytes=262140-999999", 206, 262140, 262143), ("bytes=-999999", 206, 0, 262143),
])
def test_media_range_and_head(running, range_header, expected_status, start, end):
    server, _, _, media = running
    headers = {"Range": range_header} if range_header else {}
    status, response_headers, body = http(server, "GET", "/clip.mp4", headers=headers)
    assert status == expected_status and body == media.read_bytes()[start:end + 1]
    assert response_headers["Content-Length"] == str(end - start + 1)
    assert response_headers["Content-Type"] == "video/mp4"
    assert response_headers["Accept-Ranges"] == "bytes"
    if range_header:
        assert response_headers["Content-Range"] == f"bytes {start}-{end}/262144"
    status, head_headers, body = http(server, "HEAD", "/clip.mp4", headers=headers)
    assert status == expected_status and body == b""
    assert head_headers["Content-Length"] == response_headers["Content-Length"]


@pytest.mark.parametrize("value", ["bytes=1-2,4-5", "bytes=262144-", "bytes=10-9", "bytes=-0", "bytes=-", "items=1-2", "bytes=abc-3"])
def test_invalid_single_ranges(running, value):
    server, _, _, _ = running
    status, headers, body = http(server, "GET", "/clip.mp4", headers={"Range": value})
    assert status == 416 and headers["Content-Range"] == "bytes */262144"
    assert json.loads(body)["error"]["code"] == "invalid_range"


@pytest.mark.parametrize("path", ["/../clip.mp4", "/%2e%2e/clip.mp4", "/%2Fclip.mp4", "/clip.mp4?anything=1", "/absent", "/etc/passwd"])
def test_only_literal_allowlisted_paths(running, path):
    assert http(running[0], "GET", path)[0] == 404


def test_media_rejects_symlinks_and_directory(tmp_path):
    real = tmp_path / "file"
    real.write_text("test")
    link = tmp_path / "link"
    link.symlink_to(real)
    for filename in (link, tmp_path):
        with pytest.raises(ValueError):
            create_server(Predictor(), media_manifest={"/file": str(filename)})
    with pytest.raises(ValueError):
        create_server(Predictor(), host="0.0.0.0")


def test_replaced_file_cannot_follow_symlink(running, tmp_path):
    server, _, _, media = running
    secret = tmp_path / "private"
    secret.write_text("not allowlisted")
    media.unlink()
    media.symlink_to(secret)
    status, _, body = http(server, "GET", "/clip.mp4")
    assert status == 404 and b"not allowlisted" not in body


@pytest.mark.parametrize("mutation", [
    lambda p: p.update(gt={"stage": 1}),
    lambda p: p.update(checkpoint="/other/checkpoint"),
    lambda p: p.update(path="/etc/passwd"),
    lambda p: p.update(roles=["D", "D"]),
    lambda p: p.update(roles="D"),
    lambda p: p.update(request_id="bad\nheader"),
    lambda p: p.update(schema="unknown"),
    lambda p: p.update(views=dict(reversed(list(p["views"].items())))),
    lambda p: p["views"]["top"].update(path="/etc/passwd"),
    lambda p: p["views"]["top"].update(shape=[7, 2, 3, 3]),
    lambda p: p["views"]["top"].update(shape=[8, True, 3, 3]),
    lambda p: p["views"]["top"].update(shape=[8, 0, 3, 3]),
    lambda p: p["views"]["top"].update(shape=[8, 4097, 3, 3]),
    lambda p: p["views"]["top"].update(offset=-1),
    lambda p: p["views"]["top"].update(nbytes=3),
    lambda p: p.update(stage_prompts=["<|stage|>"]),
    lambda p: p.update(stage_prompts=[]),
    lambda p: p.update(focus_stage=True),
    lambda p: p.update(focus_stage=3),
])
def test_malformed_payload_rejected_before_model(running, mutation):
    server, model, _, _ = running
    payload = payload_for(observations(), request_id="invalid")
    mutation(payload)
    status, _, body = post(server, payload)
    error = json.loads(body)
    assert status == 400 and error["schema"] == SCHEMA
    assert error["error"]["code"] == "invalid_request"
    assert not model.requests


def test_body_limits_binary_content_type_and_origin(running):
    server, model, _, _ = running
    payload = payload_for(observations())
    assert post(server, payload, {"Content-Type": "text/plain"})[0] == 415
    assert post(server, payload, {"Origin": "http://another-host"})[0] == 403
    assert post(server, payload, {"Origin": f"https://127.0.0.1:{server.server_port}"})[0] == 403
    assert http(server, "POST", "/api/predict", b"", {"Content-Type": BINARY_CONTENT_TYPE, "Content-Length": str(MAX_BINARY_BODY_BYTES + 1)})[0] == 413
    assert post(server, payload, {"Content-Type": "application/json"})[0] == 415
    for bad in (b'{"schema": 1, "schema": 2}', b'{"foo": NaN}', b"not json"):
        wire = BINARY_PREAMBLE.pack(BINARY_MAGIC, 1, len(bad)) + bad
        assert http(server, "POST", "/api/predict", wire, {"Content-Type": BINARY_CONTENT_TYPE})[0] == 400
    assert not model.requests
    assert post(server, payload, {"Origin": f"http://127.0.0.1:{server.server_port}"})[0] == 200


def test_pixel_budget_failure_is_actionable_http_400_without_request_payload(running):
    server, model, _, _ = running
    message = ("Processor resize exceeded max_pixels=1024: got 2048 spatial pixels per image; "
               "minimum size and patch rounding cannot satisfy this budget")
    model.failure = PixelBudgetError(message)
    payload = payload_for(observations(), request_id="budget-exceeded")
    status, _, body = post(server, payload)
    response = json.loads(body)
    assert status == 400 and response["schema"] == SCHEMA
    assert response["request_id"] == "budget-exceeded"
    assert response["error"] == {"code": "image_pixel_budget_exceeded", "message": message}
    assert len(model.requests) == 1
    for private in ("stage_prompts", payload["task"]):
        assert private not in body.decode()


@pytest.mark.parametrize("exception", [ValueError, TypeError, RuntimeError])
def test_failures_are_json_without_tracebacks(running, exception):
    server, model, _, _ = running
    model.failure = exception("private model detail")
    with pytest.raises(PredictionError) as error:
        predict(f"http://127.0.0.1:{server.server_port}", observations(), request_id="failed")
    assert error.value.status == 500
    response = error.value.response
    assert response["request_id"] == "failed" and response["error"]["code"] == "prediction_failed"
    assert "private model detail" not in json.dumps(response)
    assert http(server, "PUT", "/api/predict")[0] == 405


def test_prediction_thread_state_survives_connections_and_failures(running, monkeypatch):
    server, model, _, _ = running
    original = model.predict
    local = threading.local()
    calls = []

    def tracked(request, roles):
        local.count = getattr(local, "count", 0) + 1
        calls.append((threading.current_thread(), local.count))
        if len(calls) == 2:
            raise RuntimeError("one failed prediction must not discard the warmed worker")
        return original(request, roles)

    monkeypatch.setattr(model, "predict", tracked)
    url = f"http://127.0.0.1:{server.server_port}"
    # Each public client call opens and closes its own HTTP connection.
    assert predict(url, observations(), request_id="first")["request_id"] == "first"
    with pytest.raises(PredictionError) as error:
        predict(url, observations(), request_id="failed")
    assert error.value.status == 500
    assert predict(url, observations(), request_id="recovered")["request_id"] == "recovered"
    assert [count for _, count in calls] == [1, 2, 3]
    # Thread IDs can be recycled; retaining the objects catches fresh threads.
    assert all(thread is calls[0][0] for thread, _ in calls)


def test_concurrent_requests_serialize_only_model_access(running, monkeypatch):
    server, model, _, _ = running
    original = model.predict
    entered = threading.Event()
    release = threading.Event()

    def blocked(request, roles):
        entered.set()
        assert release.wait(timeout=5)
        return original(request, roles)

    monkeypatch.setattr(model, "predict", blocked)
    url = f"http://127.0.0.1:{server.server_port}"
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(predict, url, observations(), request_id=f"concurrent-{n}") for n in range(4)]
        try:
            assert entered.wait(timeout=5)
            assert http(server, "GET", "/healthz")[0] == 200
        finally:
            release.set()
        results = [future.result(timeout=5) for future in futures]
    assert model.maximum_active == 1 and len(model.requests) == 4
    assert {item["request_id"] for item in results} == {f"concurrent-{n}" for n in range(4)}


def test_server_close_waits_for_prediction_and_releases_worker(monkeypatch):
    model = Predictor()
    original = model.predict
    entered = threading.Event()
    release = threading.Event()
    closing = threading.Event()
    closed = threading.Event()
    workers = []

    def blocked(request, roles):
        workers.append(threading.current_thread())
        entered.set()
        assert release.wait(timeout=5)
        return original(request, roles)

    monkeypatch.setattr(model, "predict", blocked)
    server = create_server(model)
    serving = threading.Thread(target=server.serve_forever, daemon=True)
    serving.start()

    def close():
        closing.set()
        server.server_close()
        closed.set()

    closer = threading.Thread(target=close, daemon=True)
    try:
        with ThreadPoolExecutor(max_workers=1) as clients:
            response = clients.submit(predict, f"http://127.0.0.1:{server.server_port}",
                                      observations(), request_id="finish-before-close")
            try:
                assert entered.wait(timeout=5)
                server.shutdown()
                closer.start()
                assert closing.wait(timeout=5)
                assert not closed.wait(timeout=0.05)
                assert workers[0].is_alive()
            finally:
                release.set()
            assert response.result(timeout=5)["request_id"] == "finish-before-close"
        assert closed.wait(timeout=5)
        closer.join(timeout=5)
        assert not workers[0].is_alive()
    finally:
        release.set()
        server.shutdown()
        server.server_close()
        serving.join(timeout=5)
        if closer.ident is not None:
            closer.join(timeout=5)


def test_a_only_does_not_require_stage_annotations():
    request = InferenceRequest(observations().images, compressed_task_prompt="Move tool")
    decoded, roles, _ = decode_binary(binary_body(payload_for(request, roles=["A"])), VIEWS)
    assert roles == ["A"] and decoded.stage_prompts == [] and decoded.task is None


def test_client_cli_uses_local_files_but_sends_only_pixels(running, tmp_path):
    from grana.inference.client import main
    server, model, _, _ = running
    views = {}
    for name, frames in observations().images.items():
        filename = tmp_path / f"{name}.npy"
        np.save(filename, np.stack(frames))
        views[name] = filename.name
    request_file = tmp_path / "request.json"
    request_file.write_text(json.dumps({"views": views, "compressed_task_prompt": "Move tool"}))
    output = tmp_path / "response.json"
    args = ["--server", f"http://127.0.0.1:{server.server_port}", "--input", str(request_file),
            "--roles", "A", "--output", str(output)]
    main(args)
    assert json.loads(output.read_text())["result"]["poi_slot"] == 5
    with pytest.raises(FileExistsError):
        main(args)
    assert len(model.requests) == 1


def test_client_cli_role_frame_overrides_and_window_composition(running, tmp_path):
    from grana.inference.client import main
    server, model, _, _ = running
    full = np.arange(120, dtype=np.uint8)[:, None, None, None] * np.ones((120, 2, 3, 3), dtype=np.uint8)
    np.save(tmp_path / "full.npy", full)
    np.save(tmp_path / "eight.npy", full[:8])
    cases = [({}, [], 5, None), ({"prediction_frame": "poi"}, [], 5, None),
             ({"prediction_frame": "last"}, [], 7, None),
             ({"poi_slot": 5}, ["--prediction-frame", "last"], 7, None),
             ({"prediction_frame": "last"}, ["--poi-slot", "2"], 2, None),
             ({}, ["--frame-index", "100"], 5, (52, 68, 84, 92, 96, 100, 104, 108)),
             ({}, ["--frame-index", "100", "--prediction-frame", "last"],
              7, (44, 60, 76, 84, 88, 92, 96, 100))]
    for index, (selection, flags, slot, indices) in enumerate(cases):
        filename = "full.npy" if indices is not None else "eight.npy"
        raw = {"views": dict.fromkeys(VIEWS, filename), "task": "Move tool",
               "compressed_task_prompt": "Move tool", "stage_prompts": ["Pick", "Place"], **selection}
        path = tmp_path / f"request-{index}.json"
        path.write_text(json.dumps(raw))
        for roles in (["A"], ["C"], ["C", "D"]):
            output = tmp_path / f"response-{index}-{''.join(roles)}.json"
            main(["--server", f"http://127.0.0.1:{server.server_port}", "--input", str(path),
                  "--output", str(output), "--roles", *roles, *flags])
            received, received_roles = model.requests[-1]
            assert received_roles == roles and received.poi_slot == slot
            assert received.frame_indices == indices
            expected = list(indices) if indices is not None else list(range(8))
            assert [int(image[0, 0, 0]) for image in received.images["top"]] == expected


def test_explicit_request_policy_accepts_dynamic_camera_counts_and_reports_health(running):
    server, model, _, _ = running
    model.view_policy = "request"
    _, _, body = http(server, "GET", "/healthz")
    health = json.loads(body)
    assert health["view_policy"] == "request"
    assert health["views"] == health["checkpoint_view_names"] == list(VIEWS)
    for count in (1, 2, 3, 4, 5):
        names = tuple(f"camera_{camera}" for camera in reversed(range(count)))
        frames = {name: tuple(np.full((camera + 2, camera + 3, 3), 16 * camera + slot, dtype=np.uint8)
                              for slot in range(8)) for camera, name in enumerate(names)}
        request = InferenceRequest(frames, compressed_task_prompt="Move tool")
        predict(f"http://127.0.0.1:{server.server_port}", request, roles=["A"])
        received, roles = model.requests[-1]
        assert tuple(received.images) == names and roles == ["A"]
        for name in names:
            for actual, expected in zip(received.images[name], frames[name], strict=True):
                np.testing.assert_array_equal(actual, expected)
    assert len(model.requests) == 5 and model.view_names == VIEWS


@pytest.mark.parametrize("views", [{}, {"": {}}, {"<|stage_token|>": {}}, {"x" * 129: {}}])
def test_request_policy_still_requires_valid_named_cameras(views):
    payload = payload_for(observations(), roles=["A"])
    payload["views"] = views
    with pytest.raises(ValueError):
        decode_binary(binary_body(payload), VIEWS, view_policy="request")


def test_wire_request_cannot_enable_dynamic_policy_on_strict_server(running):
    server, model, _, _ = running
    payload = payload_for(observations(), roles=["A"])
    payload["views"] = {"wrist": payload["views"]["top"]}
    assert post(server, payload)[0] == 400
    payload["view_policy"] = "request"
    assert post(server, payload)[0] == 400
    assert not model.requests


def test_unknown_view_policy_cannot_disable_wire_validation():
    with pytest.raises(ValueError, match="view_policy"):
        decode_binary(binary_body(payload_for(observations())), VIEWS, view_policy="automatic")
