# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import hashlib
import json
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from grana.inference import client
from grana.inference.selection import prediction_frame_for_slot, select_prediction
from scripts.inference import benchmark_http as benchmark


def result_for(request, roles):
    runtime = {key: True for key in ("gpu_image_resize", "gpu_image_preprocessing", "batched_vision_sdpa",
                                   "reuse_images", "merge_lora", "cpu_position_ids", "compile_decoder", "compile_vision")}
    runtime.update(policy="cuda_optimized_v1", attn_implementation="sdpa", cpu_threads=8,
                   delta_rule_backend={"resolved": "fla"}, causal_conv1d={"active": True},
                   text_input_cache={"enabled": True}, fla_output_guard={"applied": True})
    result = {"roles": list(roles), "poi_slot": request.poi_slot,
              "prediction_frame": prediction_frame_for_slot(request.poi_slot),
              "frame_indices": list(request.frame_indices), "view_names": list(request.images),
              "view_policy": "checkpoint", "max_pixels": 76800,
              "image_preprocessing": {"resize_backend": "torchvision_cuda"},
              "checkpoint": {"weights_sha256": "a" * 64, "source_step": 1000, "inference_runtime": runtime},
              "progress": {}}
    if "C" in roles:
        ids = [0, 0, 0, 0, 0, 0, 1, 1]
        result["stage"] = {"ids": ids, "logits": [[0., float(value)] for value in ids],
                           "probabilities": [[1. - value, float(value)] for value in ids],
                           "poi_stage_id": ids[request.poi_slot]}
    for role in set(roles) - {"C"}:
        values = [slot / 9 for slot in range(8)]
        result["progress"][role] = {"values": values, "poi_score": values[request.poi_slot],
                                    "bin_probabilities": [[float(bin_id == slot) for bin_id in range(10)]
                                                          for slot in range(8)]}
        if role == "D":
            focus = result["stage"]["ids"][request.poi_slot]
            result["D_focus_stage"] = result["progress"][role]["focus_stage"] = focus
    result["selected_prediction"] = select_prediction(result)
    return result


def envelope(request, roles, request_id):
    return {"schema": client.SCHEMA, "request_id": request_id, "elapsed_seconds": .004,
            "result": result_for(request, roles)}


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    frames = np.arange(200, dtype=np.uint8)[:, None, None, None] * np.ones((200, 2, 3, 3), dtype=np.uint8)
    np.save(tmp_path / "top.npy", frames)
    np.save(tmp_path / "wrist.npy", frames[:, :, ::-1])
    source = tmp_path / "request.json"
    raw = {"views": {"top": "top.npy", "wrist": "wrist.npy"}, "task": "Assemble the car",
           "compressed_task_prompt": "Assemble the car", "stage_prompts": ["Attach wheels", "Finish car"]}
    source.write_text(json.dumps(raw))
    monkeypatch.setattr(benchmark, "source_identity", lambda: {"test_checkout": True})
    args = argparse.Namespace(server="http://127.0.0.1:9876", input=source, frame_index=100,
                              roles=["C", "D"], prediction_frame="last", warmup=1, repeats=2,
                              timeout=300, output=tmp_path / "report.json")
    return args, raw


def install_fake_client(monkeypatch, callback=None):
    calls = []

    def predict(server, request, *, roles, request_id, timeout, timing):
        started = benchmark.time.perf_counter_ns()
        calls.append((request, roles))
        response = envelope(request, roles, request_id)
        if callback is not None:
            callback(response, len(calls))
        timing["client_send_to_receive_ms"] = (benchmark.time.perf_counter_ns() - started) / 1_000_000
        timing["http_round_trip_ms"] = timing["client_send_to_receive_ms"]
        timing["client_send_ms"] = 0.
        timing["client_send_complete_to_receive_ms"] = timing["client_send_to_receive_ms"]
        return response

    monkeypatch.setattr(client, "predict", predict)
    return calls


@pytest.mark.parametrize("roles", [("A",), ("C",), ("C", "D")])
@pytest.mark.parametrize("mode,expected", [
    ("poi", (52, 68, 84, 92, 96, 100, 104, 108)),
    ("last", (44, 60, 76, 84, 88, 92, 96, 100)),
])
def test_each_workload_uses_current_evaluation_geometry_and_preserves_outputs(campaign, monkeypatch, roles, mode, expected):
    args, _ = campaign
    args.roles, args.prediction_frame = list(roles), mode
    calls = install_fake_client(monkeypatch)
    report = benchmark.run(args)
    assert len(calls) == 3 and all(actual == roles for _, actual in calls)
    assert all(request.frame_indices == expected for request, _ in calls)
    assert [int(frame[0, 0, 0]) for frame in calls[0][0].images["top"]] == list(expected)
    assert report["wire_format"] == client.WIRE_BINARY
    assert report["schema"] == "grana.http_benchmark.v3"
    assert report["timing"]["primary_metric"] == "client_send_to_receive_ms"
    assert "binary packing" in report["timing"]["excludes"]
    assert "response JSON parsing" in report["timing"]["excludes"]
    assert report["geometry"]["context_stride"] == 16 and report["geometry"]["tail_stride"] == 4
    assert report["geometry"]["has_future_observations"] is (mode == "poi")
    assert report["warmup"]["count"] == 1 and report["measured"]["count"] == 2
    result = report["sample_result"]
    assert result["selected_prediction"]["frame_index"] == 100
    if "C" in roles:
        assert len(result["stage"]["ids"]) == len(result["stage"]["logits"]) == 8
    for output in result["progress"].values():
        assert len(output["values"]) == len(output["bin_probabilities"]) == 8
    assert report["input"]["sources"]["top"]["size_bytes"] == (args.input.parent / "top.npy").stat().st_size
    assert report["input"]["sources"]["top"]["shape"] == [200, 2, 3, 3]
    assert "sha256" not in report["input"]["sources"]["top"]
    assert report["input"]["request_json_sha256"] == hashlib.sha256(args.input.read_bytes()).hexdigest()
    assert len(report["input"]["input_sha256"]) == 64
    assert json.loads(args.output.read_text()) == report


@pytest.mark.parametrize("index", [0, 1, 7, 8, 47, 199])
def test_last_frame_uses_no_future_pixels_even_with_full_video_on_disk(campaign, monkeypatch, index):
    args, _ = campaign
    args.frame_index = index
    calls = install_fake_client(monkeypatch)
    benchmark.run(args)
    request = calls[0][0]
    assert request.frame_indices[7] == index
    assert max(request.frame_indices) <= index
    assert [int(frame[0, 0, 0]) for frame in request.images["top"]] == list(request.frame_indices)


def test_public_and_send_receive_timers_exclude_loading_hashes_and_validation(campaign, monkeypatch):
    args, _ = campaign
    clock = SimpleNamespace(ns=0)
    monkeypatch.setattr(benchmark.time, "perf_counter_ns", lambda: clock.ns)
    original_prepare, original_validate = benchmark.prepare_input, benchmark.validate_response

    def prepare(*a):
        clock.ns += 10_000_000_000
        return original_prepare(*a)

    def validate(*a):
        clock.ns += 20_000_000_000
        return original_validate(*a)

    def predict(server, request, *, roles, request_id, timeout, timing):
        client.encode_request(request, roles, request_id)
        clock.ns += 3_000_000
        clock.ns += 7_000_000
        timing["client_send_to_receive_ms"] = 7.
        timing["http_round_trip_ms"] = 7.
        timing["client_send_ms"], timing["client_send_complete_to_receive_ms"] = 2., 5.
        response = json.loads(json.dumps(envelope(request, roles, request_id)))
        clock.ns += 5_000_000
        return response

    monkeypatch.setattr(benchmark, "prepare_input", prepare)
    monkeypatch.setattr(benchmark, "validate_response", validate)
    monkeypatch.setattr(client, "predict", predict)
    report = benchmark.run(args)
    samples = report["warmup"]["samples"] + report["measured"]["samples"]
    assert all(sample["client_total_ms"] == 15 for sample in samples)
    assert all(sample["client_send_to_receive_ms"] == 7 for sample in samples)
    assert all(sample["http_round_trip_ms"] == 7 for sample in samples)
    assert all(sample["server_handler_ms"] == 4 for sample in samples)


def test_real_binary_loopback_separates_send_receive_from_packing_and_parsing(campaign, monkeypatch):
    from grana.inference.server import create_server

    args, _ = campaign
    args.warmup, args.repeats = 1, 1
    clock, events = SimpleNamespace(ns=0), []
    monkeypatch.setattr(benchmark.time, "perf_counter_ns", lambda: clock.ns)
    original_encode, original_decode = client.encode_request, client.json_loads
    original_connect = client.HTTPConnection.connect

    def connect(connection):
        clock.ns += 17_000_000
        original_connect(connection)

    def encode(*a, **kw):
        events.append("pack")
        clock.ns += 3_000_000
        return original_encode(*a, **kw)

    def decode(raw):
        value = original_decode(raw)
        if isinstance(value, dict) and "result" in value:
            events.append("parse")
            clock.ns += 5_000_000
        return value

    class Predictor:
        view_names = ("top", "wrist")
        checkpoint = {"weights_sha256": "a" * 64, "source_step": 1000}

        def __init__(self):
            self.requests = []

        def predict(self, request, roles):
            events.append("server")
            clock.ns += 7_000_000
            self.requests.append(request)
            return result_for(request, roles)

    model = Predictor()
    server = create_server(model, audit_log=args.input.parent / "audit.jsonl")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    args.server = f"http://127.0.0.1:{server.server_port}"
    monkeypatch.setattr(client, "encode_request", encode)
    monkeypatch.setattr(client, "json_loads", decode)
    monkeypatch.setattr(client.HTTPConnection, "connect", connect)
    try:
        report = benchmark.run(args)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    assert events == ["pack", "server", "parse"] * 2
    assert len(model.requests) == 2
    assert report["warmup"]["samples"][0]["client_total_ms"] == 32
    assert report["measured"]["samples"][0]["client_total_ms"] == 32
    assert report["warmup"]["samples"][0]["client_send_to_receive_ms"] == 7
    assert report["measured"]["samples"][0]["client_send_to_receive_ms"] == 7
    assert report["warmup"]["samples"][0]["http_round_trip_ms"] == 24
    assert report["measured"]["samples"][0]["http_round_trip_ms"] == 24
    assert report["schema"] == "grana.http_benchmark.v3"
    assert report["timing"]["primary_metric"] == "client_send_to_receive_ms"
    assert report["measured"]["samples"][0]["server_handler_ms"] > 0
    audit = [json.loads(line) for line in (args.input.parent / "audit.jsonl").read_text().splitlines()]
    assert len(audit) == 2 and all(event["wire_format"] == client.WIRE_BINARY for event in audit)


def test_send_receive_timer_includes_upload_and_body_read_but_excludes_connect_parse_close(monkeypatch):
    clock = SimpleNamespace(ns=0)
    monkeypatch.setattr(client.time, "perf_counter_ns", lambda: clock.ns)
    original_decode = client.json_loads

    class Response:
        def __enter__(self):
            clock.ns += 2_000_000
            return self

        def read(self, limit):
            assert limit == client.MAX_RESPONSE_BYTES + 1
            clock.ns += 11_000_000
            return json.dumps({"schema": client.SCHEMA, "request_id": "timed", "result": {}}).encode()

        def __exit__(self, *args):
            clock.ns += 13_000_000

    class Socket:
        def sendall(self, data):
            clock.ns += len(data) * 1_000_000

    def open_response(wire, timeout, send_clock):
        send_clock["http_started_ns"] = clock.ns
        clock.ns += 17_000_000  # Connection established before request writes.
        sock = client._SendTimedSocket(Socket(), send_clock)
        sock.sendall(b"hdr")
        sock.sendall(b"payload")
        return Response()

    def decode(raw):
        clock.ns += 5_000_000
        return original_decode(raw)

    monkeypatch.setattr(client, "_open_with_send_timing", open_response)
    monkeypatch.setattr(client, "json_loads", decode)
    timing = {}
    client.post_encoded("http://127.0.0.1:1234", b"packed", client.BINARY_CONTENT_TYPE, "timed", timing=timing)
    assert timing == {"http_round_trip_ms": 40., "client_send_to_receive_ms": 23., "client_send_ms": 10.,
                       "client_send_complete_to_receive_ms": 13.}
    assert clock.ns == 58_000_000


@pytest.mark.parametrize("change,match", [
    ({"frame_indices": list(range(8))}, "cannot be combined"),
    ({"frame_indices": None}, "cannot be combined"),
    ({"views": {"top": ["missing.png"] * 8}}, "requires .npy"),
    ({"views": {"top": "missing.mp4"}}, "use a .npy"),
])
def test_frozen_or_non_video_inputs_fail_before_network(campaign, monkeypatch, change, match):
    args, raw = campaign
    args.input.write_text(json.dumps({**raw, **change}))
    monkeypatch.setattr(client, "predict", lambda *a, **k: pytest.fail("invalid input reached HTTP"))
    with pytest.raises(ValueError, match=match):
        benchmark.run(args)
    assert not args.output.exists()


@pytest.mark.parametrize("change,match", [
    (lambda r: r.update(roles=["A"]), "roles"),
    (lambda r: r.update(poi_slot=5), "frame selection"),
    (lambda r: r["frame_indices"].reverse(), "source indices"),
    (lambda r: r["progress"]["D"].update(values=[.5] * 7), "8 columns"),
    (lambda r: r["progress"]["D"].update(bin_probabilities=[[.1] * 10] * 7), "8 rows"),
    (lambda r: r["stage"].update(logits=[[0., 1.]] * 7), "8 rows"),
    (lambda r: r["stage"]["ids"].__setitem__(0, 99), "eight valid"),
    (lambda r: r.update(D_focus_stage=0), "D focus"),
    (lambda r: r["selected_prediction"].update(slot=5), "metadata"),
    (lambda r: r["checkpoint"].update(weights_sha256="unknown"), "SHA256"),
    (lambda r: r["checkpoint"]["inference_runtime"].update(gpu_image_resize=False), "CUDA runtime"),
])
def test_unqualified_or_changed_responses_are_not_reported(campaign, monkeypatch, change, match):
    args, _ = campaign
    install_fake_client(monkeypatch, lambda response, count: change(response["result"]))
    with pytest.raises(ValueError, match=match):
        benchmark.run(args)
    assert not args.output.exists()


def test_checkpoint_change_between_requests_invalidates_the_report(campaign, monkeypatch):
    args, _ = campaign

    def mutate(response, count):
        if count > 1:
            response["result"]["checkpoint"]["weights_sha256"] = "b" * 64

    install_fake_client(monkeypatch, mutate)
    with pytest.raises(ValueError, match="changed during"):
        benchmark.run(args)
    assert not args.output.exists()


def test_report_statistics_exclude_warmup(campaign, monkeypatch):
    args, _ = campaign
    args.warmup, args.repeats = 1, 3
    clock = SimpleNamespace(ns=0)
    monkeypatch.setattr(benchmark.time, "perf_counter_ns", lambda: clock.ns)
    durations = iter([1000, 1, 2, 9])

    def advance(response, count):
        clock.ns += next(durations) * 1_000_000

    install_fake_client(monkeypatch, advance)
    report = benchmark.run(args)
    assert report["warmup"]["summary"]["client_total_ms"] == {"mean": 1000., "p50": 1000., "p95": 1000.}
    assert report["measured"]["summary"]["client_total_ms"] == pytest.approx({"mean": 4., "p50": 2., "p95": 8.3})
    assert report["measured"]["summary"]["client_send_to_receive_ms"] == pytest.approx({"mean": 4., "p50": 2., "p95": 8.3})
    assert report["measured"]["summary"]["http_round_trip_ms"] == pytest.approx({"mean": 4., "p50": 2., "p95": 8.3})


@pytest.mark.parametrize("metric", benchmark.TRANSPORT_METRICS)
@pytest.mark.parametrize("duration", [None, float("nan"), float("inf"), True, -1.])
def test_missing_or_invalid_transport_measurement_is_not_reported(campaign, monkeypatch, metric, duration):
    args, _ = campaign

    def predict(server, request, *, roles, request_id, timeout, timing):
        timing.update({key: 4. for key in benchmark.TRANSPORT_METRICS if key != metric})
        if duration is not None:
            timing[metric] = duration
        return envelope(request, roles, request_id)

    monkeypatch.setattr(client, "predict", predict)
    with pytest.raises(ValueError, match="time"):
        benchmark.run(args)
    assert not args.output.exists()


def test_input_identity_binds_geometry_selection_and_pixels(campaign):
    args, _ = campaign
    _, last, geometry = benchmark.prepare_input(args.input, 100, "last", ("C", "D"))
    _, poi, _ = benchmark.prepare_input(args.input, 100, "poi", ("C", "D"))
    assert last["input_sha256"] != poi["input_sha256"]
    source = args.input.parent / "top.npy"
    array = np.load(source)
    array[100, 0, 0, 0] ^= 1
    np.save(source, array)
    _, changed, _ = benchmark.prepare_input(args.input, 100, "last", ("C", "D"))
    assert changed["descriptor"]["pixels"]["top"]["rgb_sha256"] != last["descriptor"]["pixels"]["top"]["rgb_sha256"]
    assert changed["input_sha256"] != last["input_sha256"]
    assert geometry["source_frame_index"] == geometry["frame_indices"][geometry["poi_slot"]] == 100


def test_unselected_video_pixels_are_not_hashed_or_loaded_for_timing_identity(campaign, monkeypatch):
    args, _ = campaign
    monkeypatch.setattr(benchmark, "file_digest", lambda *a: pytest.fail("whole source videos must not be hashed"))
    _, before, _ = benchmark.prepare_input(args.input, 100, "last", ("C", "D"))
    source = args.input.parent / "top.npy"
    array = np.load(source)
    array[199] ^= 1
    np.save(source, array)
    _, after, _ = benchmark.prepare_input(args.input, 100, "last", ("C", "D"))
    assert before["input_sha256"] == after["input_sha256"]


def test_existing_output_is_preserved_before_input_or_network(campaign, monkeypatch):
    args, _ = campaign
    args.output.write_text("existing receipt")
    monkeypatch.setattr(benchmark, "prepare_input", lambda *a: pytest.fail("existing output should fail first"))
    with pytest.raises(FileExistsError):
        benchmark.run(args)
    assert args.output.read_text() == "existing receipt"


def test_cli_defaults_and_required_selection(campaign, monkeypatch):
    args, _ = campaign
    seen = []
    monkeypatch.setattr(benchmark, "run", lambda actual: seen.append(actual) or {"geometry": {}, "measured": {"summary": {}}})
    benchmark.main(["--server", args.server, "--input", str(args.input), "--frame-index", "100", "--output", str(args.output)])
    actual = seen[0]
    assert actual.roles == ["C", "D"] and actual.prediction_frame == "poi"
    assert actual.warmup == 5 and actual.repeats == 100
    for flags in ([], ["--wire-format", "json"], ["--context-stride", "30"], ["--no-gpu-image-resize"]):
        arguments = ["--server", args.server, "--input", str(args.input), "--output", str(args.output)]
        if flags:
            arguments += ["--frame-index", "100", *flags]
        with pytest.raises(SystemExit):
            benchmark.main(arguments)


@pytest.mark.parametrize("roles", [["D"], ["A", "C"], ["A", "C", "D"], ["C", "C"]])
def test_only_supported_comparable_workloads_are_accepted(campaign, monkeypatch, roles):
    args, _ = campaign
    args.roles = roles
    monkeypatch.setattr(benchmark, "prepare_input", lambda *a: pytest.fail("invalid roles should fail first"))
    with pytest.raises(ValueError, match="roles must be"):
        benchmark.run(args)
