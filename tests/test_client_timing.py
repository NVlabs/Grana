# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Client timing covers sending through the body read, without client preprocessing."""
import io
import json
from types import SimpleNamespace
from urllib.error import HTTPError, URLError

import numpy as np
import pytest

from grana.inference import client


def observation():
    return SimpleNamespace(images={"top": np.zeros((8, 2, 3, 3), dtype=np.uint8)},
                           task="Assemble", compressed_task_prompt=None, stage_prompts=("Attach", "Finish"),
                           focus_stage=None, prediction_frame="last", poi_slot=7, frame_indices=tuple(range(8)))


@pytest.mark.parametrize("entrypoint", ["predict", "post_encoded"])
def test_timer_starts_after_request_construction_and_stops_before_parse_cleanup_or_validation(monkeypatch, entrypoint):
    clock, events = SimpleNamespace(ns=0), []
    original_encode, original_request, original_json = client.encode_request, client.Request, client.json_loads
    expected = {"schema": client.SCHEMA, "request_id": "timing", "result": {"unchanged": [1, 2, 3]}}
    encoded_response = json.dumps(expected).encode()
    timing = {"caller_value": 42., "http_round_trip_ms": -1., "client_send_to_receive_ms": -1.,
              "client_send_ms": -1., "client_send_complete_to_receive_ms": -1.}

    def advance(event, milliseconds):
        events.append(event)
        clock.ns += milliseconds * 1_000_000

    def encode(*args, **kwargs):
        advance("pack", 100)
        return original_encode(*args, **kwargs)

    def request(*args, **kwargs):
        advance("construct", 200)
        return original_request(*args, **kwargs)

    class ValidatedEnvelope(dict):
        def get(self, *args, **kwargs):
            advance("validate", 400)
            return super().get(*args, **kwargs)

    def parse(raw):
        assert raw == encoded_response
        advance("parse", 300)
        return ValidatedEnvelope(original_json(raw))

    class Response:
        def __enter__(self):
            return self

        def read(self, limit):
            assert limit == client.MAX_RESPONSE_BYTES + 1
            advance("read", 11)
            return encoded_response

        def __exit__(self, *args):
            advance("cleanup", 500)

    class Socket:
        def sendall(self, data):
            advance("send", 3)

    def open_response(wire, timeout, send_clock):
        assert wire.full_url == "http://localhost:8787/api/predict" and timeout == 15
        header_length = client.BINARY_PREAMBLE.unpack_from(wire.data)[2]
        header = json.loads(wire.data[client.BINARY_PREAMBLE.size:client.BINARY_PREAMBLE.size + header_length])
        assert header["roles"] == ["C", "D"] and header["poi_slot"] == 7
        assert header["prediction_frame"] == "last" and header["frame_indices"] == list(range(8))
        assert timing == {"caller_value": 42.}
        send_clock["http_started_ns"] = clock.ns
        advance("connect", 17)
        client._SendTimedSocket(Socket(), send_clock).sendall(wire.data)
        advance("server", 4)
        return Response()

    monkeypatch.setattr(client.time, "perf_counter_ns", lambda: clock.ns)
    monkeypatch.setattr(client, "encode_request", encode)
    monkeypatch.setattr(client, "Request", request)
    monkeypatch.setattr(client, "json_loads", parse)
    monkeypatch.setattr(client, "_open_with_send_timing", open_response)
    if entrypoint == "predict":
        result = client.predict("http://localhost:8787", observation(), request_id="timing", timeout=15, timing=timing)
    else:
        body, content_type = client.encode_request(observation(), request_id="timing")
        result = client.post_encoded("http://localhost:8787", body, content_type, "timing", 15, timing=timing)
    assert result == expected and "http_round_trip_ms" not in result
    assert timing == {"caller_value": 42., "http_round_trip_ms": 35., "client_send_to_receive_ms": 18.,
                      "client_send_ms": 3., "client_send_complete_to_receive_ms": 15.}
    assert events == ["pack", "construct", "connect", "send", "server", "read", "parse", "cleanup"] + ["validate"] * 3


@pytest.mark.parametrize("failure", ["connect", "read", "http_error", "encode"])
def test_failed_transport_does_not_leave_a_previous_successful_measurement(monkeypatch, failure):
    timing = {"http_round_trip_ms": 10., "client_send_to_receive_ms": 9., "client_send_ms": 1.,
              "client_send_complete_to_receive_ms": 8., "caller_value": 42.}
    calls = []

    class Response:
        def __enter__(self):
            return self

        def read(self, limit):
            raise TimeoutError("body read failed")

        def __exit__(self, *args):
            pass

    def urlopen(*args, **kwargs):
        calls.append("post")
        if failure == "connect":
            raise URLError("connection failed")
        if failure == "http_error":
            raise HTTPError("http://localhost/api/predict", 400, "bad request", {},
                            io.BytesIO(b'{"error":{"message":"bad request"}}'))
        return Response()

    monkeypatch.setattr(client, "_open_with_send_timing", urlopen)
    request = observation()
    if failure == "encode":
        request.images = {}
    with pytest.raises((OSError, client.PredictionError, ValueError)):
        client.predict("http://localhost", request, timing=timing)
    assert timing == {"caller_value": 42.}
    assert calls == ([] if failure == "encode" else ["post"])


def test_untimed_client_call_preserves_envelope_without_clock_overhead(monkeypatch):
    result = {"schema": client.SCHEMA, "request_id": "plain", "result": {"values": [1, 2]}}

    def no_clock():
        pytest.fail("ordinary client calls should not collect timing")

    monkeypatch.setattr(client.time, "perf_counter_ns", no_clock)
    monkeypatch.setattr(client, "urlopen", lambda *args, **kwargs: io.BytesIO(json.dumps(result).encode()))
    assert client.predict("http://localhost", observation(), request_id="plain") == result


def test_truncated_oversized_response_cannot_be_reported_as_a_complete_round_trip(monkeypatch):
    result = {"schema": client.SCHEMA, "request_id": "limit", "result": {}}
    prefix = json.dumps(result).encode()
    # The truncated prefix is valid JSON, so parsing alone cannot detect this.
    raw_response = prefix + b" " * 20
    monkeypatch.setattr(client, "MAX_RESPONSE_BYTES", len(prefix) + 1)
    monkeypatch.setattr(client, "_open_with_send_timing", lambda *args, **kwargs: io.BytesIO(raw_response))
    timing = {"http_round_trip_ms": 10., "client_send_to_receive_ms": 9., "client_send_ms": 1.,
              "client_send_complete_to_receive_ms": 8.}
    with pytest.raises(ValueError, match="response byte limit"):
        client.predict("http://localhost", observation(), request_id="limit", timing=timing)
    assert timing == {}
