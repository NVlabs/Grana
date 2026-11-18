# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Execute the browser request path and verify frame/focus semantics on raw RGB."""
from copy import deepcopy
from dataclasses import replace
import json
import re
import time

import numpy as np
from PIL import Image
import pytest

from grana.inference.predictor import InferenceRequest
from grana.inference.server import decode_binary, split_binary
from scripts.prepare_review import (HTML, JS, argument_parser, preview, reference_snapshot,
                                    write_binary_example)
from test_serving import http, running  # noqa: F401 (shared HTTP server fixture)


VIEWS = ("top", "left", "right", "bottom")
INDICES = [0, 12, 24, 36, 48, 52, 56, 60]
ROLE_MODES = [(roles, mode) for roles in (("A",), ("C",), ("C", "D")) for mode in ("poi", "last")]
STAGES = [0, 0, 0, 0, 0, 0, 1, 1]  # C[5] != C[7]: switching the selected frame must change D's focus.
A_VALUES = [i / 10 for i in range(1, 9)]
D_VALUES = {0: [i / 100 for i in range(11, 19)], 1: [i / 100 for i in range(21, 29)]}


def request():
    return InferenceRequest(
        images={view: tuple(np.full((2, 3, 3), i * 16 + j, dtype=np.uint8) for j in range(8))
                for i, view in enumerate(VIEWS)},
        task="组装 hotwheels", compressed_task_prompt="准备好", stage_prompts=("开始", "完成"))


def assert_pixels(actual, original):
    for view in VIEWS:
        for got, expected in zip(actual.images[view], original.images[view], strict=True):
            np.testing.assert_array_equal(got, expected)


def test_prepared_example_defaults_to_cd_poi_with_no_labels(tmp_path):
    original = request()
    path = tmp_path / "example.bin"
    write_binary_example(path, original, "prepared-window-0", frame_indices=INDICES)
    payload = path.read_bytes()
    header, _ = split_binary(payload)
    actual, roles, request_id = decode_binary(payload, VIEWS)
    assert payload.startswith(b"GRNB")
    assert roles == ["C", "D"] and request_id == "prepared-window-0"
    assert actual.prediction_frame == "poi" and actual.poi_slot == 5
    assert actual.frame_indices == tuple(INDICES)
    assert actual.task == original.task and tuple(actual.stage_prompts) == original.stage_prompts
    assert set(header) == {"schema", "request_id", "roles", "views", "task", "compressed_task_prompt",
                           "stage_prompts", "prediction_frame", "frame_indices"}
    assert_pixels(actual, original)


@pytest.mark.parametrize("roles,mode", ROLE_MODES)
def test_preparation_accepts_each_role_and_prediction_frame(tmp_path, roles, mode):
    args = argument_parser().parse_args(["--comparison-manifest", "manifest.json", "--output", "site", "--arm", "model", "--episode", "episode",
                                         "--roles", *roles, "--prediction-frame", mode])
    assert args.roles == list(roles) and args.prediction_frame == mode
    path = tmp_path / "example.bin"
    write_binary_example(path, request(), "prepared", roles=args.roles,
                         prediction_frame=args.prediction_frame, frame_indices=INDICES)
    actual, encoded_roles, _ = decode_binary(path.read_bytes(), VIEWS)
    assert encoded_roles == list(roles)
    assert actual.poi_slot == (5 if mode == "poi" else 7) and actual.prediction_frame == mode
    assert_pixels(actual, request())


def test_preparation_cli_defaults_and_explicit_b(tmp_path):
    args = argument_parser().parse_args(["--comparison-manifest", "manifest.json", "--output", "site", "--arm", "model", "--episode", "episode"])
    assert args.roles == ["C", "D"] and args.prediction_frame == "poi"
    path = tmp_path / "example.bin"
    write_binary_example(path, replace(request(), focus_stage=1), "prepared", roles=("B",))
    actual, roles, _ = decode_binary(path.read_bytes(), VIEWS)
    assert roles == ["B"] and actual.focus_stage == 1


def js_context():
    # QuickJS is a test dependency only: production uses the browser's native JS engine.
    quickjs = pytest.importorskip("quickjs")
    context = quickjs.Context()
    context.add_callable("python_encode", lambda value: json.dumps(list(value.encode("utf-8"))))
    context.add_callable("python_decode", lambda value: bytes(json.loads(value)).decode("utf-8"))
    context.eval("""
        class TextEncoder { encode(s) { return Uint8Array.from(JSON.parse(python_encode(s))); } }
        class TextDecoder { decode(b) { return python_decode(JSON.stringify(Array.from(b))); } }
    """)
    return context


def finish_jobs(context):
    while context.execute_pending_job():
        pass


def test_browser_binary_request_reaches_http_without_altering_pixels(tmp_path, running):
    original = request()
    path = tmp_path / "example.bin"
    write_binary_example(path, original, "prepared-window-0")
    template = path.read_bytes()
    context = js_context()
    context.eval(JS[JS.index("function binaryWithRequestId("):JS.index("\nfunction selected()")])
    output = context.eval(
        "JSON.stringify(Array.from(binaryWithRequestId(Uint8Array.from("
        + json.dumps(list(template)) + ").buffer, 'browser-new-unique-id-2')))"
    )
    payload = bytes(json.loads(output))
    actual, roles, request_id = decode_binary(payload, VIEWS)
    assert request_id == "browser-new-unique-id-2" and roles == ["C", "D"]
    assert actual.poi_slot == 5 and actual.prediction_frame == "poi"
    assert_pixels(actual, original)
    assert path.read_bytes() == template
    server, predictor, audit, _ = running
    status, _, response = http(server, "POST", "/api/predict", payload,
                               {"Content-Type": "application/vnd.grana.binary-v1"})
    assert status == 200 and json.loads(response)["request_id"] == request_id
    assert_pixels(predictor.requests[0][0], original)
    assert predictor.requests[0][0].poi_slot == 5
    for _ in range(100):
        if audit.exists() and audit.read_text().strip():
            break
        time.sleep(0.01)
    event = json.loads(audit.read_text().splitlines()[-1])
    assert event["wire_format"] == "grana-binary-v1" and event["request_id"] == request_id


def result_for(roles, mode):
    slot = 5 if mode == "poi" else 7
    result = {"checkpoint": {"weights_sha256": "a" * 64}, "roles": list(roles), "poi_slot": slot,
              "prediction_frame": mode, "frame_indices": INDICES, "progress": {}}
    if "A" in roles:
        result["progress"]["A"] = {"values": A_VALUES, "poi_score": A_VALUES[slot]}
    if "B" in roles:
        result["progress"]["B"] = {"values": D_VALUES[1], "poi_score": D_VALUES[1][slot], "focus_stage": 1}
    if "C" in roles or "D" in roles:
        result["stage"] = {"ids": STAGES, "poi_stage_id": STAGES[slot]}
    if "D" in roles:
        focus = STAGES[slot]
        result["D_focus_stage"] = focus
        result["progress"]["D"] = {"values": D_VALUES[focus], "poi_score": D_VALUES[focus][slot], "focus_stage": focus}
    return result


def browser_config():
    return {
        "model_label": "4B", "weights_sha256": "a" * 64, "videos": [],
        "examples": [{"label": "window", "previews": {"poi": "/preview-poi.png", "last": "/preview-last.png"},
                      "payload": "/example.bin", "frame_indices": INDICES, "stage_count": 2,
                      "reference": reference_snapshot(result_for(("A", "C", "D"), "poi"))}] * 3,
    }


def run_browser(tmp_path, result, *, roles=None, mode=None, config=None, http_error=False):
    path = tmp_path / "example.bin"
    write_binary_example(path, replace(request(), focus_stage=1), "prepared")
    context = js_context()
    context.eval("const fixture=" + json.dumps({"config": config or browser_config(), "result": result,
                 "template": list(path.read_bytes()), "ids": re.findall(r'id="([^"]+)"', HTML),
                 "http_error": http_error}) + ";")
    context.eval("""
        class Node {
          constructor() { this.events={}; this.value=''; this.textContent=''; this.attributes={}; }
          addEventListener(name, fn) { this.events[name]=fn; }
          setAttribute(name, value) { this.attributes[name]=value; }
          add() {} append() {}
        }
        const elements=Object.fromEntries(fixture.ids.map(id=>[id,new Node()]));
        const document={getElementById(id) {
          if (!elements[id]) throw new Error('Missing DOM node: '+id);
          return elements[id];
        },createElement:()=>new Node()};
        const window={};
        const performance={now:()=>42};
        function Option() {}
        let postedBytes, postedHeader, postedContentType;
        async function fetch(url, options) {
          if (url==='/example.bin') return {ok:true,arrayBuffer:async()=>Uint8Array.from(fixture.template).buffer};
          let value;
          if (url==='/review.json') value=fixture.config;
          else if (url==='/healthz') value={checkpoint:{weights_sha256:fixture.config.weights_sha256},
                                          wire_formats:['grana-binary-v1']};
          else if (url==='/api/predict') {
            const bytes=options.body;
            const length=new DataView(bytes.buffer).getUint32(8,true);
            postedHeader=JSON.parse(new TextDecoder().decode(bytes.subarray(12,12+length)));
            postedBytes=Array.from(bytes); postedContentType=options.headers['Content-Type'];
            if (fixture.http_error) return {ok:false,status:500,json:async()=>({error:'inference failed'})};
            value={request_id:postedHeader.request_id,result:fixture.result,elapsed_seconds:.15};
          } else throw new Error('Unexpected URL: '+url);
          return {ok:true,json:async()=>value};
        }
    """)
    context.eval(JS)
    finish_jobs(context)
    assert context.eval("elements.predict.disabled") is False
    if roles is not None:
        context.eval("elements.roles.value=" + json.dumps(" ".join(roles)))
    if mode is not None:
        context.eval("elements['prediction-frame'].value=" + json.dumps(mode))
    context.eval("elements.roles.events.change()")
    context.eval("elements.predict.events.click()")
    finish_jobs(context)
    return context


@pytest.mark.parametrize("roles,mode", ROLE_MODES)
def test_browser_six_role_frame_combinations_preserve_rgb_and_select_correct_outputs(tmp_path, roles, mode):
    result = result_for(roles, mode)
    context = run_browser(tmp_path, result, roles=roles, mode=mode)
    reply = json.loads(context.eval("JSON.stringify(window.lastGranaReply)"))
    assert reply is not None, context.eval("elements.status.textContent")
    slot = 5 if mode == "poi" else 7
    assert reply["roles"] == list(roles) and reply["prediction_frame"] == mode and reply["poi_slot"] == slot
    assert reply["frame_index"] == INDICES[slot]
    payload = bytes(json.loads(context.eval("JSON.stringify(postedBytes)")))
    actual, actual_roles, _ = decode_binary(payload, VIEWS)
    assert actual_roles == list(roles) and actual.prediction_frame == mode and actual.poi_slot == slot
    assert actual.frame_indices == tuple(INDICES)
    assert_pixels(actual, request())
    original = bytes(json.loads(context.eval("JSON.stringify(fixture.template)")))
    assert bytes(split_binary(payload)[1]) == bytes(split_binary(original)[1])
    assert context.eval("postedContentType") == "application/vnd.grana.binary-v1"
    for role in ("A", "B", "C", "D"):
        assert context.eval(f"elements['card-{role.lower()}'].hidden") is (role not in roles)
        if role in roles:
            expected = str(STAGES[slot]) if role == "C" else f"{result['progress'][role]['values'][slot]:.4f}"
            actual_text = context.eval(f"elements.{role.lower()}.textContent")
            assert actual_text.startswith(expected)
            if role == "D":
                assert actual_text.endswith(f"(focus {STAGES[slot]})")
    label = context.eval("elements['selection-label'].textContent")
    assert " + ".join(roles) in label and mode.upper() in label
    assert f"slot {slot}" in label and f"source frame {INDICES[slot]}" in label
    assert context.eval("elements.observations.src") == f"/preview-{mode}.png"
    assert f"slot {slot}" in context.eval("elements.observations.alt")
    plot = context.eval("elements['prediction-plot'].innerHTML")
    assert plot.count("<circle ") == 8 * len(roles)
    assert f'data-selected-slot="{slot}"' in plot
    assert context.eval("elements['plot-container'].hidden") is False
    assert json.loads(context.eval("elements.response.textContent"))["result"] == result
    if mode == "poi":
        assert reply["comparable"] is True
        assert reply["maxDiff"] == (0 if any(r in roles for r in ("A", "D")) else None)
        assert reply["stageMatch"] is (True if "C" in roles else None)
        assert reply["focusMatch"] is (True if "D" in roles else None)
    else:
        assert reply["comparable"] is False and reply["maxDiff"] is None
        assert "Not comparable" in context.eval("elements.parity.textContent")
        if "D" in roles:
            assert context.eval("elements.d.textContent").startswith("0.2800")
            assert D_VALUES[0][7] != D_VALUES[1][7]  # Old POI-conditioned D[-1] would be wrong.
    if roles == ("A",):
        assert "stage" not in result and "D" not in result["progress"]
    if roles == ("C",):
        assert result["progress"] == {}


def test_browser_default_and_custom_b_selection(tmp_path):
    context = run_browser(tmp_path, result_for(("C", "D"), "poi"))
    assert context.eval("postedHeader.prediction_frame") == "poi"
    assert json.loads(context.eval("JSON.stringify(postedHeader.roles)")) == ["C", "D"]
    config = browser_config()
    config.update(default_roles=["B"], default_prediction_frame="poi")
    context = run_browser(tmp_path, result_for(("B",), "poi"), config=config)
    assert context.eval("elements.b.textContent") == "0.2600 (focus 1)"
    assert context.eval("elements['card-c'].hidden") is True
    assert "B not comparable" in context.eval("elements.parity.textContent")


def test_browser_refuses_to_compare_d_under_a_different_focus(tmp_path):
    config = deepcopy(browser_config())
    for example in config["examples"]:
        example["reference"]["D_focus_stage"] = 1
        example["reference"]["progress"]["D"] = D_VALUES[1]
    context = run_browser(tmp_path, result_for(("C", "D"), "poi"), mode="poi", config=config)
    reply = json.loads(context.eval("JSON.stringify(window.lastGranaReply)"))
    assert reply["comparable"] is False and reply["focusMatch"] is False
    assert reply["maxDiff"] is None and reply["stageMatch"] is True
    assert "D not comparable" in context.eval("elements.parity.textContent")
    assert context.eval("elements.d.textContent") == "0.1600 (focus 0)"


@pytest.mark.parametrize("failure", ["http", "old_last_response", "wrong_d_focus"])
def test_browser_never_uses_saved_outputs_on_failed_or_stale_requests(tmp_path, failure):
    result = result_for(("C", "D"), "last" if failure == "old_last_response" else "poi")
    if failure == "wrong_d_focus":
        result["D_focus_stage"] = 1
    context = run_browser(tmp_path, result, http_error=failure == "http")
    assert context.eval("window.lastGranaReply") is None
    assert context.eval("elements.c.textContent") == "—"
    assert context.eval("elements.d.textContent") == "—"
    assert context.eval("elements['plot-container'].hidden") is True
    assert "Request failed" in context.eval("elements.status.textContent")
    assert "No saved prediction was used" in context.eval("elements.parity.textContent")


@pytest.mark.parametrize("slot", [5, 7])
def test_preview_highlights_only_selected_column(tmp_path, slot):
    path = tmp_path / "preview.png"
    preview(request().images, INDICES, path, selected_slot=slot)
    with Image.open(path) as image:
        for index in range(8):
            border = image.getpixel((80 + index * 160, 35))
            assert (border == (243, 208, 126)) is (index == slot)


def test_saved_reference_and_video_labels_remain_offline_poi():
    original = result_for(("A", "C", "D"), "poi")
    snapshot = reference_snapshot(original)
    assert snapshot["poi_slot"] == 5 and snapshot["prediction_frame"] == "poi"
    assert snapshot["D_focus_stage"] == 0 and snapshot["progress"]["D"] == D_VALUES[0]
    assert "Saved offline POI comparison videos" in HTML
    assert "Their measurements are unchanged" in HTML
