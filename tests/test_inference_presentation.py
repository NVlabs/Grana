# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
import argparse
import json
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from grana.inference.presentation import prediction_chart_data, render_prediction


def fixture(roles=("C", "D"), mode="last"):
    slot = 5 if mode == "poi" else 7
    indices = (72, 88, 104, 112, 116, 120, 124, 128)
    request = SimpleNamespace(poi_slot=slot, frame_indices=indices,
                              images={"top": [np.full((30, 40, 3), i * 30, dtype=np.uint8) for i in range(8)]},
                              task="Assemble the car", compressed_task_prompt="Complete the car assembly",
                              stage_prompts=("Prepare parts", "Attach wheels", "Finish assembly"))
    result = {"roles": list(roles), "poi_slot": slot, "prediction_frame": mode,
              "frame_indices": list(indices), "progress": {}, "checkpoint": {"weights_sha256": "a" * 64}}
    if "C" in roles or "D" in roles:
        result["stage"] = {"ids": [0, 0, 0, 1, 1, 1, 2, 2]}
    for role in roles:
        if role != "C":
            result["progress"][role] = {"values": [i / 10 for i in range(8)]}
    if "D" in roles:
        focus = result["stage"]["ids"][slot]
        result["D_focus_stage"] = result["progress"]["D"]["focus_stage"] = focus
    return request, result


@pytest.mark.parametrize("roles", [("A",), ("C",), ("C", "D")])
@pytest.mark.parametrize("mode", ["poi", "last"])
def test_chart_labels_selected_frame_and_task_without_discarding_other_predictions(tmp_path, roles, mode):
    request, result = fixture(roles, mode)
    before = deepcopy(result)
    output = tmp_path / "prediction.png"
    chart = render_prediction(result, request, output)
    assert result == before
    assert chart["roles"] == list(roles)
    assert chart["prediction_frame"] == mode and chart["slot"] == request.poi_slot
    assert chart["frame_index"] == request.frame_indices[request.poi_slot]
    assert [series["role"] for series in chart["series"]] == list(roles)
    for series in chart["series"]:
        assert len(series["values"]) == 8
        assert series["selected_value"] == series["values"][request.poi_slot]
    with Image.open(output) as image:
        image.load()
        metadata = json.loads(image.info["grana_prediction"])
        assert metadata == chart
        assert f"slot {request.poi_slot}" in image.info["Description"]
        assert mode.upper() in image.info["Description"]
        assert image.width == 1280 and image.height > 600


def test_cannot_relabel_poi_focused_d_as_last_frame_without_rerunning():
    request, result = fixture(mode="poi")
    request.poi_slot = 7
    result.update(poi_slot=7, prediction_frame="last")
    with pytest.raises(ValueError, match="D focus"):
        prediction_chart_data(result, request)


def test_python_override_uses_response_selection_instead_of_original_request_default():
    request, result = fixture()
    request.poi_slot = 5
    chart = prediction_chart_data(result, request)
    assert chart["slot"] == 7 and chart["frame_index"] == 128
    assert chart["D_focus_stage"] == 2


def test_plot_protects_existing_output_and_rejects_missing_predictions(tmp_path):
    request, result = fixture(("A",))
    output = tmp_path / "prediction.png"
    output.write_bytes(b"existing work")
    with pytest.raises(FileExistsError):
        render_prediction(result, request, output)
    assert output.read_bytes() == b"existing work"
    result["progress"]["A"]["values"] = [.2] * 7
    with pytest.raises(ValueError, match="eight"):
        render_prediction(result, request, tmp_path / "bad.png")
    assert not (tmp_path / "bad.png").exists()


def test_missing_source_indices_are_explicit_not_invented(tmp_path):
    request, result = fixture(("C",))
    request.frame_indices = None
    result.pop("frame_indices")
    output = tmp_path / "unknown_source.png"
    render_prediction(result, request, output)
    with Image.open(output) as image:
        assert "source frame index not supplied" in image.info["Description"]


def test_single_stage_chart_does_not_accept_id_one():
    request, result = fixture(("C",))
    request.stage_prompts = ("Only operation",)
    result["stage"]["ids"] = [0] * 7 + [1]
    with pytest.raises(ValueError, match="Stage ID exceeds"):
        prediction_chart_data(result, request)
    result["stage"]["ids"] = [0] * 8
    assert prediction_chart_data(result, request)["series"][0]["selected_value"] == 0


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1, 1.1])
def test_invalid_progress_is_not_plotted(value):
    request, result = fixture(("A",))
    result["progress"]["A"]["values"][3] = value
    with pytest.raises(ValueError):
        prediction_chart_data(result, request)


@pytest.mark.parametrize("interface", ["local", "http"])
@pytest.mark.parametrize("roles", [("A",), ("C",), ("C", "D")])
@pytest.mark.parametrize("mode", ["poi", "last"])
def test_cli_plot_matches_actual_request_selection(tmp_path, monkeypatch, interface, roles, mode):
    from grana.inference import cli, client, predictor

    np.save(tmp_path / "video.npy", np.zeros((200, 3, 4, 3), dtype=np.uint8))
    source = tmp_path / "input.json"
    source.write_text(json.dumps({"views": {"top": "video.npy"}, "task": "Assemble the car",
                                 "compressed_task_prompt": "Complete assembly",
                                 "stage_prompts": ["Prepare", "Attach", "Finish"]}))
    output, plot = tmp_path / "result.json", tmp_path / "plot.png"
    calls = []

    def predict(actual, *, roles):
        assert actual.poi_slot == (5 if mode == "poi" else 7)
        assert actual.frame_indices[actual.poi_slot] == 128
        assert len(actual.images["top"]) == 8
        calls.append((tuple(roles), actual.poi_slot))
        _, result = fixture(roles, mode)
        result["frame_indices"] = list(actual.frame_indices)
        return result

    arguments = ["--input", str(source), "--output", str(output), "--plot", str(plot),
                 "--roles", *roles, "--prediction-frame", mode, "--frame-index", "128"]
    if interface == "local":
        class Predictor:
            def __init__(self, *args, **kwargs):
                pass

            def predict(self, request, roles):
                return predict(request, roles=roles)

        monkeypatch.setattr(predictor, "GranaPredictor", Predictor)
        parser = argparse.ArgumentParser()
        cli.add_arguments(parser)
        cli.run(parser.parse_args(["--checkpoint", "unused", *arguments]))
    else:
        monkeypatch.setattr(client, "predict", lambda server, request, **kwargs: {
            "result": predict(request, roles=kwargs["roles"]), "request_id": "test"})
        client.main(["--server", "http://localhost:9876", *arguments])
    assert calls == [(roles, 5 if mode == "poi" else 7)]
    result = json.loads(output.read_text())
    result = result["result"] if interface == "http" else result
    assert result["prediction_frame"] == mode
    with Image.open(plot) as image:
        metadata = json.loads(image.info["grana_prediction"])
        assert metadata["frame_index"] == 128
        assert metadata["prediction_frame"] == mode and metadata["roles"] == list(roles)


def test_plot_path_errors_fail_before_contacting_server(tmp_path, monkeypatch):
    from grana.inference import client

    monkeypatch.setattr(client, "predict", lambda *a, **k: pytest.fail("Should validate output before inference"))
    path = tmp_path / "same.png"
    with pytest.raises(ValueError, match="different paths"):
        client.main(["--server", "http://localhost:9876", "--input", "unused.json",
                     "--output", str(path), "--plot", str(path)])
