# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest

from grana.inference.cli import load_request


def test_file_request_preserves_slot_order_and_repeated_frames(tmp_path):
    frames = np.zeros((20, 2, 3, 3), dtype=np.uint8)
    frames[:, :, :, :] = np.arange(20, dtype=np.uint8)[:, None, None, None]
    np.save(tmp_path / "top.npy", frames)
    indices = [6, 7, 8, 1, 5, 9, 13, 13]
    path = tmp_path / "request.json"
    path.write_text(json.dumps({"views": {"top": "top.npy"}, "frame_indices": indices,
                                "compressed_task_prompt": "Move the block."}))
    request = load_request(path)
    assert [int(frame[0, 0, 0]) for frame in request.images["top"]] == indices
    assert request.frame_indices == tuple(indices)
    assert request.task is None and request.focus_stage is None


def test_rgb_image_lists_need_no_cache_or_labels(tmp_path):
    Image.new("RGB", (4, 3), (10, 20, 30)).save(tmp_path / "frame.png")
    path = tmp_path / "request.json"
    path.write_text(json.dumps({"views": {"top": ["frame.png"] * 8},
                                "task": "Move block", "stage_prompts": ["Pick up", "Put down"]}))
    request = load_request(path)
    assert len(request.images["top"]) == 8
    assert request.images["top"][0].getpixel((0, 0)) == (10, 20, 30)


def test_file_request_selects_explicit_query_without_changing_frame_order(tmp_path):
    frames = np.arange(8, dtype=np.uint8)[:, None, None, None] * np.ones((8, 2, 3, 3), dtype=np.uint8)
    np.save(tmp_path / "frames.npy", frames)
    path = tmp_path / "request.json"
    path.write_text(json.dumps({"views": {"top": "frames.npy"}, "frame_indices": list(reversed(range(8))), "poi_slot": 2}))
    request = load_request(path)
    assert request.poi_slot == 2
    assert [int(frame[0, 0, 0]) for frame in request.images["top"]] == list(reversed(range(8)))


@pytest.mark.parametrize("selection,expected", [({}, 5), ({"prediction_frame": "poi"}, 5),
                                              ({"prediction_frame": "last"}, 7), ({"poi_slot": 2}, 2),
                                              ({"prediction_frame": "last", "poi_slot": 7}, 7)])
def test_file_request_selection_preserves_python_default_and_explicit_json(tmp_path, selection, expected):
    np.save(tmp_path / "frames.npy", np.zeros((8, 2, 3, 3), dtype=np.uint8))
    path = tmp_path / "request.json"
    path.write_text(json.dumps({"views": {"top": "frames.npy"}, **selection}))
    assert load_request(path).poi_slot == expected
    assert load_request(path, default_prediction_frame="last").poi_slot == (expected if selection else 7)
    assert load_request(path, prediction_frame="last").poi_slot == 7
    assert load_request(path, poi_slot=2).poi_slot == 2


@pytest.mark.parametrize("selection", [{"prediction_frame": "first"}, {"prediction_frame": None},
                                      {"prediction_frame": []}, {"prediction_frame": False},
                                      {"prediction_frame": "last", "poi_slot": 5},
                                      {"prediction_frame": "poi", "poi_slot": 7}])
def test_invalid_named_request_selection_fails_before_file_io(tmp_path, selection):
    path = tmp_path / "request.json"
    path.write_text(json.dumps({"views": {"top": "missing.npy"}, **selection}))
    with pytest.raises(ValueError, match="prediction_frame"):
        load_request(path)


@pytest.mark.parametrize("mode,expected", [
    ("poi", (52, 68, 84, 92, 96, 100, 104, 108)),
    ("last", (44, 60, 76, 84, 88, 92, 96, 100)),
])
def test_frame_index_composes_the_training_window_without_reordering(tmp_path, mode, expected):
    frames = np.arange(120, dtype=np.uint8)[:, None, None, None] * np.ones((120, 2, 3, 3), dtype=np.uint8)
    np.save(tmp_path / "frames.npy", frames)
    path = tmp_path / "request.json"
    path.write_text(json.dumps({"views": {"top": "frames.npy"}}))
    sample = load_request(path, frame_index=100, prediction_frame=mode)
    assert sample.frame_indices == expected
    assert [int(image[0, 0, 0]) for image in sample.images["top"]] == list(expected)
    assert sample.frame_indices[sample.poi_slot] == 100
    if mode == "last":
        assert max(sample.frame_indices) <= 100


@pytest.mark.parametrize("raw,options,match", [
    ({"frame_indices": list(range(8))}, {}, "cannot be combined"),
    ({"poi_slot": 2}, {}, "requires prediction_frame"),
    ({}, {"poi_slot": 2}, "requires prediction_frame"),
    ({"views": {"top": ["missing.png"] * 8}}, {}, "requires .npy"),
])
def test_frame_index_rejects_ambiguous_input_composition_before_file_io(tmp_path, raw, options, match):
    path = tmp_path / "request.json"
    path.write_text(json.dumps({"views": {"top": "missing.npy"}, **raw}))
    with pytest.raises(ValueError, match=match):
        load_request(path, frame_index=4, **options)


@pytest.mark.parametrize("frame_index", [-1, True, 1.5])
def test_invalid_frame_index_fails_before_file_io(tmp_path, frame_index):
    path = tmp_path / "request.json"
    path.write_text(json.dumps({"views": {"top": "missing.npy"}}))
    with pytest.raises(ValueError, match="frame_index"):
        load_request(path, frame_index=frame_index)


@pytest.mark.parametrize("slot", [True, -1, 8, 2.0, "2", None])
def test_invalid_query_slot_fails_before_file_io(tmp_path, slot):
    path = tmp_path / "request.json"
    path.write_text(json.dumps({"views": {"top": "missing.npy"}, "poi_slot": slot}))
    with pytest.raises(ValueError, match="poi_slot"):
        load_request(path)


@pytest.mark.parametrize("raw,match", [
    ('{"views":{"left":"missing.npy","left":"other.npy"}}', "Duplicate JSON key: left"),
    ('{"views":{"left":"missing.npy"},"views":{"right":"other.npy"}}', "Duplicate JSON key: views"),
    ('{"views":{"left":"missing.npy"},"focus_stage":NaN}', "Nonfinite JSON number"),
    ('{"views":{"left":"missing.npy"},"focus_stage":Infinity}', "Nonfinite JSON number"),
    ('{"views":{"left":"missing.npy"},"focus_stage":-Infinity}', "Nonfinite JSON number"),
    ('{"views":{"left":"missing.npy"," left ":"other.npy"}}', "whitespace"),
])
def test_rejects_ambiguous_json_or_camera_names_before_reading_images(tmp_path, raw, match):
    path = tmp_path / "request.json"
    path.write_text(raw)
    with pytest.raises(ValueError, match=match):
        load_request(path)


def test_file_request_preserves_supplied_camera_order(tmp_path):
    np.save(tmp_path / "frames.npy", np.zeros((8, 2, 3, 3), dtype=np.uint8))
    path = tmp_path / "request.json"
    path.write_text('{"views":{"wrist":"frames.npy","front":"frames.npy"}}')
    assert tuple(load_request(path).images) == ("wrist", "front")


@pytest.mark.parametrize("override", [
    {"targets": [0] * 8}, {"frame_indices": [True] * 8},
    {"frame_indices": [0] * 7}, {"frame_indices": [8] * 8},
])
def test_rejects_label_fields_and_invalid_indices(tmp_path, override):
    np.save(tmp_path / "top.npy", np.zeros((8, 2, 3, 3), dtype=np.uint8))
    path = tmp_path / "request.json"
    path.write_text(json.dumps({"views": {"top": "top.npy"}, **override}))
    with pytest.raises(ValueError):
        load_request(path)


@pytest.mark.parametrize("shape,dtype", [((8, 2, 3, 4), np.uint8), ((8, 2, 3, 3), np.float32)])
def test_rejects_non_rgb_or_non_uint8_arrays(tmp_path, shape, dtype):
    np.save(tmp_path / "top.npy", np.zeros(shape, dtype=dtype))
    path = tmp_path / "request.json"
    path.write_text(json.dumps({"views": {"top": "top.npy"}}))
    with pytest.raises(ValueError, match="RGB uint8"):
        load_request(path)


def test_infer_help_does_not_require_training_config(monkeypatch, capsys):
    from grana.cli import main
    monkeypatch.setattr("sys.argv", ["grana", "infer", "--help"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 0
    help_text = capsys.readouterr().out
    assert "--checkpoint" in help_text and "--view-policy" in help_text


def test_view_policy_cli_defaults_to_checkpoint_and_requires_explicit_request_mode():
    from argparse import ArgumentParser
    from grana.inference.cli import add_arguments
    parser = ArgumentParser()
    add_arguments(parser)
    required = ["--checkpoint", "model", "--input", "request.json"]
    assert parser.parse_args(required).view_policy == "checkpoint"
    assert parser.parse_args([*required, "--view-policy", "request"]).view_policy == "request"


def runtime_parser():
    from argparse import ArgumentParser
    from grana.inference.cli import add_arguments
    parser = ArgumentParser()
    add_arguments(parser)
    return parser, ["--checkpoint", "model", "--input", "request.json"]


REMOVED_RUNTIME_FLAGS = [
    f"--{prefix}{feature}"
    for feature in ("optimize", "batched-vision-sdpa", "reuse-images", "merge-lora",
                    "cpu-position-ids", "compile", "compile-decoder", "compile-vision",
                    "causal-conv1d", "fla-output-guard", "text-input-cache",
                    "gpu-image-preprocessing", "gpu-image-resize")
    for prefix in ("", "no-")
]


def test_predictor_has_only_checkpoint_request_and_device_configuration():
    import inspect
    from grana.inference.predictor import GranaPredictor
    assert tuple(inspect.signature(GranaPredictor).parameters) == (
        "checkpoint", "model_name_or_path", "device", "max_pixels", "view_policy")


def test_inference_defaults_to_stage_then_predicted_focus_progress():
    import inspect
    from grana.inference.predictor import GranaPredictor
    parser, required = runtime_parser()
    assert parser.parse_args(required).roles == ["C", "D"]
    assert inspect.signature(GranaPredictor.predict).parameters["roles"].default == ("C", "D")
    assert parser.parse_args([*required, "--roles", "A", "B"]).roles == ["A", "B"]


@pytest.mark.parametrize("flags", [["--prediction-frame", "first"],
                                   ["--prediction-frame", "last", "--poi-slot", "5"]])
def test_both_cli_parsers_reject_invalid_or_ambiguous_selection(tmp_path, flags):
    from grana.inference import client
    parser, required = runtime_parser()
    with pytest.raises(SystemExit) as local:
        parser.parse_args([*required, *flags])
    assert local.value.code == 2
    with pytest.raises(SystemExit) as remote:
        client.main(["--server", "http://127.0.0.1:1", "--input", "missing.json",
                     "--output", str(tmp_path / "output.json"), *flags])
    assert remote.value.code == 2


@pytest.mark.parametrize("roles", [["A"], ["C"], ["C", "D"]])
@pytest.mark.parametrize("selection,flags,slot", [({}, [], 5), ({"prediction_frame": "poi"}, [], 5),
                                                ({"prediction_frame": "last"}, [], 7),
                                                ({"poi_slot": 5}, ["--prediction-frame", "last"], 7),
                                                ({"prediction_frame": "last"}, ["--poi-slot", "2"], 2)])
def test_local_cli_passes_selected_roles_and_resolves_frame_precedence(tmp_path, monkeypatch, roles, selection, flags, slot):
    from types import SimpleNamespace
    from grana.inference import cli, predictor
    np.save(tmp_path / "frames.npy", np.zeros((8, 2, 3, 3), dtype=np.uint8))
    path = tmp_path / "request.json"
    path.write_text(json.dumps({"views": {"top": "frames.npy"}, **selection}))
    output = tmp_path / "output.json"
    seen = []

    def predict(request, roles):
        seen.append((request, roles))
        return {"poi_slot": request.poi_slot, "roles": roles}

    monkeypatch.setattr(predictor, "GranaPredictor", lambda *args, **kwargs: SimpleNamespace(predict=predict))
    parser, _ = runtime_parser()
    args = parser.parse_args(["--checkpoint", "model", "--input", str(path), "--output", str(output),
                              "--roles", *roles, *flags])
    cli.run(args)
    assert seen[0][0].poi_slot == slot and seen[0][1] == roles
    assert json.loads(output.read_text()) == {"poi_slot": slot, "roles": roles}


@pytest.mark.parametrize("flags", [[flag] for flag in REMOVED_RUNTIME_FLAGS] + [
    ["--delta-rule-backend", "torch"], ["--attn-implementation", "eager"],
])
def test_runtime_overrides_are_rejected_by_cli_server_and_comparison(flags, tmp_path, monkeypatch, capsys):
    from grana.inference import server
    from scripts import compare_inference
    parser, required = runtime_parser()
    with pytest.raises(SystemExit) as cli_exit:
        parser.parse_args([*required, *flags])
    assert cli_exit.value.code == 2
    assert f"unrecognized arguments: {' '.join(flags)}" in capsys.readouterr().err
    with pytest.raises(SystemExit) as server_exit:
        server.main(["--checkpoint", str(tmp_path), "--audit-log", str(tmp_path / "audit.jsonl"), *flags])
    assert server_exit.value.code == 2
    assert f"unrecognized arguments: {' '.join(flags)}" in capsys.readouterr().err
    monkeypatch.setattr(sys, "argv", ["compare_inference.py", "--run", "--manifest",
                                     str(tmp_path / "manifest.json"), *flags])
    with pytest.raises(SystemExit) as comparison_exit:
        compare_inference.main()
    assert comparison_exit.value.code == 2
    assert f"unrecognized arguments: {' '.join(flags)}" in capsys.readouterr().err


@pytest.mark.parametrize("flags, expected", [
    ([], dict(model_name_or_path=None, device=None, max_pixels=None, view_policy="checkpoint")),
    (["--model-name-or-path", "local/model", "--device", "cuda:2", "--max-pixels", "25600",
      "--view-policy", "request"],
     dict(model_name_or_path="local/model", device="cuda:2", max_pixels=25600, view_policy="request")),
])
def test_server_passes_only_the_supported_predictor_configuration(monkeypatch, tmp_path, flags, expected):
    from grana.inference import server
    seen = {}

    class Stop(Exception):
        pass

    def predictor(checkpoint, **kwargs):
        seen.update(kwargs)
        raise Stop

    monkeypatch.setattr(server, "GranaPredictor", predictor)
    with pytest.raises(Stop):
        server.main(["--checkpoint", str(tmp_path), "--audit-log", str(tmp_path / "audit.jsonl"), *flags])
    assert seen == expected


def test_comparison_script_uses_the_fixed_serving_runtime(monkeypatch, tmp_path):
    from scripts import compare_inference
    seen = {}
    monkeypatch.setattr(compare_inference, "run", lambda args: seen.update(vars(args)))
    monkeypatch.setattr(sys, "argv", ["compare_inference.py", "--run", "--manifest",
                                     str(tmp_path / "manifest.json")])
    compare_inference.main()
    assert seen["device"] == "cuda:0"
    for flag in REMOVED_RUNTIME_FLAGS:
        assert flag[2:].replace("-", "_") not in seen
    assert "delta_rule_backend" not in seen and "attn_implementation" not in seen


def test_optimized_server_wrapper_only_selects_checkout_and_python(tmp_path):
    # Capture arguments and relevant environment without loading a model or CUDA.
    # Isolate the shell contract from site initialization: installed FLA profiles
    # intentionally rewrite these variables when a normal Python process starts.
    executable = tmp_path / "capture-python"
    executable.write_text(
        f"#!{sys.executable} -S\nimport json, os, sys\n"
        "print(json.dumps({'args': sys.argv[1:], 'profile': {key: os.environ.get(key) "
        "for key in ('FLA_TILELANG', 'FLA_DISABLE_BACKEND_DISPATCH', 'TORCH_CUDNN_SDPA_DEPRIORITIZED')}}))\n")
    executable.chmod(0o755)
    wrapper = Path(__file__).resolve().parents[1] / "scripts/inference/serve_optimized.sh"
    profile = {"FLA_TILELANG": "1", "FLA_DISABLE_BACKEND_DISPATCH": "0", "TORCH_CUDNN_SDPA_DEPRIORITIZED": "1"}
    args = ["--checkpoint", "model", "--device", "cuda:2", "--audit-log", "audit.jsonl"]
    result = subprocess.run(["bash", str(wrapper), *args], check=True, capture_output=True, text=True,
                            env={**os.environ, **profile, "PYTHON_BIN": str(executable)})
    command = json.loads(result.stdout)
    assert command["args"] == ["-P", "-m", "grana.inference.server", *args]
    assert command["profile"] == profile


def test_optimized_server_wrapper_uses_its_checkout_from_foreign_cwd(tmp_path):
    root = Path(__file__).resolve().parents[1]
    foreign = tmp_path / "other-checkout"
    (foreign / "grana").mkdir(parents=True)
    (foreign / "grana/__init__.py").write_text("raise RuntimeError('imported the wrong checkout')\n")
    probe = tmp_path / "probe"
    probe.mkdir()
    recorded = tmp_path / "import-source.json"
    # Record the module Python actually loaded, without importing or replacing
    # Grana ourselves. --help exits before model construction or path loading.
    (probe / "sitecustomize.py").write_text(
        "import atexit, json, os, sys\n"
        "def record():\n"
        "    module = sys.modules.get('grana')\n"
        "    with open(os.environ['GRANA_IMPORT_PROBE'], 'w') as stream:\n"
        "        json.dump({'source': getattr(module, '__file__', None), 'cwd': os.getcwd()}, stream)\n"
        "atexit.register(record)\n")
    environment = {**os.environ, "PYTHON_BIN": sys.executable, "PYTHONPATH": str(probe),
                   "GRANA_IMPORT_PROBE": str(recorded), "CUDA_VISIBLE_DEVICES": "",
                   "OMP_NUM_THREADS": "2", "PYTHONDONTWRITEBYTECODE": "1"}
    environment.pop("PYTHONSAFEPATH", None)  # The launcher itself must select safe imports.
    result = subprocess.run(["bash", str(root / "scripts/inference/serve_optimized.sh"), "--help"],
                            cwd=foreign, env=environment, check=True, capture_output=True, text=True, timeout=60)
    assert "--checkpoint" in result.stdout
    assert "--fla-output-guard" not in result.stdout
    assert json.loads(recorded.read_text()) == {
        "source": str(root / "grana/__init__.py"), "cwd": str(foreign)}
