# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recorded HTTP replay must display the selected frame and its actual D focus."""
from copy import deepcopy
import json
import shutil

import numpy as np
import pytest

from scripts.inference.render_http_replay import (HEIGHT, WIDTH, curve_values, load_inputs, main,
                                                  render_frame, sha256_file, validate_case, validate_record)


VIEWS = ["top", "left", "right", "bottom"]
COMBINATIONS = [(roles, mode) for roles in (["A"], ["C"], ["C", "D"]) for mode in ("poi", "last")]


def manifest():
    return {"schema": "grana-http-replay-v1", "model_label": "HotWheels D100 / Qwen3.5-4B",
            "checkpoint": {"path": "/checkpoint-001000", "weights_sha256": "a" * 64, "step": 1000},
            "sampling": {"context_stride": 16, "tail_stride": 4, "source_fps": 10, "target_frames": [20, 40]},
            "episode": {"task": "Assemble the HotWheels track", "trajectory_id": "heldout-001",
                        "view_names": VIEWS, "num_frames": 64, "stage_prompts": ["Start", "Build", "Finish"],
                        "compressed_task_prompt": "Build the track", "frame_paths": {}}, "cases": []}


def case_for(roles, mode):
    return {"name": "".join(roles).lower() + "_" + mode, "roles": roles,
            "prediction_frame": mode, "slot": 5 if mode == "poi" else 7, "records_path": "/records.json"}


def record_for(roles, mode, frame=20, *, gt_stage=1):
    slot = 5 if mode == "poi" else 7
    offsets = [-24, -16, -12, -8, -4, 0, 4, 8] if mode == "poi" else [-32, -24, -20, -16, -12, -8, -4, 0]
    indices = [max(0, frame + offset) for offset in offsets]
    stages = [0, 0, 0, 0, 0, 0, 1, 1]  # POI focus 0, LAST focus 1.
    progress = [i / 10 for i in range(1, 9)]
    prediction = {"checkpoint": {"weights_sha256": "a" * 64}, "roles": roles,
                  "poi_slot": slot, "prediction_frame": mode, "frame_indices": indices, "progress": {}}
    if "A" in roles:
        prediction["progress"]["A"] = {"values": progress, "poi_score": progress[slot]}
    if "C" in roles:
        prediction["stage"] = {"ids": stages, "poi_stage_id": stages[slot]}
    if "D" in roles:
        values = [i / 100 + stages[slot] * 0.1 for i in range(1, 9)]
        prediction["D_focus_stage"] = stages[slot]
        prediction["progress"]["D"] = {"values": values, "poi_score": values[slot], "focus_stage": stages[slot]}
    return {"frame_index": frame, "frame_indices": indices, "slot": slot, "request_id": f"request-{frame}",
            "client_total_ms": 170.5,
            "response": {"schema": "grana.inference.v1", "request_id": f"request-{frame}", "elapsed_seconds": .16,
                         "result": prediction},
            "gt": {"whole_progress": .6, "stage_id": gt_stage, "within_stage_progress": .4}}


@pytest.mark.parametrize("roles,mode", COMBINATIONS)
def test_six_modes_select_raw_slot_with_partial_roles(roles, mode):
    record = record_for(roles, mode)
    actual = validate_record(manifest(), case_for(roles, mode), record)
    assert actual["slot"] == (5 if mode == "poi" else 7)
    assert actual["frame_index"] == actual["frame_indices"][actual["slot"]]
    if roles == ["A"]:
        assert actual["values"] == {"A": .6 if mode == "poi" else .8}
        assert curve_values(actual, roles) == (.6, actual["values"]["A"])
        assert "stage" not in record["response"]["result"]
    elif roles == ["C"]:
        assert actual["values"] == {"C": 0 if mode == "poi" else 1}
        assert curve_values(actual, roles) == (1, actual["values"]["C"])
        assert record["response"]["result"]["progress"] == {}
    else:
        assert actual["D_focus_stage"] == actual["values"]["C"] == (0 if mode == "poi" else 1)
        assert actual["values"]["D"] == pytest.approx(.06 if mode == "poi" else .18)
        assert actual["D_focus_target"] == (1.0 if mode == "poi" else .4)
        assert curve_values(actual, roles) == pytest.approx((1.4, .06 if mode == "poi" else 1.18))


@pytest.mark.parametrize("gt_stage,expected", [(0, 0), (1, .4), (2, 1)])
def test_d_truth_is_relative_to_the_actual_selected_focus(gt_stage, expected):
    actual = validate_record(manifest(), case_for(["C", "D"], "last"), record_for(["C", "D"], "last", gt_stage=gt_stage))
    assert actual["D_focus_stage"] == 1
    assert actual["D_focus_target"] == expected


@pytest.mark.parametrize("mutation,message", [
    (lambda r: r.update(frame_index=19), "Selected source frame"),
    (lambda r: r.update(slot=5), "Selected source frame"),
    (lambda r: r["response"]["result"].update(poi_slot=5), "prediction frame"),
    (lambda r: r["response"]["result"].update(prediction_frame="poi"), "prediction frame"),
    (lambda r: r["response"]["result"].update(frame_indices=[0] * 8), "source frame indices"),
    (lambda r: r["response"]["result"].update(D_focus_stage=0), "D focus"),
    (lambda r: r["response"]["result"]["progress"]["D"].update(focus_stage=0), "D focus"),
    (lambda r: r["response"]["result"]["progress"]["D"].update(values=[.1] * 7), "eight raw D"),
    (lambda r: r["response"]["result"]["progress"]["D"].update(poi_score=.7), "POI summary"),
    (lambda r: r["response"]["result"].update(roles=["C"]), "roles"),
    (lambda r: r["response"].update(request_id="other-request"), "identity mismatch"),
    (lambda r: r["response"]["result"]["checkpoint"].update(weights_sha256="b" * 64), "checkpoint digest"),
])
def test_rejects_stale_or_inconsistent_http_results(mutation, message):
    record = record_for(["C", "D"], "last")
    mutation(record)
    with pytest.raises(ValueError, match=message):
        validate_record(manifest(), case_for(["C", "D"], "last"), record)


def test_last_rejects_future_observations_in_any_input_slot():
    record = record_for(["A"], "last")
    record["frame_indices"][0] = 21
    with pytest.raises(ValueError, match="future frames"):
        validate_record(manifest(), case_for(["A"], "last"), record)
    # POI explicitly permits its later observations; preserving input order matters.
    poi = record_for(["A"], "poi")
    assert max(poi["frame_indices"]) > poi["frame_index"]
    validate_record(manifest(), case_for(["A"], "poi"), poi)


def test_selected_summary_cannot_reuse_an_old_frame_or_focus():
    record = record_for(["C", "D"], "last")
    record["response"]["result"]["selected_prediction"] = {
        "slot": 7, "frame_index": 20, "stage_id": 1, "progress": {"D": .18}, "focus_stage": {"D": 0}}
    with pytest.raises(ValueError, match="different focus"):
        validate_record(manifest(), case_for(["C", "D"], "last"), record)
    record["response"]["result"]["selected_prediction"]["focus_stage"]["D"] = 1
    validate_record(manifest(), case_for(["C", "D"], "last"), record)


def test_case_must_cover_exact_frozen_samples_and_mode():
    case = case_for(["C"], "last")
    payload = {"roles": ["C"], "prediction_frame": "last", "records": [record_for(["C"], "last")]}
    with pytest.raises(ValueError, match="coverage"):
        validate_case(manifest(), case, payload)
    payload["records"].append(record_for(["C"], "last", frame=40))
    assert len(validate_case(manifest(), case, payload)) == 2
    payload["prediction_frame"] = "poi"
    with pytest.raises(ValueError, match="prediction frame"):
        validate_case(manifest(), case, payload)


def test_render_only_reveals_values_through_current_sample():
    frozen, case = manifest(), case_for(["A"], "last")
    records = [validate_record(frozen, case, record_for(["A"], "last", frame)) for frame in (20, 40)]
    arrays = {view: np.zeros((64, 4, 6, 3), dtype=np.uint8) for view in VIEWS}
    first = render_frame(frozen, case, records, arrays, 0)
    changed = deepcopy(records)
    changed[1]["values"]["A"] = .01
    changed[1]["gt"]["whole_progress"] = .99
    assert first.size == (WIDTH, HEIGHT)
    assert first.tobytes() == render_frame(frozen, case, changed, arrays, 0).tobytes()
    assert render_frame(frozen, case, records, arrays, 1).tobytes() != render_frame(frozen, case, changed, arrays, 1).tobytes()


def test_cli_encodes_and_full_decodes_six_cases_with_hash_receipt(tmp_path):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("CPU ffmpeg and ffprobe are required for the encoding integration test")
    frozen = manifest()
    frozen["A_accuracy_note"] = "A accuracy is unqualified for this checkpoint: its HotWheels training recipe used zero A examples."
    frozen["sampling"]["target_frames"] = [20]
    for index, view in enumerate(VIEWS):
        path = tmp_path / f"{view}.npy"
        np.save(path, np.full((64, 4, 6, 3), index * 32, dtype=np.uint8))
        frozen["episode"]["frame_paths"][view] = str(path)
    for roles, mode in COMBINATIONS:
        case = case_for(roles, mode)
        path = tmp_path / f"{case['name']}.json"
        path.write_text(json.dumps({"roles": roles, "prediction_frame": mode, "records": [record_for(roles, mode)]}))
        case["records_path"] = str(path)
        frozen["cases"].append(case)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(frozen))
    loaded, cases, arrays = load_inputs(manifest_path)
    assert loaded == frozen and len(cases) == 6 and tuple(arrays) == tuple(VIEWS)
    output = tmp_path / "videos"
    receipt = main(["--manifest", str(manifest_path), "--output", str(output)])
    assert len(receipt["cases"]) == 6 and receipt["playback_fps"] == 2
    for case in receipt["cases"]:
        assert case["frames"] == 1 and case["duration_seconds"] == .5 and case["full_decode"] == "passed"
        assert case["codec"] == "h264" and case["pixel_format"] == "yuv420p"
        assert case["video_sha256"] == sha256_file(case["video"])
        assert case["poster_sha256"] == sha256_file(case["poster"])
        assert case["records_sha256"] == sha256_file(case["records_path"])
    assert receipt["manifest_sha256"] == sha256_file(manifest_path)
    assert "used zero A examples" in (output / "index.html").read_text()
    assert json.loads((output / "render_receipt.json").read_text()) == receipt
    with pytest.raises(FileExistsError):
        main(["--manifest", str(manifest_path), "--output", str(output)])
