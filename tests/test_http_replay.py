# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from io import StringIO
import json
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.inference import replay_http as replay


@pytest.fixture
def source(tmp_path):
    np.save(tmp_path / "top.npy", np.zeros((17, 2, 3, 3), dtype=np.uint8))
    request = {"views": {"top": "top.npy"}, "task": "Assemble", "compressed_task_prompt": "Finish assembly",
               "stage_prompts": ["Prepare", "Finish"]}
    metadata = {"trajectory_id": "test", "num_frames": 17, "source_fps": 10,
                "whole_progress": [i / 16 for i in range(17)], "stage_ids": [int(i >= 8) for i in range(17)],
                "within_stage_progress": [i / 16 for i in range(17)]}
    input_path, metadata_path = tmp_path / "request.json", tmp_path / "metadata.json"
    input_path.write_text(json.dumps(request))
    metadata_path.write_text(json.dumps(metadata))
    return input_path, metadata_path


def response(request, roles, request_id):
    slot = request.poi_slot
    result = {"checkpoint": {"weights_sha256": "a" * 64}, "roles": list(roles),
              "prediction_frame": "poi" if slot == 5 else "last", "poi_slot": slot,
              "frame_indices": list(request.frame_indices), "progress": {},
              "selected_prediction": {"slot": slot, "frame_index": request.frame_indices[slot],
                                      "progress": {}, "focus_stage": {}}}
    selected = result["selected_prediction"]
    if "C" in roles:
        ids = [int(frame >= 8) for frame in request.frame_indices]
        result["stage"] = {"ids": ids}
        selected["stage_id"] = ids[slot]
    for role in ("A", "D"):
        if role in roles:
            values = [i / 7 for i in range(8)]
            result["progress"][role] = {"values": values}
            selected["progress"][role] = values[slot]
    if "D" in roles:
        focus = result["stage"]["ids"][slot]
        result["D_focus_stage"] = result["progress"]["D"]["focus_stage"] = focus
        selected["focus_stage"]["D"] = focus
    return {"request_id": request_id, "elapsed_seconds": .123, "result": result}


def test_all_six_modes_use_same_targets_and_keep_gt_out_of_requests(source, tmp_path, monkeypatch):
    input_path, metadata_path = source
    health = {"ready": True, "wire_formats": ["grana-binary-v1"], "checkpoint": {
        "path": "/checkpoint", "weights_sha256": "a" * 64, "source_step": 1000,
        "inference_runtime": {"gpu_image_resize": True}}}
    monkeypatch.setattr(replay, "urlopen", lambda *a, **k: StringIO(json.dumps(health)))
    calls = []

    def predict(server, request, *, roles, request_id, timeout):
        assert not hasattr(request, "gt") and not hasattr(request, "whole_progress")
        assert not hasattr(request, "stage_ids")
        calls.append((roles, request.poi_slot, request.frame_indices[request.poi_slot]))
        if request.poi_slot == 7:
            assert max(request.frame_indices) <= request.frame_indices[7]
        return response(request, roles, request_id)

    monkeypatch.setattr(replay, "predict", predict)
    manifest_path = replay.run(SimpleNamespace(input=input_path, metadata=metadata_path, output=tmp_path / "out",
                                             server="http://localhost:8787", samples=3, name_prefix="test",
                                             model_label="Fixture", timeout=300))
    manifest = json.loads(manifest_path.read_text())
    assert len(calls) == 18 and len(manifest["cases"]) == 6
    assert "A_accuracy_note" not in manifest
    assert manifest["sampling"] == {"context_stride": 16, "tail_stride": 4, "source_fps": 10,
                                    "target_frames": [0, 8, 16]}
    for case in manifest["cases"]:
        from pathlib import Path
        records = json.loads(Path(case["records_path"]).read_text())["records"]
        assert [r["frame_index"] for r in records] == [0, 8, 16]
        assert all(r["frame_indices"][r["slot"]] == r["frame_index"] for r in records)
        assert all(r["response"]["result"]["roles"] == case["roles"] for r in records)
        assert [r["gt"]["whole_progress"] for r in records] == [0, .5, 1]


def test_reply_cannot_use_other_frame_or_old_d_focus(source):
    request = replay.load_request(source[0], frame_index=8, prediction_frame="last")
    valid = response(request, ("C", "D"), "sample")
    assert replay.validate_reply(valid, request, ("C", "D"), "last", "sample", "a" * 64)
    for mutate in (lambda r: r.update(poi_slot=5),
                   lambda r: r.update(D_focus_stage=0),
                   lambda r: r["progress"]["D"].update(focus_stage=0),
                   lambda r: r["selected_prediction"].update(frame_index=16)):
        damaged = deepcopy(valid)
        mutate(damaged["result"])
        with pytest.raises(ValueError):
            replay.validate_reply(damaged, request, ("C", "D"), "last", "sample", "a" * 64)


def test_frozen_windows_and_bad_label_lengths_are_rejected(source):
    input_path, metadata_path = source
    raw = json.loads(input_path.read_text())
    raw["frame_indices"] = list(range(8))
    input_path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="without frozen"):
        replay.load_episode(input_path, metadata_path)
    raw.pop("frame_indices")
    input_path.write_text(json.dumps(raw))
    metadata = json.loads(metadata_path.read_text())
    metadata["stage_ids"].pop()
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="one value per"):
        replay.load_episode(input_path, metadata_path)


@pytest.mark.parametrize("count,samples", [(1, 1), (17, 18), (17, 1), (True, 2), (17, True)])
def test_bad_uniform_targets_fail(count, samples):
    with pytest.raises(ValueError):
        replay.uniform_targets(count, samples)
