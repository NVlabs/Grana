# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the real two-round workflow on v2/v3 RGB, with only HTTP mocked."""
from copy import deepcopy
import json
import socket

import httpx
import pytest

from test_annotation_api import provider_configuration  # noqa: F401

from grana.annotation.api import KEY_ENV
from scripts import annotation_skill as skill
from scripts import annotation_skill_api as driver
from scripts.prepare_annotation_evidence import prepare
from test_annotation_skill_api import provider
from test_annotation_v3_evidence import VIEWS, cameras, datasets, pixels  # noqa: F401


def span(start, end, **values):
    return {"start_frame": start, "end_frame": end, "reason": "Synthetic visible state.",
            "evidence_frames": [start], **values}


@pytest.mark.parametrize("version", [2, 3])
@pytest.mark.parametrize("with_quality", [False, True])
def test_native_video_to_stage_result_then_optional_bound_quality(datasets, tmp_path, monkeypatch, version, with_quality):
    monkeypatch.setenv(KEY_ENV, "synthetic-offline-key")

    def no_network(*args, **kwargs):
        raise AssertionError("This integration study must never contact an API")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    settings = {"views": VIEWS, "task": "assembly", "task_instruction": "Place the part, then secure it.",
                "stage_semantics": "active_subtask",
                "registry": [{"id": "A", "label": "1", "prompt": "Place part"},
                             {"id": "B", "label": "2", "prompt": "Secure part"}],
                "task_policy": {"version": "synthetic-study-v1", "status": "draft", "rules":
                    "Observe the named part and its support. Do not infer retention from gripper closure. "
                    "Decide the active objective from visible manipulation. If requested, nominal means "
                    "normal handling; suboptimal means an avoidable correction; unknown means insufficient evidence."}}
    evidence = tmp_path / "evidence"
    # A clip within the second episode, not the first frames of a v3 shared file.
    prepare(datasets[version - 2], 1, settings, evidence, start_frame=2, end_frame=14,
            stride_seconds=.1, max_images=12, tile_size=(32, 24))
    bundle = json.loads((evidence / "bundle.json").read_text())
    assert bundle["mode"] == "stages"
    assert bundle["context"]["source_identity"]["source_frame_range"] == [2, 14]
    assert bundle["context"]["time_edges"] == pytest.approx([i / 10 for i in range(2, 15)])
    stages = {"stages": {"segments": [span(a, a + 4, stage_id=stage, identity_status="resolved")
                                     for a, stage in ((0, "A"), (4, "B"), (8, "A"))],
                         "boundaries": [{"frame": f, "lower_frame": f - 1, "upper_frame": f + 1,
                                         "reason": "Synthetic transition bracket.", "evidence_frames": [f - 1, f]}
                                        for f in (4, 8)],
                         "invalid_intervals": [], "omitted_stages": [], "summary": "Synthetic revisit."}}
    quality = {"stage_candidate_sha256": skill._digest(stages), "stage_annotations": [
        {"stage_segment_index": i, "aq_intervals": [span(a, a + 4, quality=q)], "aq_boundaries": []}
        for i, (a, q) in enumerate(((0, "nominal"), (4, "suboptimal"), (8, "unknown")))]}
    requests = []

    def respond(request):
        payload = json.loads(request.content)
        requests.append(payload)
        context = json.loads(payload["input"][0]["content"][0]["text"].split("\n", 1)[1])
        assert "source_identity" not in context
        assert str(tmp_path) not in json.dumps(payload)
        if context["annotation_mode"] == "quality":
            assert context["frozen_stages"] == stages["stages"]
            assert context["stage_candidate_sha256"] == skill._digest(stages)
            answer = quality
        else:
            assert context["annotation_mode"] == "stages"
            assert set(payload["text"]["format"]["schema"]["properties"]) == {"stages"}
            answer = stages
        return httpx.Response(200, json=provider(payload, answer))

    options = {"api_dir": tmp_path / "budget", "max_requests": 2, "transport": httpx.MockTransport(respond)}
    driver.run(evidence / "bundle.json", evidence / "evidence.json", tmp_path / "stages",
               request_name="stages", **options)
    saved = (tmp_path / "stages/candidate.json").read_bytes()
    validation = json.loads((tmp_path / "stages/validation.json").read_text())
    assert validation["action_quality_status"] == "default_normal_as_not_requested"
    assert validation["outcome_status"] == "not_requested"
    assert "aq_frames" not in validation
    if with_quality:
        bound = skill.prepare_quality_bundle(bundle, stages)
        path = tmp_path / "quality-bundle.json"
        path.write_text(json.dumps(bound))
        driver.run(path, evidence / "evidence.json", tmp_path / "quality", request_name="quality", **options)
        assert json.loads((tmp_path / "quality/candidate.json").read_text()) == quality
        for changed in ("hash", "boundary"):
            bad = deepcopy(quality)
            if changed == "hash":
                bad["stage_candidate_sha256"] = "0" * 64
            else:
                bad["stage_annotations"][0]["aq_intervals"][0]["end_frame"] += 1
            with pytest.raises(ValueError):
                skill.validate_bundle_candidate(bound, bad)
    else:
        assert not (tmp_path / "quality").exists()
    # Replay needs neither another request nor a credential; preserve the first-round artifact.
    monkeypatch.delenv(KEY_ENV)
    driver.run(evidence / "bundle.json", evidence / "evidence.json", tmp_path / "stages",
               request_name="stages", **options)
    assert len(requests) == (2 if with_quality else 1)
    assert (tmp_path / "stages/candidate.json").read_bytes() == saved


def test_v3_stage_refinement_and_windows_keep_native_rgb(datasets, tmp_path, monkeypatch):
    import numpy as np
    from pathlib import Path
    from scripts.prepare_annotation_windows import run as window
    from scripts.refine_annotation_evidence import run as refine

    monkeypatch.setenv(KEY_ENV, "synthetic-offline-key")
    settings = {"views": VIEWS, "task": "study", "task_instruction": "Pick, then place.",
                "stage_semantics": "active_subtask",
                "registry": [{"id": "A", "label": "1", "prompt": "Pick part"},
                             {"id": "B", "label": "2", "prompt": "Place part"}],
                "task_policy": {"version": "study-v1", "status": "draft", "rules": "Use visible objectives."}}
    evidence, result = tmp_path / "coarse", tmp_path / "candidate"
    prepare(datasets[1], 1, settings, evidence, start_frame=2, end_frame=30,
            stride_seconds=1, max_images=8, tile_size=(32, 24))
    candidate = {"stages": {
        "segments": [span(0, 14, stage_id="A", identity_status="resolved"),
                     {**span(14, 28, stage_id="B", identity_status="resolved"), "evidence_frames": [20]}],
        "boundaries": [{"frame": 14, "lower_frame": 10, "upper_frame": 20,
                        "reason": "Synthetic uncertain transition.", "evidence_frames": [10, 20]}],
        "invalid_intervals": [], "omitted_stages": [], "summary": "Synthetic stage-only candidate."}}
    driver.run(evidence / "bundle.json", evidence / "evidence.json", result, max_requests=1,
               transport=httpx.MockTransport(lambda r: httpx.Response(200, json=provider(json.loads(r.content), candidate))))
    summary = refine(evidence, result, settings, tmp_path / "dense", max_extra_frames=8,
                     max_total_images=16, radius_seconds=.3)
    assert summary["extra_frames"] > 0
    dense = json.loads((tmp_path / "dense/evidence.json").read_text())
    piece = window(evidence, result, settings, tmp_path / "windows", policy_dir=skill.POLICY_DIR,
                   window_id="window_0001", max_core_seconds=1, context_seconds=.2,
                   stride_seconds=.5, max_images=8)
    cropped = json.loads((Path(piece["output"]) / "evidence.json").read_text())
    for offset, manifest in ((2, dense), (piece["source_frame_range"][0], cropped)):
        for row in manifest["images"]:
            for view, image in cameras(row).items():
                np.testing.assert_array_equal(image, pixels(1, view)[row["frame"] + offset])
