# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real RGB/clock/receipt pipeline; only the external model is mocked."""
from copy import deepcopy
import json
from pathlib import Path

import httpx
import pytest

from test_annotation_api import provider_configuration  # noqa: F401

from grana.annotation.api import KEY_ENV
from grana.annotation.rounds import read, run_episode, verify_result
from scripts.annotation_skill import POLICY_DIR
from test_annotation_skill_api import provider
from test_annotation_v3_evidence import VIEWS, datasets  # noqa: F401


@pytest.fixture
def settings():
    return {"views": VIEWS, "task": "assembly", "task_instruction": "Place, then secure the part.",
            "stage_semantics": "active_subtask",
            "registry": [{"id": "A", "label": "1", "prompt": "Place part"},
                         {"id": "B", "label": "2", "prompt": "Secure part"}],
            "task_policy": {"version": "synthetic-v1", "status": "draft", "rules": "Use visible evidence only."}}


def answer(payload):
    ctx = json.loads(payload["input"][0]["content"][0]["text"].split("\n", 1)[1])
    count = len(ctx["time_edges"]) - 1
    row = {"start_frame": 0, "end_frame": count, "reason": "Synthetic visible evidence.", "evidence_frames": [0]}
    if ctx["annotation_mode"] == "stages":
        stage = "A" if ctx["time_edges"][0] < 1.5 else "B"
        data = {"stages": {"segments": [{**row, "stage_id": stage, "identity_status": "resolved"}],
                "boundaries": [], "invalid_intervals": [], "omitted_stages": [
                    {"stage_id": s["id"], "reason": "Not observed."} for s in ctx["registry"] if s["id"] != stage],
                "summary": "Synthetic stage-only output."}}
    else:
        assert len(ctx["frozen_stages"]["segments"]) == 1
        assert ctx["time_edges"][-1] - ctx["time_edges"][0] <= .8
        data = {"stage_candidate_sha256": ctx["stage_candidate_sha256"], "stage_annotations": [
            {"stage_segment_index": 0, "aq_intervals": [{**row, "quality": "nominal"}], "aq_boundaries": []}]}
    return ctx, data


@pytest.mark.parametrize("version", [2, 3])
def test_stages_standalone_then_bounded_quality_with_cache_and_verified_import(datasets, settings, tmp_path, monkeypatch, version):
    monkeypatch.setenv(KEY_ENV, "offline-synthetic-key")
    seen = []
    def respond(request):
        payload = json.loads(request.content)
        context, data = answer(payload)
        seen.append(context)
        assert "source_identity" not in context
        return httpx.Response(200, json=provider(payload, data))
    source, output = datasets[version - 2], tmp_path / "campaign"
    options = dict(stage_seconds=1.5, stage_stride=.5, quality_seconds=.8, quality_stride=.2,
                   max_images=8, transport=httpx.MockTransport(respond))
    first = run_episode(source, 1, settings, output, **options)
    saved = (output / "candidate/candidate.json").read_bytes()
    assert first["action_quality_status"] == "default_normal_as_not_requested" and first["stage_requests"] == 2
    assert [c["annotation_mode"] for c in seen] == ["stages", "stages"]
    assert not (output / "quality").exists()
    bundle, stages, _, _ = verify_result(first["bundle_path"], first["candidate_dir"], POLICY_DIR)
    assert [(s["stage_id"], s["start_frame"], s["end_frame"]) for s in stages["stages"]["segments"]] == [("A", 0, 15), ("B", 15, 30)]
    second = run_episode(source, 1, settings, output, with_quality=True, **options)
    assert second["quality_requests"] == 4
    assert [c["annotation_mode"] for c in seen] == ["stages"] * 2 + ["quality"] * 4
    quality_bundle, quality, _, _ = verify_result(second["bundle_path"], second["candidate_dir"], POLICY_DIR)
    assert quality_bundle["stage_lock"] == {"bundle": bundle, "candidate": stages}
    assert (output / "candidate/candidate.json").read_bytes() == saved
    assert len(quality["stage_annotations"]) == 2
    from grana.annotation.candidate_import import verified_draft
    from grana.annotation.labels import completion_problems
    cfg = {"source": str(source), "views": VIEWS, "task": settings["task"],
           "stages": [{k: s[k] for k in ("label", "prompt")} for s in settings["registry"]]}
    doc, _ = verified_draft(cfg, 1, first["bundle_path"], first["candidate_dir"])
    assert doc["quality"] == []
    assert doc["machine_import"]["action_quality_status"] == "default_normal_as_not_requested"
    doc, _ = verified_draft(cfg, 1, second["bundle_path"], second["candidate_dir"])
    assert doc["machine_import"]["action_quality_status"] == "annotated"
    from grana.annotation.drafts import verified_results
    chosen, reports = verified_results(cfg, [1], [output], skip=lambda _: None)
    assert len(chosen) == 1 and chosen[0][0].folder == output / "quality", reports
    assert len(reports) == 1 and reports[0]["status"] == "superseded"
    monkeypatch.delenv(KEY_ENV)
    assert run_episode(source, 1, settings, output, with_quality=True, **options) == second
    assert len(seen) == 6
    child = output / "requests/quality-00000-00000/candidate/candidate.json"
    bad = read(child)
    bad["stage_annotations"][0]["aq_intervals"][0]["quality"] = "suboptimal"
    child.write_text(json.dumps(bad))
    with pytest.raises(ValueError):
        verify_result(second["bundle_path"], second["candidate_dir"], POLICY_DIR)


def test_split_long_clock_preserves_coverage_and_image_limit():
    from grana.annotation.rounds import split
    from scripts.prepare_annotation_evidence import select_frames
    clock = [i / 30 for i in range(18001)]
    chunks = split(clock, 0, 18000, 10, .2, 40)
    assert chunks[0][0] == 0 and chunks[-1][1] == 18000
    assert all(a[1] == b[0] for a, b in zip(chunks, chunks[1:]))
    assert all(clock[b] - clock[a] <= 10 and len(select_frames(clock, a, b, .2, 40)) <= 40 for a, b in chunks)


def test_v3_import_editor_plays_only_the_selected_episode(datasets, settings, tmp_path):
    import subprocess
    import numpy as np
    from grana.annotation.label_server import LabelProject
    from grana.data.builder import validate_build_config
    from test_annotation_v3_evidence import pixels
    cfg = validate_build_config({"source": str(datasets[1]), "output": str(tmp_path / "cache"),
        "labels": str(tmp_path / "labels"), "episodes": "all", "task": "assembly", "views": VIEWS,
        "compressed_task_prompt": "Place, then secure.", "fps": 5,
        "stages": [{k: s[k] for k in ("label", "prompt")} for s in settings["registry"]],
        "progress": {"method": "grana_target_composition", "parameter_status": "example",
                     "parameter_source": "Synthetic test", "stage_parameters": [{"rho": 20, "p_cap": .95, "t_min_seconds": 1}] * 2}})
    project = LabelProject(cfg)
    try:
        clip = project.video(1, "wrist")
        assert project.video(1, "wrist") == clip
        raw = subprocess.check_output(["ffmpeg", "-v", "error", "-i", str(clip), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
        rgb = np.frombuffer(raw, np.uint8).reshape(-1, 24, 32, 3)
        assert len(rgb) == 30
        assert np.abs(rgb.astype(int) - pixels(1, "wrist").astype(int)).max() < 12
    finally:
        project.close()
