# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stage-only validation receipts written before the AQ status rename still verify; nothing else is relaxed."""
import json
from pathlib import Path

import httpx
import pytest

from test_annotation_api import provider_configuration  # noqa: F401

from grana.annotation.api import BASE_URL_ENV, KEY_ENV, MODEL_ENV
from grana.annotation.candidate_import import _verify_delivery, verified_draft
from grana.annotation.rounds import read, run_episode, verify_result
from scripts.annotation_skill import POLICY_DIR, stored_validation_receipt
from scripts.annotation_skill_api import prepare_submission
from test_annotation_bounded_rounds import answer, settings  # noqa: F401
from test_annotation_skill_api import provider
from test_annotation_v3_evidence import VIEWS, datasets  # noqa: F401

LEGACY = "not_requested"


@pytest.fixture
def stage_result(datasets, settings, tmp_path, monkeypatch):
    monkeypatch.setenv(KEY_ENV, "offline-synthetic-key")

    def respond(request):
        payload = json.loads(request.content)
        return httpx.Response(200, json=provider(payload, answer(payload)[1]))
    result = run_episode(datasets[1], 1, settings, tmp_path / "campaign", stage_seconds=1.5, stage_stride=.5,
                         quality_seconds=.8, quality_stride=.2, max_images=8, transport=httpx.MockTransport(respond))
    for variable in (KEY_ENV, MODEL_ENV, BASE_URL_ENV):
        monkeypatch.delenv(variable)
    return datasets[1], result


def legacy(result):
    """Rewrite every validation receipt (and the aggregate status copy) as stored before the rename."""
    aggregate = Path(result["candidate_dir"])
    delivery = read(aggregate / "delivery.json")
    for folder in [aggregate, *(Path(job["result"]) for job in delivery["jobs"])]:
        validation = read(folder / "validation.json")
        assert validation["annotation_mode"] == "stages"
        assert validation["action_quality_status"] == "default_normal_as_not_requested"
        (folder / "validation.json").write_text(json.dumps(validation | {"action_quality_status": LEGACY}))
    (aggregate / "delivery.json").write_text(json.dumps(delivery | {"action_quality_status": LEGACY}))
    return delivery["jobs"]


def verify_child(job):
    """The exact per-request check rounds.verify_result runs through candidate_import._verify_delivery."""
    path, folder = Path(job["bundle"]), Path(job["result"])
    submitted = read(folder / "delivery.json")
    template = submitted.get("template")
    bundle, content, images, _ = prepare_submission(path, path.parent / "evidence.json", POLICY_DIR,
                                                    template["template_path"] if template else None)
    _verify_delivery(bundle, read(folder / "candidate.json"), submitted, read(folder / "prepared.json"),
                     read(folder / "validation.json"), images, content, folder, POLICY_DIR)


def test_legacy_stage_only_validation_receipts_still_verify_and_import(stage_result, settings):
    source, result = stage_result
    jobs = legacy(result)
    for job in jobs:
        assert read(Path(job["result"]) / "validation.json")["action_quality_status"] == LEGACY
        verify_child(job)
    _, candidate, _, _ = verify_result(result["bundle_path"], result["candidate_dir"], POLICY_DIR)
    assert len(candidate["stages"]["segments"]) == 2
    cfg = {"source": str(source), "views": VIEWS, "task": settings["task"],
           "stages": [{k: s[k] for k in ("label", "prompt")} for s in settings["registry"]]}
    doc, _ = verified_draft(cfg, 1, result["bundle_path"], result["candidate_dir"])
    assert doc["quality"] == []
    assert doc["machine_import"]["action_quality_status"] == "default_normal_as_not_requested"


@pytest.mark.parametrize("where", ["child", "aggregate"])
@pytest.mark.parametrize("change", [{"semantic_accuracy_verified": True}, {"candidate_sha256": "0" * 64},
                                    {"outcome_status": "annotated"}, {"action_quality_status": "annotated"}])
def test_other_validation_receipt_differences_are_still_rejected(stage_result, where, change):
    _, result = stage_result
    jobs = legacy(result)
    folder = Path(jobs[0]["result"] if where == "child" else result["candidate_dir"])
    (folder / "validation.json").write_text(json.dumps(read(folder / "validation.json") | change))
    if where == "child":
        with pytest.raises(ValueError, match="Validation receipt differs"):
            verify_child(jobs[0])
    else:
        verify_child(jobs[0])
    with pytest.raises(ValueError):
        verify_result(result["bundle_path"], result["candidate_dir"], POLICY_DIR)


def test_legacy_value_is_only_renamed_for_stage_only_reports():
    current = {"annotation_mode": "stages", "action_quality_status": "default_normal_as_not_requested"}
    assert stored_validation_receipt(current | {"action_quality_status": LEGACY}, current) == current
    quality = {"annotation_mode": "quality", "action_quality_status": "annotated"}
    stored = quality | {"action_quality_status": LEGACY}
    assert stored_validation_receipt(stored, quality) is stored
    for other in ("unknown", "annotated", None):
        stored = current | {"action_quality_status": other}
        assert stored_validation_receipt(stored, current) is stored
    stored = {"annotation_mode": "quality", "action_quality_status": LEGACY}
    assert stored_validation_receipt(stored, current) is stored
