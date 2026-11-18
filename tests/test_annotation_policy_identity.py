# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stable policy names preserve existing bundles, completed requests and stage dependencies."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import httpx
import pytest

from test_annotation_api import provider_configuration  # noqa: F401

from grana.annotation.api import KEY_ENV
from grana.annotation.rounds import run_episode, verify_result
from scripts import annotation_skill as skill
from scripts import prepare_annotation_evidence as evidence
from test_annotation_bounded_rounds import answer, settings  # noqa: F401
from test_annotation_rounds import rounds  # noqa: F401
from test_annotation_skill import context, candidate, policy  # noqa: F401
from test_annotation_skill_api import provider
from test_annotation_v3_evidence import datasets  # noqa: F401


def historical(bundle):
    bundle = deepcopy(bundle)
    name = ("grana_robot_annotation_v1" if bundle["schema_version"] == "grana.annotation_bundle.v1"
            else "grana_robot_annotation_rounds_v2")
    bundle["policy"] = {"version": name, "files": bundle["policy"]["files"]}
    return bundle


@pytest.fixture(params=["stages", "quality", "combined"])
def bundle_case(request, rounds, context, candidate, policy):
    stage, stages, quality = rounds
    if request.param == "combined":
        return skill.prepare_bundle(context, policy, mode="combined"), candidate, policy
    if request.param == "quality":
        return skill.prepare_quality_bundle(stage, stages), quality, skill.POLICY_DIR
    return stage, stages, skill.POLICY_DIR


def test_new_results_name_policy_and_old_results_keep_their_original_identity(bundle_case):
    bundle, candidate, policy = bundle_case
    assert bundle["policy"]["name"] == "grana-robot-annotation"
    assert "version" not in bundle["policy"]
    current = skill.validate_bundle_candidate(bundle, candidate, policy)
    assert current["policy_name"] == bundle["policy"]["name"] and "policy_version" not in current
    old = historical(bundle)
    before = skill._digest(old)
    assert skill.verify_bundle(old, policy) == old
    assert skill.resume_bundle(bundle, old, policy) == old
    report = skill.validate_bundle_candidate(old, candidate, policy)
    assert report["policy_version"] == old["policy"]["version"]
    assert {k: v for k, v in report.items() if k != "policy_version"} == {
        k: v for k, v in current.items() if k != "policy_name"}
    assert skill._digest(old) == before


@pytest.mark.parametrize("change", [
    lambda b: b["policy"].update(version="unrecognized"),
    lambda b: b["policy"].update(name="grana-robot-annotation"),
    lambda b: b["policy"]["files"].update({"unexpected.md": "0" * 64}),
    lambda b: b["context"]["task_policy"].update(rules="Changed after the request."),
    lambda b: b.update(instructions="Changed after the request."),
    lambda b: b["schema"].update(additionalProperties=True),
])
def test_historical_name_compatibility_does_not_relax_integrity(bundle_case, change):
    bundle, _, policy = bundle_case
    old = historical(bundle)
    change(old)
    with pytest.raises(ValueError, match="integrity mismatch"):
        skill.verify_bundle(old, policy)


def test_resume_refuses_changed_inputs(bundle_case):
    bundle, _, policy = bundle_case
    expected = deepcopy(bundle)
    expected["context"]["task_policy"]["rules"] += " A new rule."
    with pytest.raises(ValueError, match="Frozen campaign bundle differs"):
        skill.resume_bundle(expected, historical(bundle), policy)


def test_new_quality_preserves_historical_stage_parent(rounds):
    stage, stages, quality = rounds
    old = historical(stage)
    bundle = skill.prepare_quality_bundle(old, stages)
    assert bundle["policy"]["name"] == "grana-robot-annotation"
    assert bundle["stage_lock"]["bundle"] == old
    assert skill.validate_bundle_candidate(bundle, quality)["action_quality_status"] == "annotated"


@pytest.mark.parametrize("with_quality", [False, True])
def test_old_campaign_resumes_without_rebilling_or_rewriting_and_can_add_quality(
        datasets, settings, tmp_path, monkeypatch, with_quality):
    monkeypatch.setenv(KEY_ENV, "offline-synthetic-key")
    requests = []

    def respond(request):
        payload = json.loads(request.content)
        requests.append(payload)
        return httpx.Response(200, json=provider(payload, answer(payload)[1]))

    options = dict(stage_seconds=1.5, stage_stride=.5, quality_seconds=.8,
                   quality_stride=.2, max_images=8, transport=httpx.MockTransport(respond))
    output = tmp_path / "campaign"
    prepare = skill.prepare_bundle
    # Reproduce the original writer's metadata on a complete RGB/request pipeline.
    with monkeypatch.context() as old_writer:
        legacy_prepare = lambda *a, **kw: historical(prepare(*a, **kw))
        old_writer.setattr(skill, "prepare_bundle", legacy_prepare)
        old_writer.setattr(evidence, "prepare_bundle", legacy_prepare)
        before = run_episode(datasets[1], 1, settings, output, with_quality=with_quality, **options)
    saved = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.rglob("*") if p.is_file()}
    request_count = len(requests)
    resumed = run_episode(datasets[1], 1, settings, output, with_quality=with_quality, **options)
    assert resumed == before and len(requests) == request_count
    assert saved == {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.rglob("*") if p.is_file()}
    bundle, _, _, _ = verify_result(resumed["bundle_path"], resumed["candidate_dir"], skill.POLICY_DIR)
    assert "version" in bundle["policy"]
    if not with_quality:
        stages = Path(resumed["bundle_path"]).read_bytes()
        enriched = run_episode(datasets[1], 1, settings, output, with_quality=True, **options)
        bundle, _, _, _ = verify_result(enriched["bundle_path"], enriched["candidate_dir"], skill.POLICY_DIR)
        assert bundle["policy"]["name"] == "grana-robot-annotation"
        assert enriched["action_quality_status"] == "annotated"
        assert Path(resumed["bundle_path"]).read_bytes() == stages
