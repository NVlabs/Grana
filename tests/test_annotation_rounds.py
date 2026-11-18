# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure API rounds, immutable stage dependencies and opt-in AQ; no live API."""
from copy import deepcopy
import hashlib
import json

import httpx
import jsonschema
from PIL import Image
import pytest

from test_annotation_api import provider_configuration  # noqa: F401

from grana.annotation.api import AnnotationAPIError, KEY_ENV
from scripts import annotation_skill as skill
from scripts import annotation_skill_api as driver
from test_annotation_skill_api import provider
from test_annotation_source import raw


def interval(start, end, **values):
    return {"start_frame": start, "end_frame": end, "reason": "Synthetic observed action",
            "evidence_frames": [start], **values}


def boundary(frame):
    return {"frame": frame, "lower_frame": frame - 1, "upper_frame": frame + 1,
            "reason": "Synthetic transition", "evidence_frames": [frame - 1, frame]}


@pytest.fixture
def rounds():
    context = {"schema_version": "grana.annotation_context.v1", "task": "assembly",
               "task_instruction": "Place a part and secure it", "stage_semantics": "active_subtask",
               "task_policy": {"version": "test-v1", "status": "draft", "rules": "Use visible objectives."},
               "source_identity": {"source_root": "/PRIVATE_SOURCE", "episode": 0},
               "time_edges": list(range(7)), "evidence_frames": list(range(6)),
               "registry": [{"id": "A", "label": "1", "prompt": "Place part"},
                            {"id": "B", "label": "2", "prompt": "Secure part"}]}
    stages = {"stages": {"segments": [interval(0, 2, stage_id="A", identity_status="resolved"),
                                      interval(2, 4, stage_id="B", identity_status="resolved"),
                                      interval(4, 6, stage_id="A", identity_status="resolved")],
                         "boundaries": [boundary(2), boundary(4)], "omitted_stages": [],
                         "invalid_intervals": [], "summary": "Synthetic revisit"}}
    bundle = skill.prepare_bundle(context)
    quality = {"stage_candidate_sha256": skill._digest(stages), "stage_annotations": [
        {"stage_segment_index": index, "aq_intervals": [interval(start, start + 2, quality="nominal")],
         "aq_boundaries": []} for index, start in enumerate((0, 2, 4))]}
    return bundle, stages, quality


def test_default_stages_is_complete_without_aq_outcome_or_nominal_inference(rounds):
    bundle, stages, _ = rounds
    jsonschema.validate(stages, bundle["schema"])
    report = skill.validate_bundle_candidate(bundle, stages)
    assert report["action_quality_status"] == "default_normal_as_not_requested"
    assert report["outcome_status"] == "not_requested"
    assert "aq_frames" not in report and "outcome_counts" not in report
    assert report["stage_coverage"]["source_frames"] == 6
    for extra in ("aq_intervals", "outcomes"):
        with pytest.raises(ValueError, match="exactly"):
            skill.validate_bundle_candidate(bundle, stages | {extra: []})


def test_aq_mode_does_not_mutate_stage_results_when_response_is_invalid(rounds, tmp_path, monkeypatch):
    bundle, stages, quality = rounds
    bundle_path, evidence = write_inputs(tmp_path, skill.prepare_quality_bundle(bundle, stages))
    original = deepcopy(stages)
    quality["stage_annotations"][0]["aq_intervals"][0]["end_frame"] = 3
    monkeypatch.setenv(KEY_ENV, "synthetic-offline-key")
    output = tmp_path / "rejected"
    with pytest.raises(ValueError, match="frozen stage"):
        driver.run(bundle_path, evidence, output,
                   transport=httpx.MockTransport(lambda r: httpx.Response(200, json=provider(json.loads(r.content), quality))))
    assert json.loads((output / "delivery.json").read_text())["status"] == "semantic_validation_failed"
    assert json.loads((output / "candidate.json").read_text()) == quality
    assert not (output / "validation.json").exists()
    assert stages == original


def test_quality_uses_occurrences_preserves_inputs_and_does_not_require_outcomes(rounds):
    bundle, stages, quality = rounds
    original = deepcopy((bundle, stages))
    aq = skill.prepare_quality_bundle(bundle, stages)
    jsonschema.Draft202012Validator.check_schema(aq["schema"])
    jsonschema.validate(quality, aq["schema"])
    report = skill.validate_bundle_candidate(aq, quality)
    assert (bundle, stages) == original
    assert report["aq_frames"]["nominal"] == 6
    assert report["action_quality_status"] == "annotated" and report["outcome_status"] == "not_requested"
    assert report["stage_candidate_sha256"] == skill._digest(stages)


@pytest.mark.parametrize("damage", ["hash", "crossing", "gap", "missing_stage", "reorder", "bool_index",
                                    "new_stages", "new_outcomes", "extra_boundary", "missing_boundary",
                                    "bad_bracket", "uncited_frame"])
def test_quality_rejects_invalid_or_unbound_results(rounds, damage):
    bundle, stages, quality = rounds
    aq = skill.prepare_quality_bundle(bundle, stages)
    first = quality["stage_annotations"][0]
    if damage == "hash":
        quality["stage_candidate_sha256"] = "0" * 64
    elif damage == "crossing":
        first["aq_intervals"][0]["end_frame"] = 3
    elif damage == "gap":
        first["aq_intervals"][0]["start_frame"] = 1
    elif damage == "missing_stage":
        quality["stage_annotations"].pop()
    elif damage == "reorder":
        quality["stage_annotations"].reverse()
    elif damage == "bool_index":
        first["stage_segment_index"] = False
    elif damage == "new_stages":
        quality["stages"] = stages["stages"]
    elif damage == "new_outcomes":
        quality["outcomes"] = []
    elif damage == "extra_boundary":
        first["aq_boundaries"] = [boundary(1)]
    elif damage in ("missing_boundary", "bad_bracket"):
        first["aq_intervals"] = [interval(0, 1, quality="nominal"), interval(1, 2, quality="suboptimal")]
        if damage == "bad_bracket":
            first["aq_boundaries"] = [boundary(1) | {"upper_frame": 3}]
    elif damage == "uncited_frame":
        first["aq_intervals"][0]["evidence_frames"] = [99]
    with pytest.raises(ValueError):
        skill.validate_bundle_candidate(aq, quality)


@pytest.mark.parametrize("uncertainty", ["uncertain", "unresolved", "invalid"])
def test_quality_distinguishes_unknown_from_not_requested(rounds, uncertainty):
    bundle, stages, quality = rounds
    if uncertainty == "invalid":
        stages["stages"]["invalid_intervals"] = [{"start_frame": 0, "end_frame": 1, "reason": "Corrupt"}]
    else:
        stages["stages"]["segments"][0]["identity_status"] = uncertainty
        if uncertainty == "unresolved":
            stages["stages"]["segments"][0]["stage_id"] = None
    quality["stage_candidate_sha256"] = skill._digest(stages)
    aq = skill.prepare_quality_bundle(bundle, stages)
    with pytest.raises(ValueError, match="unknown AQ"):
        skill.validate_bundle_candidate(aq, quality)
    quality["stage_annotations"][0]["aq_intervals"][0]["quality"] = "unknown"
    assert skill.validate_bundle_candidate(aq, quality)["aq_frames"]["unknown"] == 2


def test_stage_revision_requires_new_quality_binding(rounds):
    bundle, stages, quality = rounds
    old = skill.prepare_quality_bundle(bundle, stages)
    stages["stages"]["segments"][0]["end_frame"] = 1
    stages["stages"]["segments"][1]["start_frame"] = 1
    stages["stages"]["boundaries"][0] = boundary(1)
    new = skill.prepare_quality_bundle(bundle, stages)
    assert old["stage_lock"]["candidate"] != new["stage_lock"]["candidate"]
    with pytest.raises(ValueError, match="frozen stage"):
        skill.validate_bundle_candidate(new, quality)
    skill.validate_bundle_candidate(old, quality)


@pytest.mark.parametrize("field", ["source_identity", "registry", "time_edges", "task", "stage_semantics", "evidence_frames"])
def test_aq_cannot_switch_source_clock_vocabulary_or_drop_original_evidence(rounds, field):
    bundle, stages, _ = rounds
    context = deepcopy(bundle["context"])
    if field == "source_identity":
        context[field]["episode"] = 1
    elif field == "registry":
        context[field][0]["prompt"] = "Different objective"
    elif field == "time_edges":
        context[field] = [n + 10 for n in context[field]]
    elif field == "task":
        context[field] = "other"
    elif field == "stage_semantics":
        context[field] = "attained_state"
    else:
        context[field].pop()
    with pytest.raises(ValueError):
        skill.prepare_quality_bundle(bundle, stages, context=context)


def test_aq_can_add_evidence_and_rubric_without_changing_stages(rounds):
    bundle, stages, _ = rounds
    context = deepcopy(bundle["context"])
    context["evidence_frames"] = list(range(5))
    coarse = skill.prepare_bundle(context)
    dense = deepcopy(bundle["context"])
    dense["task_policy"] = {"version": "aq-v2", "status": "draft", "rules": "Missed acquisition is suboptimal."}
    aq = skill.prepare_quality_bundle(coarse, stages, context=dense)
    assert aq["context"]["evidence_frames"] == list(range(6))
    assert aq["stage_lock"]["candidate"] == stages
    assert skill.verify_bundle(aq) == aq


def write_inputs(tmp_path, bundle):
    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_text(json.dumps(bundle))
    rows = []
    for frame in bundle["context"]["evidence_frames"]:
        path = tmp_path / f"frame-{frame}.png"
        Image.new("RGB", (8, 4), (frame, 0, 0)).save(path)
        rows.append({"frame": frame, "timestamp": bundle["context"]["time_edges"][frame],
                     "image_path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                     "view_names": ["front"]})
    evidence = tmp_path / "evidence.json"
    evidence.write_text(json.dumps({"schema_version": "grana.annotation_evidence.v2",
                                    "context_sha256": bundle["context_sha256"], "images": rows}))
    return bundle_path, evidence


@pytest.mark.parametrize("mode", ["stages", "quality"])
def test_selected_policy_and_task_rules_reach_api_and_cannot_drift(rounds, tmp_path, monkeypatch, mode):
    original, stages, quality = rounds
    policy = tmp_path / "policy"
    (policy / "references").mkdir(parents=True)
    (policy / "SKILL.md").write_text("Local orchestration sentinel; not model instructions.")
    policies = {
        "stages": ("stage-annotation.md", "Stage policy sentinel: observe the target before choosing identity."),
        "quality": ("annotation-policy.md", "Quality policy sentinel: distinguish a miss from loss of retention."),
    }
    for name, content in policies.values():
        (policy / "references" / name).write_text(content)
    context = deepcopy(original["context"])
    context["task_policy"]["rules"] = "Task contrast: an empty moving gripper does not transport the part."
    stage_bundle = skill.prepare_bundle(context, policy)
    bundle = stage_bundle if mode == "stages" else skill.prepare_quality_bundle(stage_bundle, stages, policy)
    path, evidence = write_inputs(tmp_path, bundle)
    monkeypatch.setenv(KEY_ENV, "synthetic-offline-key")
    calls = []

    def handler(request):
        payload = json.loads(request.content)
        calls.append(payload)
        wire_context = json.loads(payload["input"][0]["content"][0]["text"].split("\n", 1)[1])
        assert wire_context["task_policy"] == context["task_policy"]
        assert policies[mode][1] in payload["instructions"]
        assert policies["quality" if mode == "stages" else "stages"][1] not in payload["instructions"]
        assert "Local orchestration sentinel" not in payload["instructions"]
        assert payload["text"]["format"]["schema"] == bundle["schema"]
        return httpx.Response(200, json=provider(payload, stages if mode == "stages" else quality))

    driver.run(path, evidence, tmp_path / "result", policy_dir=policy,
               transport=httpx.MockTransport(handler))
    assert len(calls) == 1
    (policy / "references" / policies[mode][0]).write_text("Changed decision rule after preparation.")
    with pytest.raises(ValueError, match="integrity mismatch"):
        driver.run(path, evidence, tmp_path / "changed", policy_dir=policy,
                   transport=httpx.MockTransport(lambda _: pytest.fail("Policy drift must fail before API")))
    assert not (tmp_path / "changed").exists()


def test_two_api_rounds_replay_share_budget_and_keep_private_parent_off_wire(rounds, tmp_path, monkeypatch):
    bundle, stages, quality = rounds
    bundle_path, evidence = write_inputs(tmp_path, bundle)
    monkeypatch.setenv(KEY_ENV, "synthetic-offline-key")
    calls = []

    def transport(request):
        payload = json.loads(request.content)
        assert "PRIVATE_SOURCE" not in request.content.decode()
        context = json.loads(payload["input"][0]["content"][0]["text"].split("\n", 1)[1])
        mode = context["annotation_mode"]
        calls.append(mode)
        if mode == "quality":
            assert context["frozen_stages"] == stages["stages"]
            assert "stage_lock" not in context and "source_identity" not in context
        return httpx.Response(200, json=provider(payload, stages if mode == "stages" else quality))

    options = {"api_dir": tmp_path / "campaign", "max_requests": 2, "max_attempts": 1,
               "transport": httpx.MockTransport(transport)}
    driver.run(bundle_path, evidence, tmp_path / "stages", **options)
    assert calls == ["stages"]
    stage_bytes = (tmp_path / "stages/candidate.json").read_bytes()
    quality_path = tmp_path / "quality-bundle.json"
    quality_path.write_text(json.dumps(skill.prepare_quality_bundle(bundle, stages)))
    driver.run(quality_path, evidence, tmp_path / "quality", **options)
    assert calls == ["stages", "quality"]
    assert (tmp_path / "stages/candidate.json").read_bytes() == stage_bytes
    monkeypatch.delenv(KEY_ENV)
    assert driver.run(quality_path, evidence, tmp_path / "quality", **options)["cache_hit"]
    monkeypatch.setenv(KEY_ENV, "synthetic-offline-key")
    with pytest.raises(AnnotationAPIError, match="budget"):
        driver.run(bundle_path, evidence, tmp_path / "third", request_name="third", **options)
    assert len(calls) == 2


def test_named_stage_repeat_preserves_disagreement_and_replay_does_not_resample(rounds, tmp_path, monkeypatch):
    bundle, original, _ = rounds
    alternative = deepcopy(original)
    alternative["stages"]["segments"][0]["end_frame"] = 3
    alternative["stages"]["segments"][1]["start_frame"] = 3
    alternative["stages"]["boundaries"][0] = boundary(3)
    answers = [original, alternative]
    bundle_path, evidence = write_inputs(tmp_path, bundle)
    monkeypatch.setenv(KEY_ENV, "synthetic-offline-key")
    calls = []

    def handler(request):
        payload = json.loads(request.content)
        answer = answers[len(calls)]
        calls.append(payload)
        return httpx.Response(200, json=provider(payload, answer))

    options = {"api_dir": tmp_path / "campaign", "max_requests": 2,
               "transport": httpx.MockTransport(handler)}
    reports = []
    for index, answer in enumerate(answers):
        output = tmp_path / f"sample-{index}"
        report = driver.run(bundle_path, evidence, output, request_name=f"stages-sample-{index}", **options)
        assert not report["cache_hit"]
        assert json.loads((output / "candidate.json").read_text()) == answer
        reports.append(json.loads((output / "validation.json").read_text()))
    assert calls[0] == calls[1]  # A repeat uses the same evidence, task, policy and model settings.
    assert reports[0]["context_sha256"] == reports[1]["context_sha256"]
    assert reports[0]["candidate_sha256"] != reports[1]["candidate_sha256"]
    assert all(report["action_quality_status"] == "default_normal_as_not_requested" for report in reports)
    monkeypatch.delenv(KEY_ENV)
    for index, answer in enumerate(answers):
        output = tmp_path / f"sample-{index}"
        assert driver.run(bundle_path, evidence, output, request_name=f"stages-sample-{index}", **options)["cache_hit"]
        assert json.loads((output / "candidate.json").read_text()) == answer
    assert len(calls) == 2


def test_swapped_evidence_context_rejected_before_api(rounds, tmp_path):
    bundle, _, _ = rounds
    path, evidence = write_inputs(tmp_path, bundle)
    manifest = json.loads(evidence.read_text())
    manifest["context_sha256"] = "0" * 64
    evidence.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Evidence context"):
        driver.run(path, evidence, tmp_path / "result", transport=httpx.MockTransport(lambda _: pytest.fail("No API")))
    assert not (tmp_path / "result").exists()


def test_default_cli_and_optional_quality_prepare_validate_without_api(rounds, tmp_path, capsys):
    bundle, stages, quality = rounds
    context_path = tmp_path / "context.json"
    context_path.write_text(json.dumps(bundle["context"]))
    stage_bundle = tmp_path / "stages.json"
    stage_candidate = tmp_path / "stage-candidate.json"
    stage_candidate.write_text(json.dumps(stages))
    quality_bundle = tmp_path / "quality.json"
    skill.main(["prepare", "--context", str(context_path), "--output", str(stage_bundle)])
    prepared = json.loads(capsys.readouterr().out)
    assert prepared["policy_name"] == "grana-robot-annotation"
    assert "context_sha256" not in prepared and "policy_version" not in prepared
    assert json.loads(stage_bundle.read_text())["mode"] == "stages"
    skill.main(["prepare-quality", "--stage-bundle", str(stage_bundle),
                "--stage-candidate", str(stage_candidate), "--output", str(quality_bundle)])
    prepared = json.loads(capsys.readouterr().out)
    assert prepared["stage_bundle"] == str(stage_bundle)
    assert "stage_candidate_sha256" not in prepared
    skill.validate_bundle_candidate(json.loads(quality_bundle.read_text()), quality)
    quality_candidate, report_path = tmp_path / "quality-candidate.json", tmp_path / "validation.json"
    quality_candidate.write_text(json.dumps(quality))
    skill.main(["validate", "--bundle", str(quality_bundle), "--candidate", str(quality_candidate),
                "--output", str(report_path)])
    summary = json.loads(capsys.readouterr().out)
    report = json.loads(report_path.read_text())
    assert summary["status"] == report["status"] == "valid_candidate"
    assert summary["action_quality_status"] == report["action_quality_status"] == "annotated"
    assert summary["output"] == str(report_path)
    assert not any(key in summary for key in ("source_identity", "context_sha256", "candidate_sha256", "stage_candidate_sha256"))
    assert report["candidate_sha256"] == skill._digest(quality)
    before = quality_bundle.read_bytes()
    with pytest.raises(SystemExit):
        skill.main(["prepare-quality", "--stage-bundle", str(stage_bundle),
                    "--stage-candidate", str(stage_candidate), "--output", str(quality_bundle)])
    assert quality_bundle.read_bytes() == before


def test_stage_evidence_refinement_keeps_mode_and_original_pixels(raw, tmp_path, monkeypatch):
    from scripts.prepare_annotation_evidence import prepare
    from scripts.refine_annotation_evidence import run

    source, views, _, _, _ = raw
    settings = {"views": views, "task": "Synthetic", "task_instruction": "Place tool",
                "stage_semantics": "active_subtask", "task_policy": {"version": "v1", "status": "draft", "rules": "Visible only"},
                "registry": [{"id": "S1", "label": "1", "prompt": "Acquire tool"},
                             {"id": "S2", "label": "2", "prompt": "Place tool"}]}
    baseline = tmp_path / "baseline"
    prepare(source, 0, settings, baseline, start_frame=2, end_frame=6, stride_seconds=.3,
            max_images=2, tile_size=(32, 24))
    stages = {"stages": {"segments": [interval(0, 2, stage_id="S1", identity_status="resolved"),
                                      interval(2, 4, stage_id="S2", identity_status="resolved", evidence_frames=[3])],
                         "boundaries": [boundary(2) | {"evidence_frames": [0, 3]}],
                         "omitted_stages": [], "invalid_intervals": [], "summary": "Synthetic"}}
    monkeypatch.setenv(KEY_ENV, "synthetic-offline-key")
    driver.run(baseline / "bundle.json", baseline / "evidence.json", tmp_path / "candidate",
               transport=httpx.MockTransport(lambda r: httpx.Response(200, json=provider(json.loads(r.content), stages))))
    output = tmp_path / "dense"
    run(baseline, tmp_path / "candidate", settings, output)
    old, _, before = driver.prepare_request(baseline / "bundle.json", baseline / "evidence.json")
    new, _, after = driver.prepare_request(output / "bundle.json", output / "evidence.json")
    assert old["mode"] == new["mode"] == "stages"
    assert {(r["frame"], r["sha256"]) for r in before} < {(r["frame"], r["sha256"]) for r in after}
    quality_path = tmp_path / "dense-quality.json"
    quality_path.write_text(json.dumps(skill.prepare_quality_bundle(old, stages, context=new["context"])))
    bound, _, images = driver.prepare_request(quality_path, output / "evidence.json")
    assert bound["stage_lock"]["candidate"] == stages and len(images) == 4
    plan = json.loads((output / "refinement_plan.json").read_text())
    assert not any(w["family"] == "nominal_audit" for w in plan["windows"])
