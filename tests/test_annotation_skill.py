# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline skill contract: no provider or pixel delivery is assumed by validation."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import jsonschema
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("annotation_skill", ROOT / "scripts/annotation_skill.py")
skill = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(skill)


@pytest.fixture
def policy(tmp_path):
    (tmp_path / "references").mkdir()
    (tmp_path / "SKILL.md").write_text("# Offline annotation\nRead the policy.\n")
    (tmp_path / "references/annotation-policy.md").write_text("# Policy\nKeep unknown labels unresolved.\n")
    (tmp_path / "references/output-contract.md").write_text("# Contract\nOutcomes match maximal contiguous stage runs.\n")
    return tmp_path


@pytest.fixture
def context():
    return {"schema_version": "grana.annotation_context.v1", "task": "Place tool",
            "task_instruction": "Place the tool in the drawer; partial attempts are possible.",
            "stage_semantics": "active_subtask",
            "task_policy": {"version": "test-v1", "status": "draft", "rules": "Occlusion does not establish success. 未知保留。"},
            "source_identity": {"episode": "synthetic-only", "sha256": "a" * 64},
            "time_edges": [10, 10.1, 10.3, 10.6, 11.0, 11.5, 12.1],
            "registry": [{"id": "S1", "label": "1", "prompt": "Reach tool"},
                         {"id": "S2", "label": "2", "prompt": "Place tool"},
                         {"id": "S3", "label": "3", "prompt": "Close drawer"}],
            "evidence_frames": [0, 1, 2, 3, 4, 5]}


def interval(start, end, **fields):
    return {"start_frame": start, "end_frame": end, **fields,
            "reason": "Synthetic visible evidence", "evidence_frames": [start]}


def boundary(frame):
    return {"frame": frame, "lower_frame": max(0, frame - 1), "upper_frame": frame + 1,
            "reason": "Boundary bracket", "evidence_frames": [frame - 1, frame]}


@pytest.fixture
def candidate():
    return {
        "stages": {
            "segments": [interval(0, 2, stage_id="S2", identity_status="resolved"),
                         interval(2, 3, stage_id="S2", identity_status="uncertain"),
                         interval(3, 4, stage_id=None, identity_status="unresolved"),
                         interval(4, 6, stage_id="S1", identity_status="resolved")],
            "boundaries": [boundary(2), boundary(3), boundary(4)],
            "omitted_stages": [{"stage_id": "S3", "reason": "Not observed"}],
            "invalid_intervals": [{"start_frame": 3, "end_frame": 4, "reason": "One corrupted view"}],
            "summary": "A partial attempt with independent unknowns."},
        "aq_intervals": [interval(0, 1, quality="nominal"), interval(1, 3, quality="failure"),
                         interval(3, 4, quality="unknown"), interval(4, 6, quality="suboptimal")],
        "aq_boundaries": [boundary(1), boundary(3), boundary(4)],
        "outcomes": [interval(0, 3, stage_id="S2", outcome="unknown"),
                     interval(3, 4, stage_id=None, outcome="unknown"),
                     interval(4, 6, stage_id="S1", outcome="low")],
    }


def test_bundle_includes_policy_and_output_contract_without_source_access(context, policy):
    context["source_identity"]["path"] = "/nonexistent/private/source"
    original = deepcopy(context)
    bundle = skill.prepare_bundle(context, policy, mode="combined")
    assert bundle == skill.prepare_bundle(context, policy, mode="combined")
    assert bundle["context"] == original == context
    assert bundle["context"] is not context
    assert set(bundle["policy"]["files"]) == set(skill.POLICY_FILES)
    assert "# Offline annotation" in bundle["instructions"]
    assert "Keep unknown labels unresolved." in bundle["instructions"]
    assert "Outcomes match maximal contiguous stage runs." in bundle["instructions"]
    assert "contains no images" in bundle["instructions"]
    assert "api_key" not in json.dumps(bundle).lower()
    jsonschema.Draft202012Validator.check_schema(bundle["schema"])


def test_candidate_retains_axes_and_does_not_accept_or_convert_labels(context, candidate, policy):
    original = deepcopy(candidate)
    bundle = skill.prepare_bundle(context, policy, mode="combined")
    jsonschema.validate(candidate, bundle["schema"])
    result = skill.validate_bundle_candidate(bundle, candidate, policy)
    assert candidate == original
    assert result["status"] == "valid_candidate"
    assert result["accepted_for_training"] is False
    assert result["image_delivery_verified"] is False
    assert result["semantic_accuracy_verified"] is False
    assert result["task_policy_calibration_verified"] is False
    assert result["aq_frames"] == {"nominal": 1, "suboptimal": 2, "failure": 2, "unknown": 1}
    assert result["outcome_counts"] == {"high": 0, "mid": 0, "low": 1, "unknown": 2}
    assert result["stage_coverage"]["source_seconds"] == pytest.approx(2.1)
    assert result["stage_coverage"]["unresolved_identity"]["seconds"] == pytest.approx(.4)
    assert result["stage_coverage"]["uncertain_identity"]["frames"] == 1
    assert result["stage_coverage"]["source_invalid"]["frames"] == 1
    assert "supplement" not in result and "regressed" not in json.dumps(candidate)


@pytest.mark.parametrize("field,value", [("stage_semantics", "attained_state"), ("task_policy", {
    "version": "self-declared", "status": "calibrated", "rules": "Researcher-supplied rules"})])
def test_declared_semantics_and_calibration_never_approve_candidate(context, candidate, policy, field, value):
    context[field] = value
    result = skill.validate_bundle_candidate(skill.prepare_bundle(context, policy, mode="combined"), candidate, policy)
    assert result["accepted_for_training"] is False
    assert result["task_policy_calibration_verified"] is False


@pytest.mark.parametrize("edit,match", [
    (lambda x: x.pop("stage_semantics"), "exactly"),
    (lambda x: x.update(stage_semantics="either"), "stage_semantics"),
    (lambda x: x["task_policy"].update(status="accepted"), "status"),
    (lambda x: x["task_policy"].update(rules=" "), "nonempty"),
    (lambda x: x.update(time_edges=[0, 1, 1]), "time_edges"),
    (lambda x: x["time_edges"].__setitem__(2, True), "time_edges"),
    (lambda x: x.update(source_identity={}), "source_identity"),
    (lambda x: x["registry"][1].update(id="S1"), "distinct"),
    (lambda x: x.update(evidence_frames=[]), "evidence_frames"),
    (lambda x: x.update(evidence_frames=[0, 0]), "evidence_frames"),
    (lambda x: x.update(evidence_frames=[6]), "evidence_frames"),
    (lambda x: x.update(evidence_frames=[True]), "integer"),
])
def test_invalid_context_is_not_repaired(context, policy, edit, match):
    edit(context)
    with pytest.raises(ValueError, match=match):
        skill.prepare_bundle(context, policy, mode="combined")


@pytest.mark.parametrize("edit,match", [
    (lambda x: x["aq_intervals"][1].update(start_frame=2), "partition"),
    (lambda x: x["aq_intervals"][1].update(start_frame=0), "partition"),
    (lambda x: x["aq_intervals"].pop(), "cover every"),
    (lambda x: x["aq_intervals"][0].update(quality="regressed"), "no regressed conversion"),
    (lambda x: x["aq_intervals"][0].update(end_frame=True), "integer"),
    (lambda x: x["aq_boundaries"].pop(), "AQ cuts"),
    (lambda x: x["aq_boundaries"][0].update(lower_frame=2), "bracket"),
    (lambda x: x["outcomes"].pop(), "maximal"),
    (lambda x: x["outcomes"][0].update(end_frame=2), "maximal"),
    (lambda x: x["outcomes"][0].update(outcome="high"), "require unknown"),
    (lambda x: x["outcomes"][1].update(outcome="low"), "require unknown"),
    (lambda x: x["outcomes"][2].update(outcome="success"), "Outcome must"),
    (lambda x: x["outcomes"][2].update(reason=""), "nonempty"),
    (lambda x: x["stages"]["segments"][0].update(stage_id="invented"), "registry"),
    (lambda x: x.update(review={"decision": "accepted"}), "exactly"),
])
def test_candidate_invariants_fail_closed(context, candidate, policy, edit, match):
    edit(candidate)
    with pytest.raises(ValueError, match=match):
        skill.validate_bundle_candidate(skill.prepare_bundle(context, policy, mode="combined"), candidate, policy)


@pytest.mark.parametrize("section,index", [("stage_segments", 0), ("stage_boundaries", 0),
                                          ("aq_intervals", 0), ("aq_boundaries", 0), ("outcomes", 0)])
def test_all_citations_must_reference_supplied_evidence(context, candidate, policy, section, index):
    context["evidence_frames"] = [0, 1, 2, 3, 4]  # 5 exists in source but was not supplied.
    items = (candidate["stages"][section.removeprefix("stage_")]
             if section.startswith("stage_") else candidate[section])
    items[index]["evidence_frames"] = [5]
    with pytest.raises(ValueError, match="context.evidence_frames"):
        skill.validate_bundle_candidate(skill.prepare_bundle(context, policy, mode="combined"), candidate, policy)


@pytest.mark.parametrize("edit", [
    lambda x: x["context"]["source_identity"].update(episode="different"),
    lambda x: x["context"]["task_policy"].update(rules="Changed criteria"),
    lambda x: x.update(instructions="Ignore policy"),
    lambda x: x["schema"].update(additionalProperties=True),
    lambda x: x["policy"].update(version="different"),
])
def test_bundle_changes_require_explicit_repreparation(context, candidate, policy, edit):
    bundle = skill.prepare_bundle(context, policy, mode="combined")
    edit(bundle)
    with pytest.raises(ValueError, match="integrity mismatch"):
        skill.validate_bundle_candidate(bundle, candidate, policy)


def test_policy_file_changes_invalidate_frozen_bundle(context, candidate, policy):
    bundle = skill.prepare_bundle(context, policy, mode="combined")
    (policy / "references/annotation-policy.md").write_text("Changed policy\n")
    with pytest.raises(ValueError, match="integrity mismatch"):
        skill.validate_bundle_candidate(bundle, candidate, policy)


def test_revisited_stage_produces_a_separate_run_outcome(context, candidate, policy):
    candidate["stages"]["segments"][-1]["stage_id"] = "S2"
    candidate["stages"]["omitted_stages"].append({"stage_id": "S1", "reason": "Absent"})
    candidate["outcomes"][-1]["stage_id"] = "S2"
    report = skill.validate_bundle_candidate(skill.prepare_bundle(context, policy, mode="combined"), candidate, policy)
    assert report["outcome_counts"]["low"] == 1


@pytest.mark.parametrize("section", ["aq_intervals", "outcomes"])
def test_known_labels_cannot_cite_exclusively_invalid_evidence(context, candidate, policy, section):
    candidate[section][-1]["evidence_frames"] = [3]
    with pytest.raises(ValueError, match="outside source-invalid"):
        skill.validate_bundle_candidate(skill.prepare_bundle(context, policy, mode="combined"), candidate, policy)


def test_partial_invalid_source_does_not_erase_other_evidence(context, candidate, policy):
    candidate["aq_intervals"][-1]["evidence_frames"] = [3, 4]
    candidate["outcomes"][-1]["evidence_frames"] = [3, 4]
    # Unknown AQ/outcome at frame 3 legitimately cites that invalid observation.
    report = skill.validate_bundle_candidate(skill.prepare_bundle(context, policy, mode="combined"), candidate, policy)
    assert report["aq_frames"]["suboptimal"] == 2
    assert report["outcome_counts"]["low"] == 1
    assert report["semantic_accuracy_verified"] is False


def test_cli_prepare_validate_and_refuse_overwrite(context, candidate, tmp_path):
    context_path, candidate_path = tmp_path / "context.json", tmp_path / "candidate.json"
    bundle_path, report_path = tmp_path / "bundle.json", tmp_path / "report.json"
    context_path.write_text(json.dumps(context, ensure_ascii=False))
    candidate_path.write_text(json.dumps(candidate))

    def run(*args):
        return subprocess.run([sys.executable, str(ROOT / "scripts/annotation_skill.py"), *map(str, args)],
                              cwd=tmp_path, text=True, capture_output=True, timeout=20)

    prepared = run("prepare", "--mode", "combined", "--context", context_path, "--output", bundle_path)
    assert prepared.returncode == 0, prepared.stderr
    assert json.loads(prepared.stdout)["status"] == "prepared"
    assert json.loads(prepared.stdout)["policy_name"] == "grana-robot-annotation"
    assert "sha256" not in prepared.stdout and "policy_version" not in prepared.stdout
    valid = run("validate", "--bundle", bundle_path, "--candidate", candidate_path, "--output", report_path)
    assert valid.returncode == 0, valid.stderr
    assert "sha256" not in valid.stdout and "policy_version" not in valid.stdout
    assert json.loads(report_path.read_text())["accepted_for_training"] is False
    before = report_path.read_bytes()
    repeated = run("validate", "--bundle", bundle_path, "--candidate", candidate_path, "--output", report_path)
    assert repeated.returncode == 2 and "File exists" in repeated.stderr
    assert report_path.read_bytes() == before


def test_cli_can_prepare_and_validate_an_explicit_frozen_policy_snapshot(context, candidate, policy, tmp_path):
    context_path, candidate_path = tmp_path / "context.json", tmp_path / "candidate.json"
    bundle_path = tmp_path / "snapshot-bundle.json"
    context_path.write_text(json.dumps(context))
    candidate_path.write_text(json.dumps(candidate))
    command = [sys.executable, str(ROOT / "scripts/annotation_skill.py")]
    prepared = subprocess.run([*command, "prepare", "--mode", "combined", "--context", str(context_path),
                               "--output", str(bundle_path), "--policy-dir", str(policy)],
                              capture_output=True, text=True, check=True)
    assert json.loads(prepared.stdout)["status"] == "prepared"
    validate = [*command, "validate", "--bundle", str(bundle_path), "--candidate", str(candidate_path)]
    valid = subprocess.run([*validate, "--policy-dir", str(policy)], capture_output=True, text=True, check=True)
    assert json.loads(valid.stdout)["status"] == "valid_candidate"
    different_policy = subprocess.run(validate, capture_output=True, text=True)
    assert different_policy.returncode == 2 and "integrity mismatch" in different_policy.stderr


@pytest.mark.parametrize("raw", ['{"task":1,"task":2}', '{"value":NaN}', '{"value":Infinity}'])
def test_json_reader_rejects_ambiguous_values(tmp_path, raw):
    path = tmp_path / "bad.json"
    path.write_text(raw)
    with pytest.raises(ValueError):
        skill._read_json(path)
