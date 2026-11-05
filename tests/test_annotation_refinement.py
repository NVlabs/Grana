# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded adaptive evidence never consumes labels, drops grid frames or calls an API."""
from copy import deepcopy
import json

import pytest

from grana.annotation.refinement import digest, plan_refinement
from grana.annotation.source import CLOCK_COLUMNS
from scripts.annotation_skill import POLICY_FILES, prepare_bundle, validate_bundle_candidate
from scripts.annotation_skill_api import prepare_request
from scripts.prepare_annotation_evidence import prepare
from scripts.refine_annotation_evidence import run
from test_annotation_source import raw


def segment(start, end, evidence, **extra):
    return {"start_frame": start, "end_frame": end, "reason": "Synthetic visible action",
            "evidence_frames": evidence, **extra}


def boundary(frame, evidence, lower=None, upper=None):
    return {"frame": frame, "lower_frame": frame if lower is None else lower,
            "upper_frame": frame if upper is None else upper,
            "reason": "Synthetic uncertainty bracket", "evidence_frames": evidence}


@pytest.fixture
def policy(tmp_path):
    directory = tmp_path / "policy"
    for name in POLICY_FILES:
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("Synthetic policy; unsupported evidence remains unknown.\n")
    return directory


@pytest.fixture
def planned(policy):
    frames = [*range(0, 900, 15), 899]
    context = {"schema_version": "grana.annotation_context.v1", "task": "Synthetic assembly",
        "task_instruction": "Insert a part, then place the tool", "stage_semantics": "active_subtask",
        "task_policy": {"version": "synthetic-v1", "status": "draft", "rules": "Use visible evidence."},
        "source_identity": {"source_frame_range": [150, 1050], "episode_index": 0},
        "time_edges": [100 + frame / 30 for frame in range(901)],
        "registry": [{"id": "S1", "label": "1", "prompt": "Insert part"},
                     {"id": "S2", "label": "2", "prompt": "Place tool"}], "evidence_frames": frames}
    candidate = {"stages": {
        "segments": [segment(0, 600, [0, 300], stage_id="S1", identity_status="resolved"),
                     segment(600, 900, [600, 899], stage_id="S2", identity_status="resolved")],
        "boundaries": [boundary(600, [585, 600], 585, 615)],
        "omitted_stages": [], "invalid_intervals": [], "summary": "Synthetic only"},
        "aq_intervals": [segment(0, 120, [0], quality="nominal"),
                         segment(120, 510, [120, 495], quality="unknown"),
                         segment(510, 900, [510, 899], quality="nominal")],
        "aq_boundaries": [boundary(120, [105, 120], 90, 150), boundary(510, [495, 510], 480, 540)],
        "outcomes": [segment(0, 600, [0], stage_id="S1", outcome="unknown"),
                     segment(600, 900, [600], stage_id="S2", outcome="unknown")]}
    bundle = prepare_bundle(context, policy, mode="combined")
    validation = validate_bundle_candidate(bundle, candidate, policy)
    receipt = {"source_identity": context["source_identity"], "annotation_columns_read": False,
        "sampling": {"candidate_frames": frames, "source_frames": [f + 150 for f in frames],
                     "grid_origin_seconds": 100}}
    return bundle, candidate, validation, receipt


def plan(planned, policy, **kwargs):
    return plan_refinement(*planned, policy_dir=policy, **kwargs)


def test_deterministic_bounded_windows_keep_all_original_frames_and_native_offset(planned, policy):
    result = plan(planned, policy, max_extra_frames=80, max_total_images=120)
    assert result == plan(planned, policy, max_extra_frames=80, max_total_images=120)
    assert result["human_annotations_read"] is False and result["accepted_for_training"] is False
    assert set(result["baseline_frames"]) <= set(result["expanded_frames"])
    assert not set(result["extra_frames"]) & set(result["baseline_frames"])
    assert result["extra_source_frames"] == [f + 150 for f in result["extra_frames"]]
    assert len(result["extra_frames"]) <= 80 and len(result["expanded_frames"]) <= 120
    assert result["native_time_range"] == [100, 130]
    assert all(w["native_time_range"][1] - w["native_time_range"][0] <= 2.00001 for w in result["windows"])
    assert {w["family"] for w in result["windows"]} == {
        "aq_boundary", "quality_endpoint", "stage_transition", "nominal_audit"}


def test_nominal_audit_is_hash_selected_reproducible_and_confined_to_nominal(planned, policy):
    def selected(seed):
        result = plan(planned, policy, seed=seed)
        audit = [w for w in result["windows"] if w["family"] == "nominal_audit"]
        assert len(audit) == 1 and audit[0]["status"] == "selected"
        assert result["windows"][0] == audit[0]
        assert all(f < 120 or f >= 510 for f in audit[0]["requested_frames"])
        return audit[0]["triggers"][0]["frame"]
    assert selected("repeatable") == selected("repeatable")
    assert len({selected(str(seed)) for seed in range(6)}) > 1


def test_tiny_budget_skips_whole_windows_and_reports_actual_overlap(planned, policy):
    result = plan(planned, policy, max_extra_frames=1)
    assert result["extra_frames"] == []
    assert result["budget"]["unused_extra_budget"] == 1
    assert all(w["status"] == "skipped" for w in result["windows"])
    for window in result["windows"]:
        assert window["decision_reason"] == "whole_window_exceeds_remaining_budget"
        assert window["covered_requested_frames"] == len(set(window["requested_frames"]) & set(result["baseline_frames"]))
    with pytest.raises(ValueError, match="no grid frames"):
        plan(planned, policy, max_total_images=1)


def test_long_unknown_is_only_sampled_at_bounded_endpoints_not_densified(planned, policy):
    result = plan(planned, policy, max_extra_frames=500, max_total_images=600)
    middle_unknown = {f for f in result["extra_frames"] if 200 <= f <= 400}
    assert middle_unknown == set()
    families = [w["family"] for w in result["windows"][:4]]
    assert families[0] == "nominal_audit"
    assert set(families[1:]) == {"aq_boundary", "quality_endpoint", "stage_transition"}
    assert families == result["family_order"]


def test_non_audit_family_priority_varies_reproducibly_across_seeds(planned, policy):
    orders = {tuple(plan(planned, policy, seed=str(s))["family_order"]) for s in range(8)}
    assert len(orders) > 1
    assert all(order[0] == "nominal_audit" for order in orders)


@pytest.mark.parametrize("damage", ["candidate", "receipt", "offset", "frames", "clock", "labels", "extra_field"])
def test_invalid_or_unbound_input_fails_closed(planned, policy, damage):
    bundle, candidate, validation, receipt = deepcopy(planned)
    if damage == "candidate":
        candidate["aq_intervals"][0]["end_frame"] -= 1
    elif damage == "receipt":
        validation["candidate_sha256"] = "0" * 64
    elif damage == "offset":
        receipt["source_identity"] = receipt["source_identity"] | {"source_frame_range": [0, 900]}
    elif damage == "frames":
        receipt["sampling"]["source_frames"][0] = 0
    elif damage == "clock":
        receipt["sampling"]["grid_origin_seconds"] = 0
    elif damage == "labels":
        receipt["annotation_columns_read"] = True
    else:
        candidate["human_labels"] = ["failure"]
    with pytest.raises(ValueError):
        plan_refinement(bundle, candidate, validation, receipt, policy_dir=policy)


@pytest.mark.parametrize("kwargs", [{"max_extra_frames": True}, {"max_total_images": 1.5},
    {"radius_seconds": 0}, {"dense_hz": float("nan")}, {"seed": ""}])
def test_bad_parameters_rejected(planned, policy, kwargs):
    with pytest.raises(ValueError):
        plan(planned, policy, **kwargs)


def make_source_inputs(raw, tmp_path, policy):
    source, views, _, _, _ = raw
    settings = {"views": views, "task": "Synthetic", "task_instruction": "Place tool",
        "stage_semantics": "active_subtask", "task_policy": {"version": "v1", "status": "draft", "rules": "Visible only"},
        "registry": [{"id": "S1", "label": "1", "prompt": "Place tool"}]}
    baseline = tmp_path / "baseline"
    prepare(source, 0, settings, baseline, start_frame=2, end_frame=6, stride_seconds=.3,
            max_images=2, tile_size=(32, 24), policy_dir=policy, mode="combined")
    bundle, _, verified = prepare_request(baseline / "bundle.json", baseline / "evidence.json", policy)
    candidate = {"stages": {"segments": [segment(0, 4, [0, 3], stage_id="S1", identity_status="resolved")],
        "boundaries": [], "omitted_stages": [], "invalid_intervals": [], "summary": "Synthetic"},
        "aq_intervals": [segment(0, 4, [0, 3], quality="nominal")], "aq_boundaries": [],
        "outcomes": [segment(0, 4, [0, 3], stage_id="S1", outcome="unknown")]}
    candidate_dir = tmp_path / "candidate"
    candidate_dir.mkdir()
    delivery = {"status": "completed", "accepted_for_training": False, "bundle_sha256": digest(bundle),
        "candidate_sha256": digest(candidate), "context_sha256": bundle["context_sha256"],
        "frame_order": [0, 3], "images": verified}
    for name, value in (("candidate", candidate), ("validation", validate_bundle_candidate(bundle, candidate, policy)),
                         ("delivery", delivery)):
        (candidate_dir / f"{name}.json").write_text(json.dumps(value))
    return baseline, candidate_dir, settings


def test_real_source_expansion_preserves_bytes_grid_clock_and_idempotence_without_labels(raw, tmp_path, policy, monkeypatch):
    import pyarrow.parquet as pq
    baseline, candidate_dir, settings = make_source_inputs(raw, tmp_path, policy)
    original_read = pq.read_table
    calls = []
    def clock_only(path, **kwargs):
        calls.append(kwargs["columns"])
        assert kwargs["columns"] == list(CLOCK_COLUMNS)
        return original_read(path, **kwargs)
    monkeypatch.setattr(pq, "read_table", clock_only)
    output = tmp_path / "expanded"
    kwargs = {"policy_dir": policy, "max_extra_frames": 2, "max_total_images": 4}
    result = run(baseline, candidate_dir, settings, output, **kwargs)
    assert result["extra_frames"] == 2 and result["total_images"] == 4
    old = json.loads((baseline / "evidence.json").read_text())["images"]
    new = json.loads((output / "evidence.json").read_text())["images"]
    assert [row["frame"] for row in new] == [0, 1, 2, 3]
    assert [row["timestamp"] for row in new] == pytest.approx([10.2, 10.3, 10.4, 10.5])
    assert {row["frame"]: row["sha256"] for row in old} == {row["frame"]: row["sha256"] for row in new if row["frame"] in (0, 3)}
    receipt = json.loads((output / "refinement_receipt.json").read_text())
    assert receipt["api_calls_made"] == 0 and receipt["human_annotations_read"] is False
    assert json.loads((output / "source_receipt.json").read_text())["sampling"]["source_frames"] == [2, 3, 4, 5]
    resumed = run(baseline, candidate_dir, settings, output, resume=True, **kwargs)
    assert resumed["resumed"] is True and calls
    with pytest.raises(FileExistsError):
        run(baseline, candidate_dir, settings, output, **kwargs)
    with pytest.raises(ValueError, match="different inputs"):
        run(baseline, candidate_dir, settings, output, resume=True, policy_dir=policy, max_extra_frames=0, max_total_images=4)
    new[1]["sha256"] = "0" * 64
    (output / "evidence.json").write_text(json.dumps({"schema_version": "grana.annotation_evidence.v1", "images": new}))
    with pytest.raises(ValueError, match="Evidence image changed"):
        run(baseline, candidate_dir, settings, output, resume=True, **kwargs)


@pytest.mark.parametrize("damage", ["delivery", "settings", "source", "input_pixels"])
def test_preparation_rejects_changed_bindings_before_creating_output(raw, tmp_path, policy, damage):
    baseline, candidate_dir, settings = make_source_inputs(raw, tmp_path, policy)
    if damage == "delivery":
        path = candidate_dir / "delivery.json"
        value = json.loads(path.read_text())
        value["candidate_sha256"] = "0" * 64
        path.write_text(json.dumps(value))
    elif damage == "settings":
        settings["task_instruction"] = "Changed task"
    elif damage == "source":
        path = raw[0] / "meta/episodes.jsonl"
        path.write_text(path.read_text() + "\n")
    else:
        manifest = json.loads((baseline / "evidence.json").read_text())
        from pathlib import Path
        Path(manifest["images"][0]["image_path"]).write_bytes(b"changed")
    output = tmp_path / "expanded"
    with pytest.raises(ValueError):
        run(baseline, candidate_dir, settings, output, policy_dir=policy)
    assert not output.exists()


def test_output_cannot_enter_raw_source_or_checkout(raw, tmp_path, policy):
    from scripts.refine_annotation_evidence import ROOT
    baseline, candidate_dir, settings = make_source_inputs(raw, tmp_path, policy)
    for output in (raw[0] / "forbidden", ROOT / ".local/forbidden-refinement-test"):
        with pytest.raises(ValueError, match="outside"):
            run(baseline, candidate_dir, settings, output, policy_dir=policy)
        assert not output.exists()
