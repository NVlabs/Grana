# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stage-window preparation is offline, hypothesis-based and native-clock bound."""
from copy import deepcopy
import json
from pathlib import Path
from fractions import Fraction
import random

import pytest

from grana.annotation.windows import plan_stage_windows
from grana.annotation.source import CLOCK_COLUMNS
from scripts.annotation_skill_api import prepare_request
from scripts.prepare_annotation_evidence import select_frames
from scripts.prepare_annotation_windows import run
from test_annotation_source import raw
from test_annotation_refinement import make_source_inputs, policy


@pytest.fixture
def planned():
    context = {"time_edges": [50 + frame / 10 for frame in range(1001)],
               "source_identity": {"source_frame_range": [200, 1200],
                   "frame_mapping": "native_frame = candidate_frame + source_frame_range[0]"},
               "registry": [{"id": "S1"}, {"id": "S2"}]}
    candidate = {"stages": {
        "segments": [
            {"start_frame": 0, "end_frame": 300, "stage_id": "S1", "identity_status": "resolved"},
            {"start_frame": 300, "end_frame": 730, "stage_id": "S1", "identity_status": "uncertain"},
            {"start_frame": 730, "end_frame": 1000, "stage_id": "S2", "identity_status": "resolved"}],
        "boundaries": [{"frame": 730, "lower_frame": 720, "upper_frame": 740}],
        "invalid_intervals": [{"start_frame": 800, "end_frame": 810}]}}
    return context, candidate


def test_core_partition_long_run_splits_and_context_native_offsets(planned):
    context, candidate = planned
    before = deepcopy(planned)
    result = plan_stage_windows(context, candidate)
    assert planned == before
    assert result == plan_stage_windows(context, candidate)
    windows = result["windows"]
    assert [w["core_frame_range"] for w in windows] == [[0, 300], [300, 600], [600, 730], [730, 1000]]
    assert [w["predicted_run_index"] for w in windows] == [0, 0, 0, 1]
    assert [w["context_frame_range"] for w in windows] == [[0, 320], [280, 620], [580, 750], [710, 1000]]
    assert windows[1]["source_frame_range"] == [480, 820]
    assert windows[1]["source_core_frame_range"] == [500, 800]
    assert windows[1]["window_local_core_range"] == [20, 320]
    assert windows[1]["native_core_time_range"] == [80, 110]
    assert windows[1]["native_context_time_range"] == [78, 112]
    for w in windows:
        first, last = w["context_frame_range"]
        expected = select_frames(context["time_edges"], first, last, .5, 128)
        assert w["source_evidence_frames"] == [f + 200 for f in expected]
        assert w["window_evidence_frames"] == [f - first for f in expected]
        assert w["expected_images"] == len(expected)
        assert w["stage_hypothesis"]["enforced"] is False
    assert result["accepted_for_training"] is False


def test_uncertain_invalid_and_boundary_evidence_is_flagged_not_forced(planned):
    result = plan_stage_windows(*planned)
    assert result["windows"][0]["stage_hypothesis"]["identity_statuses"] == ["resolved", "uncertain"]
    assert "coarse_stage_identity_uncertain_or_unresolved" in result["windows"][0]["review_reasons"]
    assert "coarse_stage_boundary_uncertainty_intersects_context" in result["windows"][2]["review_reasons"]
    assert "coarse_source_invalid_interval_intersects_context" in result["windows"][3]["review_reasons"]
    assert all(w["requires_review"] for w in result["windows"])
    assert result["windows"][3]["predicted_run_was_split"] is False
    assert "not a whole-stage outcome" in result["windows"][3]["outcome_scope"]


def test_unresolved_runs_still_cover_clip_with_no_stage_hint_claim(planned):
    context, candidate = deepcopy(planned)
    candidate["stages"]["segments"] = [{"start_frame": 0, "end_frame": 1000,
                                         "stage_id": None, "identity_status": "unresolved"}]
    result = plan_stage_windows(context, candidate, max_core_seconds=45)
    assert [w["core_frame_range"] for w in result["windows"]] == [[0, 450], [450, 900], [900, 1000]]
    assert all(w["stage_hypothesis"]["stage_id"] is None for w in result["windows"])
    assert all(w["requires_review"] for w in result["windows"])


def test_nonuniform_clock_and_zero_context_respect_time_cap():
    context = {"time_edges": [20, 20.1, 20.5, 21.4, 21.5, 22],
               "source_identity": {"source_frame_range": [10, 15],
                   "frame_mapping": "native_frame = candidate_frame + source_frame_range[0]"},
               "registry": [{"id": "S1"}]}
    candidate = {"stages": {"segments": [{"start_frame": 0, "end_frame": 5,
                                           "stage_id": "S1", "identity_status": "resolved"}]}}
    result = plan_stage_windows(context, candidate, max_core_seconds=1, context_seconds=0)
    assert [w["core_frame_range"] for w in result["windows"]] == [[0, 2], [2, 4], [4, 5]]
    assert all(w["core_frame_range"] == w["context_frame_range"] for w in result["windows"])
    with pytest.raises(ValueError, match="single native frame"):
        plan_stage_windows(context, candidate, max_core_seconds=.5)


def test_varied_nonuniform_clocks_keep_every_core_frame_once_and_exact_offsets():
    rng = random.Random(3817)
    for _ in range(30):
        count, offset = rng.randint(30, 120), rng.randint(1, 10000)
        ticks = [rng.randint(1000, 10000)]
        for _ in range(count):
            ticks.append(ticks[-1] + rng.choice([3, 5, 7, 11, 17]))
        times = [tick / 100 for tick in ticks]
        cuts = [0, *sorted(rng.sample(range(1, count), 3)), count]
        context = {"time_edges": times, "registry": [{"id": "S1"}, {"id": "S2"}],
                   "source_identity": {"source_frame_range": [offset, offset + count],
                       "frame_mapping": "native_frame = candidate_frame + source_frame_range[0]"}}
        candidate = {"stages": {"segments": [
            {"start_frame": a, "end_frame": b, "stage_id": "S1" if index % 2 == 0 else "S2",
             "identity_status": "resolved"} for index, (a, b) in enumerate(zip(cuts, cuts[1:]))]}}
        plan = plan_stage_windows(context, candidate, max_core_seconds=.8, context_seconds=.23,
                                  stride_seconds=.13, max_images=128)
        covered = []
        for window in plan["windows"]:
            a, b = window["core_frame_range"]; left, right = window["context_frame_range"]
            covered.extend(range(a, b))
            assert left <= a < b <= right
            assert Fraction(str(times[b])) - Fraction(str(times[a])) <= Fraction(".8")
            assert Fraction(str(times[a])) - Fraction(str(times[left])) <= Fraction(".23")
            assert Fraction(str(times[right])) - Fraction(str(times[b])) <= Fraction(".23")
            assert window["source_core_frame_range"] == [offset + a, offset + b]
            assert window["source_frame_range"] == [offset + left, offset + right]
            assert window["window_local_core_range"] == [a - left, b - left]
            assert window["window_evidence_frames"] == [f - left for f in select_frames(times, left, right, .13, 128)]
        assert covered == list(range(count))


def test_image_limit_is_explicit_including_context_no_frame_drops(planned):
    with pytest.raises(ValueError, match="context window exceeds max_images"):
        plan_stage_windows(*planned, max_images=60)
    result = plan_stage_windows(*planned, max_core_seconds=20, context_seconds=1, max_images=60)
    assert all(w["expected_images"] <= 60 for w in result["windows"])


@pytest.mark.parametrize("change,error", [
    (lambda c, p: c["time_edges"].__setitem__(2, 50), "strictly increasing"),
    (lambda c, p: c["time_edges"].__setitem__(2, float("nan")), "finite"),
    (lambda c, p: c["source_identity"].update(source_frame_range=[200, 1199]), "offset"),
    (lambda c, p: c["source_identity"].pop("frame_mapping"), "frame_mapping"),
    (lambda c, p: p["stages"]["segments"][1].update(start_frame=301), "partition"),
    (lambda c, p: p["stages"]["segments"][-1].update(end_frame=999), "partition"),
    (lambda c, p: p["stages"]["segments"][0].update(stage_id="invented"), "registry ID"),
    (lambda c, p: p["stages"]["segments"][0].update(identity_status="unresolved"), "identity_status"),
    (lambda c, p: p["stages"]["boundaries"][0].update(lower_frame=740), "bracket"),
])
def test_bad_clock_mapping_or_stage_hypothesis_fails_closed(planned, change, error):
    context, candidate = deepcopy(planned)
    change(context, candidate)
    with pytest.raises(ValueError, match=error):
        plan_stage_windows(context, candidate)


@pytest.mark.parametrize("kwargs", [{"max_core_seconds": 0}, {"context_seconds": -1},
                                    {"stride_seconds": True}, {"max_images": 1.5}])
def test_bad_parameters_fail(planned, kwargs):
    with pytest.raises(ValueError):
        plan_stage_windows(*planned, **kwargs)


def test_plan_then_prepare_one_window_preserves_native_clock_and_unforced_registry(raw, tmp_path, policy, monkeypatch):
    import pyarrow.parquet as pq
    baseline, candidate_dir, settings = make_source_inputs(raw, tmp_path, policy)
    read_table = pq.read_table
    calls = []
    def clock_only(path, **kwargs):
        calls.append(kwargs["columns"])
        assert kwargs["columns"] == list(CLOCK_COLUMNS)
        return read_table(path, **kwargs)
    monkeypatch.setattr(pq, "read_table", clock_only)
    output = tmp_path / "stage-windows"
    kwargs = {"policy_dir": policy, "max_core_seconds": .2, "context_seconds": .1,
              "stride_seconds": .1, "max_images": 4}
    planned = run(baseline, candidate_dir, settings, output, **kwargs)
    assert planned["status"] == "planned" and planned["windows"] == 2
    assert not (output / "windows").exists()
    result = run(baseline, candidate_dir, settings, output, window_id="window_0001", resume=True, **kwargs)
    destination = Path(result["output"])
    bundle, _, images = prepare_request(destination / "bundle.json", destination / "evidence.json", policy)
    assert bundle["context"]["source_identity"]["source_frame_range"] == [3, 6]
    assert bundle["context"]["time_edges"] == pytest.approx([10.3, 10.4, 10.5, 10.6])
    assert [r["frame"] for r in images] == [0, 1, 2]
    assert [r["timestamp"] for r in images] == pytest.approx([10.3, 10.4, 10.5])
    for key in ("registry", "task", "task_instruction", "task_policy", "stage_semantics"):
        assert bundle["context"][key] == settings[key]
    assert "stage_hypothesis" not in bundle["context"]
    receipt = json.loads((destination / "window_receipt.json").read_text())
    assert receipt["source_core_frame_range"] == [4, 6]
    assert receipt["window_local_core_range"] == [1, 3]
    assert receipt["stage_hint_sent_to_api_context"] is False
    assert receipt["api_calls_made"] == 0 and receipt["human_annotations_read"] is False
    assert not (output / "windows/window_0000").exists()
    again = run(baseline, candidate_dir, settings, output, window_id="window_0001", resume=True, **kwargs)
    assert again["resumed"] is True and calls
    with pytest.raises(FileExistsError):
        run(baseline, candidate_dir, settings, output, **kwargs)
    with pytest.raises(ValueError, match="different inputs or parameters"):
        run(baseline, candidate_dir, settings, output, resume=True, policy_dir=policy)


@pytest.mark.parametrize("damage", ["delivery", "validation", "settings", "source", "pixels", "source_receipt"])
def test_changed_coarse_bindings_rejected_before_output_creation(raw, tmp_path, policy, damage):
    baseline, candidate_dir, settings = make_source_inputs(raw, tmp_path, policy)
    if damage in ("delivery", "validation"):
        path = candidate_dir / (damage + ".json")
        data = json.loads(path.read_text()); data["candidate_sha256"] = "0" * 64
        path.write_text(json.dumps(data))
    elif damage == "settings":
        settings["registry"][0]["prompt"] = "A different task"
    elif damage == "source":
        path = raw[0] / "meta/episodes.jsonl"; path.write_text(path.read_text() + "\n")
    elif damage == "source_receipt":
        path = baseline / "source_receipt.json";data=json.loads(path.read_text())
        data["sampling"]["source_frames"][0] += 1;path.write_text(json.dumps(data))
    else:
        manifest = json.loads((baseline / "evidence.json").read_text())
        Path(manifest["images"][0]["image_path"]).write_bytes(b"tampered")
    output = tmp_path / "rejected"
    with pytest.raises(ValueError):
        run(baseline, candidate_dir, settings, output, policy_dir=policy)
    assert not output.exists()


def test_resume_rejects_partial_or_tampered_window_without_overwrite(raw, tmp_path, policy):
    baseline, candidate_dir, settings = make_source_inputs(raw, tmp_path, policy)
    output = tmp_path / "stage-windows"
    result = run(baseline, candidate_dir, settings, output, policy_dir=policy, window_id="window_0000")
    destination = Path(result["output"])
    manifest_path = destination / "evidence.json"
    original = manifest_path.read_bytes()
    manifest = json.loads(original);manifest["images"][0]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))
    damaged = manifest_path.read_bytes()
    with pytest.raises(ValueError, match="Evidence image changed"):
        run(baseline, candidate_dir, settings, output, policy_dir=policy, window_id="window_0000", resume=True)
    assert manifest_path.read_bytes() == damaged
    manifest_path.write_bytes(original)
    (destination / "window_receipt.json").unlink()
    with pytest.raises(FileNotFoundError):
        run(baseline, candidate_dir, settings, output, policy_dir=policy, window_id="window_0000", resume=True)


@pytest.mark.parametrize("key,value", [
    ("source_core_frame_range", [0, 4]),
    ("window_local_core_range", [1, 4]),
    ("stage_hint_sent_to_api_context", True),
    ("human_annotations_read", True),
    ("accepted_for_training", True),
])
def test_resume_rejects_tampered_core_mapping_or_provenance_flags(raw, tmp_path, policy, key, value):
    baseline, candidate_dir, settings = make_source_inputs(raw, tmp_path, policy)
    output = tmp_path / "stage-windows"
    result = run(baseline, candidate_dir, settings, output, policy_dir=policy, window_id="window_0000")
    receipt_path = Path(result["output"]) / "window_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    assert receipt[key] != value
    receipt[key] = value
    receipt_path.write_text(json.dumps(receipt))
    damaged = receipt_path.read_bytes()
    with pytest.raises(ValueError, match="completion receipt differs"):
        run(baseline, candidate_dir, settings, output, policy_dir=policy, window_id="window_0000", resume=True)
    assert receipt_path.read_bytes() == damaged


def test_output_isolation_and_unknown_window_id(raw, tmp_path, policy):
    from scripts.prepare_annotation_windows import ROOT
    baseline, candidate_dir, settings = make_source_inputs(raw, tmp_path, policy)
    for output in (raw[0] / "forbidden", ROOT / ".local/forbidden-window-test", baseline / "nested", candidate_dir / "nested"):
        with pytest.raises(ValueError, match="outside"):
            run(baseline, candidate_dir, settings, output, policy_dir=policy)
        assert not output.exists()
    output = tmp_path / "bad-id"
    with pytest.raises(ValueError, match="window_id"):
        run(baseline, candidate_dir, settings, output, policy_dir=policy, window_id="../escape")
    assert not output.exists()
