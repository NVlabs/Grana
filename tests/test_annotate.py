# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline orchestration tests: clock, evidence, resumability and honest export."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import httpx
from PIL import Image
import pytest
import yaml

from test_annotation_api import provider_configuration  # noqa: F401

from grana.annotation import workflow as annotate
from grana.annotation import api as annotation_api
from grana.annotation.schema import validate_candidate
from grana.annotation.source import Episode


IDS = ["open", "place", "close"]
STAGES = [{"id": name, "label": str(i + 1), "prompt": f"Observed {name} action"}
          for i, name in enumerate(IDS)]


def candidate(start, end, stage="open", evidence=None):
    frames = [start, end - 1] if end - start > 1 else [start]
    return {"segments": [{"start_frame": start, "end_frame": end, "stage_id": stage,
                          "identity_status": "resolved" if stage is not None else "unresolved",
                          "reason": "Visible evidence supports this identity.", "evidence_frames": evidence or frames}],
            "boundaries": [], "invalid_intervals": [],
            "omitted_stages": [{"stage_id": name, "reason": "Not observed in this window."}
                               for name in IDS if name != stage], "summary": "Conservative machine candidate."}


def split_candidate(start, cut, end, evidence):
    left = [f for f in evidence if f < cut]
    right = [f for f in evidence if f >= cut]
    result = candidate(start, end)
    result["segments"] = [candidate(start, cut, "open", left)["segments"][0],
                          candidate(cut, end, "place", right)["segments"][0]]
    result["boundaries"] = [{"frame": cut, "lower_frame": max(left), "upper_frame": min(right),
                             "reason": "Transition lies between observed frames.",
                             "evidence_frames": [max(left), min(right)]}]
    result["omitted_stages"] = [{"stage_id": "close", "reason": "Not observed."}]
    return result


@pytest.fixture
def workflow(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    raw.mkdir()
    recipe = {"source": str(raw), "output": str(tmp_path / "output"), "episodes": [0],
              "task": "Move an object", "stages": STAGES,
              "views": {"top": "camera.top", "left": "camera.left"},
              "evidence": {"window_seconds": 1.2, "stride_seconds": 0.4,
                           "boundary_radius_seconds": 0.1, "max_images_per_request": 12,
                           "tile_size": [32, 32]}, "api": {"max_requests": 24}}
    config_path = tmp_path / "annotate.yaml"
    config_path.write_text(yaml.safe_dump(recipe, sort_keys=False))
    config = annotate.load_annotation_config(config_path)
    state = {"times": (7.0, 7.2, 7.5, 8.2, 8.25, 8.8, 9.5, 9.7, 10.9),
             "source_hash": "source-original", "requests": [], "decodes": [], "make_candidate": None}

    class Source:
        def __init__(self, source, views):
            self.source, self.views = Path(source), dict(views)

        def load_episode(self, number):
            times = state["times"]
            identity = {"source_root": str(self.source), "episode_index": number,
                        "parquet_sha256": state["source_hash"],
                        "video_sha256": {view: f"sha-{view}" for view in self.views}}
            return Episode(number, len(times), times, (*times, times[-1] + times[-1] - times[-2]),
                           {view: self.source / f"{view}.mp4" for view in self.views}, tuple(self.views),
                           identity, {"terminal_timestamp_policy": "last_timestamp_plus_last_observed_positive_delta"})

    def evidence(ep, indices, directory, tile_size):
        state["decodes"].append(list(indices))
        directory.mkdir(parents=True)
        records = []
        for frame in indices:
            path = directory / f"frame-{frame}.png"
            Image.new("RGB", tile_size, (frame, 3, 4)).save(path)
            records.append({"frame": frame, "timestamp": ep.timestamps[frame], "view_names": list(ep.view_names),
                            "image_path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        return records

    def handler(request):
        payload = json.loads(request.content)
        state["requests"].append(payload)
        if "ok" in payload["text"]["format"]["schema"]["properties"]:
            data = {"ok": True}
        else:
            context = json.loads(payload["input"][0]["content"][0]["text"].split("\n", 1)[1])
            start, end = context["interval"]
            frames = [row["frame"] for row in context["images"]]
            if state["make_candidate"]:
                data = state["make_candidate"](context, frames)
            else:
                stage = {0: "open", 3: "place", 6: "open", 8: "close"}.get(start, "open")
                data = candidate(start, end, stage, frames)
        return httpx.Response(200, json={"id": f"response-{len(state['requests'])}", "status": "completed",
                                        "model": payload["model"], "reasoning": payload["reasoning"],
                                        "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                                        "output": [{"type": "message", "content": [{"type": "output_text", "text": json.dumps(data)}]}]})

    real_request = annotation_api.request_json

    def client(*args, **kwargs):
        return real_request(*args, **kwargs, transport=httpx.MockTransport(handler))

    monkeypatch.setattr(annotate, "LeRobotSource", Source)
    monkeypatch.setattr(annotate, "make_evidence", evidence)
    monkeypatch.setattr(annotate, "request_json", client)
    monkeypatch.setenv(annotation_api.KEY_ENV, "mock-annotation-secret-no-network")
    return config, state, config_path


def test_source_clock_windows_and_sparse_selection(workflow):
    config, _, _ = workflow
    source, manifest = annotate.plan_annotation(config)
    ep = source.load_episode(0)
    assert [w["interval"] for w in manifest["episodes"][0]["windows"]] == [[0, 3], [3, 6], [6, 8], [8, 9]]
    assert manifest["episodes"][0]["time_edges"] == list(ep.time_edges)
    assert ep.time_edges[0] == 7.0 and ep.time_edges[-1] == pytest.approx(12.1)
    assert manifest["planned_annotation_calls"] == 12
    assert annotate.sparse_indices(ep, 0, 3, 0.4) == [0, 2]


@pytest.mark.parametrize("window_seconds,calls,stages", [
    (1.2, 13, ["open", "place", "open", "close"]), (3.0, 7, ["open", "close"]),
])
def test_completed_resume_makes_zero_http_requests_and_zero_decodes(workflow, monkeypatch, window_seconds, calls, stages):
    config, state, _ = workflow
    config["evidence"]["window_seconds"] = window_seconds
    first = annotate.run_annotation(config)
    assert len(state["requests"]) == calls
    original_decodes = len(state["decodes"])
    root = Path(config["output"])
    original = {p.relative_to(root): p.read_bytes() for p in root.rglob("*.json")}
    state["requests"].clear()
    monkeypatch.delenv(annotation_api.KEY_ENV)
    assert annotate.run_annotation(config) == first
    assert state["requests"] == [] and len(state["decodes"]) == original_decodes
    assert {p.relative_to(root): p.read_bytes() for p in root.rglob("*.json")} == original
    report = json.loads((root / "episodes/000000/report.json").read_text())
    assert report["review_status"] == "pending_human_review"
    assert report["aq_and_outcome"] == "not_generated"
    assert report["coverage"]["source_seconds"] == pytest.approx(5.1)
    final = json.loads((root / "episodes/000000/final.json").read_text())
    assert [item["stage_id"] for item in final["segments"]] == stages
    validate_candidate(final, IDS, 9)


def test_requests_carry_exact_source_times_and_only_sent_frame_citations(workflow):
    config, state, _ = workflow
    annotate.run_annotation(config)
    for payload in state["requests"][1:]:
        content = payload["input"][0]["content"]
        context = json.loads(content[0]["text"].split("\n", 1)[1])
        for row in context["images"]:
            assert row["timestamp"] == state["times"][row["frame"]]
            assert row["view_names"] == ["top", "left"]
        assert sum(item["type"] == "input_image" for item in content) == len(context["images"])
        assert context["interval_time_seconds"][0] == state["times"][context["interval"][0]]
        assert "annotation" not in context and "ground_truth" not in context


@pytest.mark.parametrize("change", ["source", "config", "clock", "view_order"])
def test_resume_rejects_source_or_configuration_changes_before_http(workflow, change):
    config, state, _ = workflow
    annotate.run_annotation(config)
    state["requests"].clear()
    if change == "source":
        state["source_hash"] = "source-was-modified"
    elif change == "config":
        config["task_instruction"] += " Changed contract."
    elif change == "clock":
        state["times"] = tuple(x + 0.01 for x in state["times"])
    else:
        config["views"] = dict(reversed(list(config["views"].items())))
    with pytest.raises(ValueError):
        annotate.run_annotation(config)
    assert state["requests"] == []


def test_changed_cached_image_is_rejected_without_new_http(workflow):
    config, state, _ = workflow
    annotate.run_annotation(config)
    state["requests"].clear()
    image = next(Path(config["output"]).glob("episodes/000000/window_000000/draft/images/*.png"))
    image.write_bytes(b"modified bytes")
    with pytest.raises(ValueError, match="Cached evidence image changed"):
        annotate.run_annotation(config)
    assert state["requests"] == []


def test_validate_window_requires_seen_frames_and_preserves_global_indices():
    data = split_candidate(10, 14, 20, [10, 13, 16, 19])
    original = deepcopy(data)
    assert annotate.validate_window(data, IDS, 10, 20, [10, 13, 16, 19]) == data
    assert data == original
    data["segments"][0]["evidence_frames"] = [12]
    with pytest.raises(ValueError, match="absent"):
        annotate.validate_window(data, IDS, 10, 20, [10, 13, 16, 19])


def test_boundary_cannot_cite_unseen_frames():
    data = split_candidate(10, 14, 20, [10, 13, 16, 19])
    data["boundaries"][0]["evidence_frames"] = [14]
    with pytest.raises(ValueError, match="absent"):
        annotate.validate_window(data, IDS, 10, 20, [10, 13, 16, 19])


def test_combine_removes_same_identity_window_cuts_but_keeps_retries():
    chunks = [candidate(0, 3, "open"), candidate(3, 5, "open"),
              candidate(5, 8, "place"), candidate(8, 10, "open")]
    combined = annotate.combine_windows(chunks, IDS, 10)
    assert [(s["start_frame"], s["end_frame"], s["stage_id"]) for s in combined["segments"]] == [
        (0, 5, "open"), (5, 8, "place"), (8, 10, "open")]
    assert [(b["frame"], b["lower_frame"], b["upper_frame"]) for b in combined["boundaries"]] == [(5, 4, 5), (8, 7, 8)]
    assert combined["segments"][0]["evidence_frames"] == [0, 2, 3, 4]
    assert chunks[0]["segments"][0]["end_frame"] == 3


def test_combine_preserves_identity_uncertainty_and_invalid_source():
    first, second = candidate(0, 3), candidate(3, 5)
    second["segments"][0]["identity_status"] = "uncertain"
    second["invalid_intervals"] = [{"start_frame": 3, "end_frame": 4, "reason": "Occluded source."}]
    combined = annotate.combine_windows([first, second], IDS, 5)
    assert len(combined["segments"]) == 2 and len(combined["boundaries"]) == 1
    assert combined["invalid_intervals"] == second["invalid_intervals"]


def test_overview_cap_stops_plan_without_api(workflow):
    config, state, _ = workflow
    config["evidence"].update(window_seconds=60, stride_seconds=0.01, max_images_per_request=2)
    with pytest.raises(ValueError, match="Overview exceeds"):
        annotate.run_annotation(config)
    assert state["requests"] == [] and state["decodes"] == []
    assert not list(Path(config["output"]).rglob("*.json"))


def test_dense_cap_fails_without_silently_dropping_frames(workflow):
    config, state, _ = workflow
    state["times"] = tuple(float(i) for i in range(12))
    config["evidence"].update(window_seconds=60, stride_seconds=4, boundary_radius_seconds=1.1,
                              max_images_per_request=4)
    state["make_candidate"] = lambda context, frames: split_candidate(0, 6, 12, frames)
    with pytest.raises(ValueError, match="No images dropped"):
        annotate.run_annotation(config)
    assert len(state["requests"]) == 2  # Probe + draft, no refinement call.
    assert state["decodes"] == [[0, 4, 8, 11]]
    assert not (Path(config["output"]) / "episodes/000000/final.json").exists()


def test_budget_accounts_for_all_window_phases_plus_preflight(workflow):
    config, state, _ = workflow
    config["api"]["max_requests"] = 12
    with pytest.raises(ValueError, match="12 annotation calls"):
        annotate.run_annotation(config)
    assert state["requests"] == []


def test_raw_machine_export_keeps_clock_and_pending_review(workflow, tmp_path):
    config, _, _ = workflow
    annotate.run_annotation(config)
    output = tmp_path / "stages.jsonl"
    report = annotate.export_stages(config["output"], output)
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert report["segments"] == 4
    assert [row["stage_occurrence"] for row in rows] == [1, 1, 2, 1]
    assert rows[0]["start_seconds"] == 7.0 and rows[-1]["end_seconds"] == pytest.approx(12.1)
    assert all(row["review"] == {"kind": "machine", "decision": "pending_human_review"} for row in rows)
    assert all(row["label_type"] == "task_stage_only" for row in rows)
    assert "human_accepted" not in output.read_text()
    with pytest.raises(FileExistsError):
        annotate.export_stages(config["output"], output)


def revision(run_dir):
    manifest = json.loads((Path(run_dir) / "manifest.json").read_text())
    ep = manifest["episodes"][0]
    data = json.loads((Path(run_dir) / ep["candidate_file"]).read_text())
    return {"candidate": data, "candidate_sha256": annotate.digest(data), "source_identity": ep["source_identity"],
            "stages": manifest["stages"], "time_edges": ep["time_edges"], "created_at": "2026-11-01T00:00:00Z",
            "review": {"reviewer": "test-agent", "kind": "agent", "decision": "accepted", "notes": "Checked supplied evidence.",
                       "evidence_inspected": ["episodes/000000/window_000000/draft/index.json"]}}


def test_agent_revision_export_does_not_become_human_acceptance(workflow, tmp_path):
    config, _, _ = workflow
    annotate.run_annotation(config)
    wrapper = revision(config["output"])
    path = tmp_path / "revision.json"
    path.write_text(json.dumps(wrapper))
    output = tmp_path / "edited.jsonl"
    annotate.export_stages(config["output"], output, {0: path})
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert all(row["review"]["kind"] == "agent" for row in rows)
    assert "human_accepted" not in output.read_text()


def test_non_ascii_ui_revision_hash_exports_consistently(workflow, tmp_path):
    from grana.annotation.review import _json_bytes
    config, _, _ = workflow
    annotate.run_annotation(config)
    wrapper = revision(config["output"])
    wrapper["candidate"]["summary"] = "已检查视频，保留阶段边界的不确定性。"
    wrapper["candidate"]["segments"][0]["reason"] = "可见机械臂正在打开抽屉。"
    wrapper["candidate_sha256"] = hashlib.sha256(_json_bytes(wrapper["candidate"])).hexdigest()
    path = tmp_path / "unicode-revision.json"
    path.write_text(json.dumps(wrapper, ensure_ascii=False))
    output = tmp_path / "unicode-export.jsonl"
    annotate.export_stages(config["output"], output, {0: path})
    row = json.loads(output.read_text().splitlines()[0])
    assert row["candidate_sha256"] == wrapper["candidate_sha256"]
    assert row["reason"] == wrapper["candidate"]["segments"][0]["reason"]


@pytest.mark.parametrize("invalid", ["source", "registry", "clock", "hash", "review_kind", "review_fields", "evidence_empty"])
def test_revision_export_rejects_invalid_provenance(workflow, tmp_path, invalid):
    config, _, _ = workflow
    annotate.run_annotation(config)
    wrapper = revision(config["output"])
    if invalid == "source":
        wrapper["source_identity"]["parquet_sha256"] = "different"
    elif invalid == "registry":
        wrapper["stages"] = list(reversed(wrapper["stages"]))
    elif invalid == "clock":
        wrapper["time_edges"][0] -= 0.01
    elif invalid == "hash":
        wrapper["candidate_sha256"] = "incorrect"
    elif invalid == "review_kind":
        wrapper["review"]["kind"] = "machine"
        wrapper["review"]["decision"] = "human_accepted"
    elif invalid == "review_fields":
        wrapper["review"]["human_accepted"] = True
    else:
        wrapper["review"]["evidence_inspected"] = []
    path = tmp_path / "revision.json"
    path.write_text(json.dumps(wrapper))
    output = tmp_path / "invalid-export.jsonl"
    with pytest.raises(ValueError):
        annotate.export_stages(config["output"], output, {0: path})
    assert not output.exists()


def test_cli_plan_never_calls_api(workflow, monkeypatch, capsys):
    from grana import cli
    _, state, config_path = workflow
    monkeypatch.setattr(sys, "argv", ["grana", "annotate", "--config", str(config_path), "--plan"])
    cli.main()
    assert json.loads(capsys.readouterr().out)["planned_annotation_calls"] == 12
    assert state["requests"] == []


def test_cli_rejects_plan_and_preflight_together(workflow, monkeypatch):
    from grana import cli
    _, state, config_path = workflow
    monkeypatch.setattr(sys, "argv", ["grana", "annotate", "--config", str(config_path), "--plan", "--preflight"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2 and state["requests"] == []


def test_offline_plan_requires_no_provider_or_credential(workflow, monkeypatch):
    config, state, path = workflow
    for name in (annotation_api.BASE_URL_ENV, annotation_api.MODEL_ENV, annotation_api.KEY_ENV):
        monkeypatch.delenv(name, raising=False)
    recipe = yaml.safe_load(path.read_text())
    recipe["api"] = {"max_requests": 24}
    path.write_text(yaml.safe_dump(recipe))
    config = annotate.load_annotation_config(path)
    assert config["api"]["model"] is None and config["api"]["base_url"] is None
    _, plan = annotate.plan_annotation(config)
    assert plan["planned_annotation_calls"] == 12
    assert state["requests"] == []
    with pytest.raises(ValueError, match="Annotation model is required"):
        annotate.run_annotation(config)
    assert not Path(config["output"]).exists()


def test_yaml_provider_and_credential_variable_override_environment(workflow):
    _, _, path = workflow
    recipe = yaml.safe_load(path.read_text())
    recipe["api"].update(model="private/vision", base_url="https://private.example/v2", api_key_env="PRIVATE_API_KEY")
    path.write_text(yaml.safe_dump(recipe))
    selected = annotate.load_annotation_config(path)["api"]
    assert (selected["model"], selected["base_url"], selected["api_key_env"]) == (
        "private/vision", "https://private.example/v2", "PRIVATE_API_KEY")


def test_saved_explicit_config_keeps_its_identity_without_provider_environment(workflow, monkeypatch):
    config, _, path = workflow
    # Earlier saved configs have no credential-variable field: loading them must
    # not add a new default into their immutable identity.
    assert "api_key_env" not in config["api"]
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    for name in (annotation_api.MODEL_ENV, annotation_api.BASE_URL_ENV, annotation_api.KEY_ENV):
        monkeypatch.delenv(name, raising=False)
    loaded = annotate.load_annotation_config(path)
    assert loaded == config
    assert annotate.config_digest(loaded) == annotate.config_digest(config)
