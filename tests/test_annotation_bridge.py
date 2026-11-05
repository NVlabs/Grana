# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
import hashlib
import json
from pathlib import Path

from datasets import load_from_disk
import numpy as np
import pytest

from grana.annotation.bridge import read_bridge_annotations
from grana.annotation.source import LeRobotSource
from grana.data.cache import CachedTrajectories
from grana.data.builder import build_cache, validate_build_config
from test_annotation_source import raw  # Shared real unannotated, lossless-video fixture.


@pytest.fixture
def bridge(raw, tmp_path):
    source, views, _, _, pixels = raw
    episode = LeRobotSource(source, views).load_episode(0)
    stages = [{"id": "S1", "label": "1", "prompt": "Open drawer"},
              {"id": "S2", "label": "2", "prompt": "Close drawer"}]
    candidate = {"segments": [
        {"start_frame": 0, "end_frame": 3, "stage_id": "S1", "identity_status": "resolved", "reason": "Opening visible", "evidence_frames": [0, 2]},
        {"start_frame": 3, "end_frame": 6, "stage_id": "S2", "identity_status": "resolved", "reason": "Closing visible", "evidence_frames": [3, 5]},
    ], "boundaries": [{"frame": 3, "lower_frame": 2, "upper_frame": 4, "reason": "Nominal reviewed split", "evidence_frames": [2, 3, 4]}],
        "omitted_stages": [], "invalid_intervals": [], "summary": "Synthetic test stage candidate; reward labels provided separately."}
    revision = {"candidate": candidate, "candidate_sha256": "", "stages": stages, "source_identity": episode.source_identity,
                "time_edges": list(episode.time_edges), "created_at": "2026-11-01T00:00:00Z",
                "review": {"reviewer": "synthetic test fixture", "kind": "agent", "decision": "accepted",
                           "notes": "Test-only acceptance, not a real-world annotation claim.",
                           "evidence_inspected": ["Synthetic generated two-camera fixture"]}}
    supplement = {"source_identity": episode.source_identity, "reviewer": "synthetic oracle",
                  "provenance": "Explicit authored test AQ and outcomes; neither derived from stage IDs nor API output.",
                  "boundary_policy": "nominal_dense",
                  "aq_intervals": [
                      {"start_frame": 0, "end_frame": 1, "quality": "nominal", "reason": "Authored fixture"},
                      {"start_frame": 1, "end_frame": 2, "quality": "suboptimal", "reason": "Authored fixture"},
                      {"start_frame": 2, "end_frame": 6, "quality": "nominal", "reason": "Authored fixture"}],
                  "outcomes": [
                      {"start_frame": 0, "end_frame": 3, "stage_id": "S1", "outcome": "high", "reason": "Authored fixture success"},
                      {"start_frame": 3, "end_frame": 6, "stage_id": "S2", "outcome": "mid", "reason": "Authored fixture partial outcome"}]}
    spec = {"revision": str(tmp_path / "revision.json"), "supplement": str(tmp_path / "supplement.json")}
    def save(*, update_candidate_hash=True):
        if update_candidate_hash:
            revision["candidate_sha256"] = hashlib.sha256(json.dumps(revision["candidate"], sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()
        Path(spec["revision"]).write_text(json.dumps(revision))
        Path(spec["supplement"]).write_text(json.dumps(supplement))
    save()
    config = {"source": str(source), "output": str(tmp_path / "cache"), "episodes": [0], "task": "drawer",
              "compressed_task_prompt": "Open and close the drawer", "views": views, "fps": 10,
              "stages": [{k: stage[k] for k in ("label", "prompt")} for stage in stages],
              "progress": {"method": "grana_target_composition", "parameter_status": "example",
                           "parameter_source": "Synthetic formula test", "stage_parameters": [
                               {"rho": 2, "p_cap": .8, "t_min_seconds": 1}, {"rho": 2, "p_cap": .8, "t_min_seconds": 1}]},
              "annotations": {"0": spec}}
    return config, revision, supplement, save, pixels


def test_unannotated_source_to_real_cache_requires_explicit_reward_supplement(bridge):
    cfg, revision, supplement, _, pixels = bridge
    source = Path(cfg["source"])
    before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source.rglob("*") if p.is_file()}
    report = build_cache(cfg)
    row = load_from_disk(str(Path(cfg["output"]) / "processed_dataset"))[0]
    assert row["stage_ids"] == [0, 0, 0, 1, 1, 1]
    assert row["within_stage_progress"] == pytest.approx([0, 1 / 3, 1, 0, .25, .5])
    assert row["progress"] == pytest.approx([0, 1 / 6, .5, .5, .625, .75])
    assert row["action_quality"] == [0, 1, 0, 0, 0, 0]
    assert row["source_frame_indices"] == list(range(6))
    for view in cfg["views"]:
        np.testing.assert_array_equal(np.load(row[f"frames_{view}"]), pixels[view])
    provenance = report["episodes"][0]
    assert provenance["quality_source"] == "explicit_supplement.aq_intervals"
    assert provenance["boundary_policy"] == "nominal_dense"
    assert provenance["stage_review"] == revision["review"]
    assert provenance["supplement_provenance"] == supplement["provenance"]
    assert CachedTrajectories([cfg["output"]], views=list(cfg["views"])).summary()["eligible_rows"] == {"A": 1, "B": 1, "C": 1}
    assert before == {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source.rglob("*") if p.is_file()}


@pytest.mark.parametrize("edit,match", [
    (lambda r, s: r["review"].update(decision="needs_changes"), "accepted"),
    (lambda r, s: r["review"].update(kind="api"), "accepted"),
    (lambda r, s: r["review"].update(evidence_inspected=[]), "evidence"),
    (lambda r, s: r["stages"].reverse(), "registry"),
    (lambda r, s: r["time_edges"].__setitem__(1, 10.15), "time_edges"),
    (lambda r, s: r.update(source_identity={"fake": True}), "source_identity"),
    (lambda r, s: s.update(source_identity={"fake": True}), "source_identity"),
    (lambda r, s: r["candidate"]["segments"][0].update(identity_status="uncertain"), "uncertain"),
    (lambda r, s: r["candidate"]["invalid_intervals"].append({"start_frame": 1, "end_frame": 2, "reason": "Source damaged"}), "source-invalid"),
    (lambda r, s: s.update(boundary_policy="masked"), "nominal_dense"),
    (lambda r, s: s.update(provenance=""), "provenance"),
    (lambda r, s: s["aq_intervals"].clear(), "every native frame"),
    (lambda r, s: s["aq_intervals"][1].update(start_frame=0), "partition"),
    (lambda r, s: s["aq_intervals"][1].update(quality="unknown"), "quality"),
    (lambda r, s: s["outcomes"].clear(), "every maximal"),
    (lambda r, s: s["outcomes"][0].update(outcome="unknown"), "high/mid/low"),
    (lambda r, s: s["outcomes"][0].update(end_frame=2), "boundaries"),
])
def test_reward_export_fails_closed_without_fabricating_supervision(bridge, edit, match):
    cfg, revision, supplement, save, _ = bridge
    edit(revision, supplement)
    save()
    with pytest.raises(ValueError, match=match):
        read_bridge_annotations(cfg["source"], cfg["views"], 0, cfg["stages"], cfg["annotations"]["0"])
    assert not Path(cfg["output"]).exists()


def test_candidate_tamper_and_stage_only_specification_are_rejected(bridge):
    cfg, revision, _, save, _ = bridge
    revision["candidate"]["summary"] += " tampered"
    save(update_candidate_hash=False)
    with pytest.raises(ValueError, match="candidate_sha256"):
        build_cache(cfg)
    assert not Path(cfg["output"]).exists()
    broken = deepcopy(cfg)
    del broken["annotations"]["0"]["supplement"]
    with pytest.raises(ValueError, match="revision and supplement"):
        validate_build_config(broken)


def test_bridge_config_paths_resolve_and_episode_coverage_is_exact(bridge, tmp_path):
    cfg, _, _, _, _ = bridge
    cfg["annotations"] = {0: {"revision": "revision.json", "supplement": "supplement.json"}}
    normalized = validate_build_config(cfg, tmp_path)
    assert normalized["annotations"]["0"]["revision"] == str(tmp_path / "revision.json")
    for mapping in ({"1": cfg["annotations"][0]}, {"00": cfg["annotations"][0]}, {}):
        with pytest.raises(ValueError, match="annotations"):
            validate_build_config(cfg | {"annotations": mapping}, tmp_path)
