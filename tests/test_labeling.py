# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""grana label: label files, the local page's API, and building a cache from finished labels."""
from http.client import HTTPConnection
import json
from pathlib import Path
import shutil
import subprocess
import threading

from datasets import load_from_disk
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from grana.annotation.label_server import create_label_server
from grana.annotation.labels import (LEGACY_KEYS, dense_labels, label_path, new_labels, timestamps_sha256,
                                     validate_labels, write_labels)
from grana.annotation.source import LeRobotSource
from grana.data.builder import build_cache, validate_build_config

STAGES = ["1", "2"]


@pytest.fixture
def recipe(tmp_path):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("Labeling needs ffmpeg and ffprobe")
    source = tmp_path / "raw"
    (source / "meta").mkdir(parents=True)
    keys = {"top": "observation.images.top", "wrist": "observation.images.wrist"}
    info = dict(codebase_version="v2.1", fps=10, chunks_size=1000,
                data_path="data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
                video_path="videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
                features={key: {"dtype": "video", "shape": [24, 32, 3]} for key in keys.values()})
    (source / "meta/info.json").write_text(json.dumps(info))
    lengths = {0: 30, 1: 20}
    (source / "meta/episodes.jsonl").write_text("".join(json.dumps(dict(episode_index=i, length=n)) + "\n" for i, n in lengths.items()))
    for index, length in lengths.items():
        parquet = source / f"data/chunk-000/episode_{index:06d}.parquet"
        parquet.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table({"frame_index": list(range(length)), "episode_index": [index] * length,
                                 "timestamp": [i / 10 for i in range(length)]}), parquet)
        for key in keys.values():
            video = source / f"videos/chunk-000/{key}/episode_{index:06d}.mp4"
            video.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=32x24:rate=10",
                            "-frames:v", str(length), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(video)], check=True)
    return dict(source=str(source), output=str(tmp_path / "cache"), labels=str(tmp_path / "labels"), episodes="all",
                task="token-and-lid", compressed_task_prompt="Place the token, then close the lid.",
                stages=[dict(label="1", prompt="Place the token in the tray"), dict(label="2", prompt="Close the tray lid")],
                views=keys, fps=5,
                progress=dict(method="grana_target_composition", parameter_status="example", parameter_source="Test values",
                              stage_parameters=[dict(rho=50, p_cap=0.95, t_min_seconds=1)] * 2))


@pytest.fixture
def server(recipe):
    instance = create_label_server(validate_build_config(recipe), port=0)
    thread = threading.Thread(target=instance.serve_forever, daemon=True)
    thread.start()
    yield instance
    instance.shutdown()
    instance.server_close()


def save_current(project, index, body):
    return project.save(index, {**body, "project_revision": project.project()["revision"]})


def request(server, method, path, body=None, headers=None):
    if method == "POST" and path.startswith(("/api/labels?", "/api/preview?")) and isinstance(body, dict) and "project_revision" not in body:
        body = {**body, "project_revision": server.project.project()["revision"]}
    if method == "POST" and path.startswith("/api/labels?") and isinstance(body, dict) and "base_revision" not in body:
        current = json.loads(request(server, "GET", path.replace("/api/labels?", "/api/episode?"))[2])
        body = {**body, "base_revision": current.get("labels", {}).get("revision")}
    connection = HTTPConnection("127.0.0.1", server.server_address[1], timeout=30)
    data = None if body is None else json.dumps(body).encode()
    connection.request(method, path, body=data, headers={**({"Content-Type": "application/json"} if data else {}), **(headers or {})})
    response = connection.getresponse()
    payload = response.read()
    connection.close()
    return response.status, response, payload


FINISHED = {"subtasks": [{"start_frame": 0, "end_frame": 18, "stage": "1", "outcome": "high"},
                         {"start_frame": 18, "end_frame": 30, "stage": "2", "outcome": "low"}],
            "quality": [{"start_frame": 8, "end_frame": 11, "label": "regressed"}], "annotator": "Ada", "done": True, "review_confirmed": True}


def test_label_documents_validate_and_densify():
    doc = new_labels(0, "trajectory", [i / 10 for i in range(6)], STAGES,
                     source_identity={"trajectory_id": "trajectory"}, stage_registry=[])
    doc.update(subtasks=[{"start_frame": 0, "end_frame": 4, "stage": "1", "outcome": "high"},
                         {"start_frame": 4, "end_frame": 6, "stage": "2", "outcome": "low"}],
               quality=[{"start_frame": 1, "end_frame": 2, "label": "suboptimal"}], annotator="Ada", done=True, review_confirmed=True)
    validate_labels(doc, frame_count=6, stages=STAGES)
    assert dense_labels(doc) == (["1"] * 4 + ["2"] * 2, ["", "", "", "high", "", "low"], [0, 1, 0, 0, 0, 0])
    gap = {**doc, "subtasks": doc["subtasks"][1:]}
    with pytest.raises(ValueError, match="no subtask"):
        validate_labels(gap, frame_count=6, stages=STAGES)
    validate_labels({**gap, "done": False}, frame_count=6, stages=STAGES)  # drafts may be incomplete
    for broken in ({**doc, "subtasks": [{**doc["subtasks"][0], "outcome": "mid"}, doc["subtasks"][1]]},
                   {**doc, "quality": [{"start_frame": 1, "end_frame": 9, "label": "suboptimal"}]},
                   {**doc, "quality": [{"start_frame": 1, "end_frame": 2, "label": "nominal"}]},
                   {**doc, "subtasks": [{**doc["subtasks"][0], "stage": "3"}, doc["subtasks"][1]]}):
        with pytest.raises(ValueError):
            validate_labels(broken, frame_count=6, stages=STAGES)


def test_page_and_every_asset_it_references_are_served(server):
    import re

    status, _, page = request(server, "GET", "/")
    assert status == 200 and b"<title>Grana" in page
    assets = re.findall(r'(?:src|href)="([a-z][\w.-]*\.(?:js|css|svg))"', page.decode())
    assert {"grana-api.js", "agreement.js", "lab-interact.js", "grana-target.js", "grana-theme.css"} <= set(assets)
    for name in assets:
        assert request(server, "GET", f"/{name}")[0] == 200, name
    assert request(server, "GET", "/../labels.py")[0] == 404


def test_page_api_saves_labels_and_previews_the_cache_target(server, recipe):
    status, _, page = request(server, "GET", "/")
    assert status == 200 and b"Grana" in page
    project = json.loads(request(server, "GET", "/api/project")[2])
    assert [e["status"] for e in project["episodes"]] == ["new", "new"]
    episode = json.loads(request(server, "GET", "/api/episode?episode=0")[2])
    assert episode["frame_count"] == 30 and episode["labels"]["subtasks"] == []

    status, response, chunk = request(server, "GET", episode["views"]["top"], headers={"Range": "bytes=0-99"})
    assert status == 206 and len(chunk) == 100 and response.getheader("Content-Range").startswith("bytes 0-99/")

    incomplete = json.loads(request(server, "POST", "/api/preview?episode=0", {"subtasks": FINISHED["subtasks"][:1], "quality": []})[2])
    assert incomplete["problems"] and "frames" not in incomplete
    target = json.loads(request(server, "POST", "/api/preview?episode=0", {k: FINISHED[k] for k in ("subtasks", "quality")})[2])
    assert target["problems"] == [] and target["progress"][-1] == pytest.approx(0.5)  # (1 + failed) / 2 stages

    status, _, body = request(server, "POST", "/api/labels?episode=0", {**FINISHED, "annotator": ""})
    assert status == 400 and b"annotator" in body
    assert json.loads(request(server, "POST", "/api/labels?episode=0", FINISHED)[2])["status"] == "done"
    assert json.loads(request(server, "POST", "/api/labels?episode=1", {"subtasks": [], "quality": [], "done": False})[2])["status"] == "draft"
    assert [e["status"] for e in json.loads(request(server, "GET", "/api/project")[2])["episodes"]] == ["done", "draft"]
    assert request(server, "GET", "/api/episode?episode=7")[0] == 404
    assert request(server, "GET", "/video?episode=0&view=side")[0] == 404


def test_finished_labels_build_a_cache_directly(server, recipe):
    request(server, "POST", "/api/labels?episode=0", FINISHED)
    request(server, "POST", "/api/labels?episode=1", {"subtasks": FINISHED["subtasks"][:1], "quality": [], "done": False})
    report = build_cache(validate_build_config(recipe))
    assert [e["episode_index"] for e in report["episodes"]] == [0]  # the draft is left out
    assert report["episodes"][0]["annotation_source"] == "grana.labels.v1" and report["episodes"][0]["annotator"] == "Ada"
    row = load_from_disk(str(Path(recipe["output"]) / "processed_dataset"))[0]
    assert row["progress_method"] == "grana_target_composition"
    assert set(row["stage_ids"]) == {0, 1} and 2 in row["action_quality"]
    assert row["progress"][-1] == pytest.approx(0.5)


def test_submitted_legacy_label_files_build_a_cache_beside_current_ones(server, recipe):
    """An 11-key file from the first grana label is read as it is, bound to its episode's index, trajectory and clock."""
    request(server, "POST", "/api/labels?episode=0", FINISHED)
    episode = LeRobotSource(Path(recipe["source"]), recipe["views"]).load_episode(1)
    doc = new_labels(1, episode.source_identity["trajectory_id"], episode.timestamps, STAGES,
                     source_identity=episode.source_identity, stage_registry=recipe["stages"])
    doc.update(subtasks=[{"start_frame": 0, "end_frame": 12, "stage": "1", "outcome": "high"},
                         {"start_frame": 12, "end_frame": 20, "stage": "2", "outcome": "high"}],
               quality=[{"start_frame": 4, "end_frame": 6, "label": "suboptimal"}], annotator="Bo", done=True,
               updated_at="2026-11-01T10:00:00+00:00")
    legacy = {key: value for key, value in doc.items() if key in LEGACY_KEYS}
    label_path(recipe["labels"], 1).write_text(json.dumps(legacy))

    report = build_cache(validate_build_config(recipe))
    current, old = report["episodes"]
    assert (current["episode_index"], old["episode_index"]) == (0, 1)
    assert current["review_confirmed"] is True and "label_format" not in current
    assert old["label_format"] == "11-key (before revisions)" and old["annotator"] == "Bo"
    assert old["review_confirmed"] is None and old["label_revision"] is None
    rows = load_from_disk(str(Path(recipe["output"]) / "processed_dataset"))
    assert rows[1]["source_episode_index"] == 1 and set(rows[1]["stage_ids"]) == {0, 1}
    assert 1 in rows[1]["action_quality"] and rows[1]["progress"][-1] == pytest.approx(1.0)

    moved = {**legacy, "timestamps_sha256": timestamps_sha256([t + 1 for t in episode.timestamps])}
    label_path(recipe["labels"], 1).write_text(json.dumps(moved))
    with pytest.raises(ValueError, match="different source episode"):
        build_cache(validate_build_config({**recipe, "output": str(Path(recipe["output"]).with_name("cache-moved"))}))


def test_labels_must_match_the_source_episode(recipe):
    cfg = validate_build_config({**recipe, "episodes": [0]})
    doc = new_labels(0, "raw__episode_000000", [i / 10 + 1 for i in range(30)], STAGES,
                     source_identity={}, stage_registry=recipe["stages"])  # different clock
    write_labels(cfg["labels"], {**doc, **FINISHED})
    with pytest.raises(ValueError, match="different source episode"):
        build_cache(cfg)
    with pytest.raises(ValueError, match="either labels"):
        validate_build_config({**recipe, "episodes": [0], "annotations": {"0": {"revision": "r", "supplement": "s"}}})
    with pytest.raises(ValueError, match="outside the raw source"):
        validate_build_config({**recipe, "labels": str(Path(recipe["source"]) / "labels")})


def test_label_server_refuses_non_loopback_and_missing_labels_dir(recipe):
    with pytest.raises(ValueError, match="loopback"):
        create_label_server(validate_build_config(recipe), host="0.0.0.0", port=0)
    without = {key: value for key, value in recipe.items() if key != "labels"}
    without["episodes"] = [0]
    with pytest.raises(ValueError, match="labels"):
        create_label_server(validate_build_config(without), port=0)


def test_requests_must_name_this_loopback_server(server):
    port = server.server_address[1]
    # DNS rebinding: an attacker's page reaches 127.0.0.1 under its own host name.
    assert request(server, "GET", "/api/project", headers={"Host": f"attacker.example:{port}"})[0] == 403
    assert request(server, "POST", "/api/labels?episode=0", FINISHED, headers={"Host": f"attacker.example:{port}"})[0] == 403
    assert request(server, "POST", "/api/labels?episode=0", FINISHED, headers={"Origin": "http://attacker.example"})[0] == 403
    assert request(server, "GET", "/api/project", headers={"Host": f"localhost:{port}"})[0] == 200
    assert request(server, "POST", "/api/labels?episode=0", FINISHED, headers={"Origin": f"http://127.0.0.1:{port}"})[0] == 200
    # Port forwarding (ssh -L 9000:127.0.0.1:PORT) reaches the page under another local port.
    assert request(server, "GET", "/api/project", headers={"Host": "localhost:9000"})[0] == 200
    assert request(server, "POST", "/api/labels?episode=0", FINISHED,
                   headers={"Host": "localhost:9000", "Origin": "http://localhost:9000"})[0] == 200


def test_a_save_based_on_an_older_version_is_refused(server):
    first = json.loads(request(server, "POST", "/api/labels?episode=0", {**FINISHED, "done": False, "base_revision": None})[2])
    assert first["status"] == "draft" and first["updated_at"]
    # A second annotator who opened the episode before that save must reload instead of overwriting it.
    status, _, body = request(server, "POST", "/api/labels?episode=0", {**FINISHED, "annotator": "Bo", "base_revision": None})
    assert status == 409 and b"saved by Ada" in body
    # The page uses these to load and show the newer labels.
    assert json.loads(body) | {"error": None} == {"error": None, "annotator": "Ada", "updated_at": first["updated_at"], "revision": first["revision"]}
    ok = json.loads(request(server, "POST", "/api/labels?episode=0", {**FINISHED, "base_revision": first["revision"]})[2])
    assert ok["status"] == "done"


def test_write_failures_are_reported(server, recipe):
    import os
    if os.geteuid() == 0:
        pytest.skip("root can write to read-only directories")
    labels = Path(recipe["labels"])
    labels.mkdir(parents=True, exist_ok=True)
    labels.chmod(0o500)
    try:
        status, _, body = request(server, "POST", "/api/labels?episode=0", FINISHED)
    finally:
        labels.chmod(0o700)
    assert status == 500 and b"Could not read or write labels" in body


def test_complete_drafts_are_never_training_annotations(server, recipe):
    status, _, body = request(server, "POST", "/api/labels?episode=0", {**FINISHED, "done": False})
    assert status == 200 and json.loads(body)["status"] == "draft"
    with pytest.raises(ValueError, match="draft cannot be used for training"):
        build_cache({**recipe, "episodes": [0]})
    with pytest.raises(ValueError, match="No submitted episode"):
        build_cache(recipe)


def test_submit_requires_persisted_human_review(server):
    status, _, body = request(server, "POST", "/api/labels?episode=0", {**FINISHED, "review_confirmed": False})
    assert status == 400 and b"entire video was reviewed" in body
    saved = json.loads(request(server, "POST", "/api/labels?episode=0", FINISHED)[2])
    assert saved["labels"]["review_confirmed"] is True
    assert saved["labels"]["revision"] == saved["revision"]


def test_revision_tokens_change_even_at_identical_clock_times(server, monkeypatch):
    from datetime import datetime, timezone
    import grana.annotation.labels as labels

    class FixedClock:
        @staticmethod
        def now(tz):
            return datetime(2026, 11, 3, 12, 0, 0, tzinfo=timezone.utc)

    monkeypatch.setattr(labels, "datetime", FixedClock)
    first = save_current(server.project, 0, {**FINISHED, "base_revision": None})
    second = save_current(server.project, 0, {**FINISHED, "base_revision": first["revision"]})
    assert first["updated_at"] == second["updated_at"]
    assert first["revision"] != second["revision"]
    from grana.annotation.label_server import SaveConflict
    with pytest.raises(SaveConflict):
        save_current(server.project, 0, {**FINISHED, "annotator": "Stale", "base_revision": first["revision"]})
    assert server.project.episode(0)["labels"]["revision"] == second["revision"]
    with pytest.raises(ValueError, match="base_revision"):
        save_current(server.project, 0, FINISHED)


def test_separate_projects_cannot_overwrite_one_revision(server, recipe):
    from concurrent.futures import ThreadPoolExecutor
    from grana.annotation.label_server import LabelProject, SaveConflict

    first = save_current(server.project, 0, {**FINISHED, "base_revision": None})
    other = LabelProject(recipe)
    barrier = threading.Barrier(2)

    def save(project, annotator):
        barrier.wait()
        try:
            save_current(project, 0, {**FINISHED, "annotator": annotator, "base_revision": first["revision"]})
            return "saved"
        except SaveConflict:
            return "conflict"

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(save, project, name) for project, name in ((server.project, "Ada"), (other, "Bo"))]
        assert sorted(f.result(timeout=30) for f in futures) == ["conflict", "saved"]


def test_source_bytes_and_stage_prompts_are_bound(server, recipe):
    from copy import deepcopy
    from grana.annotation.label_server import LabelProject

    save_current(server.project, 0, {**FINISHED, "base_revision": None})
    changed = deepcopy(recipe)
    changed["stages"][0]["prompt"] = "Pick a different object"
    with pytest.raises(ValueError, match="stage registry"):
        LabelProject(changed).episode(0)
    with pytest.raises(ValueError, match="stage registry"):
        build_cache({**changed, "episodes": [0]})

    path = Path(recipe["source"]) / "data/chunk-000/episode_000000.parquet"
    table = pq.read_table(path)
    pq.write_table(table.append_column("extra", pa.array([1] * table.num_rows)), path)
    with pytest.raises(ValueError, match="different source episode"):
        build_cache({**recipe, "episodes": [0]})
    status, _, body = request(server, "POST", "/api/labels?episode=0", FINISHED)
    assert status == 400 and b"different source episode" in body


def test_mismatched_embedded_episode_never_writes_another_file(server, recipe):
    from grana.annotation.labels import label_path

    saved = save_current(server.project, 0, {**FINISHED, "base_revision": None})["labels"]
    saved["episode_index"] = 1
    label_path(recipe["labels"], 0).write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="different source episode"):
        save_current(server.project, 0, {**FINISHED, "base_revision": saved["revision"]})
    assert not label_path(recipe["labels"], 1).exists()


def test_atomic_save_failure_preserves_existing_labels(server, recipe, monkeypatch):
    import grana.annotation.labels as labels

    save_current(server.project, 0, {**FINISHED, "base_revision": None})
    path = labels.label_path(recipe["labels"], 0)
    before = path.read_bytes()

    def fail(*args):
        raise OSError("simulated failed rename")

    monkeypatch.setattr(labels.os, "replace", fail)
    with pytest.raises(OSError, match="failed rename"):
        labels.write_labels(recipe["labels"], json.loads(before))
    assert path.read_bytes() == before
    assert not list(path.parent.glob(".labels-*.json"))


def test_invalid_video_clocks_and_incomplete_v3_metadata_are_rejected(recipe):
    from grana.annotation.label_server import LabelProject

    source = Path(recipe["source"])
    video = source / "videos/chunk-000/observation.images.top/episode_000000.mp4"
    wrong = source / "videos/chunk-000/observation.images.top/episode_000001.mp4"
    shutil.copyfile(wrong, video)
    with pytest.raises(ValueError, match="frame count/PTS"):
        LabelProject(recipe).episode(0)
    info_path = source / "meta/info.json"
    info = json.loads(info_path.read_text())
    info["codebase_version"] = "v3.0"
    info_path.write_text(json.dumps(info))
    with pytest.raises(ValueError, match="v3 metadata is missing"):
        LabelProject(recipe)


def test_recipe_rejects_labels_in_checkout_or_output(recipe, tmp_path):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (checkout / ".git").mkdir()
    with pytest.raises(ValueError, match="outside Git"):
        validate_build_config({**recipe, "labels": str(checkout / "labels")})
    with pytest.raises(ValueError, match="cache output"):
        validate_build_config({**recipe, "labels": str(Path(recipe["output"]) / "labels")})


def test_editing_labels_does_not_inherit_old_review_confirmation(server):
    saved = save_current(server.project, 0, {**FINISHED, "base_revision": None})
    without_confirmation = {k: v for k, v in FINISHED.items() if k != "review_confirmed"}
    with pytest.raises(ValueError, match="entire video was reviewed"):
        save_current(server.project, 0, {**without_confirmation, "base_revision": saved["revision"]})


def test_submit_rejects_labels_whose_target_cannot_be_constructed(server):
    bad = {**FINISHED, "quality": [], "subtasks": [
        {"start_frame": 0, "end_frame": 1, "stage": "1", "outcome": "high"},
        {"start_frame": 1, "end_frame": 30, "stage": "2", "outcome": "high"},
    ]}
    with pytest.raises(ValueError, match="zero observed update duration"):
        save_current(server.project, 0, {**bad, "base_revision": None})
    assert server.project.status(0) == "new"


def test_browser_clock_allows_source_offset_but_rejects_video_offset():
    from types import SimpleNamespace
    from grana.data.builder import validate_label_clock

    episode = SimpleNamespace(timestamps=(3.0, 3.1, 3.2), frame_count=3,
                              clock_receipt={"cameras": {"top": {"source_minus_video_offset_seconds": 3.0}}})
    validate_label_clock(episode, 10)
    episode.clock_receipt["cameras"]["top"]["source_minus_video_offset_seconds"] = 2.0
    with pytest.raises(ValueError, match="presentation timestamp zero"):
        validate_label_clock(episode, 10)


@pytest.fixture
def editable_server(recipe, tmp_path):
    import yaml

    path = tmp_path / "labeling.yaml"
    path.write_text(yaml.safe_dump(recipe))
    instance = create_label_server(validate_build_config(recipe), port=0, config_path=path)
    thread = threading.Thread(target=instance.serve_forever)
    thread.start()
    yield instance, path
    instance.shutdown()
    instance.server_close()
    thread.join(timeout=5)


def test_edit_stage_definitions_preserves_labels_and_builds_from_new_recipe(editable_server, recipe):
    from grana.annotation.label_server import LabelProject
    from grana.annotation.stage_settings import StageConflict
    from grana.data.builder import load_build_config

    server, config_path = editable_server
    original_revision = server.project.project()["revision"]
    # Save the original definition before editing its shared vocabulary.
    save_current(server.project, 0, {**FINISHED, "base_revision": None})
    old_labels = Path(recipe["labels"]) / "episode_000000.json"
    original_bytes = old_labels.read_bytes()
    stages = [{"label": "2", "prompt": "Grasp the tray lid"}, {"label": "1", "prompt": "Place the object in the tray"},
              {"label": "3", "prompt": "Close the tray and release the gripper"}]
    status, _, payload = request(server, "POST", "/api/stages", {"base_revision": original_revision, "stages": stages})
    assert status == 200, payload
    result = json.loads(payload)
    assert result["changed"] and result["project"]["stages"] == stages
    assert result["project"]["revision"] != original_revision
    assert old_labels.read_bytes() == original_bytes
    previous = load_build_config(result["previous_recipe"])
    assert previous == validate_build_config(recipe)
    updated = load_build_config(config_path)
    assert updated["stages"] == stages and updated["labels"] != recipe["labels"]
    assert updated["output"] != recipe["output"]
    assert len(updated["progress"]["stage_parameters"]) == 3
    assert updated["progress"]["parameter_status"] == "example"
    restarted = LabelProject(updated, config_path)
    assert restarted.project() == server.project.project()
    assert restarted.settings.history == server.project.settings.history
    migrated = restarted.episode(0)["labels"]
    assert migrated["subtasks"] == FINISHED["subtasks"]
    assert migrated["quality"] == FINISHED["quality"]
    assert migrated["source_identity"] == json.loads(original_bytes)["source_identity"]
    assert migrated["stage_registry"] == stages
    assert migrated["revision"] != json.loads(original_bytes)["revision"]
    assert not migrated["done"] and not migrated["review_confirmed"]
    for project in (server.project, restarted):
        with pytest.raises(StageConflict, match="Sub-stage definitions changed"):
            project.save(0, {**FINISHED, "base_revision": None})
    # A browser with old meanings must not populate a new version, even with a null label revision.
    stale = {**FINISHED, "base_revision": None, "project_revision": original_revision}
    status, _, payload = request(server, "POST", "/api/labels?episode=0", stale)
    assert status == 409 and json.loads(payload)["project_conflict"]
    assert request(server, "POST", "/api/preview?episode=0", stale)[0] == 409
    assert restarted.episode(0)["labels"] == migrated
    submitted = {**FINISHED, "subtasks": [
        {"start_frame": i * 10, "end_frame": (i + 1) * 10, "stage": stage["label"], "outcome": "high"}
        for i, stage in enumerate(stages)]}
    assert request(server, "POST", "/api/labels?episode=0", submitted)[0] == 200
    report = build_cache(load_build_config(config_path))
    row = load_from_disk(str(Path(updated["output"]) / "processed_dataset"))[0]
    assert report["reader_validation"] == "passed"
    assert row["stage_prompts"] == [stage["prompt"] for stage in stages]
    assert set(row["stage_ids"]) == {0, 1, 2}
    assert old_labels.read_bytes() == original_bytes


def test_appending_stage_carries_every_saved_episode(editable_server, recipe):
    server, config_path = editable_server
    save_current(server.project, 0, {**FINISHED, "base_revision": None})
    draft = {**FINISHED, "done": False, "review_confirmed": False,
             "subtasks": [{"start_frame": 3, "end_frame": 14, "stage": "2", "outcome": "low"}]}
    save_current(server.project, 1, {**draft, "base_revision": None})
    before = {i: server.project.episode(i)["labels"] for i in (0, 1)}
    stages = [*recipe["stages"], {"label": "3", "prompt": "Check the closed lid"}]
    result = server.project.update_stages({"base_revision": server.project.project()["revision"], "stages": stages})
    for i, old in before.items():
        current = server.project.episode(i)["labels"]
        assert current["subtasks"] == old["subtasks"] and current["quality"] == old["quality"]
        assert current["annotator"] == old["annotator"]
        assert current["revision"] != old["revision"]
        assert current["stages"] == ["1", "2", "3"]
        assert not current["done"] and not current["review_confirmed"]
        assert json.loads((Path(recipe["labels"]) / f"episode_{i:06d}.json").read_text()) == old
    assert result["changed"]


def test_removing_used_stage_rejects_without_changing_any_labels(editable_server, recipe):
    server, config_path = editable_server
    draft = {**FINISHED, "done": False, "review_confirmed": False,
             "subtasks": [{"start_frame": 0, "end_frame": 10, "stage": "2", "outcome": "low"}]}
    save_current(server.project, 1, {**draft, "base_revision": None})
    before = config_path.read_bytes()
    doc = server.project.episode(1)["labels"]
    body = {"base_revision": server.project.project()["revision"], "stages": recipe["stages"][:1]}
    status, _, payload = request(server, "POST", "/api/stages", body)
    assert status == 400 and "episode 1" in json.loads(payload)["error"]
    assert config_path.read_bytes() == before
    assert server.project.episode(1)["labels"] == doc
    assert not list(server.project.settings.history.glob("version-*"))
    # Reassigning the interval explicitly permits removal; its quality and outcome survive.
    save_current(server.project, 1, {**draft, "subtasks": [{**draft["subtasks"][0], "stage": "1"}],
                                     "base_revision": doc["revision"]})
    assert request(server, "POST", "/api/stages", body)[0] == 200
    current = server.project.episode(1)["labels"]
    assert current["subtasks"] == [{**draft["subtasks"][0], "stage": "1"}]
    assert current["quality"] == draft["quality"]


def test_partial_label_migration_failure_preserves_active_recipe(editable_server, recipe, monkeypatch):
    import grana.annotation.stage_settings as settings

    server, config_path = editable_server
    save_current(server.project, 0, {**FINISHED, "base_revision": None})
    save_current(server.project, 1, {**FINISHED, "subtasks": [], "done": False, "base_revision": None})
    before = config_path.read_bytes()
    original = {p: p.read_bytes() for p in Path(recipe["labels"]).glob("episode_*.json")}
    write = settings.write_labels
    def fail_second(directory, doc):
        if doc["episode_index"] == 1:
            raise OSError("simulated second label migration failure")
        return write(directory, doc)
    monkeypatch.setattr(settings, "write_labels", fail_second)
    with pytest.raises(OSError, match="second label migration"):
        server.project.update_stages({"base_revision": server.project.project()["revision"],
                                     "stages": [*recipe["stages"], {"label": "3", "prompt": "Check the lid"}]})
    assert config_path.read_bytes() == before
    assert server.project.labels == Path(recipe["labels"])
    assert all(p.read_bytes() == content for p, content in original.items())
    assert not list(server.project.settings.history.glob("version-*"))


@pytest.mark.parametrize("stages", [[], None, [{"label": [], "prompt": "bad"}],
    [{"label": "1", "prompt": ""}], [{"label": "1", "prompt": "same"}, {"label": "2", "prompt": "same"}],
    [{"label": "1", "prompt": "ok", "extra": True}], [{"label": str(i), "prompt": f"stage-{i}"} for i in range(27)]])
def test_invalid_stage_settings_leave_recipe_unchanged(editable_server, stages):
    server, config_path = editable_server
    before, original = config_path.read_bytes(), server.project.project()
    status, _, _ = request(server, "POST", "/api/stages", {"base_revision": original["revision"], "stages": stages})
    assert status == 400
    assert config_path.read_bytes() == before and server.project.project() == original
    assert not list(server.project.settings.history.glob("version-*"))


def test_stage_update_conflicts_and_external_recipe_changes(editable_server):
    from grana.annotation.label_server import LabelProject
    from grana.data.builder import load_build_config
    from grana.annotation.stage_settings import StageConflict

    server, config_path = editable_server
    other = LabelProject(load_build_config(config_path), config_path)
    original = server.project.project()
    stages = [{"label": "1", "prompt": "Complete the whole task"}]
    body = {"base_revision": original["revision"], "stages": stages}
    # The same loopback protections cover recipe writes.
    assert request(server, "POST", "/api/stages", body, {"Origin": "https://example.com"})[0] == 403
    assert request(server, "POST", "/api/stages", body)[0] == 200
    before = config_path.read_bytes()
    assert request(server, "POST", "/api/stages", body)[0] == 409
    with pytest.raises(StageConflict, match="another process"):
        other.save(0, {**FINISHED, "base_revision": None, "project_revision": original["revision"]})
    assert config_path.read_bytes() == before
    same = {"base_revision": server.project.project()["revision"], "stages": stages}
    assert json.loads(request(server, "POST", "/api/stages", same)[2])["changed"] is False
    config_path.write_bytes(before + b"\n# external edit\n")
    assert request(server, "POST", "/api/stages", same)[0] == 409


def test_failed_stage_recipe_replace_preserves_active_version(editable_server, monkeypatch):
    import grana.annotation.stage_settings as settings

    server, config_path = editable_server
    original, before = server.project.project(), config_path.read_bytes()
    def fail(*args):
        raise OSError("simulated failed recipe rename")
    monkeypatch.setattr(settings.os, "replace", fail)
    with pytest.raises(OSError, match="failed recipe"):
        server.project.update_stages({"base_revision": original["revision"], "stages": [{"label": "1", "prompt": "Complete the task"}]})
    assert config_path.read_bytes() == before and server.project.project() == original
    assert not list(config_path.parent.glob(".labeling.yaml.*"))


def test_restarted_server_rejects_first_save_from_old_or_unversioned_page(recipe):
    from copy import deepcopy
    from grana.annotation.label_server import LabelProject
    from grana.annotation.stage_settings import StageConflict

    old = LabelProject(recipe)
    old_revision = old.project()["revision"]
    assert old.episode(0)["labels"]["revision"] is None
    changed = deepcopy(recipe)
    changed["stages"][0]["prompt"], changed["stages"][1]["prompt"] = changed["stages"][1]["prompt"], changed["stages"][0]["prompt"]
    restarted = LabelProject(changed)
    assert restarted.episode(0)["labels"]["revision"] is None
    for identity in ({}, {"project_revision": old_revision}):
        body = {**FINISHED, "base_revision": None, **identity}
        with pytest.raises(StageConflict):
            restarted.save(0, body)
        with pytest.raises(StageConflict):
            restarted.preview(0, body)
    assert not list(Path(recipe["labels"]).glob("episode_*.json"))
    assert save_current(restarted, 0, {**FINISHED, "base_revision": None})["status"] == "done"
