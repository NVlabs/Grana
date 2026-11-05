# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""tools/annotation_playground: recipe-driven routes, one shared request budget, the model list, explicit jobs,
per-episode import, batch import through grana draft-labels, and the seeded route: 1 to 5 seeds sent with every
request as one few-shot example (grana.annotation.templates), compared on held-out episodes through
scripts/seed_examples.py, on LeRobot v2.1 and v3.0 episodes built as test_lerobot_v3.py builds them.

The API is answered through httpx.MockTransport; every other connection fails the test.
"""
from contextlib import contextmanager
from http.client import HTTPConnection
import json
import os
from pathlib import Path
import re
import shutil
import socket
import threading
import subprocess

import httpx
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from test_annotation_api import provider_configuration  # noqa: F401

from grana.annotation.labels import LEGACY_KEYS, label_path, new_labels, validate_labels
from grana.annotation.source import LeRobotSource
from test_annotation_v3_evidence import FPS, VIEWS, encode, info, pixels
from functools import partial
LeRobotSource = partial(LeRobotSource, allow_shared_videos=True)
from tools.annotation_playground import launcher
from tools.annotation_playground.runner import (MESSAGES, PROBLEMS, REASONS, Playground, Refused, planned_attempts,
                                                read_json)
from tools.annotation_playground.server import running

LENGTHS = {0: 20, 1: 20, 2: 30, 3: 20, 4: 20}
STAGES = [{"label": "1", "prompt": "Pick the part"}, {"label": "2", "prompt": "Place the part"}]
KEY_ENV = "GRANA_PLAYGROUND_TEST_KEY"
MISSING_KEY_ENV = "GRANA_PLAYGROUND_UNSET_TEST_KEY"
DEFAULT = "default"   # the explicitly configured fixture model's id


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    """A synthetic key for the mocked API; any connection beyond this machine fails the test."""
    monkeypatch.setenv(KEY_ENV, "synthetic-test-key-not-a-credential")
    monkeypatch.delenv(MISSING_KEY_ENV, raising=False)
    connect = socket.socket.connect

    def loopback_only(self, address):
        if isinstance(address, tuple) and address[0] not in ("127.0.0.1", "::1", "localhost"):
            raise OSError(f"network access in a test: {address}")
        return connect(self, address)

    monkeypatch.setattr(socket.socket, "connect", loopback_only)


def dataset(folder, version, lengths=LENGTHS):
    """Episodes of ``lengths`` (five by default) as LeRobot v2.1 (a file per episode) or v3.0 (shared files), frames from
    test_lerobot_v3."""
    rows = lambda ep: {"frame_index": list(range(lengths[ep])), "episode_index": [ep] * lengths[ep],
                       "timestamp": [i / FPS for i in range(lengths[ep])]}
    (folder / "meta").mkdir(parents=True)
    if version == 2:
        (folder / "meta/info.json").write_text(json.dumps(info("v2.1", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
                                                               "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4")))
        (folder / "meta/episodes.jsonl").write_text("".join(json.dumps(dict(episode_index=ep, length=n, trajectory_id=f"trajectory-{ep}")) + "\n"
                                                            for ep, n in lengths.items()))
        for ep in lengths:
            path = folder / f"data/chunk-000/episode_{ep:06d}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(pa.table(rows(ep)), path)
            for view, key in VIEWS.items():
                encode(folder / f"videos/chunk-000/{key}/episode_{ep:06d}.mp4", pixels(ep, view, lengths[ep]))
        return folder
    (folder / "meta/episodes/chunk-000").mkdir(parents=True)
    (folder / "meta/info.json").write_text(json.dumps(info("v3.0", "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
                                                           "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4")))
    (folder / "data/chunk-000").mkdir(parents=True)
    pq.write_table(pa.concat_tables([pa.table(rows(ep)) for ep in lengths]), folder / "data/chunk-000/file-000.parquet")
    for view, key in VIEWS.items():
        encode(folder / f"videos/{key}/chunk-000/file-000.mp4", np.concatenate([pixels(ep, view, lengths[ep]) for ep in lengths]))
    starts = np.cumsum([0, *lengths.values()]) / FPS
    episodes = [{"episode_index": ep, "length": n, "trajectory_id": f"trajectory-{ep}", "tasks": ["demo"],
                 "data/chunk_index": 0, "data/file_index": 0,
                 **{f"videos/{key}/{name}": value for key in VIEWS.values()
                    for name, value in (("chunk_index", 0), ("file_index", 0), ("from_timestamp", float(starts[ep])),
                                        ("to_timestamp", float(starts[ep]) + n / FPS))}} for ep, n in lengths.items()]
    pq.write_table(pa.Table.from_pylist(episodes), folder / "meta/episodes/chunk-000/file-000.parquet")
    pq.write_table(pa.table({"task_index": [0], "task": ["demo"]}), folder / "meta/tasks.parquet")
    return folder


@pytest.fixture(scope="module", params=[3, 2], ids=["v3", "v2"])
def source(request, tmp_path_factory):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("needs ffmpeg and ffprobe")
    return dataset(tmp_path_factory.mktemp(f"lerobot-v{request.param}") / "raw", request.param)


@pytest.fixture(scope="module")
def seven(tmp_path_factory):
    """Seven LeRobot v3.0 episodes: room for three seeds, two held-out episodes and targets."""
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("needs ffmpeg and ffprobe")
    return dataset(tmp_path_factory.mktemp("lerobot-seven") / "raw", 3, lengths={**LENGTHS, 5: 20, 6: 20})


def recipe(source, folder):
    path = folder / "recipe.yaml"
    path.write_text(yaml.safe_dump({
        "source": str(source), "output": str(folder / "cache"), "labels": str(folder / "labels"), "episodes": "all",
        "task": "demo", "compressed_task_prompt": "Pick, then place.", "stages": STAGES, "views": VIEWS, "fps": 5,
        "progress": {"method": "grana_target_composition", "parameter_status": "example", "parameter_source": "Test values",
                     "stage_parameters": [{"rho": 20, "p_cap": 0.95, "t_min_seconds": 1}] * 2}}, sort_keys=False))
    return path


def init(source, tmp_path, *options, name="workspace", models=None):
    """Create or resume a workspace: with a --models file holding ``models``, else the explicitly configured fixture model keyed by KEY_ENV."""
    keys = ["--api-key-env", KEY_ENV]
    if models is not None:
        keys = ["--models", str(tmp_path / f"{name}-models.json")]
        Path(keys[1]).write_text(json.dumps(models))
    args = launcher.parser().parse_args(["--config", str(recipe(source, tmp_path)), "--workspace", str(tmp_path / name),
                                         *keys, *map(str, options)])
    return launcher.initialize(args)


def entry(model_id, *, key=KEY_ENV, effort="xhigh"):
    """A model list entry for a mocked Responses endpoint of its own."""
    return {"id": model_id, "label": f"Model {model_id.upper()}", "base_url": f"https://{model_id}.models.example/v1",
            "model": f"vendor/{model_id}-vision", "api_key_env": key, "reasoning_effort": effort, "protocol": "responses"}


def candidate(context, quality=None, last="high"):
    """A valid combined candidate for the clip: the registry's stages in order over equal parts, completed but for the
    last one, whose outcome is ``last``."""
    count, frames = len(context["time_edges"]) - 1, context["evidence_frames"]
    ids = [stage["id"] for stage in context["registry"]]
    cuts = [round(count * part / len(ids)) for part in range(len(ids) + 1)]
    cite = lambda *wanted: sorted({min(frames, key=lambda frame: (abs(frame - want), frame)) for want in wanted})
    cut = lambda frame: {"frame": frame, "lower_frame": frame, "upper_frame": frame, "reason": "A visible cut.",
                         "evidence_frames": cite(frame)}
    # Quality spans as given, the last one stretched to the end of this clip.
    quality = [(a, min(b, count), q) for a, b, q in quality or [(0, count, "nominal")] if a < count]
    quality[-1] = (*quality[-1][:1], count, quality[-1][2])
    return {"stages": {"segments": [{"start_frame": a, "end_frame": b, "stage_id": stage, "identity_status": "resolved",
                                     "reason": "Visible.", "evidence_frames": cite(a, b - 1)} for a, b, stage in zip(cuts, cuts[1:], ids)],
                       "boundaries": [cut(frame) for frame in cuts[1:-1]], "omitted_stages": [], "invalid_intervals": [],
                       "summary": "Synthetic test proposal."},
            "aq_intervals": [{"start_frame": a, "end_frame": b, "quality": q, "reason": "Seen.", "evidence_frames": cite(a, b - 1)}
                             for a, b, q in quality],
            "aq_boundaries": [cut(a) for a, _, _ in quality[1:]],
            "outcomes": [{"start_frame": a, "end_frame": b, "stage_id": stage, "outcome": last if b == count else "high",
                          "reason": "Done.", "evidence_frames": cite(b - 1)} for a, b, stage in zip(cuts, cuts[1:], ids)]}


class Provider:
    """The mocked Responses endpoint: a valid candidate for whatever clip it is asked about, and every request's
    payload and URL. ``damage`` may change an answer in place before it is sent."""

    def __init__(self, status=200, quality=None, last="high", damage=None):
        self.requests, self.urls, self.status, self.quality, self.last = [], [], status, quality, last
        self.damage = damage

    def __call__(self, request):
        payload = json.loads(request.content)
        self.requests.append(payload)
        self.urls.append(str(request.url))
        if self.status != 200:
            return httpx.Response(self.status, json={"error": "synthetic"})
        # The target clip's context; a request with a seed example starts with the example.
        text = next(item["text"] for item in payload["input"][0]["content"]
                    if item["type"] == "input_text" and item["text"].startswith("Annotate only"))
        context = json.loads(text.split("\n", 1)[1])
        answer = round_candidate(context, self.quality, self.last)
        if self.damage:
            self.damage(answer)
        return httpx.Response(200, json={
            "id": f"resp-{len(self.requests)}", "status": "completed", "model": payload["model"], "reasoning": payload["reasoning"],
            "output": [{"type": "message", "content": [{"type": "output_text", "text": json.dumps(answer)}]}],
            "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}})


def playground(root, provider=None, **options):
    return Playground(root, transport=httpx.MockTransport(provider or Provider()), **options)


def submit(root, episode, *, legacy=False, done=True, mode="seed"):
    """The label file grana label saves for one episode on the seed page (``legacy``: the first 11-key format)."""
    cfg = read_json_yaml(root / f"{mode}.yaml")
    loaded = LeRobotSource(Path(cfg["source"]), VIEWS).load_episode(episode)
    count = loaded.frame_count
    doc = new_labels(episode, loaded.source_identity["trajectory_id"], loaded.timestamps, ["1", "2"],
                     source_identity=loaded.source_identity, stage_registry=STAGES)
    doc.update(subtasks=[{"start_frame": 0, "end_frame": count // 2, "stage": "1", "outcome": "high"},
                         {"start_frame": count // 2, "end_frame": count, "stage": "2", "outcome": "high"}],
               quality=[{"start_frame": 3, "end_frame": 6, "label": "suboptimal"}], annotator="Ada", done=done,
               review_confirmed=done, revision="a" * 32, updated_at="2026-11-04T10:00:00+00:00")
    validate_labels(doc, frame_count=count, stages=["1", "2"])
    if legacy:
        doc = {key: value for key, value in doc.items() if key in LEGACY_KEYS}
    path = label_path(cfg["labels"], episode)
    path.write_text(json.dumps(doc))
    return path


def read_json_yaml(path):
    return yaml.safe_load(Path(path).read_text())


def rows(play, route="api"):
    status = play.status()["routes"]
    if route == "api":
        return {row["episode"]: row for row in status["api"]["episodes"]}
    steps = status["seeded"]["steps"]
    return {row["episode"]: row for row in steps["validate"]["episodes"] + steps["apply"]["episodes"]}


def result(play, episode, route="api", model=DEFAULT):
    """One model's job, blocked reason and buttons in an episode's row."""
    return rows(play, route)[episode]["models"][model]



def round_candidate(context, quality=None, last="high"):
    data = candidate(context, quality, last)
    if context.get("annotation_mode") == "stages":
        return {"stages": data["stages"]}
    if context.get("annotation_mode") == "quality":
        frozen = context["frozen_stages"]["segments"]
        assert len(frozen) == 1
        return {"stage_candidate_sha256": context["stage_candidate_sha256"], "stage_annotations": [
            {"stage_segment_index": 0, "aq_intervals": data["aq_intervals"], "aq_boundaries": data["aq_boundaries"]}]}
    return data


def test_default_stages_then_add_quality_import_and_http_contract(source, tmp_path):
    root = init(source, tmp_path)
    provider = Provider()
    play = playground(root, provider)
    play.start("api", [1], DEFAULT)
    play.idle(30)
    first = result(play, 1)
    assert first["status"] == "generated", first
    assert first["can_add_quality"]
    job = play._job("api", 1, DEFAULT)
    saved = (Path(job["output"]) / "candidate/candidate.json").read_bytes()
    assert len(provider.requests) == 1
    assert set(provider.requests[0]["text"]["format"]["schema"]["properties"]) == {"stages"}
    play.add_quality("api", 1, DEFAULT)
    play.idle(30)
    assert result(play, 1)["status"] == "generated", result(play, 1)
    assert job["action_quality_status"] == "annotated"
    assert len(provider.requests) == 3
    assert (Path(job["output"]) / "candidate/candidate.json").read_bytes() == saved
    play.import_episode("api", 1, DEFAULT)
    play.idle(30)
    assert play.state["task"]["status"] == "done", play.state["task"]
    doc = read_json(label_path(play.labels("api"), 1))
    assert doc["machine_import"]["annotation_mode"] == "quality"
    assert not doc["done"] and all(s["outcome"] is None for s in doc["subtasks"])


@pytest.mark.parametrize("selection", ["single_dataset", "route_subset"])
def test_single_episode_human_and_api_do_not_require_seed_transfer(source, tmp_path, monkeypatch, selection):
    from grana.annotation.label_server import create_label_server
    from tools.annotation_playground.label_service import PlaygroundLabelHandler

    options = ["--human-episodes", "0", "--api-episodes", "0", "--seeded-episodes", "0"]
    if selection == "single_dataset":
        version = int(read_json(source / "meta/info.json")["codebase_version"][1])
        source = dataset(tmp_path / "single", version, lengths={0: 20})
        options = []
    monkeypatch.delenv(KEY_ENV)
    root = init(source, tmp_path, *options)
    assert init(source, tmp_path, *options) == root  # The same one-episode workspace resumes.
    provider = Provider()
    play = playground(root, provider)
    status = play.status()
    assert status["routes"]["human"]["episodes"] == 1
    seeded = status["routes"]["seeded"]
    assert not seeded["available"] and seeded["blocked"] == "seeded_episode_count"
    assert all(step["state"] == "blocked" for step in seeded["steps"].values())
    assert result(play, 0)["blocked"] == "no_credential"  # No seed prerequisite on pure API.
    with pytest.raises(Refused) as error:
        play.choose([0], [0])
    assert error.value.code == "seeded_episode_count"
    assert not provider.requests

    # Both real label-page services load the selected episode without a key or seed labels.
    for mode in ("human", "api"):
        config = root / f"{mode}.yaml"
        server = create_label_server(read_json_yaml(config), port=0, config_path=config)
        server.RequestHandlerClass = PlaygroundLabelHandler
        server.playground_mode, server.playground_ports = mode, status["ports"]
        worker = threading.Thread(target=server.serve_forever)
        worker.start()
        connection = HTTPConnection("127.0.0.1", server.server_port, timeout=10)
        try:
            connection.request("GET", "/")
            response = connection.getresponse()
            assert response.status == 200
            assert f'data-playground-mode="{mode}"'.encode() in response.read()
            connection.request("GET", "/api/episode?episode=0")
            response = connection.getresponse()
            assert response.status == 200
            assert json.loads(response.read())["frame_count"] == 20
        finally:
            connection.close()
            server.shutdown()
            worker.join(5)
            server.project.close()
            server.server_close()

    human = submit(root, 0, mode="human")
    original = human.read_bytes()
    monkeypatch.setenv(KEY_ENV, "synthetic-test-key-not-a-credential")
    play.start("api", [0], DEFAULT)
    play.idle(30)
    assert result(play, 0)["status"] == "generated"
    assert len(provider.requests) == 1
    play.import_episode("api", 0, DEFAULT)
    play.idle(30)
    assert play.state["task"]["status"] == "done"
    assert read_json(label_path(play.labels("api"), 0))["machine_import"]["action_quality_status"] == "default_normal_as_not_requested"
    assert human.read_bytes() == original


def test_partial_failed_and_changed_comparisons_never_unlock_targets(source, tmp_path):
    root = init(source, tmp_path)
    for ep in (0, 1, 2):
        submit(root, ep)
    play = playground(root)
    play.choose([0], [1, 2])
    play.build_example()
    play.idle(30)
    assert play.state["task"]["status"] == "done", play.state["task"]
    play.start("seeded", [1], DEFAULT)
    play.idle(30)
    assert result(play, 1, "seeded")["status"] == "generated", result(play, 1, "seeded")
    play.compare(DEFAULT)
    play.idle(30)
    steps = play.status()["routes"]["seeded"]["steps"]
    assert steps["apply"]["validated"] == []
    assert steps["apply"]["state"] == "blocked"
    with pytest.raises(Refused):
        play.start("seeded", [3], DEFAULT)
    play.start("seeded", [2], DEFAULT)
    play.idle(30)
    play.compare(DEFAULT)
    play.idle(30)
    steps = play.status()["routes"]["seeded"]["steps"]
    assert steps["apply"]["validated"] == [DEFAULT], play.state["task"]
    compare = next(iter(play.state["compares"].values()))[DEFAULT]
    assert len([r for r in compare["results"] if r["status"] == "compared"]) == 2
    # Default normal must not become a measured AQ accuracy.
    metrics = read_json(root / compare["folder"] / "aq_metrics_000001.json")
    assert metrics["status"] == "default_normal_as_not_requested" and metrics["accuracy"] is None
    detail = play.compare_detail(Path(compare["folder"]).name, 1)
    assert detail["curves"]["problems"]["model"]
    compare["status"] = 1
    assert not play.status()["routes"]["seeded"]["steps"]["apply"]["validated"]
    compare["status"] = 0
    path = label_path(play.labels("seed"), 1)
    doc = read_json(path)
    doc["quality"] = []
    path.write_text(json.dumps(doc))
    assert not play.status()["routes"]["seeded"]["steps"]["apply"]["validated"]


def test_seed_edits_invalidate_examples_and_pending_imports_but_resaves_do_not(source, tmp_path):
    from copy import deepcopy

    root = init(source, tmp_path, "--max-requests", 12)
    seed, held = submit(root, 0), submit(root, 1)
    play = playground(root)
    play.choose([0], [1])
    play.build_example()
    play.idle(30)
    play.start("seeded", [1], DEFAULT)
    play.idle(30)
    play.compare(DEFAULT)
    play.idle(30)
    play.start("seeded", [2], DEFAULT)
    play.idle(30)
    play.import_all("seeded", "preview", model=DEFAULT)
    play.idle(30)
    preview = play.state["imports"]["seeded"]["preview"]["id"]
    original = read_json(seed)
    template = deepcopy(play.state["template"])

    def steps():
        return play.status()["routes"]["seeded"]["steps"]

    assert steps()["example"]["state"] == "done"
    assert steps()["apply"]["validated"] == [DEFAULT]
    assert result(play, 2, "seeded")["can_import"]
    # Another save/reviewer changes bookkeeping, not what the model was shown.
    seed.write_text(json.dumps({**original, "revision": "b" * 32, "annotator": "Bo",
                                "updated_at": "2026-11-05T09:00:00+00:00"}, indent=1))
    assert steps()["example"]["state"] == "done"
    assert steps()["example"]["changed"] == []
    assert steps()["apply"]["validated"] == [DEFAULT]
    assert play._version()["key"] == template["key"]
    # A preview does not authorize import after the held-out labels change either.
    held_raw = held.read_bytes()
    held_doc = read_json(held)
    held.write_text(json.dumps({**held_doc, "quality": []}))
    with pytest.raises(Refused) as error:
        play.import_all("seeded", "confirm", preview=preview)
    assert error.value.code == "compare_before_import"
    held.write_bytes(held_raw)
    # Labels, review state and source bindings all matter. Same-size replacements must be noticed too.
    for field, value in (("quality", []), ("done", False), ("review_confirmed", False),
                         ("source_identity", {**original["source_identity"], "trajectory_id": "other"}),
                         ("stage_registry", [{**STAGES[0], "prompt": "Different meaning"}, STAGES[1]]),
                         ("timestamps_sha256", "0" * 64)):
        seed.write_text(json.dumps({**original, field: value}))
        assert play._version() is None, field
        assert steps()["example"]["changed"] == [0], field
        assert not steps()["apply"]["validated"], field
        with pytest.raises(Refused):
            play.import_all("seeded", "confirm", preview=preview)
    seed.write_text(json.dumps(original))
    # The frozen example still requires exact integrity, even if live labels match the changed snapshot.
    snapshot = root / template["folder"] / "seeds/1/labels.json"
    snapshot_raw = snapshot.read_bytes()
    snapshot.write_text(json.dumps(original, indent=1))
    assert play._version() is None
    snapshot.write_bytes(snapshot_raw)
    assert play._version() is not None
    # A real boundary change marks the example stale, hides its jobs and blocks all import paths.
    changed = deepcopy(original)
    changed["subtasks"][0]["end_frame"] += 2
    changed["subtasks"][1]["start_frame"] += 2
    seed.write_text(json.dumps(changed))
    assert steps()["example"]["state"] == "stale"
    assert steps()["example"]["changed"] == [0]
    assert not steps()["validate"]["comparisons"][0]["current"]
    for operation in (lambda: play.start("seeded", [3], DEFAULT),
                      lambda: play.import_episode("seeded", 2, DEFAULT),
                      lambda: play.import_all("seeded", "preview", model=DEFAULT),
                      lambda: play.import_all("seeded", "confirm", preview=preview)):
        with pytest.raises(Refused):
            operation()
    assert not label_path(play.labels("seeded"), 2).exists()
    # Rebuilding and comparing again unlocks new results, but never an older preview.
    play.build_example()
    play.idle(30)
    assert steps()["example"]["state"] == "done" and steps()["example"]["changed"] == []
    assert play.state["template"]["key"] != template["key"]
    play.start("seeded", [1], DEFAULT)
    play.idle(30)
    play.compare(DEFAULT)
    play.idle(30)
    assert steps()["apply"]["validated"] == [DEFAULT]
    with pytest.raises(Refused) as error:
        play.import_all("seeded", "confirm", preview=preview)
    assert error.value.code == "preview_again"
    play.start("seeded", [2], DEFAULT)
    play.idle(30)
    play.import_all("seeded", "preview", model=DEFAULT)
    play.idle(30)
    current_preview = play.state["imports"]["seeded"]["preview"]["id"]
    play.import_all("seeded", "confirm", preview=current_preview)
    play.idle(30)
    assert play.state["task"]["status"] == "done", play.state["task"]
    assert label_path(play.labels("seeded"), 2).is_file()


def test_quality_failure_keeps_stages_and_never_auto_retries(source, tmp_path):
    root = init(source, tmp_path)
    provider = Provider()
    play = playground(root, provider)
    play.start("api", [1], DEFAULT)
    play.idle(30)
    provider.status = 503
    play.add_quality("api", 1, DEFAULT)
    play.idle(30)
    row = result(play, 1)
    assert row["status"] == "generated" and row["can_import"]
    assert row["job"]["phase"] == "quality_stopped"
    assert row["job"]["action_quality_status"] == "default_normal_as_not_requested"
    assert not row["can_add_quality"]
    assert len(provider.requests) == 2
    play.import_episode("api", 1, DEFAULT)
    play.idle(30)
    doc = read_json(label_path(play.labels("api"), 1))
    assert doc["machine_import"]["action_quality_status"] == "default_normal_as_not_requested"


def short_last(answer):
    """Schema-valid, semantically invalid: the answer's last stage segment or AQ interval leaves its last frame out."""
    rows = answer["stages"]["segments"] if "stages" in answer else answer["stage_annotations"][0]["aq_intervals"]
    rows[-1]["end_frame"] -= 1


def test_invalid_answer_in_a_chunk_is_invalid_candidate_not_uncertain(source, tmp_path):
    root = init(source, tmp_path)
    provider = Provider(damage=short_last)
    play = playground(root, provider)
    play.start("api", [1], DEFAULT)
    play.idle(30)
    row = result(play, 1)
    job = play._job("api", 1, DEFAULT)
    assert len(provider.requests) == 1
    # The answer is kept in its chunk's folder; the job is not the uncertain outcome of an unanswered request.
    delivery = read_json(Path(job["output"]) / "requests/stage-00000/candidate/delivery.json")
    assert delivery["status"] == "semantic_validation_failed"
    assert row["status"] == "failed" and row["blocked"] == "failed_job"
    assert (job["status"], job["error_code"]) == ("error", "invalid_candidate")
    assert row["job"]["note"]["code"] == "job_invalid_candidate"
    assert delivery["validation_error"] not in job["error"] and "cover" not in job["error"]


def test_invalid_quality_answer_keeps_stages_and_names_invalid_candidate(source, tmp_path):
    root = init(source, tmp_path)
    provider = Provider()
    play = playground(root, provider)
    play.start("api", [1], DEFAULT)
    play.idle(30)
    provider.damage = short_last
    play.add_quality("api", 1, DEFAULT)
    play.idle(30)
    row = result(play, 1)
    job = play._job("api", 1, DEFAULT)
    assert len(provider.requests) == 2
    assert read_json(Path(job["output"]) / "requests/quality-00000-00000/candidate/delivery.json")["status"] == "semantic_validation_failed"
    assert row["status"] == "generated" and row["can_import"] and not row["can_add_quality"]
    assert (job["phase"], job["error_code"]) == ("quality_stopped", "invalid_candidate")
    assert row["job"]["note"]["code"] == "job_invalid_candidate"


def test_planned_attempts_follow_round_splitting():
    from grana.annotation.rounds import DEFAULTS, split
    edges = [frame / 10 for frame in range(3001)]   # a 300 s episode at 10 fps
    count = len(edges) - 1
    stages = split(edges, 0, count, DEFAULTS["stage_seconds"], DEFAULTS["stage_stride"], DEFAULTS["max_images"])
    assert planned_attempts(edges) == len(stages) > 1
    # Before the stage answer, AQ is a lower bound for every frozen segmentation; with the segments, exact.
    bound = planned_attempts(edges, stages=False, quality=True)
    assert bound == 30 and planned_attempts(edges, quality=True) == len(stages) + bound
    for cuts in ([], [1], [5, 2000], list(range(97, count, 97))):
        segments = list(zip([0, *cuts], [*cuts, count]))
        exact = sum(len(split(edges, a, b, DEFAULTS["quality_seconds"], DEFAULTS["quality_stride"], DEFAULTS["max_images"]))
                    for a, b in segments)
        assert planned_attempts(edges, stages=False, quality=True, segments=segments) == exact >= bound


def test_budget_counts_campaign_reservations_and_admits_only_whole_campaigns(source, tmp_path):
    root = init(source, tmp_path)   # three requests
    provider, seen = Provider(), []

    def observe(request):
        seen.append(play._budget())
        return provider(request)

    play = Playground(root, transport=httpx.MockTransport(observe))
    play.start("api", [0], DEFAULT)
    play.idle(30)
    # In flight, the running job's attempt is reserved as <id>-stage-00000 and is not pending as well.
    assert [(b["used_attempts"], b["pending_attempts"], b["remaining_attempts"]) for b in seen] == [(1, 0, 2)]
    play.start("api", [1], DEFAULT)
    play.idle(30)
    assert play.status()["budget"]["remaining_attempts"] == 1
    # Stages and AQ need at least two requests: refused as a whole with one left, nothing queued or sent.
    with pytest.raises(Refused) as refused:
        play.start("api", [2], DEFAULT, with_quality=True)
    assert (refused.value.code, refused.value.values) == ("over_budget", {"asked": 2, "left": 1})
    assert result(play, 2)["status"] == "not_generated"
    # AQ of episode 1's two frozen stage segments needs two requests too.
    assert result(play, 1)["can_add_quality"]
    with pytest.raises(Refused) as refused:
        play.add_quality("api", 1, DEFAULT)
    assert (refused.value.code, refused.value.values) == ("over_budget", {"asked": 2, "left": 1})
    assert result(play, 1)["job"]["action_quality_status"] == "default_normal_as_not_requested"
    assert len(provider.requests) == 2
    play.start("api", [2], DEFAULT)
    play.idle(30)
    assert result(play, 2)["status"] == "generated" and play.status()["budget"]["remaining_attempts"] == 0
    assert len(provider.requests) == 3


def test_template_batch_default_stages_and_completed_resume(source, tmp_path):
    from grana.annotation.templates import prepare_campaign, run_campaign
    from scripts.annotation_skill_api import run
    root = init(source, tmp_path)
    for ep in (0, 1):
        submit(root, ep)
    play = playground(root)
    play.choose([0], [1])
    play.build_example()
    play.idle(30)
    template = root / play.state["template"]["folder"]
    manifest = tmp_path / "targets.json"
    manifest.write_text(json.dumps({"schema_version": "grana.annotation_targets.v1", "task": "demo",
                                   "targets": [{"name": "same-task", "source": str(source), "episodes": [2]}]}))
    destination = tmp_path / "batch"
    assert prepare_campaign(template, manifest, destination)["http_requests"] == 0
    provider = Provider()
    def worker(*args, **options):
        return run(*args, **options, transport=httpx.MockTransport(provider))
    options = dict(max_requests=1, api_key_env=KEY_ENV, run_worker=worker)
    assert run_campaign(template, manifest, destination, **options)["status"] == "completed"
    assert run_campaign(template, manifest, destination, **options)["jobs"][0]["cache_hit"]
    assert len(provider.requests) == 1
    payload = provider.requests[0]
    assert set(payload["text"]["format"]["schema"]["properties"]) == {"stages"}
    seed = next(item["text"] for item in payload["input"][0]["content"] if item.get("text", "").startswith("REFERENCE EXAMPLE ONLY"))
    assert "human_quality_exceptions" not in seed and '"outcome"' not in seed


@pytest.mark.skipif(os.environ.get("GRANA_BROWSER_SMOKE") != "1", reason="Opt-in real browser integration")
@pytest.mark.parametrize("single_episode", [False, True], ids=["multiple", "single"])
@pytest.mark.parametrize("configured", [True, False], ids=["provider", "human-only"])
def test_playground_round_controls_in_chrome(source, tmp_path, single_episode, configured, monkeypatch):
    chrome = os.environ.get("CHROME_BIN") or shutil.which("google-chrome")
    if not chrome or not shutil.which("node"):
        pytest.skip("Chrome and Node are required")
    if single_episode:
        version = int(read_json(source / "meta/info.json")["codebase_version"][1])
        source = dataset(tmp_path / "single", version, lengths={0: 20})
    if not configured:
        from grana.annotation.api import BASE_URL_ENV, MODEL_ENV
        for name in (BASE_URL_ENV, MODEL_ENV, KEY_ENV):
            monkeypatch.delenv(name, raising=False)
    root = init(source, tmp_path, "--max-requests", 12)
    provider = Provider()
    if not single_episode and configured:
        seed = submit(root, 0)
        submit(root, 1)
        play = playground(root)
        play.choose([0], [1])
        play.build_example()
        play.idle(30)
        assert play.state["task"]["status"] == "done", play.state["task"]
        doc = read_json(seed)
        doc["subtasks"][0]["end_frame"] += 2
        doc["subtasks"][1]["start_frame"] += 2
        seed.write_text(json.dumps(doc))
    output = Path(os.environ.get("GRANA_BROWSER_OUTPUT", str(tmp_path))) / ("playground-" + source.parents[1].name + "-" + source.parent.name + ("-single" if single_episode else "-multiple"))
    with running(root, port=0, transport=httpx.MockTransport(provider)) as server:
        worker = threading.Thread(target=server.serve_forever)
        worker.start()
        try:
            process = subprocess.Popen(["node", str(Path(__file__).with_name("annotation_playground_browser.cjs")),
                                      f"http://127.0.0.1:{server.server_port}/", chrome, str(output)],
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            try:
                log, _ = process.communicate(timeout=100)
            except subprocess.TimeoutExpired:
                process.terminate()  # Node's handler also stops its Chrome child.
                log, _ = process.communicate(timeout=15)
                pytest.fail("Playground browser timed out: " + log)
            assert process.returncode == 0, log
            assert len(provider.requests) == ((3 if single_episode else 6) if configured else 0)
        finally:
            server.playground.idle(30)
            server.shutdown()
            worker.join(5)


def test_unconfigured_playground_keeps_human_labeling_available(source, tmp_path, monkeypatch):
    from grana.annotation import api
    for name in (api.BASE_URL_ENV, api.MODEL_ENV, api.KEY_ENV, KEY_ENV):
        monkeypatch.delenv(name, raising=False)
    root = init(source, tmp_path)
    provider = Provider()
    play = playground(root, provider)
    status = play.status()
    assert status["models"] == []
    assert status["routes"]["human"]
    assert all(not row["models"] for row in status["routes"]["api"]["episodes"])
    submit(root, 0, mode="human")
    assert label_path(play.labels("human"), 0).is_file()
    with pytest.raises(ValueError, match="provider"):
        play.start("api", [0], "default")
    assert not provider.requests
    assert not (root / "api-budget" / "budget.json").exists()


def test_resume_uses_recorded_models_without_provider_environment(source, tmp_path, monkeypatch):
    from grana.annotation import api
    selected = [entry("custom")]
    root = init(source, tmp_path, models=selected)
    monkeypatch.setenv(api.MODEL_ENV, "different/model")
    monkeypatch.setenv(api.BASE_URL_ENV, "https://different.example/v1")
    options = launcher.parser().parse_args(["--config", str(recipe(source, tmp_path)), "--workspace", str(root)])
    assert launcher.initialize(options) == root
    assert Playground(root).config["models"] == selected
    for name in (api.MODEL_ENV, api.BASE_URL_ENV, KEY_ENV):
        monkeypatch.delenv(name, raising=False)
    assert launcher.initialize(options) == root
    assert Playground(root).config["models"] == selected


def test_playground_cli_model_and_endpoint_override_environment():
    provider = launcher.configured_models(KEY_ENV, model="chosen/model", base_url="https://chosen.example/v2")
    assert provider == [{"id": "default", "label": "model", "model": "chosen/model",
                         "base_url": "https://chosen.example/v2", "api_key_env": KEY_ENV,
                         "reasoning_effort": "xhigh", "protocol": "responses"}]


@pytest.mark.parametrize("partial", [{}, {"model": "chosen/vision"}, {"base_url": "https://chosen.example/v1"}])
def test_incomplete_provider_keeps_offline_playground_available(monkeypatch, partial):
    from grana.annotation import api
    for name in (api.BASE_URL_ENV, api.MODEL_ENV, api.KEY_ENV, KEY_ENV):
        monkeypatch.delenv(name, raising=False)
    assert launcher.configured_models(KEY_ENV, **partial) == []
