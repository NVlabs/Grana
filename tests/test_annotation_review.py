# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Review revisions are explicit, source-bound and never mutate candidates."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
from http.client import HTTPConnection
import json
import threading

import pytest

from grana.annotation.review import HTML, MAX_BODY_BYTES, ReviewStore, create_review_server


@pytest.fixture
def run(tmp_path):
    root = tmp_path / "annotation"
    original = root / "episodes/000219/final.json"
    original.parent.mkdir(parents=True)
    candidate = {
        "segments": [
            {"start_frame": 0, "end_frame": 3, "stage_id": "S1", "identity_status": "resolved",
             "reason": "Drawer opening visible", "evidence_frames": [0, 2]},
            {"start_frame": 3, "end_frame": 6, "stage_id": "S2", "identity_status": "uncertain",
             "reason": "Insertion partly occluded", "evidence_frames": [3, 5]},
        ],
        "boundaries": [{"frame": 3, "lower_frame": 2, "upper_frame": 4,
                        "reason": "Contact bracket", "evidence_frames": [2, 3, 4]}],
        "omitted_stages": [{"stage_id": "S3", "reason": "Clip ends before closing"}],
        "invalid_intervals": [], "summary": "Machine draft; insertion identity uncertain.",
    }
    original.write_text(json.dumps(candidate))
    video = tmp_path / "original.mp4"
    video.write_bytes(bytes(range(256)) * 16)
    manifest = {
        "schema_version": 1, "task": "Put spatula in drawer",
        "stages": [{"id": f"S{i}", "label": str(i), "prompt": prompt}
                   for i, prompt in enumerate(("Open drawer", "Insert spatula", "Close drawer"), 1)],
        "episodes": [{"episode_index": 219, "frame_count": 6,
                      "time_edges": [10, 10.1, 10.3, 10.4, 10.6, 10.8, 11],
                      "video_paths": {"top": str(video), "left": str(video)},
                      "source_identity": {"trajectory_id": "episode-219", "timestamps_sha256": "a" * 64,
                                          "video_sha256": {view: hashlib.sha256(video.read_bytes()).hexdigest() for view in ("top", "left")}},
                      "clock_receipt": {"cameras": {"top": {"source_minus_video_offset_seconds": 10},
                                                       "left": {"source_minus_video_offset_seconds": 9.5}}},
                      "candidate_file": "episodes/000219/final.json"}],
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root, candidate, manifest, video


def payload(candidate, kind="human", decision="needs_changes"):
    return {"episode_index": 219, "candidate": deepcopy(candidate),
            "review": {"reviewer": "Reviewer named explicitly", "kind": kind, "decision": decision,
                       "notes": "Boundary requires inspection.", "evidence_inspected": ["Top and left views, source frames 2–4"]}}


@pytest.fixture
def running(run):
    server = create_review_server(run[0], port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server, run
    server.shutdown()
    server.server_close()
    thread.join(timeout=3)


def http(server, method, path, body=None, headers=None):
    connection = HTTPConnection("127.0.0.1", server.server_port, timeout=5)
    connection.request(method, path, body=body, headers=headers or {})
    response = connection.getresponse()
    result = response.status, dict(response.getheaders()), response.read()
    connection.close()
    return result


def post(server, data, **extra):
    headers = {"Origin": f"http://127.0.0.1:{server.server_port}", "Content-Type": "application/json", **extra}
    return http(server, "POST", "/api/revision", json.dumps(data).encode(), headers)


def test_revisions_bind_source_registry_clock_hash_and_preserve_originals(run):
    root, candidate, manifest, video = run
    originals = {path: path.read_bytes() for path in (root / "manifest.json", root / "episodes/000219/final.json", video)}
    store = ReviewStore(root)
    request = payload(candidate, decision="accepted")
    first = store.save(request)
    request["candidate"]["summary"] = "Agent follow-up; not a human decision"
    request["review"].update(kind="agent", reviewer="Explicit agent", decision="needs_changes")
    second = store.save(request)
    assert first["revision"] != second["revision"]
    assert len(list((root / "revisions/episode_000219").glob("*.json"))) == 2
    saved = json.loads((root / first["relative_path"]).read_text())
    assert saved["candidate"] == candidate
    assert saved["source_identity"] == manifest["episodes"][0]["source_identity"]
    assert saved["stages"] == manifest["stages"]
    assert saved["time_edges"] == manifest["episodes"][0]["time_edges"]
    expected = hashlib.sha256(json.dumps(candidate, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    assert saved["candidate_sha256"] == first["candidate_sha256"] == expected
    assert saved["review"] == payload(candidate, decision="accepted")["review"]
    assert saved["created_at"].endswith("+00:00")
    assert store.episode(219)["review"] is None
    assert store.episode(219)["candidate"] == candidate
    assert store.episode(219, second["revision"])["review"]["kind"] == "agent"
    assert all(path.read_bytes() == data for path, data in originals.items())
    assert not list((root / "revisions/episode_000219").glob("*.tmp"))


@pytest.mark.parametrize("mutate", [
    lambda p: p["review"].update(reviewer=" "),
    lambda p: p["review"].update(kind=""),
    lambda p: p["review"].update(decision=""),
    lambda p: p["review"].pop("decision"),
    lambda p: p["review"].update(evidence_inspected=[]),
    lambda p: p["review"].update(evidence_inspected=[3]),
    lambda p: p["review"].update(evidence_inspected=["same", " same "]),
    lambda p: p["review"].update(notes=False),
    lambda p: p["review"].update(notes=" "),
    lambda p: p.update(source_identity={"forged": True}),
    lambda p: p.update(episode_index=True),
    lambda p: p["candidate"]["segments"][0].update(end_frame=2),
    lambda p: p["candidate"]["segments"][0].update(stage_id="NEW"),
    lambda p: p["candidate"]["boundaries"][0].update(lower_frame=4),
    lambda p: p["candidate"]["segments"][0].update(outcome="high"),
])
def test_invalid_review_or_candidate_creates_no_revision(run, mutate):
    store = ReviewStore(run[0])
    request = payload(run[1])
    mutate(request)
    with pytest.raises(ValueError):
        store.save(request)
    assert not (run[0] / "revisions").exists()


def test_concurrent_revisions_are_distinct_atomic_and_original_immutable(run):
    store = ReviewStore(run[0])
    with ThreadPoolExecutor(max_workers=4) as pool:
        saved = list(pool.map(lambda _: store.save(payload(run[1], "agent", "accepted")), range(8)))
    assert len({item["revision"] for item in saved}) == 8
    for item in saved:
        assert store.episode(219, item["revision"])["candidate"] == run[1]
    assert store.episode(219)["review"] is None


@pytest.mark.parametrize("candidate_path", ["../outside.json", "/tmp/outside.json", "episodes/../final.json", "episodes\\final.json"])
def test_manifest_candidate_cannot_escape_run(run, candidate_path):
    root, _, manifest, _ = run
    manifest["episodes"][0]["candidate_file"] = candidate_path
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        ReviewStore(root)


def test_manifest_requires_source_video_clock_not_guessed_fps(run):
    root, _, manifest, _ = run
    manifest["episodes"][0]["clock_receipt"]["cameras"]["top"] = {}
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="offset"):
        ReviewStore(root)


def test_startup_rejects_media_changed_since_source_manifest(run):
    run[3].write_bytes(b"Replaced after annotation")
    with pytest.raises(ValueError, match="SHA256"):
        ReviewStore(run[0])


def test_symlink_revision_directory_cannot_write_outside_run(run, tmp_path):
    root = run[0]
    store = ReviewStore(root)
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "revisions").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        store.save(payload(run[1]))
    assert list(outside.iterdir()) == []


def test_download_api_and_ui_preserve_machine_pending_state(running):
    server, (_, candidate, manifest, _) = running
    status, _, body = http(server, "GET", "/api/manifest")
    assert status == 200
    public = json.loads(body)
    assert public["stages"] == manifest["stages"]
    assert public["episodes"][0]["video_urls"] == {"top": "/media/219/top", "left": "/media/219/left"}
    assert "video_paths" not in public["episodes"][0]
    status, _, body = http(server, "GET", "/api/episode?episode=219")
    reply = json.loads(body)
    assert status == 200 and reply["candidate"] == candidate and reply["review"] is None
    assert reply["coverage"]["uncertain_identity"]["frames"] == 3
    status, headers, body = http(server, "GET", "/api/candidate?episode=219")
    assert status == 200 and json.loads(body) == candidate
    assert 'filename="episode_000219_stages.json"' in headers["Content-Disposition"]
    status, _, body = http(server, "GET", "/")
    assert status == 200 and "Machine candidate ≠ human accepted" in body.decode()
    assert '<option value="">Choose explicitly</option>' in HTML
    assert "source_minus_video_offset_seconds" in HTML
    assert "<script src=" not in HTML and "https://" not in HTML


def test_http_revision_is_explicit_and_reloadable(running):
    server, (root, candidate, _, _) = running
    status, _, body = post(server, payload(candidate))
    saved = json.loads(body)
    assert status == 201
    assert (root / saved["relative_path"]).is_file()
    status, _, body = http(server, "GET", "/api/episode?episode=219&revision=" + saved["revision"])
    assert status == 200 and json.loads(body)["review"]["decision"] == "needs_changes"
    status, _, body = http(server, "GET", "/api/candidate?episode=219&revision=" + saved["revision"])
    assert status == 200 and json.loads(body) == candidate


@pytest.mark.parametrize("origin", [None, "null", "http://evil.example", "https://127.0.0.1:8788"])
def test_writes_require_same_origin_and_do_not_default_to_acceptance(running, origin):
    server, (root, candidate, _, _) = running
    headers = {"Content-Type": "application/json"}
    if origin is not None:
        headers["Origin"] = origin
    assert http(server, "POST", "/api/revision", json.dumps(payload(candidate)), headers)[0] == 403
    assert not (root / "revisions").exists()


def test_dns_rebinding_host_and_oversized_or_duplicate_json_rejected(running):
    server, (root, candidate, _, _) = running
    assert post(server, payload(candidate), Host="evil.example", Origin="http://evil.example")[0] == 403
    origin = f"http://127.0.0.1:{server.server_port}"
    status, _, _ = http(server, "POST", "/api/revision", b"{}", {"Origin": origin, "Content-Type": "application/json", "Content-Length": str(MAX_BODY_BYTES + 1)})
    assert status == 413
    status, _, _ = http(server, "POST", "/api/revision", b'{"candidate":{},"candidate":{}}', {"Origin": origin, "Content-Type": "application/json"})
    assert status == 400
    assert not (root / "revisions").exists()


@pytest.mark.parametrize("range_header,status,start,end", [
    (None, 200, 0, 4095), ("bytes=3-12", 206, 3, 12),
    ("bytes=4090-", 206, 4090, 4095), ("bytes=-7", 206, 4089, 4095),
])
def test_allowlisted_video_supports_real_ranges_and_head(running, range_header, status, start, end):
    server, (_, _, _, video) = running
    headers = {"Range": range_header} if range_header else {}
    code, response_headers, body = http(server, "GET", "/media/219/top", headers=headers)
    assert code == status and body == video.read_bytes()[start:end + 1]
    assert response_headers["Content-Length"] == str(end - start + 1)
    assert response_headers["Accept-Ranges"] == "bytes"
    if range_header:
        assert response_headers["Content-Range"] == f"bytes {start}-{end}/4096"
    code, head_headers, body = http(server, "HEAD", "/media/219/top", headers=headers)
    assert code == status and body == b""
    assert head_headers["Content-Length"] == response_headers["Content-Length"]


@pytest.mark.parametrize("range_header", ["bytes=4096-", "bytes=5-2", "bytes=-0", "bytes=0-1,3-4"])
def test_invalid_video_ranges_return_416(running, range_header):
    code, headers, _ = http(running[0], "GET", "/media/219/top", headers={"Range": range_header})
    assert code == 416 and headers["Content-Range"] == "bytes */4096"


@pytest.mark.parametrize("url", ["/../manifest.json", "/%2e%2e/manifest.json", "/media/219/top?file=/etc/passwd", "/media/219/../../manifest.json", "/episodes/000219/final.json", "/api/episode?episode=999", "/api/episode?episode=219&revision=../../manifest.json"])
def test_only_allowlisted_media_and_revision_names_are_readable(running, url):
    assert http(running[0], "GET", url)[0] == 404


def test_replaced_original_video_is_not_served(running, tmp_path):
    server, (_, _, _, video) = running
    original = video.read_bytes()
    video.write_bytes(b"changed original")
    assert http(server, "GET", "/media/219/top")[0] == 404
    video.unlink()
    elsewhere = tmp_path / "private"
    elsewhere.write_bytes(original)
    video.symlink_to(elsewhere)
    assert http(server, "GET", "/media/219/top")[0] in (400, 404)


def test_revision_tamper_is_detected_without_modifying_original(run):
    store = ReviewStore(run[0])
    saved = store.save(payload(run[1]))
    path = run[0] / saved["relative_path"]
    wrapper = json.loads(path.read_text())
    wrapper["candidate"]["summary"] = "Changed without updating binding"
    path.write_text(json.dumps(wrapper))
    with pytest.raises(ValueError, match="hash"):
        store.episode(219, saved["revision"])
    assert store.episode(219)["candidate"] == run[1]


def test_server_rejects_nonloopback_before_binding(run):
    with pytest.raises(ValueError, match="loopback"):
        create_review_server(run[0], host="0.0.0.0", port=0)
