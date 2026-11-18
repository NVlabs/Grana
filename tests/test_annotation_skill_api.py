# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""All API traffic is mocked; fixtures are synthetic RGB and candidate JSON."""
import base64
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import httpx
from PIL import Image
import pytest

from test_annotation_api import provider_configuration  # noqa: F401

from grana.annotation.api import KEY_ENV
from scripts import annotation_skill as skill
from scripts import annotation_skill_api as driver


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    monkeypatch.setenv(KEY_ENV, "synthetic-test-key-not-a-credential")
    context = json.loads((skill.POLICY_DIR / "references/synthetic-context.json").read_text())
    context["source_identity"] = {"source_root": str(tmp_path / "PRIVATE_FAILURE_LABEL_SOURCE"),
                                  "raw_label": "PRIVATE_HUMAN_FAILURE_ANNOTATION"}
    candidate = json.loads((skill.POLICY_DIR / "references/synthetic-candidate.json").read_text())
    policy = tmp_path / "frozen-policy"
    for name in skill.POLICY_FILES:
        path = policy / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("Frozen synthetic policy: keep unsupported annotations unknown.\n")
    bundle = skill.prepare_bundle(context, policy, mode="combined")
    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_text(json.dumps(bundle))
    rows = []
    for frame in reversed(context["evidence_frames"]):
        path = tmp_path / f"PRIVATE_FAILURE_IMAGE_{frame}.png"
        Image.new("RGB", (8, 4), (frame, 20, 40)).save(path)
        rows.append({"frame": frame, "timestamp": context["time_edges"][frame],
                     "image_path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                     "view_names": ["left", "wrist"]})
    manifest = {"schema_version": driver.EVIDENCE_SCHEMA, "images": rows}
    manifest_path = tmp_path / "evidence.json"
    manifest_path.write_text(json.dumps(manifest))
    return {"bundle_path": bundle_path, "evidence_path": manifest_path,
            "output_dir": tmp_path / "result", "policy_dir": policy}, candidate, manifest


def provider(payload, candidate):
    return {"id": "resp-synthetic", "status": "completed", "model": payload["model"],
            "reasoning": payload["reasoning"],
            "output": [{"type": "message", "content": [{"type": "output_text", "text": json.dumps(candidate)}]}],
            "usage": {"input_tokens": 12, "output_tokens": 8, "total_tokens": 20, "cost": None}}


def test_verified_mosaics_are_ordered_and_private_identity_is_not_sent(inputs, monkeypatch):
    args, candidate, manifest = inputs
    seen = []

    def handler(request):
        payload = json.loads(request.content)
        seen.append(payload)
        assert request.url.path == "/v1/responses"
        assert "PRIVATE_" not in request.content.decode()
        content = payload["input"][0]["content"]
        context = json.loads(content[0]["text"].split("\n", 1)[1])
        assert "source_identity" not in context and "input_id" in context
        assert "need not be a complete physical trajectory" in content[0]["text"]
        for frame, label, image in zip((0, 2), content[1::2], content[2::2], strict=True):
            row = next(item for item in manifest["images"] if item["frame"] == frame)
            assert json.loads(label["text"]) == {key: row[key] for key in ("frame", "timestamp", "view_names")}
            raw = base64.b64decode(image["image_url"].split(",", 1)[1])
            assert hashlib.sha256(raw).hexdigest() == row["sha256"]
        return httpx.Response(200, json=provider(payload, candidate))

    result = driver.run(**args, transport=httpx.MockTransport(handler))
    assert result["status"] == "valid_candidate" and result["accepted_for_training"] is False
    assert result["image_count"] == 2 and result["cache_hit"] is False
    output = args["output_dir"]
    delivery = json.loads((output / "delivery.json").read_text())
    assert delivery["frame_order"] == [0, 2] and delivery["source_identity_sent"] is False
    assert delivery["usage"]["cost"] is None and delivery["response_id"] == "resp-synthetic"
    assert "not independently verified" in delivery["delivery_scope"]
    validation = json.loads((output / "validation.json").read_text())
    assert validation["image_delivery_verified"] is False
    assert validation["semantic_accuracy_verified"] is False
    assert json.loads((output / "candidate.json").read_text()) == candidate
    monkeypatch.delenv(KEY_ENV)
    resumed = driver.run(**args, transport=httpx.MockTransport(lambda _: pytest.fail("completed request must be reused")))
    assert resumed["cache_hit"] is True and len(seen) == 1
    assert json.loads((output / "delivery.json").read_text()) == delivery


@pytest.mark.parametrize("damage", ["missing", "duplicate", "extra", "hash", "clock", "views", "fields", "corrupt", "grayscale"])
def test_bad_evidence_fails_before_network_or_output_creation(inputs, damage):
    args, _, manifest = inputs
    rows = manifest["images"]
    if damage == "missing":
        rows.pop()
    elif damage == "duplicate":
        rows.append(deepcopy(rows[0]))
    elif damage == "extra":
        rows.append(rows[0] | {"frame": 1})
    elif damage == "hash":
        rows[0]["sha256"] = "0" * 64
    elif damage == "clock":
        rows[0]["timestamp"] += 0.01
    elif damage == "views":
        rows[0]["view_names"] = ["wrist", "left"]
    elif damage == "fields":
        rows[0]["raw_label"] = "failure"
    else:
        path = args["evidence_path"].parent / rows[0]["image_path"]
        if damage == "corrupt":
            path.write_bytes(b"not an image")
        else:
            Image.new("L", (8, 4)).save(path)
        rows[0]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    args["evidence_path"].write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        driver.run(**args, transport=httpx.MockTransport(lambda _: pytest.fail("no API for invalid evidence")))
    assert not args["output_dir"].exists()


def test_frozen_policy_mismatch_fails_before_network(inputs):
    args, _, _ = inputs
    (args["policy_dir"] / "SKILL.md").write_text("Changed policy")
    with pytest.raises(ValueError, match="Frozen bundle"):
        driver.run(**args, transport=httpx.MockTransport(lambda _: pytest.fail("no API for policy mismatch")))


def test_existing_output_is_not_reinterpreted_by_a_new_bundle(inputs):
    args, candidate, _ = inputs
    transport = httpx.MockTransport(lambda req: httpx.Response(200, json=provider(json.loads(req.content), candidate)))
    driver.run(**args, transport=transport)
    before = (args["output_dir"] / "candidate.json").read_bytes()
    bundle = json.loads(args["bundle_path"].read_text())
    bundle["context"]["task_policy"]["version"] = "new-policy"
    args["bundle_path"].write_text(json.dumps(skill.prepare_bundle(bundle["context"], args["policy_dir"], mode="combined")))
    with pytest.raises(ValueError, match="Existing artifact differs"):
        driver.run(**args, transport=httpx.MockTransport(lambda _: pytest.fail("reject before a new billable request")))
    assert (args["output_dir"] / "candidate.json").read_bytes() == before


def test_shared_request_name_cannot_reuse_a_different_private_source(inputs):
    args, candidate, _ = inputs
    requests = []

    def handler(request):
        payload = json.loads(request.content)
        requests.append(payload)
        return httpx.Response(200, json=provider(payload, candidate))

    transport = httpx.MockTransport(handler)
    args["api_dir"] = args["output_dir"].parent / "shared-campaign"
    first = driver.run(**args, transport=transport)
    first_delivery = json.loads((args["output_dir"] / "delivery.json").read_text())
    bundle = json.loads(args["bundle_path"].read_text())
    bundle["context"]["source_identity"]["private_recording_version"] = "another recording"
    args["bundle_path"].write_text(json.dumps(skill.prepare_bundle(bundle["context"], args["policy_dir"], mode="combined")))
    args["output_dir"] = args["output_dir"].parent / "second-result"
    second = driver.run(**args, transport=transport)
    second_delivery = json.loads((args["output_dir"] / "delivery.json").read_text())
    assert not first["cache_hit"] and not second["cache_hit"] and len(requests) == 2
    assert first_delivery["fingerprint"] != second_delivery["fingerprint"]
    assert len(list((args["api_dir"] / "requests/combined").iterdir())) == 2
    assert json.loads((Path(first["output_dir"]) / "delivery.json").read_text()) == first_delivery


def test_schema_valid_but_semantically_invalid_candidate_is_preserved_as_rejected(inputs):
    args, candidate, _ = inputs
    candidate["aq_intervals"][0]["end_frame"] = 2
    with pytest.raises(ValueError, match="cover every source frame"):
        driver.run(**args, transport=httpx.MockTransport(
            lambda req: httpx.Response(200, json=provider(json.loads(req.content), candidate))))
    output = args["output_dir"]
    assert json.loads((output / "candidate.json").read_text()) == candidate
    assert json.loads((output / "delivery.json").read_text())["status"] == "semantic_validation_failed"
    assert not (output / "validation.json").exists()


@pytest.mark.parametrize("url", ["https://name:secret@example.com/v1", "https://example.com/v1?key=secret"])
def test_credentials_cannot_enter_endpoint_receipts(inputs, url):
    args, _, _ = inputs
    with pytest.raises(ValueError, match="without credentials"):
        driver.run(**args, base_url=url, transport=httpx.MockTransport(lambda _: pytest.fail("invalid endpoint")))
    assert not args["output_dir"].exists()


def test_private_outputs_cannot_be_written_to_raw_source(inputs):
    args, _, _ = inputs
    args["output_dir"] = args["bundle_path"].parent / "PRIVATE_FAILURE_LABEL_SOURCE" / "outputs"
    with pytest.raises(ValueError, match="outside code checkouts and raw source"):
        driver.run(**args, transport=httpx.MockTransport(lambda _: pytest.fail("invalid output path")))


def test_cli_help_exposes_explicit_provider_and_frozen_policy_options(capsys):
    with pytest.raises(SystemExit) as exc:
        driver.main(["--help"])
    assert exc.value.code == 0
    help_text = capsys.readouterr().out
    for flag in ("--policy-dir", "--model", "--base-url", "--api-dir", "--max-requests", "--retry-uncertain"):
        assert flag in help_text
