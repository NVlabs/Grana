#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Submit a frozen annotation bundle and verified mosaics through Responses."""
from __future__ import annotations

import argparse
import base64
from copy import deepcopy
import hashlib
from io import BytesIO
import json
import math
import re
from pathlib import Path
import sys
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image

from grana.annotation.workflow import write_json
from grana.annotation.api import AnnotationAPIError, KEY_ENV, provider_settings, request_json
from scripts.annotation_skill import POLICY_DIR, _read_json, verify_bundle, validate_bundle_candidate

EVIDENCE_SCHEMA = "grana.annotation_evidence.v1"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def prepare_request(bundle_path, evidence_path, policy_dir=POLICY_DIR):
    """Verify exact input bytes locally; source paths and identity stay off wire."""
    bundle = _read_json(bundle_path)
    try:
        verify_bundle(bundle, policy_dir)
    except ValueError as error:
        raise ValueError(f"Frozen bundle differs from the supplied policy files or context: {error}") from error
    evidence_path = Path(evidence_path).expanduser().resolve()
    manifest = _read_json(evidence_path)
    version = "grana.annotation_evidence.v2" if bundle.get("mode") else EVIDENCE_SCHEMA
    fields = {"schema_version", "images"} | ({"context_sha256"} if bundle.get("mode") else set())
    if (not isinstance(manifest, dict) or set(manifest) != fields
            or manifest["schema_version"] != version or not isinstance(manifest["images"], list)):
        raise ValueError("Evidence format does not match this annotation. Prepare it again with scripts/prepare_annotation_evidence.py.")
    if bundle.get("mode") and manifest["context_sha256"] != bundle["context_sha256"]:
        raise ValueError("Evidence context differs from the frozen source, clock or task context")
    context = bundle["context"]
    available = set(context["evidence_frames"])
    seen, views, verified = set(), None, []
    for row in manifest["images"]:
        if not isinstance(row, dict) or set(row) != {"frame", "image_path", "sha256", "view_names", "timestamp"}:
            raise ValueError("Evidence image metadata is incomplete. Prepare the evidence again with scripts/prepare_annotation_evidence.py.")
        frame = row["frame"]
        if type(frame) is not int or frame not in available or frame in seen:
            raise ValueError("Evidence must contain exactly the distinct declared frame IDs")
        seen.add(frame)
        timestamp = row["timestamp"]
        if (isinstance(timestamp, bool) or not isinstance(timestamp, (int, float))
                or not math.isfinite(timestamp) or timestamp != context["time_edges"][frame]):
            raise ValueError("Evidence timestamp differs from the declared native frame clock")
        names = row["view_names"]
        if (not isinstance(names, list) or not names
                or any(not isinstance(name, str) or not name.strip() or name != name.strip() for name in names)
                or len(set(names)) != len(names) or views is not None and names != views):
            raise ValueError("Evidence requires the same nonempty ordered unique view_names at every frame")
        views = names
        if not isinstance(row["image_path"], str) or not row["image_path"].strip():
            raise ValueError("Evidence image_path must be nonempty")
        path = Path(row["image_path"]).expanduser()
        path = (evidence_path.parent / path).resolve()
        raw = path.read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        if row["sha256"] != sha:
            raise ValueError(f"Evidence image changed at frame {frame}; prepare the evidence again.")
        try:
            with Image.open(BytesIO(raw), formats=("PNG", "JPEG", "WEBP")) as image:
                if image.format not in ("PNG", "JPEG", "WEBP") or image.mode != "RGB":
                    raise ValueError("Evidence mosaics must be RGB PNG/JPEG/WebP images")
                mime = {"PNG": "png", "JPEG": "jpeg", "WEBP": "webp"}[image.format]
                size = list(image.size)
                image.verify()
            with Image.open(BytesIO(raw), formats=("PNG", "JPEG", "WEBP")) as image:
                image.load()
        except (OSError, SyntaxError) as error:
            raise ValueError(f"Evidence image cannot be decoded at frame {frame}") from error
        verified.append(({"frame": frame, "timestamp": timestamp, "view_names": names,
                          "sha256": sha, "image_path": str(path), "image_size": size}, raw, mime))
    if seen != available:
        raise ValueError("Evidence must contain exactly the declared frame set, with no missing or extra frames")
    verified.sort(key=lambda entry: entry[0]["frame"])
    wire_context = deepcopy(context)
    del wire_context["source_identity"]
    wire_context["input_id"] = bundle["context_sha256"]
    if bundle.get("mode"):
        wire_context["annotation_mode"] = bundle["mode"]
    if bundle.get("mode") == "quality":
        # Parent provenance includes private paths and stays local. Send only
        # the stage decision and its binding, never the embedded parent bundle.
        wire_context["frozen_stages"] = bundle["stage_lock"]["candidate"]["stages"]
        wire_context["stage_candidate_sha256"] = digest(bundle["stage_lock"]["candidate"])
    content = [{"type": "input_text", "text": (
        "Annotate only this declared clip frame range [0,N); it need not be a complete physical trajectory. "
        "Only the supplied sampled frames are visible. View names describe the supplied mosaics; "
        "do not assume unseen events or completion beyond this evidence.\n" + canonical(wire_context))}]
    for row, raw, mime in verified:
        content.extend([
            {"type": "input_text", "text": canonical({k: row[k] for k in ("frame", "timestamp", "view_names")})},
            {"type": "input_image", "detail": "high",
             "image_url": f"data:image/{mime};base64," + base64.b64encode(raw).decode("ascii")}])
    return bundle, content, [row for row, _, _ in verified]


def prepare_submission(bundle_path, evidence_path, policy_dir=POLICY_DIR, template_path=None):
    """Verify optional human examples separately from the target recording."""
    bundle, content, images = prepare_request(bundle_path, evidence_path, policy_dir)
    template = None
    if template_path is not None:
        # The template helper also verifies its seed evidence with prepare_request.
        from grana.annotation.templates import prepare_template_request

        references, template = prepare_template_request(template_path, bundle, policy_dir=policy_dir)
        if any(row["view_names"] != images[0]["view_names"] for seed in template["seeds"] for row in seed["seed_images"]):
            raise ValueError("Target view names/order differ from the human template; "
                             "prepare target evidence with the template's named views")
        content = [{"type": "input_text", "text": (
            "REFERENCE EXAMPLES ONLY: the following human annotations and images demonstrate the declared "
            "task policy. They are not target observations. Do not copy their frame IDs, timing, labels or "
            "outcomes onto the target. Template provenance binding: " + template["template_sha256"])},
            *references,
            {"type": "input_text", "text": (
                "END REFERENCE EXAMPLES. TARGET RECORDING FOLLOWS. Return annotations only for this "
                "target, citing only the target's supplied frame IDs. Keep unsupported target evidence unknown.")},
            *content]
    return bundle, content, images, template



def input_roots(bundle, template=None):
    """Private outputs must not change either raw source or an exported template."""
    identities = [bundle["context"]["source_identity"]]
    roots = []
    if template is not None:
        identities.extend(seed["source_identity"] for seed in template["seeds"])
        roots.append(Path(template["template_path"]).expanduser().resolve().parent)
    for identity in identities:
        if isinstance(identity.get("source_root"), str):
            roots.append(Path(identity["source_root"]).expanduser().resolve())
    return roots



def run(bundle_path, evidence_path, output_dir, *, policy_dir=POLICY_DIR, api_dir=None,
        request_name=None, model=None, base_url=None, reasoning_effort="xhigh",
        max_output_tokens=32768, timeout_seconds=480, max_attempts=2, max_requests=24,
        retry_uncertain=False, template_path=None, api_key_env=KEY_ENV, transport=None):
    provider = provider_settings(model=model, base_url=base_url)
    model, base_url = provider["model"], provider["base_url"]
    bundle, content, images, template = prepare_submission(bundle_path, evidence_path, policy_dir, template_path)
    request_name = bundle.get("mode", "combined") if request_name is None else request_name
    output = Path(output_dir).expanduser().resolve()
    protected = input_roots(bundle, template)
    if (output.is_relative_to(ROOT) or any((p / ".git").exists() for p in (output, *output.parents))
            or any(output.is_relative_to(root) for root in protected)):
        raise ValueError("Private annotation outputs must be outside code checkouts and raw source")
    api_dir = Path(api_dir).expanduser().resolve() if api_dir is not None else output / "api"
    if (api_dir.is_relative_to(ROOT) or any((p / ".git").exists() for p in (api_dir, *api_dir.parents))
            or any(api_dir.is_relative_to(root) for root in protected)):
        raise ValueError("Private API receipts must be outside code checkouts and raw source")
    url = urlsplit(base_url)
    if url.scheme != "https" or not url.hostname or url.username or url.password or url.query or url.fragment:
        raise ValueError("base_url must be HTTPS without credentials, query or fragment")
    if not isinstance(api_key_env, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,127}", api_key_env):
        raise ValueError("api_key_env must name an environment variable")
    settings = {"model": model, "base_url": base_url, "reasoning_effort": reasoning_effort,
                "max_output_tokens": max_output_tokens, "timeout_seconds": timeout_seconds,
                "max_attempts": max_attempts, "max_requests": max_requests}
    if api_key_env != KEY_ENV:
        settings["api_key_env"] = api_key_env
    prepared = {"schema_version": "grana.annotation_delivery.v1", "bundle_sha256": digest(bundle),
                "context_sha256": bundle["context_sha256"], "policy": bundle["policy"],
                "images": images, "frame_order": [row["frame"] for row in images],
                "source_identity_sent": False, "api": settings, "request_name": request_name,
                "api_dir": str(api_dir)}
    if template is not None:
        prepared["template"] = template
    write_json(output / "prepared.json", prepared)
    result = request_json(api_dir, request_name, instructions=bundle["instructions"], content=content,
                          schema=bundle["schema"], retry_uncertain=retry_uncertain, transport=transport, **settings)
    write_json(output / "candidate.json", result["data"])
    delivery = {**prepared, "status": "completed", "candidate_sha256": digest(result["data"]),
                "response_id": result["response_id"], "fingerprint": result["fingerprint"],
                "receipt": result["receipt"], "usage": result["usage"], "model": result["model"],
                "elapsed_seconds": result["elapsed_seconds"], "accepted_for_training": False,
                "delivery_scope": "Exact decoded, hash-checked mosaic bytes submitted; provider visual consumption and individual camera contents are not independently verified"}
    try:
        validation = validate_bundle_candidate(bundle, result["data"], policy_dir)
    except (ValueError, TypeError) as error:
        write_json(output / "delivery.json", delivery | {"status": "semantic_validation_failed", "validation_error": str(error)})
        raise
    write_json(output / "validation.json", validation)
    if bundle.get("mode"):
        delivery.update({key: validation[key] for key in ("annotation_mode", "action_quality_status", "outcome_status")})
    write_json(output / "delivery.json", delivery)
    return {"status": "valid_candidate", "output_dir": str(output), "response_id": result["response_id"],
            "cache_hit": result["cache_hit"], "image_count": len(images), "accepted_for_training": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "evidence", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--policy-dir", type=Path, default=POLICY_DIR)
    parser.add_argument("--api-dir", type=Path, help="Optional shared campaign directory with a non-increasing HTTP-attempt budget")
    parser.add_argument("--request-name", help="Defaults to the bundle mode: stages, quality or legacy combined")
    parser.add_argument("--template", dest="template_path", type=Path)
    parser.add_argument("--api-key-env", default=KEY_ENV)
    parser.add_argument("--model", help="Provider model name; otherwise GRANA_ANNOTATION_MODEL")
    parser.add_argument("--base-url", help="Responses API base URL; otherwise GRANA_ANNOTATION_BASE_URL")
    parser.add_argument("--reasoning-effort", choices=("high", "xhigh"), default="xhigh")
    parser.add_argument("--max-output-tokens", type=int, default=32768)
    parser.add_argument("--timeout-seconds", type=float, default=480)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--max-requests", type=int, default=24)
    parser.add_argument("--retry-uncertain", action="store_true", help="Explicitly accept possible duplicate billing after uncertain transport")
    args = vars(parser.parse_args(argv))
    args["bundle_path"], args["evidence_path"] = args.pop("bundle"), args.pop("evidence")
    try:
        print(json.dumps(run(**args), ensure_ascii=False, allow_nan=False))
    except (AnnotationAPIError, ValueError, TypeError, OSError) as error:
        parser.exit(2, f"annotation_skill_api: {error}\n")


if __name__ == "__main__":
    main()
