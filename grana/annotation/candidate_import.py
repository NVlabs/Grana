# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Import verified stage-first API candidates as unconfirmed, editable human drafts."""
from __future__ import annotations

import hashlib
from pathlib import Path

from grana.data.builder import load_build_config, validate_label_clock
from .labels import IMPORT_SCHEMA, episode_lock, label_path, new_labels, validate_labels, write_labels
from .source import LeRobotSource, same_episode
from .stage_settings import StageSettings


def display_candidate(bundle, candidate):
    """A display-only projection; never fabricate AQ or outcomes for a skipped round."""
    mode = bundle.get("mode", "combined")
    if mode == "combined":
        return candidate
    if mode == "stages":
        return {**candidate, "aq_intervals": [], "aq_boundaries": [], "outcomes": []}
    return {"stages": bundle["stage_lock"]["candidate"]["stages"], "outcomes": [],
            "aq_intervals": [row for stage in candidate["stage_annotations"] for row in stage["aq_intervals"]],
            "aq_boundaries": [row for stage in candidate["stage_annotations"] for row in stage["aq_boundaries"]]}


def _without_invalid(start, end, invalid):
    """Keep every valid piece; invalid evidence never becomes a known label."""
    pieces = [(start, end)]
    for span in invalid:
        low, high = span["start_frame"], span["end_frame"]
        pieces = [(a, b) for left, right in pieces
                  for a, b in ((left, min(right, low)), (max(left, high), right)) if a < b]
    return pieces


def _draft(candidate, context, episode, cfg):
    offset, end = context["source_identity"]["source_frame_range"]
    stage_labels = {stage["id"]: stage["label"] for stage in context["registry"]}
    doc = new_labels(episode.episode_index, episode.source_identity["trajectory_id"], episode.timestamps,
                     [stage["label"] for stage in cfg["stages"]], source_identity=episode.source_identity,
                     stage_registry=cfg["stages"])
    invalid, issues = candidate["stages"]["invalid_intervals"], []

    def issue(axis, row, value, *, native=False, boundary=False):
        shift = 0 if native else offset
        start_key, end_key = ("lower_frame", "upper_frame") if boundary else ("start_frame", "end_frame")
        issues.append({"axis": axis, "start_frame": row[start_key] + shift, "end_frame": row[end_key] + shift,
                       "value": value, "stage_id": row.get("stage_id"), "reason": row["reason"],
                       "evidence_frames": [frame + shift for frame in row.get("evidence_frames", [])],
                       "nominal_frame": row["frame"] + shift if boundary else None})

    for segment in candidate["stages"]["segments"]:
        if segment["identity_status"] != "resolved":
            issue("stage", segment, segment["identity_status"])
            continue
        for start, stop in _without_invalid(segment["start_frame"], segment["end_frame"], invalid):
            label = stage_labels[segment["stage_id"]]
            if doc["subtasks"] and doc["subtasks"][-1]["end_frame"] == start + offset and doc["subtasks"][-1]["stage"] == label:
                doc["subtasks"][-1]["end_frame"] = stop + offset
            else:
                doc["subtasks"].append({"start_frame": start + offset, "end_frame": stop + offset,
                                         "stage": label, "outcome": None})
    for outcome in candidate["outcomes"]:
        match = next((run for run in doc["subtasks"] if run["start_frame"] == outcome["start_frame"] + offset
                      and run["end_frame"] == outcome["end_frame"] + offset
                      and run["stage"] == stage_labels.get(outcome["stage_id"])), None)
        if match is not None and outcome["outcome"] in ("high", "low"):
            match["outcome"] = outcome["outcome"]
        else:
            original = outcome
            if outcome["outcome"] in ("high", "low"):
                outcome = {**outcome, "reason": "The original outcome applies to a stage run altered by invalid or uncertain frames; "
                           "choose an outcome for each remaining run. " + original["reason"]}
            issue("outcome", outcome, outcome["outcome"])
    for interval in candidate["aq_intervals"]:
        if interval["quality"] == "unknown":
            issue("quality", interval, "unknown")
        elif interval["quality"] in ("suboptimal", "failure"):
            for start, stop in _without_invalid(interval["start_frame"], interval["end_frame"], invalid):
                label = "regressed" if interval["quality"] == "failure" else "suboptimal"
                if doc["quality"] and doc["quality"][-1]["end_frame"] == start + offset and doc["quality"][-1]["label"] == label:
                    doc["quality"][-1]["end_frame"] = stop + offset
                else:
                    doc["quality"].append({"start_frame": start + offset, "end_frame": stop + offset, "label": label})
    for axis, boundaries in (("stage_boundary", candidate["stages"]["boundaries"]),
                             ("quality_boundary", candidate["aq_boundaries"])):
        for boundary in boundaries:
            if boundary["lower_frame"] < boundary["upper_frame"]:
                issue(axis, boundary, "uncertain", boundary=True)
    for span in invalid:
        issue("source", span, "invalid")
    for start, stop in ((0, offset), (end, episode.frame_count)):
        if start < stop:
            issue("unannotated", {"start_frame": start, "end_frame": stop,
                                   "reason": "Outside the imported clip; no stage, quality or outcome was proposed."},
                  "unannotated", native=True)
    issues.sort(key=lambda row: (row["start_frame"], row["end_frame"], row["axis"]))
    notes = ["Imported machine candidate: review the entire video, edit all needed labels, then explicitly confirm and submit. "
             "Unpainted quality is not confirmed nominal until that review; original unknowns and uncertainty are listed below."]
    notes.extend((f"Native boundary bracket [{row['start_frame']}, {row['end_frame']}] (inclusive)"
                  if row["axis"].endswith("_boundary") else f"Native frames [{row['start_frame']}, {row['end_frame']})")
                 + f": {row['axis']} {row['value']}"
                 + (f" (stage {row['stage_id']})" if row["stage_id"] is not None else "")
                 + f". {row['reason']}" for row in issues)
    if context["stage_semantics"] != "active_subtask":
        notes.append("The API used attained_state stage semantics. Verify that each imported interval represents the intended human subtask before submitting.")
    return doc, notes, issues


def _verify_delivery(bundle, candidate, delivery, prepared, validation, images, content, candidate_dir, policy_dir):
    """Verify local completion without making or reserving any API request."""
    from grana.annotation import api
    from scripts.annotation_skill import _read_json, stored_validation_receipt, validate_bundle_candidate
    from scripts.annotation_skill_api import digest

    actual = validate_bundle_candidate(bundle, candidate, policy_dir)
    if actual != stored_validation_receipt(validation, actual):
        raise ValueError("Validation receipt differs from this exact bundle and candidate")
    def image_contract(rows):
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            raise ValueError("Delivery images must be structured evidence records")
        return [{key: value for key, value in row.items() if key != "image_path"} for row in rows]
    if (not isinstance(delivery, dict) or delivery.get("schema_version") != "grana.annotation_delivery.v1"
            or delivery.get("status") != "completed" or delivery.get("accepted_for_training") is not False
            or delivery.get("bundle_sha256") != digest(bundle) or delivery.get("context_sha256") != bundle["context_sha256"]
            or delivery.get("candidate_sha256") != digest(candidate) or delivery.get("policy") != bundle["policy"]
            or delivery.get("source_identity_sent") is not False
            or delivery.get("frame_order") != sorted(bundle["context"]["evidence_frames"])
            or not isinstance(delivery.get("images"), list)
            or image_contract(delivery["images"]) != image_contract(images)):
        raise ValueError("Completed delivery does not bind the exact bundle, candidate and evidence")
    if not isinstance(prepared, dict) or any(delivery.get(key) != value for key, value in prepared.items()):
        raise ValueError("Prepared receipt differs from the completed delivery")
    required_prepared = {"schema_version", "bundle_sha256", "context_sha256", "policy", "images", "frame_order",
                         "source_identity_sent", "api", "request_name", "api_dir"}
    if not required_prepared <= prepared.keys():
        raise ValueError("Prepared receipt is incomplete")
    settings = delivery.get("api")
    if not isinstance(settings, dict) or any(name not in settings for name in ("model", "reasoning_effort", "max_output_tokens", "base_url")):
        raise ValueError("Delivery requires the original API request settings")
    payload, base_url = api._prepare(bundle["instructions"], content, bundle["schema"],
                                    **{name: settings[name] for name in ("model", "reasoning_effort", "max_output_tokens", "base_url")})
    fingerprint = api._digest(api._bytes({"base_url": base_url, "payload": payload}))
    if delivery.get("fingerprint") != fingerprint:
        raise ValueError("Saved response belongs to a different request; use its original inputs or generate a new result.")
    receipt_path = delivery.get("receipt")
    if not isinstance(receipt_path, str) or not receipt_path.strip():
        raise ValueError("Delivery requires its completed API receipt")
    receipt_path = Path(receipt_path).expanduser()
    if not receipt_path.is_absolute():
        receipt_path = candidate_dir / receipt_path
    receipt = _read_json(receipt_path)
    completed_path = receipt_path.parent / "completed.json"
    completed = _read_json(completed_path)
    if (not isinstance(receipt, dict) or receipt.get("status") != "completed" or receipt.get("fingerprint") != fingerprint
            or receipt.get("payload_sha256") != api._digest(api._bytes(payload))
            or receipt.get("completed_sha256") != hashlib.sha256(completed_path.read_bytes()).hexdigest()
            or not isinstance(completed, dict) or completed.get("data") != candidate):
        raise ValueError("Completed API receipt does not bind the imported candidate")
    try:
        checked = api._validated(completed["response"], bundle["schema"], settings["model"], settings["reasoning_effort"])
    except (KeyError, TypeError, api.AnnotationAPIError) as error:
        raise ValueError("Completed API response is invalid") from error
    if (checked != candidate or completed["response"].get("id") != delivery.get("response_id")
            or completed["response"].get("model") != delivery.get("model")):
        raise ValueError("Completed response differs from the candidate delivery")


def verified_draft(cfg, episode_index, bundle_path, candidate_dir, *, failure_quality=None, policy_dir=None):
    """Verify one completed combined candidate offline and build its draft for a recipe episode; write nothing.

    Returns the draft, with its machine_import record, and the episode it was checked against, for write_draft.
    """
    from scripts.annotation_skill import POLICY_DIR, _read_json
    from scripts.annotation_skill_api import digest, prepare_submission

    policy_dir = POLICY_DIR if policy_dir is None else Path(policy_dir)
    bundle_path, candidate_dir = Path(bundle_path).expanduser().resolve(), Path(candidate_dir).expanduser().resolve()
    delivery = _read_json(candidate_dir / "delivery.json")
    if not isinstance(delivery, dict):
        raise ValueError("Delivery must be a completed annotation receipt")
    template = delivery.get("template")
    if template is not None and (not isinstance(template, dict) or not isinstance(template.get("template_path"), str)):
        raise ValueError("Delivery template receipt is invalid")
    from .rounds import SCHEMA as ROUNDS_SCHEMA, verify_result
    if delivery.get("schema_version") == ROUNDS_SCHEMA:
        bundle, candidate, images, delivery = verify_result(bundle_path, candidate_dir, policy_dir)
    else:
        bundle, content, images, template_receipt = prepare_submission(
            bundle_path, bundle_path.parent / "evidence.json", policy_dir,
            template_path=template["template_path"] if template else None)
        if template_receipt != template:
            raise ValueError("Delivery template differs from the original human reference input")
        candidate = _read_json(candidate_dir / "candidate.json")
        _verify_delivery(bundle, candidate, delivery, _read_json(candidate_dir / "prepared.json"),
                         _read_json(candidate_dir / "validation.json"), images, content, candidate_dir, policy_dir)
    display = display_candidate(bundle, candidate)
    if any(row["quality"] == "failure" for row in display["aq_intervals"]) and failure_quality != "regressed":
        raise ValueError("Candidate contains failure AQ. Declare --failure-quality regressed only when that task policy means regression.")
    context = bundle["context"]
    if context["task"] != cfg["task"] or [{key: row[key] for key in ("label", "prompt")} for row in context["registry"]] != cfg["stages"]:
        raise ValueError("Candidate task and ordered stage labels/prompts must match the target recipe")
    reader = LeRobotSource(Path(cfg["source"]), cfg["views"], allow_shared_videos=True)
    episode = reader.load_episode(episode_index)
    validate_label_clock(episode, reader.info["fps"])
    identity = context["source_identity"]
    bounds = identity.get("source_frame_range")
    if (not isinstance(bounds, list) or len(bounds) != 2 or any(type(value) is not int for value in bounds)
            or not 0 <= bounds[0] < bounds[1] <= episode.frame_count
            or identity.get("frame_mapping") != "native_frame = candidate_frame + source_frame_range[0]"):
        raise ValueError("Bundle requires an explicit valid native source_frame_range and frame_mapping")
    start, end = bounds
    receipt = _read_json(bundle_path.parent / "source_receipt.json")
    sampling = receipt.get("sampling", {}) if isinstance(receipt, dict) else {}
    if (not same_episode(identity, episode.source_identity) or context["time_edges"] != list(episode.time_edges[start:end + 1])
            or any(row["view_names"] != list(episode.view_names) for row in images)
            or not isinstance(receipt, dict) or receipt.get("source_identity") != identity
            or receipt.get("clock") != episode.clock_receipt or receipt.get("annotation_columns_read") is not False
            or receipt.get("accepted_for_training") is not False
            or not isinstance(sampling, dict) or sampling.get("candidate_frames") != sorted(context["evidence_frames"])
            or sampling.get("grid_origin_seconds") != context["time_edges"][0]
            or sampling.get("source_frames") != [frame + start for frame in sorted(context["evidence_frames"])]):
        raise ValueError("Candidate source, cameras, full native clock or clip offset differs from the target episode")
    doc, notes, issues = _draft(display, context, episode, cfg)
    mode = bundle.get("mode", "combined")
    if mode == "stages":
        notes.insert(0, "AQ = default_normal_as_not_requested. Unmarked valid time defaults to Normal because AQ "
                     "was skipped, not because the model judged it normal. This is excluded from measured AQ accuracy.")
    if mode != "combined":
        notes.append("Outcomes were not requested; choose them separately if submitting full training labels.")
    doc["machine_import"] = {"schema_version": IMPORT_SCHEMA, "bundle_path": str(bundle_path), "candidate_dir": str(candidate_dir),
        "bundle_sha256": digest(bundle), "candidate_sha256": digest(candidate), "delivery_sha256": digest(delivery),
        "source_frame_range": bounds, "failure_quality": failure_quality, "model": delivery["model"],
        "response_id": delivery["response_id"], "task_policy_version": context["task_policy"]["version"],
        "stage_semantics": context["stage_semantics"], "review_notes": notes, "review_intervals": issues}
    if mode != "combined":
        doc["machine_import"].update(schema_version="grana.candidate_import.v2", annotation_mode=mode,
            action_quality_status="default_normal_as_not_requested" if mode == "stages" else "annotated", outcome_status="not_requested")
    validate_labels(doc, frame_count=episode.frame_count, stages=[stage["label"] for stage in cfg["stages"]])
    return doc, episode


def write_draft(cfg, settings, doc, episode):
    """Create the draft under the recipe and episode locks if its source is unchanged; never replace a label file or link."""
    path = label_path(cfg["labels"], episode.episode_index)
    with settings.writing(), episode_lock(cfg["labels"], episode.episode_index):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"Labels already exist at {path}; import never overwrites human work")
        refreshed = LeRobotSource(Path(cfg["source"]), cfg["views"], allow_shared_videos=True).load_episode(episode.episode_index)
        if not same_episode(episode.source_identity, refreshed.source_identity) or episode.time_edges != refreshed.time_edges:
            raise ValueError("Source changed during candidate import; no labels were written")
        write_labels(cfg["labels"], doc, new_only=True)
    return path


def import_candidate(config, episode_index, bundle_path, candidate_dir, *, failure_quality=None, policy_dir=None):
    """Write a new canonical draft; never replace saved labels or auto-accept predictions.

    The bundle's directory must retain evidence.json, source_receipt.json and its
    images. The candidate directory and original API completion receipts must
    remain available. This source-checkout helper performs no API calls.
    """
    cfg = load_build_config(config)
    if "labels" not in cfg:
        raise ValueError("Candidate import needs a human-label recipe with a separate labels directory")
    if type(episode_index) is not int or episode_index < 0:
        raise ValueError("episode must be a nonnegative integer")
    if cfg["episodes"] != "all" and episode_index not in cfg["episodes"]:
        raise ValueError("The target episode must be selected by the human-label recipe")
    if failure_quality not in (None, "regressed"):
        raise ValueError("failure_quality must be explicitly regressed or omitted")
    path = label_path(cfg["labels"], episode_index)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Labels already exist at {path}; import never overwrites human work. Use a fresh labels directory.")
    doc, episode = verified_draft(cfg, episode_index, bundle_path, candidate_dir, failure_quality=failure_quality,
                                  policy_dir=policy_dir)
    write_draft(cfg, StageSettings(cfg, config), doc, episode)
    imported = doc["machine_import"]
    return {"status": "imported_draft", "labels_file": str(path), "episode_index": episode_index,
            "done": False, "review_confirmed": False, "review_issues": len(imported["review_intervals"]),
            "source_frame_range": imported["source_frame_range"]}
