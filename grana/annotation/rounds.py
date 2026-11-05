# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded stage-first API campaigns with optional stage-local action quality.

Long recordings are split by native time before submission. AQ is requested only
after stages complete, in independent clips inside each frozen stage occurrence.
Aggregate receipts describe local composition; they never impersonate an API response.
"""
from __future__ import annotations

from bisect import bisect_left
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

from .api import KEY_ENV, provider_settings
from .schema import coalesce_candidate
from .source import LeRobotSource

SCHEMA = "grana.annotation_round_delivery.v1"
DEFAULTS = {"stage_seconds": 120.0, "stage_stride": 2.0,
            "quality_seconds": 10.0, "quality_stride": 0.2, "max_images": 96}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    from scripts.annotation_skill import _write_new
    path = Path(path)
    if path.exists():
        if read(path) != value:
            raise ValueError(f"Frozen campaign input differs: {path.name}; use a new output directory")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_new(path, value)


def split(clock, start, end, seconds, stride, max_images):
    """Bisect native intervals until both time and sampling budgets fit; never drop frames from coverage."""
    from scripts.prepare_annotation_evidence import select_frames
    for value in (seconds, stride):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError("Window duration and stride must be positive finite numbers")
    if type(max_images) is not int or max_images < 2:
        raise ValueError("max_images must be an integer of at least two")
    pending, result = [(start, end)], []
    while pending:
        a, b = pending.pop()
        fits = clock[b] - clock[a] <= seconds or b - a == 1
        if fits:
            try:
                select_frames(clock, a, b, stride, max_images)
            except ValueError:
                fits = False
        if fits:
            result.append((a, b))
        else:
            middle = max(a + 1, min(b - 1, bisect_left(clock, (clock[a] + clock[b]) / 2)))
            pending.extend([(middle, b), (a, middle)])
    return result


def shifted(row, offset):
    row = deepcopy(row)
    for key in ("start_frame", "end_frame", "frame", "lower_frame", "upper_frame"):
        if key in row:
            row[key] += offset
    if "evidence_frames" in row:
        row["evidence_frames"] = [f + offset for f in row["evidence_frames"]]
    return row


def merge_stages(parts, registry, count):
    """Join complete, verified disjoint cores and preserve uncertainty at request seams."""
    result = {"segments": [], "boundaries": [], "invalid_intervals": [], "omitted_stages": [],
              "summary": "Stage candidates composed from bounded requests; not accepted for training."}
    cursor = 0
    for start, end, bundle, candidate in parts:
        if start != cursor or end <= start or len(bundle["context"]["time_edges"]) - 1 != end - start:
            raise ValueError("Stage requests must partition the episode without gaps or overlaps")
        stages = candidate["stages"]
        if cursor:
            # Adjacent sampled endpoint frames bracket a seam; it is not an independently observed exact cut.
            result["boundaries"].append({"frame": start, "lower_frame": start - 1, "upper_frame": start,
                "reason": "Request seam between adjacent observed endpoint frames.", "evidence_frames": [start - 1, start]})
        for key in ("segments", "boundaries", "invalid_intervals"):
            result[key].extend(shifted(row, start) for row in stages[key])
        cursor = end
    if cursor != count:
        raise ValueError("Incomplete stage campaign")
    present = {s["stage_id"] for s in result["segments"]}
    result["omitted_stages"] = [{"stage_id": s["id"], "reason": "Not observed in any stage request."}
                                for s in registry if s["id"] not in present]
    return {"stages": coalesce_candidate(result, [s["id"] for s in registry], count)}


def project_stage(parent, candidate, context, index, start, end):
    """Exact local projection of one occurrence; request boundaries never become new task stages."""
    segment = candidate["stages"]["segments"][index]
    if not segment["start_frame"] <= start < end <= segment["end_frame"]:
        raise ValueError("AQ request crosses its frozen stage")
    root = parent["context"]
    for key in ("task", "task_instruction", "registry", "stage_semantics"):
        if context[key] != root[key]:
            raise ValueError(f"AQ projection changed {key}")
    if context["time_edges"] != root["time_edges"][start:end + 1]:
        raise ValueError("AQ projection changed the source clock")
    identity = deepcopy(root["source_identity"])
    offset = identity["source_frame_range"][0]
    identity["source_frame_range"] = [offset + start, offset + end]
    if context["source_identity"] != identity:
        raise ValueError("AQ projection changed the source identity")
    frames = context["evidence_frames"]
    if not frames:
        raise ValueError("AQ projection needs RGB evidence")
    # The stage decision comes from the verified parent, not another model call.
    local = {**segment, "start_frame": 0, "end_frame": end - start,
             "reason": "Frozen occurrence from the completed stage round. " + segment["reason"],
             "evidence_frames": [frames[0]]}
    invalid = [{**row, "start_frame": max(start, row["start_frame"]) - start,
                "end_frame": min(end, row["end_frame"]) - start}
               for row in candidate["stages"]["invalid_intervals"]
               if row["start_frame"] < end and row["end_frame"] > start]
    return {"stages": {"segments": [local], "boundaries": [], "invalid_intervals": invalid,
        "omitted_stages": [{"stage_id": s["id"], "reason": "Outside this frozen occurrence."}
                           for s in root["registry"] if s["id"] != segment["stage_id"]],
        "summary": "Local projection of one frozen stage occurrence."}}


def merge_quality(parent_candidate, parts):
    from scripts.annotation_skill_api import digest
    annotations = []
    for index, segment in enumerate(parent_candidate["stages"]["segments"]):
        intervals, boundaries, cursor = [], [], segment["start_frame"]
        for item, bundle, candidate in parts:
            if item["segment_index"] != index:
                continue
            a, b = item["start_frame"], item["end_frame"]
            if a != cursor:
                raise ValueError("Incomplete or overlapping AQ chunks")
            rows = candidate["stage_annotations"][0]
            if intervals:
                boundaries.append({"frame": a, "lower_frame": a - 1, "upper_frame": a,
                    "reason": "AQ request seam within the frozen stage; no independent change is asserted.",
                    "evidence_frames": [a - 1, a]})
            intervals.extend(shifted(row, a) for row in rows["aq_intervals"])
            boundaries.extend(shifted(row, a) for row in rows["aq_boundaries"])
            cursor = b
        if cursor != segment["end_frame"]:
            raise ValueError("AQ must complete every frozen stage occurrence")
        # Coalesce the same quality across chunk seams without fabricating an event.
        merged, removed = [], set()
        for row in intervals:
            if merged and merged[-1]["quality"] == row["quality"]:
                removed.add(row["start_frame"])
                merged[-1]["end_frame"] = row["end_frame"]
                merged[-1]["evidence_frames"] = sorted(set(merged[-1]["evidence_frames"] + row["evidence_frames"]))
                merged[-1]["reason"] += " | " + row["reason"]
            else:
                merged.append(row)
        annotations.append({"stage_segment_index": index, "aq_intervals": merged,
                            "aq_boundaries": [b for b in boundaries if b["frame"] not in removed]})
    return {"stage_candidate_sha256": digest(parent_candidate), "stage_annotations": annotations}


def _private(path, source):
    path = Path(path).expanduser().resolve()
    from grana._paths import PROJECT_ROOT
    if path.is_relative_to(Path(source).resolve()) or path.is_relative_to(PROJECT_ROOT) or any(
            (p / ".git").exists() for p in (path, *path.parents)):
        raise ValueError("Campaign output and receipts must be outside sources and code checkouts")
    return path


def run_episode(source, episode, settings, output, *, with_quality=False, policy_dir=None, template_path=None,
                api_dir=None, request_name="episode", model=None, base_url=None, api_key_env=KEY_ENV,
                reasoning_effort="xhigh", max_requests=24, max_attempts=1, timeout_seconds=480,
                max_output_tokens=16384, transport=None, **windows):
    """Stages are complete independently; opting into AQ later reuses them and the shared budget."""
    from scripts import annotation_skill as skill
    from scripts import annotation_skill_api as driver
    from scripts.prepare_annotation_evidence import prepare
    provider = provider_settings(model=model, base_url=base_url)
    model, base_url = provider["model"], provider["base_url"]
    if type(with_quality) is not bool:
        raise ValueError("with_quality must be a boolean")
    if windows.keys() - DEFAULTS.keys():
        raise ValueError("Unknown annotation window setting")
    limits = DEFAULTS | windows
    # Validate both modes' bounds before the first request.
    for mode in ("stage", "quality"):
        split([0.0, 1.0], 0, 1, limits[mode + "_seconds"], limits[mode + "_stride"], limits["max_images"])
    if template_path is not None:
        from .templates import _target, load_template, template_file
        policy_dir = policy_dir or template_file(template_path).parent / "policy"
        template, _ = load_template(template_path, policy_dir=policy_dir)
        bound, _, _ = _target(template, source, episode, task=settings["task"], views=settings["views"])
        if bound != settings:
            raise ValueError("Campaign settings differ from the frozen template")
    policy = Path(policy_dir or skill.POLICY_DIR).resolve()
    output = _private(output, source)
    api_dir = _private(api_dir or output / "api", source)
    reader = LeRobotSource(Path(source), settings["views"], allow_shared_videos=True)
    ep = reader.load_episode(episode)
    count = ep.frame_count
    cores = split(ep.time_edges, 0, count, limits["stage_seconds"], limits["stage_stride"], limits["max_images"])
    plan = {"schema_version": "grana.annotation_round_plan.v1", "source_identity": ep.source_identity,
            "settings": settings, "limits": limits, "stage_cores": [list(x) for x in cores],
            "policy_sha256": {str(p.relative_to(policy)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in sorted(policy.rglob('*')) if p.is_file()},
            "template_path": str(Path(template_path).resolve()) if template_path else None,
            "api": {"model": model, "base_url": base_url, "reasoning_effort": reasoning_effort,
                    "api_dir": str(api_dir), "request_name": request_name, "api_key_env": api_key_env,
                    "max_requests": max_requests, "max_attempts": max_attempts,
                    "timeout_seconds": timeout_seconds, "max_output_tokens": max_output_tokens}}
    save(output / "plan.json", plan)
    args = dict(policy_dir=policy, template_path=template_path, api_dir=api_dir, model=model, base_url=base_url,
                api_key_env=api_key_env, reasoning_effort=reasoning_effort, max_requests=max_requests,
                max_attempts=max_attempts, timeout_seconds=timeout_seconds, max_output_tokens=max_output_tokens,
                transport=transport)
    parts, images, stage_jobs = [], {}, []
    for number, (a, b) in enumerate(cores):
        folder = output / "requests" / f"stage-{number:05d}"
        evidence, result = folder / "evidence", folder / "candidate"
        if not evidence.exists():
            prepare(source, episode, settings, evidence, start_frame=a, end_frame=b,
                    stride_seconds=limits["stage_stride"], max_images=limits["max_images"], policy_dir=policy)
        driver.run(evidence / "bundle.json", evidence / "evidence.json", result,
                   request_name=f"{request_name}-stage-{number:05d}", **args)
        bundle, candidate = read(evidence / "bundle.json"), read(result / "candidate.json")
        parts.append((a, b, bundle, candidate))
        images.update({row["frame"] + a: {**row, "frame": row["frame"] + a}
                       for row in read(evidence / "evidence.json")["images"]})
        stage_jobs.append({"start_frame": a, "end_frame": b, "bundle": str(evidence / "bundle.json"),
                           "result": str(result), "delivery_sha256": driver.digest(read(result / "delivery.json"))})
    candidate = merge_stages(parts, settings["registry"], count)
    context = deepcopy(parts[0][2]["context"])
    context["source_identity"]["source_frame_range"] = [0, count]
    context.update(time_edges=list(ep.time_edges), evidence_frames=sorted(images))
    bundle = skill.prepare_bundle(context, policy)
    if (path := output / "evidence/bundle.json").exists():
        bundle = skill.resume_bundle(bundle, read(path), policy)
    _aggregate(output / "evidence", output / "candidate", bundle, candidate, images, ep, stage_jobs, plan, policy)
    result = {"status": "valid_candidate", "candidate_dir": str(output / "candidate"),
              "bundle_path": str(output / "evidence/bundle.json"), "action_quality_status": "default_normal_as_not_requested",
              "stage_requests": len(stage_jobs), "quality_requests": 0}
    if not with_quality:
        return result
    quality_jobs, quality_parts = [], []
    for index, segment in enumerate(candidate["stages"]["segments"]):
        intervals = split(ep.time_edges, segment["start_frame"], segment["end_frame"],
                          limits["quality_seconds"], limits["quality_stride"], limits["max_images"])
        for number, (a, b) in enumerate(intervals):
            folder = output / "requests" / f"quality-{index:05d}-{number:05d}"
            evidence, answer = folder / "evidence", folder / "candidate"
            if not evidence.exists():
                prepare(source, episode, settings, evidence, start_frame=a, end_frame=b,
                        stride_seconds=limits["quality_stride"], max_images=limits["max_images"], policy_dir=policy)
            local_bundle = read(evidence / "bundle.json")
            projected = project_stage(bundle, candidate, local_bundle["context"], index, a, b)
            quality_bundle = skill.prepare_quality_bundle(local_bundle, projected, policy)
            quality_path = folder / "quality-bundle.json"
            if quality_path.exists():
                quality_bundle = skill.resume_bundle(quality_bundle, read(quality_path), policy)
            save(quality_path, quality_bundle)
            driver.run(quality_path, evidence / "evidence.json", answer,
                       request_name=f"{request_name}-quality-{index:05d}-{number:05d}", **args)
            item = {"segment_index": index, "start_frame": a, "end_frame": b,
                    "bundle": str(quality_path), "evidence": str(evidence / "evidence.json"),
                    "result": str(answer), "delivery_sha256": driver.digest(read(answer / "delivery.json"))}
            quality_jobs.append(item)
            quality_parts.append((item, quality_bundle, read(answer / "candidate.json")))
            images.update({row["frame"] + a: {**row, "frame": row["frame"] + a}
                           for row in read(evidence / "evidence.json")["images"]})
    context = deepcopy(bundle["context"])
    context["evidence_frames"] = sorted(images)
    quality_bundle = skill.prepare_quality_bundle(bundle, candidate, policy, context=context)
    if (path := output / "quality/bundle.json").exists():
        quality_bundle = skill.resume_bundle(quality_bundle, read(path), policy)
    quality = merge_quality(candidate, quality_parts)
    parent = {"bundle": str(output / "evidence/bundle.json"), "result": str(output / "candidate"),
              "delivery_sha256": driver.digest(read(output / "candidate/delivery.json"))}
    _aggregate(output / "quality", output / "quality", quality_bundle, quality, images, ep,
               quality_jobs, plan, policy, parent=parent)
    return {**result, "candidate_dir": str(output / "quality"), "bundle_path": str(output / "quality/bundle.json"),
            "action_quality_status": "annotated", "quality_requests": len(quality_jobs)}


def _aggregate(evidence, output, bundle, candidate, images, ep, jobs, plan, policy, parent=None):
    from scripts.annotation_skill import validate_bundle_candidate
    from scripts.annotation_skill_api import digest
    validation = validate_bundle_candidate(bundle, candidate, policy)
    manifest = {"schema_version": "grana.annotation_evidence.v2", "context_sha256": bundle["context_sha256"],
                "images": [images[k] for k in sorted(images)]}
    save(evidence / "bundle.json", bundle)
    save(evidence / "evidence.json", manifest)
    save(evidence / "source_receipt.json", {"source_identity": bundle["context"]["source_identity"],
        "clock": ep.clock_receipt, "annotation_columns_read": False, "accepted_for_training": False,
        "sampling": {"candidate_frames": sorted(images), "source_frames": sorted(images),
                     "grid_origin_seconds": ep.time_edges[0], "method": "union_of_verified_request_samples"}})
    prepared = {"schema_version": SCHEMA, "bundle_sha256": digest(bundle), "context_sha256": bundle["context_sha256"],
        "annotation_mode": bundle["mode"], "plan": plan, "jobs": jobs, "parent": parent,
        "bundle_path": str(evidence / "bundle.json"), "accepted_for_training": False}
    save(output / "prepared.json", prepared)
    save(output / "candidate.json", candidate)
    save(output / "validation.json", validation)
    save(output / "delivery.json", {**prepared, "status": "completed", "candidate_sha256": digest(candidate),
         "images": manifest["images"], "model": plan["api"]["model"], "response_id": "local-composition:" + digest(jobs),
         "action_quality_status": validation["action_quality_status"], "outcome_status": validation["outcome_status"],
         "stage_delivery_sha256": parent["delivery_sha256"] if parent else None})


def verify_result(bundle_path, candidate_dir, policy_dir):
    """Rebuild the aggregate from individually verified API completions and exact stage projections."""
    from scripts import annotation_skill as skill
    from scripts import annotation_skill_api as driver
    from .candidate_import import _verify_delivery
    bundle, candidate = read(bundle_path), read(Path(candidate_dir) / "candidate.json")
    delivery = read(Path(candidate_dir) / "delivery.json")
    prepared, validation = (read(Path(candidate_dir) / (n + ".json")) for n in ("prepared", "validation"))
    skill.verify_bundle(bundle, policy_dir)
    if (delivery.get("schema_version") != SCHEMA or delivery.get("status") != "completed"
            or delivery.get("accepted_for_training") is not False
            or delivery.get("bundle_sha256") != driver.digest(bundle)
            or delivery.get("candidate_sha256") != driver.digest(candidate)
            or delivery.get("context_sha256") != bundle["context_sha256"]
            or any(delivery.get(k) != v for k, v in prepared.items())
            or (actual := skill.validate_bundle_candidate(bundle, candidate, policy_dir))
            != skill.stored_validation_receipt(validation, actual)):
        raise ValueError("Aggregate receipt differs from its exact bundle and candidate")
    if set(prepared) != {"schema_version", "bundle_sha256", "context_sha256", "annotation_mode", "plan", "jobs", "parent",
                         "bundle_path", "accepted_for_training"}:
        raise ValueError("Aggregate prepared receipt is incomplete")
    for axis in ("action_quality_status", "outcome_status"):
        if delivery.get(axis) != validation[axis]:
            raise ValueError("Aggregate annotation status differs from its mode")
    jobs = delivery.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("Aggregate needs completed API requests")
    parent_bundle = parent_candidate = None
    if bundle["mode"] == "quality":
        parent = delivery["parent"]
        if read(parent["bundle"]).get("mode") != "stages":
            raise ValueError("AQ parent must be a stage round")
        parent_bundle, parent_candidate, _, _ = verify_result(parent["bundle"], parent["result"], policy_dir)
        if (parent_bundle.get("mode") != "stages" or bundle["stage_lock"] != {"bundle": parent_bundle, "candidate": parent_candidate}
                or driver.digest(read(Path(parent["result"]) / "delivery.json")) != parent["delivery_sha256"]
                or read(Path(parent["result"]) / "delivery.json")["plan"] != delivery["plan"]):
            raise ValueError("Quality aggregate differs from its completed frozen stage round")
    parts, images = [], {}
    if parent_bundle:
        parent_manifest = read(Path(delivery["parent"]["bundle"]).parent / "evidence.json")
        images.update({row["frame"]: row for row in parent_manifest["images"]})
    for item in jobs:
        path, folder = Path(item["bundle"]), Path(item["result"])
        evidence = Path(item.get("evidence", path.parent / "evidence.json"))
        submitted = read(folder / "delivery.json")
        template = submitted.get("template")
        child, content, checked, reference = driver.prepare_submission(path, evidence, policy_dir,
            template["template_path"] if template else None)
        answer = read(folder / "candidate.json")
        if reference != template or driver.digest(submitted) != item["delivery_sha256"]:
            raise ValueError("Child completion or reference changed")
        for name, value in submitted["api"].items():
            if name in delivery["plan"]["api"] and delivery["plan"]["api"][name] != value:
                raise ValueError("Aggregate request settings differ from the completed child")
        if delivery["model"] != submitted["model"]:
            raise ValueError("Aggregate model differs from its completed requests")
        _verify_delivery(child, answer, submitted, read(folder / "prepared.json"), read(folder / "validation.json"),
                         checked, content, folder, policy_dir)
        a, b = item["start_frame"], item["end_frame"]
        if parent_bundle:
            projected = project_stage(parent_bundle, parent_candidate, child["context"], item["segment_index"], a, b)
            if child.get("mode") != "quality" or child["stage_lock"]["candidate"] != projected:
                raise ValueError("AQ chunk differs from its frozen occurrence")
            parts.append((item, child, answer))
        else:
            if child.get("mode") != "stages":
                raise ValueError("First-round requests must be stage-only")
            root = bundle["context"]
            expected = deepcopy(root["source_identity"])
            expected["source_frame_range"] = [a, b]
            if child["context"]["source_identity"] != expected or child["context"]["time_edges"] != root["time_edges"][a:b + 1]:
                raise ValueError("Stage request changed source or clock")
            for key in ("registry", "task", "task_instruction", "stage_semantics", "task_policy"):
                if child["context"][key] != root[key]:
                    raise ValueError("Stage request changed task policy")
            parts.append((a, b, child, answer))
        images.update({row["frame"] + a: {**row, "frame": row["frame"] + a}
                       for row in read(evidence)["images"]})
    expected = merge_quality(parent_candidate, parts) if parent_bundle else merge_stages(
        parts, bundle["context"]["registry"], len(bundle["context"]["time_edges"]) - 1)
    manifest = read(Path(bundle_path).parent / "evidence.json")
    if (expected != candidate or manifest != {"schema_version": "grana.annotation_evidence.v2",
            "context_sha256": bundle["context_sha256"], "images": [images[k] for k in sorted(images)]}
            or bundle["context"]["evidence_frames"] != sorted(images) or delivery["images"] != manifest["images"]):
        raise ValueError("Aggregate differs from its verified requests")
    return bundle, candidate, manifest["images"], delivery
