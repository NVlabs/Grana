# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private, portable human examples for same-task annotation; never learned rules.

The source checkout supplies the policy/bundle helpers. A template snapshots 1 to
5 submitted canonical label documents, the seeds, and their actual sampled RGB
evidence within one image budget. Hashes detect accidental changes, not the
authenticity of a self-declared reviewer.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from types import SimpleNamespace

from grana._paths import PROJECT_ROOT
from grana.annotation.labels import is_legacy, label_path, validate_binding, validate_labels
from grana.annotation.source import LeRobotSource
from grana.data.builder import load_build_config

SCHEMA = "grana.annotation_template.v2"
TARGETS_SCHEMA = "grana.annotation_targets.v1"
MAX_SEEDS = 5
MAPPING = {
    "human_to_api_quality": {"nominal": "nominal", "suboptimal": "suboptimal", "regressed": "failure"},
    "api_to_human_quality": {"nominal": "nominal", "suboptimal": "suboptimal", "failure": "regressed"},
    "human_to_api_outcome": {"high": "high", "low": "low"},
    "unknown_policy": "Uncertain target stages, quality and outcomes remain unknown and require review; never fill nominal.",
    "scope": "This template explicitly maps its human regressed class to wire failure; this is not a global mapping of historical failure labels.",
}
TASK_FIELDS = ("task", "task_instruction", "stage_semantics", "registry", "task_policy")


class Refusal(ValueError):
    """A refusal a page can word in its own language: ``code`` names it and ``values`` fill it in."""

    def __init__(self, code, message, **values):
        super().__init__(message)
        self.code, self.values = code, values


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def _text(value, name):
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{name} must be nonempty trimmed text")
    return value


def _private(path, sources=()):
    path = Path(path).expanduser().resolve()
    if (path.is_relative_to(PROJECT_ROOT)
            or any((parent / ".git").exists() for parent in (path, *path.parents))
            or any(path.is_relative_to(Path(source).expanduser().resolve()) for source in sources)):
        raise ValueError("Private template/campaign outputs must be outside code checkouts and raw sources")
    return path


def _file(root, relative):
    if not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError("Template files must use relative paths inside the template")
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("Template payload is missing or escapes its directory")
    return path


def template_file(path):
    path = Path(path).expanduser().resolve()
    return path / "template.json" if path.is_dir() else path


def _recording_key(identity):
    """Detect byte-identical recordings even when their directory/index is renamed."""
    videos = identity.get("video_sha256", identity.get("video_packets_sha256"))
    clock = identity.get("timestamps_sha256")
    if not isinstance(videos, dict) or not videos or not isinstance(clock, str):
        raise ValueError("Template/target requires verified video and native-clock source identity")
    return digest({"clock": clock, "videos": sorted(videos.values())})


def _source_tasks(reader, episode):
    row = reader.episodes[episode]
    values = row.get("tasks", row.get("task", []))
    if isinstance(values, str):
        values = [values]
    if values is None:
        values = []
    if not isinstance(values, list) or any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError("Source episode task metadata must be text or a list of task strings")
    return sorted(set(values))


def _reviewed_labels(doc, episode, stages):
    if is_legacy(doc):
        raise Refusal("legacy_seed", "Legacy 11-key labels cannot be seeds. Open, review and submit the episode in grana label first.")
    validate_labels(doc, frame_count=episode.frame_count, stages=[row["label"] for row in stages], require_done=True)
    try:
        validate_binding(doc, episode, stages)
    except ValueError as error:
        raise Refusal("unbound_seed", str(error)) from None
    if not doc["review_confirmed"] or not doc["revision"] or not doc["updated_at"] or not doc["annotator"].strip():
        raise Refusal("unreviewed_seed", "A seed must be saved, submitted and explicitly review_confirmed with an annotator")


def _seed_sampling(clocks, boundaries, stride_seconds, max_images):
    """Each seed's (stride_seconds, max_images) within one total image budget, checked before anything is written.

    Every seed keeps its endpoints and the frames next to each of its human boundaries. The rest of the budget is
    split evenly over the seeds' time grids at ``stride_seconds``: a seed whose grid needs less passes the rest on, and
    a seed whose grid needs more than its share is sampled with a wider stride that fits it.
    """
    from scripts.prepare_annotation_evidence import select_frames

    if isinstance(max_images, bool) or not isinstance(max_images, int) or max_images < 1:
        raise ValueError("max_images must be a positive integer")
    kept, needs = [], []
    for edges, extras in zip(clocks, boundaries, strict=True):
        count = len(edges) - 1
        kept.append(len({0, count - 1, *extras}))
        needs.append(len(select_frames(edges, 0, count, stride_seconds, count, extra_frames=extras)) - kept[-1])
    if sum(kept) > max_images:
        raise Refusal("image_budget", f"The seeds' boundary frames need {sum(kept)} images, more than max_images "
                      f"{max_images}; choose fewer seeds", needed=sum(kept), budget=max_images)
    left, shares = max_images - sum(kept), [0] * len(needs)
    for position, index in enumerate(sorted(range(len(needs)), key=needs.__getitem__)):
        shares[index] = min(needs[index], left // (len(needs) - position))
        left -= shares[index]
    sampling = []
    for edges, extras, required, need, share in zip(clocks, boundaries, kept, needs, shares):
        stride = stride_seconds
        if need > share:   # share + 1 ticks, the last on the final frame: at most ``share`` frames between the endpoints
            stride = (edges[-2] - edges[0]) / (share + 1)
        select_frames(edges, 0, len(edges) - 1, stride, required + share, extra_frames=extras)
        sampling.append((stride, required + share))
    return sampling


def export_template(config_path, episodes, output, *, criteria=None, stride_seconds=2.0,
                    max_images=64, tile_size=(320, 240), policy_dir=None):
    """Export 1 to 5 reviewed human recordings, the seeds, as one immutable few-shot reference.

    ``max_images`` is the total over all seeds; _seed_sampling shares it out.
    """
    from scripts.annotation_skill import POLICY_DIR, POLICY_FILES
    POLICY_FILES = (*POLICY_FILES, "references/stage-annotation.md")
    from scripts.prepare_annotation_evidence import prepare

    if (not isinstance(episodes, (list, tuple)) or not 1 <= len(episodes) <= MAX_SEEDS
            or any(type(episode) is not int or episode < 0 for episode in episodes) or len(set(episodes)) != len(episodes)):
        raise Refusal("seed_count", f"Seeds must be 1 to {MAX_SEEDS} distinct episode indices", max=MAX_SEEDS)
    cfg = load_build_config(config_path)
    if "labels" not in cfg:
        raise ValueError("Template export requires a human-label recipe with a labels directory")
    reader = LeRobotSource(Path(cfg["source"]), cfg["views"], allow_shared_videos=True)
    seeds, recordings = [], set()
    for episode in episodes:
        if cfg["episodes"] != "all" and episode not in cfg["episodes"]:
            raise Refusal("unselected_seed", f"Seed episode {episode} is not selected by the active recipe", episode=episode)
        ep = reader.load_episode(episode)
        labels_path = label_path(cfg["labels"], episode)
        raw_labels = labels_path.read_bytes()
        try:
            doc = json.loads(raw_labels)
            _reviewed_labels(doc, ep, cfg["stages"])
            if _recording_key(ep.source_identity) in recordings:
                raise Refusal("duplicate_seed", "it is the same recording as another seed")
        except ValueError as error:   # the page words each by its code, naming the seed
            raise Refusal(getattr(error, "code", "unreviewed_seed"), f"Seed episode {episode}: {error}",
                          episode=episode) from None
        recordings.add(_recording_key(ep.source_identity))
        extras = set()
        for span in doc["subtasks"] + doc["quality"]:
            for frame in (span["start_frame"] - 1, span["start_frame"], span["end_frame"] - 1, span["end_frame"]):
                if 0 <= frame < ep.frame_count:
                    extras.add(frame)
        seeds.append({"episode": ep, "labels_path": labels_path, "raw": raw_labels, "doc": doc, "extras": sorted(extras),
                      "tasks": _source_tasks(reader, episode)})
    sampling = _seed_sampling([seed["episode"].time_edges for seed in seeds], [seed["extras"] for seed in seeds],
                              stride_seconds, max_images)
    criteria = {"version": "human-exemplars-v1", "rules": "Use the declared stage objectives and the reviewed human examples. No additional task-specific visual rule has been inferred or calibrated."} if criteria is None else deepcopy(criteria)
    if not isinstance(criteria, dict) or set(criteria) != {"version", "rules"}:
        raise ValueError("Criteria must contain exactly version and rules")
    for key in criteria:
        _text(criteria[key], f"criteria.{key}")
    registry = [{"id": row["label"], **row} for row in cfg["stages"]]
    rules = ("Human-reference template: active subtask identity follows the declared objective, not a fixed sequence. "
             "The supplied seeds are separate reviewed demonstrations; never copy their frame boundaries, durations, outcomes or action-quality spans onto the target. "
             "Unmarked seed quality is nominal only because each whole seed was explicitly reviewed. "
             "Human examples do not prove a general visual criterion or target success. "
             "Keep unsupported target judgments unknown. No learned rules or measured calibration are claimed.\n"
             + criteria["rules"] + "\nExplicit human/API label mapping:\n" + canonical(MAPPING))
    version = digest({"task": cfg["task"], "instruction": cfg["compressed_task_prompt"], "registry": registry,
                      "label_sha256": [hashlib.sha256(seed["raw"]).hexdigest() for seed in seeds],
                      "criteria": criteria, "mapping": MAPPING})
    settings = {"task": cfg["task"], "task_instruction": cfg["compressed_task_prompt"], "views": cfg["views"],
                "stage_semantics": "active_subtask", "registry": registry,
                "task_policy": {"version": f"human-exemplars-v1:{version[:16]}", "status": "draft", "rules": rules}}
    policy = Path(policy_dir or POLICY_DIR).expanduser().resolve()
    policies = {name: (policy / name).read_bytes() for name in POLICY_FILES}
    output = _private(output, (cfg["source"],))
    output.mkdir(parents=True, exist_ok=False)
    for name, raw in policies.items():
        target = output / "policy" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    _write(output / "recipe.json", cfg)
    listed, provenance, images = [], [], 0
    for number, (seed, (stride, cap)) in enumerate(zip(seeds, sampling), 1):
        ep, folder = seed["episode"], output / "seeds" / str(number)
        prepare(cfg["source"], ep.episode_index, settings, folder, stride_seconds=stride, max_images=cap,
                tile_size=tile_size, policy_dir=output / "policy", extra_frames=seed["extras"])
        (folder / "labels.json").write_bytes(seed["raw"])
        evidence = json.loads((folder / "evidence.json").read_text())
        for row in evidence["images"]:
            row["image_path"] = str(Path(row["image_path"]).relative_to(folder))
        (folder / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
        images += len(evidence["images"])
        listed.append({"episode": ep.episode_index, "folder": str(folder.relative_to(output))})
        backups = [{"name": path.name, "sha256": _sha(path)} for path in sorted(
            (Path(cfg["labels"]) / ".v1-backup").glob(f"{seed['labels_path'].stem}.*.json"))]
        provenance.append({"episode": ep.episode_index, "original_labels": str(seed["labels_path"].resolve()),
                           "source_identity": ep.source_identity, "legacy_backups": backups,
                           "machine_import": seed["doc"].get("machine_import")})
    files = {str(path.relative_to(output)): _sha(path) for path in sorted(output.rglob("*")) if path.is_file()}
    template = {"schema_version": SCHEMA, "created_at": datetime.now(timezone.utc).isoformat(),
                "settings": settings, "label_mapping": deepcopy(MAPPING), "criteria": criteria,
                "source_tasks": sorted({task for seed in seeds for task in seed["tasks"]}), "seeds": listed, "files": files,
                "provenance": {"original_recipe": str(Path(config_path).expanduser().resolve()), "seeds": provenance,
                               "sampling": {"stride_seconds": stride_seconds, "max_images": max_images,
                                            "tile_size": list(tile_size),
                                            "rule": "Each seed keeps its endpoints and boundary-adjacent frames; the rest of "
                                                    "max_images is split evenly over the seeds' grids, a wider stride where "
                                                    "a grid exceeds its share."},
                               "review_status": "submitted_and_review_confirmed",
                               "review_identity": "Self-declared human review; not authenticated or independently verified.",
                               "method": "Few-shot human exemplars with declared criteria; no inferred learned policy."}}
    template["template_id"] = digest(template)
    _write(output / "template.json", template)
    load_template(output)
    return {"status": "exported", "template": str(output / "template.json"), "template_id": template["template_id"],
            "task": cfg["task"], "seeds": list(episodes), "seed_images": images, "policy_status": "draft"}


def load_template(path, *, policy_dir=None):
    """Validate all frozen payloads without requiring the original raw source.

    Returns the template and its seeds in order, each a dict of its bundle, labels, request content and images.
    """
    from scripts.annotation_skill import POLICY_FILES
    POLICY_FILES = (*POLICY_FILES, "references/stage-annotation.md")
    from scripts.annotation_skill_api import prepare_request

    path = template_file(path)
    root = path.parent
    template = json.loads(path.read_text())
    required = {"schema_version", "created_at", "settings", "label_mapping", "criteria", "source_tasks", "seeds", "files",
                "provenance", "template_id"}
    if isinstance(template, dict) and template.get("schema_version") == "grana.annotation_template.v1":
        raise ValueError("This one-seed template predates templates of several seeds; export it again")
    if not isinstance(template, dict) or set(template) != required or template["schema_version"] != SCHEMA:
        raise ValueError(f"Unsupported annotation template; export a fresh {SCHEMA} directory")
    if template["template_id"] != digest({key: value for key, value in template.items() if key != "template_id"}):
        raise ValueError("Template content hash mismatch")
    listed, provenance = template["seeds"], template["provenance"]
    if (not isinstance(listed, list) or not 1 <= len(listed) <= MAX_SEEDS
            or any(not isinstance(seed, dict) or set(seed) != {"episode", "folder"} for seed in listed)
            or [seed["folder"] for seed in listed] != [f"seeds/{number}" for number in range(1, len(listed) + 1)]
            or not isinstance(provenance, dict) or not isinstance(provenance.get("seeds"), list)
            or [row.get("episode") if isinstance(row, dict) else None for row in provenance["seeds"]]
            != [seed["episode"] for seed in listed]):
        raise ValueError(f"Template must list 1 to {MAX_SEEDS} seeds in seeds/1.. with their provenance")
    files = template["files"]
    mandatory = {"recipe.json", *(f"policy/{name}" for name in POLICY_FILES),
                 *(f"{seed['folder']}/{name}" for seed in listed
                   for name in ("labels.json", "bundle.json", "evidence.json", "source_receipt.json"))}
    if not isinstance(files, dict) or not mandatory <= files.keys():
        raise ValueError("Template must include recipe, labels, source evidence and frozen policy")
    for relative, expected in files.items():
        if _sha(_file(root, relative)) != expected:
            raise ValueError(f"Template payload hash mismatch: {relative}")
    settings = template["settings"]
    if not isinstance(settings, dict) or set(settings) != {*TASK_FIELDS, "views"}:
        raise ValueError("Template settings are invalid")
    if template["label_mapping"] != MAPPING:
        raise ValueError("Template must declare the supported human/API quality and outcome mapping")
    if any(row["id"] != row["label"] for row in settings["registry"]):
        raise ValueError("Template IDs must preserve the stable human stage labels")
    stages = [{key: row[key] for key in ("label", "prompt")} for row in settings["registry"]]
    recipe = json.loads((root / "recipe.json").read_text())
    if (recipe.get("task") != settings["task"] or recipe.get("compressed_task_prompt") != settings["task_instruction"]
            or recipe.get("stages") != stages or recipe.get("views") != settings["views"]):
        raise ValueError("Template task/stages/views differ from its source recipe")
    if (not isinstance(template["source_tasks"], list)
            or any(not isinstance(value, str) or not value for value in template["source_tasks"])):
        raise ValueError("Template source task aliases must be explicit strings")
    if provenance.get("review_status") != "submitted_and_review_confirmed":
        raise ValueError("Template provenance differs from its reviewed seed source")
    policy = Path(policy_dir).expanduser().resolve() if policy_dir is not None else root / "policy"
    seeds, recordings = [], set()
    for seed, origin in zip(listed, provenance["seeds"], strict=True):
        folder = root / seed["folder"]
        bundle, content, images = prepare_request(folder / "bundle.json", folder / "evidence.json", policy)
        if any(settings[key] != bundle["context"][key] for key in TASK_FIELDS):
            raise ValueError("Template settings differ from the frozen seed bundle")
        if any(row["view_names"] != list(settings["views"]) for row in images):
            raise ValueError("Template views differ from the seed evidence")
        for image in images:
            location = Path(image["image_path"])
            if not location.is_relative_to(root) or str(location.relative_to(root)) not in files:
                raise ValueError("Seed images must be hash-bound payloads inside the portable template")
        context = bundle["context"]
        count = len(context["time_edges"]) - 1
        if context["source_identity"].get("source_frame_range") != [0, count]:
            raise ValueError("Seed labels require the complete source episode")
        source_identity = {key: value for key, value in context["source_identity"].items()
                           if key not in ("source_frame_range", "frame_mapping")}
        if origin.get("source_identity") != source_identity or source_identity.get("episode_index") != seed["episode"]:
            raise ValueError("Template provenance differs from its reviewed seed source")
        doc = json.loads((folder / "labels.json").read_text())
        ep = SimpleNamespace(episode_index=context["source_identity"]["episode_index"], frame_count=count,
                             timestamps=context["time_edges"][:-1], source_identity=context["source_identity"])
        _reviewed_labels(doc, ep, stages)
        recording = _recording_key(context["source_identity"])
        if recording in recordings:
            raise ValueError("Two seeds of the template are the same recording")
        recordings.add(recording)
        seeds.append({"bundle": bundle, "labels": doc, "content": content, "images": images})
    return template, seeds


def prepare_template_request(template_path, target_bundle, policy_dir=None):
    """Return separate seed content and a private provenance receipt for the API.

    Each seed reference has its own frame namespace, in the template's order. Only target frames may be cited by
    candidate output. Private paths, identities and annotators stay local.
    """
    template, seeds = load_template(template_path, policy_dir=policy_dir)
    context = target_bundle.get("context", {}) if isinstance(target_bundle, dict) else {}
    if any(context.get(key) != template["settings"][key] for key in TASK_FIELDS):
        raise ValueError("Target task, ontology or policy differs from the selected template")
    from scripts.annotation_skill import verify_bundle
    verify_bundle(target_bundle, template_file(template_path).parent / "policy")
    if _recording_key(context.get("source_identity", {})) in _seed_recordings(template):
        raise ValueError("The target is a seed recording; choose a separate same-task recording")
    content, receipts = [], []
    for number, seed in enumerate(seeds, 1):
        name, doc, images = f"human-seed-{number}", seed["labels"], seed["images"]
        reference = {"reference_id": name, "task": template["settings"]["task"],
                     "stage_semantics": "active_subtask", "registry": template["settings"]["registry"],
                     "time_edges": seed["bundle"]["context"]["time_edges"], "sampled_seed_frames": [row["frame"] for row in images],
                     "human_subtasks": doc["subtasks"], "human_quality_exceptions": doc["quality"],
                     "unmarked_seed_quality": "nominal (the entire seed was explicitly reviewed)",
                     "label_mapping": template["label_mapping"], "criteria": template["criteria"]}
        if target_bundle.get("mode") == "stages":
            reference["human_subtasks"] = [{k: v for k, v in run.items() if k != "outcome"} for run in doc["subtasks"]]
            del reference["human_quality_exceptions"], reference["unmarked_seed_quality"]
            reference.pop("label_mapping", None)
        elif target_bundle.get("mode") == "quality":
            reference["human_subtasks"] = [{k: v for k, v in run.items() if k != "outcome"} for run in doc["subtasks"]]
        content.append({"type": "input_text", "text": (
            f"REFERENCE EXAMPLE ONLY — {name}, example {number} of {len(seeds)}, is a different, human-reviewed "
            f"recording. All following SEED frame numbers until END REFERENCE {name} belong only to {name}. "
            "Learn no timing/duration schedule from it; never copy its intervals or outcomes to the target. "
            "Human labels illustrate the declared criteria, not proof of any target event. Return annotations only "
            "for the TARGET clip supplied after the last END REFERENCE. Output evidence_frames must cite TARGET images "
            "only. Unsupported target judgments remain unknown.\n" + canonical(reference))})
        for row, image in zip(images, seed["content"][2::2], strict=True):
            content.extend([{"type": "input_text", "text": canonical({"reference_id": name, "seed_frame": row["frame"],
                              "timestamp": row["timestamp"], "view_names": row["view_names"]})}, image])
        content.append({"type": "input_text", "text": f"END REFERENCE {name}." + (
            " The next context and images are the TARGET to annotate." if number == len(seeds) else "")})
        receipts.append({"reference_id": name, "source_identity": seed["bundle"]["context"]["source_identity"],
                         "seed_labels_sha256": template["files"][f"seeds/{number}/labels.json"],
                         "seed_images": [{key: row[key] for key in ("frame", "timestamp", "sha256", "view_names")} for row in images],
                         "human_review": {key: doc[key] for key in ("done", "review_confirmed", "annotator", "revision", "updated_at")}})
    path = template_file(template_path)
    receipt = {"template_path": str(path), "template_sha256": _sha(path), "template_id": template["template_id"],
               "label_mapping": deepcopy(MAPPING), "seeds": receipts, "provenance": template["provenance"],
               "accepted_for_training": False}
    return content, receipt


def _seed_recordings(template):
    return {_recording_key(seed["source_identity"]) for seed in template["provenance"]["seeds"]}


def _seed_roots(template):
    """The seeds' raw sources: a private output must be outside them."""
    return [seed["source_identity"]["source_root"] for seed in template["provenance"]["seeds"]]


def _target(template, source, episode, *, task=None, source_task=None, views=None):
    settings = deepcopy(template["settings"])
    if task is not None and task != settings["task"]:
        raise ValueError("Target task differs from the template task")
    if views is not None:
        if not isinstance(views, dict) or list(views) != list(settings["views"]):
            raise ValueError("Target views must preserve the template's ordered camera names; only feature keys may change")
        settings["views"] = deepcopy(views)
    reader = LeRobotSource(Path(source), settings["views"], allow_shared_videos=True)
    ep = reader.load_episode(episode)
    known = _source_tasks(reader, episode)
    accepted = {settings["task"], settings["task_instruction"], *template["source_tasks"]}
    if source_task is not None:
        accepted.add(_text(source_task, "source_task"))
        if source_task not in known:
            raise ValueError("Explicit source_task does not match the source episode task metadata")
    if known and not set(known) <= accepted:
        raise ValueError("Source episode task metadata differs from the template; provide an explicit matching source_task alias only for the same task")
    if not known and task != settings["task"]:
        raise ValueError("Source has no task metadata; explicitly bind it with --task matching the template task")
    if _recording_key(ep.source_identity) in _seed_recordings(template):
        raise ValueError("The target is a seed recording; choose a separate same-task recording")
    binding = {"task": settings["task"], "source_tasks": known, "explicit_task": task,
               "explicit_source_task_alias": source_task, "scope": "Declared same-task binding; semantic equivalence is not automatically inferred."}
    return settings, ep, binding


def prepare_target(template_path, source, episode, output, *, task=None, source_task=None, views=None,
                   start_frame=0, end_frame=None, stride_seconds=.5, max_images=96,
                   tile_size=(320, 240), policy_dir=None):
    from scripts.prepare_annotation_evidence import prepare

    template, _ = load_template(template_path, policy_dir=policy_dir)
    settings, _, binding = _target(template, source, episode, task=task, source_task=source_task, views=views)
    output = _private(output, (source, *_seed_roots(template), template_file(template_path).parent))
    policy = Path(policy_dir).expanduser().resolve() if policy_dir else template_file(template_path).parent / "policy"
    result = prepare(source, episode, settings, output, start_frame=start_frame, end_frame=end_frame,
                     stride_seconds=stride_seconds, max_images=max_images, tile_size=tile_size, policy_dir=policy)
    _write(output / "template_binding.json", {"schema_version": "grana.annotation_template_binding.v1",
           "template_id": template["template_id"], "template_sha256": _sha(template_file(template_path)),
           "task_binding": binding, "context_sha256": result["context_sha256"], "accepted_for_training": False})
    return result | {"template_id": template["template_id"], "task": settings["task"]}


def campaign_plan(template_path, manifest_path, output, *, stride_seconds=.5, max_images=96,
                  tile_size=(320, 240), policy_dir=None):
    """Read every source and validate the complete same-task inventory before writes."""
    from scripts.prepare_annotation_evidence import select_frames
    from scripts.annotation_skill import prepare_bundle

    template, seeds = load_template(template_path, policy_dir=policy_dir)
    policy = Path(policy_dir).expanduser().resolve() if policy_dir else template_file(template_path).parent / "policy"
    manifest_path = Path(manifest_path).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text())
    if (not isinstance(manifest, dict) or set(manifest) != {"schema_version", "task", "targets"}
            or manifest["schema_version"] != TARGETS_SCHEMA or manifest["task"] != template["settings"]["task"]):
        raise ValueError("Target manifest requires grana.annotation_targets.v1, the template task and targets")
    if not isinstance(manifest["targets"], list) or not manifest["targets"]:
        raise ValueError("Target manifest must contain at least one target directory")
    jobs, names, recordings, sources = [], set(), set(), []
    for target in manifest["targets"]:
        if (not isinstance(target, dict) or not {"name", "source", "episodes"} <= target.keys()
                or target.keys() - {"name", "source", "episodes", "views", "source_task"}):
            raise ValueError("Each target requires name, source, episodes; optional views/source_task")
        name = target["name"]
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", name) or name in names:
            raise ValueError("Target names must be unique simple identifiers up to 80 characters")
        names.add(name)
        source = Path(_text(target["source"], "source")).expanduser()
        source = (manifest_path.parent / source).resolve()
        sources.append(source)
        episodes = target["episodes"]
        if (not isinstance(episodes, list) or not episodes or any(type(value) is not int or value < 0 for value in episodes)
                or len(set(episodes)) != len(episodes)):
            raise ValueError("Target episodes must be distinct nonnegative integers")
        for episode in episodes:
            settings, ep, binding = _target(template, source, episode, task=manifest["task"],
                                            source_task=target.get("source_task"), views=target.get("views"))
            identity = _recording_key(ep.source_identity)
            if identity in recordings:
                raise ValueError("Duplicate physical recording in target manifest (same video and clock hashes)")
            recordings.add(identity)
            frames = select_frames(ep.time_edges, 0, ep.frame_count, stride_seconds, max_images)
            context = {key: settings[key] for key in TASK_FIELDS}
            context.update(schema_version="grana.annotation_context.v1", time_edges=list(ep.time_edges), evidence_frames=frames,
                           source_identity={**ep.source_identity, "source_frame_range": [0, ep.frame_count],
                                            "frame_mapping": "native_frame = candidate_frame + source_frame_range[0]"})
            bundle = prepare_bundle(context, policy)
            jobs.append({"name": f"{name}-episode-{episode:06d}", "source": str(source), "episode": episode,
                         "views": settings["views"], "source_identity": ep.source_identity,
                         "task_binding": binding, "evidence_frames": frames,
                         "context_sha256": bundle["context_sha256"]})
    output = _private(output, (*sources, *_seed_roots(template), template_file(template_path).parent))
    if (not isinstance(tile_size, (tuple, list)) or len(tile_size) != 2
            or any(type(value) is not int or not 1 <= value <= 4096 for value in tile_size)):
        raise ValueError("tile_size must contain two positive dimensions no larger than 4096")
    return {"schema_version": "grana.annotation_campaign.v1", "template_id": template["template_id"],
            "template_sha256": _sha(template_file(template_path)), "manifest_sha256": _sha(manifest_path),
            "task": manifest["task"], "policy": seeds[0]["bundle"]["policy"], "output": str(output),
            "seeds": len(seeds), "seed_images": sum(len(seed["images"]) for seed in seeds),
            "sampling": {"stride_seconds": stride_seconds, "max_images": max_images, "tile_size": list(tile_size)},
            "jobs": jobs, "accepted_for_training": False}


def prepare_campaign(template_path, manifest_path, output, **options):
    plan = campaign_plan(template_path, manifest_path, output, **options)
    output = Path(plan["output"])
    output.mkdir(parents=True, exist_ok=False)
    jobs = []
    for job in plan["jobs"]:
        destination = output / "evidence" / job["name"]
        prepare_target(template_path, job["source"], job["episode"], destination, task=plan["task"],
                       source_task=job["task_binding"]["explicit_source_task_alias"], views=job["views"], **options)
        jobs.append({"name": job["name"], "bundle": str(destination / "bundle.json"),
                     "evidence": str(destination / "evidence.json"), "output_dir": str(output / "candidates" / job["name"])})
    _write(output / "jobs.json", jobs)
    _write(output / "plan.json", plan)
    return {"status": "prepared", "output": str(output), "jobs": len(jobs), "http_requests": 0,
            "next_step": "Run annotate_task.py with the same inputs and --run to submit API requests."}


def run_campaign(template_path, manifest_path, output, *, stride_seconds=.5, max_images=96,
                 tile_size=(320, 240), policy_dir=None, **api_options):
    from scripts.annotation_windows_api import run

    plan = campaign_plan(template_path, manifest_path, output, stride_seconds=stride_seconds,
                         max_images=max_images, tile_size=tile_size, policy_dir=policy_dir)
    output = Path(plan["output"])
    if not (output / "plan.json").is_file() or json.loads((output / "plan.json").read_text()) != plan:
        raise ValueError("Campaign is not completely prepared for these exact inputs; prepare a fresh output directory first")
    expected = [{"name": job["name"], "bundle": str(output / "evidence" / job["name"] / "bundle.json"),
                 "evidence": str(output / "evidence" / job["name"] / "evidence.json"),
                 "output_dir": str(output / "candidates" / job["name"])} for job in plan["jobs"]]
    if json.loads((output / "jobs.json").read_text()) != expected:
        raise ValueError("Prepared campaign jobs differ from their frozen source inventory")
    for job, paths in zip(plan["jobs"], expected, strict=True):
        bundle = json.loads(Path(paths["bundle"]).read_text())
        binding = json.loads((Path(paths["bundle"]).parent / "template_binding.json").read_text())
        expected_binding = {"schema_version": "grana.annotation_template_binding.v1", "template_id": plan["template_id"],
                            "template_sha256": plan["template_sha256"], "task_binding": job["task_binding"],
                            "context_sha256": job["context_sha256"], "accepted_for_training": False}
        if bundle.get("context_sha256") != job["context_sha256"] or binding != expected_binding:
            raise ValueError("Prepared job source/context differs from the frozen campaign inventory")
    policy = Path(policy_dir).expanduser().resolve() if policy_dir else template_file(template_path).parent / "policy"
    return run(output / "jobs.json", api_dir=output / "api", template_path=template_file(template_path),
               policy_dir=policy, **api_options)
