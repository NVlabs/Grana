#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""API results for held-out episodes labeled in grana label, compared with people.

compare: skill API results for held-out episodes, found and checked as grana draft-labels checks them (with a seed
template's --policy-dir when they were asked with one), are scored against the episodes' submitted labels with
evaluate_annotation_aq.py and evaluate_annotation_events.py.

It writes private labels into a new folder outside code checkouts and calls no API.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml

from grana.annotation.drafts import recipe_source, verified_results
from grana.annotation.labels import (SCHEMA as LABELS_SCHEMA, finished_episodes, is_legacy_file, label_path,
                                     read_episode_labels, read_legacy_episode_labels)
from grana.annotation.refinement import digest
from grana.data.builder import load_build_config, validate_label_clock
from scripts.evaluate_annotation_aq import INPUT_VERSION as AQ_INPUT, evaluate_aq
from scripts.evaluate_annotation_events import INPUT_VERSION as EVENTS_INPUT, evaluate_events

NO_LABELS = "Add `labels: <directory>` to the recipe; grana label saves one file per episode there"
# Quality codes of the label readers: the page's label and the skill's AQ class.
QUALITY = {0: ("nominal", "nominal"), 1: ("suboptimal", "suboptimal"), 2: ("regressed", "failure")}
# What a label file that started as an imported API draft records about that draft.
MODEL_ASSISTED = ("model", "response_id", "task_policy_version", "candidate_sha256", "source_frame_range", "failure_quality")


def submitted_labels(cfg: dict, episode, fps):
    """One episode's submitted labels, read and checked as grana build-cache reads them; an 11-key file is bound to
    the episode's index, trajectory and clock. Returns per-frame subtask labels, each run's outcome on its last
    frame, per-frame quality codes and the provenance."""
    validate_label_clock(episode, fps)
    if is_legacy_file(cfg["labels"], episode.episode_index):
        return read_legacy_episode_labels(cfg["labels"], episode.episode_index, times=episode.timestamps,
                                          trajectory_id=episode.source_identity["trajectory_id"], stages=cfg["stages"])
    return read_episode_labels(cfg["labels"], episode, cfg["stages"])


def nominal(provenance) -> str:
    """How unmarked time came to be Nominal: the review tick, or the submission of an 11-key file, which has none."""
    return "explicit" if provenance["review_confirmed"] else "legacy_implied"


def model_assisted(provenance):
    """The imported API draft a submitted label file started from, or None for labels made from a blank episode."""
    imported = provenance.get("machine_import")
    return None if imported is None else {key: imported[key] for key in MODEL_ASSISTED}


def spans(values):
    """Maximal runs of equal values as [start_frame, end_frame, value], half-open."""
    found = []
    for frame, value in enumerate(values):
        if found and found[-1][2] == value:
            found[-1][1] = frame + 1
        else:
            found.append([frame, frame + 1, value])
    return found


def output_folder(path, cfg: dict) -> Path:
    """--output: a new folder outside code checkouts and the raw source, since it holds private labels."""
    if os.path.lexists(Path(path).expanduser()):
        raise FileExistsError(f"--output {path} exists; give a new folder")
    output = Path(path).expanduser().resolve()
    if (output.is_relative_to(ROOT) or output.is_relative_to(Path(cfg["source"]))
            or any((parent / ".git").exists() for parent in (output, *output.parents))):
        raise ValueError("--output must be outside code checkouts and the raw source")
    return output


@contextmanager
def staged(output: Path):
    """A folder beside ``output`` that becomes ``output`` once the block succeeds, and is removed otherwise."""
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(f".{output.name}.partial-{uuid.uuid4().hex[:8]}")
    staging.mkdir()
    try:
        yield staging
        if os.path.lexists(output):
            raise FileExistsError(f"{output} appeared while writing; nothing was written")
        staging.rename(output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def write_new(path: Path, value) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


# ---------- compare: API results against people ----------

def evaluation_inputs(context, candidate, names, quality):
    """The two evaluators' inputs for one result's clip [s, e), with the submitted labels as reference: unmarked time
    is nominal and regressed is failure; human runs and spans become stage and AQ windows, and the spans events."""
    start, end = context["source_identity"]["source_frame_range"]
    ids = {stage["label"].strip(): stage["id"] for stage in context["registry"]}
    classes = [(a, b, QUALITY[code][1]) for a, b, code in spans(quality[start:end])]
    aq = {"schema_version": AQ_INPUT, "time_edges": context["time_edges"],
          "human_intervals": [{"start_frame": a, "end_frame": b, "label": name} for a, b, name in classes],
          "predicted_intervals": candidate["aq_intervals"]}
    windows = [{"id": f"quality_{number}", "start_frame": a, "end_frame": b, "quality": name,
                "role": "normal" if name == "nominal" else "event"} for number, (a, b, name) in enumerate(classes)]
    reference = {"status": "human_reviewed", "policy_version": LABELS_SCHEMA, "aq_windows": windows,
                 "stage_windows": [{"id": f"subtask_{number}", "start_frame": a, "end_frame": b, "stage_id": ids[name]}
                                   for number, (a, b, name) in enumerate(spans(names[start:end]))],
                 "events": [{"id": window["id"], "quality": window["quality"], "start_frame": window["start_frame"],
                             "end_frame": window["end_frame"], "start_boundary_bracket": [window["start_frame"]] * 2,
                             "end_boundary_bracket": [window["end_frame"]] * 2}
                            for window in windows if window["role"] == "event"]}
    events = {"schema_version": EVENTS_INPUT, "time_edges": context["time_edges"], "reference": reference,
              "candidate": {key: candidate[key] for key in ("stages", "aq_intervals", "aq_boundaries")}}
    return aq, events


def headline(aq: dict, events: dict) -> str:
    """One line of agreement: AQ time and per-class recall/precision, subtask time, human events found and how far
    the matched predictions' edges are from them."""
    percent = lambda value: "–" if value is None else f"{100 * value:.0f}%"
    if aq.get("status") == "default_normal_as_not_requested":
        return f"Subtasks {percent(events['stage']['resolved_agreement'])} agree · AQ defaults to Normal (not requested, not scored)"
    rates = ", ".join(f"{name} {percent(row['recall'])}/{percent(row['precision'])}"
                      for name, row in aq["per_class"].items() if row["support_seconds"])
    found = events["events"]["by_class"].values()
    errors = [match[f"{side}_boundary"]["absolute_distance_to_reference_bracket_seconds"]
              for match in events["events"]["matches"] for side in ("start", "end")]
    line = (f"AQ {percent(aq['accuracy'])} agree, recall/precision {rates} · "
            f"subtasks {percent(events['stage']['resolved_agreement'])} agree · "
            f"{sum(row['matched_events'] for row in found)}/{sum(row['reference_events'] for row in found)} human events found")
    return line + (f", boundary error ≤ {max(errors):.2f} s" if errors else "")


def evaluate_result(bundle, candidate, names, quality):
    """Score requested axes only; skipped AQ is a default, never a measured prediction."""
    from grana.annotation.candidate_import import display_candidate
    display = display_candidate(bundle, candidate)
    if bundle.get("mode") != "stages":
        inputs = evaluation_inputs(bundle["context"], display, names, quality)
        return inputs, (evaluate_aq(inputs[0]), evaluate_events(inputs[1]))
    context = bundle["context"]
    a, b = context["source_identity"]["source_frame_range"]
    ids = {s["label"]: s["id"] for s in context["registry"]}
    prediction = [None] * (b - a)
    invalid = display["stages"]["invalid_intervals"]
    for segment in display["stages"]["segments"]:
        if segment["identity_status"] == "resolved":
            for f in range(segment["start_frame"], segment["end_frame"]):
                if not any(row["start_frame"] <= f < row["end_frame"] for row in invalid):
                    prediction[f] = segment["stage_id"]
    clock = context["time_edges"]
    weights = [right - left for left, right in zip(clock, clock[1:])]
    resolved = sum(w for p, w in zip(prediction, weights) if p is not None)
    correct = sum(w for p, n, w in zip(prediction, names[a:b], weights) if p == ids[n])
    stage = {"resolved_agreement": correct / resolved if resolved else None,
             "coverage": resolved / sum(weights), "agreement_over_all_time": correct / sum(weights)}
    skipped = {"status": "default_normal_as_not_requested", "accuracy": None, "per_class": {}}
    events = {"status": "stages_only", "stage": stage, "events": {"status": "not_requested", "by_class": {}, "matches": []}}
    return ({"status": skipped["status"]}, {"time_edges": clock, "candidate": {"stages": display["stages"]}}), (skipped, events)


def compare(recipe, sources, output, *, failure_quality=None, policy_dir=None, log=None) -> int:
    """Score each skill API result against its episode's submitted labels; return 1 if a result needs a look, else 0.

    Results are found and checked as grana draft-labels checks them, one per episode, with the same
    ``failure_quality`` and ``policy_dir``. An episode without submitted labels is skipped. Per compared episode the
    output folder holds aq_NNNNNN.json and events_NNNNNN.json (the evaluators' inputs) and aq_metrics_NNNNNN.json and
    events_metrics_NNNNNN.json (their reports); report.json lists every result. Nothing is written if nothing is
    compared.
    """
    started = datetime.now(timezone.utc)
    recipe = Path(recipe).expanduser().resolve()
    cfg = load_build_config(recipe)
    if "labels" not in cfg:
        raise ValueError(NO_LABELS)
    labels = Path(cfg["labels"])
    output = output_folder(output, cfg)
    if log is not None:
        log(f"Source {cfg['source']} · labels {labels}")
    _, fps, selected = recipe_source(cfg)
    submitted = set(finished_episodes(labels))

    def unlabeled(index):
        """Why an episode has no labels to compare with, or None."""
        if index in submitted:
            return None
        return "its labels are a draft" if label_path(labels, index).is_file() else "no labels"

    chosen, rows = verified_results(cfg, selected, sources, skip=unlabeled, failure_quality=failure_quality,
                                    policy_dir=policy_dir)
    compared = []
    for result, draft, episode in chosen:
        row = {"episode": result.episode, "status": "compared", "from": str(result.folder), "model": result.model}
        imported = draft["machine_import"]
        try:
            bundle, candidate = (json.loads(path.read_text(encoding="utf-8"))
                                 for path in (result.bundle_path, result.folder / "candidate.json"))
            if (digest(bundle), digest(candidate)) != (imported["bundle_sha256"], imported["candidate_sha256"]):
                raise ValueError("bundle.json or candidate.json changed while it was checked")
        except (OSError, ValueError) as error:
            rows.append({**row, "status": "rejected", "reason": str(error)})
            continue
        try:
            names, _, quality, provenance = submitted_labels(cfg, episode, fps)
        except ValueError as error:
            rows.append({**row, "status": "skipped", "reason": f"its labels: {error}"})
            continue
        inputs, reports = evaluate_result(bundle, candidate, names, quality)
        assisted = model_assisted(provenance)
        # Labels a person made by correcting a draft of this very result agree with it more than blank-page labels.
        note = ("" if assisted is None else " · labels started from this result's draft"
                if assisted["candidate_sha256"] == imported["candidate_sha256"] else " · labels started from an API draft")
        rows.append({**row, "candidate_sha256": imported["candidate_sha256"], "delivery_sha256": imported["delivery_sha256"],
                     "native_frames": imported["source_frame_range"],
                     "labels_file": provenance["labels_file"], "labels_sha256": provenance["labels_sha256"],
                     "nominal": nominal(provenance), "labels_model_assisted": assisted,
                     "summary": headline(*reports) + note})
        compared.append((result.episode, inputs, reports))
    rows.sort(key=lambda item: (item["episode"] is None, item["episode"] or 0, item["from"]))
    scored = {episode for episode, _, _ in compared}
    if compared:
        with staged(output) as folder:
            for episode, (aq, events), (aq_report, events_report) in compared:
                for name, value in (("aq", aq), ("events", events), ("aq_metrics", aq_report),
                                    ("events_metrics", events_report)):
                    write_new(folder / f"{name}_{episode:06d}.json", value)
            write_new(folder / "report.json", {"recipe": str(recipe), "source": cfg["source"], "labels": str(labels),
                                               "created_at": started.isoformat(), "failure_quality": failure_quality,
                                               "results": rows})
    lines, ambiguous = [], set()
    for row in rows:
        where = f"episode {row['episode']}" if row["episode"] is not None else row["from"]
        if row["status"] == "compared":
            lines.append(f"{where}: {row['summary']}")
        elif row["status"] == "ambiguous":
            if row["episode"] not in ambiguous:
                ambiguous.add(row["episode"])
                lines.append(f"{where}: ambiguous · {row['reason']}")
        elif row["status"] not in ("not_in_recipe", "superseded"):
            lines.append(f"{where}: {row['status']} · {row['reason']}")
    grouped = [f"{count} {name}" for name, count in (
        ("not in recipe", sum(row["status"] == "not_in_recipe" for row in rows)),
        ("superseded", sum(row["status"] == "superseded" for row in rows))) if count]
    lines.append(" · ".join([f"{len(compared)} compared", *grouped])
                 + (f" · inputs and reports in {output} (private labels)" if compared else "; nothing written"))
    if log is not None:
        for line in lines:
            log(line)
    # As for drafting, a result the recipe leaves out needs no look, nor one replaced by a refinement that was compared.
    return int(any(row["status"] in ("rejected", "failed", "ambiguous", "skipped")
                   or row["status"] == "superseded" and row["episode"] not in scored for row in rows))


# ---------- command line ----------

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    scores = commands.add_parser("compare", help="Score skill API results against submitted labels of held-out episodes")
    scores.add_argument("--config", type=Path, required=True, help="The labeling recipe, as for grana label")
    scores.add_argument("--from", dest="sources", required=True, action="append", type=Path, metavar="PATH",
                        help="A skill API result folder or file, a jobs list, or a folder of results; repeatable")
    scores.add_argument("--output", type=Path, required=True, help="New private folder outside code checkouts")
    scores.add_argument("--failure-quality", choices=("regressed",),
                        help="Explicit task-policy mapping; required if candidate AQ contains failure")
    scores.add_argument("--policy-dir", type=Path, help="Exact frozen annotation policy used by preparation and submission")
    args = parser.parse_args(argv)
    say = lambda line: print(line, file=sys.stderr, flush=True)
    try:
        status = compare(args.config, args.sources, args.output, failure_quality=args.failure_quality,
                         policy_dir=args.policy_dir, log=say)
    except (OSError, ValueError, subprocess.SubprocessError, yaml.YAMLError) as error:
        # Unusable inputs, found before anything was written: one line.
        parser.exit(2, f"seed_examples {args.command}: {' '.join(str(error).split()) or type(error).__name__}\n")
    if status:
        raise SystemExit(status)


if __name__ == "__main__":
    main()
