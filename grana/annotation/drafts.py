# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``grana draft-labels``: import completed skill API results as drafts for ``grana label``.

Each result is verified, converted and written by the import core (``candidate_import``): create-only, into
a recipe episode with no label file, as an unconfirmed draft whose ``machine_import`` lists what to review.
This module finds the results under ``--from``, pairs each with its bundle by digest, keeps one result per
episode among those that pass, and reports every result. Nothing calls an API.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess

from grana.data.builder import load_build_config
from .candidate_import import verified_draft, write_draft
from .labels import is_legacy, label_path
from .refinement import digest
from .source import LeRobotSource
from .stage_settings import StageConflict, StageSettings

RESULT_FILES = ("delivery.json", "candidate.json", "validation.json", "prepared.json")   # in a skill API result folder
STATUSES = ("written", "would_write", "skipped", "not_in_recipe", "superseded", "ambiguous", "rejected", "failed",
            "not_attempted")
PROBLEMS = {"ambiguous", "rejected", "failed", "not_attempted"}   # any of these: exit status 1
LABELED = "has labels"


class Rejected(Exception):
    """A result that cannot be drafted, with the episode it names when that is known."""

    def __init__(self, reason: str, episode: int | None = None):
        super().__init__(reason)
        self.episode = episode


@dataclass
class Result:
    """One completed skill API result and the bundle file it was asked with."""
    folder: Path
    bundle_path: Path
    episode: int
    model: str | None
    key: tuple   # its bundle and candidate digests: the same result reached twice is drafted once
    delivery_sha256: str
    refines: str | None = None   # digest of the delivery.json its refined evidence was planned from


def _object(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as error:
        raise ValueError(f"{path.name} is not JSON ({error})") from None
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} is not a JSON object")
    return value


def _failure(error: OSError) -> str:
    return f"{error.strerror or error}" + (f" ({error.filename})" if error.filename else "")


# ---------- finding results ----------

def _from_file(path: Path, given):
    """What one --from file stands for: its API result folder, or a jobs list's results."""
    if path.name in RESULT_FILES:
        return {path.parent: None}, set()
    try:
        jobs = json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        jobs = None
    if (not isinstance(jobs, list) or not jobs
            or any(not isinstance(job, dict) or not all(isinstance(job.get(key), str) for key in ("bundle", "output_dir"))
                   for job in jobs)):
        raise ValueError(f"--from {given}: expected a folder, an API result file ({', '.join(RESULT_FILES)}) "
                         "or a jobs list [{name, bundle, evidence, output_dir}]")
    # The batch runner's jobs list pairs each result folder with its bundle (relative paths: from the list's folder).
    return ({(path.parent / Path(job["output_dir"]).expanduser()).resolve():
             (path.parent / Path(job["bundle"]).expanduser()).resolve() for job in jobs}, set())


def _walk(top: Path):
    """API result folders and bundles under ``top``; evidence images and dot folders are skipped."""
    results, bundles = {}, set()
    for folder, subfolders, names in os.walk(top):   # never follows symbolic links
        here = Path(folder)
        if "plan.json" in names:
            plan = _object(here / "plan.json")
            if plan.get("schema_version") == "grana.annotation_round_plan.v1":
                # Internal chunks are evidence for these aggregates, not competing episode candidates.
                for result, bundle in ((here / "candidate", here / "evidence/bundle.json"),
                                       (here / "quality", here / "quality/bundle.json")):
                    if (result / "delivery.json").is_file():
                        results[result] = bundle
                        bundles.add(bundle)
                subfolders.clear()
                continue
        if "bundle.json" in names:
            bundles.add(here / "bundle.json")
        if {"delivery.json", "prepared.json", "candidate.json"} & set(names):
            results[here] = None
        else:
            subfolders[:] = [name for name in subfolders if not name.startswith(".") and name != "images"]
            continue
        subfolders.clear()
    return results, bundles


def _discover(sources):
    """Skill result folders (with the bundle a jobs list names) and bundle files under --from."""
    results, bundles = {}, set()
    for source in sources:
        path = Path(source).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"--from {source}: no such file or folder")
        found = _from_file(path.resolve(), source) if path.is_file() else _walk(path.resolve())
        if not any(found):
            raise ValueError(f"--from {source}: no skill API result found")
        for folder, bundle in found[0].items():
            results[folder] = bundle or results.get(folder)
        bundles |= found[1]
    if not results:
        raise ValueError("--from: found bundles but no skill API result")
    return results, bundles


def _bundle_digest(path: Path, digests):
    """The canonical digest of a bundle file, or None if it cannot be read; once per file."""
    if path not in digests:
        try:
            digests[path] = digest(_object(path))
        except (OSError, ValueError):
            digests[path] = None
    return digests[path]


def _bundle(paths, wanted, digests):
    """The first bundle file among ``paths`` whose canonical digest is ``wanted``: a path alone is never trusted."""
    if not isinstance(wanted, str):
        return None
    return next((path for path in paths if _bundle_digest(path, digests) == wanted), None)


def _refined_from(path: Path, digests):
    """The digest of the delivery.json a refined bundle was planned from (its refinement_receipt.json), or None."""
    receipt = path.parent / "refinement_receipt.json"
    try:
        binding = _object(receipt).get("binding") if receipt.is_file() else None
    except (OSError, ValueError):
        return None
    if not isinstance(binding, dict) or binding.get("expanded_bundle_sha256") != _bundle_digest(path, digests):
        return None
    baseline = binding.get("baseline_delivery_sha256")
    return baseline if isinstance(baseline, str) else None


def _result(folder: Path, hint: Path | None, bundles, digests) -> Result:
    """A result folder of scripts/annotation_skill_api.py and the bundle its delivery.json names; the import core
    checks everything else."""
    episode = None
    try:
        # The driver writes prepared.json before its API call and delivery.json once the call is over.
        receipt = next((folder / name for name in ("delivery.json", "prepared.json") if (folder / name).is_file()), None)
        if receipt is None:   # a job not run yet has no folder
            raise ValueError("unfinished: no delivery.json yet" if not folder.exists()
                             else "no delivery.json; only results of scripts/annotation_skill_api.py are drafted")
        delivery = _object(receipt)
        problem = ("unfinished: no delivery.json yet" if receipt.name == "prepared.json"
                   else None if delivery.get("status") == "completed"
                   else f"the API result is {delivery.get('status')!r}, not completed")
        wanted, candidate = delivery.get("bundle_sha256"), delivery.get("candidate_sha256")
        listed = delivery.get("images") if isinstance(delivery.get("images"), list) else []
        images = [Path(row["image_path"]) for row in listed
                  if isinstance(row, dict) and isinstance(row.get("image_path"), str)][:1]
        # Evidence images live in <evidence>/images/ beside the bundle; a jobs list names the bundle itself.
        recorded = delivery.get("bundle_path")
        places = [*([hint] if hint else []), *([Path(recorded)] if isinstance(recorded, str) else []),
                  folder / "bundle.json", *(place / "bundle.json" for image in images for place in image.parents[:2]),
                  *sorted(bundles)]
        bundle_path = _bundle(places, wanted, digests)
        if bundle_path is None:
            raise ValueError(problem or "no bundle found; add its evidence folder to --from")
        context = _object(bundle_path).get("context")
        identity = context.get("source_identity") if isinstance(context, dict) else None
        if not isinstance(identity, dict) or type(identity.get("episode_index")) is not int or identity["episode_index"] < 0:
            raise ValueError("the bundle's source identity names no episode")
        episode = identity["episode_index"]
        if problem:   # reported with its episode: one that has labels, or is not in the recipe, needs no look
            raise ValueError(problem)
    except ValueError as error:
        raise Rejected(str(error), episode) from None
    # Without a candidate digest a result is its own key: the import core rejects it, so none is dropped unseen.
    return Result(folder=folder, bundle_path=bundle_path, episode=episode, model=delivery.get("model"),
                  key=(wanted, candidate if isinstance(candidate, str) else str(folder)), delivery_sha256=digest(delivery),
                  refines=delivery.get("stage_delivery_sha256") or _refined_from(bundle_path, digests))


def _row(status: str, origin, episode=None, reason=None, **fields) -> dict:
    return {"episode": episode, "status": status, "from": str(origin), **({"reason": reason} if reason else {}), **fields}


def _result_row(result: Result, status: str, reason=None, **fields) -> dict:
    return _row(status, result.folder, result.episode, reason, model=result.model, **fields)


def _collect(sources):
    """Results under the --from paths without duplicates, rows for those that cannot be used, and the refinements
    there: the digest of the delivery.json each was planned from, with its evidence folder."""
    found, bundles = _discover(sources)
    results, rows, seen, digests = [], [], set(), {}
    for folder, hint in sorted(found.items()):
        try:
            result = _result(folder, hint, bundles, digests)
        except Rejected as rejected:
            rows.append(_row("rejected", folder, rejected.episode, str(rejected)))
        except OSError as error:
            rows.append(_row("failed", folder, reason=_failure(error)))
        else:
            if result.key not in seen:   # the same result reached twice, e.g. through a folder and a jobs list
                seen.add(result.key)
                results.append(result)
    # Every bundle seen: found under --from, named by a jobs list, or located for a result (finished or not).
    refinements = {}
    for path in sorted({*bundles, *filter(None, found.values()), *(path for path, value in digests.items() if value)}):
        baseline = _refined_from(path, digests)
        if baseline:
            refinements.setdefault(baseline, path.parent)
    return results, rows, refinements


# ---------- checking them against the recipe ----------

def recipe_source(cfg: dict):
    """The recipe's LeRobot source, its frame rate and the episodes the recipe selects, checked as grana label checks
    them, with the FFmpeg tools that check episode videos on PATH."""
    dataset = LeRobotSource(Path(cfg["source"]), cfg["views"], allow_shared_videos=True)
    fps = dataset.info.get("fps")
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not 0 < fps < float("inf"):
        raise ValueError("Source fps must be finite and positive")
    if cfg["fps"] > fps:
        raise ValueError("Output fps cannot exceed source fps")
    selected = set(dataset.episodes) if cfg["episodes"] == "all" else set(cfg["episodes"])
    if selected - dataset.episodes.keys():
        raise ValueError(f"Episodes absent from the dataset: {sorted(selected - dataset.episodes.keys())}")
    for tool in ("ffprobe", "ffmpeg") if dataset.shared is not None else ("ffprobe",):
        if not shutil.which(tool):
            raise FileNotFoundError(f"{tool} (part of FFmpeg) must be on PATH to check the episode videos")
    return dataset, fps, selected


def verified_results(cfg: dict, selected, sources, *, skip, failure_quality=None, policy_dir=None):
    """The results under ``sources`` that pass the import core's checks, at most one per episode, as
    grana draft-labels finds and checks them; ``selected`` holds the recipe's episodes (recipe_source).

    ``skip(index)`` is why an episode needs no result (for drafting: it has labels), or None. Returns the
    (result, draft, episode) triples of verified_draft in episode order, and a report row (STATUSES) for every other
    result.
    """
    results, rows, refinements = _collect(sources)
    # A result that cannot be used needs no look in an episode the recipe leaves out or one that ``skip`` names.
    for row in rows:
        if row["episode"] is not None and row["episode"] not in selected:
            row.update(status="not_in_recipe")
            row.pop("reason", None)
        elif row["episode"] is not None and (reason := skip(row["episode"])):
            row.update(status="skipped", reason=reason)
    # Cheap checks first: a skipped episode is never probed.
    passed = {}
    for result in sorted(results, key=lambda item: (item.episode, str(item.folder))):
        if result.episode not in selected:
            rows.append(_result_row(result, "not_in_recipe"))
        elif reason := skip(result.episode):
            rows.append(_result_row(result, "skipped", reason))
        else:
            try:
                draft, episode = verified_draft(cfg, result.episode, result.bundle_path, result.folder,
                                                failure_quality=failure_quality, policy_dir=policy_dir)
            except (ValueError, TypeError) as error:
                rows.append(_result_row(result, "rejected", str(error)))
            except FileNotFoundError as error:   # a receipt or image the result needs
                rows.append(_result_row(result, "rejected", _failure(error)))
            except subprocess.SubprocessError:
                rows.append(_result_row(result, "failed", "could not check the episode videos with ffprobe"))
            except OSError as error:
                rows.append(_result_row(result, "failed", _failure(error)))
            else:
                passed.setdefault(result.episode, []).append((result, draft, episode))
    # One result per episode, chosen among those that pass: a refinement (denser evidence planned from one result)
    # replaces that result, and any other choice is the user's. Drafts are create-only, so a result whose refinement
    # has no result yet waits for it (passed alone with --from, it is drafted).
    chosen = []
    for _, candidates in sorted(passed.items()):
        refiners = {result.refines: result for result, _, _ in candidates if result.refines}
        kept = []
        for item in candidates:
            result = item[0]
            refiner = refiners.get(result.delivery_sha256)
            if refiner is not None and refiner is not result:
                rows.append(_result_row(result, "superseded", f"refined by {refiner.folder}"))
            elif result.delivery_sha256 in refinements:
                rows.append(_result_row(result, "superseded",
                                        f"its refinement {refinements[result.delivery_sha256]} has no result to draft yet"))
            else:
                kept.append(item)
        if len(kept) > 1:
            rows += [_result_row(result, "ambiguous", f"{len(kept)} results for this episode; pass one with --from")
                     for result, _, _ in kept]
        elif kept:
            chosen.append(kept[0])
    return chosen, rows


def _check_labels_directory(labels: Path, cfg: dict) -> None:
    """Stop if the labels directory holds files made with other stages: a recipe whose ``labels`` resolves elsewhere
    (an environment variable from another project) would put drafts that project's page cannot open."""
    order = [stage["label"] for stage in cfg["stages"]]
    for path in sorted(labels.glob("episode_*.json")):
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if is_legacy(doc):
            same = doc["stages"] == order
        elif isinstance(doc, dict) and "stage_registry" in doc:
            same = doc["stage_registry"] == cfg["stages"]
        else:
            continue
        if not same:
            raise ValueError(f"{path} was labeled with other subtasks than this recipe's; "
                             "check that the recipe's `labels:` is this project's directory")


def _save_report(labels: Path, report: dict, when: datetime) -> Path:
    folder = labels / ".draft-labels"
    folder.mkdir(exist_ok=True)
    stamp = when.strftime("%Y%m%dT%H%M%SZ")
    for suffix in ("", *(f"-{n}" for n in range(2, 100))):
        path = folder / f"run-{stamp}{suffix}.json"
        try:
            with path.open("x", encoding="utf-8") as stream:
                json.dump({**report, "report": str(path)}, stream, indent=2, ensure_ascii=False)
                stream.write("\n")
            return path
        except FileExistsError:
            continue
    raise FileExistsError(f"Too many reports at {stamp} in {folder}")


def draft_labels(recipe, sources, *, dry_run: bool = False, failure_quality=None, policy_dir=None, log=None) -> dict:
    """Import the completed skill API results under ``sources`` as drafts of one labeling recipe; return the report.

    Unusable inputs raise ValueError or OSError before anything is written. Every result then gets a status
    (STATUSES): a failure for one episode is reported and the others continue. ``log`` receives one line first:
    the source and labels directories the recipe resolves to.
    """
    started = datetime.now(timezone.utc)
    recipe = Path(recipe).expanduser().resolve()
    cfg = load_build_config(recipe)
    if "labels" not in cfg:
        raise ValueError("Add `labels: <directory>` to the recipe; grana label saves one file per episode there")
    if failure_quality not in (None, "regressed"):
        raise ValueError("failure_quality must be explicitly regressed or omitted")
    labels = Path(cfg["labels"])
    if log is not None:
        log(f"Source {cfg['source']} · labels {labels}{' (dry run)' if dry_run else ''}")
    settings = StageSettings(cfg, recipe)
    selected = recipe_source(cfg)[2]
    _check_labels_directory(labels, cfg)
    has_labels = lambda index: LABELED if os.path.lexists(label_path(labels, index)) else None
    chosen, rows = verified_results(cfg, selected, sources, skip=has_labels, failure_quality=failure_quality,
                                    policy_dir=policy_dir)
    # Create-only writes under the locks page saves take, in the same order.
    for position, (result, draft, episode) in enumerate(chosen):
        fields = {"labels_file": str(label_path(labels, result.episode)),
                  "review_issues": len(draft["machine_import"]["review_intervals"])}
        if dry_run:
            rows.append(_result_row(result, "would_write", **fields))
            continue
        try:
            write_draft(cfg, settings, draft, episode)
        except StageConflict:
            # A stage edit moved the project to a new labels directory; drafts written before it moved along.
            rows += [_result_row(other, "not_attempted", "the recipe changed (a stage edit?); run again")
                     for other, _, _ in chosen[position:]]
            break
        except FileExistsError:   # saved meanwhile
            rows.append(_result_row(result, "skipped", LABELED))
        except ValueError as error:   # the source changed meanwhile
            rows.append(_result_row(result, "rejected", str(error)))
        except subprocess.SubprocessError:
            rows.append(_result_row(result, "failed", "could not check the episode videos with ffprobe"))
        except OSError as error:
            rows.append(_result_row(result, "failed", _failure(error)))
        else:
            rows.append(_result_row(result, "written", **fields))
    rows.sort(key=lambda item: (item["episode"] is None, item["episode"] or 0, item["from"]))
    counts = {status: sum(item["status"] == status for item in rows) for status in STATUSES}
    hints = []
    if rows and all(item["status"] == "skipped" for item in rows):
        labeled = sorted({item["episode"] for item in rows})
        which, them = ((f"all {len(labeled)} episodes already have", "them") if len(labeled) > 1
                       else (f"episode {labeled[0]} already has", "it"))
        hints.append(f"Nothing drafted: {which} labels. To import results into {them}, "
                     "copy the recipe with a new `labels:` directory.")
    report = {"recipe": str(recipe), "source": cfg["source"], "labels": str(labels), "dry_run": dry_run,
              "failure_quality": failure_quality,
              "policy_dir": None if policy_dir is None else str(Path(policy_dir).expanduser().resolve()),
              "created_at": started.isoformat(), "counts": {status: n for status, n in counts.items() if n},
              "results": rows, "hints": hints, "report": None}
    if not dry_run and labels.is_dir():
        try:
            report["report"] = str(_save_report(labels, report, started))
        except OSError as error:
            report["report_error"] = _failure(error)
    return report


def exit_status(report: dict) -> int:
    """0 when every result was drafted or rightly left alone; 1 when one needs a look."""
    return 1 if PROBLEMS & report["counts"].keys() or report.get("report_error") else 0


# The statuses summed up on one line, as (status, name).
GROUPED = (("skipped", "skipped (has labels)"), ("not_in_recipe", "not in recipe"), ("superseded", "superseded"),
           ("not_attempted", "not attempted (the recipe changed; run again)"))


def summary_lines(report: dict) -> list[str]:
    """The report for a terminal: a line for each episode to look at, one line for the rest, then the total."""
    lines, ambiguous = [], set()
    for row in report["results"]:
        where = f"episode {row['episode']}" if row["episode"] is not None else row["from"]
        if row["status"] in ("written", "would_write"):
            issues = row["review_issues"]
            lines.append(f"{where}: {'draft' if row['status'] == 'written' else 'would draft'} · "
                         f"{issues} review issue{'' if issues == 1 else 's'}")
        elif row["status"] in ("rejected", "failed"):
            lines.append(f"{where}: {row['status']} · {row['reason']}")
        elif row["status"] == "ambiguous" and row["episode"] not in ambiguous:
            ambiguous.add(row["episode"])
            lines.append(f"{where}: ambiguous · {row['reason']}")
    grouped = [f"{n} {name}" for status, name in GROUPED
               if (n := sum(row["status"] == status for row in report["results"]))]
    if grouped:
        lines.append(" · ".join(grouped))
    counts = report["counts"]
    drafted = counts.get("would_write" if report["dry_run"] else "written", 0)
    where = "would be written to" if report["dry_run"] else "in"
    total = f"{drafted} draft{'' if drafted == 1 else 's'} {where} {report['labels']}"
    if report.get("report"):
        total += f" · report {report['report']}"
    elif report.get("report_error"):
        total += f" · report not saved: {report['report_error']}"
    return [*lines, total, *report["hints"]]
