# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Episode label files written by ``grana label`` and read by ``grana build-cache``.

One JSON file per episode, ``episode_<index>.json`` in the recipe's ``labels``
directory. Subtask runs cover every native frame in order and each run ends with
a binary outcome: ``high`` (completed, 1) or ``low`` (failed, 0). Frames outside
the quality spans are nominal. A
finished file yields the same per-frame stage labels, sparse outcome scores and
quality codes that annotated LeRobot parquet provides.
"""
from __future__ import annotations

from datetime import datetime, timezone
from copy import deepcopy
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
import uuid

from .source import same_episode

SCHEMA = "grana.labels.v1"
OUTCOMES = ("high", "low")  # completed / failed, as in the paper
QUALITY = {"suboptimal": 1, "regressed": 2}
KEYS = {"schema", "episode_index", "trajectory_id", "frame_count", "timestamps_sha256", "stages",
        "subtasks", "quality", "annotator", "done", "updated_at", "revision", "review_confirmed",
        "source_identity", "stage_registry"}
# Files from the first grana label (before revisions and the review tick) have only these 11 keys. They stay
# readable until grana.labels.v2: the label page upgrades one on its next save, build-cache reads them as they are.
LEGACY_KEYS = KEYS - {"revision", "review_confirmed", "source_identity", "stage_registry"}
IMPORT_SCHEMA = "grana.candidate_import.v1"
IMPORT_KEYS = {"schema_version", "bundle_path", "candidate_dir", "bundle_sha256", "candidate_sha256",
               "delivery_sha256", "source_frame_range", "failure_quality", "model", "response_id",
               "task_policy_version", "stage_semantics", "review_notes", "review_intervals"}
IMPORT_INTERVAL_KEYS = {"axis", "start_frame", "end_frame", "value", "stage_id", "reason",
                        "evidence_frames", "nominal_frame"}
MAX_IMPORT_BYTES = 512 * 1024


def validate_machine_import(value, frame_count):
    """Bounded provenance and original review issues, never an acceptance decision.

    Optional metadata has its own closed schema. Original API payloads stay in
    their immutable files; exports retain their hashes and native-frame issues.
    """
    v2 = isinstance(value, dict) and value.get("schema_version") == "grana.candidate_import.v2"
    keys = IMPORT_KEYS | {"annotation_mode", "action_quality_status", "outcome_status"} if v2 else IMPORT_KEYS
    if not isinstance(value, dict) or set(value) != keys or value["schema_version"] not in (IMPORT_SCHEMA, "grana.candidate_import.v2"):
        raise ValueError(f"machine_import requires schema {IMPORT_SCHEMA} and exactly {sorted(IMPORT_KEYS)}")
    if v2 and (value["annotation_mode"] not in ("stages", "quality") or value["outcome_status"] != "not_requested"
               or value["action_quality_status"] != ("default_normal_as_not_requested" if value["annotation_mode"] == "stages" else "annotated")):
        raise ValueError("machine_import annotation axes disagree with its mode")
    for name in ("bundle_path", "candidate_dir", "model", "response_id", "task_policy_version"):
        if not isinstance(value[name], str) or not value[name].strip():
            raise ValueError(f"machine_import.{name} must be nonempty text")
    for name in ("bundle_sha256", "candidate_sha256", "delivery_sha256"):
        if (not isinstance(value[name], str) or len(value[name]) != 64
                or any(char not in "0123456789abcdef" for char in value[name])):
            raise ValueError(f"machine_import.{name} must be a SHA256 digest")
    bounds = value["source_frame_range"]
    if (not isinstance(bounds, list) or len(bounds) != 2 or any(type(frame) is not int for frame in bounds)
            or not 0 <= bounds[0] < bounds[1] <= frame_count):
        raise ValueError("machine_import.source_frame_range must lie within the source episode")
    if value["failure_quality"] not in (None, "regressed"):
        raise ValueError("machine_import.failure_quality must be null or explicitly regressed")
    if value["stage_semantics"] not in ("active_subtask", "attained_state"):
        raise ValueError("machine_import.stage_semantics is unsupported")
    if (not isinstance(value["review_notes"], list) or not value["review_notes"]
            or any(not isinstance(note, str) or not note.strip() for note in value["review_notes"])):
        raise ValueError("machine_import.review_notes must be nonempty text entries")
    if not isinstance(value["review_intervals"], list):
        raise ValueError("machine_import.review_intervals must be a list")
    for interval in value["review_intervals"]:
        if not isinstance(interval, dict) or set(interval) != IMPORT_INTERVAL_KEYS:
            raise ValueError("machine_import review intervals use the exact native-frame issue schema")
        if interval["axis"] not in ("stage", "quality", "outcome", "source", "stage_boundary", "quality_boundary", "unannotated"):
            raise ValueError("machine_import review interval axis is unsupported")
        start, end = interval["start_frame"], interval["end_frame"]
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= frame_count:
            raise ValueError("machine_import review interval is outside the native episode")
        for name in ("value", "reason"):
            if not isinstance(interval[name], str) or not interval[name].strip():
                raise ValueError(f"machine_import review interval {name} must be nonempty text")
        if interval["stage_id"] is not None and (not isinstance(interval["stage_id"], str) or not interval["stage_id"].strip()):
            raise ValueError("machine_import review interval stage_id must be text or null")
        frames = interval["evidence_frames"]
        if (not isinstance(frames, list) or any(type(frame) is not int or not 0 <= frame < frame_count for frame in frames)
                or len(set(frames)) != len(frames)):
            raise ValueError("machine_import evidence frames must be distinct native frames")
        nominal = interval["nominal_frame"]
        if nominal is not None and (type(nominal) is not int or not start <= nominal <= end):
            raise ValueError("machine_import nominal boundary must lie in its uncertainty bracket")
    if len(json.dumps(value, ensure_ascii=False, allow_nan=False).encode()) > MAX_IMPORT_BYTES:
        raise ValueError("machine_import exceeds the 512 KiB provenance limit; import a shorter clip")
    return value


def label_path(directory, episode_index: int) -> Path:
    if isinstance(episode_index, bool) or not isinstance(episode_index, int) or episode_index < 0:
        raise ValueError("episode_index must be a nonnegative integer")
    return Path(directory) / f"episode_{episode_index:06d}.json"


def timestamps_sha256(times) -> str:
    return hashlib.sha256(json.dumps([float(t) for t in times]).encode()).hexdigest()


def _spans(items, name, frame_count, keys):
    if not isinstance(items, list):
        raise ValueError(f"{name} must be a list")
    end = 0
    for item in items:
        if not isinstance(item, dict) or set(item) != keys:
            raise ValueError(f"Each {name} entry needs exactly {sorted(keys)}")
        start, stop = item["start_frame"], item["end_frame"]
        if any(isinstance(v, bool) or not isinstance(v, int) for v in (start, stop)) or not 0 <= start < stop <= frame_count:
            raise ValueError(f"{name} frames must satisfy 0 <= start_frame < end_frame <= {frame_count}")
        if start < end:
            raise ValueError(f"{name} entries must be sorted and must not overlap")
        end = stop


def validate_labels(doc: dict, *, frame_count: int, stages: list[str], require_done: bool = False) -> dict:
    """Check a label document; finished documents must be complete."""
    if not isinstance(doc, dict) or set(doc) not in (KEYS, KEYS | {"machine_import"}) or doc["schema"] != SCHEMA:
        raise ValueError(f"Label files use schema {SCHEMA} with exactly {sorted(KEYS)} and optional machine_import")
    label_path(".", doc["episode_index"])
    if isinstance(doc["frame_count"], bool) or not isinstance(doc["frame_count"], int) or doc["frame_count"] != frame_count:
        raise ValueError("Label file frame_count differs from the source episode")
    if "machine_import" in doc:
        validate_machine_import(doc["machine_import"], frame_count)
    if doc["stages"] != list(stages):
        raise ValueError("Label file subtasks differ from the recipe's stage labels")
    if not isinstance(doc["done"], bool) or not isinstance(doc["review_confirmed"], bool) or not isinstance(doc["annotator"], str):
        raise ValueError("done and review_confirmed must be booleans and annotator must be text")
    if require_done and not doc["done"]:
        raise ValueError("Labels are not submitted; a draft cannot be used for training")
    if not isinstance(doc["source_identity"], dict) or not isinstance(doc["stage_registry"], list):
        raise ValueError("Labels need source_identity and the complete stage_registry")
    revision = doc["revision"]
    if revision is not None and (not isinstance(revision, str) or len(revision) != 32
                                 or any(char not in "0123456789abcdef" for char in revision)):
        raise ValueError("revision must be a saved revision token or null")
    if doc["updated_at"] is not None and not isinstance(doc["updated_at"], str):
        raise ValueError("updated_at must be text or null")
    _spans(doc["subtasks"], "subtasks", frame_count, {"start_frame", "end_frame", "stage", "outcome"})
    _spans(doc["quality"], "quality", frame_count, {"start_frame", "end_frame", "label"})
    for run in doc["subtasks"]:
        if run["stage"] not in stages:
            raise ValueError(f"Unknown subtask label: {run['stage']!r}")
        if run["outcome"] is not None and run["outcome"] not in OUTCOMES:
            raise ValueError("Subtask outcome must be high (completed), low (failed), or null in a draft")
    if any(not isinstance(span["label"], str) or span["label"] not in QUALITY for span in doc["quality"]):
        raise ValueError("Quality spans must be suboptimal or regressed; unmarked frames are nominal")
    if doc["done"] or require_done:
        problems = completion_problems(doc)
        if problems:
            raise ValueError("Labels are not finished: " + "; ".join(problems))
    return doc


def completion_problems(doc: dict) -> list[str]:
    """Reasons a structurally valid document cannot be submitted yet."""
    problems, covered, previous = [], 0, None
    for run in doc["subtasks"]:
        if run["start_frame"] != covered:
            problems.append(f"frames {covered}-{run['start_frame'] - 1} have no subtask")
        if previous == run["stage"]:
            problems.append(f"touching runs of subtask {run['stage']} must be merged")
        if run["outcome"] is None:
            problems.append(f"subtask {run['stage']} ending at frame {run['end_frame'] - 1} has no outcome")
        covered, previous = run["end_frame"], run["stage"]
    if covered != doc["frame_count"]:
        problems.append(f"frames {covered}-{doc['frame_count'] - 1} have no subtask")
    if not doc["annotator"].strip():
        problems.append("annotator name is empty")
    if not doc["review_confirmed"]:
        problems.append("confirm that the entire video was reviewed")
    return problems


def dense_labels(doc: dict) -> tuple[list[str], list[str], list[int]]:
    """Per-frame stage labels, outcome scores on each run's last frame, and quality codes."""
    count = doc["frame_count"]
    labels, scores, quality = [""] * count, [""] * count, [0] * count
    for run in doc["subtasks"]:
        labels[run["start_frame"]:run["end_frame"]] = [run["stage"]] * (run["end_frame"] - run["start_frame"])
        if run["outcome"] is not None:
            scores[run["end_frame"] - 1] = run["outcome"]
    for span in doc["quality"]:
        quality[span["start_frame"]:span["end_frame"]] = [QUALITY[span["label"]]] * (span["end_frame"] - span["start_frame"])
    return labels, scores, quality


def new_labels(episode_index: int, trajectory_id: str, times, stages: list[str], *, source_identity: dict,
               stage_registry: list[dict]) -> dict:
    return {"schema": SCHEMA, "episode_index": episode_index, "trajectory_id": trajectory_id,
            "frame_count": len(times), "timestamps_sha256": timestamps_sha256(times), "stages": list(stages),
            "subtasks": [], "quality": [], "annotator": "", "done": False, "updated_at": None,
            "revision": None, "review_confirmed": False, "source_identity": deepcopy(source_identity),
            "stage_registry": deepcopy(stage_registry)}


def is_legacy(doc) -> bool:
    """An 11-key file written by the first ``grana label``."""
    return isinstance(doc, dict) and set(doc) == LEGACY_KEYS and doc.get("schema") == SCHEMA


def legacy_revision(raw: bytes) -> str:
    """A revision token for an 11-key file, taken from its bytes: the page opens the file as saved, and a save
    based on it conflicts if the file changed meanwhile (a null base would overwrite it unchecked)."""
    return hashlib.sha256(raw).hexdigest()[:32]


def from_legacy(doc: dict, *, source_identity: dict, stage_registry: list[dict], revision: str | None = None) -> dict:
    """An 11-key file as a current document; a submitted file counts as reviewed."""
    return {**doc, "revision": revision, "review_confirmed": doc["done"] is True,
            "source_identity": deepcopy(source_identity), "stage_registry": deepcopy(stage_registry)}


def validate_legacy_binding(doc, episode_index: int, trajectory_id: str, times) -> None:
    """11-key files are bound to the episode index, trajectory id and native clock, as when they were written."""
    if (doc["episode_index"], doc["trajectory_id"], doc["timestamps_sha256"]) != (episode_index, trajectory_id, timestamps_sha256(times)):
        raise ValueError("Label file was made for a different source episode")


def validate_binding(doc, episode, stages):
    """Bind labels to the episode's source evidence, wherever the dataset lives, and to the ordered stage descriptions."""
    if (doc["episode_index"] != episode.episode_index or not same_episode(doc["source_identity"], episode.source_identity)
            or doc["trajectory_id"] != episode.source_identity["trajectory_id"]
            or doc["timestamps_sha256"] != timestamps_sha256(episode.timestamps)):
        raise ValueError("Label file was made for a different source episode")
    if doc["stage_registry"] != stages:
        raise ValueError("Label file stage registry differs from the recipe's labels or prompts")


@contextmanager
def episode_lock(directory, episode_index):
    """Serialize compare-and-save across tabs, threads, and local server processes."""
    path = label_path(directory, episode_index)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_name(f".{path.name}.lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def write_labels(directory, doc: dict, *, keep_updated_at: bool = False, new_only: bool = False) -> Path:
    """Replace one episode's label file atomically under a new revision. ``updated_at`` becomes the time of this
    write, or stays the document's own with ``keep_updated_at`` (labels carried over as they were saved).
    ``new_only`` publishes atomically without replacing even a concurrently created file or symbolic link."""
    path = label_path(directory, doc["episode_index"])
    path.parent.mkdir(parents=True, exist_ok=True)
    updated_at = doc["updated_at"] if keep_updated_at else datetime.now(timezone.utc).isoformat()
    doc = {**doc, "updated_at": updated_at, "revision": uuid.uuid4().hex}
    descriptor, temporary = tempfile.mkstemp(prefix=".labels-", suffix=".json", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(doc, stream, indent=1, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        if new_only:
            os.link(temporary, path)   # atomic no-clobber publication, including concurrent non-UI writers
        else:
            os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return path


def finished_episodes(directory) -> list[int]:
    """Episode indices whose label files are submitted (done)."""
    found = []
    for path in sorted(Path(directory).glob("episode_*.json")):
        try:
            doc = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if isinstance(doc, dict) and doc.get("done") is True:
            index = doc.get("episode_index")
            if isinstance(index, int) and not isinstance(index, bool) and index >= 0 and path == label_path(directory, index):
                found.append(index)
    return found


def read_episode_labels(directory, episode, stages: list[dict]):
    """Return dense labels, scores and quality for a finished episode, plus provenance."""
    episode_index = episode.episode_index
    path = label_path(directory, episode_index)
    if not path.is_file():
        raise FileNotFoundError(f"No labels for episode {episode_index} ({path}); label it with `grana label`")
    raw = path.read_bytes()
    doc = validate_labels(json.loads(raw), frame_count=episode.frame_count,
                          stages=[stage["label"] for stage in stages], require_done=True)
    validate_binding(doc, episode, stages)
    labels, scores, quality = dense_labels(doc)
    provenance = {"annotation_source": SCHEMA, "labels_file": str(path), "labels_sha256": hashlib.sha256(raw).hexdigest(),
                  "annotator": doc["annotator"], "labeled_at": doc["updated_at"],
                  "label_revision": doc["revision"], "review_confirmed": doc["review_confirmed"],
                  "source_identity": episode.source_identity, "clock_receipt": episode.clock_receipt,
                  "stage_registry": doc["stage_registry"],
                  "quality_source": "labels.quality", "quality_disagreement_frames": None}
    if "machine_import" in doc:
        provenance["machine_import"] = deepcopy(doc["machine_import"])
    return labels, scores, quality, provenance


def is_legacy_file(directory, episode_index: int) -> bool:
    try:
        return is_legacy(json.loads(label_path(directory, episode_index).read_text()))
    except (OSError, ValueError):
        return False


def read_legacy_episode_labels(directory, episode_index: int, *, trajectory_id: str, times, stages: list[dict]):
    """Dense labels from a submitted 11-key file; works for LeRobot v2 and v3 sources."""
    path = label_path(directory, episode_index)
    raw = path.read_bytes()
    stored = json.loads(raw)
    if not is_legacy(stored):
        raise ValueError(f"{path} is not an 11-key grana.labels.v1 file")
    doc = validate_labels(from_legacy(stored, source_identity={}, stage_registry=stages), frame_count=len(times),
                          stages=[stage["label"] for stage in stages], require_done=True)
    validate_legacy_binding(doc, episode_index, trajectory_id, times)
    labels, scores, quality = dense_labels(doc)
    provenance = {"annotation_source": SCHEMA, "label_format": "11-key (before revisions)", "labels_file": str(path),
                  "labels_sha256": hashlib.sha256(raw).hexdigest(), "annotator": doc["annotator"],
                  "labeled_at": doc["updated_at"], "label_revision": None, "review_confirmed": None,
                  "quality_source": "labels.quality", "quality_disagreement_frames": None}
    return labels, scores, quality, provenance
