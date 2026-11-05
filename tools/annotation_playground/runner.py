# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Playground state, API jobs and local steps; status polling never starts work.

Generation runs only after an explicit click, one request at a time, with a model from the workspace's model
list. Every route and model spends the same request budget and keeps its receipts in the same folder. Imports and
the seeded route's local steps run one at a time on a second worker. What the page shows of a refusal or a failure
is a note: a message code with its values, which the page words in its language, and the English text.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import threading
from urllib.parse import urlsplit
import uuid

SCHEMA = "grana.annotation_playground.v3"
ROUTES = ("human", "api", "seeded")
SERVICES = ("human", "api", "seeded", "seed")   # label pages, on the ports after the home page
# Target evidence: every configured view once a second, at most 96 mosaics of 320x240 pixels per view. A seed
# example samples its seeds the same way, its seeds sharing the 96 mosaics.
EVIDENCE = {"stride_seconds": 1.0, "max_images": 96, "tile_size": (320, 240)}
FAILURE_QUALITY = "regressed"   # the playground's declared meaning of the API's failure class
# A model list entry; api_key_env names the variable holding the key, never the key itself.
MODEL_FIELDS = ("id", "label", "base_url", "model", "api_key_env", "reasoning_effort", "protocol")
MAX_MODELS = 16
MODEL_ID = r"[a-z0-9][a-z0-9._-]{0,31}"
KEY_NAME = r"[A-Za-z_][A-Za-z0-9_]{0,127}"
# Each protocol's request driver: a module whose run() takes the arguments of scripts/annotation_skill_api.run.
# Responses is the only one; another protocol needs its own driver here.
PROTOCOLS = {"responses": "scripts.annotation_skill_api"}
# The English of the refusals by code; the page words each code in its other languages. Refusals of
# grana.annotation.templates (seed_count, legacy_seed, image_budget, ...) bring their own code and English.
MESSAGES = {
    "seeded_episode_count": "Seed examples need at least two episodes: one seed and one held-out episode. Human labeling and pure API annotation are available independently.",
    "seed_count": "Choose 1 to {max} seeds",
    "no_held_out": "Choose at least one held-out episode",
    "seed_and_held_out": "An episode cannot be both a seed and held out",
    "not_in_route": "Episode {episode} is not one of this route's episodes",
    "not_submitted": "Episode {episode} is not submitted on the seed page",
    "choose_first": "Choose 1 to {max} submitted seeds and held-out episodes first",
    "busy": "Another local step is running; wait for it to finish",
    "cannot_generate": "Episode {episode} cannot be generated in this route now",
    "no_credential": "Episode {episode}: this model's key variable {name} is not set in the local service",
    "budget": "Episode {episode}: the shared request budget is used up",
    "failed_job": "Episode {episode}: an earlier request needs inspection",
    "example": "Episode {episode}: build the example first",
    "validate": "Episode {episode}: compare the held-out episodes with this model first",
    "has_job": "Episode {episode}: it already has a job or labels",
    "over_budget": "{asked} requests asked, {left} left in the shared budget",
    "worker": "The local worker could not start; no request was launched",
    "no_result": "Episode {episode} has no generated result of this model to import, or already has labels",
    "preview_again": "Preview the import again; the one confirmed is no longer current",
    "compare_before_import": "Compare the held-out episodes with this model before importing its targets",
    "nothing_generated": "Nothing generated with this model to import yet",
    "no_folder": "{folder} is not an absolute path to existing results",
    "generate_held_out": "Generate at least one held-out episode with this model under the current example first",
    "no_comparison": "This comparison is no longer listed; compare again",
    "not_compared": "Episode {episode} was not compared there",
    "result_changed": "Episode {episode}'s result changed after this comparison; compare again",
}
# Reasons of grana draft-labels and seed_examples.py compare the page words in its languages, by their English.
REASONS = {"has labels": "has_labels", "no labels": "no_labels", "its labels are a draft": "labels_draft"}


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with temp.open("x", encoding="utf-8") as stream:
        os.chmod(temp, 0o600)
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def comparison_current(compare, version, held_out):
    """A completed, unchanged comparison of the entire selected cohort; never an accuracy claim."""
    if (not version or not held_out or compare.get("key") != version["key"]
            or compare.get("held_out") != held_out or type(compare.get("status")) is not int or compare["status"] != 0):
        return False
    results = compare.get("results", [])
    scored = [row for row in results if row.get("status") == "compared"]
    if (len(scored) != len(held_out) or {row.get("episode") for row in scored} != set(held_out)
            or any(row.get("status") not in ("compared", "superseded", "not_in_recipe") for row in results)):
        return False
    try:
        for row in scored:
            if (hashlib.sha256(Path(row["labels_file"]).read_bytes()).hexdigest() != row["labels_sha256"]
                    or digest(read_json(Path(row["from"]) / "delivery.json")) != row["delivery_sha256"]
                    or digest(read_json(Path(row["from"]) / "candidate.json")) != row["candidate_sha256"]):
                return False
    except (OSError, ValueError, TypeError, KeyError):
        return False
    return True


def result_folder(job):
    return Path(job.get("candidate_dir", Path(job["output"]) / "candidate"))


def result_bundle(job):
    return Path(job.get("bundle_path", Path(job["output"]) / "evidence/bundle.json"))


def planned_attempts(time_edges, *, stages=True, quality=False, segments=None):
    """HTTP attempts grana.annotation.rounds.run_episode reserves for one episode as the playground runs it: its default
    windows, one attempt per request (max_attempts=1). The stage requests are run_episode's own split of the episode's
    clock, exactly. AQ is asked per frozen stage segment, so given those ``segments`` it is exact too. Before the stage
    answer exists the segments are unknown, so AQ is a conservative lower bound: its clips partition the episode and
    each spans at most quality_seconds of the clock (or one frame), so there are at least duration / that span."""
    from grana.annotation.rounds import DEFAULTS, split

    count, images = len(time_edges) - 1, DEFAULTS["max_images"]
    attempts = len(split(time_edges, 0, count, DEFAULTS["stage_seconds"], DEFAULTS["stage_stride"], images)) if stages else 0
    if quality and segments is not None:
        attempts += sum(len(split(time_edges, a, b, DEFAULTS["quality_seconds"], DEFAULTS["quality_stride"], images))
                        for a, b in segments)
    elif quality:
        widest = max(DEFAULTS["quality_seconds"], *(b - a for a, b in zip(time_edges, time_edges[1:])))
        attempts += max(1, math.ceil((time_edges[-1] - time_edges[0]) / widest - 1e-9))
    return attempts


def spent(job, reservations):
    """Attempts a job's requests reserved in the shared budget. grana.annotation.api reserves each under its request's
    name: the job's id for the one request of an earlier playground, <id>-stage-<n> or <id>-quality-<i>-<n> for each
    request of a campaign."""
    return sum(isinstance(row, dict) and isinstance(row.get("name"), str)
               and (row["name"] == job["id"] or row["name"].startswith(job["id"] + "-")) for row in reservations)


def answered_invalid(output):
    """Whether the API answered one of a job's requests with a candidate that failed validation. A campaign of
    grana.annotation.rounds keeps a delivery per request under requests/<chunk>/candidate; a job of an earlier
    playground kept its one request's in candidate/."""
    for path in (output / "candidate" / "delivery.json", *sorted(output.glob("requests/*/candidate/delivery.json"))):
        try:
            if read_json(path).get("status") == "semantic_validation_failed":
                return True
        except (OSError, ValueError, AttributeError):
            continue
    return False


def one_line(error, limit=800):
    """A local error for the page: one line, bounded."""
    return " ".join(str(error).split())[:limit] or type(error).__name__


class Refused(ValueError):
    """A well-formed request this state does not allow; nothing was started or written. Its ``code`` names its
    message in MESSAGES, which ``values`` fill in."""

    def __init__(self, code, **values):
        super().__init__(MESSAGES[code].format(**values))
        self.code, self.values = code, values


def note(error):
    """An error as the page shows it: its message code and values when it has them, and its English text."""
    code, values = getattr(error, "code", None), getattr(error, "values", None)
    coded = isinstance(code, str) and isinstance(values, dict)
    return {"code": code if coded else None, "values": values if coded else {}, "text": one_line(error)}


def reason_note(reason):
    """A result's reason as the page shows it."""
    return {"code": REASONS.get(reason), "values": {}, "text": reason}


# Why the label page's preview has no curve yet, by its English (grana.annotation.labels.completion_problems), for the
# page's languages.
PROBLEMS = ((r"frames (\d+)-(\d+) have no subtask", "curve_gap", ("start", "end")),
            (r"touching runs of subtask (.+) must be merged", "curve_merge", ("stage",)),
            (r"subtask (.+) ending at frame (\d+) has no outcome", "curve_outcome", ("stage", "frame")))


def problem_note(text):
    """A preview problem as the page shows it."""
    for pattern, code, names in PROBLEMS:
        match = re.fullmatch(pattern, text)
        if match:
            return {"code": code, "values": dict(zip(names, match.groups())), "text": text}
    return {"code": None, "values": {}, "text": text}


def metrics(aq, events):
    """One compared episode's scores from the evaluators' reports: AQ agreement and recall/precision by class, subtask
    agreement, human events found and the largest boundary error."""
    found = events["events"]["by_class"].values()
    errors = [match[f"{side}_boundary"]["absolute_distance_to_reference_bracket_seconds"]
              for match in events["events"]["matches"] for side in ("start", "end")]
    return {"aq_agreement": aq["accuracy"],
            "classes": {name: {key: value[key] for key in ("recall", "precision", "support_seconds")}
                        for name, value in aq["per_class"].items()},
            "subtask_agreement": events["stage"]["resolved_agreement"],
            "action_quality_status": aq.get("status", "annotated"),
            "events_found": None if events.get("status") == "stages_only" else sum(item["matched_events"] for item in found),
            "events_total": None if events.get("status") == "stages_only" else sum(item["reference_events"] for item in found),
            "boundary_error_seconds": max(errors) if errors else None}


def check_models(models):
    """A workspace's model list: 0 to MAX_MODELS entries with exactly MODEL_FIELDS. Each problem is one line naming
    the entry, repeating no value but an id or a protocol: an api_key_env, for one, may hold a key by mistake."""
    from grana.annotation.api import _validate_base_url, _validate_model

    if not isinstance(models, list) or not 0 <= len(models) <= MAX_MODELS:
        raise ValueError(f"The model list must be a JSON array of 0 to {MAX_MODELS} entries")
    ids = set()
    for number, entry in enumerate(models, 1):
        if (not isinstance(entry, dict) or set(entry) != set(MODEL_FIELDS)
                or any(not isinstance(value, str) for value in entry.values())):
            raise ValueError(f"Model {number} needs exactly these text fields: {', '.join(MODEL_FIELDS)}")
        name = f"Model {number}" + (f" ({entry['id']})" if re.fullmatch(MODEL_ID, entry["id"]) else "")
        try:
            url = urlsplit(entry["base_url"])
        except ValueError:
            url = urlsplit("")
        for wrong, problem in (
                (not re.fullmatch(MODEL_ID, entry["id"]), "id must be 1-32 lowercase letters, digits, '.', '_' or '-'"),
                (entry["id"] in ids, "this id is used twice"),
                (not entry["label"].strip() or len(entry["label"]) > 80 or not entry["label"].isprintable(),
                 "label must be one line of 1-80 characters"),
                (url.scheme != "https" or not url.hostname or url.username or url.password or url.query or url.fragment
                 or not entry["base_url"].isprintable() or " " in entry["base_url"],
                 "base_url must be an HTTPS URL without credentials or query"),
                (not entry["model"].strip() or entry["model"] != entry["model"].strip() or len(entry["model"]) > 200
                 or not entry["model"].isprintable(), "model must be the provider's model name"),
                (not re.fullmatch(KEY_NAME, entry["api_key_env"]),
                 "api_key_env must name an environment variable, not contain its value"),
                (entry["reasoning_effort"] not in ("high", "xhigh"), "reasoning_effort must be high or xhigh"),
                (entry["protocol"] not in PROTOCOLS,
                 f"protocol {entry['protocol']!r} is not supported; the playground sends only \"responses\" requests")):
            if wrong:
                raise ValueError(f"{name}: {problem}")
        ids.add(entry["id"])
        _validate_base_url(entry["base_url"])
        _validate_model(entry["model"])
    return models


def label_state(path):
    """A label file as the home page shows it: none, draft, submitted or unreadable. A submitted file also says how
    its unmarked time came to be reviewed: explicit (the review tick), legacy (an 11-key file, submitted before the
    tick existed) or model_assisted (it started as an imported API draft). An imported file names the model and the
    result folder it was drafted from."""
    from grana.annotation.labels import is_legacy

    if not os.path.lexists(path):
        return {"label": "none"}
    try:
        doc = read_json(path)
    except (OSError, ValueError):
        doc = None
    if not isinstance(doc, dict):
        return {"label": "unreadable"}
    imported = doc.get("machine_import") if isinstance(doc.get("machine_import"), dict) else None
    legacy = is_legacy(doc)
    submitted = doc.get("done") is True and (legacy or doc.get("review_confirmed") is True)
    state = {"label": "submitted" if submitted else "draft", "machine_import": imported is not None,
             "provenance": "model_assisted" if imported else "legacy" if legacy else "explicit"}
    if imported:
        state["proposal"] = {"model": imported.get("model"), "candidate_dir": imported.get("candidate_dir")}
    return state


class Playground:
    def __init__(self, root, *, recover=False, transport=None):
        self.root = Path(root).resolve()
        self.config = read_json(self.root / "run-state.json")
        if self.config.get("schema_version") != SCHEMA:
            raise ValueError("This workspace was made by an earlier playground; choose a fresh --workspace")
        # Every API request goes through this httpx transport; None sends real HTTPS requests.
        self.transport = transport
        # The models fixed when the workspace was created, by id; the first is the page's default.
        self.models = {entry["id"]: entry for entry in check_models(self.config.get("models"))}
        self.settings = read_json(self.root / "settings.json")
        self.task = self.settings["task"]
        self.frames = {row["episode"]: row["frames"] for row in self.config["episodes"]}
        self.lock = threading.RLock()
        self.worker = self.local = None
        self._recipes = {}
        self.path, self.state_path = self.root / "dashboard-jobs.json", self.root / "playground-state.json"
        self.jobs = read_json(self.path) if self.path.exists() else {"version": 2, "jobs": {}}
        if self.jobs.get("version") != 2 or not isinstance(self.jobs.get("jobs"), dict):
            raise ValueError("Invalid playground job state; inspect it before starting.")
        # compares: the latest comparison of each example (key) and model (id).
        self.state = read_json(self.state_path) if self.state_path.exists() else {
            "version": 1, "seeded": {"seeds": [], "held_out": []}, "template": None, "compares": {},
            "imports": {"api": None, "seeded": None}, "task": None}
        if self.state.get("version") != 1:
            raise ValueError("Invalid playground step state; inspect it before starting.")
        # Steps saved by earlier playgrounds: their rules, exports and rules comparisons are dropped, and an example of
        # one seed (an earlier template format) is built again; its comparisons stay, out of date.
        for name in ("mode", "export", "rules"):
            self.state.pop(name, None)
        if self.state.get("template") is not None and "seeds" not in self.state["template"]:
            self.state["template"] = None
        if (self.state.get("task") or {}).get("kind") == "export":
            self.state["task"] = None
        compares = {}
        for key, by_model in self.state.get("compares", {}).items():
            kept = {model: {**compare, "key": compare.get("key", key), "seeds": compare.get("seeds", [compare.get("seed")])}
                    for model, compare in by_model.items() if "seeds" in compare or compare.get("mode") == "examples"}
            if kept:
                compares[key] = kept
        self.state["compares"] = compares
        if recover:
            changed = False
            for job in self.jobs["jobs"].values():
                if job.get("status") == "running":
                    job.update(status="uncertain", phase="interrupted", updated_at=now(),
                               error="The service stopped during this job. Inspect its receipts; automatic retry is disabled.")
                    changed = True
                elif job.get("status") == "queued":
                    # Never sent: a restart must not spend budget without a new click.
                    job.update(status="cancelled", phase="not_started", updated_at=now())
                    changed = True
            task = self.state.get("task")
            if task and task.get("status") == "running":
                message = "The service stopped during this step; run it again."
                task.update(status="error", finished_at=now(), message=message,
                            note={"code": "step_interrupted", "values": {}, "text": message})
                changed = True
            if changed:
                atomic_json(self.path, self.jobs)
                atomic_json(self.state_path, self.state)

    # ---------- recipes, labels and budget ----------

    def recipe(self, mode):
        """A label page's current recipe; stage edits there rewrite it, so it is reread when it changes."""
        from grana.data.builder import load_build_config

        path = self.root / f"{mode}.yaml"
        stat = path.stat()
        key = (stat.st_mtime_ns, stat.st_size)
        if self._recipes.get(mode, (None,))[0] != key:
            self._recipes[mode] = (key, load_build_config(path))
        return self._recipes[mode][1]

    def labels(self, mode):
        return Path(self.recipe(mode)["labels"])

    def label(self, mode, episode):
        from grana.annotation.labels import label_path

        return label_state(label_path(self.labels(mode), episode))

    def _check_recipe(self, mode, settings):
        cfg = self.recipe(mode)
        expected_stages = [{key: row[key] for key in ("label", "prompt")} for row in settings["registry"]]
        if (Path(cfg["source"]).resolve() != Path(self.config["source"]).resolve()
                or not Path(cfg["labels"]).resolve().is_relative_to(self.root)
                or cfg["task"] != settings["task"] or cfg["views"] != settings["views"]
                or cfg["stages"] != expected_stages or cfg["compressed_task_prompt"] != settings["task_instruction"]):
            error = ValueError("Playground recipe differs from the fixed task, source, views or registry")
            error.code = "recipe_changed"
            raise error

    def _reservations(self):
        """The shared budget's reservations and saved cap; none before the first request."""
        path = self.root / "api-budget" / "budget.json"
        if not path.exists():
            return [], self.config["max_requests"]
        value = read_json(path)
        rows, saved = value["reservations"], value["max_requests"]
        if value.get("version") != 1 or not isinstance(rows, list) or type(saved) is not int or saved < 1:
            raise ValueError("Invalid budget")
        return rows, saved

    def _budget(self):
        cap = self.config["max_requests"]
        active = [job for job in self.jobs["jobs"].values() if job["status"] in ("queued", "running")]
        try:
            rows, saved = self._reservations()
            # An admitted job reserves its planned attempts (one for a job of an earlier playground) one request at a
            # time, just before each request: count those it has not reserved yet as pending.
            pending = sum(max(0, job.get("planned_attempts", 1) - spent(job, rows)) for job in active)
            cap = min(cap, saved)
            return {"max_attempts": cap, "used_attempts": len(rows), "pending_attempts": pending,
                    "remaining_attempts": max(0, cap - len(rows) - pending)}
        except (OSError, ValueError, KeyError, TypeError):
            return {"max_attempts": cap, "used_attempts": cap, "pending_attempts": 0, "remaining_attempts": 0,
                    "error": "Saved budget needs inspection; new requests are blocked."}

    # ---------- status ----------

    def _model(self, model):
        if not self.models:
            raise ValueError("No annotation provider configured; create a workspace with an explicit endpoint and model to generate")
        if not isinstance(model, str) or model not in self.models:
            raise ValueError("Name one of this workspace's models: " + ", ".join(self.models))

    @staticmethod
    def _key(route, episode, model, example=None):
        """A job's key in dashboard-jobs.json: one per route, episode, model and, in the seeded route, example."""
        return "-".join(map(str, (route, episode, *([example] if example else []), model)))

    def _version(self):
        """The selected example while its submitted seed labels and source bindings remain current."""
        template = self.state.get("template")
        if template and self.state["seeded"]["seeds"] == template["seeds"] and not self._changed_seeds(template):
            return {"key": template["key"], "seeds": template["seeds"], "template": template["folder"]}
        return None

    def _changed_seeds(self, template):
        """Compare reviewed content, not save timestamps. Frozen snapshots still require exact integrity; current
        labels must also keep their source clock and stage definitions. Read the small label files on each check so
        atomic replacements and same-size edits cannot leave the dashboard using a stale cached value."""
        from grana.annotation.labels import KEYS, label_path, validate_labels

        seeds = template["seeds"]
        folder = self.root / template["folder"]
        try:
            frozen = read_json(folder / "template.json")
            if (frozen["template_id"] != template["template_id"]
                    or frozen["template_id"] != digest({k: v for k, v in frozen.items() if k != "template_id"})
                    or [seed["episode"] for seed in frozen["seeds"]] != seeds):
                return list(seeds)
            changed = []
            fields = KEYS - {"annotator", "updated_at", "revision"}
            for number, episode in enumerate(seeds, 1):
                relative = f"seeds/{number}/labels.json"
                try:
                    raw = (folder / relative).read_bytes()
                    if hashlib.sha256(raw).hexdigest() != frozen["files"][relative]:
                        raise ValueError("Saved example changed")
                    saved = json.loads(raw)
                    current = read_json(label_path(self.labels("seed"), episode))
                    validate_labels(current, frame_count=saved["frame_count"], stages=saved["stages"], require_done=True)
                    if not current["annotator"].strip() or not current["revision"] or not current["updated_at"]:
                        raise ValueError("Seed needs review")
                    if any(current[name] != saved[name] for name in fields):
                        changed.append(episode)
                except (OSError, ValueError, KeyError, TypeError):
                    changed.append(episode)
            return changed
        except (OSError, ValueError, KeyError, TypeError):
            return list(seeds)

    def _job(self, route, episode, model):
        """The episode's job with one model: for the seeded route, the one under the current example. A cancelled job
        never ran."""
        version = self._version() if route == "seeded" else None
        if route == "seeded" and version is None:
            return None
        job = self.jobs["jobs"].get(self._key(route, episode, model, version and version["key"]))
        return None if job is None or job["status"] == "cancelled" else job

    @staticmethod
    def _public(job):
        public = {key: job[key] for key in ("id", "route", "episode", "model", "example", "status", "phase", "created_at",
                                            "updated_at", "error", "error_code", "imported", "with_quality",
                                            "action_quality_status", "stage_requests", "quality_requests")
                  if job.get(key) is not None}
        if job.get("error"):   # why it stopped, as the page words it
            kind = ("interrupted" if job.get("phase") == "interrupted" else "uncertain" if job["status"] == "uncertain"
                    else job["error_code"] if job.get("error_code") in ("invalid_candidate", "recipe_changed")
                    else "not_started" if job.get("phase") == "not_started" else "stopped")
            public["note"] = {"code": "job_" + kind, "values": {"code": job.get("error_code")}, "text": job["error"]}
        return public

    def _row(self, route, episode, context, gate=lambda model: None, *, scored=False):
        """One episode of a generated route: its label file and, per model, its job and what may be clicked; ``gate``
        says why a model may not generate here yet. A ``scored`` (held-out) episode is generated to be compared with
        people, so a draft imported under an earlier example does not count. The row's own status is its label file's, or
        else its furthest job's; a draft also names the model it came from."""
        label = self.label(route, episode)
        proposal = label.pop("proposal", None)
        row = {"episode": episode, "frames": self.frames.get(episode), **label, "models": {}}
        labeled = label["label"] != "none" and not scored
        for model in self.models:
            job = self._job(route, episode, model)
            result = {"status": ({"generated": "generated", "queued": "queued", "running": "running"}.get(job["status"], "failed")
                                 if job else "not_generated")}
            if job:
                result["job"] = self._public(job)
            blocked = None
            if result["status"] == "not_generated" and not labeled:
                blocked = (gate(model) or (None if context["credentials"][model] else "no_credential")
                           or ("budget" if context["budget"]["remaining_attempts"] <= 0 else None))
            elif result["status"] == "failed":
                blocked = "failed_job"
            if blocked:
                result["blocked"] = blocked
            result["can_generate"] = result["status"] == "not_generated" and not labeled and blocked is None
            result["can_import"] = result["status"] == "generated" and label["label"] == "none" and not context["busy"]
            result["can_add_quality"] = bool(result["can_import"] and context["credentials"][model]
                and context["budget"]["remaining_attempts"] > 0 and job
                and job.get("action_quality_status") == "default_normal_as_not_requested" and not job.get("error"))
            row["models"][model] = result
        states = {result["status"] for result in row["models"].values()}
        row["status"] = label["label"] if labeled else next(
            (state for state in ("running", "queued", "generated", "failed") if state in states), "not_generated")
        if proposal:
            made_by = context["results"].get(proposal["candidate_dir"])   # None: a result made elsewhere
            row["draft_model"] = {"id": made_by, "label": self.models[made_by]["label"] if made_by else proposal["model"]}
        return row

    def _context(self):
        task = self.state.get("task")
        return {"budget": self._budget(), "busy": bool(task and task.get("status") == "running"),
                "credentials": {model: bool(os.environ.get(entry["api_key_env"], "").strip())
                                for model, entry in self.models.items()},
                # Which model made each result folder, to name the model of a draft imported from it.
                "results": {str(result_folder(job)): job["model"] for job in self.jobs["jobs"].values()}}

    def status(self):
        with self.lock:
            context = self._context()
            human = [self.label("human", episode)["label"] for episode in self.config["routes"]["human"]]
            routes = {
                "human": {"episodes": len(human), "submitted": human.count("submitted"), "drafts": human.count("draft")},
                "api": {"episodes": [self._row("api", episode, context) for episode in self.config["routes"]["api"]],
                        "import": self.state["imports"]["api"]},
                "seeded": self._seeded(context)}
            return {"status": "ready", "task": self.task, "task_instruction": self.settings["task_instruction"],
                    "recipe": self.config.get("recipe"), "workspace": str(self.root),
                    "episodes": self.config["recipe_episodes"], "budget": context["budget"],
                    "models": [{**entry, "credential": context["credentials"][model]} for model, entry in self.models.items()],
                    "ports": self.config["ports"], "routes": routes, "local_task": self.state.get("task"),
                    "updated_at": now()}

    # ---------- the seeded route: seeds → example → API ----------

    def _seeded(self, context):
        from grana.annotation.templates import MAX_SEEDS

        route = self.config["routes"]["seeded"]
        chosen = self.state["seeded"]
        seeds, held_out = chosen["seeds"], chosen["held_out"]
        labels = {episode: self.label("seed", episode) for episode in route}
        submitted = {episode for episode in route if labels[episode]["label"] == "submitted"}
        roles = {**dict.fromkeys(seeds, "seed"), **dict.fromkeys(held_out, "held_out")}
        labeled = bool(seeds and held_out) and set(seeds + held_out) <= submitted
        available = len(route) >= 2
        chosen_ok = available and labeled and len(seeds) <= MAX_SEEDS
        # The targets are the episodes without submitted labels; a submitted one is a seed, held out or unused.
        targets = [episode for episode in route if episode not in submitted and episode not in roles]
        steps = {"choose": {"state": "blocked" if not available else "done" if chosen_ok else "attention" if roles else "todo",
                            "seeds": seeds, "held_out": held_out, "targets": targets, "max_seeds": MAX_SEEDS,
                            "episodes": [{"episode": episode,
                                          "role": roles.get(episode, "unused" if episode in submitted else "target"),
                                          **{key: value for key, value in labels[episode].items() if key != "proposal"}}
                                         for episode in route]},
                 "example": self._example_step(chosen_ok, seeds)}
        version = self._version()
        ready = chosen_ok and version is not None
        rows = {episode: self._row("seeded", episode, context, lambda model: None if ready else "example", scored=True)
                for episode in held_out} if chosen_ok else {}
        # Per model: its comparison under the current example, else its latest one, which is then out of date.
        compares, shown = self.state["compares"], {}
        for model in self.models:
            made = [by_model[model] for by_model in compares.values() if model in by_model]
            if made:
                shown[model] = (compares.get((version or {}).get("key"), {}).get(model)
                                or max(made, key=lambda compare: compare["created_at"]))
        current = lambda compare: bool(labeled and comparison_current(compare, version, held_out))
        validated = [model for model, compare in shown.items() if current(compare)]
        task = self.state.get("task") or {}
        running = lambda *kinds: task.get("status") == "running" and task.get("kind") in kinds
        steps["validate"] = {"state": "blocked" if not ready else "running" if running("compare") else "done" if validated
                             else "stale" if shown else "todo",
                             "episodes": [rows[episode] for episode in held_out if episode in rows],
                             "comparisons": [self._compare_view(compare, current(compare)) for compare in shown.values()]}
        # A model drafts targets once it has a comparison of the current example and held-out episodes.
        gate = lambda model: None if model in validated else "validate"
        steps["apply"] = {"state": "blocked" if not validated else "todo", "validated": validated,
                          "episodes": [self._row("seeded", episode, context, gate) for episode in targets] if chosen_ok else [],
                          "import": self.state["imports"]["seeded"]}
        if steps["apply"]["state"] == "todo" and running("preview", "confirm") and task.get("route") == "seeded":
            steps["apply"]["state"] = "running"
        elif steps["apply"]["state"] == "todo" and any(row["status"] in ("draft", "submitted") for row in steps["apply"]["episodes"]):
            steps["apply"]["state"] = "done"
        return {"episodes": len(route), "available": available,
                "blocked": None if available else "seeded_episode_count", "steps": steps}

    def _example_step(self, chosen_ok, seeds):
        """The second step: the example built from the seeds, or why building it was refused (a seed's refusal names
        that seed)."""
        template, task = self.state.get("template"), self.state.get("task") or {}
        failed = task.get("kind") == "example" and task.get("status") == "error" and task.get("seeds") == seeds
        changed = self._changed_seeds(template) if template and template["seeds"] == seeds else []
        if task.get("kind") == "example" and task.get("status") == "running":
            state = "running"
        elif not chosen_ok:
            state = "blocked"
        elif failed:
            state = "error"
        elif template is None:
            state = "todo"
        else:
            state = "done" if template["seeds"] == seeds and not changed else "stale"
        error = None
        if state == "error":
            refusal = task.get("note") or {"code": None, "values": {}, "text": task.get("message")}
            error = {"episode": refusal["values"].get("episode"), "note": refusal}
        return {"state": state, "template": template, "error": error, "changed": changed}

    def _compare_view(self, compare, current):
        """Per held-out episode of one model's comparison: AQ recall/precision by class, subtask agreement, human
        events found and the largest boundary error, read from the comparison's evaluator reports."""
        rows = []
        folder = self.root / compare["folder"] if compare.get("folder") else None
        for result in compare.get("results", []):
            row = {key: result.get(key) for key in ("episode", "status", "reason", "summary")}
            if result.get("reason"):
                row["note"] = reason_note(result["reason"])
            if result.get("status") == "compared" and folder is not None:
                try:
                    aq = read_json(folder / f"aq_metrics_{result['episode']:06d}.json")
                    events = read_json(folder / f"events_metrics_{result['episode']:06d}.json")
                except (OSError, ValueError, TypeError):
                    rows.append({**row, "status": "unreadable"})
                    continue
                row.update(metrics(aq, events))
            rows.append(row)
        # Its id, the comparison folder's name, opens one of its episodes in detail.
        return {**{key: compare.get(key) for key in ("folder", "model", "key", "seeds", "held_out", "created_at", "lines",
                                                      "status")},
                "id": Path(compare["folder"]).name if compare.get("folder") else None, "current": current, "episodes": rows}

    def choose(self, seeds, held_out):
        """Step 1: submitted seed-page episodes as 1 to MAX_SEEDS seeds and at least one held-out episode, disjoint."""
        from grana.annotation.templates import MAX_SEEDS

        for name, values in (("seeds", seeds), ("held_out", held_out)):
            if (not isinstance(values, list) or len(values) > 10000
                    or any(type(value) is not int for value in values) or len(set(values)) != len(values)):
                raise ValueError(f"{name} must be a list of distinct episode indices")
        with self.lock:
            route = set(self.config["routes"]["seeded"])
            if len(route) < 2:
                raise Refused("seeded_episode_count")
            if not 1 <= len(seeds) <= MAX_SEEDS:
                raise Refused("seed_count", max=MAX_SEEDS)
            if not held_out:
                raise Refused("no_held_out")
            if set(seeds) & set(held_out):
                raise Refused("seed_and_held_out")
            for episode in seeds + held_out:
                if episode not in route:
                    raise Refused("not_in_route", episode=episode)
                if self.label("seed", episode)["label"] != "submitted":
                    raise Refused("not_submitted", episode=episode)
            self.state["seeded"] = {"seeds": sorted(seeds), "held_out": sorted(held_out)}
            self._save()
            return {"seeds": self.state["seeded"]["seeds"], "held_out": self.state["seeded"]["held_out"]}

    def build_example(self):
        """Step 2: grana.annotation.templates.export_template of the seeds into a new folder under seeded/templates,
        with the playground's criteria and evidence sampling (EVIDENCE: max_images shared by the seeds)."""
        from grana.annotation.templates import MAX_SEEDS

        with self.lock:
            choose = self.status()["routes"]["seeded"]["steps"]["choose"]
            if choose["state"] != "done":
                raise Refused("choose_first", max=MAX_SEEDS)
            seeds, output = choose["seeds"], self._next_folder("templates", "template")
            return self._local_task("example", lambda: self._build_example(seeds, output), seeds=seeds)

    def _build_example(self, seeds, output):
        from grana.annotation.templates import export_template

        result = export_template(self.root / "seed.yaml", seeds, output, criteria=read_json(self.root / "criteria.json"),
                                 **EVIDENCE)
        with self.lock:
            self.state["template"] = {"folder": str(output.relative_to(self.root)), "seeds": seeds,
                                      "seed_images": result["seed_images"], "template_id": result["template_id"],
                                      "key": "examples-" + result["template_id"][:12], "created_at": now()}
            self._save()
        return {"seeds": seeds, "seed_images": result["seed_images"]}

    # ---------- generation ----------

    def _plan(self, episodes, *, stages=True, quality=False, segments=None):
        """The attempts each episode's campaign will reserve (planned_attempts), from the episode's own clock. An
        episode whose clock cannot be read here counts the least a campaign sends, one request per round."""
        from grana.data.lerobot import LeRobotDataset

        try:
            dataset = LeRobotDataset(self.config["source"])
        except (OSError, ValueError, LookupError, TypeError):
            dataset = None
        plans = {}
        for episode in episodes:
            try:
                if dataset is None:
                    raise ValueError("Unreadable source")
                times = [float(value[0] if isinstance(value, (list, tuple)) and len(value) == 1 else value)
                         for value in dataset.rows(episode, ["timestamp"])["timestamp"]]
                # The time edges grana.annotation.source.LeRobotSource.load_episode gives run_episode.
                edges = [*times, times[-1] + (times[-1] - times[-2])]
                plans[episode] = planned_attempts(edges, stages=stages, quality=quality, segments=segments)
            except (OSError, ValueError, LookupError, TypeError, ArithmeticError):
                plans[episode] = stages + quality
        return plans

    def _rows(self, route):
        status = self.status()
        if route == "api":
            return status["routes"]["api"]["episodes"]
        steps = status["routes"]["seeded"]["steps"]
        return steps["validate"]["episodes"] + steps["apply"]["episodes"]

    def start(self, route, episodes, model, with_quality=False):
        """Queue bounded stage-first requests after an explicit click; refused as
        a whole unless every one may run."""
        if type(with_quality) is not bool:
            raise ValueError("with_quality must be a boolean")
        if (route not in ("api", "seeded") or not isinstance(episodes, list) or not 1 <= len(episodes) <= 1000
                or any(type(episode) is not int for episode in episodes) or len(set(episodes)) != len(episodes)):
            raise ValueError("Send route api or seeded and a list of distinct episode indices")
        self._model(model)
        # Each episode's whole campaign must fit: long episodes need several stage requests, AQ at least one more.
        # The clocks are read before taking the lock that status polling waits on.
        plans = self._plan(episodes, quality=with_quality)
        with self.lock:
            rows = {row["episode"]: row for row in self._rows(route)}
            for episode in episodes:
                row = rows.get(episode)
                if row is None:
                    raise Refused("cannot_generate", episode=episode)
                result = row["models"][model]
                if not result["can_generate"]:
                    blocked = result.get("blocked")
                    raise Refused(blocked if blocked in MESSAGES else "has_job", episode=episode,
                                  **({"name": self.models[model]["api_key_env"]} if blocked == "no_credential" else {}))
            budget = self._budget()
            if sum(plans.values()) > budget["remaining_attempts"]:
                raise Refused("over_budget", asked=sum(plans.values()), left=budget["remaining_attempts"])
            # A seeded job runs under the current example: its template folder.
            version = self._version() if route == "seeded" else None
            jobs = []
            for episode in episodes:
                job_id = f"{route}-{episode}-{model}-{uuid.uuid4().hex[:12]}"
                job = {"id": job_id, "route": route, "episode": episode, "model": model, "status": "queued",
                       "phase": "queued", "with_quality": with_quality, "planned_attempts": plans[episode],
                       "created_at": now(), "updated_at": now(), "output": str(self.root / "jobs" / job_id)}
                if version:
                    job.update(example=version["key"], template=version["template"])
                self.jobs["jobs"][self._key(route, episode, model, version and version["key"])] = job
                jobs.append(job)
            atomic_json(self.path, self.jobs)
            if self.worker is None or not self.worker.is_alive():
                self.worker = threading.Thread(target=self._drain, name="playground-api", daemon=False)
                try:
                    self.worker.start()
                except RuntimeError:
                    for job in jobs:
                        job.update(status="error", phase="not_started", updated_at=now(),
                                   error="The local worker could not start. No request was launched.")
                    atomic_json(self.path, self.jobs)
                    self.worker = None
                    raise Refused("worker") from None
            return {"accepted": True, "jobs": [self._public(job) for job in jobs]}

    def _drain(self):
        while True:
            with self.lock:
                queued = sorted((job for job in self.jobs["jobs"].values() if job["status"] == "queued"),
                                key=lambda job: job["created_at"])
                if not queued:
                    self.worker = None
                    return
                job = queued[0]
                self._update(job, status="running", phase="preparing")
            self._run(job)

    def add_quality(self, route, episode, model):
        """An explicit second-round click reuses the completed first round and its budget."""
        self._model(model)
        with self.lock:
            row = next((r for r in self._rows(route) if r["episode"] == episode), None)
            if row is None or not row["models"][model].get("can_add_quality"):
                raise Refused("no_result", episode=episode)
            job = self._job(route, episode, model)
            # Its stage requests are answered from their receipts; AQ is planned on the frozen stage segments.
            try:
                segments = [(segment["start_frame"], segment["end_frame"])
                            for segment in read_json(result_folder(job) / "candidate.json")["stages"]["segments"]]
            except (OSError, ValueError, LookupError, TypeError):
                segments = None
            asked = self._plan([episode], stages=False, quality=True, segments=segments)[episode]
            budget = self._budget()
            if asked > budget["remaining_attempts"]:
                raise Refused("over_budget", asked=asked, left=budget["remaining_attempts"])
            self._update(job, status="queued", phase="queued", with_quality=True,
                         planned_attempts=spent(job, self._reservations()[0]) + asked)
            if self.worker is None or not self.worker.is_alive():
                self.worker = threading.Thread(target=self._drain, name="playground-api", daemon=False)
                self.worker.start()
            return {"accepted": True, "jobs": [self._public(job)]}

    def _update(self, job, **fields):
        with self.lock:
            job.update(**fields, updated_at=now())
            atomic_json(self.path, self.jobs)

    def _job_settings(self, job):
        if job["route"] == "seeded":   # the example's own settings, its stage labels as ids
            return read_json(self.root / job["template"] / "template.json")["settings"]
        criteria = read_json(self.root / "criteria.json")
        if criteria != {key: self.settings["task_policy"][key] for key in ("version", "rules")}:
            raise ValueError("Playground criteria and standalone policy differ")
        return self.settings

    def _run(self, job):
        from grana.annotation.labels import label_path

        output = Path(job["output"])
        try:
            from grana.annotation.rounds import run_episode
            from scripts.annotation_skill import POLICY_DIR
            entry = self.models[job["model"]]
            settings = self._job_settings(job)
            mode, episode = job["route"], job["episode"]
            template = self.root / job["template"] if job.get("template") else None
            self._check_recipe(mode, settings)
            # A label file blocks the standalone route's import; seeded results still score held-out episodes.
            if mode == "api" and os.path.lexists(label_path(self.labels(mode), episode)):
                raise FileExistsError("Saved labels must not be overwritten")
            output.mkdir(parents=True, exist_ok=True)
            if (output / "settings.json").exists() and read_json(output / "settings.json") != settings:
                raise ValueError("Saved settings changed; use a new campaign")
            atomic_json(output / "settings.json", settings)
            # Editors can change their recipe while evidence is being extracted.
            # Recheck immediately before the only potentially paid operation.
            self._check_recipe(mode, settings)
            self._update(job, phase="requesting")
            result = run_episode(self.config["source"], episode, settings, output,
                with_quality=job.get("with_quality", False),
                policy_dir=POLICY_DIR if template is None else template / "policy", template_path=template,
                api_dir=self.root / "api-budget", request_name=job["id"],
                max_requests=self.config["max_requests"], max_attempts=1,
                model=entry["model"], base_url=entry["base_url"], reasoning_effort=entry["reasoning_effort"],
                api_key_env=entry["api_key_env"], transport=self.transport)
            self._update(job, **{**result, "status": "generated", "phase": "complete"})
        except Exception as exc:
            # Do not expose exception text: a provider, source path or environment value may be embedded.
            code = getattr(exc, "code", None)
            if not isinstance(code, str) or not re.fullmatch(r"[A-Za-z0-9_]{1,100}", code):
                code = "saved_labels_exist" if isinstance(exc, FileExistsError) else "local_validation_or_preparation_failed"
            if answered_invalid(output):
                code = "invalid_candidate"
            uncertain = "uncertain" in code or job.get("phase") == "requesting" and code == "local_validation_or_preparation_failed"
            message = ("Request outcome is uncertain. Inspect the saved receipts; this job will not be retried." if uncertain
                       else "Annotation stopped: " + code + ". Saved artifacts remain available for inspection; no automatic retry will run.")
            if code == "invalid_candidate":
                message = "The API answered, but its candidate failed validation. Inspect the job's delivery.json; no automatic retry will run."
            if code == "recipe_changed":
                message = "The editor task, stages, cameras or source differ from the fixed playground settings. No request was sent. Reconcile those definitions before creating a new run; user edits were preserved."
            self._update(job, status="uncertain" if uncertain else "error", phase="stopped", error=message, error_code=code)
            # A failed optional round never hides or replaces the useful first-round result.
            if (output / "candidate/delivery.json").is_file():
                self._update(job, status="generated", phase="quality_stopped", candidate_dir=str(output / "candidate"),
                             bundle_path=str(output / "evidence/bundle.json"),
                             action_quality_status="default_normal_as_not_requested")

    # ---------- imports and other local steps ----------

    def _save(self):
        atomic_json(self.state_path, self.state)

    def _local_task(self, kind, work, **fields):
        """Run ``work`` on the local worker, one step at a time; its result or error is kept for the page."""
        with self.lock:
            task = self.state.get("task")
            if task and task.get("status") == "running":
                raise Refused("busy")
            task = {"id": uuid.uuid4().hex[:12], "kind": kind, "status": "running", "started_at": now(), **fields}
            self.state["task"] = task
            self._save()
            self.local = threading.Thread(target=self._do_local, args=(task, work), name="playground-" + kind, daemon=False)
            self.local.start()
            return {"accepted": True, "task": dict(task)}

    def _do_local(self, task, work):
        try:
            fields = {"status": "done", "result": work()}
        except Exception as error:
            fields = {"status": "error", "message": one_line(error), "note": note(error)}
        with self.lock:
            task.update(fields, finished_at=now())
            self._save()

    def import_episode(self, route, episode, model):
        """Import one model's generated result as a draft with the candidate import core."""
        if route not in ("api", "seeded") or type(episode) is not int:
            raise ValueError("Send route api or seeded and an integer episode")
        self._model(model)
        with self.lock:
            row = next((row for row in self._rows(route) if row["episode"] == episode), None)
            if row is None or not row["models"][model]["can_import"]:
                raise Refused("no_result", episode=episode)
            job = self._job(route, episode, model)
            return self._local_task("import", lambda: self._import(job), route=route, episode=episode, model=model)

    def _import(self, job):
        from grana.annotation.candidate_import import import_candidate
        from scripts.annotation_skill import POLICY_DIR

        output = Path(job["output"])
        # A result asked with an example is checked against that example's frozen policy.
        policy = self.root / job["template"] / "policy" if job.get("template") else POLICY_DIR
        result = import_candidate(self.root / f"{job['route']}.yaml", job["episode"], result_bundle(job),
                                  result_folder(job), failure_quality=FAILURE_QUALITY, policy_dir=policy)
        atomic_json(output / "import.json", result)
        self._update(job, imported=result["labels_file"])
        return {"episode": job["episode"], "review_issues": result["review_issues"]}

    def _generated(self, route, episodes, model):
        """Folders of one model's generated results for these episodes (seeded: under the current example)."""
        return [str(result_folder(job)) for episode in episodes
                if (job := self._job(route, episode, model)) is not None and job["status"] == "generated"]

    def import_all(self, route, action, *, model=None, folder=None, preview=None):
        """Batch import with grana draft-labels: a dry run lists what would be written and why the rest is skipped;
        only a confirmation of that preview writes, from the same results. The route's own results are those of one
        ``model``; results from a ``folder`` name their models themselves."""
        if route not in ("api", "seeded") or action not in ("preview", "confirm", "cancel"):
            raise ValueError("Send route api or seeded and action preview, confirm or cancel")
        with self.lock:
            imports = self.state["imports"]
            if action == "cancel":
                imports[route] = None
                self._save()
                return {"imports": None}
            # Results asked with an example are checked with its frozen policy; others with the checkout's (None).
            policy = None
            if action == "confirm":
                pending = (imports.get(route) or {}).get("preview")
                if not pending or pending["id"] != preview:
                    raise Refused("preview_again")
                sources, model, policy = pending["sources"], pending.get("model"), pending.get("policy_dir")
                if route == "seeded":
                    steps = self.status()["routes"]["seeded"]["steps"]
                    if model not in steps["apply"]["validated"]:
                        raise Refused("compare_before_import")
                    if policy != str(self.root / self._version()["template"] / "policy"):
                        raise Refused("preview_again")
            elif folder is not None:
                if route != "api" or not isinstance(folder, str) or model is not None:
                    raise ValueError("Results from a folder go to the API route's drafts, with the models they name")
                path = Path(folder).expanduser()
                if not path.is_absolute() or not path.exists():
                    raise Refused("no_folder", folder=folder)
                sources = [str(path)]
            else:
                self._model(model)
                steps = self.status()["routes"]["seeded"]["steps"] if route == "seeded" else None
                if steps is not None and model not in steps["apply"]["validated"]:
                    raise Refused("compare_before_import")
                episodes = self.config["routes"]["api"] if route == "api" else steps["choose"]["targets"]
                sources = self._generated(route, episodes, model)
                if not sources:
                    raise Refused("nothing_generated")
                if route == "seeded":
                    policy = str(self.root / self._version()["template"] / "policy")
            kind = "confirm" if action == "confirm" else "preview"
            return self._local_task(kind, lambda: self._draft_labels(route, sources, kind, folder, model, policy),
                                    route=route, model=model)

    def _draft_labels(self, route, sources, kind, folder, model, policy):
        from grana.annotation.drafts import draft_labels

        report = draft_labels(self.root / f"{route}.yaml", sources, dry_run=kind == "preview", failure_quality=FAILURE_QUALITY,
                              policy_dir=policy)
        rows = [{**{key: row[key] for key in ("episode", "status", "from", "reason", "review_issues", "labels_file", "model")
                    if key in row}, **({"note": reason_note(row["reason"])} if row.get("reason") else {})}
                for row in report["results"]]
        # Its one hint: nothing drafted, every episode has labels.
        hints = [{"code": "all_labeled" if all(row["status"] == "skipped" for row in rows) else None, "values": {},
                  "text": hint} for hint in report["hints"]]
        summary = {"id": uuid.uuid4().hex[:12], "created_at": now(), "folder": folder, "model": model,
                   "counts": report["counts"], "results": rows, "hints": hints, "report": report.get("report")}
        with self.lock:
            self.state["imports"][route] = ({"preview": {**summary, "sources": sources, "policy_dir": policy}, "result": None}
                                            if kind == "preview" else {"preview": None, "result": summary})
            self._save()
        return {"counts": report["counts"]}

    def _next_folder(self, parent, name):
        parent = self.root / "seeded" / parent
        parent.mkdir(parents=True, exist_ok=True)
        numbers = [int(match.group(1)) for path in parent.iterdir() if (match := re.fullmatch(name + r"-(\d+)", path.name))]
        return parent / f"{name}-{max(numbers, default=0) + 1:03d}"

    def compare(self, model):
        """Validate: scripts/seed_examples.py compare of one model's held-out results with their submitted seed-page
        labels. It replaces that model's earlier comparison under the same example; other models' are kept."""
        self._model(model)
        with self.lock:
            steps = self.status()["routes"]["seeded"]["steps"]
            version = self._version()
            held_out = steps["choose"]["held_out"]
            sources = self._generated("seeded", held_out, model) if steps["validate"]["state"] != "blocked" else []
            if not sources:
                raise Refused("generate_held_out")
            output = self._next_folder("compare", "compare")
            return self._local_task("compare", lambda: self._compare(sources, output, version, held_out, model),
                                    model=model)

    def _compare(self, sources, output, version, held_out, model):
        from scripts.seed_examples import compare

        lines = []
        # Results asked with an example are scored with its frozen policy, as grana draft-labels imports them.
        status = compare(self.root / "seed.yaml", sources, output, failure_quality=FAILURE_QUALITY,
                         policy_dir=self.root / version["template"] / "policy", log=lines.append)
        results = read_json(output / "report.json")["results"] if output.exists() else []
        with self.lock:
            self.state["compares"].setdefault(version["key"], {})[model] = {
                "folder": str(output.relative_to(self.root)) if output.exists() else None, "model": model,
                "key": version["key"], "seeds": version["seeds"], "held_out": held_out, "status": status,
                "lines": lines, "results": results, "created_at": now()}
            self._save()
        return {"status": status, "compared": sum(row["status"] == "compared" for row in results)}

    # ---------- one compared episode in detail ----------

    def compare_detail(self, name, episode):
        """One held-out episode of a listed comparison, for its detail view, from what the comparison stored: the
        result's candidate and bundle (checked against the report's hashes), the human label file it compared with
        (flagged when it changed since) and the evaluators' reports.

        Subtask runs, quality spans and cuts are in seconds on the episode's time axis (frame / fps, as the label page
        shows them); each human cut has the nearest model cut and their difference. The model's labels are the draft
        its import would write. Both progress curves come from the label page's own preview, the target grana
        build-cache builds, with the model's unset outcomes counted as completed."""
        from grana.annotation.candidate_import import _draft, display_candidate
        from grana.annotation.label_server import LabelProject
        from grana.annotation.stage_settings import revision

        if not isinstance(name, str) or not re.fullmatch(r"compare-\d{3,9}", name) or type(episode) is not int:
            raise ValueError("Send compare (a comparison folder's name) and an integer episode")
        with self.lock:
            compare = next((compare for by_model in self.state["compares"].values() for compare in by_model.values()
                            if compare.get("folder") == f"seeded/compare/{name}"), None)
            cfg, imported = self.recipe("seed"), self.label("seeded", episode)["label"] != "none"
        if compare is None:
            raise Refused("no_comparison")
        folder = self.root / compare["folder"]
        row = next((row for row in read_json(folder / "report.json")["results"]
                    if row.get("episode") == episode and row.get("status") == "compared"), None)
        if row is None:
            raise Refused("not_compared", episode=episode)
        aq, events = (read_json(folder / f"{kind}_metrics_{episode:06d}.json") for kind in ("aq", "events"))
        result = Path(row["from"])
        candidate, delivery = read_json(result / "candidate.json"), read_json(result / "delivery.json")
        # The playground's job folders keep the evidence the result was asked with beside it.
        bundle = read_json(delivery.get("bundle_path", result.parent / "evidence" / "bundle.json"))
        if (digest(candidate), digest(bundle)) != (row["candidate_sha256"], delivery.get("bundle_sha256")):
            raise Refused("result_changed", episode=episode)
        raw = Path(row["labels_file"]).read_bytes()
        human = json.loads(raw)
        project = LabelProject(cfg, self.root / "seed.yaml")
        try:
            fps, source = project.fps, project.dataset.load_episode(episode)
            draft, _, issues = _draft(display_candidate(bundle, candidate), bundle["context"], source, cfg)
            unset = sum(run["outcome"] is None for run in draft["subtasks"])
            assumed = {**draft, "subtasks": [{**run, "outcome": run["outcome"] or "high"} for run in draft["subtasks"]]}
            curves = {}
            for side, doc in (("human", human), ("model", assumed)):
                if side == "model" and bundle.get("mode") in ("stages", "quality"):
                    curves[side] = {"problems": ["Outcomes were not requested; no model-derived training curve is available."]}
                    continue
                try:
                    curves[side] = project.preview(episode, {"subtasks": doc["subtasks"], "quality": doc["quality"],
                                                             "project_revision": revision(project.cfg)})
                except ValueError as error:
                    curves[side] = {"problems": [str(error)]}
        finally:
            project.close()
        stages = [stage["label"] for stage in cfg["stages"]]
        seconds = lambda frame: frame / fps
        runs = lambda doc: [{"start": seconds(run["start_frame"]), "end": seconds(run["end_frame"]), "stage": run["stage"],
                             "index": stages.index(run["stage"]), "outcome": run["outcome"]} for run in doc["subtasks"]]
        spans = lambda rows: [{"start": seconds(span["start_frame"]), "end": seconds(span["end_frame"]), "label": span["label"]}
                              for span in rows]
        count = source.frame_count
        model_cuts = sorted({run["start_frame"] for run in draft["subtasks"] if run["start_frame"] > 0}
                            | {run["end_frame"] for run in draft["subtasks"] if run["end_frame"] < count})
        cuts = []
        for before, after in zip(human["subtasks"], human["subtasks"][1:]):
            cut = after["start_frame"]
            nearest = min(model_cuts, key=lambda frame: (abs(frame - cut), frame), default=None)
            cuts.append({"seconds": seconds(cut), "from": before["stage"], "to": after["stage"],
                         "model_seconds": None if nearest is None else seconds(nearest),
                         "delta_seconds": None if nearest is None else seconds(nearest - cut)})
        # One time axis for both curves: each preview frame's middle, as the label page plots it.
        frames = sorted({*curves["human"].get("frames", []), *curves["model"].get("frames", [])})
        values = {side: dict(zip(curve.get("frames", []), curve.get("progress", []))) for side, curve in curves.items()}
        return {"compare": name, "episode": episode, "model_id": compare["model"], "seeds": compare.get("seeds"),
                "fps": fps, "duration_seconds": seconds(count),
                "labels_changed": hashlib.sha256(raw).hexdigest() != row["labels_sha256"],
                "human": {"runs": runs(human), "quality": spans(human["quality"])},
                "model": {"runs": runs(draft), "quality": spans(draft["quality"]),
                          "unknown_quality": [{"start": seconds(issue["start_frame"]), "end": seconds(issue["end_frame"])}
                                              for issue in issues if issue["axis"] == "quality" and issue["value"] == "unknown"]},
                "cuts": cuts,
                "curves": {"seconds": [(frame + 0.5) / fps for frame in frames],
                           "human": [values["human"].get(frame) for frame in frames],
                           "model": [values["model"].get(frame) for frame in frames],
                           "problems": {side: [problem_note(text) for text in curve.get("problems", [])]
                                        for side, curve in curves.items()},
                           "assumed_completed": unset},
                "metrics": metrics(aq, events), "seeded_draft": imported}

    def idle(self, timeout=None):
        """Wait for the running request and local step (tests and scripted runs)."""
        for thread in (self.worker, self.local):
            if thread is not None:
                thread.join(timeout)


if __name__ == "__main__":
    # Read-only diagnostic; only the home page's POST requests start work.
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    print(json.dumps(Playground(parser.parse_args().root).status(), indent=2))
