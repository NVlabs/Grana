# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Persist browser stage edits as new recipes without reinterpreting saved labels."""
from contextlib import contextmanager
from copy import deepcopy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import threading

import yaml

from grana.data.builder import load_build_config, validate_build_config
from grana.data.targets.composition import PROGRESS_METHOD as COMPOSITION_METHOD
from grana.modeling.prompts import text_field
from .labels import validate_labels, write_labels


class StageConflict(ValueError):
    """The browser or server is still using a previous project definition."""


def revision(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class StageSettings:
    def __init__(self, cfg, config_path=None):
        self.path = Path(config_path).expanduser().resolve() if config_path is not None else None
        self.lock = threading.RLock()
        self.file_digest = None
        if self.path is not None:
            if load_build_config(self.path) != cfg:
                raise ValueError("The labeling recipe changed before the server started; restart with its current contents")
            self.file_digest = hashlib.sha256(self.path.read_bytes()).hexdigest()
        parent = Path(cfg["labels"]).parent
        # A newly selected version lives in <root>/.stage-versions/<version>/labels.
        self.history = parent.parent if parent.parent.name == ".stage-versions" else parent / ".stage-versions"

    @contextmanager
    def writing(self):
        """Serialize recipe changes with saves, including other local server processes."""
        with self.lock:
            if self.path is None:
                yield
                return
            self.history.mkdir(parents=True, exist_ok=True)
            key = hashlib.sha256(str(self.path).encode()).hexdigest()[:20]
            with (self.history / f".{key}.lock").open("a") as stream:
                fcntl.flock(stream, fcntl.LOCK_EX)
                try:
                    if hashlib.sha256(self.path.read_bytes()).hexdigest() != self.file_digest:
                        raise StageConflict("The recipe changed in another process. Restart this label server to load it; your saved labels are preserved.")
                    yield
                finally:
                    fcntl.flock(stream, fcntl.LOCK_UN)

    def check(self, cfg, body):
        body = dict(body)
        supplied = body.pop("project_revision", None)
        # A blank label document also has a null revision in a different recipe.
        # Require the loaded project identity even on the very first save.
        if supplied != revision(cfg):
            raise StageConflict("Sub-stage definitions changed. Your edits are still here; export unsaved labels, then reload the page before editing the new version.")
        return body

    def update(self, cfg, body, load_documents):
        if self.path is None:
            raise ValueError("Stage editing needs the recipe path; launch this page with grana label --config")
        if not isinstance(body, dict) or set(body) != {"base_revision", "stages"}:
            raise ValueError("Send base_revision and the ordered stages list")
        stages = body["stages"]
        if (not isinstance(stages, list) or not 1 <= len(stages) <= 26
                or any(not isinstance(stage, dict) or set(stage) != {"label", "prompt"}
                       or any(not isinstance(stage[key], str) for key in ("label", "prompt")) for stage in stages)):
            raise ValueError("Define 1–26 sub-stages, each with a text label and description")
        with self.writing():
            if body["base_revision"] != revision(cfg):
                raise StageConflict("Sub-stage definitions changed in another tab. Reload the definitions before applying your edits.")
            # Compare the trimmed text the recipe keeps: surrounding spaces change no stage.
            stages = [{key: text_field(stage[key], f"stage {key}") for key in ("label", "prompt")} for stage in stages]
            if stages == cfg["stages"]:
                return cfg, None, False
            candidate = deepcopy(cfg)
            candidate["stages"] = deepcopy(stages)
            # Text changes can change stage meaning, even when IDs stay put. Only the trimmed no-op above
            # preserves review/calibration; keep every interval but require review under the new definitions.
            reopened = True
            progress = candidate["progress"]
            if reopened and progress["method"] == COMPOSITION_METHOD and isinstance(stages, list):
                previous = {stage["label"]: params for stage, params in zip(cfg["stages"], progress["stage_parameters"])}
                progress["stage_parameters"] = [deepcopy(previous.get(stage.get("label"), progress["stage_parameters"][-1]))
                                                for stage in stages if isinstance(stage, dict)]
                progress["parameter_status"] = "example"
                progress["parameter_source"] = "Stage editor: inherited preview parameters; new stage definitions need task-specific review before training."
            # Validate all user input before creating a version or touching the active recipe.
            candidate = validate_build_config(candidate)
            labels = [stage["label"] for stage in candidate["stages"]]
            documents = []
            # Read under the same cross-process lock as saves. Stage IDs, rather
            # than their display positions, keep intervals attached when reordered.
            for original in load_documents():
                removed = {run["stage"] for run in original["subtasks"]} - set(labels)
                if removed:
                    names = ", ".join(f"S{i + 1}" for i, stage in enumerate(cfg["stages"]) if stage["label"] in removed)
                    raise ValueError(f"Cannot remove stages {names}: episode {original['episode_index']} still has intervals assigned to them. Reassign or remove those intervals first.")
                doc = deepcopy(original)
                doc.update(stages=labels, stage_registry=deepcopy(candidate["stages"]))
                if reopened:
                    doc.update(done=False, review_confirmed=False)
                documents.append(validate_labels(doc, frame_count=doc["frame_count"], stages=labels))
            version = Path(tempfile.mkdtemp(prefix="version-", dir=self.history))
            temporary = None
            try:
                candidate.update(labels=str(version / "labels"), output=str(version / "cache"))
                candidate = validate_build_config(candidate)
                (version / "previous-recipe.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
                content = yaml.safe_dump(candidate, allow_unicode=True, sort_keys=False)
                (version / "recipe.yaml").write_text(content)
                (version / "labels").mkdir()
                # Commit the recipe only after every migrated draft is durable.
                # A failed copy leaves the active recipe and all old files intact.
                for doc in documents:
                    write_labels(candidate["labels"], doc)
                descriptor, temporary = tempfile.mkstemp(prefix=f".{self.path.name}.", dir=self.path.parent)
                with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                    stream.write(content)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.chmod(temporary, self.path.stat().st_mode & 0o777)
                os.replace(temporary, self.path)
            except Exception:
                shutil.rmtree(version)
                raise
            finally:
                if temporary is not None and os.path.exists(temporary):
                    os.unlink(temporary)
            self.file_digest = hashlib.sha256(content.encode()).hexdigest()
            return candidate, version, reopened
