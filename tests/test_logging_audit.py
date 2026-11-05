# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compact curves must not weaken verification of full cloud diagnostics."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from grana.training.evaluation_reporting import compact_evaluation_metrics
from grana.training.logging_audit import audit_evaluation_artifacts


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture(params=["grana-compact-eval-v2", "grana-compact-eval-v3"])
def logged(tmp_path, request):
    entries, rows, uploads, events = [], [], {}, []
    for step in (2, 4):
        metrics = {"accumulated/overall/raw_rmse": .4, "accumulated/overall/normalized_rmse": .04,
                   "diagnostics/overall/D/incorrect_C/rmse": .8}
        evaluation = {"optimizer_steps": step, "metrics": metrics,
                      "records": [{"cache": "/cache", "task": "task", "trajectory_id": "traj", "poi": 0,
                                   "prediction": step, "role": "D"}],
                      "diagnostics": {"full": [step]}, "plan_sha256": "cohort"}
        event = {"optimizer_steps": step, **metrics, "elapsed_seconds": 2.,
                 "evaluation_and_persistence_seconds": 3.}
        events.append(event)
        envelope = {"schema": "grana-evaluation-log-v1", "optimizer_steps": step, "purpose": "primary",
                    "metrics": {key: value for key, value in event.items() if key != "optimizer_steps"},
                    "evaluation": evaluation, "diagnostics": evaluation["diagnostics"]}
        write(tmp_path / f"evaluation-{step:06d}.json", evaluation)
        path = tmp_path / "evaluation_artifacts" / f"primary-{step:06d}.json"
        write(path, envelope)
        compact = {"eval/" + key: value for key, value in
                   compact_evaluation_metrics(envelope["metrics"], evaluation=evaluation, schema=request.param).items()}
        reference = f"team/project/grana-run-evaluation-primary:step-{step:06d}"
        entries.append({"purpose": "primary", "optimizer_steps": step, "local_path": str(path),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "artifact_reference": reference, "logged_scalar_keys": sorted(compact)})
        uploads[reference] = path.read_bytes()
        rows.append({"_step": step, "optimizer_steps": step, **compact})
    (tmp_path / "evaluation_metrics.jsonl").write_text("\n".join(json.dumps(event) for event in events))

    class Artifact:
        def __init__(self, reference):
            self.reference = reference

        def download(self, root):
            (Path(root) / "evaluation.json").write_bytes(uploads[self.reference])
            return root

    return SimpleNamespace(output=tmp_path, entries=entries, rows=rows, uploads=uploads,
                           api=SimpleNamespace(artifact=Artifact),
                           run=SimpleNamespace(id="run", entity="team", project="project",
                                               scan_history=lambda **kwargs: iter(rows)),
                           receipt={"evaluation_logging_schema": request.param,
                                    "history_step_schema": "grana-optimizer-step-history-v1",
                                    "evaluation_artifacts": entries})


def audit(logged):
    return audit_evaluation_artifacts(logged.api, logged.run, logged.receipt, logged.output, expected_steps=[2, 4])


def test_downloaded_full_diagnostics_verified_without_redundant_scalar_history(logged):
    result = audit(logged)
    assert [item["optimizer_steps"] for item in result] == [2, 4]
    assert all(item["full_metric_count"] == 3 and item["record_count"] == 1 for item in result)
    assert all("eval/diagnostics/overall/D/incorrect_C/rmse" not in row for row in logged.rows)


@pytest.mark.parametrize("damage", ["schema", "missing", "duplicate", "cloud_bytes", "history", "duplicate_history",
                                    "records", "detail", "event_time", "other_run", "step_schema", "internal_step",
                                    "missing_optimizer_step"])
def test_compact_audit_rejects_missing_or_altered_evidence(logged, damage):
    entry = logged.entries[0]
    path = Path(entry["local_path"])
    if damage == "schema":
        logged.receipt.pop("evaluation_logging_schema")
    elif damage == "step_schema":
        logged.receipt.pop("history_step_schema")
    elif damage == "internal_step":
        logged.rows[0]["_step"] = 3
    elif damage == "missing_optimizer_step":
        logged.rows[0].pop("optimizer_steps")
    elif damage == "missing":
        logged.entries.pop()
    elif damage == "duplicate":
        logged.entries.append(deepcopy(entry))
    elif damage == "cloud_bytes":
        logged.uploads[entry["artifact_reference"]] = b"{}"
    elif damage == "history":
        key = next(key for key in logged.rows[0] if key.endswith("raw_rmse"))
        logged.rows[0][key] = .9
    elif damage == "duplicate_history":
        logged.rows.append(dict(logged.rows[0]))
    elif damage == "other_run":
        entry["artifact_reference"] = "team/project/grana-other-evaluation-primary:step-000002"
    else:
        envelope = json.loads(path.read_text())
        if damage == "records":
            envelope["evaluation"]["records"] = []
        elif damage == "detail":
            del envelope["metrics"]["diagnostics/overall/D/incorrect_C/rmse"]
        elif damage == "event_time":
            envelope["metrics"]["evaluation_and_persistence_seconds"] = 4.
        write(path, envelope)
        # A new checksum cannot legitimize loss of actual evaluator evidence.
        entry["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        logged.uploads[entry["artifact_reference"]] = path.read_bytes()
    with pytest.raises(ValueError):
        audit(logged)
