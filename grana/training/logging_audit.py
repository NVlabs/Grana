# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify compact history against complete, downloaded evaluation artifacts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def audit_evaluation_artifacts(api, run, receipt, output, *, expected_steps):
    """New-format runs must retain every evaluation in a cloud artifact.

    A checksum alone is insufficient: first compare the local artifact to the
    actual evaluator output and completed event, then download the cloud bytes.
    Historical scalar aliases are deliberately not an alternative contract.
    """
    from .evaluation_reporting import (EVALUATION_PURPOSE_PREFIXES, SUPPORTED_COMPACT_EVALUATION_SCHEMAS,
                                       compact_evaluation_metrics)

    output = Path(output).resolve()
    schema = receipt.get("evaluation_logging_schema")
    _require(schema in SUPPORTED_COMPACT_EVALUATION_SCHEMAS,
             "Requires the compact evaluation logging schema")
    _require(receipt.get("history_step_schema") == "grana-optimizer-step-history-v1",
             "Requires explicit optimizer-step history")
    entries = receipt.get("evaluation_artifacts", [])
    identities = [(entry["purpose"], entry["optimizer_steps"]) for entry in entries]
    _require(len(identities) == len(set(identities)), "Duplicate evaluation artifact receipt")
    _require(sorted(step for purpose, step in identities if purpose == "primary") == sorted(expected_steps),
             "Missing primary evaluation artifact")
    rows = list(run.scan_history(page_size=1000))
    steps = [row.get("optimizer_steps") for row in rows]
    _require(all(isinstance(step, int) and not isinstance(step, bool) and step >= 0 for step in steps),
             "History row is missing a valid optimizer step")
    _require(steps == sorted(set(steps)), "Optimizer steps must have one ordered history row each")
    _require(all(row.get("_step") == row["optimizer_steps"] for row in rows),
             "W&B media/history step differs from optimizer step")
    events_path = output / "evaluation_metrics.jsonl"
    events = {row["optimizer_steps"]: row for row in
              (json.loads(line) for line in events_path.read_text().splitlines())} if events_path.exists() else {}
    verified = []
    for entry in entries:
        purpose, step = entry["purpose"], entry["optimizer_steps"]
        _require(purpose in ("primary", "renders", "dense"), "Unknown evaluation artifact purpose")
        local = Path(entry["local_path"]).resolve()
        _require(local.is_relative_to(output / "evaluation_artifacts"), "Evaluation artifact escapes run directory")
        payload = local.read_bytes()
        checksum = hashlib.sha256(payload).hexdigest()
        _require(checksum == entry["sha256"], "Local evaluation artifact checksum differs")
        envelope = json.loads(payload)
        _require(envelope.get("schema") == "grana-evaluation-log-v1"
                 and envelope.get("optimizer_steps") == step and envelope.get("purpose") == purpose,
                 "Evaluation artifact identity differs")
        prefix = {"primary": "evaluation", "renders": "render-evaluation", "dense": "deploy-dense-evaluation"}[purpose]
        evaluation = json.loads((output / f"{prefix}-{step:06d}.json").read_text())
        _require(envelope.get("evaluation") == evaluation, "Artifact lost or changed full evaluation records")
        _require(envelope.get("diagnostics") == evaluation.get("diagnostics"), "Artifact diagnostics differ")
        metrics = envelope["metrics"]
        _require(all(metrics.get(key) == value for key, value in evaluation["metrics"].items()),
                 "Artifact lost or changed detailed evaluation metrics")
        if purpose == "primary" and step in events:
            _require(metrics == {key: value for key, value in events[step].items() if key != "optimizer_steps"},
                     "Artifact differs from completed evaluation timing/event")
        projected = {EVALUATION_PURPOSE_PREFIXES[purpose] + key: value
                     for key, value in compact_evaluation_metrics(metrics, evaluation=evaluation, schema=schema).items()}
        _require(sorted(projected) == sorted(entry["logged_scalar_keys"]), "Compact history key receipt differs")
        from .campaign_report import close
        for key, value in projected.items():
            matching = [row[key] for row in rows if row.get("optimizer_steps") == step and row.get(key) is not None]
            _require(len(matching) == 1 and close(matching[0], value), f"Compact evaluation history differs: {key}")
        reference = entry["artifact_reference"]
        _require(reference.startswith(f"{run.entity}/{run.project}/grana-{run.id}-evaluation-{purpose}:"),
                 "Evaluation artifact belongs to a different run")
        with tempfile.TemporaryDirectory(prefix="grana-evaluation-audit-") as temporary:
            downloaded = Path(api.artifact(reference).download(root=temporary)) / "evaluation.json"
            _require(hashlib.sha256(downloaded.read_bytes()).hexdigest() == checksum,
                     "Downloaded evaluation artifact checksum differs")
        verified.append({"optimizer_steps": step, "purpose": purpose, "artifact_reference": reference,
                         "sha256": checksum, "full_metric_count": len(evaluation["metrics"]),
                         "record_count": len(evaluation.get("records", [])), "compact_scalar_count": len(projected)})
    return verified
