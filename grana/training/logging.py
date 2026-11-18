# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rank-zero W&B history and explicit checkpoint manifests, without hidden fallback."""
from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
from typing import Any

from .evaluation_reporting import COMPACT_EVALUATION_SCHEMA


def environment_summary() -> dict:
    """Record runtime versions, never credentials or the process environment."""
    from .runtime_fingerprint import capture_runtime_fingerprint

    runtime = capture_runtime_fingerprint()
    packages = {name: package.get("version") for name, package in runtime["packages"].items()}
    import torch

    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": packages, "cuda_runtime": torch.version.cuda,
            "runtime_fingerprint": runtime,
            "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}


class RunLogger:
    """Use as a context manager around the complete worker lifecycle.

    Only rank zero initializes W&B. Disabled mode does not import its SDK.
    Online initialization, upload, and finish failures propagate to the caller.
    Checkpoint artifacts contain metadata only unless explicitly opted in.
    History payloads accumulate locally until commit_step publishes one row at
    that optimizer step, including media. Finish flushes a trailing measured
    payload even on failure; it never creates a row or retries a failed write.
    """

    def __init__(self, config: dict, *, rank: int = 0, world_size: int = 1):
        self.config = config
        self.settings = config["logging"]
        self.rank, self.world_size = rank, world_size
        self.mode = self.settings["mode"]
        if self.mode not in ("disabled", "offline", "online"):
            raise ValueError("logging.mode must be disabled, offline, or online")
        self.output = Path(config["training"]["output_dir"]).expanduser().resolve()
        self.run = None
        self._sdk = None
        self._finished = False
        self._last_step = 0
        self._pending_step: int | None = None
        self._pending_history: dict[str, Any] = {}
        self._committed_steps: list[int] = []
        self._history_error: BaseException | None = None
        self._artifacts = []
        self._checkpoint_records = []
        self._render_records = []
        self._dense_records = []
        self._replay_dense_records = []
        self._evaluation_records = []
        self._receipt: dict[str, Any] = {"mode": self.mode, "rank": rank, "world_size": world_size,
                                         "evaluation_logging_schema": COMPACT_EVALUATION_SCHEMA,
                                         "history_step_schema": "grana-optimizer-step-history-v1",
                                         "committed_optimizer_steps": [],
                                         "status": "disabled" if self.mode == "disabled" else "pending"}

    @property
    def enabled(self) -> bool:
        return self.rank == 0 and self.mode != "disabled"

    def _write_receipt(self) -> None:
        if self.rank == 0:
            self.output.mkdir(parents=True, exist_ok=True)
            (self.output / "wandb_run.json").write_text(json.dumps(self._receipt, indent=2) + "\n")

    def __enter__(self):
        if self.rank != 0:
            return self
        self.output.mkdir(parents=True, exist_ok=True)
        try:
            # Persist the actual worker environment even when cloud logging is
            # disabled. Reuse this exact payload online, rather than recapture
            # a potentially different environment after SDK initialization.
            environment = environment_summary()
            environment_path = self.output / "environment.json"
            encoded = (json.dumps(environment, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
            temporary = environment_path.with_suffix(".json.tmp")
            temporary.write_bytes(encoded)
            temporary.replace(environment_path)
            self._receipt.update(environment_path=str(environment_path),
                                 environment_sha256=hashlib.sha256(encoded).hexdigest(),
                                 runtime_fingerprint_sha256=environment.get("runtime_fingerprint", {}).get("sha256"))
            if not self.enabled:
                self._write_receipt()
                return self
            import wandb

            self._sdk = wandb
            self.run = wandb.init(
                mode=self.mode, project=self.settings["project"], entity=self.settings["entity"],
                group=self.settings["group"], name=self.settings["name"], tags=self.settings["tags"],
                job_type=self.settings["job_type"] or self.config["training"]["stage"],
                config=self.config, dir=str(self.output), resume="never", reinit="create_new",
                force=self.mode == "online", save_code=False,
                settings=wandb.Settings(console="off", disable_git=True, disable_code=True,
                                        symlink=False, allow_media_symlink=False,
                                        init_timeout=90, finish_timeout=180, finish_timeout_raises=True),
            )
            self.run.define_metric("optimizer_steps")
            self.run.define_metric("*", step_metric="optimizer_steps")
            self.run.summary.update({"status": "running", "world_size": self.world_size,
                                     "evaluation_logging_schema": COMPACT_EVALUATION_SCHEMA,
                                     "history_step_schema": "grana-optimizer-step-history-v1",
                                     "environment": environment})
            self._receipt.update(status="running", id=self.run.id, entity=self.run.entity,
                                 project=self.run.project, url=self.run.url if self.mode == "online" else None,
                                 local_directory=self.run.dir)
            self._write_receipt()
        except BaseException as error:
            self._receipt.update(status="failed", failure_type=type(error).__name__)
            self._write_receipt()
            if self.run is not None:
                self.finish(exit_code=1)
            raise
        return self

    def start(self, *, model_summary: dict, data_summary: dict, initialization: dict | None = None) -> None:
        if self.enabled:
            self.run.summary.update({"model": model_summary, "data": data_summary,
                                     "roles": self.config["roles"], "loss_configuration": self.config["loss"],
                                     "initialization": initialization})

    def _validate_history_step(self, step: int) -> None:
        if self._finished:
            raise ValueError("Cannot log history after the run has finished")
        if self._history_error is not None:
            raise RuntimeError("A history write failed; refusing to retry or append history") from self._history_error
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError("History requires a nonnegative integer optimizer step")
        if self._committed_steps and step <= self._committed_steps[-1]:
            raise ValueError("History optimizer step was already committed or is in the past")
        if self._pending_step is not None and step != self._pending_step:
            raise ValueError(f"Commit pending optimizer step {self._pending_step} before logging step {step}")

    def _validate_history_row(self, row: dict, *, step: int) -> None:
        self._validate_history_step(step)
        if row.get("optimizer_steps") != step:
            raise ValueError("History payload optimizer step differs from its explicit step")
        duplicates = (self._pending_history.keys() & row.keys()) - {"optimizer_steps"}
        if duplicates:
            raise ValueError(f"Duplicate history keys for optimizer step {step}: {sorted(duplicates)}")

    def _queue_history(self, row: dict, *, step: int) -> None:
        self._validate_history_row(row, step=step)
        self._pending_history.update(row)
        self._pending_step = step
        self._receipt["pending_optimizer_step"] = step
        self._write_receipt()

    def commit_step(self, *, step: int) -> None:
        """Publish one complete optimizer-step row, once, at W&B's real step.

        Call after every scalar and media producer for this step has finished.
        Explicit step plus commit=True aligns media selectors as well as scalar
        axes. A failed SDK call is not retried because remote acceptance may be
        unknown. Empty, repeated, past, and out-of-order commits are errors.
        """
        if not self.enabled:
            return
        self._validate_history_step(step)
        if self._pending_step is None:
            raise ValueError("Cannot commit an optimizer step without a history payload")
        try:
            self.run.log(dict(self._pending_history), step=step, commit=True)
        except BaseException as error:
            self._history_error = error
            self._receipt.update(history_write_failed=True, history_write_failure_type=type(error).__name__)
            self._write_receipt()
            raise
        self._committed_steps.append(step)
        self._pending_history.clear()
        self._pending_step = None
        self._receipt.update(committed_optimizer_steps=list(self._committed_steps), pending_optimizer_step=None)
        self.run.summary["committed_optimizer_steps"] = list(self._committed_steps)
        self._write_receipt()

    def log_step(self, metrics: dict, *, step: int) -> None:
        if not self.enabled:
            return
        if isinstance(step, bool) or not isinstance(step, int) or step <= self._last_step:
            raise ValueError("W&B optimizer steps must increase strictly")
        row = dict(metrics)
        row["optimizer_steps"] = step
        for role, weight in self.config["loss"]["role_weights"].items():
            key = f"role_{role}_loss"
            if key in row:
                row[f"role_{role}_weighted_loss"] = row[key] * weight
        self._queue_history(row, step=step)
        self._last_step = step
        self._receipt["last_optimizer_step"] = step
        self._write_receipt()

    def _log_json_artifact(self, payload: dict, *, name: str, kind: str,
                           file_name: str, local_path: Path, metadata: dict, aliases: list[str]) -> dict:
        encoded = (json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
        local_path.parent.mkdir(parents=True, exist_ok=True)
        with local_path.open("xb") as stream:
            stream.write(encoded)
        receipt = {**metadata, "sha256": hashlib.sha256(encoded).hexdigest(), "file_name": file_name,
                   "local_path": str(local_path),
                   "artifact_reference": f"{self.run.entity}/{self.run.project}/{name}:{aliases[-1]}"}
        artifact = self._sdk.Artifact(name=name, type=kind, metadata=receipt)
        artifact.add_file(str(local_path), name=file_name)
        self._artifacts.append(self.run.log_artifact(artifact, aliases=aliases))
        return receipt

    def log_evaluation(self, metrics: dict, *, step: int, diagnostics: dict | None = None,
                       evaluation: dict | None = None, purpose: str = "primary") -> dict | None:
        if self.enabled:
            from .evaluation_reporting import (EVALUATION_LOG_SCHEMA, EVALUATION_PURPOSE_PREFIXES,
                                                compact_evaluation_metrics, evaluation_report)

            if (purpose not in EVALUATION_PURPOSE_PREFIXES or isinstance(step, bool)
                    or not isinstance(step, int) or step < 0):
                raise ValueError("Evaluation logging requires a valid purpose and optimizer step")
            previous = [row["optimizer_steps"] for row in self._evaluation_records if row["purpose"] == purpose]
            if previous and step <= previous[-1]:
                raise ValueError("Evaluation steps must increase strictly within each purpose")
            self._validate_history_step(step)
            if evaluation is not None:
                if evaluation.get("optimizer_steps", step) != step:
                    raise ValueError("Evaluation artifact optimizer step differs from logged step")
                if any(key not in metrics or metrics[key] != value for key, value in evaluation.get("metrics", {}).items()):
                    raise ValueError("Evaluation artifact metrics differ from the supplied complete event")
                if diagnostics is None:
                    diagnostics = evaluation.get("diagnostics")
                elif evaluation.get("diagnostics", diagnostics) != diagnostics:
                    raise ValueError("Evaluation artifact diagnostics differ from the supplied event")
            prefix = EVALUATION_PURPOSE_PREFIXES[purpose]
            compact = compact_evaluation_metrics(metrics, evaluation=evaluation)
            row = {"optimizer_steps": step, **{prefix + key: value for key, value in compact.items()}}
            scope = self.config.get("evaluation", {}).get("mode") if purpose == "primary" else purpose
            report = evaluation_report(metrics, scope=scope)
            if diagnostics is not None:
                confusion, support = [], []
                scopes = [("overall", "all", diagnostics["overall"])]
                for scope in ("per_task", "per_group"):
                    scopes.extend((scope, name, value) for name, value in diagnostics[scope].items())
                for scope, name, value in scopes:
                    stage = value["C"]
                    for true_stage, counts in enumerate(stage["confusion_matrix"]):
                        for predicted_stage, count in enumerate(counts):
                            confusion.append([step, scope, name, true_stage, predicted_stage, count])
                    for index, count in enumerate(stage["stage_support"]):
                        support.append([step, scope, name, index, count, stage["stage_accuracy"][index]])
                row[prefix + "stage_confusion"] = self._sdk.Table(
                    columns=["optimizer_steps", "scope", "group_or_task", "GT_stage", "predicted_stage", "count"],
                    data=confusion)
                row[prefix + "stage_support"] = self._sdk.Table(
                    columns=["optimizer_steps", "scope", "group_or_task", "stage", "support", "accuracy"],
                    data=support)
            self._validate_history_row(row, step=step)
            metadata = {"schema": EVALUATION_LOG_SCHEMA, "optimizer_steps": step, "purpose": purpose,
                        "plan_sha256": (evaluation or {}).get("plan_sha256", metrics.get("plan_sha256")),
                        "logged_scalar_keys": sorted(prefix + key for key in compact)}
            receipt = self._log_json_artifact(
                {"schema": EVALUATION_LOG_SCHEMA, "optimizer_steps": step, "purpose": purpose,
                 "metrics": metrics, "diagnostics": diagnostics, "evaluation": evaluation, "report": report,
                 "logging_provenance": {"mode": "current_run"}},
                name=f"grana-{self.run.id}-evaluation-{purpose}", kind="evaluation-details", file_name="evaluation.json",
                local_path=self.output / "evaluation_artifacts" / f"{purpose}-{step:06d}.json", metadata=metadata,
                aliases=["latest", f"step-{step:06d}"])
            self._queue_history(row, step=step)
            if purpose == "primary":
                self.run.summary["evaluation_report"] = {key: report[key] for key in ("primary", "oracle_accumulated_comparison")}
                self.run.summary["primary_evaluation"] = {"optimizer_steps": step, **report["primary"]}
            self._evaluation_records.append(receipt)
            self.run.summary["evaluation_artifacts"] = list(self._evaluation_records)
            self._receipt["evaluation_artifacts"] = list(self._evaluation_records)
            self._write_receipt()
            return receipt

    def log_evaluation_renders(self, bundle: dict, *, step: int) -> None:
        """Publish this step's held-out RGB/GT/prediction replays in run history.

        Media use stable trajectory keys so W&B's step selector compares the
        same held-out example over training. The table includes the optimizer
        step, elapsed training time, and per-trajectory POI RMSE/accuracy. The
        artifact retains numeric plotted samples and source provenance.
        """
        if not self.enabled:
            return
        if (bundle.get("schema") != "grana-training-eval-renders-v1"
                or bundle.get("optimizer_steps") != step or isinstance(step, bool)
                or not isinstance(step, int) or step < 0):
            raise ValueError("Evaluation render bundle must match this optimizer step")
        if self._render_records and step <= self._render_records[-1]["optimizer_steps"]:
            raise ValueError("Evaluation render steps must increase strictly")
        self._validate_history_step(step)
        row = {"optimizer_steps": step, "training_elapsed_seconds": bundle["training_elapsed_seconds"],
               "eval/rendering_seconds": bundle["rendering_seconds"],
               "eval/rendered_trajectories": bundle["rendered_trajectories"]}
        columns = ["optimizer_steps", "training_elapsed_seconds", "task", "trajectory_id", "image", "sampled_replay",
                   "A_whole_task_RMSE_diagnostic", "B_oracle_focus_relative_RMSE_diagnostic",
                   "C_stage_accuracy_diagnostic", "D_predicted_focus_relative_RMSE_diagnostic",
                   "evaluated_POIs", "replay_POIs"]
        table_rows = []
        for item in bundle["items"]:
            caption = (f"{item['task']} | held-out {item['trajectory_id']} | step {step} | "
                       f"training elapsed {bundle['training_elapsed_seconds']:.1f}s | "
                       "sampled evaluation windows, not original-speed video")
            image = self._sdk.Image(item["image"], caption=caption)
            video = self._sdk.Video(item["video"], format=item["video_format"], caption=caption)
            row[f"eval/renders/{item['key']}"] = image
            row[f"eval/replays/{item['key']}"] = video
            metrics = item["metrics"]
            table_rows.append([step, bundle["training_elapsed_seconds"], item["task"], item["trajectory_id"], image, video,
                               metrics.get("role_A_poi_rmse"), metrics.get("role_B_poi_rmse"),
                               metrics.get("role_C_poi_accuracy"), metrics.get("role_D_poi_rmse"),
                               len(item["sampled_pois"]), len(item["replay_pois"])])
        row["eval/render_examples"] = self._sdk.Table(columns=columns, data=table_rows)
        self._validate_history_row(row, step=step)
        metadata = {key: bundle[key] for key in ("optimizer_steps", "training_elapsed_seconds", "plan_sha256",
                                                 "replay_type", "available_trajectories", "rendered_trajectories")}
        # W&B collection names have a fixed artifact type. The render cohort's
        # metric JSON uses evaluation-renders (evaluation-details), so media
        # must have its own collection even though both share optimizer steps.
        artifact = self._sdk.Artifact(name=f"grana-{self.run.id}-evaluation-render-media", type="evaluation-renders",
                                      description="Numeric held-out samples and provenance for training-step RGB replays.",
                                      metadata=metadata)
        artifact.add_file(bundle["manifest"], name="manifest.json")
        self._artifacts.append(self.run.log_artifact(artifact, aliases=["latest", f"step-{step:06d}"]))
        self._queue_history(row, step=step)
        record = {**metadata, "local_manifest": bundle["manifest"],
                  "media_keys": [item["key"] for item in bundle["items"]]}
        self._render_records.append(record)
        self.run.summary["evaluation_renders"] = list(self._render_records)
        self._receipt["evaluation_renders"] = list(self._render_records)
        self._write_receipt()

    def log_deploy_dense(self, bundle: dict, *, step: int) -> None:
        if not self.enabled:
            return
        from .deploy_dense import log_dense_bundle
        if bundle.get("optimizer_steps") != step or bundle["provenance"]["mode"] != "during_training":
            raise ValueError("Training dense evaluation must match the current optimizer step")
        if self._dense_records and step <= self._dense_records[-1]["optimizer_steps"]:
            raise ValueError("Dense evaluation steps must increase strictly")
        self._validate_history_step(step)
        self._artifacts.append(log_dense_bundle(self.run, self._sdk, bundle, history_sink=self._queue_history))
        self._dense_records.append({"optimizer_steps": step, "plan_sha256": bundle["plan_sha256"],
                                   "evaluated_pois": bundle["evaluated_pois"], "mode": "during_training",
                                   "records_sha256": bundle.get("records_sha256"),
                                   "metric_scope": bundle.get("metric_scope"),
                                   "manifest": bundle["manifest"], "media_pairs": len(bundle["items"])})
        self.run.summary["deploy_dense_evaluations"] = list(self._dense_records)
        self._receipt["deploy_dense_evaluations"] = list(self._dense_records)
        self._write_receipt()

    def log_checkpoint(self, path: str | Path, *, step: int, final: bool = False) -> dict | None:
        if not self.enabled:
            return None
        path = Path(path).expanduser().resolve()
        metadata_path = path / "metadata.json"
        metadata = json.loads(metadata_path.read_text())
        include_weights = self.settings["upload_checkpoint_weights"]
        name = f"grana-{self.run.id}-checkpoints"
        alias = f"step-{step:06d}"
        record = {"step": step, "final": final, "local_checkpoint": str(path),
                  "artifact_reference": f"{self.run.entity}/{self.run.project}/{name}:{alias}",
                  "payload_included": include_weights,
                  "artifact_kind": "model" if include_weights else "checkpoint-manifest",
                  "weights_sha256": metadata["weights_sha256"],
                  "files": [{"name": p.name, "bytes": p.stat().st_size,
                             "uploaded": p.name == "metadata.json" or include_weights}
                            for p in (metadata_path, path / "weights.safetensors")]}
        manifest_dir = self.output / "checkpoint_manifests"
        manifest_dir.mkdir(exist_ok=True)
        manifest_path = manifest_dir / f"checkpoint-{step:06d}.json"
        manifest_path.write_text(json.dumps(record, indent=2) + "\n")
        artifact = self._sdk.Artifact(name=name, type=record["artifact_kind"],
                                      description="Grana checkpoint weights and metadata" if include_weights else
                                      "Metadata-only checkpoint manifest; model weights remain on the training host.",
                                      metadata=record)
        artifact.add_file(str(metadata_path), name="metadata.json")
        artifact.add_file(str(manifest_path), name="checkpoint-manifest.json")
        if include_weights:
            artifact.add_file(str(path / "weights.safetensors"), name="weights.safetensors")
        logged = self.run.log_artifact(artifact, aliases=["latest", alias] + (["final"] if final else []))
        self._artifacts.append(logged)
        self._checkpoint_records.append(record)
        self.run.summary["checkpoints"] = list(self._checkpoint_records)
        self._receipt["checkpoints"] = list(self._checkpoint_records)
        self._write_receipt()
        return record

    def log_checkpoint_replay_dense(self, bundle: dict, *, step: int) -> None:
        """Publish replay media in a dedicated evaluation run, never as training."""
        if not self.enabled:
            return
        from .deploy_dense import log_dense_bundle

        provenance = bundle.get("provenance", {})
        if (self.settings.get("job_type") != "checkpoint-evaluation"
                or isinstance(step, bool) or not isinstance(step, int) or step <= 0
                or bundle.get("optimizer_steps") != step
                or provenance.get("mode") != "checkpoint_replay"
                or provenance.get("source_optimizer_step") != step
                or provenance.get("new_optimizer_updates") != 0
                or not all(provenance.get(key) for key in ("checkpoint", "weights_sha256", "source_run"))):
            raise ValueError("Checkpoint replay requires a dedicated evaluation run and matching saved-step provenance")
        if self._replay_dense_records and step <= self._replay_dense_records[-1]["optimizer_steps"]:
            raise ValueError("Checkpoint replay steps must increase strictly")
        self._validate_history_step(step)
        self._artifacts.append(log_dense_bundle(self.run, self._sdk, bundle, history_sink=self._queue_history))
        self._replay_dense_records.append({"optimizer_steps": step, "plan_sha256": bundle["plan_sha256"],
                                           "manifest": bundle["manifest"], "media_pairs": len(bundle["items"]),
                                           "provenance": dict(provenance)})
        self.run.summary["checkpoint_replay_evaluations"] = list(self._replay_dense_records)
        self._receipt["checkpoint_replay_evaluations"] = list(self._replay_dense_records)
        self._write_receipt()

    def finish(self, summary: dict | None = None, *, exit_code: int = 0) -> None:
        if self._finished or not self.enabled or self.run is None:
            return
        status = "finished" if exit_code == 0 else "failed"
        try:
            if self._history_error is not None:
                if exit_code == 0:
                    raise RuntimeError("Cannot finish successfully after a failed history write") from self._history_error
            elif self._pending_step is not None:
                self.commit_step(step=self._pending_step)
            if summary:
                receipt = self._log_json_artifact(
                    {"schema": "grana-completion-log-v1", "summary": summary},
                    name=f"grana-{self.run.id}-completion", kind="run-completion", file_name="completion.json",
                    local_path=self.output / "evaluation_artifacts" / "completion.json",
                    metadata={"schema": "grana-completion-log-v1"}, aliases=["final"])
                self._receipt["completion_artifact"] = receipt
                self.run.summary["completion_artifact"] = receipt
                allowed = {"optimizer_steps", "status", "failure_type", "checkpoint_reload_equal",
                           "checkpoint_prediction_parity", "checkpoint_prediction_max_abs_diff",
                           "sampled_parameter_delta_l1", "total_elapsed_seconds", "training_compute_seconds",
                           "new_optimizer_updates", "new_model_forwards", "source_optimizer_step"}
                allowed.update(key for record in self._evaluation_records for key in record["logged_scalar_keys"])
                self.run.summary.update({key: value for key, value in summary.items() if key in allowed})
            self.run.summary.update({"status": status, "exit_code": exit_code,
                                     "logged_optimizer_steps": self._last_step})
            if self.mode == "online" and exit_code == 0:
                for artifact in self._artifacts:
                    artifact.wait(timeout=180)
            self.run.finish(exit_code=exit_code)
        except BaseException as error:
            self._receipt.update(status="failed", failure_type=type(error).__name__, exit_code=1)
            self._write_receipt()
            # An artifact upload can fail before finish is reached. Tell W&B
            # that the run failed, while preserving the original exception.
            try:
                self.run.summary.update({"status": "failed", "exit_code": 1,
                                         "failure_type": type(error).__name__})
                self.run.finish(exit_code=1)
                self._finished = True
            except BaseException:
                pass
            raise
        self._finished = True
        self._receipt.update(status=status, exit_code=exit_code)
        self._write_receipt()

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None and self.enabled and self.run is not None:
            self._receipt["failure_type"] = exc_type.__name__
            self.finish({"failure_type": exc_type.__name__}, exit_code=1)
        else:
            self.finish()
        return False
