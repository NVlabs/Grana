# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluate an immutable saved checkpoint, including from an unfinished run.

The saved training plan is reconstructed and verified before applying execution
overrides. Output and online media use a new evaluation run; the source training
run, its completion status and historical evaluation events are never rewritten.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import time

from .campaign import atomic_json


def read_json(path):
    return json.loads(Path(path).read_text())


def prepare_checkpoint_evaluation(source: Path, step: int, *, batch_size: int,
                                  batched_vision_sdpa: bool):
    """Validate saved provenance without requiring a completed trainer result."""
    from grana.data.cache import CachedTrajectories
    from grana.data.sampling import CTSP9
    from .evaluation import build_evaluation_plan
    from .deploy_dense import build_dense_plan

    if isinstance(step, bool) or not isinstance(step, int) or step <= 0:
        raise ValueError("Checkpoint step must be a positive integer")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("Evaluation batch size must be a positive integer")
    if type(batched_vision_sdpa) is not bool:
        raise ValueError("batched_vision_sdpa must be a bool")
    source = Path(source).resolve()
    config = read_json(source / "resolved_config.json")
    if step > config["training"]["max_steps"]:
        raise ValueError("Checkpoint step exceeds the source training schedule")
    checkpoint = source / f"checkpoint-{step:06d}"
    metadata = read_json(checkpoint / "metadata.json")
    if (metadata.get("format") != "grana-weights-v1"
            or type(metadata.get("source_step")) is not int or metadata["source_step"] != step):
        raise ValueError("Checkpoint format or source step does not match")
    if metadata["source_stage"] != config["training"]["stage"]:
        raise ValueError("Checkpoint training stage does not match the source run")
    for name in ("name_or_path", "revision"):
        if metadata["model"].get(name) != config["model"].get(name):
            raise ValueError(f"Checkpoint model {name} does not match the source run")
    # This also prevents accidentally pointing at a different run's checkpoint.
    compared_keys = ("model", "data", "roles", "evaluation", "training")
    # Current transferable weights omit the run's loss configuration. Older
    # artifacts may include it; when present it must agree for an exact replay.
    # Otherwise resolved_config.json is authoritative and hashed in provenance.
    if "loss" in metadata["config"]:
        compared_keys += ("loss",)
    for key in compared_keys:
        if metadata["config"][key] != config[key]:
            raise ValueError(f"Checkpoint saved {key} configuration differs from the source run")
    if not (checkpoint / "weights.safetensors").is_file():
        raise FileNotFoundError(checkpoint / "weights.safetensors")
    rows = [json.loads(line) for line in (source / "training_metrics.jsonl").read_text().splitlines() if line]
    matches = [row for row in rows if type(row.get("optimizer_steps")) is int and row["optimizer_steps"] == step]
    if len(matches) != 1 or Path(matches[0].get("checkpoint_path", "")).resolve() != checkpoint:
        raise ValueError("Checkpoint must have one committed save receipt at the requested step")
    elapsed = matches[0].get("elapsed_seconds")
    if isinstance(elapsed, bool) or not isinstance(elapsed, (float, int)) or not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError("Committed save receipt needs finite nonnegative training elapsed time")
    train = CachedTrajectories(config["data"]["caches"], views=config["data"]["views"])
    heldout = CachedTrajectories(config["evaluation"]["caches"], views=config["data"]["views"])
    source_manifest = read_json(source / "evaluation_plan.json")
    source_evaluation = {**config["evaluation"]}
    # Legacy eval used data.stride_pairs (or CTSP9 when omitted). Rebuild from
    # the saved config, then verify the frozen plan without trusting its inputs.
    source_evaluation.setdefault("stride_pairs", config["data"].get("stride_pairs", CTSP9))
    base = build_evaluation_plan(heldout, source_evaluation, config["data"], training_dataset=train,
                                 roles=config["roles"])
    plan = base
    if config["evaluation"]["mode"] == "dense":
        dense = config["evaluation"]["deploy_dense"]
        plan = build_dense_plan(base, max_trajectories_per_task=dense["max_trajectories_per_task"],
                                frame_stride=dense["frame_stride"], include_oracle=dense["include_oracle"],
                                group_by_cache=dense["group_by_cache"])
    if plan.manifest() != source_manifest:
        raise ValueError("Reconstructed source evaluation plan differs from the saved plan")
    source_manifest = plan.manifest()
    settings = {**plan.settings}
    if batch_size == 1:
        settings.pop("batch_size", None)
    else:
        settings["batch_size"] = batch_size
    execution_plan = replace(plan, settings=settings)
    if [w.manifest() for w in execution_plan.windows] != source_manifest["windows"]:
        raise RuntimeError("Execution override changed evaluation windows")
    return config, checkpoint, execution_plan, {
        "mode": "checkpoint_replay", "source_directory": str(source), "checkpoint": str(checkpoint),
        "source_optimizer_step": step, "weights_sha256": metadata["weights_sha256"],
        "source_plan_sha256": source_manifest["plan_sha256"],
        "execution_plan_sha256": execution_plan.manifest()["plan_sha256"],
        "batch_size": batch_size, "batched_vision_sdpa": batched_vision_sdpa,
        "same_source_windows": True, "training_elapsed_seconds": elapsed,
        "source_config_sha256": sha256((source / "resolved_config.json").read_bytes()).hexdigest(),
        "source_run_completion_required": False,
        "new_optimizer_updates": 0,
    }


def run(args):
    import torch
    from grana.runtime import require_owned_environment
    from grana.modeling.model import build_model
    from grana.modeling.checkpoint import load_weights
    from grana.modeling.vision_sdpa import configure_batched_vision_sdpa, batched_vision_sdpa_stats
    from .evaluation import evaluate
    from .deploy_dense import render_dense_evaluation, require_encoder
    from .logging import RunLogger

    require_owned_environment()
    config, checkpoint, plan, provenance = prepare_checkpoint_evaluation(
        args.source_run, args.step, batch_size=args.batch_size, batched_vision_sdpa=args.batched_vision_sdpa)
    source = Path(provenance["source_directory"])
    output = args.output.resolve()
    if output == source or source in output.parents:
        raise ValueError("Checkpoint evaluation output must be outside the immutable source run")
    if args.dry_run:
        print(json.dumps({**provenance, "windows": len(plan.windows), "model_calls": plan.manifest()["model_calls"]}))
        return
    if config["evaluation"]["mode"] == "dense":
        require_encoder()
    config = deepcopy(config)
    config["training"]["output_dir"] = str(output)
    config["training"]["batched_vision_sdpa"] = False
    config["evaluation"].update(batch_size=args.batch_size, batched_vision_sdpa=args.batched_vision_sdpa)
    config["logging"].update(mode=args.logging_mode, job_type="checkpoint-evaluation",
                              name=f"{args.source_run.name}-checkpoint-{args.step:06d}-bs{args.batch_size}")
    source_receipt = source / "wandb_run.json"
    source_wandb = read_json(source_receipt) if source_receipt.is_file() else {}
    identity = [source_wandb.get(key) for key in ("entity", "project", "id")]
    provenance["source_run"] = "/".join(identity) if all(isinstance(value, str) and value for value in identity) else f"local:{source}"
    provenance["execution_device"] = args.device
    provenance["execution_world_size"] = 1
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "provenance": provenance, "created_utc": datetime.now(timezone.utc).isoformat()}
    atomic_json(output / "checkpoint_evaluation.json", report)
    atomic_json(output / "config.json", config)
    atomic_json(output / "evaluation_plan.json", plan.manifest())
    try:
        torch.set_num_threads(config["training"]["cpus_per_worker"])
        device = torch.device(args.device)
        if device.type == "cuda":
            torch.cuda.set_device(device)
        model, processor = build_model(config)
        load = load_weights(model, checkpoint)
        if load["adapter_tensor_count"] == 0 or any(value["outcome"] != "loaded" for value in load["heads"].values()):
            raise ValueError("Checkpoint evaluation requires all adapters and both complete saved heads")
        model.to(device)
        configure_batched_vision_sdpa(model, enabled=args.batched_vision_sdpa, training_enabled=False, profile=True)
        # This entrypoint owns a fresh evaluation run and service lifetime.
        # Parent-service bindings or a shell-level run ID must not reconnect it
        # to the source training run or make it depend on a launcher surviving.
        for key in ("WANDB_SERVICE", "_WANDB_SERVICE", "WANDB_RUN_ID"):
            os.environ.pop(key, None)
        with RunLogger(config) as logger:
            logger.start(model_summary=model.model_spec, data_summary={"heldout_windows": len(plan.windows)},
                         initialization={**load, **provenance})
            last_update = time.monotonic()
            def progress(done, total):
                nonlocal last_update
                if time.monotonic() - last_update >= 30 or done == total:
                    print(json.dumps({"checkpoint_step": args.step, "role_rows_completed": done, "role_rows": total}), flush=True)
                    last_update = time.monotonic()
            result = evaluate(model, processor, plan, device=device, max_pixels=config["data"]["max_pixels"],
                              frame_weights=config["loss"]["frame_weights"], role_weights=config["loss"]["role_weights"],
                              loss_config=config["loss"], progress_callback=progress)
            result.update(optimizer_steps=args.step, provenance=provenance,
                          training_elapsed_seconds=provenance["training_elapsed_seconds"],
                          vision_sdpa=batched_vision_sdpa_stats(model, synchronize=True))
            atomic_json(output / f"evaluation-{args.step:06d}.json", result)
            from .engine import _evaluation_runtime_metrics
            logger.log_evaluation({**result["metrics"], **_evaluation_runtime_metrics(result),
                                   "elapsed_seconds": result["elapsed_seconds"]}, step=args.step,
                                  diagnostics=result.get("diagnostics"), evaluation=result)
            if config["evaluation"]["mode"] == "dense":
                settings = config["evaluation"]["deploy_dense"]
                bundle = render_dense_evaluation(result, plan, output_dir=output / "deploy_dense", step=args.step,
                    training_elapsed_seconds=provenance["training_elapsed_seconds"], max_frames=settings["max_frames"],
                    replay_fps=settings["replay_fps"], provenance=provenance)
                logger.log_checkpoint_replay_dense(bundle, step=args.step)
                report.update(media_manifest=bundle["manifest"], media_pairs=len(bundle["items"]))
            logger.commit_step(step=args.step)
            report.update(status="completed", elapsed_seconds=result["elapsed_seconds"],
                          source_optimizer_step=args.step, new_optimizer_updates=0,
                          metrics=result["metrics"], checkpoint_load=load,
                          completed_utc=datetime.now(timezone.utc).isoformat())
            logger.finish(summary={"checkpoint_evaluation": report})
        atomic_json(output / "checkpoint_evaluation.json", report)
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        atomic_json(output / "checkpoint_evaluation.json", report)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", required=True, type=Path)
    parser.add_argument("--step", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", required=True, type=int)
    parser.add_argument("--batched-vision-sdpa", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--logging-mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--dry-run", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
