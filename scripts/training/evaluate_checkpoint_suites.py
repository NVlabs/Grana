#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluate fixed dense or bounded cohorts on all Ray workers, loading weights once."""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def read(path):
    return json.loads(Path(path).read_text())


def write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def normalize_specs(config, specs):
    """Resolve explicit suite budgets without changing the source configuration."""
    if not isinstance(specs, list) or not specs:
        raise ValueError("Suite specs must be a nonempty JSON list")
    allowed = {"name", "caches", "roles", "max_trajectories_per_task", "batch_size",
               "frame_stride", "points_per_trajectory", "seed", "stride_pairs", "include_oracle"}
    resolved, names = [], set()
    for value in specs:
        if not isinstance(value, dict) or set(value) - allowed:
            raise ValueError("Suite specs contain unknown fields or are not objects")
        spec = deepcopy(value)
        name = spec.get("name")
        if (not isinstance(name, str) or not name or name in (".", "..") or name in names
                or Path(name).name != name or name in {"plans", "ray", "driver.json", "config.json",
                                                       "suite_specs.json", "worker_setup.json", "suite_reports.json"}
                or "\\" in name):
            raise ValueError("Suite names must be unique, unreserved path components")
        names.add(name)
        if not isinstance(spec.get("caches"), list) or not spec["caches"]:
            raise ValueError(f"Suite {name!r} needs an explicit nonempty caches list")
        spec.setdefault("max_trajectories_per_task", None)
        spec.setdefault("batch_size", 64)
        spec.setdefault("roles", ["C", "D"])
        spec.setdefault("seed", config["evaluation"]["seed"])
        spec.setdefault("stride_pairs", deepcopy(config["evaluation"]["stride_pairs"]))
        spec.setdefault("include_oracle", False)
        if spec.get("points_per_trajectory") is not None and "frame_stride" in spec:
            raise ValueError("Choose points_per_trajectory or frame_stride, not both")
        if spec.get("points_per_trajectory") is None:
            spec.setdefault("frame_stride", config["evaluation"]["deploy_dense"]["frame_stride"])
        for key, minimum in (("batch_size", 1), ("max_trajectories_per_task", 1),
                             ("frame_stride", 1), ("points_per_trajectory", 2)):
            value = spec.get(key)
            if value is None and key != "batch_size":
                continue
            if type(value) is not int or value < minimum:
                raise ValueError(f"Suite {key} must be an integer >= {minimum}")
        if type(spec["seed"]) is not int or type(spec["include_oracle"]) is not bool:
            raise ValueError("Suite seed must be an integer and include_oracle must be boolean")
        roles = spec["roles"]
        if (not isinstance(roles, list) or not roles or any(role not in ("A", "B", "C", "D") for role in roles)
                or len(set(roles)) != len(roles) or ("D" in roles and "C" not in roles)):
            raise ValueError("Suite roles must be unique A/B/C/D values; D requires C")
        resolved.append(spec)
    return resolved


def prepare(config, specs):
    from grana.data.cache import CachedTrajectories
    from grana.training.evaluation import build_evaluation_plan
    from grana.training.deploy_dense import build_dense_plan
    from grana.training.render_plan import build_render_plan
    from grana.training.distributed_evaluation import split_plan

    training = CachedTrajectories(config["data"]["caches"], views=config["data"]["views"])
    plans, summaries = {}, {}
    for spec in normalize_specs(config, specs):
        name = spec["name"]
        heldout = CachedTrajectories(spec["caches"], views=config["data"]["views"])
        options = deepcopy(config["evaluation"])
        options.update(max_trajectories_per_task=spec["max_trajectories_per_task"],
                       batch_size=spec["batch_size"], stride_pairs=spec["stride_pairs"], seed=spec["seed"])
        roles = tuple(spec["roles"])
        base = build_evaluation_plan(heldout, options, config["data"],
                                     training_dataset=training, roles=roles)
        if spec.get("points_per_trajectory") is not None:
            # Apply the same deployment-role policy as the dense path before
            # choosing bounded POIs. Endpoint planning does not read RGB.
            routes = build_dense_plan(base, max_trajectories_per_task=spec["max_trajectories_per_task"],
                                      frame_stride=max(w.trajectory.num_frames for w in base.windows),
                                      include_oracle=spec["include_oracle"])
            sampled = build_render_plan(routes, max_trajectories_per_task=spec["max_trajectories_per_task"],
                                        points_per_trajectory=spec["points_per_trajectory"])
            plan = replace(sampled, settings={**sampled.settings,
                "purpose": "sampled_evaluation", "metric_scope": "fixed_heldout_uniform_poi_cohort",
                "frame_stride": "uniform POIs", "source_fps": 10, "include_final_frame": True,
                "include_oracle": spec["include_oracle"]})
        else:
            plan = build_dense_plan(base, max_trajectories_per_task=spec["max_trajectories_per_task"],
                                    frame_stride=spec["frame_stride"], include_oracle=spec["include_oracle"])
        if any("A" not in window.roles and not {"C", "D"} <= set(window.roles) for window in plan.windows):
            raise ValueError("Media suites require an A or C+D progress route for every window")
        shards = split_plan(plan, workers=config["training"]["num_workers"])
        plans[name] = plan
        summaries[name] = {
            "plan_sha256": plan.manifest()["plan_sha256"],
            "task_count": len(plan.tasks), "windows": len(plan.windows),
            "role_rows": plan.manifest()["model_calls"],
            "trajectories": len({(str(w.trajectory.cache_root), w.trajectory.trajectory_id)
                                 for w in plan.windows}),
            "world_size": len(shards), "shards": [s.manifest() for s in shards],
            "active_workers": sum(bool(s.plan.windows) for s in shards),
            "batch_size": spec["batch_size"], "frame_stride": plan.settings["frame_stride"],
            "points_per_trajectory": spec.get("points_per_trajectory"),
            "metric_scope": plan.settings["metric_scope"],
            "roles": list(roles), "eval_augmentation": False,
        }
    return plans, summaries


def checkpoint_source(config, source, step):
    """Read the saved checkpoint receipt; a finished training run is not required."""
    if type(step) is not int or step <= 0:
        raise ValueError("Checkpoint step must be a positive integer")
    source = Path(source).resolve()
    checkpoint = source / f"checkpoint-{step:06d}"
    metadata = read(checkpoint / "metadata.json")
    if (metadata.get("format") != "grana-weights-v1" or type(metadata.get("source_step")) is not int
            or metadata["source_step"] != step):
        raise ValueError("Checkpoint format or step mismatch")
    for key in ("name_or_path", "revision", "lora_rank", "lora_alpha"):
        if metadata["model"].get(key) != config["model"].get(key):
            raise ValueError(f"Checkpoint model {key} differs from evaluation config")
    if not (checkpoint / "weights.safetensors").is_file():
        raise FileNotFoundError(checkpoint / "weights.safetensors")
    events = [json.loads(line) for line in (source / "training_metrics.jsonl").read_text().splitlines() if line.strip()]
    events = [event for event in events if event.get("optimizer_steps") == step]
    if len(events) != 1 or Path(events[0].get("checkpoint_path", "")).resolve() != checkpoint:
        raise ValueError("Checkpoint needs one committed source-run save receipt")
    elapsed = events[0].get("elapsed_seconds")
    if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError("Checkpoint save receipt needs finite nonnegative elapsed_seconds")
    receipt_path = source / "wandb_run.json"
    receipt = read(receipt_path) if receipt_path.is_file() else {}
    identity = [receipt.get(key) for key in ("entity", "project", "id")]
    source_run = "/".join(identity) if all(isinstance(value, str) and value for value in identity) else f"local:{source}"
    return checkpoint, {"source_run": source_run, "training_elapsed_seconds": elapsed,
                        "weights_sha256": metadata["weights_sha256"]}


class IncompleteEvaluationError(RuntimeError):
    """A launcher can exit normally even when its workers were interrupted."""


def verify_completion(result, output, specs, prepared):
    if result.error:
        raise RuntimeError(str(result.error))
    if result.metrics.get("completed_suites") != len(specs):
        raise IncompleteEvaluationError("Workers did not report all requested suites complete")
    for spec in specs:
        path = output / spec["name"] / "report.json"
        report = read(path) if path.is_file() else {}
        if (report.get("status") != "succeeded"
                or report.get("plan_sha256") != prepared[spec["name"]]["plan_sha256"]
                or report.get("wandb", {}).get("status") not in ("finished", "disabled")):
            raise IncompleteEvaluationError(f"Suite {spec['name']!r} lacks a completed result and logging receipt")


def close_incomplete_runs(output, specs, *, status, error):
    """After Ray shuts down, close only this attempt's unfinished online runs."""
    errors = []
    for spec in specs:
        path = output / spec["name"] / "wandb_run.json"
        try:
            receipt = read(path) if path.is_file() else {}
            if receipt.get("mode") != "online" or receipt.get("status") != "running":
                continue
            import wandb
            run = wandb.init(id=receipt["id"], project=receipt["project"], entity=receipt["entity"],
                             resume="must", mode="online", dir=str(path.parent), reinit="create_new",
                             settings=wandb.Settings(console="off", disable_git=True, disable_code=True,
                                                     init_timeout=30, finish_timeout=60, finish_timeout_raises=True))
            run.summary.update({"evaluation_status": status, "evaluation_complete": False,
                                "termination_reason": error})
            code = 130 if status == "cancelled" else 1
            run.finish(exit_code=code)
            write(path, {**receipt, "status": "failed", "evaluation_status": status, "exit_code": code})
        except Exception as exc:
            errors.append({"suite": spec["name"], "error": f"{type(exc).__name__}: {exc}"})
    return errors


def worker(payload):
    import torch
    import torch.distributed as dist
    from ray import train
    from ray.train.torch import get_device
    from grana.runtime import require_ray_worker
    from grana.modeling.model import build_model
    from grana.modeling.checkpoint import load_weights
    from grana.modeling.vision_sdpa import batched_vision_sdpa_stats, configure_batched_vision_sdpa
    from grana.training.evaluation import evaluate
    from grana.training.distributed_evaluation import evaluate_distributed
    from grana.training.deploy_dense import render_dense_evaluation, require_encoder, dense_metrics, dense_series
    from grana.training.engine import _evaluation_runtime_metrics
    from grana.training.logging import RunLogger

    require_ray_worker()
    context = train.get_context()
    rank, world = context.get_world_rank(), context.get_world_size()
    config, specs = payload["config"], payload["specs"]
    output, checkpoint = Path(payload["output"]), Path(payload["checkpoint"])
    device = get_device()
    visibility = os.environ.get("CUDA_VISIBLE_DEVICES")
    mappings = {item["device_uuid"]: item for item in payload["device_mappings"].values()}
    if visibility not in mappings or device.index != 0 or torch.cuda.device_count() != 1:
        raise RuntimeError("Evaluation workers must be isolated to one assigned GPU UUID at cuda:0")
    torch.cuda.set_device(device)
    torch.set_num_threads(config["training"]["cpus_per_worker"])
    started = time.perf_counter()
    plans, summaries = prepare(config, specs)
    if summaries != payload["prepared"]:
        raise ValueError("Worker rebuilt a different evaluation plan")
    if rank == 0:
        require_encoder()
        for name, plan in plans.items():
            write(output / "plans" / f"{name}.json", plan.manifest())
    model_started = time.perf_counter()
    model, processor = build_model(config)
    loaded = load_weights(model, checkpoint)
    if not loaded["adapter_tensor_count"] or any(x["outcome"] != "loaded" for x in loaded["heads"].values()):
        raise ValueError("All checkpoint adapters and heads must be loaded")
    model.to(device)
    model.eval()
    configure_batched_vision_sdpa(model, enabled=config["evaluation"]["batched_vision_sdpa"],
                                 training_enabled=False, profile=True)
    torch.cuda.synchronize(device)
    setup = {"rank": rank, "model_load_seconds": time.perf_counter() - model_started,
             "worker_setup_seconds": time.perf_counter() - started, "device_mapping": mappings[visibility]}
    setups = [None] * world
    dist.all_gather_object(setups, setup)
    if rank == 0:
        write(output / "worker_setup.json", setups)
    reports = {}
    for spec in specs:
        name, plan = spec["name"], plans[spec["name"]]
        cfg = deepcopy(config)
        suite_output = output / name
        cfg["training"]["output_dir"] = str(suite_output)
        cfg["training"]["init_checkpoint"] = str(checkpoint)
        cfg["evaluation"].update(caches=spec["caches"], manifest=None,
                                 batch_size=spec["batch_size"], max_trajectories_per_task=spec["max_trajectories_per_task"],
                                 seed=spec["seed"], stride_pairs=spec["stride_pairs"], mode="dense")
        cfg["evaluation"]["deploy_dense"].update(max_trajectories_per_task=spec["max_trajectories_per_task"],
                                                  frame_stride=spec.get("frame_stride", config["evaluation"]["deploy_dense"]["frame_stride"]),
                                                  include_oracle=spec["include_oracle"])
        if spec.get("points_per_trajectory") is not None:
            cfg["evaluation"].update(mode="sampled", windows_per_trajectory=spec["points_per_trajectory"])
            cfg["logging"]["tags"] = list(cfg["logging"].get("tags", [])) + [
                f"fixed-{spec['max_trajectories_per_task']}traj-{spec['points_per_trajectory']}poi", "bounded-eval"]
        cfg["logging"].update(name=f"checkpoint-{payload['step']:06d}-{name}-{world}gpu",
                                job_type="checkpoint-evaluation")
        suite_started = time.perf_counter()
        with RunLogger(cfg, rank=rank, world_size=world) as logger:
            provenance = {"mode": "checkpoint_replay", "checkpoint": str(checkpoint),
                          "weights_sha256": loaded["weights_sha256"], "source_run": payload["source_run"],
                          "source_optimizer_step": payload["step"], "suite": name,
                          "execution_world_size": world, "new_optimizer_updates": 0}
            logger.start(model_summary=model.model_spec, data_summary=summaries[name],
                         initialization={**loaded, **provenance})
            if rank == 0:
                write(suite_output / "config.json", cfg)
                write(suite_output / "evaluation_plan.json", plan.manifest())
            progress_path = suite_output / f"progress-rank-{rank}.json"

            def local(shard):
                last = [0.0]
                local_started = time.perf_counter()
                write(progress_path, {"rank": rank, "completed_role_rows": 0,
                      "total_role_rows": shard.manifest()["model_calls"], "updated_unix": time.time()})

                def progress(done, total):
                    now = time.perf_counter()
                    if now - last[0] >= 15 or done == total:
                        write(progress_path, {"rank": rank, "completed_role_rows": done,
                              "total_role_rows": total, "elapsed_seconds": now - local_started,
                              "updated_unix": time.time()})
                        last[0] = now

                batched_vision_sdpa_stats(model, reset=True)
                result = evaluate(model, processor, shard, device=device,
                                  max_pixels=cfg["data"]["max_pixels"],
                                  frame_weights=cfg["loss"]["frame_weights"],
                                  role_weights=cfg["loss"]["role_weights"], loss_config=cfg["loss"],
                                  progress_callback=progress)
                result["vision_sdpa"] = batched_vision_sdpa_stats(model, synchronize=True)
                return result

            result = evaluate_distributed(plan, local, output_dir=suite_output / "shards",
                                          frame_weights=cfg["loss"]["frame_weights"],
                                          role_weights=cfg["loss"]["role_weights"], loss_config=cfg["loss"])
            error, report = None, None
            if rank == 0:
                try:
                    persistence_started = time.perf_counter()
                    result.update(optimizer_steps=payload["step"], provenance=provenance)
                    if plan.settings["purpose"] == "sampled_evaluation":
                        result["metrics"].update(dense_metrics(dense_series(result, plan)))
                    write(suite_output / "evaluation.json", result)
                    bundle = render_dense_evaluation(
                        result, plan, output_dir=suite_output / "media", step=payload["step"],
                        training_elapsed_seconds=payload["training_elapsed_seconds"],
                        max_frames=cfg["evaluation"]["deploy_dense"]["max_frames"],
                        replay_fps=cfg["evaluation"]["deploy_dense"]["replay_fps"], provenance=provenance)
                    logger.log_checkpoint_replay_dense(bundle, step=payload["step"])
                    metrics = {**result["metrics"], **_evaluation_runtime_metrics(result),
                               "elapsed_seconds": result["elapsed_seconds"]}
                    logger.log_evaluation(metrics, step=payload["step"],
                                          diagnostics=result.get("diagnostics"), evaluation=result)
                    logger.commit_step(step=payload["step"])
                    report = {"status": "succeeded", "suite": name, **summaries[name],
                              "evaluation_seconds": result["elapsed_seconds"],
                              "rendering_seconds": bundle["rendering_seconds"],
                              "persistence_media_enqueue_seconds": time.perf_counter() - persistence_started,
                              "metrics": metrics, "provenance": provenance,
                              "parallel_execution": result["parallel_execution"], "worker_setup": setups}
                    cloud_started = time.perf_counter()
                    logger.finish(summary={"checkpoint_evaluation": report})
                    report.update(cloud_finish_seconds=time.perf_counter() - cloud_started,
                                  suite_end_to_end_seconds=time.perf_counter() - suite_started,
                                  wandb=read(suite_output / "wandb_run.json"))
                    write(suite_output / "report.json", report)
                    reports[name] = report
                    write(output / "suite_reports.json", reports)
                    print(json.dumps({"suite_completed": name, "evaluation_seconds": report["evaluation_seconds"],
                                      "end_to_end_seconds": report["suite_end_to_end_seconds"]}), flush=True)
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
            shared = [error]
            dist.broadcast_object_list(shared, src=0)
            if shared[0]:
                raise RuntimeError(shared[0])
        dist.barrier()
    train.report({"completed_suites": len(specs), "worker_total_seconds": time.perf_counter() - started})


def main(argv=None):
    from grana.config import load_config
    from grana.runtime import require_owned_environment

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--specs", required=True, type=Path)
    parser.add_argument("--prepared", required=True, type=Path)
    parser.add_argument("--source-run", required=True, type=Path)
    parser.add_argument("--step", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--num-workers", type=int,
                        help="One GPU per worker; defaults to all GPUs in CUDA_VISIBLE_DEVICES")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args(argv)
    require_owned_environment()
    config = load_config(args.config)
    specs = normalize_specs(config, read(args.specs))
    checkpoint, source_info = checkpoint_source(config, args.source_run, args.step)
    import torch
    workers = torch.cuda.device_count() if args.num_workers is None else args.num_workers
    if type(workers) is not int or workers <= 0:
        raise ValueError("At least one GPU worker is required; specify --num-workers for CPU-only preparation")
    config["training"]["num_workers"] = workers
    source, output = args.source_run.resolve(), args.output.resolve()
    for path in (output, args.prepared.resolve()):
        if path == source or source in path.parents:
            raise ValueError("Evaluation output and prepared plan must be outside the immutable source run")
    if args.prepare_only:
        started = time.perf_counter()
        plans, summaries = prepare(config, specs)
        write(args.prepared, {"suites": summaries, "prepare_seconds": time.perf_counter() - started,
                              "source_commit": subprocess.check_output(
                                  ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()})
        print(json.dumps(summaries, indent=2), flush=True)
        return
    prepared = read(args.prepared)["suites"]
    _, rebuilt = prepare(config, specs)
    if rebuilt != prepared:
        raise ValueError("Prepared suites differ from current inputs or worker count; run --prepare-only again")
    if workers > torch.cuda.device_count():
        raise ValueError("Requested workers exceed visible CUDA devices")
    from scripts.inference.evaluate_frames import isolated_device_mappings
    mappings = isolated_device_mappings(range(workers), torch.cuda.get_device_properties)
    output.mkdir(parents=True, exist_ok=False)
    for key in ("WANDB_SERVICE", "_WANDB_SERVICE", "WANDB_RUN_ID"):
        os.environ.pop(key, None)
    # Preserve Grana's Ray Train V1 worker-session contract.
    environment = {"TOKENIZERS_PARALLELISM": "false", "USE_TF": "0", "TRANSFORMERS_NO_TF": "1",
                   "RAY_TRAIN_V2_ENABLED": "0",
                   "CUDA_VISIBLE_DEVICES": ",".join(item["device_uuid"] for item in mappings.values()),
                   "TRAIN_ENABLE_SHARE_CUDA_VISIBLE_DEVICES": "0", "TUNE_DISABLE_SIGINT_HANDLER": "1"}
    previous_environment = {key: os.environ.get(key) for key in environment}
    os.environ.update(environment)
    import ray
    from ray.train import RunConfig, ScalingConfig, FailureConfig
    from ray.train.torch import TorchConfig, TorchTrainer
    started = time.time()
    report = {"status": "running", "driver_pid": os.getpid(), "started_unix": started,
              "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "device_mappings": mappings,
              "source_checkpoint": str(checkpoint), "prepared": prepared,
              "script_sha256": sha256(Path(__file__).read_bytes()).hexdigest()}
    write(output / "driver.json", report)
    write(output / "suite_specs.json", specs)
    write(output / "config.json", config)
    previous_sigterm = signal.getsignal(signal.SIGTERM)

    def terminate(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminate)
    try:
        ray.init(address="local", _temp_dir=tempfile.mkdtemp(prefix="grana-eval-suites-"),
                 include_dashboard=False, _node_ip_address="127.0.0.1",
                 num_cpus=workers * config["training"]["cpus_per_worker"] + 1,
                 num_gpus=workers)
        payload = {"config": config, "specs": specs, "output": str(output), "prepared": prepared,
                   "checkpoint": str(checkpoint), "step": args.step, "device_mappings": mappings, **source_info}
        trainer = TorchTrainer(worker, train_loop_config=payload,
                               torch_config=TorchConfig(timeout_s=config["training"]["distributed_timeout_seconds"]),
                               scaling_config=ScalingConfig(num_workers=workers, use_gpu=True,
                                 resources_per_worker={"CPU": config["training"]["cpus_per_worker"], "GPU": 1}),
                               run_config=RunConfig(name="checkpoint_suites", storage_path=str(output / "ray"),
                                                    failure_config=FailureConfig(max_failures=0)))
        result = trainer.fit()
        verify_completion(result, output, specs, prepared)
        report.update(status="succeeded", metrics=result.metrics)
    except BaseException as exc:
        status = "cancelled" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else (
            "incomplete" if isinstance(exc, IncompleteEvaluationError) else "failed")
        report.update(status=status, error=f"{type(exc).__name__}: {exc}")
        write(output / "driver.json", report)
        raise
    finally:
        try:
            ray.shutdown()
            if report["status"] != "succeeded":
                report["cloud_cleanup_errors"] = close_incomplete_runs(
                    output, specs, status=report["status"], error=report.get("error", "driver interrupted"))
        except Exception as exc:
            report["shutdown_error"] = f"{type(exc).__name__}: {exc}"
            if report["status"] == "succeeded":
                report["status"] = "failed"
            raise
        finally:
            signal.signal(signal.SIGTERM, previous_sigterm)
            for key, value in previous_environment.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            report["end_to_end_seconds"] = time.time() - started
            write(output / "driver.json", report)


if __name__ == "__main__":
    main()
