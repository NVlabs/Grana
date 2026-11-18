# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Portable launch, status, wait, and report commands for training campaigns."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time

from grana._paths import PROJECT_ROOT
from .campaign import Coordinator, atomic_json, identity_is_live, last_json_line

SIZES = {size.lower(): size for size in ("0.8B", "2B", "4B", "9B", "27B")}
MODEL_FAMILIES = {"qwen35": tuple(SIZES.values()), "qwen38": ("27B",)}
TERMINAL = {"succeeded", "failed", "cancelled", "skipped", "interrupted"}


def size_name(value):
    try:
        return SIZES[value.lower()]
    except KeyError:
        raise argparse.ArgumentTypeError("Model size must be 0.8B, 2B, 4B, 9B, or 27B") from None


def positive_seconds(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("Seconds must be finite and positive")
    return number


def lane(value):
    size, sep, raw = value.partition("=")
    gpus = raw.split(",")
    if not sep or not gpus or len(set(gpus)) != len(gpus) or any(
        re.fullmatch(r"(?:[0-9]+|(?:GPU|MIG)-[A-Za-z0-9/-]+)", gpu) is None for gpu in gpus
    ):
        raise argparse.ArgumentTypeError("Use SIZE=GPU[,GPU,...] with distinct CUDA indices/UUIDs")
    return size_name(size), gpus


def posttrain_recipe(value):
    name, separator, location = value.partition("=")
    if (not separator or not location.strip() or name == "pretrain"
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name) is None):
        raise argparse.ArgumentTypeError("Use NAME=CONFIG_PATH; NAME is a unique letter/digit/underscore/hyphen label other than pretrain")
    return name, Path(location).expanduser().resolve()


def read_json(path):
    return json.loads(Path(path).read_text())


def resolve_control(path):
    path = Path(path).expanduser().resolve()
    return path / "control" if (path / "control/campaign.json").is_file() else path


def load_campaign(path):
    control = resolve_control(path)
    source = control / "campaign.json"
    if not source.is_file():
        source = control / "plan.json"  # Also observes retained model-support runs.
    return control, read_json(source), read_json(control / "status.json")


def validate_checkpoint_transfer(source, target, *, source_name, target_name):
    """Preflight declared model identity without loading a model or checkpoint.

    The weights loader checks adapter rank/scaling, tokenizer IDs and tensor
    shapes. Model location/revision select the tokenizer and backbone structure;
    require the same declared identity for a campaign's mandatory transfer.
    Runtime checks still verify actual tokens, tensor coverage and digests.
    Objectives, input preprocessing and explicit fresh-head allowances are not
    part of adapter compatibility and may differ between these phases.
    """
    for field in ("name_or_path", "revision", "lora_rank", "lora_alpha"):
        if source[field] != target[field]:
            raise ValueError(
                f"Campaign checkpoint transfer {source_name!r} -> {target_name!r} requires matching "
                f"model.{field}: pretraining={source[field]!r}, adaptation={target[field]!r}")


def freeze_runtime_locations(config):
    """Preserve planning-CWD path semantics when train.sh switches to the repo."""
    for section in ("data", "evaluation"):
        for index, entry in enumerate(config[section]["caches"]):
            raw = entry.get("path") if isinstance(entry, dict) else entry
            if not isinstance(raw, str) or not raw.strip():
                raise ValueError(f"{section} cache entry requires a nonempty path")
            # Inline caches are CWD-relative; manifest entries are already
            # absolute. Match CachedTrajectories' expanduser/resolve behavior.
            resolved = str(Path(raw).expanduser().resolve())
            if isinstance(entry, str):
                config[section]["caches"][index] = resolved
            else:
                entry["path"] = resolved
    location = config["model"]["name_or_path"]
    candidate = Path(location)
    # Bare org/name values remain HF identifiers unless a local directory
    # actually exists, matching from_pretrained's local-directory precedence.
    if candidate.is_absolute() or location.startswith(("./", "../")) or candidate.is_dir():
        config["model"]["name_or_path"] = str(candidate.resolve())


def build_plan(args):
    import yaml
    from grana.config import _read_recipe, load_config, supervision_budget

    if args.timeout_seconds <= 20:
        raise ValueError("Campaign timeout must exceed the 20-second cleanup reserve")
    sizes = [size for size, _ in args.lane]
    model_family = getattr(args, "model_family", "qwen35")
    if model_family not in MODEL_FAMILIES:
        raise ValueError(f"Unknown model family: {model_family}")
    unsupported_sizes = set(sizes) - set(MODEL_FAMILIES[model_family])
    if unsupported_sizes:
        raise ValueError(f"{model_family} supports model sizes: {', '.join(MODEL_FAMILIES[model_family])}")
    gpus = [gpu for _, devices in args.lane for gpu in devices]
    if len(set(sizes)) != len(sizes) or len(set(gpus)) != len(gpus):
        raise ValueError("Each size and each physical CUDA device may appear in only one lane")
    posttrains = args.posttrain_config
    labels = [label for label, _ in posttrains]
    if len(set(labels)) != len(labels) or "pretrain" in labels:
        raise ValueError("Posttraining recipe names must be unique and cannot be pretrain")
    recipes = [("pretrain", args.pretrain_config.expanduser().resolve()), *posttrains]
    models = {}
    for supplied in args.model:
        raw_size, sep, location = supplied.partition("=")
        size = size_name(raw_size)
        if not sep or not location.strip() or size not in sizes or size in models:
            raise ValueError("--model must be a unique SIZE=HF_ID_OR_LOCAL_PATH for a selected lane")
        models[size] = location
    root = args.output_root.expanduser().resolve()
    if root.exists():
        raise FileExistsError(f"Use a fresh campaign output root: {root}")
    name = args.name or root.name
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name) is None:
        raise ValueError("Campaign name must start with a letter/digit and contain only letters, digits, dots, hyphens or underscores")
    control = root / "control"
    jobs, configs = [], {}
    for size, devices in args.lane:
        label = size.lower().replace(".", "p")
        pretrain_name = f"{name}-{label}-pretrain"
        pretrain_checkpoint = None
        previous_by_device = {}
        model_recipe = PROJECT_ROOT / "configs/models" / model_family / f"{size.lower()}.yaml"
        model_defaults = yaml.safe_load(model_recipe.read_text())["model"]
        for index, (task, recipe) in enumerate(recipes):
            pretrain = index == 0
            job_name = f"{name}-{label}-{task}"
            output = root / "runs" / job_name
            assigned = devices if pretrain else [devices[(index - 1) % len(devices)]]
            dependency = None if pretrain else previous_by_device.get(assigned[0], pretrain_name)
            values = {
                "model.name_or_path": models.get(size, model_defaults["name_or_path"]),
                "training.output_dir": str(output), "training.num_workers": len(assigned),
                "training.init_checkpoint": None if pretrain else str(pretrain_checkpoint),
                "logging.group": args.group or name, "logging.name": job_name,
                "logging.tags": [f"{model_family}-{size.lower()}", task, "support-smoke" if args.smoke else "campaign"],
            }
            for key, value in (("mode", args.logging_mode or ("disabled" if args.smoke else None)),
                               ("entity", args.entity), ("project", args.project)):
                if value is not None:
                    values[f"logging.{key}"] = value
            if args.smoke:
                values.update({"training.max_steps": 2, "training.save_steps": 2, "training.warmup_steps": 0,
                               "training.global_batch_size": None, "training.gradient_accumulation_steps": 2,
                               "roles": {"A": 1, "B": 1, "C": 1, "D": 0} if pretrain else
                                        {"A": 0, "B": 2, "C": 1, "D": 1},
                               "evaluation.every_steps": 2,
                               "evaluation.max_trajectories_per_task": 1, "evaluation.windows_per_trajectory": 1,
                               "evaluation.renders.max_trajectories_per_task": 1,
                               "evaluation.renders.points_per_trajectory": 4,
                               "evaluation.renders.max_frames": 4, "evaluation.renders.replay_fps": 4,
                               "evaluation.deploy_dense.max_trajectories_per_task": 1,
                               "evaluation.deploy_dense.frame_stride": 1000000,
                               "evaluation.deploy_dense.max_frames": 2})
            stage_overrides = getattr(args, "pretrain_set" if pretrain else "posttrain_set", [])
            supplied_model = _read_recipe(recipe)[0].get("model", {})
            if not isinstance(supplied_model, dict):
                raise ValueError(f"model must be a mapping: {recipe}")
            # Size overlays provide defaults; inherited/user recipe backend
            # choices and explicit phase overrides retain their precedence.
            model_overrides = [f"model.{key}={json.dumps(value)}" for key, value in model_defaults.items()
                               if key not in supplied_model]
            cfg = load_config(recipe, model_overrides + stage_overrides
                              + [f"{key}={json.dumps(value)}" for key, value in values.items()])
            freeze_runtime_locations(cfg)
            if cfg["training"]["stage"] != ("pretrain" if pretrain else "posttrain"):
                raise ValueError("Campaign recipes and phase overrides must preserve the expected training.stage")
            if not pretrain:
                validate_checkpoint_transfer(configs[pretrain_name]["model"], cfg["model"],
                                             source_name=pretrain_name, target_name=job_name)
            steps = cfg["training"]["max_steps"]
            if pretrain:
                pretrain_checkpoint = output / f"checkpoint-{steps:06d}"
            else:
                previous_by_device[assigned[0]] = job_name
            manifests = [cfg[section]["manifest"] for section in ("data", "evaluation") if cfg[section]["manifest"]]
            if args.check_data:
                from grana.data.cache import CachedTrajectories
                from .evaluation import build_evaluation_plan
                training_data = CachedTrajectories(cfg["data"]["caches"], views=cfg["data"]["views"])
                if cfg["evaluation"]["enabled"]:
                    heldout = CachedTrajectories(cfg["evaluation"]["caches"], views=cfg["data"]["views"])
                    build_evaluation_plan(heldout, cfg["evaluation"], cfg["data"],
                                          training_dataset=training_data, roles=cfg["roles"])
            for section in ("data", "evaluation"):
                cfg[section]["manifest"] = None  # Freeze resolved caches in each runtime recipe.
            config_path = control / "configs" / f"{label}-{task}.yaml"
            job = {"name": job_name, "size": size, "task": task,
                   "stage": "pretrain" if pretrain else "posttrain", "gpus": assigned,
                   "model": cfg["model"]["name_or_path"], "config": str(config_path),
                   "portable_config": str(recipe), "model_config": str(model_recipe), "manifests": manifests,
                   "output_dir": str(output), "expected_steps": steps,
                   "expected_checkpoint": str(output / f"checkpoint-{steps:06d}"),
                   "dependency": dependency, "source_pretrain": None if pretrain else pretrain_name,
                   "init_checkpoint": cfg["training"]["init_checkpoint"],
                   "supervision_budget": supervision_budget(cfg),
                   "log": str(control / "logs" / f"{label}-{task}.log"),
                   "receipt_path": str(control / "receipts" / f"{label}-{task}.json")}
            jobs.append(job)
            configs[job_name] = cfg
    plan = {"schema": "grana-training-campaign-v1", "campaign": name, "repo": str(PROJECT_ROOT),
            "output_root": str(root), "control_dir": str(control),
            "deadline_utc": (datetime.now(timezone.utc) + timedelta(seconds=args.timeout_seconds)).isoformat(),
            "cleanup_reserve_seconds": 20, "jobs": jobs,
            "scope": "Two-update support smoke; not full training or quality qualification" if args.smoke else
                     "User-supplied pretraining and optional independent task adaptation; recipe-defined budgets",
            "environment": {"TOKENIZERS_PARALLELISM": "false", "NCCL_NET_PLUGIN": "none",
                            "NCCL_IB_DISABLE": "1", "NCCL_SOCKET_FAMILY": "AF_INET",
                            "NCCL_NVLS_ENABLE": "0", "NCCL_CUMEM_ENABLE": "0", "NCCL_DEBUG": "WARN"}}
    return plan, configs


def prepare_run(plan, configs):
    import yaml
    from grana.config import config_sources
    from .runtime_fingerprint import capture_runtime_fingerprint

    root, control = Path(plan["output_root"]), Path(plan["control_dir"])
    # mkdir is exclusive even if another process races with the dry-run check.
    root.mkdir(parents=True, exist_ok=False)
    for name in ("configs", "logs", "receipts"):
        (control / name).mkdir(parents=True)
    runtime_path = control / "runtime_fingerprint.json"
    atomic_json(runtime_path, capture_runtime_fingerprint())
    plan["source_fingerprint"] = str(control / "runtime_source_fingerprint.json")
    plan["runtime_fingerprint"] = str(runtime_path)
    files = list((PROJECT_ROOT / "grana").rglob("*.py"))
    files += [path for path in (PROJECT_ROOT / "scripts").rglob("*") if path.is_file() and path.suffix in (".py", ".sh")]
    files.append(runtime_path)
    for job in plan["jobs"]:
        path = Path(job["config"])
        path.write_text(yaml.safe_dump(configs[job["name"]], sort_keys=False))
        files += [path, *config_sources(job["portable_config"]),
                  *config_sources(job["model_config"]), *map(Path, job["manifests"])]
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(set(files))}
    atomic_json(control / "runtime_source_fingerprint.json", {"files_sha256": hashes})
    atomic_json(control / "campaign.json", plan)
    return control


def status_snapshot(control):
    control, plan, state = load_campaign(control)
    live = identity_is_live(state.get("coordinator"))
    recorded_status = state["status"]
    interrupted = recorded_status == "running" and not live
    effective_status = "interrupted" if interrupted else recorded_status
    completion = None
    if plan.get("completion_command") and recorded_status == "succeeded":
        receipt_path = control / "acceptance.json"
        watchdog_path = control / "watchdog.json"
        receipt = read_json(receipt_path) if receipt_path.is_file() else {}
        watchdog = read_json(watchdog_path) if watchdog_path.is_file() else {}
        if receipt.get("finished_utc"):
            completion = "succeeded" if receipt.get("passed") else "failed"
        elif watchdog.get("status") in ("succeeded", "failed", "cancelled"):
            completion = watchdog["status"]
        else:
            completion = "auditing" if watchdog.get("status") == "auditing" or receipt.get("started_utc") else "awaiting_acceptance"
        effective_status = completion
    owned = {record["pid"]: record for record in
             [*state.get("observed_owned_processes", []), *state.get("owned_live_processes", [])]
             if identity_is_live(record)}
    jobs = []
    for job in plan["jobs"]:
        observed = state["jobs"][job["name"]]
        metric = last_json_line(Path(job["output_dir"]) / "training_metrics.jsonl") or {}
        jobs.append({"name": job["name"], "stage": job["stage"], "size": job["size"],
                     "status": "interrupted" if interrupted and observed["status"] == "running" else observed["status"],
                     "recorded_status": observed["status"], "optimizer_steps": metric.get("optimizer_steps", 0),
                     "expected_steps": job["expected_steps"], "training_compute_seconds": metric.get("time/training_seconds"),
                     "elapsed_seconds": metric.get("elapsed_seconds"), "log": observed.get("active_log", job["log"]),
                     "attempts": len(observed.get("attempts", [])), "restart_mode": observed.get("restart_mode"),
                     "error": observed.get("error") or observed.get("reason")})
    return {"campaign": plan["campaign"], "control_dir": str(control),
            "status": effective_status, "recorded_status": recorded_status,
            "completion_status": completion,
            "coordinator_live": live, "owned_live_processes": len(owned),
            "deadline_utc": plan["deadline_utc"], "jobs": jobs}


def wait_for_campaign(control, *, until="all", timeout_seconds=28800, poll_seconds=10):
    started = time.monotonic()
    while True:
        try:
            snapshot = status_snapshot(control)
        except FileNotFoundError:
            snapshot = {"status": "waiting_for_status", "control_dir": str(control), "jobs": []}
        print(json.dumps(snapshot), flush=True)
        selected = [job for job in snapshot["jobs"] if until == "all" or job["stage"] == "pretrain"]
        ready = selected and all(job["status"] == "succeeded" for job in selected)
        if ready and (until == "pretrain" or (snapshot["status"] == "succeeded"
                                               and not snapshot["coordinator_live"]
                                               and snapshot["owned_live_processes"] == 0)):
            # A stale success label alone is not sufficient completion evidence.
            _, plan, _ = load_campaign(control)
            for job in plan["jobs"]:
                if until == "all" or job["stage"] == "pretrain":
                    result = read_json(Path(job["output_dir"]) / "result.json")
                    if (result.get("status") != "completed" or result.get("optimizer_steps") != job["expected_steps"]
                            or result.get("checkpoint_reload_equal") is not True
                            or result.get("checkpoint_prediction_parity") is not True):
                        return 1
            return 0
        if (any(job["status"] in TERMINAL - {"succeeded"} for job in selected)
                or snapshot["status"] in TERMINAL - {"succeeded"}
                or snapshot.get("coordinator_live") is False and not ready):
            return 1
        remaining = timeout_seconds - (time.monotonic() - started)
        if remaining <= 0:
            return 124  # An observer timeout never signals or stops training.
        time.sleep(min(poll_seconds, remaining))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Launch a new foreground campaign through scripts/train.sh")
    run.add_argument("--lane", type=lane, action="append", required=True, metavar="SIZE=GPU[,GPU,...]")
    run.add_argument("--pretrain-config", type=Path, required=True,
                     help="User-owned pretraining recipe; no dataset inventory is bundled")
    run.add_argument("--posttrain-config", type=posttrain_recipe, action="append", default=[], metavar="NAME=PATH",
                     help="Optional named adaptation recipe; repeat for independent tasks")
    run.add_argument("--model", action="append", default=[], metavar="SIZE=HF_ID_OR_LOCAL_PATH")
    run.add_argument("--model-family", choices=tuple(MODEL_FAMILIES), default="qwen35",
                     help="Model overlays and logging family; qwen38 currently supports 27B")
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--name")
    run.add_argument("--timeout-seconds", type=positive_seconds, default=28800)
    run.add_argument("--smoke", action="store_true", help="Two updates/stage; bound evaluation if enabled by the supplied recipe")
    run.add_argument("--pretrain-set", action="append", default=[], metavar="KEY=VALUE",
                     help="Override pretraining defaults; campaign model/output/device identity remains authoritative")
    run.add_argument("--posttrain-set", action="append", default=[], metavar="KEY=VALUE",
                     help="Override task-adaptation defaults; checkpoint dependency is computed from pretraining")
    run.add_argument("--dry-run", action="store_true", help="Resolve/validate configs without writes or training")
    run.add_argument("--check-data", action="store_true", help="Check caches and held-out separation before launch")
    run.add_argument("--logging-mode", choices=("online", "offline", "disabled"))
    run.add_argument("--entity", default=os.environ.get("WANDB_ENTITY"))
    run.add_argument("--project", help="Override recipe logging project")
    run.add_argument("--group")
    for command in ("status", "wait", "report", "audit-wandb"):
        child = commands.add_parser(command)
        child.add_argument("--control-dir", type=Path, required=True, help="Control directory or campaign output root")
        if command == "wait":
            child.add_argument("--until", choices=("pretrain", "all"), default="all")
            child.add_argument("--timeout-seconds", type=positive_seconds, default=28800)
            child.add_argument("--poll-seconds", type=positive_seconds, default=10)
        if command in ("report", "audit-wandb"):
            child.add_argument("--output", type=Path, help="Write a new JSON report; default prints only")
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            plan, configs = build_plan(args)
            if args.dry_run:
                print(json.dumps({"training_launched": False, "files_written": False, "plan": plan,
                                  "delta_rule_backends": {name: cfg["model"]["delta_rule_backend"] for name, cfg in configs.items()}}, indent=2))
                return 0
            from grana.runtime import require_owned_environment
            require_owned_environment()
            prepare_run(plan, configs)
            return Coordinator(plan).run()
        if args.command == "status":
            print(json.dumps(status_snapshot(args.control_dir), indent=2))
            return 0
        if args.command == "wait":
            return wait_for_campaign(args.control_dir, until=args.until, timeout_seconds=args.timeout_seconds,
                                     poll_seconds=args.poll_seconds)
        from .campaign_report import report_campaign, audit_wandb
        report = report_campaign(args.control_dir) if args.command == "report" else audit_wandb(args.control_dir)
        if args.output:
            with args.output.expanduser().open("x") as stream:
                stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
            print(json.dumps({"passed": report["passed"], "jobs": len(report["jobs"]),
                              "failures": report["failures"], "report": str(args.output.expanduser().resolve())}, indent=2))
        else:
            print(json.dumps(report, indent=2, allow_nan=False))
        return 0 if report["passed"] else 1
    except (OSError, ValueError, KeyError, RuntimeError, argparse.ArgumentTypeError) as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
