# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Composable recipes with distinct pretraining and task-adaptation defaults."""
from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
from typing import Any

import yaml

from grana.modeling.prompts import text_field
from grana.modeling.inputs import validate_max_pixels
from grana.data.frame_sampling import (DEFAULT_TRAIN_STRIDE_PAIRS, DEFAULT_EVAL_STRIDE_PAIRS,
                                      default_train_frame_sampling, normalize_train_frame_sampling)

DEFAULTS = {
    "model": {
        "name_or_path": "Qwen/Qwen3.5-0.8B",
        "revision": None,
        "attn_implementation": "sdpa",
        "delta_rule_backend": "auto",
        "lora_rank": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.05,
        # Training, canaries, profiling and throughput timing default to on.
        # Compare disabled checkpointing only on an explicit user request.
        "gradient_checkpointing": True,
    },
    "data": {
        "caches": [],
        "manifest": None,
        "stride_pairs": [list(pair) for pair in DEFAULT_TRAIN_STRIDE_PAIRS],
        "train_frame_sampling": default_train_frame_sampling(),
        "max_pixels": None,
        # One policy for training and evaluation; saved checkpoints carry it
        # into inference. Older recipes retain host resizing unless opted in.
        "gpu_image_resize": False,
        "views": ["top", "left", "right", "bottom"],
        "seed": 42,
    },
    "roles": {"A": 10, "B": 10, "C": 12, "D": 0},
    "loss": {
        "role_weights": {"A": 1.0, "B": 1.0, "C": 0.25, "D": 1.0},
        "frame_weights": [0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 2.0, 2.0],
        "stage": {"cdf_weight": 1.0},
        "progress": {"cdf_weight": 1.0, "ceiling_weight": 0.0},
    },
    "evaluation": {
        "enabled": False, "caches": [], "manifest": None, "every_steps": 500,
        "stride_pairs": [list(pair) for pair in DEFAULT_EVAL_STRIDE_PAIRS],
        "mode": "sampled", "batch_size": 1, "batched_vision_sdpa": False,
        "max_trajectories_per_task": 2, "windows_per_trajectory": 2, "seed": 1729,
        "renders": {"enabled": False, "max_trajectories_per_task": 4,
                    "points_per_trajectory": 24, "max_frames": 24, "replay_fps": 8},
        "deploy_dense": {"enabled": False, "max_trajectories_per_task": 4,
                         "frame_stride": 10, "include_oracle": True, "max_frames": 96, "replay_fps": 8,
                         "group_by_cache": {}},
    },
    "logging": {
        "mode": "disabled", "project": "grana", "entity": None, "group": None,
        "name": None, "job_type": None, "tags": [], "upload_checkpoint_weights": False,
    },
    "training": {
        "stage": "pretrain",
        "output_dir": "outputs/pretrain",
        "max_steps": 5000,
        # Count all role rows across workers and forwards in one optimizer update.
        "global_batch_size": 128,
        "gradient_accumulation_steps": None,
        "num_workers": 1,
        "cpus_per_worker": 8,
        "distributed_timeout_seconds": 3600,
        "learning_rate": 2e-5,
        "weight_decay": 0.01,
        "lr_schedule": "cosine",
        "warmup_steps": 100,
        "max_grad_norm": 1.0,
        "save_steps": 500,
        "profile_steps": 0,
        "prefetch_samples": True,
        "batched_vision_sdpa": False,
        "seed": 42,
        "init_checkpoint": None,
        "allow_fresh_heads": [],
    },
}

POSTTRAIN_DEFAULTS = {
    "roles": {"A": 0, "B": 10, "C": 3, "D": 3},
    "loss": {"role_weights": {"A": 1.0, "B": 1.0, "C": 10.0, "D": 1.0}},
    "training": {"stage": "posttrain", "output_dir": "outputs/posttrain", "max_steps": 200,
                 "warmup_steps": 10, "save_steps": 25},
    "evaluation": {"every_steps": 50, "max_trajectories_per_task": None, "windows_per_trajectory": 8},
}


def _merge(base: dict, supplied: dict, prefix: str = "") -> dict:
    for key, value in supplied.items():
        path = f"{prefix}{key}"
        if key not in base:
            raise ValueError(f"Unknown configuration key: {path}")
        if path == "evaluation.deploy_dense.group_by_cache":
            if not isinstance(value, dict):
                raise ValueError(f"{path} must be a mapping")
            base[key] = copy.deepcopy(value)
        elif isinstance(base[key], dict):
            if not isinstance(value, dict):
                raise ValueError(f"{path} must be a mapping")
            _merge(base[key], value, path + ".")
        else:
            base[key] = value
    return base


def _expand_location(value: Any, name: str) -> str:
    """Expand portable recipe paths while preserving relative paths and HF IDs."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty path or model identifier")
    expanded = os.path.expandvars(value)
    if "$" in expanded:
        raise ValueError(f"Unresolved environment variable in {name}: {expanded}")
    return os.path.expanduser(expanded)


def _overlay(base: dict, supplied: dict) -> dict:
    """Compose supplied layers before validating against the complete schema."""
    for key, value in supplied.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _overlay(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def _parse_values(raw: str) -> Any:
    # JSON exports use exponents such as 1e-06 that PyYAML interprets as text.
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return yaml.safe_load(raw)


def _read_recipe(path: Path, stack: tuple[Path, ...] = ()) -> tuple[dict, dict, set[Path]]:
    path = path.expanduser().resolve()
    if path in stack:
        raise ValueError("Configuration inheritance cycle: " + " -> ".join(map(str, (*stack, path))))
    values = _parse_values(path.read_text())
    if not isinstance(values, dict):
        raise ValueError(f"Configuration must be a YAML mapping: {path}")
    parents = values.pop("extends", [])
    if isinstance(parents, str):
        parents = [parents]
    if not isinstance(parents, list) or any(not isinstance(p, str) or not p.strip() for p in parents):
        raise ValueError(f"extends must be a path or a list of paths: {path}")
    combined, origins, sources = {}, {}, {path}
    for parent in parents:
        parent_path = Path(_expand_location(parent, "extends"))
        layer, locations, files = _read_recipe(path.parent / parent_path, (*stack, path))
        _overlay(combined, layer)
        origins.update(locations)
        sources.update(files)
    _overlay(combined, values)
    for section in ("data", "evaluation"):
        if isinstance(values.get(section), dict) and "manifest" in values[section]:
            origins[section] = path.parent
    return combined, origins, sources


def config_sources(path: str | Path) -> list[Path]:
    """Include every inherited recipe in a launcher's source fingerprint."""
    return sorted(_read_recipe(Path(path))[2])


def _supplied_values(path: Path, overrides: list[str]) -> tuple[dict, dict]:
    values, origins, _ = _read_recipe(path)
    for override in overrides:
        key, sep, raw = override.partition("=")
        pieces = key.split(".")
        if not sep or any(not piece for piece in pieces):
            raise ValueError(f"Override must be dotted.key=value: {override!r}")
        layer = cursor = {}
        for part in pieces[:-1]:
            cursor[part] = {}
            cursor = cursor[part]
        cursor[pieces[-1]] = _parse_values(raw)
        _overlay(values, layer)
        for section in ("data", "evaluation"):
            if isinstance(layer.get(section), dict) and "manifest" in layer[section]:
                origins[section] = path.parent
    return values, origins


def _resolve_batch(cfg: dict) -> None:
    training, roles = cfg["training"], cfg["roles"]
    for role, count in roles.items():
        _integer(count, f"roles.{role}", 0)
    if not sum(roles.values()):
        raise ValueError("At least one role must be active")
    if roles["D"] and (roles["A"] % 2 or roles["B"] % 2 or roles["C"] < roles["D"]):
        raise ValueError("D requires even aggregate A/B counts and C>=D>0")
    _integer(training["num_workers"], "training.num_workers")
    target = training["global_batch_size"]
    if target is None:
        return  # Explicit accumulation/count mode, including historical recipes.
    _integer(target, "training.global_batch_size")
    # With D, one complete two-forward cycle contains exactly sum(roles).
    # For odd independent C quotas the peak physical batch is not the mean.
    width = sum(roles.values()) * training["num_workers"]
    cycles, remainder = divmod(target, width)
    accumulation = cycles * 2 if roles["D"] else cycles
    if remainder or accumulation < 1:
        cycles_required = "complete even forward cycles" if roles["D"] else "complete forward cycles"
        raise ValueError(f"training.global_batch_size must fit {cycles_required}; "
                         f"got target={target}, cycle_rows_across_workers={width}")
    supplied = training["gradient_accumulation_steps"]
    if supplied is not None:
        _integer(supplied, "training.gradient_accumulation_steps")
        if supplied != accumulation:
            raise ValueError("training.gradient_accumulation_steps conflicts with training.global_batch_size; "
                             "use null accumulation to derive it, or null global_batch_size for explicit count mode")
    training["gradient_accumulation_steps"] = accumulation


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    values, origins = _supplied_values(path, overrides or [])
    supplied_training = values.get("training", {})
    if not isinstance(supplied_training, dict):
        raise ValueError("training must be a mapping")
    stage = supplied_training.get("stage", "pretrain")
    if stage not in ("pretrain", "posttrain"):
        raise ValueError("training.stage must be pretrain or posttrain")
    defaults = copy.deepcopy(DEFAULTS)
    if stage == "posttrain":
        _merge(defaults, POSTTRAIN_DEFAULTS)
    # Old fully resolved runs and user-owned explicit GA recipes retain their
    # original count semantics; an inherited global target still takes priority.
    if supplied_training.get("gradient_accumulation_steps") is not None and "global_batch_size" not in supplied_training:
        defaults["training"]["global_batch_size"] = None
    result = _merge(defaults, values)
    _resolve_batch(result)
    result["model"]["name_or_path"] = _expand_location(result["model"]["name_or_path"], "model.name_or_path")
    result["training"]["output_dir"] = _expand_location(result["training"]["output_dir"], "training.output_dir")
    if result["training"]["init_checkpoint"] is not None:
        result["training"]["init_checkpoint"] = _expand_location(result["training"]["init_checkpoint"], "training.init_checkpoint")
    groups = result["evaluation"]["deploy_dense"]["group_by_cache"]
    result["evaluation"]["deploy_dense"]["group_by_cache"] = {
        str(Path(_expand_location(cache, "evaluation.deploy_dense.group_by_cache key")).resolve()): group
        for cache, group in groups.items()
    }
    for section in ("data", "evaluation"):
        manifest = result[section]["manifest"]
        if manifest:
            manifest_path = Path(_expand_location(manifest, f"{section}.manifest"))
            if not manifest_path.is_absolute():
                manifest_path = origins.get(section, path.parent) / manifest_path
            manifest_path = manifest_path.resolve()
            body = yaml.safe_load(manifest_path.read_text())
            if not isinstance(body, dict) or set(body) != {"caches"} or not isinstance(body["caches"], list):
                raise ValueError(f"Manifest must contain only a caches list: {manifest_path}")
            entries = copy.deepcopy(body["caches"])
            for index, entry in enumerate(entries):
                if not isinstance(entry, (str, dict)) or isinstance(entry, dict) and "path" not in entry:
                    raise ValueError(f"Cache manifest entry must be a path or mapping with path: {manifest_path}")
                raw = entry if isinstance(entry, str) else entry["path"]
                resolved = Path(_expand_location(raw, f"{section} cache path"))
                if not resolved.is_absolute():
                    resolved = manifest_path.parent / resolved
                if isinstance(entry, str):
                    entries[index] = str(resolved.resolve())
                else:
                    entry["path"] = str(resolved.resolve())
            if result[section]["caches"]:
                raise ValueError(f"Choose {section}.manifest or {section}.caches, not both")
            result[section]["caches"] = entries
            result[section]["manifest"] = str(manifest_path)
    validate_config(result)
    return result


def _integer(value: Any, name: str, minimum: int = 1) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _views(value: Any, name: str) -> None:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must contain unique camera names")
    names = [text_field(view, name) for view in value]
    if any(clean != raw or len(clean) > 128 or "<|" in clean or "|>" in clean
           for raw, clean in zip(value, names, strict=True)):
        raise ValueError(f"{name} must have no surrounding whitespace or token delimiters and at most 128 characters")
    if len(set(names)) != len(names):
        raise ValueError(f"{name} must contain unique camera names")


def validate_config(cfg: dict) -> None:
    training, data, roles, loss = (cfg[key] for key in ("training", "data", "roles", "loss"))
    unknown_data_keys = set(data) - set(DEFAULTS["data"])
    if unknown_data_keys:
        raise ValueError(f"Unknown configuration key: data.{sorted(unknown_data_keys)[0]}")
    for key in ("max_steps", "gradient_accumulation_steps", "num_workers", "cpus_per_worker", "save_steps", "distributed_timeout_seconds"):
        _integer(training[key], f"training.{key}")
    _integer(training.get("profile_steps", 0), "training.profile_steps", 0)
    if type(training.get("prefetch_samples", True)) is not bool:
        raise ValueError("training.prefetch_samples must be boolean")
    if not isinstance(training.get("batched_vision_sdpa", False), bool):
        raise ValueError("training.batched_vision_sdpa must be boolean")
    if roles["D"] and training["gradient_accumulation_steps"] % 2:
        raise ValueError("gradient_accumulation_steps must be even when Role D is enabled")
    if training.get("global_batch_size") is not None:
        _integer(training["global_batch_size"], "training.global_batch_size")
        multiplier = training["gradient_accumulation_steps"] // (2 if roles["D"] else 1)
        actual = sum(roles.values()) * multiplier * training["num_workers"]
        if actual != training["global_batch_size"]:
            raise ValueError("Resolved accumulation does not match training.global_batch_size")
    for role, count in roles.items():
        _integer(count, f"roles.{role}", 0)
    if sum(roles.values()) == 0:
        raise ValueError("At least one role must be active")
    if roles["D"]:
        if roles["A"] % 2 or roles["B"] % 2 or roles["C"] < roles["D"]:
            raise ValueError("D requires even aggregate A/B counts and C>=D>0")
        if training["gradient_accumulation_steps"] % 2:
            raise ValueError("D requires even gradient_accumulation_steps (complete ABC→ABD pairs)")
    if training["stage"] not in ("pretrain", "posttrain"):
        raise ValueError("training.stage must be pretrain or posttrain")
    # The stage chooses supervision defaults. A task may start from its raw
    # pretrained backbone, or explicitly transfer same-size Grana weights.
    if not isinstance(training["output_dir"], str) or not training["output_dir"].strip():
        raise ValueError("training.output_dir must be a nonempty path")
    if not isinstance(data["caches"], list) or not data["caches"]:
        raise ValueError("data.caches must list existing cached datasets")
    validate_max_pixels(data["max_pixels"])
    if type(data.get("gpu_image_resize", False)) is not bool:
        raise ValueError("data.gpu_image_resize must be boolean")
    for section in ("data", "evaluation"):
        pairs = cfg[section]["stride_pairs"]
        if not isinstance(pairs, list) or not pairs:
            raise ValueError(f"{section}.stride_pairs must be a nonempty list")
        for pair in pairs:
            if not isinstance(pair, list) or len(pair) != 2:
                raise ValueError(f"Each {section}.stride_pairs entry is [context_stride, tail_stride]")
            for value in pair:
                _integer(value, f"{section}.stride_pairs stride")
        if len({tuple(pair) for pair in pairs}) != len(pairs):
            raise ValueError(f"{section}.stride_pairs must contain unique pairs")
    data["train_frame_sampling"] = normalize_train_frame_sampling(data["train_frame_sampling"])
    _views(data["views"], "data.views")
    for section in ("data", "evaluation"):
        for entry in cfg[section]["caches"]:
            if isinstance(entry, dict) and "views" in entry:
                _views(entry["views"], f"{section} cache.views")
    for key in ("learning_rate", "max_grad_norm"):
        value = training[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"training.{key} must be finite and positive")
    for key in ("weight_decay",):
        value = training[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"training.{key} must be finite and nonnegative")
    if len(loss["frame_weights"]) != 8 or any(not isinstance(w, (int, float)) or not math.isfinite(w) or w <= 0 for w in loss["frame_weights"]):
        raise ValueError("loss.frame_weights must contain eight finite positive weights")
    for role, value in loss["role_weights"].items():
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"loss.role_weights.{role} must be finite and nonnegative")
    for head in ("stage", "progress"):
        objective = loss[head]
        for key in (("cdf_weight", "ceiling_weight") if head == "progress" else ("cdf_weight",)):
            value = objective[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"loss.{head}.{key} must be finite and nonnegative")
    _integer(training["warmup_steps"], "training.warmup_steps", 0)
    if training["warmup_steps"] >= training["max_steps"]:
        raise ValueError("warmup_steps must be smaller than max_steps")
    if training["lr_schedule"] not in ("constant", "cosine"):
        raise ValueError("training.lr_schedule must be constant or cosine")
    for section in ("data", "training", "evaluation"):
        _integer(cfg[section]["seed"], f"{section}.seed", 0)
    evaluation = cfg["evaluation"]
    if not isinstance(evaluation["enabled"], bool):
        raise ValueError("evaluation.enabled must be boolean")
    if not isinstance(evaluation.get("batched_vision_sdpa", False), bool):
        raise ValueError("evaluation.batched_vision_sdpa must be boolean")
    if evaluation["mode"] not in ("sampled", "dense"):
        raise ValueError("evaluation.mode must be sampled or dense")
    _integer(evaluation.get("batch_size", 1), "evaluation.batch_size")
    for key in ("every_steps", "windows_per_trajectory"):
        _integer(evaluation[key], f"evaluation.{key}")
    if evaluation["max_trajectories_per_task"] is not None:
        _integer(evaluation["max_trajectories_per_task"], "evaluation.max_trajectories_per_task")
    if evaluation["enabled"] and not evaluation["caches"]:
        raise ValueError("Enabled evaluation requires held-out caches or a manifest")
    renders = evaluation["renders"]
    if not isinstance(renders["enabled"], bool):
        raise ValueError("evaluation.renders.enabled must be boolean")
    if renders["enabled"] and not evaluation["enabled"]:
        raise ValueError("Enabled evaluation renders require enabled held-out evaluation")
    for key in ("max_trajectories_per_task", "max_frames"):
        _integer(renders[key], f"evaluation.renders.{key}")
    _integer(renders["points_per_trajectory"], "evaluation.renders.points_per_trajectory", 2)
    fps = renders["replay_fps"]
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not math.isfinite(fps) or not .1 <= fps <= 20:
        raise ValueError("evaluation.renders.replay_fps must be in [0.1, 20]")
    dense = evaluation["deploy_dense"]
    for key in ("enabled", "include_oracle"):
        if not isinstance(dense[key], bool):
            raise ValueError(f"evaluation.deploy_dense.{key} must be boolean")
    if dense["enabled"] and not evaluation["enabled"]:
        raise ValueError("Enabled dense evaluation requires enabled held-out evaluation")
    if dense["max_trajectories_per_task"] is not None:
        _integer(dense["max_trajectories_per_task"], "evaluation.deploy_dense.max_trajectories_per_task")
    for key in ("frame_stride", "max_frames"):
        _integer(dense[key], f"evaluation.deploy_dense.{key}")
    groups = dense["group_by_cache"]
    if not isinstance(groups, dict) or any(
        not isinstance(path, str) or not path.strip() or not isinstance(group, str) or not group.strip()
        for path, group in groups.items()
    ):
        raise ValueError("evaluation.deploy_dense.group_by_cache must map cache paths to nonempty group names")
    if evaluation["mode"] == "dense" and not (evaluation["enabled"] and dense["enabled"]):
        raise ValueError("evaluation.mode=dense requires enabled held-out evaluation and deploy_dense")
    fps = dense["replay_fps"]
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not math.isfinite(fps) or not .1 <= fps <= 20:
        raise ValueError("evaluation.deploy_dense.replay_fps must be in [0.1, 20]")
    logging = cfg["logging"]
    if logging["mode"] not in ("disabled", "offline", "online"):
        raise ValueError("logging.mode must be disabled, offline, or online")
    if not isinstance(logging["project"], str) or not logging["project"].strip():
        raise ValueError("logging.project must be nonempty")
    if not isinstance(logging["tags"], list) or any(not isinstance(tag, str) for tag in logging["tags"]):
        raise ValueError("logging.tags must be a list of strings")
    if not isinstance(logging["upload_checkpoint_weights"], bool):
        raise ValueError("logging.upload_checkpoint_weights must be boolean")
    for key in ("entity", "group", "name", "job_type"):
        if logging[key] is not None and (not isinstance(logging[key], str) or not logging[key].strip()):
            raise ValueError(f"logging.{key} must be nonempty text or null")
    if cfg["model"]["attn_implementation"] != "sdpa":
        raise ValueError("This release supports the qualified sdpa attention backend")
    revision = cfg["model"]["revision"]
    if revision is not None and (not isinstance(revision, str) or not revision.strip()):
        raise ValueError("model.revision must be nonempty text or null")
    if not isinstance(cfg["model"]["gradient_checkpointing"], bool):
        raise ValueError("model.gradient_checkpointing must be boolean")
    if cfg["model"]["delta_rule_backend"] not in ("auto", "torch"):
        raise ValueError("model.delta_rule_backend must be auto or torch")
    for key in ("lora_rank", "lora_alpha"):
        _integer(cfg["model"][key], f"model.{key}")
    if not 0 <= cfg["model"]["lora_dropout"] < 1:
        raise ValueError("model.lora_dropout must be in [0, 1)")


def physical_batch_size(cfg: dict) -> int:
    return max(forward_batch_sizes(cfg))


def forward_batch_sizes(cfg: dict) -> tuple[int, ...]:
    """Physical row counts of a single scheduling cycle on one worker."""
    roles = cfg["roles"]
    if not roles["D"]:
        return (sum(roles.values()),)
    independent = roles["C"] - roles["D"]
    paired = roles["A"] // 2 + roles["B"] // 2 + roles["D"]
    return (paired + (independent + 1) // 2, paired + independent // 2)


def supervision_budget(cfg: dict) -> dict:
    """Expose actual independent stage supervision, including C-derived D rows."""
    training, roles = cfg["training"], cfg["roles"]
    accumulation, workers = training["gradient_accumulation_steps"], training["num_workers"]
    multiplier = (accumulation // 2 if roles["D"] else accumulation) * workers
    counts = {role: count * multiplier for role, count in roles.items()}
    return {"stage": training["stage"], "physical_batch_size_per_worker": physical_batch_size(cfg),
            "forward_batch_sizes_per_worker": list(forward_batch_sizes(cfg)),
            "num_workers": workers, "gradient_accumulation_steps": accumulation,
            "global_batch_size": sum(counts.values()), "role_rows_per_update_global": counts,
            "role_rows_over_run_global": {role: count * training["max_steps"] for role, count in counts.items()},
            "independent_C_windows_per_update": counts["C"],
            "paired_C_rows_per_update_global": counts["D"],
            "unpaired_C_rows_per_update_global": counts["C"] - counts["D"],
            "stage_label_slots_per_update": 8 * counts["C"],
            "semantics": "A/B/C are independent windows; D reuses C observations and adds no independent stage labels."}
