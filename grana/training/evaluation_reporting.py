# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Name measured deployment quality separately from conditional-head diagnostics."""
from __future__ import annotations

import math
import re
from collections import defaultdict


CONDITIONAL_RMSE = {
    "role_B_poi_rmse": "diagnostics/progress/B_oracle_focus_relative_rmse",
    "role_D_poi_rmse": "diagnostics/progress/D_predicted_focus_relative_rmse",
}


def diagnostic_metric_aliases(metrics: dict) -> dict:
    """Keep historical scalar names while giving new consumers explicit labels."""
    return {alias: metrics[key] for key, alias in CONDITIONAL_RMSE.items() if key in metrics}


EVALUATION_LOG_SCHEMA = "grana-evaluation-log-v1"
COMPACT_EVALUATION_SCHEMA = "grana-compact-eval-v3"
SUPPORTED_COMPACT_EVALUATION_SCHEMAS = frozenset({"grana-compact-eval-v2", COMPACT_EVALUATION_SCHEMA})
EVALUATION_PURPOSE_PREFIXES = {"primary": "eval/", "dense": "eval/dense/", "renders": "eval/render_subset/"}


ACCUMULATED_ROUTE_NAMES = {
    "predicted_stage_C_to_D": "predicted_stage_route_accumulated_progress",
    "whole_task_A": "compressed_whole_task_progress",
    "mixed_C_to_D_and_A": "mixed_route_accumulated_progress",
    "unspecified": "unclassified_route_accumulated_progress",
}


def _combined_route(routes):
    if routes == {"predicted_stage_C_to_D", "whole_task_A"}:
        return "mixed_C_to_D_and_A"
    if len(routes) == 1:
        return next(iter(routes))
    return "unspecified"


def _task_slug(task):
    # Match dense_metrics' historical task aliases without importing its renderer.
    return re.sub(r"[^a-zA-Z0-9]+", "-", task).strip("-").lower() or "task"


def accumulated_scope_routes(metrics: dict, *, evaluation: dict | None = None) -> dict:
    """Identify each measured scope's route without changing its RMSE formula.

    Actual C+D records select the staged route even when A is also scored on
    the same window, matching dense_series. Historical metrics without records
    can identify task routes; a mixed group with unknown membership stays
    explicitly unclassified rather than being labelled as C→D.
    """
    scopes = defaultdict(set)
    windows = {}
    for row in (evaluation or {}).get("records", []):
        if "role" not in row:
            continue
        identity = row["cache"], row["task"], row["trajectory_id"], row["poi"]
        roles, group = windows.setdefault(identity, (set(), row.get("group", row["task"])))
        roles.add(row["role"])
    for (_, task, _, _), (roles, group) in windows.items():
        route = ("predicted_stage_C_to_D" if {"C", "D"} <= roles else
                 "whole_task_A" if "A" in roles else "unspecified")
        for scope in ("overall", f"task/{task}", f"group/{group}"):
            scopes[scope].add(route)
    if windows:
        return {scope: _combined_route(routes) for scope, routes in scopes.items()}

    task_routes = {}
    suffixes = {"/predicted_stage_route_accumulated_progress_raw_rmse": "predicted_stage_C_to_D",
                "/compressed_whole_task_progress_raw_rmse": "whole_task_A"}
    for key in metrics:
        for suffix, route in suffixes.items():
            if key.startswith("eval_quality/") and key.endswith(suffix):
                task_routes[key[len("eval_quality/"):-len(suffix)]] = route
    task_names = [key[len("accumulated/task/"):-len("/raw_rmse")] for key in metrics
                  if key.startswith("accumulated/task/") and key.endswith("/raw_rmse")]
    routes = ({task_routes.get(_task_slug(task), "unspecified") for task in task_names}
              if task_routes and task_names else set(task_routes.values()))
    if not routes:
        if "role_D_poi_rmse" in metrics:
            routes.add("predicted_stage_C_to_D")
        if "role_A_poi_rmse" in metrics:
            routes.add("whole_task_A")
    overall = _combined_route(routes)
    result = {"overall": overall}
    for key in metrics:
        if not key.startswith("accumulated/") or not key.endswith("/raw_rmse"):
            continue
        scope = key[len("accumulated/"):-len("/raw_rmse")]
        if scope.startswith("task/"):
            result[scope] = task_routes.get(_task_slug(scope[len("task/"):]),
                                           overall if not task_routes and len(routes) == 1 else "unspecified")
        elif scope.startswith("group/"):
            result[scope] = overall if len(routes) == 1 else "unspecified"
    return result


def compact_evaluation_metrics(metrics: dict, *, evaluation: dict | None = None,
                               schema: str = COMPACT_EVALUATION_SCHEMA) -> dict:
    """Project measured values for charts; full detail belongs in the artifact.

    Never substitute a conditional progress metric for accumulated quality.
    Group/task names, routes and worker counts come from the event, not a recipe.
    Original metric keys remain unchanged inside the full evaluation artifact.
    New charts name both aggregation and scale. The v2 projection is retained
    only for reading/auditing existing runs; new logging always uses v3.
    """
    if schema not in SUPPORTED_COMPACT_EVALUATION_SCHEMAS:
        raise ValueError(f"Unsupported compact evaluation schema: {schema}")
    aggregation = "poi_micro_" if schema == COMPACT_EVALUATION_SCHEMA else ""
    selected = {}

    def include(key, *, chart_key=None):
        if key not in metrics:
            return
        value = metrics[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"Compact evaluation scalar must be finite: {key}")
        selected[chart_key or key] = value

    routes = accumulated_scope_routes(metrics, evaluation=evaluation)

    def pair(prefix):
        keys = [prefix + suffix for suffix in ("raw_rmse", "normalized_rmse")]
        if all(key in metrics for key in keys):
            scope = prefix[len("accumulated/"):-1]
            name = ACCUMULATED_ROUTE_NAMES[routes.get(scope, "unspecified")]
            for key, suffix in zip(keys, ("raw_rmse", "normalized_rmse"), strict=True):
                include(key, chart_key=f"{scope}/{name}_{aggregation}{suffix}")

    pair("accumulated/overall/")
    # Held-out pretraining tasks have unequal numbers of POIs. Keep the
    # evaluator's equal-task average visible alongside the pooled POI RMSE.
    if routes.get("overall") == "whole_task_A":
        include("role_A_task_macro_poi_rmse",
                chart_key="overall/compressed_whole_task_progress_task_macro_"
                          + ("raw_rmse" if schema == COMPACT_EVALUATION_SCHEMA else "rmse"))
    for key in metrics:
        if key.startswith("accumulated/group/") and key.endswith("/raw_rmse"):
            pair(key[:-len("raw_rmse")])
    # Multiple tasks can differ even when an evaluation group spans tasks.
    task_prefixes = [key[:-len("raw_rmse")] for key in metrics
                     if key.startswith("accumulated/task/") and key.endswith("/raw_rmse")]
    task_names = {_task_slug(prefix[len("accumulated/task/"):-1]): prefix[len("accumulated/task/"):-1]
                  for prefix in task_prefixes}
    if len(task_prefixes) > 1:
        for prefix in task_prefixes:
            pair(prefix)
    for key in ("accumulated/overall/support", "accumulated/overall/trajectory_support",
                "role_C_poi_accuracy", "execution_world_size", "elapsed_seconds",
                "evaluation_and_persistence_seconds", "windows_per_second"):
        include(key)
    for key in metrics:
        if key.startswith("eval_quality/") and key.endswith("/oracle_stage_route_accumulated_progress_raw_rmse"):
            alias, metric_name = key[len("eval_quality/"):].rsplit("/", 1)
            metric_name = metric_name.removesuffix("raw_rmse") + aggregation + "raw_rmse"
            include(key, chart_key=f"task/{task_names.get(alias, alias)}/{metric_name}")
        elif re.fullmatch(r"runtime/(?:worker_\d+/)?cuda_memory/(?:peak_reserved_bytes|device_free_after_bytes)", key):
            include(key)
    if evaluation is not None and "accumulated/overall/support" not in selected:
        records = evaluation.get("records", [])
        if records:
            windows = {(row["cache"], row["task"], row["trajectory_id"], row["poi"]) for row in records}
            selected["coverage/windows"] = len(windows)
            selected["coverage/trajectories"] = len({identity[:3] for identity in windows})
    return selected


def evaluation_report(metrics: dict, *, scope: str | None = None) -> dict:
    """Select existing measured values; never infer accumulated error from B/D RMSE.

    Sampled evaluations without accumulated metrics stay explicitly unavailable.
    A-only and mixed inventories retain their actual whole-task deployment route.
    The original metric dictionary and archived artifacts are never modified.
    """
    raw, normalized = "accumulated/overall/raw_rmse", "accumulated/overall/normalized_rmse"
    staged = ("role_D_poi_rmse" in metrics or any(
        key.startswith("eval_quality/") and key.endswith("/predicted_stage_route_stage_accuracy") for key in metrics))
    whole_task = ("role_A_poi_rmse" in metrics or any(
        key.startswith("eval_quality/") and key.endswith("/compressed_whole_task_progress_raw_rmse") for key in metrics))
    if staged and whole_task:
        route, label = "mixed_C_to_D_and_A", "Deployment progress RMSE (C→D accumulated; A whole-task)"
    elif staged:
        route, label = "predicted_stage_C_to_D", "Predicted-stage C→D accumulated progress RMSE"
    elif whole_task:
        route, label = "whole_task_A", "A whole-task progress RMSE"
    else:
        route, label = "unspecified", "Accumulated progress RMSE"
    label += " (POI micro)"
    primary = {"status": "unavailable", "label": label, "scope": scope, "aggregation": "poi_micro",
               "reason": "This evaluation has no measured accumulated raw/normalized RMSE pair."}
    if raw in metrics and normalized in metrics:
        primary = {"status": "assessed", "label": label, "scope": scope, "route": route,
                   "aggregation": "poi_micro",
                   "raw_rmse": metrics[raw], "normalized_rmse": metrics[normalized],
                   "metric_keys": {"raw_rmse": raw, "normalized_rmse": normalized},
                   "normalization": "Each POI error is divided by its trajectory's stage count; A uses scale 1."}
    suffix = "/oracle_stage_route_accumulated_progress_raw_rmse"
    oracle = {key[len("eval_quality/"):-len(suffix)]: {"raw_rmse": value, "metric_key": key}
              for key, value in metrics.items() if key.startswith("eval_quality/") and key.endswith(suffix)}
    return {"primary": primary,
            "oracle_accumulated_comparison": {"label": "Oracle accumulated progress RMSE (true stage + B; POI micro)",
                                               "aggregation": "poi_micro",
                                               "per_task": oracle},
            "conditional_diagnostics": diagnostic_metric_aliases(metrics),
            "legacy_metric_semantics": {
                "role_B_poi_rmse": "Oracle true-stage focus-relative progress diagnostic; not predicted-route accumulated RMSE.",
                "role_D_poi_rmse": "Predicted-stage focus-relative progress diagnostic; target is 1/within-stage progress/0 for earlier/equal/later focus."}}
