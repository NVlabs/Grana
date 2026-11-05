# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Separate full-range rendering samples from a fixed held-out evaluation plan."""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from hashlib import sha256

from .evaluation import EvaluationPlan, EvaluationWindow, validate_evaluation_stage_routing
from .losses import ROLES


def build_render_plan(base_plan: EvaluationPlan, *, max_trajectories_per_task: int | None = 4,
                      points_per_trajectory: int = 24) -> EvaluationPlan:
    """Sample only trajectories already selected by ``base_plan``.

    Real source-frame POIs cover both endpoints, with evenly spaced interior
    samples and no duplicates on short trajectories. Selection uses metadata
    only: it neither inspects RGB/labels nor enumerates the dataset inventory.
    The base plan's preprocessing and stride support remain in force. The
    resulting metrics describe a rendering cohort, not the headline evaluation.
    """
    validate_evaluation_stage_routing(base_plan)
    for name, value, minimum in (("max_trajectories_per_task", max_trajectories_per_task, 1),
                                 ("points_per_trajectory", points_per_trajectory, 2)):
        if name == "max_trajectories_per_task" and value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"render {name} must be an integer >= {minimum}")
    if not base_plan.windows:
        raise ValueError("A render plan requires a nonempty held-out base plan")

    # Use windows rather than dataset.records: the rendering cohort cannot
    # expand beyond the exact trajectories admitted by the fixed evaluation.
    grouped = defaultdict(dict)
    source_groups = {}
    for window in base_plan.windows:
        row = window.trajectory
        key = (str(row.cache_root), row.trajectory_id)
        if key in source_groups and source_groups[key] != window.group:
            raise ValueError("A trajectory must have one fixed evaluation group")
        source_groups[key] = window.group
        if key not in grouped[row.task]:
            grouped[row.task][key] = (row, set())
        grouped[row.task][key][1].update(window.roles)

    settings = deepcopy(base_plan.settings)
    seed = settings["seed"]
    pairs = tuple(tuple(pair) for pair in settings["stride_pairs"])
    windows, tasks = [], []
    for task, members in sorted(grouped.items()):
        candidates = sorted(members.values(), key=lambda item: (
            sha256(f"{seed}:{task}:{item[0].cache_root}:{item[0].trajectory_id}".encode()).digest(),
            str(item[0].cache_root), item[0].trajectory_id))
        task_roles = set().union(*(eligible for _, eligible in candidates))
        # At most sixteen role subsets exist. Reserve a smallest deterministic
        # covering cohort before filling the cap, so rare eligible roles survive.
        covers = {frozenset(): ()}
        for index, (_, eligible) in enumerate(candidates):
            for covered, chosen in tuple(covers.items()):
                combined, proposed = covered | eligible, chosen + (index,)
                previous = covers.get(combined)
                if previous is None or (len(proposed), proposed) < (len(previous), previous):
                    covers[combined] = proposed
        selected_indices = set(covers[frozenset(task_roles)])
        cap = len(candidates) if max_trajectories_per_task is None else max_trajectories_per_task
        if len(selected_indices) > cap:
            raise ValueError(f"Render trajectory cap omits eligible roles for {task!r}; raise max_trajectories_per_task")
        for index in range(len(candidates)):
            if len(selected_indices) >= cap:
                break
            selected_indices.add(index)
        selected = [candidates[index] for index in sorted(selected_indices)]
        for row, eligible in selected:
            count = min(points_per_trajectory, row.num_frames)
            pois = [0] if count == 1 else [index * (row.num_frames - 1) // (count - 1) for index in range(count)]
            roles = tuple(role for role in ROLES if role in eligible)
            for poi in pois:
                key = f"{seed}:{task}:{row.cache_root}:{row.trajectory_id}:{poi}"
                pair = pairs[int.from_bytes(sha256(key.encode()).digest()[:8], "big") % len(pairs)]
                windows.append(EvaluationWindow(row, poi, pair, roles, source_groups[(str(row.cache_root), row.trajectory_id)]))
        tasks.append({"task": task, "available_trajectories": len(candidates),
                      "selected_trajectories": len(selected), "roles": sorted(task_roles),
                      "cache_paths": sorted({str(row.cache_root) for row, _ in candidates}),
                      "selected_ids": [row.trajectory_id for row, _ in selected]})
        source_report = next((item for item in base_plan.tasks if item["task"] == task), {})
        if "stage_routing" in source_report:
            tasks[-1]["stage_routing"] = deepcopy(source_report["stage_routing"])
            tasks[-1]["stage_routing"]["role_window_counts"] = {
                role: sum(role in window.roles for window in windows if window.trajectory.task == task)
                for role in ROLES}

    settings.update(max_trajectories_per_task=max_trajectories_per_task,
                    windows_per_trajectory=points_per_trajectory,
                    purpose="sampled_rendering", metric_scope="sampled_heldout_render_cohort",
                    source_plan_sha256=base_plan.manifest()["plan_sha256"],
                    selection="sha256-ranked held-out base-plan trajectories with role coverage; evenly spaced real POIs including both endpoints")
    settings["view_sets"] = [list(names) for names in sorted({window.trajectory.view_names for window in windows})]
    return EvaluationPlan(base_plan.dataset, tuple(windows), tuple(tasks), settings)
