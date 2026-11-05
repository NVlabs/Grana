# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read existing HF metadata and RGB NPY caches; never construct or rewrite data."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
import hashlib
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from grana.modeling.prompts import text_field


def _view_name(value: Any, name: str) -> str:
    clean = text_field(value, name)
    if clean != value or len(clean) > 128 or "<|" in clean or "|>" in clean:
        raise ValueError(f"{name} must have no surrounding whitespace or token delimiters and at most 128 characters")
    return clean


def _sequence(value: Any, name: str, length: int | None = None) -> tuple:
    if not isinstance(value, (list, tuple, np.ndarray)):
        raise ValueError(f"{name} must be a sequence")
    values = tuple(value)
    if length is not None and len(values) != length:
        raise ValueError(f"{name} must have {length} frame-aligned entries, got {len(values)}")
    return values


def unit_values(value: Any, name: str, length: int) -> tuple[float, ...]:
    values = _sequence(value, name, length)
    if any(isinstance(x, (bool, np.bool_)) or not isinstance(x, Real)
           or not np.isfinite(x) or not 0 <= x <= 1 for x in values):
        raise ValueError(f"{name} must contain finite real numbers in [0, 1]")
    return tuple(float(x) for x in values)


def resolve_frame_path(raw_path: Any, cache_root: Path) -> Path:
    """Prefer an existing explicit absolute path, then this cache's frames/basename."""
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError("Each requested view needs an explicit NPY frame path")
    path = Path(raw_path).expanduser()
    if path.is_absolute() and path.is_file():
        return path
    local = cache_root / "frames" / path.name
    if local.is_file():
        return local
    raise FileNotFoundError(f"Frame file missing: {path}; same-cache fallback: {local}")


@lru_cache(maxsize=32)
def _mmap(path: str) -> np.ndarray:
    return np.load(path, mmap_mode="r", allow_pickle=False)


@dataclass(frozen=True)
class Trajectory:
    cache_root: Path
    trajectory_id: str
    task: str
    num_frames: int
    frame_paths: tuple[Path, ...]
    frame_shape: tuple[int, ...]
    view_names: tuple[str, ...]
    roles: tuple[str, ...]
    progress: tuple[float, ...]
    stage_ids: tuple[int, ...]
    within_stage_progress: tuple[float, ...]
    stage_prompts: tuple[str, ...]
    compressed_task_prompt: str | None

    def images(self, indices: tuple[int, ...]) -> tuple[Image.Image, ...]:
        """Full RGB views in timestep-major order, including duplicated slots."""
        arrays = tuple(_mmap(str(path)) for path in self.frame_paths)
        for path, array in zip(self.frame_paths, arrays, strict=True):
            if array.dtype != np.uint8 or tuple(array.shape) != self.frame_shape:
                raise ValueError(f"RGB cache {path} must be uint8 with shape {self.frame_shape}; got {array.dtype} {array.shape}")
        if any(isinstance(i, bool) or not isinstance(i, Integral) or not 0 <= i < self.num_frames for i in indices):
            raise ValueError("Frame indices must refer to real source frames")
        return tuple(Image.fromarray(array[index]) for index in indices for array in arrays)


def _trajectory(row: dict, root: Path, views: tuple[str, ...], roles: tuple[str, ...], prompt: str | None) -> Trajectory:
    count = row.get("num_frames")
    if isinstance(count, bool) or not isinstance(count, Integral) or count <= 0:
        raise ValueError("num_frames must be a positive integer")
    count = int(count)
    shape = _sequence(row.get("frames_shape"), "frames_shape")
    if (len(shape) != 4 or any(isinstance(x, bool) or not isinstance(x, Integral) or x <= 0 for x in shape)
            or shape[0] != count or shape[-1] != 3):
        raise ValueError("frames_shape must be [num_frames, height, width, 3]")
    paths = tuple(resolve_frame_path(row.get(f"frames_{view}", row.get("frames") if view == "top" else None), root) for view in views)
    task = text_field(row.get("task"), "task")
    identity = text_field(row.get("id"), "id")
    progress, stage_ids, within, stages = (), (), (), ()
    compressed = None
    if "A" in roles:
        progress = unit_values(row.get("progress"), "progress", count)
        compressed = text_field(prompt if prompt is not None else row.get("compressed_task_prompt"), "compressed_task_prompt (or manifest prompt)")
    if "B" in roles or "C" in roles:
        stages = tuple(text_field(x, "stage prompt") for x in _sequence(row.get("stage_prompts"), "stage_prompts"))
        if not stages or len(stages) > 26:
            raise ValueError("B/C/D need 1..26 ordered stages")
        raw_ids = _sequence(row.get("stage_ids"), "stage_ids", count)
        if any(isinstance(x, (bool, np.bool_)) or not isinstance(x, Integral) or not 0 <= x < len(stages) for x in raw_ids):
            raise ValueError("stage_ids must be integer slots in the ordered stage_prompts range")
        stage_ids = tuple(int(x) for x in raw_ids)
        within = unit_values(row.get("within_stage_progress"), "within_stage_progress", count)
    return Trajectory(root, identity, task, count, paths, tuple(int(x) for x in shape), views, roles,
                      progress, stage_ids, within, stages, compressed)


def _hash_split_ids(source_ids: list[str], spec: Any) -> set[str]:
    """Select a reproducible cohort without publishing source trajectory IDs."""
    if not isinstance(spec, dict) or set(spec) != {"seed", "eval_count", "subset"}:
        raise ValueError("hash_split must contain exactly seed, eval_count, and subset")
    seed, count, subset = spec["seed"], spec["eval_count"], spec["subset"]
    if not isinstance(seed, str) or not seed.strip():
        raise ValueError("hash_split seed must be a nonempty string")
    if not isinstance(subset, str) or subset not in ("train", "eval"):
        raise ValueError("hash_split subset must be train or eval")
    if isinstance(count, (bool, np.bool_)) or not isinstance(count, Integral) or count <= 0:
        raise ValueError("hash_split eval_count must be a positive integer")
    if any(not isinstance(identity, str) or not identity.strip() or identity != identity.strip()
           for identity in source_ids):
        raise ValueError("hash_split source IDs must be nonempty strings without surrounding whitespace")
    if len(set(source_ids)) != len(source_ids):
        raise ValueError("Duplicate cached trajectory IDs in hash_split source inventory")
    if count >= len(source_ids):
        raise ValueError("hash_split eval_count must be smaller than the source inventory")
    ranked = sorted(source_ids, key=lambda identity: (
        hashlib.sha256(f"{seed}|{identity}".encode("utf-8")).digest(), identity))
    evaluation = set(ranked[:int(count)])
    return evaluation if subset == "eval" else set(source_ids) - evaluation


class CachedTrajectories:
    """Validated role-conditional task inventory over complete source trajectories.

    Each entry is a path or {path, prompt?, roles?, views?, ...}. `views` overrides
    the ordered global camera default for that cache, allowing mixed inventories
    in one dataset and physical batch. `roles` contains independent
    lanes A/B/C; D always inherits C's inventory. An A-only cache needs no stage
    labels. A BC-only cache needs no compressed prompt or global progress.
    """

    def __init__(self, caches: list, views: tuple[str, ...] | list[str] = ("top", "left", "right", "bottom")):
        from datasets import Dataset, load_from_disk

        if not isinstance(caches, (list, tuple)) or not caches:
            raise ValueError("caches must be a nonempty list of paths or manifest entries")
        if not isinstance(views, (list, tuple)) or not views:
            raise ValueError("views must be unique ordered names")
        self.views = tuple(_view_name(x, "view") for x in views)
        if len(set(self.views)) != len(self.views):
            raise ValueError("views must be unique ordered names")
        groups = {role: defaultdict(list) for role in "ABC"}
        task_weights = {role: {} for role in "ABC"}
        records, summaries, seen = [], [], set()
        for entry in caches:
            spec = {"path": str(entry)} if isinstance(entry, (str, Path)) else entry
            if not isinstance(spec, dict) or set(spec) - {"path", "prompt", "roles", "views", "task_weights", "include_ids", "exclude_ids", "hash_split"} or "path" not in spec:
                raise ValueError("Cache entry has missing path or unknown fields")
            if "hash_split" in spec and any(key in spec for key in ("include_ids", "exclude_ids")):
                raise ValueError("Choose hash_split or include_ids/exclude_ids, not both")
            root = Path(spec["path"]).expanduser().resolve()
            # Accept either the cache root or its processed_dataset subdirectory.
            if root.name == "processed_dataset":
                root = root.parent
            selected_views = spec.get("views", self.views)
            if not isinstance(selected_views, (list, tuple)) or not selected_views:
                raise ValueError("Cache views must be nonempty unique ordered names")
            selected_views = tuple(_view_name(x, "cache view") for x in selected_views)
            if len(set(selected_views)) != len(selected_views):
                raise ValueError("Cache views must be nonempty unique ordered names")
            role_values = spec.get("roles", ["A", "B", "C"])
            if (not isinstance(role_values, (list, tuple)) or not role_values
                    or any(role not in ("A", "B", "C") for role in role_values)
                    or len(set(role_values)) != len(role_values)):
                raise ValueError("Cache roles must be a nonempty subset of [A, B, C]; D derives from C")
            roles = tuple(role for role in "ABC" if role in role_values)
            override = text_field(spec["prompt"], "manifest prompt") if "prompt" in spec else None
            data = load_from_disk(str(root / "processed_dataset"))
            if not isinstance(data, Dataset) or not len(data):
                raise ValueError(f"{root} must contain a nonempty HF Dataset, not a DatasetDict")
            declared_weights = spec.get("task_weights", {})
            if not isinstance(declared_weights, dict) or set(declared_weights) - set(roles):
                raise ValueError("task_weights must map only eligible A/B/C roles")
            weights = {role: declared_weights.get(role, 1.0) for role in roles}
            if any(isinstance(w, bool) or not isinstance(w, Real) or not np.isfinite(w) or w <= 0 for w in weights.values()):
                raise ValueError("Eligible task weights must be finite and positive")
            selectors = {}
            for key in ("include_ids", "exclude_ids"):
                values = spec.get(key)
                if values is not None:
                    if not isinstance(values, list) or not values or any(not isinstance(x, str) or not x for x in values) or len(set(values)) != len(values):
                        raise ValueError(f"{key} must be a nonempty list of unique trajectory IDs")
                    selectors[key] = set(values)
            if len(selectors) > 1:
                raise ValueError("Choose include_ids or exclude_ids, not both")
            source_id_list = list(data["id"])
            if "hash_split" in spec:
                selectors["include_ids"] = _hash_split_ids(source_id_list, spec["hash_split"])
            source_ids = set(source_id_list)
            for key, values in selectors.items():
                if values - source_ids:
                    raise ValueError(f"{key} contains absent trajectory IDs: {sorted(values - source_ids)}")
            tasks, selected_rows = set(), 0
            for row in data:
                identity = row.get("id")
                if "include_ids" in selectors and identity not in selectors["include_ids"]:
                    continue
                if identity in selectors.get("exclude_ids", set()):
                    continue
                selected_rows += 1
                try:
                    record = _trajectory(row, root, selected_views, roles, override)
                except (ValueError, FileNotFoundError) as exc:
                    raise type(exc)(f"{root}, trajectory {row.get('id')!r}: {exc}") from exc
                key = (root, record.trajectory_id)
                if key in seen:
                    raise ValueError(f"Duplicate cached trajectory: {key}")
                seen.add(key)
                records.append(record)
                tasks.add(record.task)
                for role in roles:
                    if record.task in task_weights[role] and task_weights[role][record.task] != weights[role]:
                        raise ValueError(f"Conflicting declared weight for role {role}, task {record.task}")
                    task_weights[role][record.task] = float(weights[role])
                    groups[role][record.task].append(record)
            if not selected_rows:
                raise ValueError(f"Trajectory selection is empty: {root}")
            summaries.append({"path": str(root), "rows": selected_rows, "source_rows": len(data),
                              "tasks": sorted(tasks), "roles": list(roles), "views": list(selected_views), "task_weights": weights})
        self.records = tuple(records)
        self.by_role = {role: {task: tuple(rows) for task, rows in sorted(tasks.items())} for role, tasks in groups.items()}
        self.task_weights = task_weights
        self.max_num_stages = max(len(row.stage_prompts) for row in records)
        self._summaries = summaries

    def __len__(self) -> int:
        return len(self.records)

    def summary(self) -> dict:
        return {"trajectories": len(self), "views": list(self.views), "caches": self._summaries,
                "view_sets": [list(names) for names in sorted({row.view_names for row in self.records})],
                "eligible_rows": {role: sum(map(len, groups.values())) for role, groups in self.by_role.items()},
                "max_num_stages": self.max_num_stages, "task_weights": self.task_weights}
