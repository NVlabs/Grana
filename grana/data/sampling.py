# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Whole-trajectory sampling and eight-slot A/B/C/D supervision."""
from __future__ import annotations

from dataclasses import dataclass, replace
from copy import deepcopy
from hashlib import sha256
import json
import math
from numbers import Integral, Real
import random

from PIL import Image

from .cache import CachedTrajectories, Trajectory, unit_values
from .frame_sampling import (DEFAULT_TRAIN_STRIDE_PAIRS, FRAME_SELECTION_PROFILES, QUERY_ORDER_PROFILES, FRAME_SELECTION_KEY,
                             QUERY_ORDER_KEY, IDENTITY_ORDER, FrameSamplingDecision, choose_profile,
                             normalize_train_frame_sampling, regular_frame_indices, sample_frame_indices)
from grana.modeling.inputs import serialize_messages
from grana.modeling.prompts import (PROGRESS_MARKER, STAGE_MARKER, progress_prompt, stage_progress_prompt,
                                    stage_prompt, with_views)

# Historical explicit stride support for replay/fixtures; never an implicit default.
CTSP9 = ((10, 1), (20, 2), (20, 4), (30, 3), (30, 4), (40, 4), (50, 4), (50, 5), (60, 6))
POI_SLOT = 5


def frame_indices(poi: int, num_frames: int, context_stride: int, tail_stride: int) -> tuple[int, ...]:
    """Fixed evaluation geometry; also the default training selector's base order."""
    return regular_frame_indices(poi, num_frames, context_stride, tail_stride)


def stage_relative_targets(stage_ids: tuple[int, ...], within: tuple[float, ...], focus: int, num_stages: int) -> tuple[float, ...]:
    """For each observation: requested focus earlier/same/later than GT -> 1/WSP/0."""
    if isinstance(num_stages, bool) or not isinstance(num_stages, Integral) or num_stages <= 0:
        raise ValueError("num_stages must be positive")
    if isinstance(focus, bool) or not isinstance(focus, Integral) or not 0 <= focus < num_stages:
        raise ValueError("focus must be a task-local integer stage")
    if not stage_ids or any(isinstance(x, bool) or not isinstance(x, Integral) or not 0 <= x < num_stages for x in stage_ids):
        raise ValueError("stage_ids must be task-local integer stages")
    values = unit_values(within, "within_stage_progress", len(stage_ids))
    return tuple(1.0 if focus < gt else value if focus == gt else 0.0 for gt, value in zip(stage_ids, values, strict=True))


def _image_digest(images: tuple[Image.Image, ...]) -> str:
    digest = sha256()
    for image in images:
        digest.update(f"{image.mode}:{image.size};".encode())
        digest.update(image.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class RoleExample:
    role: str
    images: tuple[Image.Image, ...]
    frame_indices: tuple[int, ...]
    targets: tuple[float, ...] | tuple[int, ...]
    num_stages: int
    task: str
    trajectory_id: str
    prompt: str
    view_names: tuple[str, ...]
    stage_prompts: tuple[str, ...]
    stage_ids: tuple[int, ...]
    within_stage_progress: tuple[float, ...]
    stride_pair: tuple[int, int]
    focus_stage: int | None = None
    observation_digest: str | None = None
    poi_slot: int = POI_SLOT
    frame_weight_order: tuple[int, ...] = IDENTITY_ORDER
    frame_selection_profile: str = "regular_ctsp"
    query_order_profile: str = "forward"

    def __post_init__(self) -> None:
        if isinstance(self.poi_slot, bool) or not isinstance(self.poi_slot, Integral) or not 0 <= self.poi_slot < 8:
            raise ValueError("poi_slot must be an integer in 0..7")
        if (not isinstance(self.frame_weight_order, tuple) or len(self.frame_weight_order) != 8
                or any(isinstance(index, bool) or not isinstance(index, Integral) for index in self.frame_weight_order)
                or set(self.frame_weight_order) != set(IDENTITY_ORDER)):
            raise ValueError("frame_weight_order must be an eight-slot integer permutation")
        if self.frame_selection_profile not in FRAME_SELECTION_PROFILES:
            raise ValueError("Unknown frame_selection_profile")
        if self.query_order_profile not in QUERY_ORDER_PROFILES:
            raise ValueError("Unknown query_order_profile")

    @property
    def poi(self) -> int:
        return self.frame_indices[self.poi_slot]

    def messages(self) -> list[dict]:
        """Labels stay backend-only; each marker follows one full view group."""
        if len(self.frame_indices) != 8 or len(self.targets) != 8 or len(self.images) != 8 * len(self.view_names):
            raise ValueError("Each example must have eight fully supervised view groups")
        return serialize_messages(self.prompt, self.role, self.images, self.view_names)


def _replay_inventory_digest(dataset) -> str:
    """Bind replay to loaded metadata and the actual ordered sampling law.

    This runs only when saving/restoring a sampler snapshot. It never opens
    frame files, decodes pixels, changes an RNG, or enters frame-sampling seeds.
    Per-frame labels are already resident metadata; pixel content integrity is
    outside this digest and remains the cache's responsibility.
    """
    digest = sha256(b"grana-sampler-replay-inventory-v1\n")
    record_digests = {}

    def encoded(value):
        return json.dumps(value, sort_keys=True, ensure_ascii=False,
                          separators=(",", ":"), allow_nan=False).encode("utf-8")

    def record_digest(record):
        # Object identity is only a local memo key, never serialized or seeded.
        key = id(record)
        if key not in record_digests:
            payload = {name: getattr(record, name) for name in (
                "trajectory_id", "task", "num_frames", "frame_shape", "view_names", "roles",
                "stage_prompts", "compressed_task_prompt", "progress", "stage_ids", "within_stage_progress")}
            payload.update(cache_root=str(record.cache_root), frame_paths=[str(path) for path in record.frame_paths])
            record_digests[key] = sha256(encoded(payload)).hexdigest()
        return record_digests[key]

    for record in dataset.records:
        digest.update(encoded(["record", record_digest(record)]) + b"\n")
    groups = getattr(dataset, "by_role", None)
    if groups is None:
        digest.update(b"metadata-only-inventory\n")
    else:
        for role in ("A", "B", "C"):
            for task, records in groups[role].items():
                weight = dataset.task_weights[role][task]
                if isinstance(weight, bool) or not isinstance(weight, Real) or not math.isfinite(weight) or weight <= 0:
                    raise ValueError("Sampler replay task weights must be finite and positive")
                digest.update(encoded(["task", role, task, float(weight),
                                       [record_digest(record) for record in records]]) + b"\n")
    return digest.hexdigest()


def materialize_example(trajectory: Trajectory, role: str, poi: int, pair: tuple[int, int], *,
                        frame_decision: FrameSamplingDecision | None = None,
                        execution_mode: str = "eval") -> RoleExample:
    """Build aligned observations/targets; absent a decision, use frozen evaluation geometry."""
    if execution_mode not in ("train", "eval", "inference"):
        raise ValueError("execution_mode must be train, eval, or inference")
    if role not in ("A", "B", "C") or role not in trajectory.roles:
        raise ValueError(f"Trajectory {trajectory.trajectory_id!r} is not eligible for role {role}")
    indices = frame_indices(poi, trajectory.num_frames, *pair) if frame_decision is None else frame_decision.frame_indices
    poi_slot = POI_SLOT if frame_decision is None else frame_decision.poi_slot
    if indices[poi_slot] != poi:
        raise ValueError("Frame decision must retain the designated physical POI")
    stages = tuple(trajectory.stage_ids[i] for i in indices) if trajectory.stage_ids else ()
    within = tuple(trajectory.within_stage_progress[i] for i in indices) if trajectory.within_stage_progress else ()
    focus = None
    if role == "A":
        targets = tuple(trajectory.progress[i] for i in indices)
        prompt = progress_prompt(trajectory.compressed_task_prompt)
    elif role == "B":
        focus = trajectory.stage_ids[poi]
        targets = stage_relative_targets(stages, within, focus, len(trajectory.stage_prompts))
        prompt = stage_progress_prompt(trajectory.task, trajectory.stage_prompts, focus)
    else:
        targets = stages
        prompt = stage_prompt(trajectory.task, trajectory.stage_prompts)
    images = trajectory.images(indices)
    metadata = {} if frame_decision is None else {
        "poi_slot": frame_decision.poi_slot, "frame_weight_order": frame_decision.frame_weight_order,
        "frame_selection_profile": frame_decision.frame_selection_profile,
        "query_order_profile": frame_decision.query_order_profile,
    }
    return RoleExample(role, images, indices, targets, len(trajectory.stage_prompts), trajectory.task,
                       trajectory.trajectory_id, with_views(prompt, trajectory.view_names), trajectory.view_names,
                       trajectory.stage_prompts, stages, within, pair, focus,
                       _image_digest(images) if role == "C" else None, **metadata)


class RoleSampler:
    def __init__(self, dataset: CachedTrajectories, data_config: dict | None = None, seed: int | None = None, *,
                 execution_mode: str = "train", worker_rank: int = 0):
        self.dataset = dataset
        cfg = data_config or {}
        seed = cfg.get("seed", 42) if seed is None else seed
        self.seed = seed
        if execution_mode not in ("train", "eval", "inference"):
            raise ValueError("execution_mode must be train, eval, or inference")
        if isinstance(worker_rank, bool) or not isinstance(worker_rank, Integral) or worker_rank < 0:
            raise ValueError("worker_rank must be a nonnegative integer")
        self.execution_mode = execution_mode
        self.worker_rank = int(worker_rank)
        if execution_mode == "train":
            c_tasks = dataset.by_role["C"]
        else:
            # Frozen eval/render plans may expose only their trajectory records.
            # They materialize fixed windows directly; no training inventory or
            # task-weight sampler is required merely to construct paired D.
            c_tasks = {}
            for record in dataset.records:
                if "C" in record.roles:
                    c_tasks.setdefault(record.task, []).append(record)
            c_tasks = {task: tuple(rows) for task, rows in sorted(c_tasks.items())}
        self.paired_c_tasks = dict(c_tasks)
        self.train_frame_sampling = normalize_train_frame_sampling(cfg.get("train_frame_sampling", {}))
        pairs = cfg.get("stride_pairs", DEFAULT_TRAIN_STRIDE_PAIRS)
        self.stride_pairs = tuple(tuple(pair) for pair in pairs)
        if not self.stride_pairs or len(set(self.stride_pairs)) != len(self.stride_pairs):
            raise ValueError("stride_pairs must be nonempty and unique")
        for pair in self.stride_pairs:
            if len(pair) != 2 or any(isinstance(x, bool) or not isinstance(x, Integral) or x <= 0 for x in pair):
                raise ValueError("Each stride pair needs two positive integers")
        # Independent role and stride streams keep A/B/C laws separate and keep
        # source/POI draws unchanged when the stride support changes.
        self._streams = ("A", "B", "C", "C_independent")
        self._rng = {role: random.Random(f"grana:{seed}:{role}:selection") for role in self._streams}
        self._stride_rng = {role: random.Random(f"grana:{seed}:{role}:stride") for role in self._streams}
        self._draw_ordinals = {role: 0 for role in self._streams}

    def validate_role_budget(self, roles: dict) -> None:
        """Fail before model initialization when a task route cannot fit the budget."""
        if roles.get("D", 0):
            if roles.get("C", 0) < roles["D"]:
                raise ValueError("Role D requires at least as many C rows for one-to-one pairing")
            if not self.paired_c_tasks:
                raise ValueError("Role D has no eligible task for paired C")

    def state_dict(self) -> dict:
        """Snapshot sampler randomness for qualification/replay; does not save training state."""
        return {"schema": "grana-role-sampler-state-v3", "seed": self.seed,
                "replay_inventory_sha256": _replay_inventory_digest(self.dataset),
                "stride_pairs": self.stride_pairs, "train_frame_sampling": deepcopy(self.train_frame_sampling),
                "worker_rank": self.worker_rank,
                "selection_rng": {role: rng.getstate() for role, rng in self._rng.items()},
                "stride_rng": {role: rng.getstate() for role, rng in self._stride_rng.items()},
                "draw_ordinals": dict(self._draw_ordinals)}

    def load_state_dict(self, state: dict) -> None:
        """Restore all streams atomically, rejecting a different sampling configuration."""
        if (state.get("schema") != "grana-role-sampler-state-v3" or state.get("seed") != self.seed
                or state.get("replay_inventory_sha256") != _replay_inventory_digest(self.dataset)
                or state.get("stride_pairs") != self.stride_pairs
                or state.get("train_frame_sampling") != self.train_frame_sampling
                or state.get("worker_rank") != self.worker_rank):
            raise ValueError("Sampler state configuration does not match")
        streams = {}
        for key in ("selection_rng", "stride_rng"):
            values = state.get(key)
            if not isinstance(values, dict) or set(values) != set(self._streams):
                raise ValueError(f"Sampler state requires all role streams: {key}")
            streams[key] = {}
            for role, value in values.items():
                restored = random.Random()
                restored.setstate(value)
                streams[key][role] = restored
        ordinals = state.get("draw_ordinals")
        if (not isinstance(ordinals, dict) or set(ordinals) != set(self._streams)
                or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in ordinals.values())):
            raise ValueError("Sampler state requires nonnegative role draw ordinals")
        self._rng, self._stride_rng = streams["selection_rng"], streams["stride_rng"]
        self._draw_ordinals = dict(ordinals)

    def sample(self, role: str) -> RoleExample:
        request = role
        if request not in ("A", "B", "C", "C_paired", "C_independent"):
            raise ValueError("Only A/B/C are independently sampled; D derives from paired C")
        role = "C" if request in ("C_paired", "C_independent") else request
        # Pairing reuses the historical C stream. The independent mixed-budget
        # lane owns a separate stream; plain C preserves the D=0 legacy path.
        stream = "C" if request == "C_paired" else request
        tasks = self.paired_c_tasks if request == "C_paired" else self.dataset.by_role[role]
        if not tasks:
            raise ValueError(f"No eligible cached trajectories for request {request}")
        rng = self._rng[stream]
        task_names = tuple(tasks)
        weights = [self.dataset.task_weights[role][name] for name in task_names]
        task = rng.choice(task_names) if len(set(weights)) == 1 else rng.choices(task_names, weights=weights, k=1)[0]
        trajectory = rng.choice(tasks[task])
        poi = rng.randrange(trajectory.num_frames)
        pair = self._stride_rng[stream].choice(self.stride_pairs)
        ordinal = self._draw_ordinals[stream]
        # Per-request namespaces prevent variable selector work from affecting
        # subsequent draws. No configuration fingerprint enters these seeds.
        def independent_rng(namespace: str) -> random.Random:
            return random.Random(f"grana:{self.seed}:{stream}:frames-v1:{ordinal}:{namespace}")
        family = choose_profile(self.train_frame_sampling[FRAME_SELECTION_KEY], independent_rng("family"))
        order = choose_profile(self.train_frame_sampling[QUERY_ORDER_KEY], independent_rng("order"))
        decision = sample_frame_indices(poi, trajectory.num_frames, *pair, frame_selection_profile=family,
                                        query_order_profile=order, rng=independent_rng("frame-choice"))
        self._draw_ordinals[stream] += 1
        return materialize_example(trajectory, role, poi, pair, frame_decision=decision,
                                   execution_mode=self.execution_mode)

    def make_d(self, source_c: RoleExample, focus_stage: int) -> RoleExample:
        """Use the caller's detached hard argmax at C's designated POI; never redraw."""
        if source_c.role != "C" or source_c.focus_stage is not None:
            raise ValueError("D must derive from an unfocused C example")
        if source_c.observation_digest != _image_digest(source_c.images):
            raise ValueError("Paired C observations changed before D construction")
        targets = stage_relative_targets(source_c.stage_ids, source_c.within_stage_progress, focus_stage, source_c.num_stages)
        prompt = with_views(stage_progress_prompt(source_c.task, source_c.stage_prompts, int(focus_stage)), source_c.view_names)
        return replace(source_c, role="D", targets=targets, prompt=prompt, focus_stage=int(focus_stage))
