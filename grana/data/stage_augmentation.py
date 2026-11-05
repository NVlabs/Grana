# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Training-only, contiguous native answer choices from immutable stage text.

This module never reads pixels or labels. The caller supplies the already
filtered training records; held-out records must never be added to this pool.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from numbers import Integral
import random
import unicodedata

from grana.modeling.prompts import text_field


ALGORITHM_VERSION = "contiguous-shift-v1"
TEXT_NORMALIZATION_VERSION = "nfc-whitespace-v1"


def normalized_stage_text(value: str) -> str:
    """Versioned equality rule; keep validated original text for presentation."""
    return " ".join(unicodedata.normalize("NFC", text_field(value, "stage prompt")).split())


@dataclass(frozen=True)
class StageChoiceMapping:
    candidate_prompts: tuple[str, ...]
    stage_to_choice: tuple[int, ...]
    choice_to_stage: tuple[int, ...]
    offset: int
    pool_digest: str
    algorithm_version: str
    draw_id: str
    # The real task's entries have no donor source; every foreign entry retains
    # all other training tasks contributing the selected unique description.
    candidate_source_tasks: tuple[tuple[str, ...], ...] = ()

    def __post_init__(self) -> None:
        if (not isinstance(self.candidate_prompts, tuple)
                or not 1 <= len(self.candidate_prompts) <= 26):
            raise ValueError("candidate_prompts must contain 1..26 immutable choices")
        count = len(self.candidate_prompts)
        normalized = tuple(normalized_stage_text(value) for value in self.candidate_prompts)
        if len(set(normalized)) != count:
            raise ValueError("Stage choices must have unique normalized descriptions")
        if (not isinstance(self.stage_to_choice, tuple) or not self.stage_to_choice
                or any(isinstance(index, bool) or not isinstance(index, Integral)
                       for index in self.stage_to_choice)):
            raise ValueError("stage_to_choice must be a nonempty integer tuple")
        if (isinstance(self.offset, bool) or not isinstance(self.offset, Integral)
                or not 0 <= self.offset <= count - len(self.stage_to_choice)):
            raise ValueError("offset must fit the complete real-stage block")
        if self.stage_to_choice != tuple(range(self.offset, self.offset + len(self.stage_to_choice))):
            raise ValueError("stage_to_choice must be contiguous and preserve canonical order")
        inverse = tuple(index - self.offset if index in self.stage_to_choice else -1 for index in range(count))
        if (not isinstance(self.choice_to_stage, tuple) or self.choice_to_stage != inverse
                or any(isinstance(index, bool) or not isinstance(index, Integral)
                       for index in self.choice_to_stage)):
            raise ValueError("choice_to_stage must invert the real-stage block; foreign choices use -1")
        if self.algorithm_version != ALGORITHM_VERSION:
            raise ValueError("Unknown stage-choice algorithm version")
        if not isinstance(self.pool_digest, str) or not self.pool_digest:
            raise ValueError("pool_digest must identify the training text inventory")
        if not isinstance(self.draw_id, str) or not self.draw_id:
            raise ValueError("draw_id must identify the sampler draw")
        if not isinstance(self.candidate_source_tasks, tuple):
            raise ValueError("candidate_source_tasks must be immutable")
        if self.candidate_source_tasks:
            if len(self.candidate_source_tasks) != count:
                raise ValueError("candidate_source_tasks must align with candidate_prompts")
            for index, tasks in enumerate(self.candidate_source_tasks):
                if (not isinstance(tasks, tuple)
                        or any(not isinstance(task, str) or not task for task in tasks)
                        or tasks != tuple(sorted(set(tasks)))
                        or bool(tasks) != (self.choice_to_stage[index] == -1)):
                    raise ValueError("candidate_source_tasks must mark only foreign choices with sorted task names")


@dataclass(frozen=True)
class StageDistractor:
    prompt: str
    source_tasks: tuple[str, ...]


@dataclass(frozen=True)
class StageDistractorPool:
    # Every normalized description appears once. Distinct original variants
    # remain available so display text always comes from an eligible donor.
    entries: tuple[tuple[str, tuple[tuple[str, str], ...]], ...]
    digest: str

    @classmethod
    def from_records(cls, records) -> StageDistractorPool:
        index: dict[str, set[tuple[str, str]]] = {}
        for record in records:
            for description in record.stage_prompts:
                text = text_field(description, "stage prompt")
                normalized = normalized_stage_text(text)
                index.setdefault(normalized, set()).add((record.task, text))
        entries = tuple((key, tuple(sorted(values))) for key, values in sorted(index.items()))
        payload = {"normalization": TEXT_NORMALIZATION_VERSION, "entries": entries}
        digest = sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True,
                                   separators=(",", ":")).encode("utf-8")).hexdigest()
        return cls(entries, digest)

    def eligible(self, task: str, stages: tuple[str, ...], candidate_count: int) -> tuple[StageDistractor, ...]:
        """Preflight one task vocabulary and return its unique, ordered donors."""
        if (isinstance(candidate_count, bool) or not isinstance(candidate_count, Integral)
                or not 1 <= candidate_count <= 26):
            raise ValueError("candidate_count must be an integer in 1..26")
        count = len(stages)
        if not 1 <= count <= candidate_count:
            raise ValueError(f"Task {task!r} has K={count} stages but candidate_count={candidate_count}; require 1 <= K <= N")
        true_text = tuple(normalized_stage_text(value) for value in stages)
        if len(set(true_text)) != count:
            raise ValueError(f"Task {task!r} has duplicate normalized real stage descriptions; answer mapping is ambiguous")
        true_text_set = set(true_text)
        eligible = []
        for normalized, sources in self.entries:
            if normalized in true_text_set:
                continue
            other_sources = tuple((name, text) for name, text in sources if name != task)
            if other_sources:
                eligible.append(StageDistractor(min(text for _, text in other_sources),
                                                tuple(sorted({name for name, _ in other_sources}))))
        needed = candidate_count - count
        if len(eligible) < needed:
            raise ValueError(f"Task {task!r} needs {needed} unique other-training-task distractors, "
                             f"but only {len(eligible)} are available")
        return tuple(eligible)


def make_stage_choice_mapping(task: str, stages: tuple[str, ...], candidate_count: int, *,
                              pool_digest: str, donors: tuple[StageDistractor, ...],
                              seed: int, worker_rank: int, lane: str, draw_ordinal: int) -> StageChoiceMapping:
    """Draw one mapping per observation, independently of frames and GT labels."""
    if (isinstance(candidate_count, bool) or not isinstance(candidate_count, Integral)
            or not 1 <= len(stages) <= candidate_count <= 26):
        raise ValueError("A contiguous stage block requires 1 <= K <= N <= 26")
    if (isinstance(draw_ordinal, bool) or not isinstance(draw_ordinal, Integral) or draw_ordinal < 0
            or isinstance(worker_rank, bool) or not isinstance(worker_rank, Integral) or worker_rank < 0):
        raise ValueError("Stage-choice rank and draw ordinal must be nonnegative integers")
    if lane not in ("C", "C_independent"):
        raise ValueError("Only independent training C draws may use stage augmentation")
    needed = candidate_count - len(stages)
    if len(donors) < needed:
        raise ValueError(f"Task {task!r} has too few unique training distractors")
    draw_id = f"rank={worker_rank}:lane={lane}:draw={draw_ordinal}"
    # Task, GT, inventory digest and candidate count deliberately do not enter
    # seed namespaces: donor changes cannot perturb the offset random stream.
    def rng(namespace: str) -> random.Random:
        return random.Random(f"grana:{seed}:rank={worker_rank}:{lane}:stage-choice-v1:{draw_ordinal}:{namespace}")

    offset = rng("offset").randrange(candidate_count - len(stages) + 1)
    selected = rng("distractors").sample(donors, needed)
    rng("placement").shuffle(selected)
    candidate_prompts, source_tasks, choice_to_stage = [], [], []
    foreign = iter(selected)
    for choice in range(candidate_count):
        stage = choice - offset
        if 0 <= stage < len(stages):
            candidate_prompts.append(stages[stage])
            source_tasks.append(())
            choice_to_stage.append(stage)
        else:
            donor = next(foreign)
            if task in donor.source_tasks:
                raise ValueError("Distractor source includes the current task")
            candidate_prompts.append(donor.prompt)
            source_tasks.append(donor.source_tasks)
            choice_to_stage.append(-1)
    return StageChoiceMapping(tuple(candidate_prompts), tuple(range(offset, offset + len(stages))),
                              tuple(choice_to_stage), offset, pool_digest, ALGORITHM_VERSION, draw_id,
                              tuple(source_tasks))
