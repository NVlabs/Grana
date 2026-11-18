# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The retained A/B/C/D prompt contract, without labels in model inputs."""
from __future__ import annotations

from string import ascii_uppercase

PROGRESS_MARKER = "<|prog_token|>"
STAGE_MARKER = "<|stage_token|>"


def text_field(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty text")
    if any(marker in value for marker in (PROGRESS_MARKER, STAGE_MARKER)):
        raise ValueError(f"{name} cannot contain reserved answer markers")
    return value.strip()


def progress_prompt(compressed_task_prompt: str) -> str:
    task = text_field(compressed_task_prompt, "compressed_task_prompt")
    return (
        f"The task for the robot is '{task}'. "
        "The following trajectory shows multiple camera views at each "
        "timestep. Predict task progress at each frame, how far along the "
        "robot is towards completing the task, a float between 0 and 1, "
        "where 0 is the starting state and 1 is when the task is "
        "completed. If the robot is not performing the same task, "
        "predict 0 progress."
    )


def stage_progress_prompt(task: str, stages: tuple[str, ...], focus: int) -> str:
    task = text_field(task, "task")
    if isinstance(focus, bool) or not isinstance(focus, int) or not 0 <= focus < len(stages):
        raise ValueError("focus must be an integer in the task's stage range")
    stages = tuple(text_field(value, "stage prompt") for value in stages)
    lines = [f"The robot is solving a multi-stage task ({task}). The ordered should-do stages are:"]
    lines.extend(f"Stage {index}: {prompt}" for index, prompt in enumerate(stages))
    lines.extend((
        f"Focus stage: Stage {focus}: {stages[focus]}",
        "For each frame independently, predict progress relative to this focus stage as a float between 0 and 1:",
        "- Predict 1, if this focus stage has already been completed and the robot should move on the later/higher stage.",
        "- Predict the within-stage progress (between 0 to 1), if the robot is currently doing this focus stage.",
        "- Predict 0, if the robot should still focus on earlier/lower prerequisites stages, not even able to start consider the focus stage at all.",
    ))
    return "\n".join(lines)


def stage_prompt(task: str, stages: tuple[str, ...]) -> str:
    """Canonical native A..Z choices; no focus, answer, or DONE option."""
    task = text_field(task, "task")
    stages = tuple(text_field(value, "stage prompt") for value in stages)
    if not 1 <= len(stages) <= 26:
        raise ValueError("native stage prediction requires 1..26 ordered stages")
    task_name = f" ({task})" if task != " | ".join(stages) else ""
    lines = [f"The robot is solving a multi-stage task{task_name} with {len(stages)} stages in chronological task order:"]
    letters = ascii_uppercase[:len(stages)]
    lines.extend(f"{letter}: Stage {index}: {prompt}" for index, (letter, prompt) in enumerate(zip(letters, stages, strict=True)))
    lines.extend((
        "For each frame, use the available observations to identify the first unfinished stage: "
        "the earliest stage in the task's chronological stage list that has not yet been completed at that frame. "
        "A failed or incomplete stage remains unfinished even if the robot starts a later stage. "
        "If all stages are complete, select the final stage in the task's stage list.",
        "Follow the task's stage order, not the answer-letter order. "
        "After each frame group, the stage marker requests exactly one answer letter "
        f"from {', '.join(letters)}. Output the letter whose description matches "
        "the selected stage. Output only the letter, without explanation.",
    ))
    return "\n".join(lines)


def with_views(prompt: str, views: tuple[str, ...]) -> str:
    if len(views) == 1:
        prompt = prompt.replace(
            "The following trajectory shows multiple camera views at each timestep.",
            "The following trajectory shows one camera view at each timestep.",
        )
        return prompt + (
            f"\nThe following trajectory is shown from one camera view ({views[0]}) "
            "at each timestep, and there are total of 8 timesteps."
        )
    return prompt + (
        f"\nThe following trajectory is shown from {len(views)} camera views "
        f"({', '.join(views)}) at each timestep, and there are total of 8 timesteps."
    )
