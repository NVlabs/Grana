# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Augmented answer text is complete, ordered and independent of labels."""
import inspect
import re

import pytest

from grana.modeling.inputs import serialize_messages
from grana.modeling.prompts import STAGE_MARKER, stage_choice_prompt, stage_prompt, with_views


STAGES = ("Pick up the wing.", "Align it with the fuselage.", "Insert and secure the wing.")


def candidates(stages=STAGES, count=10, offset=3):
    choices = [f"Unrelated operation {index}." for index in range(count)]
    choices[offset:offset + len(stages)] = stages
    return tuple(choices)


@pytest.mark.parametrize("count,stage_count", [(1, 1), (10, 1), (10, 3), (10, 8), (10, 10), (26, 3)])
def test_augmented_prompt_lists_every_answer_and_preserves_true_stage_order(count, stage_count):
    stages = tuple(f"True operation {index}." for index in range(stage_count))
    for offset in range(count - stage_count + 1):
        choices = candidates(stages, count, offset)
        prompt = stage_choice_prompt("Assemble the object", stages, choices)
        assert f"with {stage_count} stages in chronological task order:" in prompt
        assert [line for line in prompt.splitlines() if re.match(r"\d+\. ", line)] == [
            f"{index + 1}. {stage}" for index, stage in enumerate(stages)]
        choice_lines = [line for line in prompt.splitlines() if re.match(r"[A-Z]: ", line)]
        assert choice_lines == [f"{chr(65 + index)}: {text}" for index, text in enumerate(choices)]
        assert "Stage 0:" not in prompt
        assert "Focus stage" not in prompt and "DONE" not in prompt
        assert "first unfinished stage" in prompt and "select the final stage" in prompt
        assert not any(word in prompt for word in ("offset", "choice_to_stage", "target", "distractor", "source_task"))


def test_def_prompt_fills_abc_and_ghij_with_identical_choice_format():
    prompt = stage_choice_prompt("Install the wing", STAGES, candidates())
    lines = [line for line in prompt.splitlines() if re.match(r"[A-Z]: ", line)]
    assert len(lines) == 10
    assert lines[3:6] == [f"{letter}: {text}" for letter, text in zip("DEF", STAGES, strict=True)]
    for index in (0, 1, 2, 6, 7, 8, 9):
        assert lines[index] == f"{chr(65 + index)}: Unrelated operation {index}."


def test_augmented_messages_keep_the_same_mapping_for_all_eight_frames():
    prompt = with_views(stage_choice_prompt("Install the wing", STAGES, candidates()), ("top", "side"))
    images = tuple(object() for _ in range(16))
    content = serialize_messages(prompt, "C", images, ("top", "side"))[0]["content"]
    assert content[0] == {"type": "text", "text": prompt}
    assert sum(item.get("text") == STAGE_MARKER for item in content) == 8
    assert [item["image"] for item in content if item["type"] == "image"] == list(images)
    # The text API cannot consume current frame labels or mapping provenance.
    assert list(inspect.signature(stage_choice_prompt).parameters) == ["task", "stages", "candidate_prompts"]


@pytest.mark.parametrize("stages,choices,match", [
    ((), (), "1..26"),
    (STAGES, STAGES[:2], "1..26"),
    (STAGES, tuple(f"choice {i}" for i in range(27)), "1..26"),
    (STAGES, tuple(reversed(STAGES)), "contiguous ordered block"),
    (STAGES, (STAGES[0], "foreign", *STAGES[1:]), "contiguous ordered block"),
    (("Only",), ("Only", "Only"), "contiguous ordered block"),
    (STAGES, (*STAGES, "bad <|stage_token|>"), "reserved"),
    (STAGES, (*STAGES, ""), "nonempty"),
])
def test_invalid_choice_text_fails_before_serialization(stages, choices, match):
    with pytest.raises(ValueError, match=match):
        stage_choice_prompt("task", stages, choices)


def test_canonical_prompt_keeps_original_letter_stage_mapping():
    prompt = stage_prompt("Install the wing", STAGES)
    assert [line for line in prompt.splitlines() if re.match(r"[A-Z]: ", line)] == [
        f"{letter}: Stage {index}: {stage}" for index, (letter, stage) in enumerate(zip("ABC", STAGES, strict=True))]
    assert "Answer choices:" not in prompt
    assert "from A, B, C." in prompt
