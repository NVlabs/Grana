# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy

import pytest

from grana.annotation.task_cards import compile_task_settings


def document():
    def profile(prompt, tasks):
        return {"tasks": tasks, "task_instruction": "Assemble the toy.",
                "registry": [{"id": "S1", "label": "1", "prompt": prompt}],
                "cards": [{"id": "S1", "objective": prompt, "entry": "Approach the part.",
                           "completion": "Part retained at the target.",
                           "normal_operations": ["Acquire", "Align", "Recover"],
                           "visibility_limits": "Do not infer hidden contact."}]}
    return {"schema_version": "grana.annotation_task_cards.v1", "version": "test-v1", "status": "draft",
            "stage_semantics": "active_subtask", "common_rules": "Normal recovery ends the defective attempt.",
            "aq_contract": {key: "Explicit criterion." for key in
                            ("normal", "missed_acquisition", "state_loss", "normal_recovery",
                             "unknown", "temporal_unit", "outcomes")},
            "profiles": {"wheels": profile("Install wheels.", ["car"]),
                         "rims": profile("Fit rims first.", ["car"]),
                         "tractor": profile("Insert screws.", ["tractor", "tractor-recovery"])},
            "task_owner_override": {"task": "tractor", "profile": "tractor", "reference": "owner.json"}}


VIEWS = {"top": "observation.images.top"}
OWNER = {"version": "owner-v1", "status": "draft", "rules": "Scoped owner rule applies only to the declared stage."}


def test_same_task_name_requires_explicit_variant_and_compiles_cards_into_rules():
    cards = document()
    wheels = compile_task_settings(cards, "wheels", "car", VIEWS)
    rims = compile_task_settings(cards, "rims", "car", VIEWS)
    assert wheels["registry"] != rims["registry"]
    assert "Install wheels." in wheels["task_policy"]["rules"]
    assert "Fit rims first." in rims["task_policy"]["rules"]
    assert "Fit rims first." not in wheels["task_policy"]["rules"]
    assert wheels["task_policy"]["version"] != rims["task_policy"]["version"]
    assert wheels == compile_task_settings(cards, "wheels", "car", VIEWS)
    assert cards == document()
    wheels["registry"][0]["prompt"] = "Changed downstream"
    assert cards == document()


def test_variant_mismatch_is_rejected_instead_of_silently_remapping():
    cards = document()
    with pytest.raises(ValueError, match="expected variant"):
        compile_task_settings(cards, "wheels", "car", VIEWS,
                              expected_registry=cards["profiles"]["rims"]["registry"])
    with pytest.raises(ValueError, match="explicit alias"):
        compile_task_settings(cards, "wheels", "tractor", VIEWS)


def test_owner_override_only_enters_exact_declared_task_and_profile():
    cards = document()
    with pytest.raises(ValueError, match="explicit task-owner policy"):
        compile_task_settings(cards, "tractor", "tractor", VIEWS)
    ordinary = compile_task_settings(cards, "tractor", "tractor", VIEWS, owner_policy=OWNER)
    recovery = compile_task_settings(cards, "tractor", "tractor-recovery", VIEWS, owner_policy=OWNER)
    assert OWNER["rules"] in ordinary["task_policy"]["rules"]
    assert OWNER["rules"] not in recovery["task_policy"]["rules"]
    assert ordinary["task_policy"]["status"] == "draft"


def test_generic_tasks_do_not_require_an_invented_owner_override():
    cards = document()
    expected = compile_task_settings(cards, "wheels", "car", VIEWS)
    del cards["task_owner_override"]
    assert compile_task_settings(cards, "wheels", "car", VIEWS) == expected
    with pytest.raises(ValueError, match="explicit task_owner_override scope"):
        compile_task_settings(cards, "wheels", "car", VIEWS, owner_policy=OWNER)


@pytest.mark.parametrize("scope", [{}, {"task": "car"}])
def test_an_explicit_but_incomplete_override_is_not_silently_ignored(scope):
    cards = document()
    cards["task_owner_override"] = scope
    with pytest.raises(ValueError, match="exact task/profile/reference scope"):
        compile_task_settings(cards, "wheels", "car", VIEWS)


@pytest.mark.parametrize("mutation", [
    lambda d: d.update(stage_semantics="attained_state"),
    lambda d: d["profiles"]["wheels"]["cards"][0].update(id="S2"),
    lambda d: d["profiles"]["wheels"]["cards"][0].update(normal_operations=[]),
    lambda d: d["profiles"]["wheels"]["registry"][0].update(prompt=""),
    lambda d: d["aq_contract"].pop("normal_recovery"),
])
def test_incomplete_or_incompatible_task_contracts_fail(mutation):
    cards = deepcopy(document())
    mutation(cards)
    with pytest.raises(ValueError):
        compile_task_settings(cards, "wheels", "car", VIEWS)
