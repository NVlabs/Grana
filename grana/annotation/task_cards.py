# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compile explicit task variants into the existing annotation context contract."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json


def _text(value, name):
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{name} must be nonempty text without surrounding whitespace")
    return value


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def compile_task_settings(document, profile_name, task, views, *, owner_policy=None,
                          expected_registry=None):
    """Bind one task/profile explicitly; no task-name-only variant inference.

    The result can be passed directly to ``prepare_annotation_evidence.prepare``.
    The cards are included in task_policy.rules, so the vision API actually sees
    them. This operation does not calibrate or accept annotations.
    """
    if not isinstance(document, dict) or document.get("schema_version") != "grana.annotation_task_cards.v1":
        raise ValueError("Unsupported task-card schema_version")
    _text(document.get("version"), "task-card version")
    if document.get("status") not in ("draft", "calibrated"):
        raise ValueError("Task-card status must be draft or calibrated")
    if document.get("stage_semantics") != "active_subtask":
        raise ValueError("Task cards require active_subtask semantics")
    _text(document.get("common_rules"), "common_rules")
    contract = document.get("aq_contract")
    required_contract = {"normal", "missed_acquisition", "state_loss", "normal_recovery",
                         "unknown", "temporal_unit", "outcomes"}
    if not isinstance(contract, dict) or set(contract) != required_contract:
        raise ValueError("aq_contract must declare normal, acquisition, loss, recovery, uncertainty, time and outcomes")
    for key, value in contract.items():
        _text(value, f"aq_contract.{key}")
    profiles = document.get("profiles")
    if not isinstance(profiles, dict) or profile_name not in profiles:
        raise ValueError("Choose an explicit supported task-card profile")
    profile = profiles[profile_name]
    aliases = profile.get("tasks") if isinstance(profile, dict) else None
    if (not isinstance(aliases, list) or not aliases
            or any(not isinstance(x, str) or not x.strip() for x in aliases)
            or len(set(aliases)) != len(aliases)):
        raise ValueError("Profile tasks must be distinct explicit task names")
    if task not in aliases:
        raise ValueError("Task is not an explicit alias of the selected profile")
    _text(profile.get("task_instruction"), "task_instruction")
    registry, cards = profile.get("registry"), profile.get("cards")
    if not isinstance(registry, list) or not 1 <= len(registry) <= 26:
        raise ValueError("Profile registry must contain 1..26 stages")
    for stage in registry:
        if not isinstance(stage, dict) or set(stage) != {"id", "label", "prompt"}:
            raise ValueError("Registry stages require exactly id, label and prompt")
        for key, value in stage.items():
            _text(value, f"registry.{key}")
    for key in ("id", "label"):
        if len({stage[key] for stage in registry}) != len(registry):
            raise ValueError(f"Registry {key} values must be unique")
    if expected_registry is not None and registry != expected_registry:
        raise ValueError("Selected task-card registry differs from the expected variant")
    if not isinstance(cards, list) or len(cards) != len(registry):
        raise ValueError("Every registry stage needs a task card")
    required_card = {"id", "objective", "entry", "completion", "normal_operations", "visibility_limits"}
    for stage, card in zip(registry, cards, strict=True):
        if not isinstance(card, dict) or set(card) != required_card or card.get("id") != stage["id"]:
            raise ValueError("Cards must match registry order and declare all decision fields")
        for key in required_card:
            value = card[key]
            if key == "normal_operations" and isinstance(value, list):
                if not value:
                    raise ValueError("normal_operations must not be empty")
                for operation in value:
                    _text(operation, "normal operation")
            else:
                _text(value, f"card.{key}")
    if (not isinstance(views, dict) or not views
            or any(not isinstance(k, str) or not k.strip() or not isinstance(v, str) or not v.strip()
                   for k, v in views.items()) or len(set(views.values())) != len(views)):
        raise ValueError("views must map distinct view names to distinct camera features")
    resolved = {"task": task, "profile": profile_name, "aq_contract": contract,
                "cards": cards, "stage_semantics": "active_subtask"}
    rules = document["common_rules"] + "\n\nResolved task cards:\n" + _canonical(resolved)
    override = document.get("task_owner_override")
    if override is not None and (not isinstance(override, dict)
            or not {"task", "profile", "reference"} <= set(override)):
        raise ValueError("task_owner_override must declare its exact task/profile/reference scope")
    if override is None and owner_policy is not None:
        raise ValueError("An owner policy requires an explicit task_owner_override scope")
    applied = override is not None and task == override["task"] and profile_name == override["profile"]
    if applied:
        if (not isinstance(owner_policy, dict) or set(owner_policy) != {"version", "status", "rules"}
                or owner_policy.get("status") not in ("draft", "calibrated")):
            raise ValueError("This exact task/profile requires its explicit task-owner policy")
        _text(owner_policy["version"], "owner policy version")
        _text(owner_policy["rules"], "owner policy rules")
        rules += "\n\nTask-owner override takes precedence within its declared stage scope:\n" + owner_policy["rules"]
    version_hash = hashlib.sha256(_canonical({"rules": rules, "registry": registry}).encode()).hexdigest()[:12]
    return {"views": deepcopy(views), "task": task, "task_instruction": profile["task_instruction"],
            "stage_semantics": "active_subtask", "registry": deepcopy(registry),
            "task_policy": {"version": f"{document['version']}:{profile_name}:{version_hash}",
                            "status": "draft", "rules": rules}}
