# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Domain boundaries preserve lightweight imports and existing integrations."""
import importlib
import pickle
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_domain_and_annotation_imports_do_not_load_model_runtimes():
    # A fresh interpreter catches accidental eager imports hidden by the rest
    # of the suite, which legitimately imports torch during collection.
    code = """
import importlib
import sys
for name in (
    'grana.annotation', 'grana.data', 'grana.data.targets', 'grana.modeling',
    'grana.training', 'grana.inference', 'grana.annotation.workflow',
    'grana.annotation_api', 'grana.annotation_schema', 'grana.progress',
):
    importlib.import_module(name)
assert not {'torch', 'ray', 'transformers', 'peft'} & sys.modules.keys()
assert 'grana.inference.predictor' not in sys.modules
"""
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("legacy_name,canonical_name", [
    ("grana.annotate", "grana.annotation.workflow"),
    ("grana.annotation_api", "grana.annotation.api"),
    ("grana.cache", "grana.data.cache"),
    ("grana.progress", "grana.data.targets.legacy"),
    ("grana.model", "grana.modeling.model"),
    ("grana.train", "grana.training.engine"),
    ("grana.logging", "grana.training.logging"),
    ("grana.client", "grana.inference.client"),
    ("grana.serve", "grana.inference.server"),
])
def test_legacy_imports_share_module_state_and_keep_canonical_metadata(legacy_name, canonical_name, monkeypatch):
    legacy = importlib.import_module(legacy_name)
    canonical = importlib.import_module(canonical_name)
    assert legacy is canonical
    assert canonical.__name__ == canonical.__spec__.name == canonical_name
    marker = object()
    monkeypatch.setattr(legacy, "_integration_probe", marker, raising=False)
    assert canonical._integration_probe is marker


def test_legacy_worker_and_data_types_remain_pickleable():
    from grana.cache import Trajectory
    from grana.train import train

    assert Trajectory.__module__ == "grana.data.cache"
    assert train.__module__ == "grana.training.engine"
    for value in (Trajectory, train):
        assert pickle.loads(pickle.dumps(value)) is value


def test_public_predictor_imports_preserve_type_identity():
    from grana.inference import GranaPredictor, InferenceRequest
    from grana.inference import predictor

    assert GranaPredictor is predictor.GranaPredictor
    assert InferenceRequest is predictor.InferenceRequest


@pytest.mark.parametrize("module,option", [
    ("grana.serve", "--checkpoint"),
    ("grana.inference.server", "--checkpoint"),
    ("grana.client", "--server"),
    ("grana.inference.client", "--server"),
    ("grana.annotation_review", "--run-dir"),
    ("grana.annotation.review", "--run-dir"),
])
def test_legacy_and_domain_module_commands_show_help_without_loading_a_checkpoint(module, option):
    result = subprocess.run([sys.executable, "-m", module, "--help"], cwd=ROOT,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert option in result.stdout
    assert "RuntimeWarning" not in result.stderr
