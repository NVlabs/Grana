# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The human-labeling entry point stays independent of model and review launches."""
import builtins
from pathlib import Path
import sys
from types import ModuleType

import pytest

from grana.cli import main


def test_label_help_is_available_without_loading_the_server(monkeypatch, capsys):
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name in {"grana.annotation.label_server", "grana.data.builder", "grana.training.engine",
                    "grana.inference.predictor"}:
            pytest.fail(f"Label help attempted to load {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(sys, "argv", ["grana", "label", "--help"])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 0
    help_text = capsys.readouterr().out
    assert "--config" in help_text and "--port" in help_text and "8790" in help_text


@pytest.mark.parametrize("port_args,expected_port", [([], 8790), (["--port", "9012"], 9012)])
def test_label_loads_the_cache_recipe_and_starts_only_its_server(monkeypatch, tmp_path, port_args, expected_port):
    recipe = tmp_path / "human-labeling.yaml"
    config = {"source": "/data/raw", "labels": "/data/labels"}
    calls = []
    builder = ModuleType("grana.data.builder")
    server = ModuleType("grana.annotation.label_server")
    builder.load_build_config = lambda path: calls.append(("load", path)) or config
    server.serve_labels = lambda cfg, *, port, config_path: calls.append(("serve", cfg, port, config_path))
    monkeypatch.setitem(sys.modules, builder.__name__, builder)
    monkeypatch.setitem(sys.modules, server.__name__, server)
    monkeypatch.setattr("grana.cli.load_config", lambda *_: pytest.fail("Labeling loaded a training recipe"))
    monkeypatch.setattr("grana.cli.subprocess.run", lambda *_, **__: pytest.fail("Labeling launched a subprocess"))
    monkeypatch.setattr(sys, "argv", ["grana", "label", "--config", str(recipe), *port_args])
    main()
    assert calls == [("load", Path(recipe)), ("serve", config, expected_port, recipe)]


def test_label_requires_an_explicit_recipe(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["grana", "label"])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert "--config" in capsys.readouterr().err
