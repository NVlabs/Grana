# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve historical flat imports without loading the training stack eagerly.

New code uses the domain packages. Aliases return the canonical module itself,
so existing callers and patches share one set of classes and module state.
"""
from importlib import import_module
from importlib.abc import Loader, MetaPathFinder
from importlib.util import spec_from_loader
import sys


_ALIASES = {
    "grana.annotate": "grana.annotation.workflow",
    "grana.annotation_api": "grana.annotation.api",
    "grana.annotation_bridge": "grana.annotation.bridge",
    "grana.annotation_refinement": "grana.annotation.refinement",
    "grana.annotation_review": "grana.annotation.review",
    "grana.annotation_schema": "grana.annotation.schema",
    "grana.annotation_source": "grana.annotation.source",
    "grana.annotation_task_cards": "grana.annotation.task_cards",
    "grana.annotation_windows": "grana.annotation.windows",
    "grana.cache": "grana.data.cache",
    "grana.preprocess": "grana.data.builder",
    "grana.roles": "grana.data.sampling",
    "grana.progress": "grana.data.targets.legacy",
    "grana.target_composition": "grana.data.targets.composition",
    "grana.checkpoint": "grana.modeling.checkpoint",
    "grana.inputs": "grana.modeling.inputs",
    "grana.model": "grana.modeling.model",
    "grana.prompts": "grana.modeling.prompts",
    "grana.train": "grana.training.engine",
    "grana.evaluate": "grana.training.evaluation",
    "grana.losses": "grana.training.losses",
    "grana.logging": "grana.training.logging",
    "grana.inference_cli": "grana.inference.cli",
    "grana.client": "grana.inference.client",
    "grana.serve": "grana.inference.server",
}


class _AliasLoader(Loader):
    def __init__(self, destination):
        self.destination = destination

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        # Keep the canonical module's spec and name intact for pickling,
        # inspection and Ray serialization.
        sys.modules[module.__name__] = import_module(self.destination)

    def get_code(self, fullname):
        # Preserve python -m grana.serve/client/annotation_review as well as
        # imports, without duplicating their command-line implementations.
        source = (
            "from runpy import run_module\n"
            f"run_module({self.destination!r}, run_name='__main__', alter_sys=True)\n"
        )
        return compile(source, f"<{fullname} compatibility entry point>", "exec")


class _LegacyImports(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        destination = _ALIASES.get(fullname)
        if destination is not None:
            return spec_from_loader(fullname, _AliasLoader(destination))
        return None


def install():
    """Add a fallback for the explicit legacy names once per interpreter."""
    if not any(isinstance(finder, _LegacyImports) for finder in sys.meta_path):
        sys.meta_path.append(_LegacyImports())
