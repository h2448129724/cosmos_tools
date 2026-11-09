"""Public-entry adapter for models owned by the Cosmos CAB-F project."""

from __future__ import annotations

from functools import lru_cache
import importlib
from types import ModuleType
from typing import Any

from cosmos_toolbox.paths import ensure_import_paths


class _LegacyProjectEntry(ModuleType):
    """One lazy adapter for the pre-project-package Cosmos checkout layout."""

    def __getattr__(self, name):
        runtime = {
            'SewPointDetector': 'algo.cab_f.sew_point_detector',
            'SewPointConnector': 'algo.cab_f.sew_point_connector',
        }
        if name in runtime:
            module = runtime[name]
        elif name in {'EdgeGraphNet', 'EdgeGraphCore', 'EdgePatchEncoder'}:
            module = 'train.models.sew_point_connector'
        elif name in {'build_model', 'compare_split_graph', 'compare_torch_models',
                      'export_graph_core', 'export_patch_encoder', 'load_checkpoint',
                      'load_real_graph', 'make_probe_graph'}:
            module = 'train.tools.export_sew_point_connector_onnx'
        else:
            raise AttributeError(name)
        return getattr(importlib.import_module(module), name)


@lru_cache(maxsize=1)
def project_entry() -> ModuleType:
    """Return the CAB-F public entry without importing any lazy model export."""

    ensure_import_paths()
    try:
        projects = importlib.import_module('projects')
    except ModuleNotFoundError as exc:
        if exc.name != 'projects':
            raise
        projects = None
    loader = getattr(projects, 'load_project_package', None)
    if callable(loader):
        # Do not hide initialization failures in a modern project package.
        return loader('CAB-F')
    return _LegacyProjectEntry('cosmos_toolbox.training.legacy_cab_f')


def create_sew_point_detector(**kwargs: Any) -> Any:
    return project_entry().SewPointDetector(**kwargs)


def create_sew_point_connector(**kwargs: Any) -> Any:
    return project_entry().SewPointConnector(**kwargs)


def create_edge_graph_net(**kwargs: Any) -> Any:
    return project_entry().EdgeGraphNet(**kwargs)


__all__ = [
    "create_edge_graph_net",
    "create_sew_point_connector",
    "create_sew_point_detector",
    "project_entry",
]
