"""Public-entry adapter for models owned by the Cosmos CAB-F project."""

from __future__ import annotations

from functools import lru_cache
from types import ModuleType
from typing import Any

from cosmos_toolbox.paths import ensure_import_paths


@lru_cache(maxsize=1)
def project_entry() -> ModuleType:
    """Return the CAB-F public entry without importing any lazy model export."""

    ensure_import_paths()
    from projects import load_project_package

    return load_project_package("CAB-F")


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
