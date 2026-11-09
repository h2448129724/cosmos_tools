"""Bootstrap for CAB-F CLI scripts."""
from __future__ import annotations

try:
    from scripts._bootstrap import REPO_ROOT, ensure_import_paths
except ImportError:  # pragma: no cover - direct script execution from scripts/
    from _bootstrap import REPO_ROOT, ensure_import_paths

__all__ = ["REPO_ROOT"]


ensure_import_paths(include_repo=False, include_shared=True)
