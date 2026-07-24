"""Bootstrap: cabf_flow 顶层入口，负责路径发现。"""
from __future__ import annotations

# ruff: noqa: E402  -- path bootstrap must run before importing the application

try:
    from scripts._bootstrap import ensure_import_paths
except ImportError:  # pragma: no cover - direct script execution from scripts/
    from _bootstrap import ensure_import_paths


ensure_import_paths(include_repo=True, include_shared=False)

from apps.cabf_flow.flow import main


if __name__ == "__main__":
    raise SystemExit(main())
