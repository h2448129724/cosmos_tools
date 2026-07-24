"""Bootstrap: labeling_ui 顶层入口，负责路径发现。"""
from __future__ import annotations

try:
    from scripts._bootstrap import ensure_import_paths
except ImportError:  # pragma: no cover - direct script execution from scripts/
    from _bootstrap import ensure_import_paths


ensure_import_paths(include_repo=True, include_shared=True)


def main() -> int:
    from apps.labeling_ui.main import main as run_main

    return int(run_main() or 0)


if __name__ == "__main__":
    raise SystemExit(main())
