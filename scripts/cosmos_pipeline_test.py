"""Run the complete Cosmos inspection flow without production pages or hardware."""
from __future__ import annotations

try:
    from scripts._bootstrap import ensure_import_paths
except ImportError:  # pragma: no cover
    from _bootstrap import ensure_import_paths


def _main() -> int:
    ensure_import_paths(include_repo=True, include_shared=True)
    from cosmos_toolbox.paths import ensure_import_paths as ensure_cosmos_import_paths

    ensure_cosmos_import_paths()
    from apps.cosmos_pipeline.runner import main

    return main()


if __name__ == "__main__":
    raise SystemExit(_main())
