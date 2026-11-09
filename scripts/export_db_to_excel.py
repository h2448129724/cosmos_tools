"""CLI adapter for the toolbox read-only SQLite Excel exporter."""
from _bootstrap import ensure_import_paths

ensure_import_paths(include_shared=False)

from cosmos_toolbox.database_excel import main  # noqa: E402

if __name__ == '__main__':
    raise SystemExit(main())
