"""CLI adapter for the toolbox Excel NG image extractor."""
from _bootstrap import ensure_import_paths

ensure_import_paths(include_shared=False)

from cosmos_toolbox.excel_ng_images import main  # noqa: E402

if __name__ == '__main__':
    raise SystemExit(main())
