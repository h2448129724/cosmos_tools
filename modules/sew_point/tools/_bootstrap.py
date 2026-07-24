"""Pre-import bootstrap for direct execution of module tool scripts.

Each ``modules/<name>/tools/`` directory carries its own copy of this shim so
that ``python modules/<name>/tools/<script>.py`` works before ``sys.path`` is
configured. It is a no-op when the tool is launched via ``python -m``.

The duplication across tool directories is intentional: the shim must live in
the script's own directory to be importable as ``from _bootstrap import ...``
during direct execution.
"""
from __future__ import annotations

import sys
from pathlib import Path


def _prepend(paths: tuple[Path, ...]) -> None:
    for path in reversed(paths):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)


def ensure_module_paths(script_file: str) -> None:
    """Make a module tool runnable directly.

    Adds the ``modules/`` root (so package imports such as
    ``from segmentation.X`` / ``from sew_point.X`` resolve) and
    ``shared/cabf_common`` (so ``import cabf`` resolves).
    """
    file_path = Path(script_file).resolve()
    modules_root = file_path.parents[2]
    repo_root = file_path.parents[3]
    shared_cabf_root = repo_root / "shared" / "cabf_common"
    _prepend((modules_root, shared_cabf_root))
