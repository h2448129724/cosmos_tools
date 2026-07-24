"""Shared bootstrap helpers for repository-level script entry points."""
from __future__ import annotations

import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SHARED_ROOT = REPO_ROOT / "shared" / "cabf_common"


def ensure_import_paths(include_repo: bool = True, include_shared: bool = True) -> None:
    paths: list[Path] = []
    if include_repo:
        paths.append(REPO_ROOT)
    if include_shared:
        paths.append(SHARED_ROOT)

    for path in reversed(paths):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)


def python_env(include_repo: bool = True, include_shared: bool = True) -> dict[str, str]:
    env = dict(os.environ)
    paths: list[str] = []
    if include_repo:
        paths.append(str(REPO_ROOT))
    if include_shared:
        paths.append(str(SHARED_ROOT))

    current = env.get("PYTHONPATH", "")
    if current:
        paths.append(current)
    env["PYTHONPATH"] = os.pathsep.join(paths)
    return env
