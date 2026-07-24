from __future__ import annotations

import os
from pathlib import Path


# Canonical image-extension set. Duplicated from cabf.constants.IMAGE_SUFFIXES
# intentionally: this is the dependency-free foundation module, so it must not
# import the shared cabf package at module load time. The values are stable.
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def get_image_files(path: str, extensions: set[str] | None = None) -> list[str]:
    """Recursively find all image files in a directory."""
    extensions = extensions or _IMAGE_SUFFIXES
    files: list[str] = []
    p = Path(path)
    if p.is_file():
        return [str(p)] if p.suffix.lower() in extensions else []
    for root, _, filenames in os.walk(path):
        for f in filenames:
            if Path(f).suffix.lower() in extensions:
                files.append(os.path.join(root, f))
    return sorted(files)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)
