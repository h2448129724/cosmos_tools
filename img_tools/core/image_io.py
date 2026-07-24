"""Unicode-safe OpenCV image IO."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"})


def read_image(path: str | Path) -> np.ndarray | None:
    """Read an image from a Unicode path without using ``cv2.imread``."""
    try:
        raw = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if raw.size == 0:
        return None
    return cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)


def write_image(path: str | Path, image: np.ndarray, *, quality: int = 95) -> None:
    """Write an image to a Unicode path, creating its parent directory."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    suffix = output.suffix.lower()
    if suffix not in IMAGE_SUFFIXES:
        raise ValueError(f"不支持的图片格式：{suffix or '无扩展名'}")
    params: list[int] = []
    if suffix in {".jpg", ".jpeg"}:
        params = [cv2.IMWRITE_JPEG_QUALITY, max(0, min(100, quality))]
    elif suffix == ".webp":
        params = [cv2.IMWRITE_WEBP_QUALITY, max(0, min(100, quality))]
    elif suffix == ".png":
        params = [cv2.IMWRITE_PNG_COMPRESSION, 3]
    ok, encoded = cv2.imencode(suffix, image, params)
    if not ok:
        raise OSError(f"无法编码图片：{output}")
    encoded.tofile(str(output))


def iter_image_files(folder: str | Path, *, recursive: bool = False):
    """Yield supported image files in deterministic name order."""
    root = Path(folder)
    paths = root.rglob("*") if recursive else root.glob("*")
    yield from sorted(path for path in paths if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
