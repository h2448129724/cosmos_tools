from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .constants import IMAGE_SUFFIXES


def read_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, data: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def read_image_bgr(image_path: str | Path) -> np.ndarray:
    """Decode an image into a BGR ``uint8`` array, Unicode-path safe.

    Uses ``np.fromfile`` + ``cv2.imdecode(IMREAD_COLOR)`` so non-ASCII paths work
    on Windows (``cv2.imread`` itself is not Unicode-safe there). Raises
    ``FileNotFoundError`` if the file cannot be decoded.
    """
    image = cv2.imdecode(np.fromfile(str(Path(image_path)), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"无法读取图片: {image_path}")
    return image


def write_image(image_path: str | Path, img: np.ndarray, params=None) -> None:
    """Encode and write an image, Unicode-path safe via ``imencode`` + ``tofile``.

    Creates parent directories. ``params`` is forwarded to ``cv2.imencode``
    (e.g. ``[cv2.IMWRITE_PNG_COMPRESSION, 9]``); omit it for codec defaults.
    Raises ``ValueError`` if encoding fails.
    """
    path = Path(image_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encode_args = (path.suffix, img, params) if params is not None else (path.suffix, img)
    try:
        success, buf = cv2.imencode(*encode_args)
    except cv2.error as exc:  # e.g. unsupported extension
        raise ValueError(f"无法编码图片: {path}") from exc
    if not success:
        raise ValueError(f"无法编码图片: {path}")
    buf.tofile(str(path))


def read_image_size(image_path: str | Path) -> tuple[int, int]:
    image = read_image_bgr(image_path)
    h, w = image.shape[:2]
    return int(w), int(h)


def iter_image_files(image_dir: str | Path) -> list[Path]:
    folder = Path(image_dir)
    if not folder.is_dir():
        raise FileNotFoundError(f"图片目录不存在: {folder}")
    files = []
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            files.append(path)
    return files


def iter_json_files(annotation_dir: str | Path) -> list[Path]:
    folder = Path(annotation_dir)
    if not folder.is_dir():
        raise FileNotFoundError(f"标注目录不存在: {folder}")
    return sorted([path for path in folder.iterdir() if path.is_file() and path.suffix.lower() == ".json"])
