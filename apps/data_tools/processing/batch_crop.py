"""批量裁剪 — 使用相同 ROI 矩形批量裁剪文件夹中的所有图片。"""
from __future__ import annotations

import os
from pathlib import Path


from .image_io import read_image, write_image
from ..common.helpers import ensure_dir, get_image_files


def crop_single_image(
    image_path: str,
    rects: list[tuple[int, int, int, int]],
    ref_width: int,
    ref_height: int,
    output_dir: str,
) -> int:
    """Crop one image using the shared ROI rectangles and flattened naming."""
    if not rects:
        return 0

    img = read_image(image_path)
    if img is None:
        return 0

    if ref_width <= 0 or ref_height <= 0:
        return 0

    ensure_dir(output_dir)

    h, w = img.shape[:2]
    sx = w / ref_width
    sy = h / ref_height
    total = 0

    for ri, (x1, y1, x2, y2) in enumerate(rects):
        cx1 = max(0, round(x1 * sx))
        cy1 = max(0, round(y1 * sy))
        cx2 = min(w, round(x2 * sx))
        cy2 = min(h, round(y2 * sy))
        if cx2 <= cx1 or cy2 <= cy1:
            continue
        crop = img[cy1:cy2, cx1:cx2]
        name = f"{Path(image_path).stem}_roi_{ri + 1}{Path(image_path).suffix}"
        out_path = os.path.join(output_dir, name)
        write_image(out_path, crop)
        total += 1

    return total


def batch_crop(
    input_dir: str,
    rects: list[tuple[int, int, int, int]],
    ref_width: int,
    ref_height: int,
    output_dir: str,
    *,
    progress_callback=None,
) -> int:
    """Crop all images using the same ROI rects, scaling for different sizes.

    All crops are written into output_dir. Each file is suffixed with roi_N
    to avoid collisions when multiple ROI regions are exported.
    """
    files = get_image_files(input_dir)
    if not files or not rects:
        return 0

    ensure_dir(output_dir)

    total = 0
    for fi, fpath in enumerate(files):
        total += crop_single_image(fpath, rects, ref_width, ref_height, output_dir)

        if progress_callback:
            progress_callback(fi + 1, len(files))

    return total
