"""自动裁剪 — 将图片按固定尺寸自动切割成多个小块。"""
from __future__ import annotations

import math
import os
from pathlib import Path

import cv2

from .image_io import read_image, write_image
from ..common.helpers import ensure_dir, get_image_files


def compute_tile_grid(
    img_w: int, img_h: int, tile_w: int, tile_h: int, allow_overlap: bool = True,
) -> list[tuple[int, int, int, int, int, int]]:
    """Compute tile positions for an image. Returns list of (x, y, w, h, row, col).

    If allow_overlap is true and the image can't be evenly divided, the last tile
    shifts back so a full tile still fits in bounds.
    """
    cols = max(1, math.ceil(img_w / tile_w))
    rows = max(1, math.ceil(img_h / tile_h))
    tiles = []
    for r in range(rows):
        for c in range(cols):
            x = c * tile_w
            y = r * tile_h
            # Shift the last tile so it remains in bounds, or keep a truncated edge tile.
            if allow_overlap and x + tile_w > img_w:
                x = max(0, img_w - tile_w)
            if allow_overlap and y + tile_h > img_h:
                y = max(0, img_h - tile_h)
            if allow_overlap:
                actual_w = tile_w
                actual_h = tile_h
            elif img_w > x:
                actual_w = min(tile_w, img_w - x)
                actual_h = min(tile_h, img_h - y) if img_h > y else tile_h
            else:
                actual_w = tile_w
                actual_h = tile_h
            tiles.append((x, y, actual_w, actual_h, r, c))
    return tiles


def batch_tile_crop(
    input_dir: str,
    output_dir: str,
    tile_w: int,
    tile_h: int,
    allow_overlap: bool = True,
    *,
    progress_callback=None,
) -> int:
    """Tile-crop all images in input_dir.

    Output naming: {original_name}_tile_{row}_{col}.ext
    Returns total tiles written.
    """
    files = get_image_files(input_dir)
    if not files:
        return 0

    ensure_dir(output_dir)
    total = 0
    for fi, fpath in enumerate(files):
        img = read_image(fpath)
        if img is None:
            continue
        h, w = img.shape[:2]
        tiles = compute_tile_grid(w, h, tile_w, tile_h, allow_overlap=allow_overlap)
        stem = Path(fpath).stem
        ext = Path(fpath).suffix
        for x, y, tw, th, row, col in tiles:
            # Skip undersized edge tiles when overlap is disabled so output keeps a fixed size.
            if tw < tile_w or th < tile_h:
                continue
            tile_img = img[y:y + th, x:x + tw]
            if allow_overlap and (tile_img.shape[1] < tile_w or tile_img.shape[0] < tile_h):
                # A tile cannot be shifted far enough when the source image itself is
                # smaller than the requested size. Keep the source pixels unchanged
                # at the top-left and pad the missing right/bottom area with black.
                pad_right = tile_w - tile_img.shape[1]
                pad_bottom = tile_h - tile_img.shape[0]
                tile_img = cv2.copyMakeBorder(
                    tile_img,
                    0,
                    pad_bottom,
                    0,
                    pad_right,
                    borderType=cv2.BORDER_CONSTANT,
                    value=0,
                )
            out_name = f"{stem}_tile_{row}_{col}{ext}"
            write_image(os.path.join(output_dir, out_name), tile_img)
            total += 1
        if progress_callback:
            progress_callback(fi + 1, len(files))

    return total
