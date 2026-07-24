"""Polygon-to-mask generation."""
from __future__ import annotations

import cv2
import numpy as np


def polygon_mask(width: int, height: int, polygons: list[list[tuple[int, int]]]) -> np.ndarray:
    """Create a single-channel mask with white filled polygons."""
    if width <= 0 or height <= 0:
        raise ValueError("Mask 宽高必须大于 0")
    mask = np.zeros((height, width), dtype=np.uint8)
    for polygon in polygons:
        if len(polygon) >= 3:
            cv2.fillPoly(mask, [np.asarray(polygon, dtype=np.int32)], 255)
    return mask
