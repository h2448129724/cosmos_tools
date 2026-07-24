"""Display-only downsampling for very large images."""
from __future__ import annotations

import cv2
import numpy as np


def make_display_preview(image: np.ndarray, *, max_pixels: int = 12_000_000) -> np.ndarray:
    """Return the original or a downsampled copy suitable for interactive display."""
    height, width = image.shape[:2]
    if width * height <= max_pixels:
        return image
    scale = (max_pixels / (width * height)) ** 0.5
    return cv2.resize(image, (max(1, round(width * scale)), max(1, round(height * scale))), interpolation=cv2.INTER_AREA)
