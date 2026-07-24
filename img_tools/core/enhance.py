"""Common image enhancement operations, reusable by preview and batch pipelines."""
from __future__ import annotations


import cv2
import numpy as np


def enhance_image(image: np.ndarray, *, brightness: int = 0, contrast: float = 1.0, gamma: float = 1.0, blur_radius: int = 0, sharpen: bool = False, threshold: int | None = None) -> np.ndarray:
    """Apply deterministic, composable image enhancement operations."""
    if gamma <= 0:
        raise ValueError("Gamma 必须大于 0")
    result = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)
    if gamma != 1.0:
        table = np.array([((index / 255.0) ** (1.0 / gamma)) * 255 for index in range(256)]).astype(np.uint8)
        result = cv2.LUT(result, table)
    if blur_radius:
        kernel = max(1, blur_radius * 2 + 1)
        result = cv2.GaussianBlur(result, (kernel, kernel), 0)
    if sharpen:
        blurred = cv2.GaussianBlur(result, (0, 0), 2)
        result = cv2.addWeighted(result, 1.6, blurred, -0.6, 0)
    if threshold is not None:
        gray = cv2.cvtColor(result, cv2.COLOR_BGRA2GRAY if result.ndim == 3 and result.shape[2] == 4 else cv2.COLOR_BGR2GRAY) if result.ndim == 3 else result
        _, result = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return result
