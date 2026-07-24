"""ROI geometry with one explicit coordinate convention.

Every rectangle is xywh: its origin is inclusive and its right/bottom edges are
exclusive.  That convention maps directly to NumPy slicing and prevents
off-by-one ambiguity.
"""
from __future__ import annotations

from .models import Roi


def clamp_roi(roi: Roi, image_width: int, image_height: int) -> Roi | None:
    """Return the part of *roi* inside an image, or ``None`` if it misses it."""
    if image_width <= 0 or image_height <= 0:
        raise ValueError("图片宽高必须大于 0")
    x1 = max(0, roi.x)
    y1 = max(0, roi.y)
    x2 = min(image_width, roi.x2)
    y2 = min(image_height, roi.y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return Roi(x1, y1, x2 - x1, y2 - y1, roi.name)


def scale_roi(roi: Roi, reference_size: tuple[int, int], target_size: tuple[int, int]) -> Roi:
    """Scale an ROI from a reference image size to a target image size."""
    ref_w, ref_h = reference_size
    target_w, target_h = target_size
    if ref_w <= 0 or ref_h <= 0:
        raise ValueError("参考图宽高必须大于 0")
    return Roi(
        round(roi.x * target_w / ref_w),
        round(roi.y * target_h / ref_h),
        max(1, round(roi.width * target_w / ref_w)),
        max(1, round(roi.height * target_h / ref_h)),
        roi.name,
    )
