"""Deterministic CAB-F template/calibration decisions.

This module is deliberately independent from OpenCV, model runtimes, and the
filesystem.  The CAB-F shell computes small image/statistics facts and invokes
the actual model or warp; this module owns the rules which decide whether those
facts are usable and how half-resolution matcher offsets map back to the
full-size calibration reference.

Coordinates in this module follow the existing CAB-F convention:

* image sizes are ``(width, height)``;
* offsets and points are ``(row, column)``;
* a positive offset moves a source pixel down/right in the full-size
  calibration transform.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Literal, TypeAlias


ImageSize: TypeAlias = tuple[int, int]
Offset: TypeAlias = tuple[int, int]
SourceShape: TypeAlias = tuple[int, ...]


@dataclass(frozen=True, slots=True)
class SourceImageFacts:
    """Facts computed by the image-reading shell before source validation.

    ``shape`` uses NumPy's native ``(height, width, channels)`` order.  The
    statistics are intentionally scalar so no array/model dependency leaks
    into the functional core.
    """

    shape: SourceShape
    channel_spread_max: float
    near_binary_ratio: float


SourceImageRejection = Literal["invalid_shape", "binary_template"]


@dataclass(frozen=True, slots=True)
class SourceImageDecision:
    """Stable decision for accepting an initial original colour image."""

    accepted: bool
    rejection: SourceImageRejection | None = None


def decide_source_image(
    facts: SourceImageFacts,
    *,
    max_channel_spread: float = 1.0,
    min_near_binary_ratio: float = 0.995,
) -> SourceImageDecision:
    """Reject malformed images and existing black/white matching templates.

    The thresholds mirror the production rule in ``cabf_config`` exactly:
    channel spread must be at most one and at least 99.5% of pixels in the
    first channel must be near black/near white for an image to be considered
    a binary template.
    """

    shape = facts.shape
    if len(shape) != 3 or shape[0] <= 0 or shape[1] <= 0 or shape[2] != 3:
        return SourceImageDecision(False, "invalid_shape")
    if not isfinite(facts.channel_spread_max) or not isfinite(facts.near_binary_ratio):
        return SourceImageDecision(False, "invalid_shape")
    if facts.channel_spread_max <= max_channel_spread and facts.near_binary_ratio >= min_near_binary_ratio:
        return SourceImageDecision(False, "binary_template")
    return SourceImageDecision(True)


@dataclass(frozen=True, slots=True)
class ForegroundFacts:
    """Mask cardinality facts computed by the segmentation shell."""

    foreground_pixels: int
    total_pixels: int


@dataclass(frozen=True, slots=True)
class ForegroundDecision:
    """Whether the segmentation mask contains usable foreground."""

    accepted: bool
    foreground_pixels: int
    total_pixels: int


def decide_foreground(facts: ForegroundFacts) -> ForegroundDecision:
    """Apply CAB-F's foreground rule (at least one pixel below 128)."""

    foreground = max(int(facts.foreground_pixels), 0)
    total = max(int(facts.total_pixels), 0)
    return ForegroundDecision(total > 0 and foreground > 0, foreground, total)


def _validate_size(size: ImageSize, *, name: str) -> ImageSize:
    if len(size) != 2:
        raise ValueError(f"{name} must be (width, height)")
    width, height = (int(size[0]), int(size[1]))
    if width <= 0 or height <= 0:
        raise ValueError(f"{name} must contain positive dimensions")
    return width, height


def _opencv_half_dimension(value: int) -> int:
    """Match ``cv2.resize(..., fx=.5)``'s cvRound dimension calculation.

    OpenCV's default ``cvRound`` on the supported Python builds uses
    nearest-even rounding; Python's ``round`` has the same tie behaviour.
    Keeping this explicit prevents a later shell change to an implicit dsize
    from silently changing odd-size calibration geometry.
    """

    return int(round(value * 0.5))


@dataclass(frozen=True, slots=True)
class HalfScalePlan:
    """Plan for the model shell's half-resolution segmentation input."""

    source_size: ImageSize
    resized_size: ImageSize
    scale: float = 0.5
    interpolation: Literal["area"] = "area"


def plan_half_scale(source_size: ImageSize) -> HalfScalePlan:
    """Return the deterministic half-resolution dimensions for a source."""

    width, height = _validate_size(source_size, name="source_size")
    return HalfScalePlan(
        source_size=(width, height),
        resized_size=(_opencv_half_dimension(width), _opencv_half_dimension(height)),
    )


def half_to_full_offset(
    half_offset: Offset,
    *,
    source_size: ImageSize,
    template_size: ImageSize,
) -> Offset:
    """Convert matcher offset (on the half image/template) to full-size.

    The matcher reports ``(row, column)``.  Each axis is scaled independently
    using the original source and the actual refined template dimensions; this
    preserves the historical CAB-F behaviour when the matcher crops or pads
    its template.  ``int(round(...))`` intentionally preserves Python/OpenCV
    nearest-even rounding and the sign of negative offsets.
    """

    source_width, source_height = _validate_size(source_size, name="source_size")
    template_width, template_height = _validate_size(template_size, name="template_size")
    if len(half_offset) != 2:
        raise ValueError("half_offset must be (row, column)")
    half_row, half_column = (float(half_offset[0]), float(half_offset[1]))
    if not isfinite(half_row) or not isfinite(half_column):
        raise ValueError("half_offset must contain finite values")
    return (
        int(round(half_row * source_height / template_height)),
        int(round(half_column * source_width / template_width)),
    )


@dataclass(frozen=True, slots=True)
class CalibrationTransformPlan:
    """Pure transform contract consumed by the OpenCV shell adapter."""

    source_size: ImageSize
    template_size: ImageSize
    half_offset: Offset
    full_offset: Offset
    output_size: ImageSize
    angle_degrees: float = 0.0
    scale: float = 1.0
    border_value: int = 0


def plan_calibration_transform(
    source_size: ImageSize,
    template_size: ImageSize,
    half_offset: Offset,
    *,
    angle_degrees: float = 0.0,
    scale: float = 1.0,
    border_value: int = 0,
) -> CalibrationTransformPlan:
    """Build the full-size calibration transform/ROI coordinate contract."""

    source_size = _validate_size(source_size, name="source_size")
    template_size = _validate_size(template_size, name="template_size")
    if not isfinite(float(angle_degrees)) or not isfinite(float(scale)) or float(scale) <= 0:
        raise ValueError("angle_degrees must be finite and scale must be positive")
    full_offset = half_to_full_offset(
        half_offset,
        source_size=source_size,
        template_size=template_size,
    )
    return CalibrationTransformPlan(
        source_size=source_size,
        template_size=template_size,
        half_offset=(int(half_offset[0]), int(half_offset[1])),
        full_offset=full_offset,
        output_size=source_size,
        angle_degrees=float(angle_degrees),
        scale=float(scale),
        border_value=int(border_value),
    )


def translate_point(point: Offset, offset: Offset) -> Offset:
    """Apply a CAB-F row/column translation to an ROI point."""

    if len(point) != 2 or len(offset) != 2:
        raise ValueError("point and offset must be (row, column)")
    return int(point[0]) + int(offset[0]), int(point[1]) + int(offset[1])


__all__ = [
    "CalibrationTransformPlan",
    "ForegroundDecision",
    "ForegroundFacts",
    "HalfScalePlan",
    "ImageSize",
    "Offset",
    "SourceImageDecision",
    "SourceImageFacts",
    "decide_foreground",
    "decide_source_image",
    "half_to_full_offset",
    "plan_calibration_transform",
    "plan_half_scale",
    "translate_point",
]
