"""Deterministic ROI crop planning shared by image-workspace shells.

The plan describes which half-open pixel boxes should be sliced and how each
output should be named.  Image decoding, NumPy slicing, collision checks and
file writes remain the responsibility of imperative adapters.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePath
from typing import Iterable, Literal


CoordinateMode = Literal["absolute", "scaled"]
ScaleSemantics = Literal["edges", "origin_size"]
SuffixMode = Literal["preserve", "lower"]


@dataclass(frozen=True, slots=True)
class CropRegion:
    """A requested half-open rectangle in reference-image coordinates."""

    x1: int
    y1: int
    x2: int
    y2: int
    name: str = "ROI"


@dataclass(frozen=True, slots=True)
class CropOperation:
    """One valid pixel slice and its deterministic output filename."""

    region_index: int
    region_name: str
    box: tuple[int, int, int, int]
    output_name: str


@dataclass(frozen=True, slots=True)
class CropRejection:
    """A requested region that cannot produce pixels for this image."""

    region_index: int
    region_name: str
    reason: str


@dataclass(frozen=True, slots=True)
class ImageCropPlan:
    """All valid operations and rejected regions for one decoded image."""

    operations: tuple[CropOperation, ...]
    rejections: tuple[CropRejection, ...]


def build_image_crop_plan(
    *,
    source_name: str,
    image_size: tuple[int, int],
    regions: Iterable[CropRegion],
    reference_size: tuple[int, int],
    coordinate_mode: CoordinateMode,
    scale_semantics: ScaleSemantics,
    suffix_mode: SuffixMode,
) -> ImageCropPlan:
    """Map requested regions to a decoded image without performing effects.

    ``edges`` preserves the workspace adapter's historical rule of scaling
    both rectangle edges independently. ``origin_size`` preserves the legacy
    image tool's historical rule of scaling the origin and size separately.
    The distinction matters for fractional scale factors and is therefore an
    explicit policy rather than an adapter-local implementation detail.
    """

    image_width, image_height = image_size
    reference_width, reference_height = reference_size
    if image_width <= 0 or image_height <= 0:
        raise ValueError("图片宽高必须大于 0")
    if coordinate_mode not in {"absolute", "scaled"}:
        raise ValueError(f"未知的坐标模式：{coordinate_mode}")
    if scale_semantics not in {"edges", "origin_size"}:
        raise ValueError(f"未知的缩放语义：{scale_semantics}")
    if suffix_mode not in {"preserve", "lower"}:
        raise ValueError(f"未知的扩展名策略：{suffix_mode}")
    if coordinate_mode == "scaled" and (
        reference_width <= 0 or reference_height <= 0
    ):
        raise ValueError("参考图宽高必须大于 0")

    path = PurePath(source_name)
    suffix = path.suffix.lower() if suffix_mode == "lower" else path.suffix
    operations: list[CropOperation] = []
    rejections: list[CropRejection] = []

    for region_index, region in enumerate(regions, start=1):
        mapped = _map_region(
            region,
            image_size=image_size,
            reference_size=reference_size,
            coordinate_mode=coordinate_mode,
            scale_semantics=scale_semantics,
        )
        x1 = max(0, mapped[0])
        y1 = max(0, mapped[1])
        x2 = min(image_width, mapped[2])
        y2 = min(image_height, mapped[3])
        if x2 <= x1 or y2 <= y1:
            rejections.append(
                CropRejection(
                    region_index=region_index,
                    region_name=region.name,
                    reason="ROI 与图片没有重叠区域",
                )
            )
            continue
        operations.append(
            CropOperation(
                region_index=region_index,
                region_name=region.name,
                box=(x1, y1, x2, y2),
                output_name=f"{path.stem}_roi_{region_index}{suffix}",
            )
        )

    return ImageCropPlan(tuple(operations), tuple(rejections))


def _map_region(
    region: CropRegion,
    *,
    image_size: tuple[int, int],
    reference_size: tuple[int, int],
    coordinate_mode: CoordinateMode,
    scale_semantics: ScaleSemantics,
) -> tuple[int, int, int, int]:
    if coordinate_mode == "absolute":
        return region.x1, region.y1, region.x2, region.y2

    image_width, image_height = image_size
    reference_width, reference_height = reference_size
    scale_x = image_width / reference_width
    scale_y = image_height / reference_height
    x1 = round(region.x1 * scale_x)
    y1 = round(region.y1 * scale_y)
    if scale_semantics == "edges":
        return (
            x1,
            y1,
            round(region.x2 * scale_x),
            round(region.y2 * scale_y),
        )

    width = max(1, round((region.x2 - region.x1) * scale_x))
    height = max(1, round((region.y2 - region.y1) * scale_y))
    return x1, y1, x1 + width, y1 + height


__all__ = [
    "CoordinateMode",
    "CropOperation",
    "CropRegion",
    "CropRejection",
    "ImageCropPlan",
    "ScaleSemantics",
    "SuffixMode",
    "build_image_crop_plan",
]
