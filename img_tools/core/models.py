"""Shared data models for image inspection and ROI operations."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Roi:
    """A rectangle expressed as an inclusive origin and exclusive width/height."""

    x: int
    y: int
    width: int
    height: int
    name: str = "ROI"

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("ROI 的宽和高必须大于 0")

    @property
    def x2(self) -> int:
        return self.x + self.width

    @property
    def y2(self) -> int:
        return self.y + self.height

    def with_name(self, name: str) -> "Roi":
        return Roi(self.x, self.y, self.width, self.height, name)

    def as_xywh(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.width, self.height

    def as_xyxy(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.x2, self.y2


@dataclass(frozen=True, slots=True)
class BatchCropResult:
    processed_files: int
    written_files: int
    errors: tuple[str, ...]
    cancelled: bool = False
