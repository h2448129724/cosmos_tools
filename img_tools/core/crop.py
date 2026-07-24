"""Single-image and folder ROI crop operations."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Literal

import numpy as np

from .geometry import clamp_roi, scale_roi
from .image_io import iter_image_files, read_image, write_image
from .models import BatchCropResult, Roi
from .output import ConflictPolicy, resolve_output_path

CoordinateMode = Literal["absolute", "scaled"]


def crop_image(image: np.ndarray, roi: Roi) -> tuple[np.ndarray, Roi]:
    """Crop an image, clipping the ROI to its actual bounds."""
    height, width = image.shape[:2]
    effective = clamp_roi(roi, width, height)
    if effective is None:
        raise ValueError("ROI 与图片没有重叠区域")
    return image[effective.y:effective.y2, effective.x:effective.x2].copy(), effective


def batch_crop(
    input_dir: str | Path,
    output_dir: str | Path,
    rois: list[Roi],
    *,
    reference_size: tuple[int, int],
    coordinate_mode: CoordinateMode = "absolute",
    recursive: bool = False,
    preserve_structure: bool = True,
    conflict_policy: ConflictPolicy = "rename",
    progress: Callable[[int, int], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> BatchCropResult:
    """Crop every image in a directory and return a complete result summary."""
    if not rois:
        raise ValueError("请至少添加一个 ROI")
    source_root, destination_root = Path(input_dir), Path(output_dir)
    if source_root.resolve() == destination_root.resolve():
        raise ValueError("输出目录不能与输入目录相同，以免修改原图")
    files = list(iter_image_files(source_root, recursive=recursive))
    errors: list[str] = []
    written = 0
    cancelled = False
    processed = 0
    for index, source in enumerate(files, start=1):
        if should_cancel and should_cancel():
            cancelled = True
            break
        processed = index
        image = read_image(source)
        if image is None:
            errors.append(f"无法读取：{source}")
        else:
            image_h, image_w = image.shape[:2]
            relative_parent = source.relative_to(source_root).parent if preserve_structure else Path()
            for roi_index, roi in enumerate(rois, start=1):
                mapped = scale_roi(roi, reference_size, (image_w, image_h)) if coordinate_mode == "scaled" else roi
                try:
                    cropped, _ = crop_image(image, mapped)
                    suffix = source.suffix.lower()
                    output_name = f"{source.stem}_roi_{roi_index}{suffix}"
                    output_path = resolve_output_path(destination_root / relative_parent / output_name, conflict_policy)
                    if output_path is not None:
                        write_image(output_path, cropped)
                        written += 1
                except (ValueError, OSError) as error:
                    errors.append(f"{source.name} / {roi.name}: {error}")
        if progress is not None:
            progress(index, len(files))
    return BatchCropResult(processed, written, tuple(errors), cancelled)
