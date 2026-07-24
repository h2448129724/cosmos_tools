"""Common non-destructive resize, rotation, flip and format conversion operations."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Literal

import cv2
import numpy as np
from dataclasses import dataclass

from .image_io import IMAGE_SUFFIXES, iter_image_files, read_image, write_image
from .output import ConflictPolicy, resolve_output_path

Rotation = Literal["none", "cw90", "ccw90", "180"]
Flip = Literal["none", "horizontal", "vertical"]


@dataclass(frozen=True, slots=True)
class BatchTransformResult:
    processed_files: int
    written_files: int
    errors: tuple[str, ...]
    cancelled: bool = False


def transform_image(image: np.ndarray, *, width: int | None = None, height: int | None = None, keep_aspect: bool = True, rotation: Rotation = "none", flip: Flip = "none", grayscale: bool = False) -> np.ndarray:
    """Apply common geometric/color operations in a predictable order."""
    result = image
    if rotation == "cw90":
        result = cv2.rotate(result, cv2.ROTATE_90_CLOCKWISE)
    elif rotation == "ccw90":
        result = cv2.rotate(result, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif rotation == "180":
        result = cv2.rotate(result, cv2.ROTATE_180)
    if flip == "horizontal":
        result = cv2.flip(result, 1)
    elif flip == "vertical":
        result = cv2.flip(result, 0)
    if width or height:
        source_h, source_w = result.shape[:2]
        if keep_aspect:
            if width and height:
                scale = min(width / source_w, height / source_h)
            else:
                scale = (width / source_w) if width else (height / source_h)
            target_w, target_h = max(1, round(source_w * scale)), max(1, round(source_h * scale))
        else:
            target_w, target_h = width or source_w, height or source_h
        interpolation = cv2.INTER_AREA if target_w < source_w or target_h < source_h else cv2.INTER_CUBIC
        result = cv2.resize(result, (target_w, target_h), interpolation=interpolation)
    if grayscale and result.ndim == 3:
        result = cv2.cvtColor(result, cv2.COLOR_BGRA2GRAY if result.shape[2] == 4 else cv2.COLOR_BGR2GRAY)
    return result


def batch_transform(input_dir: str | Path, output_dir: str | Path, *, output_suffix: str, width: int | None = None, height: int | None = None, keep_aspect: bool = True, rotation: Rotation = "none", flip: Flip = "none", grayscale: bool = False, recursive: bool = False, preserve_structure: bool = True, conflict_policy: ConflictPolicy = "rename", progress: Callable[[int, int], None] | None = None, should_cancel: Callable[[], bool] | None = None) -> BatchTransformResult:
    """Apply one transformation to every top-level image in a folder."""
    source, destination = Path(input_dir), Path(output_dir)
    if source.resolve() == destination.resolve():
        raise ValueError("输出目录不能与输入目录相同，以免修改原图")
    if output_suffix.lower() not in IMAGE_SUFFIXES:
        raise ValueError(f"不支持的输出格式：{output_suffix}")
    written, errors = 0, []
    files = list(iter_image_files(source, recursive=recursive))
    processed = 0
    cancelled = False
    for index, file in enumerate(files, start=1):
        if should_cancel and should_cancel():
            cancelled = True
            break
        processed = index
        image = read_image(file)
        if image is None:
            errors.append(f"无法读取：{file}")
            continue
        try:
            relative_parent = file.relative_to(source).parent if preserve_structure else Path()
            output = resolve_output_path(destination / relative_parent / f"{file.stem}{output_suffix.lower()}", conflict_policy)
            if output is not None:
                write_image(output, transform_image(image, width=width, height=height, keep_aspect=keep_aspect, rotation=rotation, flip=flip, grayscale=grayscale))
                written += 1
        except (ValueError, OSError) as error:
            errors.append(f"{file.name}: {error}")
        if progress:
            progress(index, len(files))
    return BatchTransformResult(processed, written, tuple(errors), cancelled)
