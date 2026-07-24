"""Portable multi-step image-processing pipelines."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable
from dataclasses import dataclass

import numpy as np

from .enhance import enhance_image
from .transform import transform_image
from .image_io import IMAGE_SUFFIXES, iter_image_files, read_image, write_image
from .output import ConflictPolicy, resolve_output_path


@dataclass(frozen=True, slots=True)
class BatchPipelineResult:
    processed_files: int
    written_files: int
    errors: tuple[str, ...]
    cancelled: bool = False


def run_pipeline(image: np.ndarray, steps: list[dict[str, Any]]) -> np.ndarray:
    """Apply a validated ordered sequence of transform/enhance steps."""
    result = image
    for index, step in enumerate(steps, start=1):
        operation = step.get("operation")
        options = step.get("options", {})
        if not isinstance(options, dict):
            raise ValueError(f"第 {index} 步 options 必须是对象")
        if operation == "transform":
            result = transform_image(result, **options)
        elif operation == "enhance":
            result = enhance_image(result, **options)
        else:
            raise ValueError(f"第 {index} 步包含未知操作：{operation}")
    return result


def load_pipeline(path: str | Path) -> list[dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("version") != 1 or not isinstance(data.get("steps"), list):
        raise ValueError("不是有效的处理流水线文件")
    return data["steps"]


def save_pipeline(path: str | Path, steps: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"version": 1, "steps": steps}, ensure_ascii=False, indent=2), encoding="utf-8")


def batch_pipeline(input_dir: str | Path, output_dir: str | Path, steps: list[dict[str, Any]], *, output_suffix: str = ".png", recursive: bool = False, preserve_structure: bool = True, conflict_policy: ConflictPolicy = "rename", progress: Callable[[int, int], None] | None = None, should_cancel: Callable[[], bool] | None = None) -> BatchPipelineResult:
    """Run a saved pipeline over every image in a directory."""
    source, destination = Path(input_dir), Path(output_dir)
    if source.resolve() == destination.resolve():
        raise ValueError("输出目录不能与输入目录相同")
    if output_suffix.lower() not in IMAGE_SUFFIXES:
        raise ValueError(f"不支持的输出格式：{output_suffix}")
    files = list(iter_image_files(source, recursive=recursive))
    processed = written = 0
    errors: list[str] = []
    for index, file in enumerate(files, start=1):
        if should_cancel and should_cancel():
            return BatchPipelineResult(processed, written, tuple(errors), True)
        processed = index
        image = read_image(file)
        if image is None:
            errors.append(f"无法读取：{file}")
        else:
            try:
                relative = file.relative_to(source).parent if preserve_structure else Path()
                target = resolve_output_path(destination / relative / f"{file.stem}{output_suffix.lower()}", conflict_policy)
                if target is not None:
                    write_image(target, run_pipeline(image, steps))
                    written += 1
            except (OSError, ValueError, TypeError) as error:
                errors.append(f"{file.name}: {error}")
        if progress:
            progress(index, len(files))
    return BatchPipelineResult(processed, written, tuple(errors))
