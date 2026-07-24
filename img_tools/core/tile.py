"""Deterministic fixed-size image tiling."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

import cv2

from .image_io import iter_image_files, read_image, write_image
from .output import ConflictPolicy, resolve_output_path

EdgePolicy = Literal["shift", "discard", "partial", "pad"]


@dataclass(frozen=True, slots=True)
class Tile:
    x: int
    y: int
    width: int
    height: int
    row: int
    column: int


@dataclass(frozen=True, slots=True)
class TileResult:
    processed_files: int
    written_files: int
    errors: tuple[str, ...]
    cancelled: bool = False


def compute_tiles(image_width: int, image_height: int, tile_width: int, tile_height: int, *, overlap_percent: int = 0, edge_policy: EdgePolicy = "shift") -> list[Tile]:
    """Compute tiles that cover an image according to the selected edge policy."""
    if min(image_width, image_height, tile_width, tile_height) <= 0:
        raise ValueError("图片和切片的宽高必须大于 0")
    if not 0 <= overlap_percent < 100:
        raise ValueError("重叠率必须在 0 到 99 之间")
    step_x = max(1, round(tile_width * (100 - overlap_percent) / 100))
    step_y = max(1, round(tile_height * (100 - overlap_percent) / 100))
    xs = _axis_positions(image_width, tile_width, step_x, edge_policy)
    ys = _axis_positions(image_height, tile_height, step_y, edge_policy)
    tiles: list[Tile] = []
    for row, y in enumerate(ys):
        for column, x in enumerate(xs):
            width = min(tile_width, image_width - x)
            height = min(tile_height, image_height - y)
            if edge_policy in {"shift", "discard"} and (width < tile_width or height < tile_height):
                continue
            tiles.append(Tile(x, y, width, height, row, column))
    return tiles


def _axis_positions(length: int, tile_length: int, step: int, policy: EdgePolicy) -> list[int]:
    if length < tile_length:
        return [0] if policy in {"partial", "pad"} else []
    positions = list(range(0, length, step))
    if policy == "shift" and positions and positions[-1] + tile_length > length:
        positions[-1] = length - tile_length
    return list(dict.fromkeys(positions))


def batch_tile(
    input_dir: str | Path,
    output_dir: str | Path,
    tile_width: int,
    tile_height: int,
    *,
    overlap_percent: int = 0,
    edge_policy: EdgePolicy = "shift",
    recursive: bool = False,
    preserve_structure: bool = True,
    conflict_policy: ConflictPolicy = "rename",
    progress: Callable[[int, int], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> TileResult:
    """Tile every image in a directory; output names include their grid position."""
    source_root, destination_root = Path(input_dir), Path(output_dir)
    if source_root.resolve() == destination_root.resolve():
        raise ValueError("输出目录不能与输入目录相同，以免修改原图")
    files = list(iter_image_files(source_root, recursive=recursive))
    written, errors = 0, []
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
            height, width = image.shape[:2]
            try:
                for tile in compute_tiles(width, height, tile_width, tile_height, overlap_percent=overlap_percent, edge_policy=edge_policy):
                    piece = image[tile.y:tile.y + tile.height, tile.x:tile.x + tile.width]
                    if edge_policy == "pad" and (tile.width < tile_width or tile.height < tile_height):
                        piece = cv2.copyMakeBorder(piece, 0, tile_height - tile.height, 0, tile_width - tile.width, cv2.BORDER_CONSTANT)
                    relative_parent = source.relative_to(source_root).parent if preserve_structure else Path()
                    output = resolve_output_path(destination_root / relative_parent / f"{source.stem}_tile_{tile.row}_{tile.column}{source.suffix.lower()}", conflict_policy)
                    if output is not None:
                        write_image(output, piece)
                        written += 1
            except (ValueError, OSError) as error:
                errors.append(f"{source.name}: {error}")
        if progress:
            progress(index, len(files))
    return TileResult(processed, written, tuple(errors), cancelled)
