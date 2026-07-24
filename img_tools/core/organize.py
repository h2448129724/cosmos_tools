"""Safe file classification and validation operations."""
from __future__ import annotations

import shutil
import hashlib
import cv2
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .image_io import iter_image_files, read_image
from .output import ConflictPolicy, resolve_output_path

TransferMode = Literal["copy", "move"]


@dataclass(frozen=True, slots=True)
class OrganizeResult:
    processed_files: int
    changed_files: int
    unmatched_files: int
    errors: tuple[str, ...]


def classify_by_keywords(input_dir: str | Path, output_dir: str | Path, keywords: list[str], *, mode: TransferMode = "copy", conflict_policy: ConflictPolicy = "rename") -> OrganizeResult:
    """Classify images by the first matching filename keyword."""
    source, destination = Path(input_dir), Path(output_dir)
    if source.resolve() == destination.resolve():
        raise ValueError("输出目录不能与输入目录相同")
    normalized = [keyword.strip() for keyword in keywords if keyword.strip()]
    if not normalized:
        raise ValueError("请至少提供一个关键字")
    changed = unmatched = 0
    errors: list[str] = []
    files = list(iter_image_files(source))
    for file in files:
        matched = next((keyword for keyword in normalized if keyword.lower() in file.name.lower()), None)
        if not matched:
            unmatched += 1
            continue
        target = resolve_output_path(destination / matched / file.name, conflict_policy)
        if target is None:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            if mode == "move":
                shutil.move(str(file), str(target))
            else:
                shutil.copy2(file, target)
            changed += 1
        except OSError as error:
            errors.append(f"{file.name}: {error}")
    return OrganizeResult(len(files), changed, unmatched, tuple(errors))


def find_invalid_images(input_dir: str | Path, *, recursive: bool = False) -> list[Path]:
    """Return files that have an image suffix but cannot be decoded."""
    return [path for path in iter_image_files(input_dir, recursive=recursive) if read_image(path) is None]


def find_exact_duplicates(input_dir: str | Path, *, recursive: bool = False) -> list[list[Path]]:
    """Group byte-identical images by SHA-256 without modifying any file."""
    groups: dict[str, list[Path]] = {}
    for path in iter_image_files(input_dir, recursive=recursive):
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        groups.setdefault(digest.hexdigest(), []).append(path)
    return [paths for paths in groups.values() if len(paths) > 1]


def find_similar_images(input_dir: str | Path, *, max_distance: int = 4, recursive: bool = False) -> list[list[Path]]:
    """Group visually similar images with an 8×8 average hash (read-only)."""
    items: list[tuple[Path, int]] = []
    for path in iter_image_files(input_dir, recursive=recursive):
        image = read_image(path)
        if image is None:
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY if image.ndim == 3 and image.shape[2] == 4 else cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        small = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA)
        bits = small >= small.mean()
        items.append((path, sum(int(bit) << index for index, bit in enumerate(bits.ravel()))))
    groups: list[list[Path]] = []
    consumed: set[Path] = set()
    for path, value in items:
        if path in consumed:
            continue
        group = [other for other, other_value in items if (value ^ other_value).bit_count() <= max_distance]
        if len(group) > 1:
            groups.append(group)
            consumed.update(group)
    return groups


def batch_rename(input_dir: str | Path, prefix: str, *, start: int = 1, digits: int = 4, recursive: bool = False) -> list[tuple[Path, Path]]:
    """Rename image files deterministically, preserving their individual suffixes."""
    if not prefix.strip():
        raise ValueError("前缀不能为空")
    files = list(iter_image_files(input_dir, recursive=recursive))
    targets = [(path, path.with_name(f"{prefix}_{start + index:0{digits}d}{path.suffix.lower()}")) for index, path in enumerate(files)]
    if len({target for _, target in targets}) != len(targets):
        raise ValueError("重命名目标存在冲突")
    temporary = [(source, source.with_name(f".__img_tools_tmp_{index}{source.suffix}"), target) for index, (source, target) in enumerate(targets)]
    for source, temp, _ in temporary:
        source.rename(temp)
    for _, temp, target in temporary:
        temp.rename(target)
    return [(source, target) for source, _, target in temporary]
