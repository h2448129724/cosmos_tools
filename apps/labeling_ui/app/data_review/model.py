"""Project-neutral dataset review model and file operations."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


IMAGE_SUFFIXES = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"})


@dataclass(frozen=True)
class ReviewSpec:
    image_source: Path
    annotation_source: Path | None = None
    accepted_output: Path | None = None
    trash_root: Path | None = None
    require_annotation: bool = False
    annotation_suffix: str = ".json"


@dataclass
class ReviewItem:
    image_path: Path
    annotation_path: Path | None = None

    @property
    def stem(self) -> str:
        return self.image_path.stem

    @property
    def has_annotation(self) -> bool:
        return self.annotation_path is not None

    # Compatibility with the former CAB-F-specific filter item.
    @property
    def label_path(self) -> Path | None:
        return self.annotation_path

    @property
    def has_label(self) -> bool:
        return self.has_annotation


@dataclass(frozen=True)
class ReviewResult:
    source_image_dir: Path
    active_image_dir: Path
    accepted_output: Path | None
    accepted_count: int
    removed_count: int
    remaining_count: int


def collect_review_items(spec: ReviewSpec) -> list[ReviewItem]:
    annotation_root = spec.annotation_source or spec.image_source
    items: list[ReviewItem] = []
    for path in sorted(spec.image_source.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        annotation_path = annotation_root / f"{path.stem}{spec.annotation_suffix}"
        if annotation_path.is_file():
            items.append(ReviewItem(image_path=path, annotation_path=annotation_path))
        elif not spec.require_annotation:
            items.append(ReviewItem(image_path=path))
    return items


def make_review_trash_dir(root: Path, namespace: str = "dataset_review") -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    trash_dir = root / ".trash" / timestamp / namespace
    trash_dir.mkdir(parents=True, exist_ok=True)
    return trash_dir


def _resolve_unique_path(destination: Path, source_name: str) -> Path:
    candidate = destination / source_name
    if not candidate.exists():
        return candidate
    source = Path(source_name)
    index = 1
    while True:
        candidate = destination / f"{source.stem}_{index}{source.suffix}"
        if not candidate.exists():
            return candidate
        index += 1


def move_file_safe(source: Path, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    target = _resolve_unique_path(destination, source.name)
    if target.resolve() == source.resolve():
        target = _resolve_unique_path(destination, f"{source.stem}_moved{source.suffix}")
    shutil.move(str(source), str(target))
    return target


def move_review_item(item: ReviewItem, destination: Path) -> tuple[Path, Path | None]:
    moved_image = move_file_safe(item.image_path, destination)
    moved_annotation = move_file_safe(item.annotation_path, destination) if item.annotation_path else None
    return moved_image, moved_annotation


def build_review_result(
    spec: ReviewSpec,
    *,
    accepted_count: int,
    removed_count: int,
    remaining_count: int,
) -> ReviewResult:
    active_dir = spec.accepted_output if accepted_count > 0 and spec.accepted_output else spec.image_source
    return ReviewResult(
        source_image_dir=spec.image_source,
        active_image_dir=active_dir,
        accepted_output=spec.accepted_output,
        accepted_count=accepted_count,
        removed_count=removed_count,
        remaining_count=remaining_count,
    )
