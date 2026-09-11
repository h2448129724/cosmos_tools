"""Project-neutral dataset review model and file operations."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from img_tools.core.review_session import (
    ReviewItem,
    ReviewMoveIntent,
    build_review_move_intents,
)


IMAGE_SUFFIXES = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"})


@dataclass(frozen=True)
class ReviewSpec:
    image_source: Path
    annotation_source: Path | None = None
    accepted_output: Path | None = None
    trash_root: Path | None = None
    require_annotation: bool = False
    annotation_suffix: str = ".json"


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


def _resolve_unique_path(
    destination: Path,
    source_name: str,
    *,
    reserved: set[Path] | None = None,
) -> Path:
    reserved = reserved or set()
    candidate = destination / source_name
    if not candidate.exists() and candidate.resolve() not in reserved:
        return candidate
    source = Path(source_name)
    index = 1
    while True:
        candidate = destination / f"{source.stem}_{index}{source.suffix}"
        if not candidate.exists() and candidate.resolve() not in reserved:
            return candidate
        index += 1


def move_file_safe(
    source: Path,
    destination: Path,
    *,
    preferred_name: str | None = None,
) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    target = _resolve_unique_path(destination, preferred_name or source.name)
    if target.resolve() == source.resolve():
        target = _resolve_unique_path(destination, f"{source.stem}_moved{source.suffix}")
    shutil.move(str(source), str(target))
    return target


def execute_review_intents(intents: tuple[ReviewMoveIntent, ...]) -> tuple[Path, ...]:
    targets: list[tuple[ReviewMoveIntent, Path]] = []
    reserved: set[Path] = set()
    for intent in intents:
        if intent.conflict_policy != "rename":
            raise ValueError(
                f"数据审阅 adapter 不支持重名策略：{intent.conflict_policy}"
            )
        target = _resolve_unique_path(
            intent.destination_dir,
            intent.preferred_name,
            reserved=reserved,
        )
        if target.resolve() == intent.source.resolve():
            target = _resolve_unique_path(
                intent.destination_dir,
                f"{intent.source.stem}_moved{intent.source.suffix}",
                reserved=reserved,
            )
        reserved.add(target.resolve())
        targets.append((intent, target))

    for _intent, target in targets:
        target.parent.mkdir(parents=True, exist_ok=True)

    completed: list[tuple[Path, Path]] = []
    try:
        for intent, target in targets:
            shutil.move(str(intent.source), str(target))
            completed.append((intent.source, target))
    except Exception as exc:
        rollback_errors: list[str] = []
        for source, target in reversed(completed):
            try:
                source.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(target), str(source))
            except Exception as rollback_exc:  # Preserve every recovery failure for the UI.
                rollback_errors.append(f"{target} -> {source}: {rollback_exc}")
        if rollback_errors:
            raise RuntimeError(
                f"审阅文件移动失败且回滚不完整: {exc}; " + "; ".join(rollback_errors)
            ) from exc
        raise
    return tuple(target for _intent, target in targets)


def move_review_item(item: ReviewItem, destination: Path) -> tuple[Path, Path | None]:
    moved = execute_review_intents(
        build_review_move_intents(
            item,
            destination,
            decision="review",
            conflict_policy="rename",
        )
    )
    moved_image = moved[0]
    moved_annotation = moved[1] if len(moved) > 1 else None
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
