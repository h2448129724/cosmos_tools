"""Reversible manual review file moves."""
from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from .output import ConflictPolicy, resolve_output_path


@dataclass(frozen=True, slots=True)
class ReviewMove:
    source: Path
    target: Path
    decision: str


def move_for_review(source: str | Path, destination_dir: str | Path, decision: str, *, conflict_policy: ConflictPolicy = "rename") -> ReviewMove | None:
    path = Path(source)
    target = resolve_output_path(Path(destination_dir) / path.name, conflict_policy)
    if target is None:
        return None
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(path), str(target))
    return ReviewMove(path, target, decision)


def undo_review_move(move: ReviewMove) -> Path:
    """Restore a reviewed file to its original directory without overwriting it."""
    target = resolve_output_path(move.source, "rename")
    assert target is not None
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(move.target), str(target))
    return target
