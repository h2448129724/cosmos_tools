"""Reversible manual review file moves."""
from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from .output import resolve_output_path
from .review_session import (
    ConflictPolicy,
    ReviewMoveIntent,
    plan_review_undo,
    plan_standalone_review_move,
)


@dataclass(frozen=True, slots=True)
class ReviewMove:
    source: Path
    target: Path
    decision: str


def move_for_review(source: str | Path, destination_dir: str | Path, decision: str, *, conflict_policy: ConflictPolicy = "rename") -> ReviewMove | None:
    path = Path(source)
    intent = plan_standalone_review_move(
        path,
        Path(destination_dir),
        decision,
        conflict_policy=conflict_policy,
    )
    target = _execute_move_intent(intent)
    if target is None:
        return None
    return ReviewMove(path, target, decision)


def undo_review_move(move: ReviewMove) -> Path:
    """Restore a reviewed file to its original directory without overwriting it."""
    intent = plan_review_undo(
        original_source=move.source,
        moved_target=move.target,
        decision=move.decision,
    )
    target = _execute_move_intent(intent)
    assert target is not None
    return target


def _execute_move_intent(intent: ReviewMoveIntent) -> Path | None:
    target = resolve_output_path(
        intent.destination_dir / intent.preferred_name,
        intent.conflict_policy,
    )
    if target is None:
        return None
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(intent.source), str(target))
    return target
