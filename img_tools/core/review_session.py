"""Pure review-session transitions and filesystem move intents.

Adapters supply discovered review items and execute the returned intents.
This module never inspects paths, resolves collisions, moves files, or reads
the clock.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal


ReviewDisposition = Literal["accepted", "removed"]
ConflictPolicy = Literal["rename", "skip", "overwrite"]


@dataclass(frozen=True, slots=True)
class ReviewItem:
    image_path: Path
    annotation_path: Path | None = None

    @property
    def stem(self) -> str:
        return self.image_path.stem

    @property
    def has_annotation(self) -> bool:
        return self.annotation_path is not None

    @property
    def label_path(self) -> Path | None:
        """Compatibility name used by the CAB-F review adapter."""

        return self.annotation_path

    @property
    def has_label(self) -> bool:
        """Compatibility name used by the CAB-F review adapter."""

        return self.has_annotation


@dataclass(frozen=True, slots=True)
class ReviewMoveIntent:
    """A requested move whose collision handling belongs to a shell."""

    source: Path
    destination_dir: Path
    preferred_name: str
    decision: str
    conflict_policy: ConflictPolicy


@dataclass(frozen=True, slots=True)
class ReviewSession:
    """Immutable ordering, selection and outcome counts for one review."""

    items: tuple[ReviewItem, ...] = ()
    current_index: int = -1
    accepted_count: int = 0
    removed_count: int = 0

    @property
    def current_item(self) -> ReviewItem | None:
        if 0 <= self.current_index < len(self.items):
            return self.items[self.current_index]
        return None


@dataclass(frozen=True, slots=True)
class ReviewTransition:
    """State to commit after all move intents execute successfully."""

    session: ReviewSession
    item: ReviewItem
    removed_index: int
    intents: tuple[ReviewMoveIntent, ...]


def begin_review(items: Iterable[ReviewItem]) -> ReviewSession:
    return ReviewSession(items=tuple(items))


def select_review_index(session: ReviewSession, index: int) -> ReviewSession:
    if not session.items:
        return ReviewSession(
            items=session.items,
            current_index=-1,
            accepted_count=session.accepted_count,
            removed_count=session.removed_count,
        )
    selected = max(0, min(int(index), len(session.items) - 1))
    return ReviewSession(
        items=session.items,
        current_index=selected,
        accepted_count=session.accepted_count,
        removed_count=session.removed_count,
    )


def clear_review_selection(session: ReviewSession) -> ReviewSession:
    return ReviewSession(
        items=session.items,
        current_index=-1,
        accepted_count=session.accepted_count,
        removed_count=session.removed_count,
    )


def build_review_move_intents(
    item: ReviewItem,
    destination_dir: Path,
    *,
    decision: str,
    conflict_policy: ConflictPolicy = "rename",
) -> tuple[ReviewMoveIntent, ...]:
    sources = (item.image_path,) + (
        (item.annotation_path,) if item.annotation_path is not None else ()
    )
    return tuple(
        ReviewMoveIntent(
            source=source,
            destination_dir=destination_dir,
            preferred_name=source.name,
            decision=decision,
            conflict_policy=conflict_policy,
        )
        for source in sources
    )


def plan_review_decision(
    session: ReviewSession,
    disposition: ReviewDisposition,
    destination_dir: Path,
    *,
    decision: str,
    conflict_policy: ConflictPolicy = "rename",
) -> ReviewTransition | None:
    """Plan removal of the selected item and selection of its successor."""

    item = session.current_item
    if item is None:
        return None
    if disposition not in {"accepted", "removed"}:
        raise ValueError(f"未知的审阅处置：{disposition}")

    removed_index = session.current_index
    remaining = session.items[:removed_index] + session.items[removed_index + 1 :]
    next_index = min(removed_index, len(remaining) - 1) if remaining else -1
    next_session = ReviewSession(
        items=remaining,
        current_index=next_index,
        accepted_count=session.accepted_count + (disposition == "accepted"),
        removed_count=session.removed_count + (disposition == "removed"),
    )
    return ReviewTransition(
        session=next_session,
        item=item,
        removed_index=removed_index,
        intents=build_review_move_intents(
            item,
            destination_dir,
            decision=decision,
            conflict_policy=conflict_policy,
        ),
    )


def plan_standalone_review_move(
    source: Path,
    destination_dir: Path,
    decision: str,
    *,
    conflict_policy: ConflictPolicy = "rename",
) -> ReviewMoveIntent:
    return ReviewMoveIntent(
        source=source,
        destination_dir=destination_dir,
        preferred_name=source.name,
        decision=decision,
        conflict_policy=conflict_policy,
    )


def plan_review_undo(
    *,
    original_source: Path,
    moved_target: Path,
    decision: str,
) -> ReviewMoveIntent:
    return ReviewMoveIntent(
        source=moved_target,
        destination_dir=original_source.parent,
        preferred_name=original_source.name,
        decision=f"undo:{decision}",
        conflict_policy="rename",
    )


__all__ = [
    "ConflictPolicy",
    "ReviewDisposition",
    "ReviewItem",
    "ReviewMoveIntent",
    "ReviewSession",
    "ReviewTransition",
    "begin_review",
    "build_review_move_intents",
    "clear_review_selection",
    "plan_review_decision",
    "plan_review_undo",
    "plan_standalone_review_move",
    "select_review_index",
]
