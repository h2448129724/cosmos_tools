"""Pure point/edge annotation edit commands.

The Qt canvases own interaction state (selection, pending edge starts and
signals), while this module owns the deterministic state transitions.  Every
command returns a new list and leaves its input lists and dictionaries alone.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class AnnotationState:
    """Point/edge data owned by the functional core.

    Construction and every transition deep-copy dictionaries so a caller can
    safely retain and reuse the lists it supplied to the editor.
    """

    points: list[dict[str, Any]] = field(default_factory=list)
    edges: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        object.__setattr__(self, "points", copy.deepcopy(list(self.points)))
        object.__setattr__(self, "edges", copy.deepcopy(list(self.edges)))


@dataclass(frozen=True, slots=True)
class AddPoint:
    x: float
    y: float
    score: float = 1.0
    source: str = "manual"


@dataclass(frozen=True, slots=True)
class MovePoint:
    point_id: int
    x: float
    y: float
    source: str = "manual"


@dataclass(frozen=True, slots=True)
class DeletePoint:
    point_id: int


@dataclass(frozen=True, slots=True)
class ToggleEdge:
    src: int
    dst: int
    label: int = 1
    source: str = "manual"


@dataclass(frozen=True, slots=True)
class ClearEdges:
    pass


@dataclass(frozen=True, slots=True)
class UndoLastEdge:
    pass


EditCommand = AddPoint | MovePoint | DeletePoint | ToggleEdge | ClearEdges | UndoLastEdge


@dataclass(frozen=True, slots=True)
class EditTransition:
    next_state: AnnotationState
    changed: bool
    action: str
    removed_points: list[dict[str, Any]] = field(default_factory=list)
    removed_edges: list[dict[str, Any]] = field(default_factory=list)


def next_point_id(points: Sequence[dict[str, Any]]) -> int:
    """Return the next point id using the editor's max-id-plus-one rule."""

    if not points:
        return 0
    return max(int(point["id"]) for point in points) + 1


def add_point(
    points: Sequence[dict[str, Any]],
    x: float,
    y: float,
    *,
    score: float = 1.0,
    source: str = "manual",
) -> list[dict[str, Any]]:
    """Append a point using the schema and defaults used by the Qt editor."""

    return apply_edit(
        AnnotationState(points=list(points)), AddPoint(x=x, y=y, score=score, source=source)
    ).next_state.points


def move_point(
    points: Sequence[dict[str, Any]],
    point_id: int,
    x: float,
    y: float,
    *,
    source: str = "manual",
) -> list[dict[str, Any]]:
    """Move one point, preserving order and all unrelated point fields."""

    return apply_edit(
        AnnotationState(points=list(points)), MovePoint(point_id=point_id, x=x, y=y, source=source)
    ).next_state.points


def delete_point(points: Sequence[dict[str, Any]], point_id: int) -> list[dict[str, Any]]:
    """Delete a point while preserving the order of all remaining points."""

    return apply_edit(AnnotationState(points=list(points)), DeletePoint(point_id)).next_state.points


def delete_point_and_incident_edges(
    points: Sequence[dict[str, Any]],
    edges: Sequence[dict[str, Any]],
    point_id: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Delete a point and every edge whose source or destination is that point."""

    transition = apply_edit(
        AnnotationState(points=list(points), edges=list(edges)), DeletePoint(point_id)
    )
    return transition.next_state.points, transition.next_state.edges


# The shorter name reads naturally at call sites and keeps compatibility with
# callers that describe this operation as removing a point and its edges.
remove_point_and_edges = delete_point_and_incident_edges


def _edge_key(src: int, dst: int) -> tuple[int, int]:
    return tuple(sorted((int(src), int(dst))))


def has_edge(edges: Sequence[dict[str, Any]], src: int, dst: int) -> bool:
    """Return whether an undirected edge exists, irrespective of endpoint order."""

    key = _edge_key(src, dst)
    return any(_edge_key(int(edge["src"]), int(edge["dst"])) == key for edge in edges)


def toggle_edge(
    edges: Sequence[dict[str, Any]],
    src: int,
    dst: int,
    *,
    label: int = 1,
    source: str = "manual",
) -> list[dict[str, Any]]:
    """Toggle a normalized undirected edge, preserving existing edge order.

    A new edge id intentionally uses ``len(edges) + 1`` to match the existing
    editor behavior, including its numbering after an earlier deletion.
    """

    return apply_edit(
        AnnotationState(edges=list(edges)), ToggleEdge(src=src, dst=dst, label=label, source=source)
    ).next_state.edges


def clear_edges(edges: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove all edges and return a fresh list."""

    return apply_edit(AnnotationState(edges=list(edges)), ClearEdges()).next_state.edges


def undo_last_edge(edges: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop the last edge, or return an equivalent copy when there are none."""

    return apply_edit(AnnotationState(edges=list(edges)), UndoLastEdge()).next_state.edges


def apply_edit(state: AnnotationState, command: EditCommand) -> EditTransition:
    """Apply one typed edit and return the complete deterministic transition."""

    points = copy.deepcopy(state.points)
    edges = copy.deepcopy(state.edges)
    removed_points: list[dict[str, Any]] = []
    removed_edges: list[dict[str, Any]] = []
    action = "noop"

    if isinstance(command, AddPoint):
        points.append(
            {
                "id": next_point_id(points),
                "x": float(command.x),
                "y": float(command.y),
                "score": float(command.score),
                "source": command.source,
            }
        )
        action = "add_point"
    elif isinstance(command, MovePoint):
        target_id = int(command.point_id)
        for point in points:
            if int(point["id"]) == target_id:
                point["x"] = float(command.x)
                point["y"] = float(command.y)
                point["source"] = command.source
                action = "move_point"
                break
    elif isinstance(command, DeletePoint):
        target_id = int(command.point_id)
        removed_points = [copy.deepcopy(point) for point in points if int(point["id"]) == target_id]
        removed_edges = [
            copy.deepcopy(edge)
            for edge in edges
            if int(edge["src"]) == target_id or int(edge["dst"]) == target_id
        ]
        points = [point for point in points if int(point["id"]) != target_id]
        edges = [
            edge
            for edge in edges
            if int(edge["src"]) != target_id and int(edge["dst"]) != target_id
        ]
        action = "delete_point"
    elif isinstance(command, ToggleEdge):
        key = _edge_key(command.src, command.dst)
        for index, edge in enumerate(edges):
            if _edge_key(int(edge["src"]), int(edge["dst"])) == key:
                removed_edges = [copy.deepcopy(edges.pop(index))]
                action = "remove_edge"
                break
        else:
            edges.append(
                {
                    "edge_id": f"edge_{len(edges) + 1:04d}",
                    "src": key[0],
                    "dst": key[1],
                    "label": int(command.label),
                    "source": command.source,
                }
            )
            action = "add_edge"
    elif isinstance(command, ClearEdges):
        removed_edges = copy.deepcopy(edges)
        edges = []
        action = "clear_edges"
    elif isinstance(command, UndoLastEdge):
        if edges:
            removed_edges = [copy.deepcopy(edges[-1])]
            edges = edges[:-1]
            action = "undo_last_edge"

    next_state = AnnotationState(points=points, edges=edges)
    changed = next_state.points != state.points or next_state.edges != state.edges
    return EditTransition(
        next_state=next_state,
        changed=changed,
        action=action,
        removed_points=removed_points,
        removed_edges=removed_edges,
    )


__all__ = [
    "AddPoint",
    "AnnotationState",
    "ClearEdges",
    "DeletePoint",
    "EditCommand",
    "EditTransition",
    "MovePoint",
    "ToggleEdge",
    "UndoLastEdge",
    "add_point",
    "apply_edit",
    "clear_edges",
    "delete_point",
    "delete_point_and_incident_edges",
    "has_edge",
    "move_point",
    "next_point_id",
    "remove_point_and_edges",
    "toggle_edge",
    "undo_last_edge",
]
