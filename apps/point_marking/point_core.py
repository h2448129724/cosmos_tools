"""Ordered image point marking rules and the exported two-level coordinate list.

The core owns the ordered point list, the projection into the exported
payload, and the text encoding of that payload.  Qt widgets, OpenCV decoding,
and filesystem writes stay in the shell that adapts this module to a page.

Coordinates are always expressed in source-image pixels.  A shell that draws
a downscaled preview converts between display pixels and source pixels before
calling into this module.
"""

from __future__ import annotations

import json
from dataclasses import dataclass


Point = tuple[int, int]

# The exported document is a two-level list: one entry per marked point, each
# entry holding the x and y coordinate in that order.
EXPORT_FORMAT = "json"


@dataclass(frozen=True, slots=True)
class MarkingSnapshot:
    """Immutable projection of ordered points and their export payload."""

    points: tuple[Point, ...]
    payload: tuple[tuple[int, int], ...]
    text: str

    @property
    def count(self) -> int:
        return len(self.points)


def normalize_point(x: float, y: float) -> Point:
    """Round a floating coordinate into an integer pixel coordinate."""

    return int(round(x)), int(round(y))


def clamp_point(point: Point, width: int, height: int) -> Point:
    """Keep a point inside the source image bounds."""

    if width <= 0 or height <= 0:
        return point
    x = max(0, min(int(point[0]), width - 1))
    y = max(0, min(int(point[1]), height - 1))
    return x, y


def append_point(points: tuple[Point, ...], x: float, y: float) -> tuple[Point, ...]:
    """Append a point at the end of the ordered list."""

    return (*points, normalize_point(x, y))


def insert_point(
    points: tuple[Point, ...],
    index: int,
    x: float,
    y: float,
) -> tuple[Point, ...]:
    """Insert a point at ``index``; an out-of-range index appends."""

    position = max(0, min(int(index), len(points)))
    return (*points[:position], normalize_point(x, y), *points[position:])


def remove_point(points: tuple[Point, ...], index: int) -> tuple[Point, ...]:
    """Remove one point; an out-of-range index keeps the list unchanged."""

    if not 0 <= index < len(points):
        return points
    return (*points[:index], *points[index + 1:])


def move_point(
    points: tuple[Point, ...],
    index: int,
    x: float,
    y: float,
) -> tuple[Point, ...]:
    """Move one point to a new coordinate; an out-of-range index is unchanged."""

    if not 0 <= index < len(points):
        return points
    updated = list(points)
    updated[index] = normalize_point(x, y)
    return tuple(updated)


def build_payload(points: tuple[Point, ...]) -> tuple[tuple[int, int], ...]:
    """Project ordered points into the exported two-level list."""

    return tuple((int(x), int(y)) for x, y in points)


def render_export_text(payload: tuple[tuple[int, int], ...]) -> str:
    """Encode the payload as the exported JSON two-level list.

    Every point stays on one line so the document keeps the shape readers
    expect: a list whose entries are ``[x, y]`` pairs.
    """

    if not payload:
        return "[]"
    lines = ["["]
    last = len(payload) - 1
    for index, point in enumerate(payload):
        suffix = "," if index < last else ""
        lines.append(f"  [{int(point[0])}, {int(point[1])}]{suffix}")
    lines.append("]")
    return "\n".join(lines)


def parse_export_text(text: str) -> tuple[tuple[Point, ...], str]:
    """Decode an exported two-level list.

    Returns ``(points, error)``.  A successful decode carries an empty error
    message; a failed decode carries the reason and an empty point list.
    """

    raw = (text or "").strip()
    if not raw:
        return (), "内容为空"
    try:
        document = json.loads(raw)
    except json.JSONDecodeError as exc:
        return (), f"JSON 解析失败：{exc.msg}（第 {exc.lineno} 行）"
    if not isinstance(document, list):
        return (), "顶层必须是一个列表"
    points: list[Point] = []
    for position, entry in enumerate(document, start=1):
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            return (), f"第 {position} 项必须是 [x, y]"
        x, y = entry
        if isinstance(x, bool) or isinstance(y, bool):
            return (), f"第 {position} 项必须是数字"
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            return (), f"第 {position} 项必须是数字"
        points.append(normalize_point(x, y))
    return tuple(points), ""


def build_snapshot(points: tuple[Point, ...]) -> MarkingSnapshot:
    """Return the points together with their payload and exported text."""

    payload = build_payload(points)
    return MarkingSnapshot(points=points, payload=payload, text=render_export_text(payload))


def point_label(index: int) -> str:
    """Return the 1-based label shown next to a marked point."""

    return str(int(index) + 1)


def describe_points(points: tuple[Point, ...]) -> str:
    """Return a short human readable summary of the ordered points."""

    if not points:
        return "尚未标注任何点"
    first_x, first_y = points[0]
    last_x, last_y = points[-1]
    return f"共 {len(points)} 个点，首点 ({first_x}, {first_y})，末点 ({last_x}, {last_y})"
