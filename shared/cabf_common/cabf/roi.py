"""Pure CAB-F ROI document semantics.

The coordinates owned by this module are always pixels in the full-size,
calibrated reference image.  Preview canvases and half-size match templates
are adapter concerns and must not change the values stored here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


CABF_ROI_COORDINATE_SPACE = "full-size-calibrated-reference"
Rect = tuple[int, int, int, int]


class RoiCardinalityError(ValueError):
    """Raised when more rectangles are assigned to a single-ROI field."""


class RoiTextPatchError(ValueError):
    """Raised when an identified YAML ROI line cannot be patched safely."""


@dataclass(frozen=True, slots=True)
class RoiField:
    path_parts: tuple[Any, ...]
    display_name: str
    side: str
    is_multi: bool

    @property
    def path_key(self) -> str:
        return ".".join(str(part) for part in self.path_parts)


@dataclass(frozen=True, slots=True)
class RoiIssue:
    field: str
    roi_index: int
    message: str


def _is_number(value: Any, *, allow_boolean: bool = False) -> bool:
    return isinstance(value, (int, float)) and (allow_boolean or not isinstance(value, bool))


def _is_rect_list(value: Any, *, allow_boolean: bool = False) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 4
        and all(_is_number(item, allow_boolean=allow_boolean) for item in value)
    )


def _is_multi_rect_list(value: Any, *, allow_boolean: bool = False) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(_is_rect_list(item, allow_boolean=allow_boolean) for item in value)
    )


def _detect_side(path_parts: tuple[Any, ...], *, case_sensitive: bool = False) -> str:
    names = {str(part) if case_sensitive else str(part).lower() for part in path_parts}
    if "top" in names:
        return "top"
    if "bottom" in names:
        return "bottom"
    return "unknown"


def collect_roi_fields(
    node: Any,
    path_parts: tuple[Any, ...] = (),
    *,
    roi_keys: tuple[str, ...] = ("roi",),
    empty_roi_is_multi: bool = True,
    allow_boolean: bool = False,
    side_case_sensitive: bool = False,
) -> list[RoiField]:
    """Discover ROI fields using an explicit adapter policy."""
    fields: list[RoiField] = []
    lowered_keys = {key.lower() for key in roi_keys}
    if isinstance(node, dict):
        for key, value in node.items():
            next_path = path_parts + (key,)
            key_name = str(key).lower()
            is_empty_multi = (
                empty_roi_is_multi
                and key_name == "roi"
                and isinstance(value, list)
                and not value
            )
            if key_name in lowered_keys and (
                _is_rect_list(value, allow_boolean=allow_boolean)
                or _is_multi_rect_list(value, allow_boolean=allow_boolean)
                or is_empty_multi
            ):
                fields.append(
                    RoiField(
                        path_parts=next_path,
                        display_name=".".join(str(part) for part in next_path),
                        side=_detect_side(next_path, case_sensitive=side_case_sensitive),
                        is_multi=(
                            _is_multi_rect_list(value, allow_boolean=allow_boolean)
                            or is_empty_multi
                        ),
                    )
                )
            fields.extend(
                collect_roi_fields(
                    value,
                    next_path,
                    roi_keys=roi_keys,
                    empty_roi_is_multi=empty_roi_is_multi,
                    allow_boolean=allow_boolean,
                    side_case_sensitive=side_case_sensitive,
                )
            )
    elif isinstance(node, list):
        for index, value in enumerate(node):
            fields.extend(
                collect_roi_fields(
                    value,
                    path_parts + (index,),
                    roi_keys=roi_keys,
                    empty_roi_is_multi=empty_roi_is_multi,
                    allow_boolean=allow_boolean,
                    side_case_sensitive=side_case_sensitive,
                )
            )
    return fields


def normalize_rects(value: Any, *, allow_boolean: bool = False) -> list[Rect]:
    """Return integer rectangle tuples for either supported YAML shape."""
    if _is_rect_list(value, allow_boolean=allow_boolean):
        return [tuple(int(item) for item in value)]
    if _is_multi_rect_list(value, allow_boolean=allow_boolean):
        return [tuple(int(item) for item in rect) for rect in value]
    return []


def get_value(data: Any, path_parts: tuple[Any, ...]) -> Any:
    current = data
    for part in path_parts:
        current = current[part]
    return current


def set_rects(
    data: Any,
    field: RoiField,
    rects: list[Rect],
    *,
    reject_multiple: bool = True,
) -> None:
    """Replace a field without changing the full-size reference coordinates."""
    if not field.is_multi and len(rects) > 1 and reject_multiple:
        raise RoiCardinalityError(f"{field.display_name} is a single-ROI field")
    current = data
    for part in field.path_parts[:-1]:
        current = current[part]
    normalized = [[int(value) for value in rect] for rect in rects]
    if field.is_multi:
        current[field.path_parts[-1]] = normalized
    else:
        current[field.path_parts[-1]] = normalized[0] if normalized else []


def _line_indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _find_yaml_key_line(lines: list[str], path_parts: tuple[Any, ...]) -> int:
    if not path_parts or any(not isinstance(part, str) for part in path_parts):
        return -1
    start = 0
    end = len(lines)
    parent_indent = -1
    for depth, key in enumerate(path_parts):
        found_index = -1
        found_indent = -1
        expected_prefix = f"{key}:"
        for index in range(start, end):
            stripped = lines[index].strip()
            if not stripped or stripped.startswith("#") or not stripped.startswith(expected_prefix):
                continue
            indent = _line_indent(lines[index])
            if indent <= parent_indent:
                continue
            if found_index == -1 or indent < found_indent:
                found_index = index
                found_indent = indent
        if found_index < 0:
            return -1
        if depth == len(path_parts) - 1:
            return found_index
        start = found_index + 1
        end = len(lines)
        for index in range(start, len(lines)):
            stripped = lines[index].strip()
            if stripped and not stripped.startswith("#") and _line_indent(lines[index]) <= found_indent:
                end = index
                break
        parent_indent = found_indent
    return -1


def _format_roi_value(field: RoiField, rects: list[Rect]) -> str:
    if field.is_multi:
        return "[" + ", ".join(
            f"[{x1}, {y1}, {x2}, {y2}]" for x1, y1, x2, y2 in rects
        ) + "]"
    if not rects:
        return "[]"
    return "[" + ", ".join(str(value) for value in rects[0]) + "]"


def _split_line_ending(line: str) -> tuple[str, str]:
    if line.endswith("\r\n"):
        return line[:-2], "\r\n"
    if line.endswith(("\n", "\r")):
        return line[:-1], line[-1]
    return line, ""


def _replace_yaml_value_line(line: str, value: str) -> str:
    body, newline = _split_line_ending(line)
    comment = ""
    if "#" in body:
        body, comment = body.split("#", 1)
        comment = "#" + comment
    prefix, separator, _old_value = body.partition(":")
    if not separator:
        raise RoiTextPatchError(f"cannot update YAML line: {line.rstrip()}")
    replaced = f"{prefix}: {value}"
    if comment:
        replaced += f"  {comment}"
    return replaced + newline


def patch_roi_text(
    text: str,
    data: Any,
    fields: list[RoiField],
    *,
    allow_boolean: bool = False,
) -> str:
    """Patch only discovered ROI scalars, leaving other YAML bytes intact."""
    lines = text.splitlines(keepends=True)
    for field in fields:
        index = _find_yaml_key_line(lines, field.path_parts)
        if index >= 0:
            lines[index] = _replace_yaml_value_line(
                lines[index],
                _format_roi_value(
                    field,
                    normalize_rects(
                        get_value(data, field.path_parts),
                        allow_boolean=allow_boolean,
                    ),
                ),
            )
    return "".join(lines)


class CabfRoiDocument:
    """Mutable functional core for CAB-F ROI values."""

    coordinate_space = CABF_ROI_COORDINATE_SPACE

    def __init__(self, data: dict[str, Any]) -> None:
        self.data = data
        self._fields = collect_roi_fields(data)

    def fields(self, side: str | None = None) -> list[RoiField]:
        if side is None:
            return list(self._fields)
        return [field for field in self._fields if field.side == side]

    def rects(self, field: RoiField) -> list[Rect]:
        return normalize_rects(get_value(self.data, field.path_parts))

    def set_rects(self, field: RoiField, rects: list[Rect]) -> None:
        set_rects(self.data, field, rects)

    def validate(
        self,
        side: str,
        *,
        calibrated_reference_size: tuple[int, int],
    ) -> list[RoiIssue]:
        """Validate stored pixels against the full-size calibrated reference."""
        width, height = calibrated_reference_size
        issues: list[RoiIssue] = []
        for field in self.fields(side):
            for index, (x1, y1, x2, y2) in enumerate(self.rects(field)):
                if x2 <= x1 or y2 <= y1:
                    issues.append(RoiIssue(field.display_name, index, "坐标顺序无效"))
                elif x1 < 0 or y1 < 0 or x2 > width or y2 > height:
                    issues.append(RoiIssue(field.display_name, index, f"超出基准图 {width}×{height}"))
        return issues

    def patch_text(self, original_text: str) -> str:
        return patch_roi_text(original_text, self.data, self._fields)


__all__ = [
    "CABF_ROI_COORDINATE_SPACE",
    "CabfRoiDocument",
    "Rect",
    "RoiCardinalityError",
    "RoiField",
    "RoiIssue",
    "RoiTextPatchError",
    "collect_roi_fields",
    "get_value",
    "normalize_rects",
    "patch_roi_text",
    "set_rects",
]
