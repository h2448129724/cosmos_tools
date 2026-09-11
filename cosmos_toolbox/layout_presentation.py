"""Pure viewport-density and pane presentation rules.

This module sits outside the Qt-backed :mod:`cosmos_toolbox.ui` package so the
workbench reducer can be imported without initializing any desktop toolkit.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class LayoutDensity(StrEnum):
    COMPACT = "compact"
    REGULAR = "regular"
    SPACIOUS = "spacious"


@dataclass(frozen=True, slots=True)
class LayoutPresentation:
    density: LayoutDensity
    navigation_compact: bool
    inspector_overlay: bool
    page_margin: int
    page_spacing: int


def layout_presentation(viewport_width: int) -> LayoutPresentation:
    """Project a host width into one consistent desktop presentation."""

    width = max(0, int(viewport_width))
    if width < 1180:
        return LayoutPresentation(LayoutDensity.COMPACT, True, True, 12, 8)
    if width < 1480:
        return LayoutPresentation(LayoutDensity.REGULAR, False, width < 1320, 16, 10)
    return LayoutPresentation(LayoutDensity.SPACIOUS, False, False, 20, 12)


__all__ = ["LayoutDensity", "LayoutPresentation", "layout_presentation"]
