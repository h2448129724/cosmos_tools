"""Shared desktop UI foundation for the Cosmos workbench and workspaces.

The public interface is intentionally small.  Pages describe their business
layout with primitives; this package owns visual tokens, widget roles,
responsive action layout, and common presentation behaviour.
"""

from .primitives import (
    ActionBar,
    CollapsibleLogPanel,
    EmptyState,
    MetricGrid,
    PageHeader,
    PageScaffold,
    PathField,
    SectionSurface,
    StatusBanner,
    set_tone,
    set_ui_role,
)
from .responsive import LayoutDensity, LayoutPresentation, layout_presentation
from .theme import (
    DEFAULT_TOKENS,
    LEGACY_TOKENS,
    StatusTone,
    ThemeTokens,
    build_labeling_stylesheet,
    build_toolbox_stylesheet,
    build_trainer_stylesheet,
    build_workspace_stylesheet,
    status_badge_stylesheet,
)

__all__ = [
    "ActionBar",
    "CollapsibleLogPanel",
    "DEFAULT_TOKENS",
    "EmptyState",
    "LEGACY_TOKENS",
    "LayoutDensity",
    "LayoutPresentation",
    "MetricGrid",
    "PageHeader",
    "PageScaffold",
    "PathField",
    "SectionSurface",
    "StatusBanner",
    "StatusTone",
    "ThemeTokens",
    "build_labeling_stylesheet",
    "build_toolbox_stylesheet",
    "build_trainer_stylesheet",
    "build_workspace_stylesheet",
    "layout_presentation",
    "set_tone",
    "set_ui_role",
    "status_badge_stylesheet",
]
