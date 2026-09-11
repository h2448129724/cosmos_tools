"""Trainer compatibility interface backed by the shared Cosmos UI theme."""

from __future__ import annotations

from cosmos_toolbox.ui.theme import (
    DEFAULT_TOKENS,
    StatusTone,
    build_trainer_stylesheet,
    status_badge_stylesheet as _shared_status_badge_stylesheet,
)

BG_WINDOW = DEFAULT_TOKENS.surface
BG_CARD = DEFAULT_TOKENS.surface
BG_HOVER = DEFAULT_TOKENS.hover
BG_SELECTED = DEFAULT_TOKENS.primary_selected
BG_INPUT = DEFAULT_TOKENS.surface
BG_LOG = DEFAULT_TOKENS.canvas_dark
BG_HEADER = DEFAULT_TOKENS.surface_muted
BG_TAB = DEFAULT_TOKENS.surface_muted

BORDER_CARD = DEFAULT_TOKENS.border
BORDER_INPUT = DEFAULT_TOKENS.border_strong
GRID_LINE = DEFAULT_TOKENS.border_weak

TEXT_PRIMARY = DEFAULT_TOKENS.text
TEXT_SECONDARY = DEFAULT_TOKENS.text_secondary
TEXT_MUTED = DEFAULT_TOKENS.text_secondary
TEXT_ON_PRIMARY = "#ffffff"
TEXT_ON_DARK = "#e2e8f0"

PRIMARY = DEFAULT_TOKENS.primary
PRIMARY_HOVER = DEFAULT_TOKENS.primary_hover

BTN_SECONDARY_BG = DEFAULT_TOKENS.surface_muted
BTN_SECONDARY_FG = DEFAULT_TOKENS.text
BTN_SECONDARY_HOVER = DEFAULT_TOKENS.hover
BTN_DANGER = DEFAULT_TOKENS.danger
BTN_DANGER_HOVER = "#8e332c"
BTN_GHOST_BG = DEFAULT_TOKENS.surface
BTN_GHOST_FG = DEFAULT_TOKENS.text
BTN_GHOST_BORDER = DEFAULT_TOKENS.border_strong
BTN_GHOST_HOVER = DEFAULT_TOKENS.hover
BTN_DISABLED_BG = "#d7dadd"
BTN_DISABLED_FG = DEFAULT_TOKENS.text_weak

STATUS_COLORS = {
    "running": (DEFAULT_TOKENS.primary, "#e5edf4"),
    "cancelling": (DEFAULT_TOKENS.warning, DEFAULT_TOKENS.warning_bg),
    "success": (DEFAULT_TOKENS.success, DEFAULT_TOKENS.success_bg),
    "business_ng": (DEFAULT_TOKENS.warning, DEFAULT_TOKENS.warning_bg),
    "failed": (DEFAULT_TOKENS.danger, DEFAULT_TOKENS.danger_bg),
    "stopped": (DEFAULT_TOKENS.warning, DEFAULT_TOKENS.warning_bg),
    "pending": (DEFAULT_TOKENS.neutral, DEFAULT_TOKENS.neutral_bg),
    "idle": (DEFAULT_TOKENS.neutral, DEFAULT_TOKENS.neutral_bg),
}

RADIUS_CARD = 3
RADIUS_INPUT = 3
RADIUS_BUTTON = 3
RADIUS_BADGE = 2
RADIUS_COMPACT = 2
FONT_SIZE_TITLE = 20
FONT_SIZE_SECTION = 16
FONT_SIZE_LOG = 12
FONT_SIZE_COMPACT = 11


def _tone_for_status(status: str) -> StatusTone:
    return {
        "running": StatusTone.RUNNING,
        "cancelling": StatusTone.WARNING,
        "success": StatusTone.SUCCESS,
        "business_ng": StatusTone.WARNING,
        "failed": StatusTone.DANGER,
        "stopped": StatusTone.WARNING,
        "pending": StatusTone.NEUTRAL,
        "idle": StatusTone.NEUTRAL,
    }.get(status, StatusTone.NEUTRAL)


def status_badge_stylesheet(status: str) -> str:
    return _shared_status_badge_stylesheet(_tone_for_status(status))


def build_app_stylesheet() -> str:
    return build_trainer_stylesheet()


def log_stylesheet() -> str:
    return (
        "font-family: Consolas, 'Courier New', monospace; "
        f"font-size: {FONT_SIZE_LOG}px; background: {BG_LOG}; color: {TEXT_ON_DARK};"
    )
