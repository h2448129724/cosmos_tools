"""Compatibility imports for the shared Cosmos desktop UI foundation.

New pages should import from :mod:`cosmos_toolbox.ui`.  These names remain so
standalone labeling entry points and existing project adapters use the same
theme without a flag-day migration.
"""

from __future__ import annotations

from cosmos_toolbox.ui.theme import (
    LEGACY_TOKENS,
    StatusTone,
    build_labeling_stylesheet,
    status_badge_stylesheet,
)

TOKENS = LEGACY_TOKENS
APP_STYLESHEET = build_labeling_stylesheet()


def badge_style(kind: str) -> str:
    aliases = {
        "neutral": StatusTone.NEUTRAL,
        "info": StatusTone.INFO,
        "running": StatusTone.RUNNING,
        "success": StatusTone.SUCCESS,
        "warning": StatusTone.WARNING,
        "danger": StatusTone.DANGER,
        "failed": StatusTone.DANGER,
    }
    return status_badge_stylesheet(aliases.get(kind, StatusTone.NEUTRAL))
