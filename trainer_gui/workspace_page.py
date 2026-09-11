"""Embeddable training workspace page.

The implementation lives in :mod:`trainer_gui.main_window` for backwards
compatibility with existing imports.  This module provides the explicit seam
used by the Cosmos toolbox so callers do not need to know about the standalone
``QMainWindow`` adapter.
"""

from pathlib import Path

from PySide6.QtWidgets import QWidget

from cosmos_toolbox.paths import TOOLBOX_ROOT

from .main_window import TrainingWorkspacePage


def workspace_page(
    project_root: str | Path = TOOLBOX_ROOT,
    parent: QWidget | None = None,
) -> TrainingWorkspacePage:
    """Create the native training page used by the project shell.

    Keeping construction behind a small function makes the embedding seam
    explicit and mirrors the image workspace adapter without exposing a
    nested top-level window.
    """
    return TrainingWorkspacePage(project_root, parent=parent)

__all__ = ["TrainingWorkspacePage", "workspace_page"]
