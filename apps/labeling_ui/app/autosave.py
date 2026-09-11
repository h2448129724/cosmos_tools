"""Reusable autosave status helpers for editor dialogs."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtWidgets import QLabel, QPushButton
from cosmos_toolbox.ui.primitives import set_tone, set_ui_role


class AutoSaveStatusController:
    """Track dirty/saving/saved states with a small label and optional button hint."""

    def __init__(self, label: QLabel, save_button: Optional[QPushButton] = None) -> None:
        self._label = label
        self._save_button = save_button
        self._base_button_text = save_button.text() if save_button is not None else ""
        self.mark_pristine("未修改")

    def mark_pristine(self, message: str = "未修改") -> None:
        self._apply(message, "success", dirty=False)

    def mark_dirty(self, message: str = "未保存修改") -> None:
        self._apply(message, "warning", dirty=True)

    def mark_saving(self, *, auto: bool) -> None:
        self._apply("自动保存中" if auto else "正在保存", "running", dirty=True)

    def mark_saved(self, path: str | Path | None, *, auto: bool) -> None:
        prefix = "已自动保存" if auto else "已保存"
        suffix = f"：{Path(path).name}" if path else ""
        self._apply(prefix + suffix, "success", dirty=False)

    def mark_error(self, message: str = "保存失败") -> None:
        self._apply(message, "danger", dirty=True)

    def _apply(self, text: str, tone: str, *, dirty: bool) -> None:
        self._label.setObjectName("statusBadge")
        set_ui_role(self._label, "statusBadge")
        set_tone(self._label, tone)
        self._label.setText(f"● {text}")
        self._label.setAccessibleName(f"自动保存状态：{text}")
        self._label.setToolTip(text)
        if self._save_button is not None:
            self._save_button.setText(self._base_button_text + (" *" if dirty else ""))
