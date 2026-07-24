"""Reusable autosave status helpers for editor dialogs."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtWidgets import QLabel, QPushButton


class AutoSaveStatusController:
    """Track dirty/saving/saved states with a small label and optional button hint."""

    def __init__(self, label: QLabel, save_button: Optional[QPushButton] = None) -> None:
        self._label = label
        self._save_button = save_button
        self._base_button_text = save_button.text() if save_button is not None else ""
        self.mark_pristine("未修改")

    def mark_pristine(self, message: str = "未修改") -> None:
        self._apply(message, "#2d6a4f", "#e8f3ec", dirty=False)

    def mark_dirty(self, message: str = "未保存修改") -> None:
        self._apply(message, "#a15c00", "#fff1dc", dirty=True)

    def mark_saving(self, *, auto: bool) -> None:
        self._apply("自动保存中" if auto else "正在保存", "#155a9c", "#e6f0fb", dirty=True)

    def mark_saved(self, path: str | Path | None, *, auto: bool) -> None:
        prefix = "已自动保存" if auto else "已保存"
        suffix = f"：{Path(path).name}" if path else ""
        self._apply(prefix + suffix, "#1f7a1f", "#e9f6ea", dirty=False)

    def mark_error(self, message: str = "保存失败") -> None:
        self._apply(message, "#a12626", "#fbeaea", dirty=True)

    def _apply(self, text: str, color: str, background: str, *, dirty: bool) -> None:
        dot = f"<span style='color:{color}; font-size:16px;'>●</span>"
        content = f"{dot} <span style='color:{color}; font-weight:600;'>{text}</span>"
        self._label.setText(content)
        self._label.setStyleSheet(
            "padding: 4px 10px;"
            "border-radius: 2px;"
            f"background-color: {background};"
        )
        if self._save_button is not None:
            self._save_button.setText(self._base_button_text + (" *" if dirty else ""))
