"""Base class for CAB-F tool pages and shared worker thread."""
from __future__ import annotations

from PySide6.QtCore import QThread, Signal, Qt
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QPlainTextEdit, QProgressBar, QPushButton, QSizePolicy, QWidget, QVBoxLayout,
)


class BaseToolPage(QWidget):
    tool_key: str = ""
    tool_title: str = ""
    tool_nav_title: str = ""
    tool_icon: str = ">"
    tool_summary: str = ""
    tool_tags: tuple[str, ...] = ()

    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self._mw = main_window

    def on_activated(self):
        pass

    def on_deactivated(self):
        pass

    def make_progress_bar(self) -> tuple[QWidget, QProgressBar, QLabel]:
        """Create a standard progress bar with label. Returns (container, bar, label)."""
        container = QFrame()
        container.setObjectName("card")
        lay = QVBoxLayout(container)
        lay.setContentsMargins(14, 10, 14, 10)
        lay.setSpacing(5)

        row = QHBoxLayout()
        self._progress_label = QLabel("准备就绪")
        self._progress_label.setStyleSheet("color:#6B7280;font-size:13px;")
        row.addWidget(self._progress_label)
        row.addStretch()
        self._progress_count = QLabel("")
        self._progress_count.setStyleSheet("color:#9CA3AF;font-size:12px;")
        row.addWidget(self._progress_count)
        lay.addLayout(row)

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setTextVisible(False)
        self._progress_bar.setFixedHeight(6)
        self._progress_bar.setStyleSheet(
            "QProgressBar{background:#EEF0F3;border:none;border-radius:3px;}"
            "QProgressBar::chunk{background:#2563EB;border-radius:3px;}"
        )
        lay.addWidget(self._progress_bar)

        container.hide()
        return container, self._progress_bar, self._progress_label

    def show_progress(self, container: QWidget, label: QLabel,
                      count_label: QLabel, bar: QProgressBar,
                      current: int, total: int, msg: str = ""):
        """Update progress bar state."""
        container.show()
        if total > 0:
            bar.setValue(int(current / total * 100))
            count_label.setText(f"{current}/{total}")
        if msg:
            label.setText(msg)

    def hide_progress(self, container: QWidget, label: QLabel, msg: str = "完成"):
        """Hide progress bar after completion."""
        label.setText(msg)
        from PySide6.QtCore import QTimer
        QTimer.singleShot(1500, container.hide)


class FuncWorker(QThread):
    finished = Signal(object)
    progress = Signal(int, int)
    log = Signal(str)

    def __init__(self, func, *args, **kwargs):
        super().__init__()
        self._func = func
        self._args = args
        self._kwargs = kwargs

    def run(self):
        try:
            result = self._func(*self._args, **self._kwargs)
            self.finished.emit(result)
        except Exception as e:
            self.finished.emit(e)


# ---------------------------------------------------------------------------
# Shared UI helpers
# ---------------------------------------------------------------------------

def make_card() -> QFrame:
    """Create a card-styled container frame (uses #card stylesheet)."""
    f = QFrame()
    f.setObjectName("card")
    return f


def make_header(title: str, desc: str = "") -> QLabel:
    """Create a styled page header with optional description."""
    html = f"<div style='font-size:18px;font-weight:700;color:#0f172a;'>{title}</div>"
    if desc:
        html += (
            f"<div style='font-size:12px;color:#64748b;margin-top:4px;line-height:1.5;'>"
            f"{desc}</div>"
        )
    return QLabel(html)


def make_page_header(title: str, desc: str = "", status: str = "") -> QFrame:
    """Create a lightweight page header with optional status badge."""
    frame = QFrame()
    lay = QVBoxLayout(frame)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(2)

    title_lbl = QLabel(title)
    title_lbl.setStyleSheet("font-size:18px;font-weight:700;color:#111827;")
    lay.addWidget(title_lbl)

    row = QHBoxLayout()
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(10)
    desc_lbl = QLabel(desc)
    desc_lbl.setWordWrap(True)
    desc_lbl.setStyleSheet("font-size:12px;color:#6B7280;")
    row.addWidget(desc_lbl, 1)
    if status:
        status_lbl = QLabel(status)
        status_lbl.setStyleSheet(
            "background:#E9EBED;color:#555D64;border:1px solid #CFD3D7;border-radius:2px;"
            "padding:3px 7px;font-size:11px;font-weight:600;"
        )
        row.addWidget(status_lbl, 0, Qt.AlignTop)
    lay.addLayout(row)
    return frame


def set_primary(btn: QPushButton) -> QPushButton:
    """Mark a button as the primary action (blue accent via stylesheet)."""
    btn.setProperty("primary", "true")
    return btn


def make_hint_panel(title: str, body: str) -> QFrame:
    """Create a compact info panel for guidance or empty states."""
    panel = QFrame()
    panel.setObjectName("hintPanel")
    lay = QVBoxLayout(panel)
    lay.setContentsMargins(12, 10, 12, 10)
    lay.setSpacing(4)

    head = QLabel(title)
    head.setStyleSheet("color:#0f172a;font-weight:700;")
    lay.addWidget(head)

    text = QLabel(body)
    text.setWordWrap(True)
    text.setStyleSheet("color:#475569;line-height:1.5;")
    lay.addWidget(text)
    return panel


def make_log_box(placeholder: str = "日志...", height: int = 112) -> QPlainTextEdit:
    box = QPlainTextEdit()
    box.setReadOnly(True)
    box.setFixedHeight(height)
    box.setPlaceholderText(placeholder)
    return box


def make_log_card(box: QPlainTextEdit, title: str = "运行日志") -> QFrame:
    card = QFrame()
    card.setObjectName("logCard")
    card_height = box.height() + 46
    card.setFixedHeight(card_height)
    card.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
    lay = QVBoxLayout(card)
    lay.setContentsMargins(14, 12, 14, 12)
    lay.setSpacing(6)
    lay.setAlignment(Qt.AlignTop)
    title_lbl = QLabel(title)
    title_lbl.setStyleSheet("font-size:12px;font-weight:600;color:#111827;")
    lay.addWidget(title_lbl)
    lay.addWidget(box)
    return card


def make_empty_state(icon: str, title: str, hint: str) -> QFrame:
    """Create a centered empty-state placeholder with icon, title and hint text."""
    frame = QFrame()
    frame.setObjectName("hintPanel")
    lay = QVBoxLayout(frame)
    lay.setContentsMargins(24, 28, 24, 28)
    lay.setSpacing(8)
    lay.setAlignment(Qt.AlignCenter)

    icon_lbl = QLabel(icon)
    icon_lbl.setAlignment(Qt.AlignCenter)
    icon_lbl.setStyleSheet("font-size:40px;color:#D1D5DB;")
    lay.addWidget(icon_lbl)

    title_lbl = QLabel(title)
    title_lbl.setAlignment(Qt.AlignCenter)
    title_lbl.setStyleSheet("color:#6B7280;font-size:16px;font-weight:600;")
    lay.addWidget(title_lbl)

    hint_lbl = QLabel(hint)
    hint_lbl.setAlignment(Qt.AlignCenter)
    hint_lbl.setWordWrap(True)
    hint_lbl.setStyleSheet("color:#9CA3AF;font-size:13px;")
    lay.addWidget(hint_lbl)

    return frame
