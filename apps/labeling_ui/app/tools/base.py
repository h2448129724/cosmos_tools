"""Base class for CAB-F tool pages and shared worker thread."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QPlainTextEdit, QProgressBar, QPushButton, QWidget, QVBoxLayout,
)

from cosmos_toolbox.ui.primitives import (
    ActionBar,
    CollapsibleLogPanel,
    EmptyState,
    PageHeader,
    StatusBanner,
    set_ui_role,
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
        self._worker: FuncWorker | None = None

    def on_activated(self):
        pass

    def on_deactivated(self):
        pass

    def worker_running(self) -> bool:
        return self._worker is not None and self._worker.isRunning()

    def run_background(
        self,
        func: Callable[..., Any],
        *args: Any,
        on_result: Callable[[object], None],
        **kwargs: Any,
    ) -> bool:
        """Run one shell operation and retain it until QThread teardown.

        Result delivery and thread lifecycle are separate signals.  This keeps
        a completed operation alive until Qt has emitted its real ``finished``
        signal and prevents pages from destroying a still-running QThread.
        """
        if self.worker_running():
            return False
        worker = FuncWorker(func, *args, parent=self, **kwargs)
        self._worker = worker
        worker.result_ready.connect(on_result)
        worker.finished.connect(lambda: self._release_worker(worker))
        worker.start()
        return True

    def _release_worker(self, worker: FuncWorker) -> None:
        if self._worker is worker:
            self._worker = None
        worker.deleteLater()

    def shutdown(self) -> None:
        """Complete an in-flight non-cancellable file operation before teardown."""
        worker = self._worker
        if worker is None:
            return
        if worker.isRunning():
            worker.requestInterruption()
            worker.wait()
        if self._worker is worker:
            self._worker = None
        worker.deleteLater()

    def make_progress_bar(self) -> tuple[QWidget, QProgressBar, QLabel]:
        """Create a standard progress bar with label. Returns (container, bar, label)."""
        container = make_card()
        lay = QVBoxLayout(container)
        lay.setContentsMargins(14, 10, 14, 10)
        lay.setSpacing(5)

        row = QHBoxLayout()
        self._progress_label = QLabel("准备就绪")
        self._progress_label.setProperty("uiRole", "muted")
        row.addWidget(self._progress_label)
        row.addStretch()
        self._progress_count = QLabel("")
        self._progress_count.setProperty("uiRole", "muted")
        row.addWidget(self._progress_count)
        lay.addLayout(row)

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setTextVisible(False)
        self._progress_bar.setMinimumHeight(6)
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
    result_ready = Signal(object)
    progress = Signal(int, int)
    log = Signal(str)

    def __init__(self, func, *args, parent: QWidget | None = None, **kwargs):
        super().__init__(parent)
        self._func = func
        self._args = args
        self._kwargs = kwargs

    def run(self):
        try:
            result = self._func(*self._args, **self._kwargs)
            self.result_ready.emit(result)
        except Exception as e:
            self.result_ready.emit(e)


# ---------------------------------------------------------------------------
# Shared UI helpers
# ---------------------------------------------------------------------------

def make_card() -> QFrame:
    """Create a shared-surface frame while retaining the legacy QFrame API."""
    f = QFrame()
    f.setObjectName("card")
    set_ui_role(f, "sectionSurface")
    f.setAccessibleName("内容区域")
    return f


def make_header(title: str, desc: str = "") -> QLabel:
    """Legacy single-label header; use :func:`make_page_header` for new pages."""
    label = QLabel(title)
    set_ui_role(label, "pageTitle")
    label.setAccessibleName(title)
    if desc:
        label.setToolTip(desc)
    return label


def make_page_header(title: str, desc: str = "", status: str = "") -> QFrame:
    """Create the shared page header with a backwards-compatible return type."""
    header = PageHeader(title, desc, status)
    return header


def set_primary(btn: QPushButton) -> QPushButton:
    """Mark a button as the primary action (blue accent via stylesheet)."""
    btn.setProperty("primary", "true")
    return btn


def make_action_bar(*widgets: QWidget) -> ActionBar:
    """Compatibility factory for pages that build wrapping action rows."""
    return ActionBar(widgets)


def make_status_banner(text: str = "", tone: str = "neutral") -> StatusBanner:
    """Compatibility factory for shared status feedback presentation."""
    return StatusBanner(text, tone)


def make_hint_panel(title: str, body: str) -> QFrame:
    """Create a compact info panel for guidance or empty states."""
    panel = QFrame()
    panel.setObjectName("hintPanel")
    set_ui_role(panel, "hint")
    panel.setAccessibleName(title)
    lay = QVBoxLayout(panel)
    lay.setContentsMargins(12, 10, 12, 10)
    lay.setSpacing(4)

    head = QLabel(title)
    set_ui_role(head, "sectionTitle")
    lay.addWidget(head)

    text = QLabel(body)
    text.setWordWrap(True)
    set_ui_role(text, "muted")
    lay.addWidget(text)
    return panel


def make_log_box(placeholder: str = "日志...", height: int = 112) -> QPlainTextEdit:
    box = QPlainTextEdit()
    box.setReadOnly(True)
    box.setMinimumHeight(height)
    box.setPlaceholderText(placeholder)
    box.setProperty("uiRole", "logViewer")
    box.setAccessibleName("运行日志")
    return box


def make_log_card(box: QPlainTextEdit, title: str = "运行日志") -> QFrame:
    panel = CollapsibleLogPanel(title, expanded=True)
    panel.setObjectName("logCard")
    # Replace the primitive's default viewer so existing callers retain the
    # exact QPlainTextEdit instance used by their worker callbacks.
    default_log = panel.log
    panel.body_layout.removeWidget(default_log)
    default_log.setParent(None)
    default_log.deleteLater()
    panel.log = box
    panel.body_layout.addWidget(box)
    return panel


def make_empty_state(icon: str, title: str, hint: str) -> QFrame:
    """Create a centered empty-state placeholder with icon, title and hint text."""
    frame = EmptyState(title, hint)
    frame.setObjectName("hintPanel")
    frame.setAccessibleDescription(f"{icon} {hint}" if icon else hint)
    return frame
