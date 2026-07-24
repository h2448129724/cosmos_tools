from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Iterator

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from .workspaces import request_shutdown


class EmbeddedPageHost(QFrame):
    back_requested = Signal()

    def __init__(self, page: QWidget, title: str, subtitle: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.page = page
        self.setObjectName("embeddedPageHost")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 14, 18, 14)
        layout.setSpacing(10)

        header = QFrame()
        header.setObjectName("embeddedPageHeader")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        self.back_button = QPushButton("← 返回")
        self.back_button.setObjectName("embeddedBackButton")
        self.back_button.clicked.connect(self.back_requested)
        header_layout.addWidget(self.back_button)
        text_layout = QVBoxLayout()
        text_layout.setSpacing(1)
        title_label = QLabel(title)
        title_label.setObjectName("embeddedPageTitle")
        text_layout.addWidget(title_label)
        if subtitle:
            subtitle_label = QLabel(subtitle)
            subtitle_label.setObjectName("embeddedPageSubtitle")
            subtitle_label.setWordWrap(True)
            text_layout.addWidget(subtitle_label)
        header_layout.addLayout(text_layout, 1)
        layout.addWidget(header)

        page.setParent(self)
        if isinstance(page, QDialog):
            page.setModal(False)
            page.setWindowFlag(Qt.WindowType.Dialog, False)
            page.setWindowFlag(Qt.WindowType.Window, False)
            page.setWindowFlag(Qt.WindowType.Widget, True)
        page.setObjectName(page.objectName() or "embeddedBusinessPage")
        layout.addWidget(page, 1)
        page.show()


@dataclass(slots=True)
class _RouteEntry:
    host: EmbeddedPageHost
    page: QWidget
    source_key: str
    on_finished: Callable[[int], None] | None = None
    closing: bool = False
    finished: bool = False
    callback_called: bool = False
    shutdown_called: bool = False


class PageRouter(QObject):
    """Stack business pages inside the main shell and preserve nested return paths."""

    depth_changed = Signal(int)

    def __init__(
        self,
        stack: QStackedWidget,
        return_to_workspace: Callable[[str], None],
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.stack = stack
        self.return_to_workspace = return_to_workspace
        self._routes: list[_RouteEntry] = []
        self._return_target_override: str | None = None

    @property
    def depth(self) -> int:
        return len(self._routes)

    @property
    def current_page(self) -> QWidget | None:
        return self._routes[-1].page if self._routes else None

    @property
    def current_return_target(self) -> str | None:
        return self._routes[-1].source_key if self._routes else None

    @contextmanager
    def returning_to(self, capability_key: str | None) -> Iterator[None]:
        """Temporarily make newly opened pages return to a shell capability."""
        previous = self._return_target_override
        self._return_target_override = capability_key or previous
        try:
            yield
        finally:
            self._return_target_override = previous

    def open_page(
        self,
        widget: QWidget,
        title: str,
        source_key: str,
        subtitle: str = "",
    ) -> EmbeddedPageHost:
        return self._push(widget, title, source_key, subtitle, None).host

    def open_dialog(
        self,
        dialog: QDialog,
        title: str,
        source_key: str,
        subtitle: str = "",
        on_finished: Callable[[int], None] | None = None,
    ) -> EmbeddedPageHost:
        entry = self._push(dialog, title, source_key, subtitle, on_finished)
        dialog.finished.connect(lambda result, route=entry: self._dialog_finished(route, int(result)))
        return entry.host

    def close_current(self, *, return_to_source: bool = True) -> None:
        if not self._routes:
            return
        entry = self._routes[-1]
        if entry.closing:
            return
        entry.closing = True
        self._shutdown_entry(entry)
        if isinstance(entry.page, QDialog) and not entry.finished:
            entry.page.reject()
            if not entry.finished:
                self._notify_dialog_finished(entry, int(QDialog.DialogCode.Rejected))
        if entry not in self._routes:
            return
        self._routes.remove(entry)
        self.stack.removeWidget(entry.host)
        entry.host.hide()
        entry.host.deleteLater()
        self.depth_changed.emit(len(self._routes))
        if self._routes:
            self.stack.setCurrentWidget(self._routes[-1].host)
        elif return_to_source:
            self.return_to_workspace(entry.source_key)

    def reset(self) -> None:
        while self._routes:
            self.close_current(return_to_source=False)

    def _push(
        self,
        page: QWidget,
        title: str,
        source_key: str,
        subtitle: str,
        on_finished: Callable[[int], None] | None,
    ) -> _RouteEntry:
        host = EmbeddedPageHost(page, title, subtitle, self.stack)
        host.back_requested.connect(self.close_current)
        entry = _RouteEntry(host, page, self._return_target_override or source_key, on_finished)
        self._routes.append(entry)
        self.stack.addWidget(host)
        self.stack.setCurrentWidget(host)
        self.depth_changed.emit(len(self._routes))
        return entry

    def _dialog_finished(self, entry: _RouteEntry, result: int) -> None:
        if entry not in self._routes:
            return
        self._notify_dialog_finished(entry, result)
        if entry.closing:
            return
        if self._routes and self._routes[-1] is entry:
            self.close_current()

    @staticmethod
    def _notify_dialog_finished(entry: _RouteEntry, result: int) -> None:
        entry.finished = True
        if entry.callback_called:
            return
        entry.callback_called = True
        if entry.on_finished is not None:
            entry.on_finished(result)

    def _shutdown_entry(self, entry: _RouteEntry) -> None:
        if entry.shutdown_called:
            return
        entry.shutdown_called = True
        self._shutdown_page(entry.page)

    @staticmethod
    def _shutdown_page(page: QWidget) -> None:
        """Ask an embedded page to release work before its host is destroyed."""
        request_shutdown(page)
