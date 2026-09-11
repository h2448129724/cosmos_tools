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

from .route_state import RouteState, close_top, finish_dialog, push
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
    route_id: int
    host: EmbeddedPageHost
    page: QWidget
    source_key: str
    on_finished: Callable[[int], None] | None = None
    closing: bool = False
    finished: bool = False
    callback_called: bool = False
    shutdown_called: bool = False
    return_on_close: bool = True


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
        self._state = RouteState()
        self._entries: dict[int, _RouteEntry] = {}
        self._return_target_override: str | None = None

    @property
    def depth(self) -> int:
        return len(self._state.routes)

    @property
    def current_page(self) -> QWidget | None:
        if not self._state.routes:
            return None
        entry = self._entries.get(self._state.routes[-1].route_id)
        return entry.page if entry is not None else None

    @property
    def current_return_target(self) -> str | None:
        return self._state.routes[-1].source_key if self._state.routes else None

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
        if not self._state.routes:
            return
        route = self._state.routes[-1]
        entry = self._entries.get(route.route_id)
        if entry is None:
            self._state = close_top(self._state).state
            return
        if entry.closing:
            return
        entry.closing = True
        entry.return_on_close = return_to_source
        self._shutdown_entry(entry)
        if isinstance(entry.page, QDialog) and not entry.finished:
            entry.page.reject()
            if entry.route_id in self._entries and not entry.finished:
                self._finish_dialog(entry, int(QDialog.DialogCode.Rejected))
            return
        transition = close_top(self._state)
        self._state = transition.state
        self._apply_transition(transition, return_to_source=return_to_source)

    def reset(self) -> None:
        while self._state.routes:
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
        transition = push(self._state, self._return_target_override or source_key, is_dialog=isinstance(page, QDialog))
        self._state = transition.state
        route = transition.added
        assert route is not None
        entry = _RouteEntry(route.route_id, host, page, route.source_key, on_finished)
        self._entries[route.route_id] = entry
        self.stack.addWidget(host)
        self.stack.setCurrentWidget(host)
        self.depth_changed.emit(len(self._state.routes))
        return entry

    def _dialog_finished(self, entry: _RouteEntry, result: int) -> None:
        if entry.route_id not in self._entries or entry.finished:
            return
        self._finish_dialog(entry, result)

    def _finish_dialog(self, entry: _RouteEntry, result: int) -> None:
        transition = finish_dialog(self._state, entry.route_id, result)
        if not transition.changed:
            return
        self._state = transition.state
        try:
            self._notify_dialog_finished(entry, result)
        finally:
            self._apply_transition(
                transition,
                return_to_source=entry.return_on_close,
            )

    def _apply_transition(self, transition, *, return_to_source: bool = True) -> None:
        if not transition.changed:
            return
        for route in transition.removed:
            entry = self._entries.pop(route.route_id, None)
            if entry is None:
                continue
            self._shutdown_entry(entry)
            self.stack.removeWidget(entry.host)
            entry.host.hide()
            entry.host.deleteLater()
        self.depth_changed.emit(len(self._state.routes))
        if self._state.routes:
            top = self._entries.get(self._state.routes[-1].route_id)
            if top is not None:
                self.stack.setCurrentWidget(top.host)
        elif return_to_source and transition.return_to_source:
            self.return_to_workspace(transition.return_to_source)

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
