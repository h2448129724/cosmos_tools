"""Page-level single-key shortcuts that survive child focus.

``keyPressEvent`` on a page only sees keys its focused child ignores, so a list
widget (type-ahead search) or a disabled-then-re-enabled page silently swallows
A/D/S/W.  ``QShortcut`` with ``WidgetWithChildrenShortcut`` context fires while
focus is anywhere inside the page; text inputs still get plain typing because
they claim printable keys through ``ShortcutOverride``.
"""
from __future__ import annotations

from typing import Callable, Mapping

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import QWidget


def bind_page_shortcuts(page: QWidget, mapping: Mapping[Qt.Key | str, Callable[[], object]]) -> list[QShortcut]:
    shortcuts = []
    for key, handler in mapping.items():
        sequence = QKeySequence(key) if isinstance(key, str) else QKeySequence(int(key))
        shortcut = QShortcut(sequence, page)
        shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        shortcut.activated.connect(handler)
        shortcuts.append(shortcut)
    return shortcuts
