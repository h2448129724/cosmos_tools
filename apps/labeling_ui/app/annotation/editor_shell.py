"""Shared chrome for the point and graph annotation editors.

The canvases remain deliberately dumb: this module only owns the dialog
layout and presentation primitives.  Keeping the shell in one place means a
short viewport behaves the same for both editors while preserving the legacy
widget attributes used by adapters and tests.
"""

from __future__ import annotations

from PySide6.QtCore import QSize, Qt
from PySide6.QtWidgets import QFrame, QScrollArea, QSplitter, QVBoxLayout, QWidget

from cosmos_toolbox.ui.primitives import ActionBar, PageHeader, SectionSurface, StatusBanner


class _WrappingActionBar(ActionBar):
    """ActionBar variant that reports its wrapped height to parent layouts."""

    def hasHeightForWidth(self) -> bool:  # noqa: N802 - Qt interface
        return True

    def heightForWidth(self, width: int) -> int:  # noqa: N802 - Qt interface
        return max(20, self.flow.heightForWidth(max(width, 1)))

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt interface
        width = max(self.width(), 480)
        return QSize(width, self.heightForWidth(width))

    def minimumSizeHint(self) -> QSize:  # noqa: N802 - Qt interface
        return QSize(80, 20)


class AnnotationEditorShell(QWidget):
    """Header, wrapping actions, scrollable inspector and canvas splitter.

    ``left_panel`` intentionally remains a :class:`QScrollArea`; older
    callers use that public name to hide/show the inspector.  The content
    widget and layout are exposed as ``sidebar_content`` and
    ``sidebar_layout`` for incremental migration of existing dialogs.
    """

    def __init__(self, title: str, description: str = "", parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("annotationEditorShell")
        self.page_header = PageHeader(title, description, parent=self)
        self.page_header.setAccessibleName(title)
        self.action_bar = _WrappingActionBar(parent=self)
        self.action_bar.setAccessibleName("标注编辑器操作栏")

        self.splitter = QSplitter(Qt.Orientation.Horizontal, self)
        self.left_panel = QScrollArea(self.splitter)
        self.left_panel.setObjectName("annotationEditorSidebar")
        self.left_panel.setWidgetResizable(True)
        self.left_panel.setFrameShape(QFrame.Shape.NoFrame)
        self.left_panel.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.left_panel.setAccessibleName("编辑器侧栏")
        self.sidebar_content = QWidget()
        self.sidebar_content.setObjectName("annotationEditorSidebarContent")
        self.sidebar_layout = QVBoxLayout(self.sidebar_content)
        self.sidebar_layout.setContentsMargins(0, 0, 6, 0)
        self.sidebar_layout.setSpacing(8)
        self.sidebar_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.left_panel.setWidget(self.sidebar_content)

        self.canvas_host = QWidget(self.splitter)
        self.canvas_layout = QVBoxLayout(self.canvas_host)
        self.canvas_layout.setContentsMargins(0, 0, 0, 0)
        self.canvas_layout.setSpacing(0)
        self.canvas_host.setAccessibleName("标注画布")
        self.status_banner = StatusBanner("准备就绪", parent=self)
        self.status_banner.setAccessibleName("编辑器状态")

        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(8)
        outer.addWidget(self.page_header)
        outer.addWidget(self.action_bar)
        outer.addWidget(self.splitter, 1)
        outer.addWidget(self.status_banner)

    def add_action(self, widget: QWidget) -> QWidget:
        return self.action_bar.add_widget(widget)

    def add_sidebar(self, widget: QWidget, stretch: int = 0) -> QWidget:
        self.sidebar_layout.addWidget(widget, stretch)
        return widget

    def add_sidebar_section(
        self,
        widget: QWidget,
        title: str = "",
        description: str = "",
        *,
        nested: bool = False,
    ) -> SectionSurface:
        section = SectionSurface(title, description, nested=nested, parent=self.sidebar_content)
        section.body_layout.addWidget(widget)
        self.add_sidebar(section)
        return section

    def set_canvas(self, widget: QWidget) -> QWidget:
        self.canvas_layout.addWidget(widget)
        return widget

    def set_status(self, text: str, tone: str = "neutral") -> None:
        self.status_banner.set_status(text, tone)


__all__ = ["AnnotationEditorShell"]
