"""Stable Qt primitives used by project activities and compatibility pages."""

from __future__ import annotations

from collections.abc import Iterable

from PySide6.QtCore import QPoint, QRect, QSize, Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QLayout,
    QLayoutItem,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QWidgetItem,
)

from .theme import StatusTone


def _refresh_style(widget: QWidget) -> None:
    style = widget.style()
    style.unpolish(widget)
    style.polish(widget)
    widget.update()


def set_ui_role(widget: QWidget, role: str) -> QWidget:
    widget.setProperty("uiRole", role)
    _refresh_style(widget)
    return widget


def set_tone(widget: QWidget, tone: StatusTone | str) -> QWidget:
    widget.setProperty("tone", StatusTone(tone).value)
    _refresh_style(widget)
    return widget


class FlowLayout(QLayout):
    """Small wrapping layout for action bars and metric groups."""

    def __init__(self, parent: QWidget | None = None, margin: int = 0, h_spacing: int = 8, v_spacing: int = 6) -> None:
        super().__init__(parent)
        self._items: list[QLayoutItem] = []
        self._h_spacing = h_spacing
        self._v_spacing = v_spacing
        self.setContentsMargins(margin, margin, margin, margin)

    def addItem(self, item: QLayoutItem) -> None:  # noqa: N802 - Qt interface
        self._items.append(item)

    def addWidget(self, widget: QWidget) -> None:  # noqa: N802 - Qt interface
        self.addChildWidget(widget)
        self.addItem(QWidgetItem(widget))

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index: int) -> QLayoutItem | None:  # noqa: N802 - Qt interface
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index: int) -> QLayoutItem | None:  # noqa: N802 - Qt interface
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def expandingDirections(self) -> Qt.Orientations:  # noqa: N802 - Qt interface
        return Qt.Orientations()

    def hasHeightForWidth(self) -> bool:  # noqa: N802 - Qt interface
        return True

    def heightForWidth(self, width: int) -> int:  # noqa: N802 - Qt interface
        return self._do_layout(QRect(0, 0, width, 0), test_only=True)

    def setGeometry(self, rect: QRect) -> None:  # noqa: N802 - Qt interface
        super().setGeometry(rect)
        self._do_layout(rect, test_only=False)

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt interface
        return self.minimumSize()

    def minimumSize(self) -> QSize:  # noqa: N802 - Qt interface
        size = QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        margins = self.contentsMargins()
        size += QSize(margins.left() + margins.right(), margins.top() + margins.bottom())
        return size

    def _do_layout(self, rect: QRect, *, test_only: bool) -> int:
        margins = self.contentsMargins()
        effective = rect.adjusted(margins.left(), margins.top(), -margins.right(), -margins.bottom())
        x = effective.x()
        y = effective.y()
        line_height = 0
        for item in self._items:
            widget = item.widget()
            if widget is not None and not widget.isVisible() and not test_only:
                continue
            hint = item.sizeHint()
            next_x = x + hint.width() + self._h_spacing
            if x > effective.x() and next_x - self._h_spacing > effective.right() + 1:
                x = effective.x()
                y += line_height + self._v_spacing
                next_x = x + hint.width() + self._h_spacing
                line_height = 0
            if not test_only:
                item.setGeometry(QRect(QPoint(x, y), hint))
            x = next_x
            line_height = max(line_height, hint.height())
        return y + line_height - rect.y() + margins.bottom()


class PageHeader(QFrame):
    def __init__(self, title: str, description: str = "", status: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        set_ui_role(self, "pageHeader")
        self.setAccessibleName(title)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        text = QVBoxLayout()
        text.setSpacing(2)
        self.title_label = QLabel(title)
        set_ui_role(self.title_label, "pageTitle")
        self.title_label.setWordWrap(True)
        self.description_label = QLabel(description)
        set_ui_role(self.description_label, "pageDescription")
        self.description_label.setWordWrap(True)
        text.addWidget(self.title_label)
        if description:
            text.addWidget(self.description_label)
        layout.addLayout(text, 1)
        self.status_label = QLabel(status)
        self.status_label.setVisible(bool(status))
        self.status_label.setAccessibleName("页面状态")
        set_ui_role(self.status_label, "statusBadge")
        layout.addWidget(self.status_label, 0, Qt.AlignmentFlag.AlignTop)

    def set_status(self, text: str, tone: StatusTone | str = StatusTone.NEUTRAL) -> None:
        self.status_label.setText(text)
        self.status_label.setVisible(bool(text))
        set_tone(self.status_label, tone)


class SectionSurface(QFrame):
    def __init__(
        self,
        title: str = "",
        description: str = "",
        *,
        nested: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        set_ui_role(self, "sectionSurface")
        self.setProperty("nested", nested)
        self.setAccessibleName(title or "内容区域")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(8)
        self.header = QWidget()
        header_layout = QVBoxLayout(self.header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(2)
        self.title_label = QLabel(title)
        set_ui_role(self.title_label, "sectionTitle")
        self.description_label = QLabel(description)
        self.description_label.setWordWrap(True)
        set_ui_role(self.description_label, "muted")
        if title:
            header_layout.addWidget(self.title_label)
        if description:
            header_layout.addWidget(self.description_label)
        self.header.setVisible(bool(title or description))
        layout.addWidget(self.header)
        self.body = QWidget()
        self.body_layout = QVBoxLayout(self.body)
        self.body_layout.setContentsMargins(0, 0, 0, 0)
        self.body_layout.setSpacing(8)
        layout.addWidget(self.body)


class PathField(QWidget):
    browse_requested = Signal()
    open_requested = Signal()

    def __init__(
        self,
        label: str,
        *,
        placeholder: str = "",
        browse_text: str = "选择",
        show_open: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setAccessibleName(label)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        self.label = QLabel(label)
        set_ui_role(self.label, "fieldLabel")
        layout.addWidget(self.label)
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        self.line_edit = QLineEdit()
        self.line_edit.setPlaceholderText(placeholder)
        self.line_edit.setAccessibleName(label)
        row.addWidget(self.line_edit, 1)
        self.browse_button = QPushButton(browse_text)
        self.browse_button.setProperty("buttonRole", "secondary")
        self.browse_button.setAccessibleName(f"选择{label}")
        self.browse_button.clicked.connect(self.browse_requested)
        row.addWidget(self.browse_button)
        self.open_button = QPushButton("打开")
        self.open_button.setProperty("buttonRole", "ghost")
        self.open_button.setAccessibleName(f"打开{label}")
        self.open_button.clicked.connect(self.open_requested)
        self.open_button.setVisible(show_open)
        row.addWidget(self.open_button)
        layout.addLayout(row)

    def text(self) -> str:
        return self.line_edit.text()

    def setText(self, value: str) -> None:  # noqa: N802 - mirrors QLineEdit
        self.line_edit.setText(value)

    def set_validation(self, state: str, message: str = "") -> None:
        self.line_edit.setProperty("validationState", state)
        self.line_edit.setToolTip(message)
        self.line_edit.setAccessibleDescription(message)
        _refresh_style(self.line_edit)


class ActionBar(QFrame):
    def __init__(self, widgets: Iterable[QWidget] = (), parent: QWidget | None = None) -> None:
        super().__init__(parent)
        set_ui_role(self, "actionBar")
        self.flow = FlowLayout(self, h_spacing=8, v_spacing=6)
        for widget in widgets:
            self.add_widget(widget)

    def add_widget(self, widget: QWidget) -> QWidget:
        self.flow.addWidget(widget)
        return widget

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt interface
        super().resizeEvent(event)
        self.flow.setGeometry(self.rect())

    def showEvent(self, event) -> None:  # noqa: N802 - Qt interface
        super().showEvent(event)
        self.flow.setGeometry(self.rect())


class StatusBanner(QFrame):
    def __init__(self, text: str = "", tone: StatusTone | str = StatusTone.NEUTRAL, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        set_ui_role(self, "statusBanner")
        self.setAccessibleName("状态")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 7, 10, 7)
        layout.setSpacing(8)
        self.label = QLabel(text)
        self.label.setWordWrap(True)
        layout.addWidget(self.label, 1)
        self.set_status(text, tone)

    def set_status(self, text: str, tone: StatusTone | str = StatusTone.NEUTRAL) -> None:
        self.label.setText(text)
        self.setAccessibleDescription(text)
        set_tone(self, tone)


class MetricGrid(QFrame):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.flow = FlowLayout(self, h_spacing=8, v_spacing=8)
        self._metrics: dict[str, tuple[QLabel, QLabel]] = {}

    def set_metrics(self, metrics: Iterable[tuple[str, str]]) -> None:
        while self.flow.count():
            item = self.flow.takeAt(0)
            if item is not None and item.widget() is not None:
                item.widget().deleteLater()
        self._metrics.clear()
        for key, value in metrics:
            frame = QFrame()
            set_ui_role(frame, "metric")
            frame.setMinimumWidth(118)
            layout = QVBoxLayout(frame)
            layout.setContentsMargins(10, 8, 10, 8)
            layout.setSpacing(1)
            value_label = QLabel(str(value))
            set_ui_role(value_label, "metricValue")
            label = QLabel(str(key))
            set_ui_role(label, "metricLabel")
            layout.addWidget(value_label)
            layout.addWidget(label)
            self.flow.addWidget(frame)
            self._metrics[str(key)] = (value_label, label)


class EmptyState(QFrame):
    action_requested = Signal()

    def __init__(
        self,
        title: str,
        description: str,
        action_text: str = "",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        set_ui_role(self, "emptyState")
        self.setObjectName("emptyState")
        self.setAccessibleName(title)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 28, 24, 28)
        layout.setSpacing(8)
        layout.addStretch(1)
        title_label = QLabel(title)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        set_ui_role(title_label, "emptyStateTitle")
        description_label = QLabel(description)
        description_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        description_label.setWordWrap(True)
        set_ui_role(description_label, "emptyStateDescription")
        layout.addWidget(title_label)
        layout.addWidget(description_label)
        self.action_button = QPushButton(action_text)
        self.action_button.setProperty("buttonRole", "secondary")
        self.action_button.clicked.connect(self.action_requested)
        self.action_button.setVisible(bool(action_text))
        layout.addWidget(self.action_button, 0, Qt.AlignmentFlag.AlignHCenter)
        layout.addStretch(1)


class CollapsibleLogPanel(SectionSurface):
    def __init__(self, title: str = "运行日志", *, expanded: bool = True, parent: QWidget | None = None) -> None:
        super().__init__(parent=parent)
        self.header.setVisible(True)
        header_layout = self.header.layout()
        self.toggle = QToolButton()
        self.toggle.setText(title)
        self.toggle.setCheckable(True)
        self.toggle.setChecked(expanded)
        self.toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.toggle.setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)
        self.toggle.setAccessibleName(f"{title}展开状态")
        header_layout.addWidget(self.toggle)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setProperty("uiRole", "logViewer")
        self.log.setAccessibleName(title)
        self.body_layout.addWidget(self.log)
        self.body.setVisible(expanded)
        self.toggle.toggled.connect(self._set_expanded)

    def _set_expanded(self, expanded: bool) -> None:
        self.body.setVisible(expanded)
        self.toggle.setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)


class PageScaffold(QWidget):
    """Header + one scrolling content owner + optional sticky action bar."""

    def __init__(self, title: str, description: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setProperty("ownsPageHeader", True)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(10)
        self.page_header = PageHeader(title, description)
        outer.addWidget(self.page_header)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.content = QWidget()
        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.setSpacing(10)
        self.content_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.scroll.setWidget(self.content)
        outer.addWidget(self.scroll, 1)
        self.action_bar = ActionBar()
        self.action_bar.hide()
        outer.addWidget(self.action_bar)

    def add_action(self, widget: QWidget) -> QWidget:
        self.action_bar.show()
        return self.action_bar.add_widget(widget)
