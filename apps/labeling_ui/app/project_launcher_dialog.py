"""Project-specific tool hub dialogs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
    QSizePolicy,
)
from .tools.base import make_page_header
from cosmos_toolbox.ui.primitives import set_ui_role


@dataclass(frozen=True)
class ProjectToolEntry:
    key: str
    title: str
    description: str
    launch: Callable[[], None]
    category: str = "工具"
    featured: bool = False


class ProjectToolsHubDialog(QDialog):
    """Lightweight navigation hub for project-specific tools."""

    def __init__(self, project_name: str, tools: Sequence[ProjectToolEntry], parent=None):
        super().__init__(parent)
        self._project_name = project_name
        self._tools = list(tools)
        self._category_titles: list[str] = []
        self._featured_tool_keys: list[str] = []
        self._setup_ui()
        self._populate_tools()

    def _display_project_name(self) -> str:
        return "CAB-F" if self._project_name.startswith("CAB-F") else self._project_name

    def category_titles(self) -> list[str]:
        return list(self._category_titles)

    def featured_tool_keys(self) -> list[str]:
        return list(self._featured_tool_keys)

    def _setup_ui(self) -> None:
        display_name = self._display_project_name()
        self.setWindowTitle(f"{display_name} 项目适配")
        self.resize(980, 620)
        # Visuals come from the shared foundation stylesheet; keep this dialog
        # free of one-off colors so it follows application theme changes.

        root = QVBoxLayout(self)
        root.setContentsMargins(24, 22, 24, 22)
        root.setSpacing(16)

        hero = make_page_header(
            f"{display_name} 项目适配",
            "先从主流程开始，需要精修或复核时再进入对应工具。",
        )
        self.meta_label = QLabel("")
        set_ui_role(self.meta_label, "muted")
        self.meta_label.setAccessibleName("工具注册状态")
        hero.layout().addWidget(self.meta_label)
        root.addWidget(hero)

        content = QHBoxLayout()
        content.setSpacing(10)
        root.addLayout(content, 1)

        scroll = QScrollArea()
        scroll.setObjectName("projectToolScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        grid_widget = QWidget()
        grid_widget.setObjectName("projectToolGrid")
        self._grid_layout = QGridLayout(grid_widget)
        self._grid_layout.setHorizontalSpacing(14)
        self._grid_layout.setVerticalSpacing(10)
        self._grid_layout.setContentsMargins(0, 0, 0, 0)
        self._grid_layout.setColumnStretch(0, 1)
        self._grid_layout.setColumnStretch(1, 1)
        scroll.setWidget(grid_widget)
        content.addWidget(scroll, 1)

    def _populate_tools(self) -> None:
        cols = 2
        grouped: dict[str, list[ProjectToolEntry]] = {}
        for tool in self._tools:
            grouped.setdefault(tool.category or "工具", []).append(tool)

        grid_row = 0
        for category, tools in grouped.items():
            self._category_titles.append(category)
            section = QLabel(category)
            set_ui_role(section, "sectionTitle")
            section.setAccessibleName(f"工具分类：{category}")
            self._grid_layout.addWidget(section, grid_row, 0, 1, cols)
            grid_row += 1
            for i, tool in enumerate(tools):
                if tool.featured:
                    self._featured_tool_keys.append(tool.key)
                card = self._make_tool_card(tool)
                col = i % cols
                if tool.featured:
                    self._grid_layout.addWidget(card, grid_row, 0, 1, cols)
                    grid_row += 1
                    continue
                self._grid_layout.addWidget(card, grid_row, col)
                if col == cols - 1 or i == len(tools) - 1:
                    grid_row += 1

        if self._tools:
            self.meta_label.setText(f"已注册 {len(self._tools)} 个可用工具。")
        else:
            self.meta_label.setText("当前项目暂无已注册工具。")

    def _make_tool_card(self, tool: ProjectToolEntry) -> QFrame:
        card = QFrame()
        card.setProperty("tool_key", tool.key)
        if tool.featured:
            card.setObjectName("featuredProjectToolCard")
        else:
            card.setObjectName("projectToolCard")
        card.setCursor(Qt.PointingHandCursor)
        set_ui_role(card, "sectionSurface")
        card.setToolTip(tool.description)
        card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        lay = QHBoxLayout(card)
        lay.setContentsMargins(18, 14, 16, 14)
        lay.setSpacing(14)

        text_col = QVBoxLayout()
        text_col.setContentsMargins(0, 0, 0, 0)
        text_col.setSpacing(5)

        title = QLabel(tool.title)
        set_ui_role(title, "sectionTitle")
        title.setAccessibleName(f"工具：{tool.title}")
        text_col.addWidget(title)

        desc = QLabel(tool.description)
        desc.setWordWrap(True)
        set_ui_role(desc, "muted")
        text_col.addWidget(desc)
        text_col.addStretch(1)
        lay.addLayout(text_col, 1)

        btn = QPushButton("进入" if tool.featured else "打开")
        btn.setMinimumWidth(72)
        btn.setProperty("buttonRole", "primary" if tool.featured else "default")
        btn.setAccessibleName(f"{('进入' if tool.featured else '打开')}工具：{tool.title}")
        btn.setToolTip(tool.description)
        btn.clicked.connect(lambda checked=False, t=tool: t.launch())
        lay.addWidget(btn, 0, Qt.AlignVCenter)
        return card
