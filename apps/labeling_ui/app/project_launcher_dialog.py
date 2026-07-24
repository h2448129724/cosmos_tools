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
)
from .tools.base import make_page_header


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
        self.setStyleSheet(
            "ProjectToolsHubDialog{background:#f6f8fb;}"
            "#projectToolScroll{background:transparent;border:none;}"
            "#projectToolGrid{background:transparent;}"
            "#featuredProjectToolCard{background:#f8fbff;border:1px solid #b8cdf8;"
            "border-left:3px solid #356a9a;border-radius:2px;}"
            "#projectToolCard{background:#ffffff;border:1px solid #cfd3d7;border-radius:3px;}"
            "#projectToolCard:hover{border-color:#94a3b8;background:#fbfdff;}"
        )

        root = QVBoxLayout(self)
        root.setContentsMargins(24, 22, 24, 22)
        root.setSpacing(16)

        hero = make_page_header(
            f"{display_name} 项目适配",
            "先从主流程开始，需要精修或复核时再进入对应工具。",
        )
        self.meta_label = QLabel("")
        self.meta_label.setStyleSheet("color:#64748b;font-size:12px;")
        hero.layout().addWidget(self.meta_label)
        root.addWidget(hero)

        content = QHBoxLayout()
        content.setSpacing(10)
        root.addLayout(content, 1)

        scroll = QScrollArea()
        scroll.setObjectName("projectToolScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.viewport().setStyleSheet("background:transparent;")
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
            section.setStyleSheet("color:#0f172a;font-size:15px;font-weight:700;padding:10px 0 4px;")
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
        card.setMinimumHeight(112 if tool.featured else 96)
        lay = QHBoxLayout(card)
        lay.setContentsMargins(18, 14, 16, 14)
        lay.setSpacing(14)

        text_col = QVBoxLayout()
        text_col.setContentsMargins(0, 0, 0, 0)
        text_col.setSpacing(5)

        title = QLabel(tool.title)
        title.setStyleSheet(
            f"color:#0f172a;font-size:{'18' if tool.featured else '16'}px;font-weight:700;"
            "background:transparent;border:none;"
        )
        text_col.addWidget(title)

        desc = QLabel(tool.description)
        desc.setWordWrap(True)
        desc.setStyleSheet("color:#475569;font-size:13px;line-height:1.5;background:transparent;border:none;")
        text_col.addWidget(desc)
        text_col.addStretch(1)
        lay.addLayout(text_col, 1)

        btn = QPushButton("进入" if tool.featured else "打开")
        btn.setFixedSize(92, 36)
        btn.setStyleSheet(
            "QPushButton{background:#2563eb;color:white;border:none;border-radius:6px;font-size:14px;font-weight:600;}"
            "QPushButton:hover{background:#1d4ed8;}"
            if tool.featured
            else
            "QPushButton{background:#ffffff;color:#0f172a;border:1px solid #cbd5e1;border-radius:6px;font-size:14px;}"
            "QPushButton:hover{background:#f8fafc;border-color:#94a3b8;}"
        )
        btn.clicked.connect(lambda checked=False, t=tool: t.launch())
        lay.addWidget(btn, 0, Qt.AlignVCenter)
        return card
