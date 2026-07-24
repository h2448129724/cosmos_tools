"""历史记录面板：可筛选表格 + 状态过滤 + 搜索。"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QHeaderView,
    QHBoxLayout,
    QLineEdit,
    QTableWidget,
    QVBoxLayout,
    QWidget,
)


class HistoryPanel(QWidget):
    """历史任务面板，支持按状态过滤和搜索。"""

    COLUMNS = ["项目名", "功能", "子功能", "状态", "开始时间", "耗时", "输出目录", "操作"]
    COL_COUNT = len(COLUMNS)

    # 状态过滤选项
    STATUS_FILTERS = [
        ("全部", None),
        ("成功", "success"),
        ("失败", "failed"),
        ("停止", "stopped"),
    ]

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # --- 工具栏：过滤 + 搜索 ---
        toolbar = QHBoxLayout()
        toolbar.setSpacing(8)

        self.status_filter = QComboBox()
        self.status_filter.setMinimumHeight(34)
        for label, _ in self.STATUS_FILTERS:
            self.status_filter.addItem(label)

        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("搜索项目名 / 功能...")
        self.search_edit.setMinimumHeight(34)
        self.search_edit.setMaximumWidth(220)

        toolbar.addWidget(self.status_filter)
        toolbar.addWidget(self.search_edit)
        toolbar.addStretch(1)

        # --- 表格 ---
        self.table = QTableWidget(0, self.COL_COUNT)
        self.table.setHorizontalHeaderLabels(self.COLUMNS)
        header = self.table.horizontalHeader()
        for col in range(6):
            header.setSectionResizeMode(col, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(6, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(7, QHeaderView.ResizeMode.Fixed)

        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.table.verticalHeader().setVisible(False)
        self.table.setWordWrap(False)

        layout.addLayout(toolbar)
        layout.addWidget(self.table)

    def current_status_filter(self) -> str | None:
        """返回当前选中的状态过滤值，None 表示全部。"""
        idx = self.status_filter.currentIndex()
        if 0 <= idx < len(self.STATUS_FILTERS):
            return self.STATUS_FILTERS[idx][1]
        return None

    def current_search_text(self) -> str:
        """返回当前搜索框文本（小写）。"""
        return self.search_edit.text().strip().lower()
