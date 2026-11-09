"""One NG image extraction page with two data sources.

The database source copies straight from SQLite with a resumable ledger; the
Excel source works on a machine that only has the exported workbook and the
original image drive.  Both engines stay unchanged; this page only gives them a
single entry and a shared header.
"""
from __future__ import annotations

from PySide6.QtWidgets import QLabel, QTabWidget, QVBoxLayout, QWidget

from .data_transfer_ui import DataTransferPage
from .db_ng_ui import DbNgPage


class NgExtractPage(QWidget):
    capability_key = "data.ng_extract"

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setProperty("ownsPageHeader", True)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        title = QLabel("NG 原图提取")
        title.setStyleSheet("font-size: 22px; font-weight: 600;")
        layout.addWidget(title)
        hint = QLabel(
            "按 NG 产品记录复制原图。能直接访问数据库和原图时用“数据库”来源（可暂停续跑）；"
            "只拿到导出的 Excel 时用“Excel”来源，也可以在这里先把数据库导出为 Excel。"
        )
        hint.setWordWrap(True)
        hint.setProperty("uiRole", "muted")
        layout.addWidget(hint)

        self.database_source = DbNgPage(self)
        self.excel_source = DataTransferPage(self)
        for page in (self.database_source, self.excel_source):
            for widget in getattr(page, "header_widgets", ()):
                widget.hide()
        self.sources = QTabWidget()
        self.sources.setAccessibleName("NG 数据来源")
        self.sources.addTab(self.database_source, "来源：数据库")
        self.sources.addTab(self.excel_source, "来源：Excel / 导出 Excel")
        layout.addWidget(self.sources, 1)

    def bind_runtime(self, task_center, project_session, window) -> None:
        for page in (self.database_source, self.excel_source):
            page.task_center = task_center
            page.protect_window_close(window)
        self.database_source.project_session = project_session
