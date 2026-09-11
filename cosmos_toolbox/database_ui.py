"""Native read-only database activity for the toolbox."""

from PySide6.QtCore import QEvent, QThread, Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .database_reader import inspect_database, read_inspection_details
from .inspection_detail_ui import InspectionDetailPanel
from .paths import COSMOS_ROOT
from .ui.primitives import FlowLayout


class DatabaseWorker(QThread):
    def __init__(self, options, parent=None, reader=inspect_database):
        super().__init__(parent)
        self.options = options
        self.reader = reader
        self.result = None
        self.error = ""

    def run(self):
        try:
            self.result = self.reader(**self.options)
        except Exception as exc:
            self.error = str(exc)


class DatabasePage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.worker = None
        self.offset = 0
        self.total = 0
        self.rows = []
        self.columns = []
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("只读浏览 SQLite 数据库 · 每页 200 条 · 双击单元格查看完整内容"))
        self.controls = QWidget()
        control_layout = QVBoxLayout(self.controls)
        control_layout.setContentsMargins(0, 0, 0, 0)
        path_row = QHBoxLayout()
        self.path = QLineEdit(str(COSMOS_ROOT / "cosmos.db"))
        self.path.setReadOnly(True)
        self.path.setMinimumWidth(100)
        path_row.addWidget(self.path, 1)
        choose = QPushButton("选择数据库")
        choose.clicked.connect(self.choose_database)
        path_row.addWidget(choose)
        self.open_button = QPushButton("打开 / 刷新")
        self.open_button.clicked.connect(lambda: self.load(reset=True))
        path_row.addWidget(self.open_button)
        self.inspect_button = QPushButton("查看检查项")
        self.inspect_button.clicked.connect(self.inspect_selected)
        path_row.addWidget(self.inspect_button)
        control_layout.addLayout(path_row)
        filters = FlowLayout()
        self.table = QComboBox()
        self.table.setMinimumWidth(150)
        self.table.setAccessibleName("数据表")
        self.column = QComboBox()
        self.column.setAccessibleName("筛选字段")
        self.column.addItem("所有字段", None)
        self.keyword = QLineEdit()
        self.keyword.setPlaceholderText("包含文本（区分大小写）")
        self.keyword.returnPressed.connect(lambda: self.load(reset=True))
        self.order = QComboBox()
        self.order.setAccessibleName("排序字段")
        self.order.addItem("默认排序", None)
        self.direction = QComboBox()
        self.direction.addItems(["升序", "降序"])
        apply = QPushButton("查询")
        apply.clicked.connect(lambda: self.load(reset=True))
        for widget in (
            QLabel("数据表"),
            self.table,
            self.column,
            self.keyword,
            QLabel("排序"),
            self.order,
            self.direction,
            apply,
        ):
            filters.addWidget(widget)
        control_layout.addLayout(filters)
        layout.addWidget(self.controls)
        self.table.currentTextChanged.connect(self.table_changed)
        splitter = QSplitter(Qt.Orientation.Vertical)
        tabs = QTabWidget()
        self.data = self.make_table()
        self.schema = self.make_table()
        tabs.addTab(self.data, "表数据")
        tabs.addTab(self.schema, "字段结构")
        splitter.addWidget(tabs)
        self.detail = QPlainTextEdit()
        self.detail.setReadOnly(True)
        self.detail.setPlaceholderText("双击数据单元格，在此查看完整内容（NULL 与空字符串分别显示）。")
        self.detail_tabs = QTabWidget()
        self.inspector = InspectionDetailPanel()
        self.detail_tabs.addTab(self.inspector, "检查项详情")
        self.detail_tabs.addTab(self.detail, "单元格原文")
        splitter.addWidget(self.detail_tabs)
        splitter.setSizes([300, 320])
        layout.addWidget(splitter, 1)
        self.data.cellDoubleClicked.connect(self.show_detail)
        navigation = QHBoxLayout()
        self.previous = QPushButton("上一页")
        self.next = QPushButton("下一页")
        self.previous.clicked.connect(lambda: self.load(offset=max(0, self.offset - 200)))
        self.next.clicked.connect(lambda: self.load(offset=self.offset + 200))
        self.status = QLabel("选择数据库后点击“打开 / 刷新”。")
        self.status.setWordWrap(True)
        navigation.addWidget(self.previous)
        navigation.addWidget(self.next)
        navigation.addWidget(self.status, 1)
        layout.addLayout(navigation)
        self.previous.setEnabled(False)
        self.next.setEnabled(False)

    @staticmethod
    def make_table():
        table = QTableWidget()
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setAlternatingRowColors(True)
        table.setWordWrap(False)
        return table

    def choose_database(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择 SQLite 数据库",
            self.path.text(),
            "SQLite 数据库 (*.db *.sqlite *.sqlite3);;所有文件 (*)",
        )
        if path:
            self.path.setText(path)
            self.table.blockSignals(True)
            self.table.clear()
            self.table.blockSignals(False)
            self.reset_fields()
            self.load(reset=True)

    def reset_fields(self):
        self.column.clear()
        self.column.addItem("所有字段", None)
        self.order.clear()
        self.order.addItem("默认排序", None)
        self.keyword.clear()

    def table_changed(self):
        self.reset_fields()
        self.load(reset=True)

    def load(self, *, reset=False, offset=None):
        if self.worker is not None:
            return
        self.controls.setEnabled(False)
        self.previous.setEnabled(False)
        self.next.setEnabled(False)
        self.status.setText("正在只读查询…")
        self.inspector.clear()
        self.worker = DatabaseWorker(
            dict(
                path=self.path.text(),
                table=self.table.currentText() or None,
                column=self.column.currentData(),
                keyword=self.keyword.text(),
                order=self.order.currentData(),
                descending=self.direction.currentIndex() == 1,
                offset=0 if reset else (self.offset if offset is None else offset),
            ),
            self,
        )
        self.worker.finished.connect(self.loaded)
        self.worker.start()

    def loaded(self):
        worker = self.worker
        self.worker = None
        self.controls.setEnabled(True)
        if worker.error:
            self.rows = []
            self.data.setRowCount(0)
            self.schema.setRowCount(0)
            self.detail.clear()
            self.status.setText(f"读取失败：{worker.error}。大表查询超时可选择具体字段缩小筛选范围。")
            worker.deleteLater()
            return
        result = worker.result
        worker.deleteLater()
        self.table.blockSignals(True)
        self.table.clear()
        self.table.addItems(result["tables"])
        if result["table"]:
            self.table.setCurrentText(result["table"])
        self.table.blockSignals(False)
        for combo, label in ((self.column, "所有字段"), (self.order, "默认排序")):
            selected = combo.currentData()
            combo.clear()
            combo.addItem(label, None)
            for name in result["columns"]:
                combo.addItem(name, name)
            combo.setCurrentIndex(max(0, combo.findData(selected)))
        self.rows = result["rows"]
        self.columns = result["columns"]
        self.fill_table(self.data, result["columns"], self.rows)
        self.fill_table(self.schema, ["序号", "字段", "类型", "非空", "默认值", "主键序号"], result["schema"])
        self.detail.clear()
        self.offset, self.total = result["offset"], result["total"]
        self.status.setText(
            f"共 {self.total:,} 条 · 第 {self.offset + 1 if self.total else 0}–{self.offset + len(self.rows)} 条"
        )
        self.previous.setEnabled(self.offset > 0)
        self.next.setEnabled(self.offset + len(self.rows) < self.total)

    @staticmethod
    def display(value):
        if value is None:
            return "NULL"
        if isinstance(value, bytes):
            return f"[BLOB · {len(value)} 字节]"
        return str(value)

    def fill_table(self, table, columns, rows):
        table.clear()
        table.setColumnCount(len(columns))
        table.setHorizontalHeaderLabels(columns)
        table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            for col_index, value in enumerate(row):
                table.setItem(row_index, col_index, QTableWidgetItem(self.display(value)[:500]))

    def show_detail(self, row, column):
        value = self.rows[row][column]
        self.detail.setPlainText(value.hex() if isinstance(value, bytes) else self.display(value))
        if self.columns[column] == "result_detail":
            self.inspector.set_documents([("选中明细", value)])
            self.detail_tabs.setCurrentIndex(0)
        else:
            self.detail_tabs.setCurrentIndex(1)

    def inspect_selected(self):
        if self.worker is not None:
            return
        index = self.data.currentRow()
        self.detail_tabs.setCurrentIndex(0)
        if not 0 <= index < len(self.rows):
            self.inspector.clear("请先单击选择一条记录，或使用“粘贴 JSON”。")
            return
        record = dict(zip(self.columns, self.rows[index]))
        if "result_detail" in record:
            self.inspector.set_documents([(f"明细 #{record.get('id', index + 1)}", record["result_detail"])])
            return
        result_id = record.get("id") if self.table.currentText() == "inspection_result" else record.get("result_id")
        if result_id is None:
            self.inspector.clear("该表没有 result_detail 或可关联的 result_id；可使用“粘贴 JSON”。")
            return
        self.controls.setEnabled(False)
        self.data.setEnabled(False)
        self.previous.setEnabled(False)
        self.next.setEnabled(False)
        self.inspector.clear(f"正在读取检测记录 #{result_id} 的所有明细…")
        self.worker = DatabaseWorker(
            dict(path=self.path.text(), result_id=result_id), self, reader=read_inspection_details
        )
        self.worker.finished.connect(self.inspection_loaded)
        self.worker.start()

    def inspection_loaded(self):
        worker = self.worker
        self.worker = None
        self.controls.setEnabled(True)
        self.data.setEnabled(True)
        self.previous.setEnabled(self.offset > 0)
        self.next.setEnabled(self.offset + len(self.rows) < self.total)
        if worker.error:
            self.inspector.clear(f"读取检查项失败：{worker.error}")
        else:
            self.inspector.set_documents([(f"明细 #{key}", raw) for key, raw in worker.result])
        worker.deleteLater()

    def closeEvent(self, event):
        if self.worker is not None:
            self.worker.wait()
        super().closeEvent(event)

    def protect_window_close(self, window):
        window.installEventFilter(self)

    def eventFilter(self, watched, event):
        if event.type() == QEvent.Type.Close:
            self.wait_for_query()
        return super().eventFilter(watched, event)

    def wait_for_query(self):
        if self.worker is not None:
            self.worker.wait()
