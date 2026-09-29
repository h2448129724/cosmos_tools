"""PySide6 page for previewing and exporting NG images from a Cosmos DB.

The database/export implementation deliberately lives in :mod:`db_ng_export`.
This module is only the desktop shell around that read-only planner and its
cooperative copy operation.  Keeping the workers here small also makes the
page usable from the toolbox shell and as a standalone module.
"""
from __future__ import annotations

import json
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

from PySide6.QtCore import QEvent, QThread, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QScrollArea, QTabWidget, QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
)

from .paths import COSMOS_ROOT


def _new_control():
    return SimpleNamespace(paused=threading.Event(), stopped=threading.Event())


class DbNgWorker(QThread):
    """Run one planner or copy operation outside the Qt GUI thread.

    ``options`` is the exact keyword dictionary accepted by ``scan`` for a
    scan worker.  For a copy worker it contains ``plan`` and ``output``.  The
    shape mirrors the other toolbox workers, which is useful to callers that
    want to exercise a worker without creating a page.
    """

    progress = Signal(dict)
    result = Signal(dict)
    error = Signal(str)

    def __init__(self, options: dict, control=None, scan_only: bool = True, parent=None):
        super().__init__(parent)
        self.options = dict(options)
        self.control = control or _new_control()
        self.scan_only = scan_only

    def run(self):  # pragma: no cover - Qt invokes this in a real thread
        try:
            from . import db_ng_export

            if self.scan_only:
                plan = db_ng_export.scan(**self.options)
                self.result.emit(plan)
            else:
                report = db_ng_export.copy_plan(
                    self.options["plan"],
                    self.options["output"],
                    control=self.control,
                    on_progress=self.progress.emit,
                )
                self.result.emit(report)
        except Exception as exc:
            self.error.emit(f"{type(exc).__name__}: {exc}")


class DbNgPage(QWidget):
    """Scan a database, review an immutable plan, and copy its images."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setProperty("ownsPageHeader", True)
        self.setWindowTitle("COSMOS NG 图片提取")
        self.worker: DbNgWorker | None = None
        self.control = _new_control()
        self.plan: dict | None = None
        self._plan_options: dict | None = None
        self._closing_window = None
        self._initializing = True

        layout = QVBoxLayout(self)
        title = QLabel("COSMOS NG 图片提取")
        title.setStyleSheet("font-size: 22px; font-weight: 600;")
        layout.addWidget(title)
        explanation = QLabel(
            "先选择输出目录并扫描生成只读复制计划，再按同一计划导出。数据库不会被修改；NG 是产品级判定，"
            "关联图片不代表每张图片都是 NG。筛选条件或输出目录变化后需要重新扫描。"
        )
        explanation.setWordWrap(True)
        layout.addWidget(explanation)

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs, 1)

        self.settings = QGroupBox("输入与筛选")
        form = QFormLayout(self.settings)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        form.setVerticalSpacing(10)

        self.database = QLineEdit(str(COSMOS_ROOT / "cosmos.db"))
        self.database_path = self.database
        self.database.setPlaceholderText("SQLite 数据库路径")
        form.addRow("数据库", self._file_row(self.database, self._browse_database))

        self.output = QLineEdit()
        self.output_dir = self.output
        self.output.setPlaceholderText("复制目标文件夹")
        form.addRow("输出目录", self._folder_row(self.output))

        self.old_root = QLineEdit()
        self.old_root.setPlaceholderText("可选：数据库中的旧路径前缀")
        form.addRow("旧路径前缀", self._folder_row(self.old_root, existing_only=False))
        self.new_root = QLineEdit()
        self.new_root.setPlaceholderText("可选：映射到机器上的新路径前缀")
        form.addRow("新路径前缀", self._folder_row(self.new_root, existing_only=False))

        self.project = QLineEdit("CAB-F")
        self.project.setPlaceholderText("项目（默认 CAB-F）")
        form.addRow("项目", self.project)
        self.product = QLineEdit()
        self.product.setPlaceholderText("留空表示全部产品")
        form.addRow("产品", self.product)
        self.start = QLineEdit()
        self.start.setPlaceholderText("ISO 时间；留空表示不设下限")
        form.addRow("开始时间", self.start)
        self.end = QLineEdit()
        self.end.setPlaceholderText("ISO 时间；结束时间不包含此时刻")
        form.addRow("结束时间（不含）", self.end)

        self.exclude_misjudged = QCheckBox("排除人工改判记录")
        self.exclude_misjudged_checkbox = self.exclude_misjudged
        self.exclude_misjudged.setToolTip("只导出原始判定为 NG 的产品；默认包含人工改判记录。")
        form.addRow("判定", self.exclude_misjudged)

        kinds_widget = QWidget()
        kinds_layout = QHBoxLayout(kinds_widget)
        kinds_layout.setContentsMargins(0, 0, 0, 0)
        self.raw = QCheckBox("原图")
        self.raw.setChecked(True)
        self.raw_checkbox = self.raw
        self.result = QCheckBox("结果图")
        self.result_checkbox = self.result
        kinds_layout.addWidget(self.raw)
        kinds_layout.addWidget(self.result)
        kinds_layout.addStretch(1)
        form.addRow("图片类型", kinds_widget)
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.addWidget(self.settings)
        content_layout.addStretch()
        self.settings_scroll = QScrollArea()
        self.settings_scroll.setWidgetResizable(True)
        self.settings_scroll.setWidget(content)
        self.tabs.addTab(self.settings_scroll, '输入与筛选')

        from .ui.primitives import ActionBar
        actions = ActionBar()
        self.scan_button = QPushButton("扫描预览")
        self.copy_button = QPushButton("按计划复制")
        self.pause_button = QPushButton("暂停")
        self.stop_button = QPushButton("停止")
        self.open_button = QPushButton("打开输出目录")
        for button in (self.scan_button, self.copy_button, self.pause_button, self.stop_button, self.open_button):
            actions.add_widget(button)
        layout.addWidget(actions)
        self.scan_button.clicked.connect(self.scan)
        self.copy_button.clicked.connect(self.copy)
        self.pause_button.clicked.connect(self._toggle_pause)
        self.stop_button.clicked.connect(self._stop)
        self.open_button.clicked.connect(self._open_output)

        self.status = QLabel("请先扫描数据库，确认数量和容量后再复制。")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress = self.progress_bar  # convenient compatibility alias
        layout.addWidget(self.progress_bar)

        self.preview = QPlainTextEdit()
        self.preview.setReadOnly(True)
        self.preview.setPlaceholderText("扫描计划、缺失路径和复制结果会显示在这里。")
        self.preview.setMaximumBlockCount(2000)
        self.tabs.addTab(self.preview, '计划与日志')
        self.files_table = QTableWidget(0, 3)
        self.files_table.setHorizontalHeaderLabels(['状态', '原图路径', '容量'])
        self.files_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.files_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.tabs.addTab(self.files_table, '文件清单')
        self.scan_preview = self.preview
        self.log = self.preview

        for widget in (
            self.database,
            self.output,
            self.old_root,
            self.new_root,
            self.project,
            self.product,
            self.start,
            self.end,
        ):
            widget.textChanged.connect(self._options_changed)
        for widget in (self.exclude_misjudged, self.raw, self.result):
            widget.toggled.connect(self._options_changed)
        self._initializing = False
        self._set_busy(False)
        self._update_copy_enabled()

    @staticmethod
    def _file_row(edit, callback):
        row = QWidget()
        box = QHBoxLayout(row)
        box.setContentsMargins(0, 0, 0, 0)
        box.addWidget(edit, 1)
        browse = QPushButton("浏览…")
        browse.clicked.connect(callback)
        box.addWidget(browse)
        return row

    def _folder_row(self, edit, existing_only=True):
        row = QWidget()
        box = QHBoxLayout(row)
        box.setContentsMargins(0, 0, 0, 0)
        box.addWidget(edit, 1)
        browse = QPushButton("浏览…")
        browse.clicked.connect(lambda: self._browse_folder(edit, existing_only))
        box.addWidget(browse)
        return row

    def _browse_database(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择 SQLite 数据库",
            self.database.text() or str(COSMOS_ROOT),
            "SQLite 数据库 (*.db *.sqlite *.sqlite3);;所有文件 (*)",
        )
        if path:
            self.database.setText(path)

    def _browse_folder(self, edit, existing_only=True):
        if existing_only:
            value = QFileDialog.getExistingDirectory(self, "选择文件夹", edit.text() or str(COSMOS_ROOT))
        else:
            value = QFileDialog.getExistingDirectory(self, "选择路径前缀", edit.text() or str(COSMOS_ROOT))
        if value:
            edit.setText(value)

    def _options(self):
        kinds = tuple(kind for kind, check in (("raw", self.raw), ("result", self.result)) if check.isChecked())
        return {
            "database": self.database.text().strip(),
            "project": self.project.text().strip() or "CAB-F",
            "product": self.product.text().strip(),
            "start": self.start.text().strip(),
            "end": self.end.text().strip(),
            "exclude_misjudged": self.exclude_misjudged.isChecked(),
            "kinds": kinds,
            "old_root": self.old_root.text().strip(),
            "new_root": self.new_root.text().strip(),
        }

    def _signature(self):
        """Scan filters plus destination, both of which matter to this page."""
        return {"scan": self._options(), "output": self.output.text().strip()}

    def _options_changed(self, *_):
        if self._initializing:
            return
        if self.plan is not None:
            self.plan = None
            self._plan_options = None
            self.preview.appendPlainText("筛选条件已改变，旧扫描计划已失效；请重新扫描。")
            self.status.setText("筛选条件已改变，请重新扫描后再复制。")
        self._update_copy_enabled()

    def _update_copy_enabled(self):
        valid = self.plan is not None and self._plan_options == self._signature()
        self.copy_button.setEnabled(valid and not (self.worker and self.worker.isRunning()))

    def _validate_scan(self):
        database = Path(self.database.text().strip()).expanduser()
        if not database.is_file():
            QMessageBox.warning(self, "数据库", f"数据库不存在：{database}")
            return None
        if not self.raw.isChecked() and not self.result.isChecked():
            QMessageBox.warning(self, "图片类型", "至少勾选原图或结果图。")
            return None
        options = self._options()
        options["database"] = str(database.resolve())
        return options

    def scan(self):
        if self.worker and self.worker.isRunning():
            return
        options = self._validate_scan()
        if options is None:
            return
        self.plan = None
        self._plan_options = None
        self.control = _new_control()
        self.worker = DbNgWorker(options, self.control, scan_only=True, parent=self)
        self.worker.result.connect(self._on_scan_result)
        self.worker.progress.connect(self._on_progress)
        self.worker.error.connect(self._on_error)
        self.worker.finished.connect(self._finished)
        self._set_busy(True, scan_only=True)
        self.progress_bar.setRange(0, 0)
        self.status.setText("正在只读扫描数据库…")
        self.preview.clear()
        from .worker_task_bridge import bind_worker_task
        bind_worker_task(self, '数据库 NG：扫描', 'cabf.db_ng_export')
        self.worker.start()

    # Small method aliases make this page convenient for shell adapters and
    # preserve the verb-style entry points used by older toolbox pages.
    def _start_scan(self):
        self.scan()

    def copy(self):
        if self.worker and self.worker.isRunning():
            return
        if self.plan is None or self._plan_options != self._signature():
            QMessageBox.warning(self, "扫描计划", "请先扫描；筛选条件变化后必须重新扫描。")
            self._update_copy_enabled()
            return
        output = self.output.text().strip()
        if not output:
            QMessageBox.warning(self, "输出目录", "请选择输出目录。")
            return
        self.control = _new_control()
        options = {"plan": self.plan, "output": str(Path(output).expanduser())}
        self.worker = DbNgWorker(options, self.control, scan_only=False, parent=self)
        self.worker.result.connect(self._on_copy_result)
        self.worker.progress.connect(self._on_progress)
        self.worker.error.connect(self._on_error)
        self.worker.finished.connect(self._finished)
        self._set_busy(True, scan_only=False)
        self.progress_bar.setRange(0, 0)
        self.status.setText("正在复制计划中的图片…")
        from .worker_task_bridge import bind_worker_task
        bind_worker_task(self, '数据库 NG：复制', 'cabf.db_ng_export')
        self.worker.start()

    def _start_copy(self):
        self.copy()

    def _on_scan_result(self, plan):
        self.plan = plan
        self._plan_options = self._signature()
        summary = plan.get("summary", {}) if isinstance(plan, dict) else {}
        self.status.setText(self._summary_text(summary, scanned=True))
        self.preview.setPlainText(self._plan_text(plan))
        files = plan.get('files', [])
        self.files_table.setRowCount(len(files))
        for row, item in enumerate(files):
            for column, value in enumerate((item.get('status', ''), item.get('source', ''),
                                             self._format_bytes(item.get('size', 0)))):
                cell = QTableWidgetItem(str(value))
                cell.setToolTip(str(value))
                self.files_table.setItem(row, column, cell)
        self.tabs.setCurrentWidget(self.files_table)
        self._update_copy_enabled()

    def _on_copy_result(self, report):
        report = report if isinstance(report, dict) else {}
        scan_summary = report.get("scan", {})
        stopped = bool(report.get("stopped")) or self.control.stopped.is_set()
        copied = report.get("copied", 0)
        skipped = report.get("skipped", 0)
        failed = report.get("failed", 0)
        missing = report.get("missing", 0)
        state = "已停止，已保留已完成文件；可按同一计划再次复制。" if stopped else "复制完成。"
        self.status.setText(
            f"{state} 已复制 {copied:,}，跳过 {skipped:,}，失败 {failed:,}，缺失 {missing:,}。"
            + (" " + self._summary_text(scan_summary, scanned=False) if scan_summary else "")
        )
        self.preview.appendPlainText("\n复制报告：\n" + json.dumps(report, ensure_ascii=False, indent=2, default=str))
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0 if stopped else 100)

    def _summary_text(self, summary, scanned=False):
        prefix = "扫描完成" if scanned else "复制统计"
        ng = summary.get("ng_records", 0)
        images = summary.get("image_records", summary.get("files", 0))
        ready = summary.get("ready_files", 0)
        missing = summary.get("missing_files", 0)
        capacity = self._format_bytes(summary.get("bytes", 0))
        caveat = "NG 为产品级判定，关联图片不代表逐张 NG。"
        return f"{prefix}：{ng:,} 个 NG 产品，{images:,} 个图片记录；可复制 {ready:,}，缺失 {missing:,}，容量 {capacity}。{caveat}"

    @staticmethod
    def _format_bytes(value):
        try:
            value = float(value or 0)
        except (TypeError, ValueError):
            return "未知"
        units = ("B", "KiB", "MiB", "GiB", "TiB")
        for unit in units:
            if abs(value) < 1024 or unit == units[-1]:
                return f"{value:.2f} {unit}"
            value /= 1024
        return "未知"

    def _plan_text(self, plan):
        summary = plan.get("summary", {}) if isinstance(plan, dict) else {}
        missing_records = plan.get("missing_records", []) if isinstance(plan, dict) else []
        lines = [self._summary_text(summary, scanned=True), "", "扫描计划（复制将严格使用此计划）："]
        for item in plan.get("files", []) if isinstance(plan, dict) else []:
            source = item.get("source", "")
            status = item.get("status", "")
            size = item.get("size", 0)
            lines.append(f"[{status}] {source}" + (f" ({self._format_bytes(size)})" if size else ""))
        if missing_records:
            lines.extend(["", "无图片关联记录（产品仍为 NG）："])
            lines.extend(str(item) for item in missing_records)
        return "\n".join(lines)

    def _on_progress(self, data):
        if not isinstance(data, dict):
            self.preview.appendPlainText(str(data))
            return
        total = data.get("total")
        current = data.get("current", data.get("done", data.get("processed")))
        if isinstance(total, int) and total > 0 and isinstance(current, int):
            self.progress_bar.setRange(0, total)
            self.progress_bar.setValue(current)
        message = data.get("message") or json.dumps(data, ensure_ascii=False, default=str)
        self.preview.appendPlainText(str(message))
        if not self.control.paused.is_set() and not self.control.stopped.is_set():
            self.status.setText(str(message))

    def _on_error(self, message):
        self.progress_bar.setRange(0, 100)
        self.status.setText("任务异常，请查看详情。")
        self.preview.appendPlainText("错误：" + str(message))
        self.tabs.setCurrentWidget(self.preview)

    def _set_busy(self, busy, scan_only=False):
        self.settings.setEnabled(not busy)
        self.scan_button.setEnabled(not busy)
        self.copy_button.setEnabled(not busy and self.plan is not None and self._plan_options == self._signature())
        self.pause_button.setEnabled(busy and not scan_only)
        self.stop_button.setEnabled(busy and not scan_only)
        self.open_button.setEnabled(not busy)

    def _finished(self):
        worker = self.worker
        self._set_busy(False)
        self.pause_button.setText("暂停")
        if worker is not None:
            worker.deleteLater()
        self.worker = None
        self._update_copy_enabled()

    def _toggle_pause(self):
        if not self.worker or not self.worker.isRunning():
            return
        if self.control.paused.is_set():
            self.control.paused.clear()
            self.pause_button.setText("暂停")
            self.status.setText("继续处理中…")
        else:
            self.control.paused.set()
            self.pause_button.setText("继续")
            self.status.setText("已请求暂停，将在安全处理点等待。")
        if getattr(self, '_task_bridge', None):
            self._task_bridge.note(self.status.text())

    def _stop(self):
        bridge = getattr(self, '_task_bridge', None)
        if bridge and bridge.center.cancel(bridge.task_id):
            return
        self.control.stopped.set()
        self.control.paused.clear()
        self.pause_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self.status.setText("已请求停止，正在完成当前安全处理点…")

    def _open_output(self):
        value = self.output.text().strip()
        if value and Path(value).is_dir():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(value).resolve())))

    def protect_window_close(self, window):
        self._closing_window = window
        window.installEventFilter(self)

    def eventFilter(self, watched, event):
        if watched is self._closing_window and event.type() == QEvent.Type.Close and self.worker and self.worker.isRunning():
            self._stop()
            event.ignore()
            self.preview.appendPlainText("任务停止后请再次关闭窗口。")
            return True
        return super().eventFilter(watched, event)

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self._stop()
            event.ignore()
        else:
            event.accept()


# Names used by a few toolbox shells and by older page-factory conventions.
DbNgExportPage = DbNgPage
DatabaseNgPage = DbNgPage
DatabaseNgWorker = DbNgWorker


def main():
    app = QApplication.instance() or QApplication(sys.argv)
    page = DbNgPage()
    page.resize(1080, 780)
    page.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
