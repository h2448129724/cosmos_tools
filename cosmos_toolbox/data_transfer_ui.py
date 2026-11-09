"""Native workbench page for SQLite → Excel → NG originals."""
from __future__ import annotations

import threading
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from PySide6.QtCore import QEvent, QThread, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QFileDialog, QFormLayout, QHBoxLayout,
    QHeaderView, QLabel, QLineEdit, QMessageBox, QPlainTextEdit, QProgressBar,
    QPushButton, QScrollArea, QTableWidget, QTableWidgetItem, QTabWidget,
    QVBoxLayout, QWidget,
)

from .paths import COSMOS_ROOT
from .task_center import TaskStatus


STATUS_LABELS = {
    'copied': '复制成功', 'ready': '可复制', 'existing_same': '已存在且相同',
    'duplicate': '重复来源', 'missing': '源图缺失', 'conflict': '同名冲突',
    'copy_error': '复制/访问失败', 'no_source_path': '无原图路径',
    'no_image_row': '无关联图片行', 'pending': '未处理',
}


class DataTransferWorker(QThread):
    progress = Signal(dict)
    result = Signal(dict)
    error = Signal(str)

    def __init__(self, operation, options, parent=None):
        super().__init__(parent)
        self.operation = operation
        self.options = dict(options)
        self.stopped = threading.Event()

    def run(self):
        try:
            if self.operation == 'export':
                from .database_excel import export_database

                reports = export_database(**self.options, on_progress=self.progress.emit)
                self.result.emit({'operation': 'export', 'reports': reports,
                                  'output': str(Path(self.options['output']).expanduser().resolve())})
                return

            from .excel_ng_images import FAILED_STATUSES, collect_images, process_images, write_report

            options = dict(self.options)
            dry_run = options.pop('dry_run')
            report = options.pop('report')
            self.progress.emit({'message': '正在读取 Excel 并关联 NG 记录…'})
            records, count = collect_images(**options)
            if not dry_run and not report:
                report = Path(options['output']) / ('_copy_ng_report_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '.csv')
            if report:
                report = Path(report).expanduser().resolve()
                if report.suffix.lower() != '.csv' or report.exists():
                    raise ValueError('清单路径必须是尚不存在的 .csv 文件')
                if any(str(report).lower() in {row['source_path'].lower(), row['destination_path'].lower()}
                       for row in records):
                    raise ValueError('清单路径不能与源图或目标图片相同')
            summary = process_images(records, dry_run, on_progress=self.progress.emit,
                                     should_stop=self.stopped.is_set)
            if report:
                write_report(records, report)
            self.result.emit({
                'operation': self.operation, 'ng_count': count, 'total': len(records),
                'summary': dict(summary), 'failed': sum(summary[key] for key in FAILED_STATUSES),
                'stopped': self.stopped.is_set(), 'preview': records[:200],
                'report': str(report) if report else '', 'output': str(options['output']),
            })
        except Exception as exc:
            self.error.emit(f'{type(exc).__name__}: {exc}')


class DataTransferPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setProperty('ownsPageHeader', True)
        self.worker = None
        self.task_center = None
        self._task_id = None
        self._closing_window = None
        self._result = None
        self._error = None
        self._last_output = ''
        self.setWindowTitle('数据库导出与 Excel NG 提取')
        layout = QVBoxLayout(self)
        title = QLabel(self.windowTitle())
        title.setStyleSheet('font-size: 22px; font-weight: 600;')
        layout.addWidget(title)
        hint = QLabel('先将数据库导出为 Excel，再在能访问原图的电脑上提取 NG 图片。也可直接使用已有 Excel。')
        hint.setWordWrap(True)
        layout.addWidget(hint)
        self.header_widgets = (title, hint)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        self.database = QLineEdit(str(COSMOS_ROOT / 'cosmos.db'))
        self.excel_output = QLineEdit(str(COSMOS_ROOT / 'cosmos.xlsx'))
        self.tables = QPlainTextEdit()
        self.tables.setMaximumHeight(70)
        self.tables.setPlaceholderText('留空导出全部用户表；如需指定，每行一个完整表名')
        self.overwrite = QCheckBox('允许覆盖已有 Excel（默认保留已有文件）')
        self.export_button = QPushButton('导出 Excel')
        self.export_button.clicked.connect(self.start_export)
        self.export_settings, form = self._form_tab('1. 数据库 → Excel')
        form.addRow('数据库', self._path_row(self.database, 'database'))
        form.addRow('输出 Excel', self._path_row(self.excel_output, 'save_excel'))
        form.addRow('指定表（可选）', self.tables)
        form.addRow('', self.overwrite)
        note = QLabel('只读导出，每张表一个工作表。保留关联 ID、JSON 和路径；长文本自动拆列，不嵌入图片。')
        note.setWordWrap(True)
        form.addRow(note)
        form.addRow(self.export_button)

        self.excel = QLineEdit()
        self.excel.setPlaceholderText('可选择已有 Excel，或使用上一步的导出结果')
        self.output = QLineEdit()
        self.source_drive = QLineEdit('E:')
        self.source_root = QLineEdit()
        self.source_root.setPlaceholderText('可选，例如 F:/ 或 X:/machine_E；保留原盘符下的相对路径')
        self.only_ng_faces = QCheckBox('只提取 ng_error 指向的 NG 面')
        self.start_date = QLineEdit()
        self.end_date = QLineEdit()
        for edit in (self.start_date, self.end_date):
            edit.setPlaceholderText('可留空；YYYY-MM-DD 或 YYYY-MM-DD HH:MM:SS')
        self.report = QLineEdit()
        self.report.setPlaceholderText('可选：尚不存在的 .csv；复制时默认在输出目录生成清单')
        self.check_button = QPushButton('仅检查（不复制）')
        self.copy_button = QPushButton('复制 NG 原图')
        self.check_button.clicked.connect(lambda: self.start_images(True))
        self.copy_button.clicked.connect(lambda: self.start_images(False))
        self.image_settings, form = self._form_tab('2. Excel → NG 原图')
        form.addRow('输入 Excel', self._path_row(self.excel, 'excel'))
        form.addRow('图片输出目录', self._path_row(self.output, 'directory'))
        drive_row = QWidget()
        drive_layout = QHBoxLayout(drive_row)
        drive_layout.setContentsMargins(0, 0, 0, 0)
        self.source_drive.setMaximumWidth(80)
        drive_layout.addWidget(self.source_drive)
        drive_layout.addWidget(QLabel('映射根目录'))
        drive_layout.addWidget(self._path_row(self.source_root, 'directory'), 1)
        form.addRow('原图盘符', drive_row)
        form.addRow('开始时间（含）', self.start_date)
        form.addRow('结束时间（含）', self.end_date)
        form.addRow('', self.only_ng_faces)
        form.addRow('CSV 清单', self._path_row(self.report, 'report'))
        note = QLabel('按检测时间筛选；结束日期包含当天。默认提取 NG 产品的全部面，包含误判记录。'
                      '优先 filename，其次 raw_image；不使用结果图。同名异源加目录前缀，已有异内容文件不覆盖。')
        note.setWordWrap(True)
        form.addRow(note)
        actions = QWidget()
        buttons = QHBoxLayout(actions)
        buttons.setContentsMargins(0, 0, 0, 0)
        buttons.addWidget(self.check_button)
        buttons.addWidget(self.copy_button)
        buttons.addStretch()
        form.addRow(actions)

        actions = QHBoxLayout()
        self.stop_button = QPushButton('停止图片任务')
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self._stop)
        self.open_button = QPushButton('打开输出位置')
        self.open_button.setEnabled(False)
        self.open_button.clicked.connect(self._open_output)
        actions.addWidget(self.stop_button)
        actions.addWidget(self.open_button)
        actions.addStretch()
        layout.addLayout(actions)
        self.status = QLabel('请选择数据库导出，或切换到第二步选择已有 Excel。')
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)
        self.results = QTabWidget()
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(2000)
        self.results.addTab(self.log, '任务日志')
        self.preview = QTableWidget(0, 4)
        self.preview.setHorizontalHeaderLabels(['状态', '源图片', '目标图片', '说明'])
        self.preview.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.preview.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.results.addTab(self.preview, '图片清单（前 200 条）')
        layout.addWidget(self.results, 1)

    def _form_tab(self, title):
        content = QWidget()
        form = QFormLayout(content)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content)
        scroll.setMinimumHeight(220)
        self.tabs.addTab(scroll, title)
        return content, form

    def _path_row(self, edit, kind):
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(edit, 1)
        button = QPushButton('浏览…')
        button.clicked.connect(lambda: self._browse(edit, kind))
        layout.addWidget(button)
        return row

    def _browse(self, edit, kind):
        initial = edit.text().strip() or str(COSMOS_ROOT)
        filters = {'database': 'SQLite (*.db *.sqlite *.sqlite3);;所有文件 (*)',
                   'excel': 'Excel (*.xlsx)', 'save_excel': 'Excel (*.xlsx)', 'report': 'CSV (*.csv)'}
        if kind == 'directory':
            value = QFileDialog.getExistingDirectory(self, '选择目录', initial)
        elif kind in {'save_excel', 'report'}:
            value, _ = QFileDialog.getSaveFileName(self, '选择输出路径', initial, filters[kind])
        else:
            value, _ = QFileDialog.getOpenFileName(self, '选择输入文件', initial, filters[kind])
        if value:
            edit.setText(value)

    def start_export(self):
        if not self.database.text().strip() or not self.excel_output.text().strip():
            QMessageBox.warning(self, '路径缺失', '请选择数据库和输出 Excel 路径。')
            return
        self._start('export', {
            'database': self.database.text().strip(), 'output': self.excel_output.text().strip(),
            'tables': [line.strip() for line in self.tables.toPlainText().splitlines() if line.strip()] or None,
            'overwrite': self.overwrite.isChecked(),
        })

    def start_images(self, dry_run):
        if not self.excel.text().strip() or not self.output.text().strip():
            QMessageBox.warning(self, '路径缺失', '请选择输入 Excel 和图片输出目录。')
            return
        self._start('check' if dry_run else 'copy', {
            'excel': self.excel.text().strip(), 'output': self.output.text().strip(),
            'source_drive': self.source_drive.text().strip(), 'source_root': self.source_root.text().strip() or None,
            'only_ng_faces': self.only_ng_faces.isChecked(),
            'start_date': self.start_date.text().strip() or None, 'end_date': self.end_date.text().strip() or None,
            'dry_run': dry_run, 'report': self.report.text().strip() or None,
        })

    def _start(self, operation, options):
        if self.worker is not None:
            return
        self._result, self._error = None, None
        self._last_output = ''
        self.log.clear()
        self.preview.setRowCount(0)
        self.results.setCurrentWidget(self.log)
        self.worker = DataTransferWorker(operation, options, self)
        self.worker.progress.connect(self._on_progress)
        self.worker.result.connect(self._on_result)
        self.worker.error.connect(self._on_error)
        self.worker.finished.connect(self._finished)
        self.export_settings.setEnabled(False)
        self.image_settings.setEnabled(False)
        self.stop_button.setEnabled(operation != 'export')
        self.open_button.setEnabled(False)
        self.progress_bar.setRange(0, 0)
        self.status.setText('正在导出数据库，请等待完成后关闭窗口。' if operation == 'export' else '正在处理 Excel…')
        self._task_id = None
        if self.task_center is not None:
            self._task_id = 'data-transfer-' + uuid4().hex[:12]
            title = {'export': '数据库导出 Excel', 'check': 'Excel NG：检查', 'copy': 'Excel NG：复制'}[operation]
            self.task_center.start(self._task_id, title, 'data.ng_extract', options['output'] if operation != 'check' else '',
                                   cancel=self._stop if operation != 'export' else None)
        self.worker.start()

    def _append_log(self, message):
        self.log.appendPlainText(message)
        if self.task_center is not None and self._task_id:
            self.task_center.log(self._task_id, message)

    def _on_progress(self, fact):
        if fact.get('total', 0) > 0:
            self.progress_bar.setRange(0, fact['total'])
            self.progress_bar.setValue(fact['current'])
            if self.task_center is not None and self._task_id:
                self.task_center.progress(self._task_id, fact['current'], fact['total'])
        self._append_log(fact['message'])

    def _on_result(self, result):
        self._result = result
        if result['operation'] == 'export':
            self.excel.setText(result['output'])
            if not self.output.text().strip():
                self.output.setText(str(Path(result['output']).parent / 'NG_images'))
            for table, sheet, count, extra in result['reports']:
                self._append_log(f'{table} → {sheet}：{count} 条；长文本增加 {extra} 列')
            self.status.setText('Excel 导出完成，已将路径填入第二步。')
            self.tabs.setCurrentIndex(1)
        else:
            summary = '；'.join(f'{STATUS_LABELS.get(key, key)}：{value}' for key, value in result['summary'].items())
            state = '已停止' if result['stopped'] else ('检查完成' if result['operation'] == 'check' else '复制完成')
            self.status.setText(f'{state}；NG 检测记录：{result["ng_count"]}；{summary or "无关联图片"}')
            rows = result['preview']
            self.preview.setRowCount(len(rows))
            for index, row in enumerate(rows):
                values = (STATUS_LABELS.get(row['status'], row['status']), row['source_path'],
                          row['destination_path'], row['message'])
                for column, value in enumerate(values):
                    item = QTableWidgetItem(value)
                    item.setToolTip(value)
                    self.preview.setItem(index, column, item)
            if result['report']:
                self._append_log('CSV 清单：' + result['report'])
            self._append_log(f'共 {result["total"]} 条清单记录，页面最多预览前 200 条。')
        self._last_output = result['output'] if result['operation'] != 'check' else result.get('report', '')
        self._append_log(self.status.text())
        if self._last_output:
            self._append_log('输出：' + self._last_output)

    def _on_error(self, message):
        self._error = message
        self.status.setText('任务失败，请查看日志。')
        self._append_log(message)

    def _finished(self):
        result = self._result or {}
        failed = bool(self._error or self._result is None or result.get('failed'))
        state = TaskStatus.FAILED if failed else (TaskStatus.STOPPED if result.get('stopped') else TaskStatus.SUCCESS)
        if self.task_center is not None and self._task_id:
            output = result.get('output', '') if result.get('operation') != 'check' else result.get('report', '')
            self.task_center.finish(self._task_id, state, output)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(100 if state == TaskStatus.SUCCESS else 0)
        self.export_settings.setEnabled(True)
        self.image_settings.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.open_button.setEnabled(bool(self._last_output))
        self.worker.deleteLater()
        self.worker = None

    def _stop(self):
        if self.worker is not None and self.worker.operation != 'export':
            if self.task_center is not None and self._task_id and self.task_center.cancel(self._task_id):
                return
            self.worker.stopped.set()
            self.stop_button.setEnabled(False)
            self.status.setText('已请求停止，正在完成当前读取或复制；已处理文件会保留。')

    def _open_output(self):
        if self._last_output:
            path = Path(self._last_output).expanduser().resolve()
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path if path.is_dir() else path.parent)))

    def protect_window_close(self, window):
        self._closing_window = window
        window.installEventFilter(self)

    def eventFilter(self, watched, event):
        if watched is self._closing_window and event.type() == QEvent.Type.Close and self.worker is not None:
            self._stop()
            self._append_log('任务结束后请再次关闭窗口。')
            event.ignore()
            return True
        return super().eventFilter(watched, event)

    def closeEvent(self, event):
        if self.worker is not None:
            self._stop()
            event.ignore()
        else:
            event.accept()
