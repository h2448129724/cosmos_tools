"""Native batch PNG compression page, with cooperative worker lifecycle."""
from __future__ import annotations

import json
import os
import threading
from types import SimpleNamespace

from PySide6.QtCore import QEvent, QThread, Signal
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QPlainTextEdit, QPushButton,
                              QCheckBox, QComboBox, QLabel, QFileDialog, QMessageBox,
                              QProgressBar, QScrollArea, QSpinBox, QTreeView, QListView, QAbstractItemView)

from .ui.primitives import ActionBar, PageHeader
from .worker_task_bridge import WorkerTaskBridge
from . import png_compress


class PngWorker(QThread):
    progress = Signal(dict)
    result = Signal(dict)
    error = Signal(str)

    def __init__(self, options, scan_only, parent=None):
        super().__init__(parent)
        self.options, self.scan_only = options, scan_only
        self.control = SimpleNamespace(paused=threading.Event(), stopped=threading.Event())

    def run(self):
        report = dict(count=0, processed=0, failed=0, compressed=0, converted=0, saved=0, potential_saved=0, stopped=False)
        try:
            files = png_compress.scan_folders(self.options['sources'], self.options['recursive'], self.control)
            report.update(count=len(files), bytes=sum(p.stat().st_size for p in files))
            if not self.scan_only:
                for index, path in enumerate(files):
                    png_compress.checkpoint(self.control)
                    self.progress.emit(dict(current=index, total=len(files), message=f'处理中：{path}'))
                    try:
                        if self.options.get('mode') == 'jpg':
                            result = png_compress.convert_jpeg(path, quality=self.options['quality'],
                                                              delete_source=self.options['delete_source'], control=self.control)
                        else:
                            result = png_compress.compress_file(path, level=self.options['level'], control=self.control,
                                                               optimize=self.options.get('optimize', True))
                        report['compressed'] += result['status'] == 'compressed'
                        report['converted'] += result['status'] == 'converted'
                        report['potential_saved'] += result.get('potential_saved', 0)
                        report['saved'] += result['saved']
                    except png_compress.Cancelled:
                        raise
                    except Exception as exc:
                        report['failed'] += 1
                        result = dict(path=str(path), error=f'{type(exc).__name__}: {exc}')
                    report['processed'] += 1
                    self.progress.emit(dict(current=index + 1, total=len(files), message=json.dumps(result, ensure_ascii=False)))
        except png_compress.Cancelled:
            report['stopped'] = True
        except Exception as exc:
            report['failed'] += 1
            self.error.emit(f'{type(exc).__name__}: {exc}')
        self.result.emit(report)


class PngCompressPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.worker = None
        self._task_bridge = None
        self.setProperty('ownsPageHeader', True)
        outer = QVBoxLayout(self)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        layout = QVBoxLayout(content)
        scroll.setWidget(content)
        outer.addWidget(scroll)
        layout.addWidget(PageHeader('图片压缩 · PNG / JPG'))
        explanation = QLabel('选择一个或多个文件夹 → 设置质量数值 → 开始压缩。\n'
                             'JPG 保持分辨率，数值越低通常文件越小、细节损失越多；默认保留原 PNG。')
        explanation.setToolTip('JPG 仅支持 8 位灰度/RGB，不保留 PNG 元数据。请停止其他写入程序；大图需要数 GB 内存，'
                               '暂停或停止需等待当前编码返回。不支持的文件会保留并报告。')
        explanation.setWordWrap(True)
        layout.addWidget(explanation)
        self.sources = QPlainTextEdit()
        self.sources.setPlaceholderText('每行一个文件夹，可多次添加；也可直接粘贴多个路径')
        self.sources.setMinimumHeight(75)
        self.sources.setMaximumHeight(120)
        layout.addWidget(self.sources)
        self.recursive = QCheckBox('包含子文件夹')
        self.recursive.setChecked(True)
        layout.addWidget(self.recursive)
        self.mode = QComboBox()
        self.mode.addItem('PNG 无损原地压缩', 'png')
        self.mode.addItem('PNG → JPG 有损压缩（同目录、同名）', 'jpg')
        self.mode.setCurrentIndex(1)
        layout.addWidget(self.mode)
        self.quality = QSpinBox()
        self.quality.setRange(1, 100)
        self.quality.setValue(95)
        self.quality.setPrefix('JPG 质量（1–100）：')
        self.quality.setToolTip('数值越高通常体积越大；100 仍是有损。默认 95，支持时使用 4:4:4 色度采样。')
        layout.addWidget(self.quality)
        self.delete_source = QCheckBox('JPG 校验成功且更小后，永久删除对应 PNG（不可恢复）')
        layout.addWidget(self.delete_source)
        self.optimize = QCheckBox('重新优化 PNG 滤波（推荐；使用现有 OpenCV，解码像素校验）')
        self.optimize.setChecked(True)
        layout.addWidget(self.optimize)
        self.level = QComboBox()
        for title, value in [('标准无损压缩（推荐）', 6), ('快速无损压缩', 1), ('最大压缩（更慢，不保证明显更小）', 9)]:
            self.level.addItem(title, value)
        layout.addWidget(self.level)
        actions = ActionBar()
        layout.addWidget(actions)
        self.add = QPushButton('选择文件夹（可多选）')
        self.scan = QPushButton('扫描')
        self.start = QPushButton('开始原地压缩')
        self.pause = QPushButton('暂停')
        self.stop = QPushButton('停止')
        for button in (self.add, self.scan, self.start, self.pause, self.stop):
            actions.add_widget(button)
        self.pause.setEnabled(False)
        self.stop.setEnabled(False)
        self.add.clicked.connect(self._add)
        self.scan.clicked.connect(lambda: self._start(True))
        self.start.clicked.connect(lambda: self._start(False))
        self.pause.clicked.connect(self._pause)
        self.stop.clicked.connect(self._stop)
        self.progress = QProgressBar()
        layout.addWidget(self.progress)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(2000)
        layout.addWidget(self.log, 1)
        self.mode.currentIndexChanged.connect(self._mode_changed)
        self._mode_changed()

    def _mode_changed(self):
        jpg = self.mode.currentData() == 'jpg'
        self.quality.setVisible(jpg)
        self.delete_source.setVisible(jpg)
        self.level.setVisible(not jpg)
        self.optimize.setVisible(not jpg)
        self.start.setText('开始转换 JPG' if jpg else '开始原地压缩')

    def _add(self):
        dialog = QFileDialog(self, '选择文件夹（Ctrl / Shift 多选，也可多次添加）')
        dialog.setOption(QFileDialog.Option.DontUseNativeDialog, True)
        dialog.setOption(QFileDialog.Option.ShowDirsOnly, True)
        dialog.setFileMode(QFileDialog.FileMode.Directory)
        for view in dialog.findChildren(QTreeView) + dialog.findChildren(QListView):
            view.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        if dialog.exec():
            self._append_sources(dialog.selectedFiles())
        dialog.deleteLater()

    def _append_sources(self, folders):
        paths = self.sources.toPlainText().splitlines() + list(folders)
        unique = {}
        for raw in paths:
            path = raw.strip().strip('"')
            if path:
                unique.setdefault(os.path.normcase(os.path.normpath(path)), path)
        self.sources.setPlainText('\n'.join(unique.values()))

    def _start(self, scan_only):
        if self.worker is not None:
            return
        sources = [line.strip().strip('"') for line in self.sources.toPlainText().splitlines() if line.strip()]
        if not sources:
            QMessageBox.warning(self, '未选择输入', '请添加至少一个文件夹。')
            return
        warning = 'PNG 校验成功且变小后替换原文件，不保留备份。'
        if self.mode.currentData() == 'jpg':
            warning = f'转换为 JPG，质量 {self.quality.value()}，像素有损且不保留 PNG 元数据。\n'
            warning += '校验成功且更小后，永久删除源 PNG，不可恢复！' if self.delete_source.isChecked() else '保留源 PNG，在同目录生成同名 JPG。'
        if not scan_only and QMessageBox.question(self, '确认图片压缩',
                warning + '\n同名 JPG 不覆盖。请停止其他写入程序。是否继续？') != QMessageBox.StandardButton.Yes:
            return
        self.log.clear()
        self.progress.setRange(0, 0)
        self.worker = PngWorker(dict(sources=sources, recursive=self.recursive.isChecked(), level=self.level.currentData(),
                                    optimize=self.optimize.isChecked(), mode=self.mode.currentData(), quality=self.quality.value(),
                                    delete_source=self.delete_source.isChecked()), scan_only, self)
        self.worker.progress.connect(self._progress)
        self.worker.result.connect(self._result)
        self.worker.error.connect(self.log.appendPlainText)
        center = getattr(self, 'task_center', None)
        if center:
            if self._task_bridge:
                self._task_bridge.deleteLater()
            self._task_bridge = WorkerTaskBridge(self, center, self.worker, 'PNG 扫描' if scan_only else '图片压缩 / 转换',
                                                  'images.png_compress', '', self._stop)
        self.worker.finished.connect(self._finished)
        for widget in (self.sources, self.recursive, self.optimize, self.level, self.mode, self.quality, self.delete_source, self.add, self.scan, self.start):
            widget.setEnabled(False)
        self.pause.setEnabled(True)
        self.stop.setEnabled(True)
        self.worker.start()

    def _progress(self, fact):
        self.progress.setRange(0, fact['total'])
        self.progress.setValue(fact['current'])
        self.log.appendPlainText(fact['message'])

    def _result(self, result):
        self.log.appendPlainText(
            f"扫描 {result['count']} 张，原始总量 {result.get('bytes', 0) / 1024**3:.2f} GiB；"
            f"处理 {result['processed']} 张，无损变小 {result['compressed']} 张，生成 JPG {result['converted']} 张，"
            f"失败/不支持 {result['failed']} 张，节省 {result['saved'] / 1024**2:.2f} MiB。"
            + (' 已停止。' if result['stopped'] else ''))
        self.log.appendPlainText('详细汇总：' + json.dumps(result, ensure_ascii=False))
        if result['potential_saved']:
            self.log.appendPlainText(f"JPG 相对 PNG 减少 {result['potential_saved'] / 1024**2:.2f} MiB；保留 PNG 时不释放磁盘空间。")

    def _pause(self):
        if self.worker:
            paused = self.worker.control.paused
            if paused.is_set():
                paused.clear()
                self.pause.setText('暂停')
            else:
                paused.set()
                self.pause.setText('继续')

    def _stop(self):
        if self.worker:
            self.worker.control.stopped.set()
            self.worker.control.paused.clear()
            self.stop.setEnabled(False)
            if self._task_bridge:
                self._task_bridge.center.cancel(self._task_bridge.task_id)

    def _finished(self):
        self.worker.deleteLater()
        self.worker = None
        for widget in (self.sources, self.recursive, self.optimize, self.level, self.mode, self.quality, self.delete_source, self.add, self.scan, self.start):
            widget.setEnabled(True)
        self.pause.setEnabled(False)
        self.pause.setText('暂停')
        self.stop.setEnabled(False)
        if self.progress.maximum() == 0:
            self.progress.setRange(0, 1)
            self.progress.setValue(1)

    def protect_window_close(self, window):
        window.installEventFilter(self)

    def closeEvent(self, event):
        if self.worker is not None:
            self._stop()
            event.ignore()
            return
        super().closeEvent(event)

    def eventFilter(self, watched, event):
        if event.type() == QEvent.Type.Close and self.worker is not None:
            self._stop()
            self.log.appendPlainText('正在安全停止，请任务结束后再次关闭窗口。')
            event.ignore()
            return True
        return super().eventFilter(watched, event)
