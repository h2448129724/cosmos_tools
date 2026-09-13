"""Desktop entry point for CAB-F field dataset collection."""
from __future__ import annotations

import json
import sqlite3
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

from PySide6.QtCore import QEvent, QThread, QUrl, Signal, Qt, QSize
from PySide6.QtGui import QDesktopServices, QImageReader, QPixmap, QPainter, QPen, QColor
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFileDialog, QFormLayout, QGridLayout,
    QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPlainTextEdit,
    QProgressBar, QPushButton, QScrollArea, QSpinBox, QVBoxLayout, QWidget,
    QListWidget, QListWidgetItem, QTabWidget,
)

from .paths import ensure_import_paths


class DatasetWorker(QThread):
    progress = Signal(dict)
    result = Signal(dict)
    error = Signal(str)

    def __init__(self, options: dict, control, scan_only: bool = False, parent=None, environment_name=None):
        super().__init__(parent)
        self.options = options
        self.control = control
        self.scan_only = scan_only
        self.environment_name = environment_name

    def run(self):
        try:
            if self.environment_name:
                from .field_dataset_runtime import run_in_environment
                self.result.emit(run_in_environment(self.options, self.environment_name, self.control,
                                                    self.progress.emit, self.scan_only))
                return
            from .field_dataset import run, scan

            if self.scan_only:
                files = scan(self.options["source"], face=self.options["face"])
                total_bytes = sum(Path(p).stat().st_size for p in files)
                self.result.emit({"scan": True, "count": len(files), "bytes": total_bytes})
            else:
                result = run(**self.options, control=self.control, on_progress=self.progress.emit)
                self.result.emit(result)
        except Exception as exc:
            self.error.emit(f"{type(exc).__name__}: {exc}")


class FieldDatasetPage(QWidget):
    """Independent model selection with a cooperative background runner."""

    def __init__(self, parent=None):
        super().__init__(parent)
        ensure_import_paths()
        from .field_dataset import MODEL_IDS

        self.setProperty("ownsPageHeader", True)
        self.setWindowTitle("CAB-F 现场数据集生成")
        self.worker = None
        self.control = SimpleNamespace(paused=threading.Event(), stopped=threading.Event())
        self._closing_window = None
        layout = QVBoxLayout(self)
        title = QLabel("CAB-F 现场数据集生成")
        title.setStyleSheet("font-size: 22px; font-weight: 600;")
        layout.addWidget(title)
        help_text = QLabel("单独勾选任一模型，或组合生成多套数据；必要的定位步骤自动执行。自动标签需要人工复核。")
        help_text.setWordWrap(True)
        layout.addWidget(help_text)

        self.settings = QGroupBox("输入与生成设置")
        form = QFormLayout(self.settings)
        self.source = QLineEdit()
        self.output = QLineEdit()
        form.addRow("原图文件夹", self._folder_row(self.source))
        form.addRow("输出文件夹", self._folder_row(self.output))
        self.environment = QComboBox()
        self.environment.setEditable(True)
        self.environment.addItem('onnx-gpu')
        self.environment.setToolTip('任务在所选 Conda 环境中运行；缺少依赖会报错，不会自动安装。')
        environment_row = QWidget()
        environment_layout = QHBoxLayout(environment_row)
        environment_layout.setContentsMargins(0, 0, 0, 0)
        environment_layout.addWidget(self.environment)
        refresh_environments = QPushButton('加载环境列表')
        refresh_environments.clicked.connect(self._load_environments)
        environment_layout.addWidget(refresh_environments)
        form.addRow('执行 Conda 环境', environment_row)
        self.product = QComboBox()
        self.product.addItems(["D01-R", "D01-L"])
        self.face = QComboBox()
        for label, value in [("全部", "all"), ("Top", "top"), ("Bottom", "bottom")]:
            self.face.addItem(label, value)
        self.mode = QComboBox()
        self.mode.addItem("提取并自动标注（伪标签）", "auto")
        self.mode.addItem("仅提取图片（待标注）", "images")
        self.limit = QSpinBox()
        self.limit.setRange(0, 10000000)
        self.limit.setSpecialValueText("全部图片")
        self.limit.setToolTip("设置少量图片进行试运行；0 表示全部。按扫描顺序取样。")
        form.addRow("产品配置", self.product)
        form.addRow("正反面", self.face)
        form.addRow("生成方式", self.mode)
        form.addRow("本次最多处理", self.limit)
        layout.addWidget(self.settings)

        self.models_group = QGroupBox("选择要导出的模型数据集")
        models_layout = QVBoxLayout(self.models_group)
        shortcuts = QHBoxLayout()
        for label, selection in [("全选", "all"), ("清空", "none"), ("尾部全套", "tail"), ("耳片全套", "ear")]:
            button = QPushButton(label)
            button.clicked.connect(lambda checked=False, value=selection: self._select(value))
            shortcuts.addWidget(button)
        models_layout.addLayout(shortcuts)
        container = QWidget()
        grid = QGridLayout(container)
        self.checks = {}
        for index, (model_id, label) in enumerate(MODEL_IDS.items()):
            check = QCheckBox(f"{label}\n{model_id}")
            check.setToolTip("仅导出勾选项；必要的前置定位模型由执行器自动补齐。")
            self.checks[model_id] = check
            check.toggled.connect(self._update_dependencies)
            grid.addWidget(check, index // 2, index % 2)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(container)
        scroll.setMinimumHeight(180)
        models_layout.addWidget(scroll)
        self.dependencies = QLabel("自动依赖：无")
        self.dependencies.setWordWrap(True)
        models_layout.addWidget(self.dependencies)
        layout.addWidget(self.models_group, 2)

        actions = QHBoxLayout()
        self.scan_button = QPushButton("扫描图片")
        self.start_button = QPushButton("开始生成")
        self.preview_button = QPushButton("试跑前 2 张")
        self.pause_button = QPushButton("暂停")
        self.stop_button = QPushButton("安全停止")
        self.open_button = QPushButton("打开输出目录")
        for button in (self.scan_button, self.preview_button, self.start_button, self.pause_button, self.stop_button, self.open_button):
            actions.addWidget(button)
        layout.addLayout(actions)
        self.scan_button.clicked.connect(lambda: self._start(scan_only=True))
        self.start_button.clicked.connect(lambda: self._start())
        self.preview_button.clicked.connect(self._preview_run)
        self.pause_button.clicked.connect(self._toggle_pause)
        self.stop_button.clicked.connect(self._stop)
        self.open_button.clicked.connect(self._open_output)
        self.status = QLabel("请选择文件夹和模型；可先限制处理数量进行试运行。")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(2000)
        self.tabs = QTabWidget()
        self.tabs.addTab(self.log, "运行日志")
        preview = QWidget()
        preview_layout = QVBoxLayout(preview)
        self.load_preview_button = QPushButton("加载输出目录中的样本")
        self.load_preview_button.clicked.connect(self._load_previews)
        preview_layout.addWidget(self.load_preview_button)
        preview_row = QHBoxLayout()
        self.sample_list = QListWidget()
        self.sample_list.setMaximumWidth(320)
        self.sample_list.currentItemChanged.connect(self._show_sample)
        preview_row.addWidget(self.sample_list)
        self.sample_image = QLabel("生成后可查看裁片与标签叠加；最多载入 500 个样本。")
        self.sample_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        preview_scroll = QScrollArea()
        preview_scroll.setWidgetResizable(True)
        preview_scroll.setWidget(self.sample_image)
        preview_row.addWidget(preview_scroll, 1)
        preview_layout.addLayout(preview_row)
        self.tabs.addTab(preview, "样本预览")
        layout.addWidget(self.tabs, 2)
        self._set_busy(False)

    def _load_environments(self):
        from shared.conda_runtime import CondaEnvManager
        current = self.environment.currentText().strip() or 'onnx-gpu'
        names = [item.name for item in CondaEnvManager().list_envs(refresh=True)]
        self.environment.clear()
        self.environment.addItems(list(dict.fromkeys([current, 'onnx-gpu', *names])))
        self.environment.setCurrentText(current)
        if not names:
            QMessageBox.warning(self, 'Conda 环境', '未读取到环境列表，请确认 Conda 可用；也可手动输入环境名称。')

    def _update_dependencies(self):
        from .field_dataset import MODEL_DEPENDENCIES, MODEL_IDS
        selected = {key for key, check in self.checks.items() if check.isChecked()}
        required = set()
        pending = list(selected)
        while pending:
            for dependency in MODEL_DEPENDENCIES.get(pending.pop(), []):
                if dependency not in required:
                    required.add(dependency)
                    pending.append(dependency)
        labels = [MODEL_IDS[key] for key in MODEL_IDS if key in required - selected]
        self.dependencies.setText("自动依赖（只运行，不额外导出）：" + ("、".join(labels) or "无"))

    def _preview_run(self):
        previous = self.limit.value()
        self.limit.setValue(2)
        self._start()
        self.limit.setValue(previous)

    def _load_previews(self):
        self.sample_list.clear()
        root = Path(self.output.text().strip()).resolve()
        database = root / "run.db"
        if not database.is_file():
            self.sample_image.setText("输出目录尚无 run.db，请先生成数据。")
            return
        try:
            connection = sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)
            try:
                rows = connection.execute("SELECT model,details FROM items WHERE status='complete' ORDER BY model,key")
                count = 0
                for model, details in rows:
                    for record in json.loads(details):
                        directory = Path(record["directory"]).resolve()
                        if root not in directory.parents:
                            continue
                        item = QListWidgetItem(f"{model}\n{Path(record.get('source', '')).name} / {directory.name}")
                        item.setData(Qt.ItemDataRole.UserRole, str(directory))
                        self.sample_list.addItem(item)
                        count += 1
                        if count >= 500:
                            break
                    if count >= 500:
                        break
            finally:
                connection.close()
            self.tabs.setCurrentIndex(1)
            if self.sample_list.count():
                self.sample_list.setCurrentRow(0)
            else:
                self.sample_image.setText("暂无已提交样本。")
        except Exception as exc:
            self.log.appendPlainText(f"样本预览读取失败：{exc}")

    def _show_sample(self, item, previous=None):
        if item is None:
            return
        try:
            directory = Path(item.data(Qt.ItemDataRole.UserRole))
            reader = QImageReader(str(directory / "image.png"))
            original = reader.size()
            if not original.isValid():
                raise ValueError("裁片无法读取")
            scaled = original.scaled(QSize(1200, 1200), Qt.AspectRatioMode.KeepAspectRatio)
            reader.setScaledSize(scaled)
            pixmap = QPixmap.fromImage(reader.read())
            if pixmap.isNull():
                raise ValueError(reader.errorString())
            annotation = json.loads((directory / "image.json").read_text(encoding="utf-8"))
            painter = QPainter(pixmap)
            painter.setPen(QPen(QColor("#ff6038"), 2))
            sx, sy = pixmap.width() / original.width(), pixmap.height() / original.height()
            for shape in annotation.get("shapes", []):
                points = [(int(x * sx), int(y * sy)) for x, y in shape.get("points", [])]
                if shape.get("shape_type") == "rectangle" and len(points) == 2:
                    (x1, y1), (x2, y2) = points
                    painter.drawRect(min(x1, x2), min(y1, y2), abs(x2-x1), abs(y2-y1))
                elif shape.get("shape_type") == "point" and points:
                    painter.drawEllipse(points[0][0]-3, points[0][1]-3, 6, 6)
                else:
                    segments = list(zip(points, points[1:]))
                    if shape.get("shape_type") == "polygon" and len(points) > 2:
                        segments.append((points[-1], points[0]))
                    for start, end in segments:
                        painter.drawLine(*start, *end)
                if points:
                    painter.drawText(points[0][0], max(12, points[0][1]-4), str(shape.get("label", "")))
            painter.end()
            self.sample_image.setPixmap(pixmap)
        except Exception as exc:
            self.sample_image.setText(f"预览失败：{exc}")

    def _folder_row(self, edit):
        row = QWidget()
        box = QHBoxLayout(row)
        box.setContentsMargins(0, 0, 0, 0)
        box.addWidget(edit)
        browse = QPushButton("浏览…")
        browse.clicked.connect(lambda: self._browse(edit))
        box.addWidget(browse)
        return row

    def _browse(self, edit):
        chosen = QFileDialog.getExistingDirectory(self, "选择文件夹", edit.text())
        if chosen:
            edit.setText(chosen)

    def _select(self, selection):
        tail = {"tail_roi_detector", "tail_placement_classifier", "tail_cloth_roi_detector", "tail_cloth_seam_classifier", "hook_detector"}
        ear = {"ear_placement_classifier", "knife_segment"}
        for key, check in self.checks.items():
            check.setChecked(selection == "all" or (selection == "tail" and key in tail) or (selection == "ear" and key in ear))

    def _start(self, scan_only=False):
        if self.worker and self.worker.isRunning():
            return
        source = self.source.text().strip()
        output = self.output.text().strip()
        selected = [key for key, check in self.checks.items() if check.isChecked()]
        if not source or not Path(source).is_dir():
            QMessageBox.warning(self, "输入目录", "请选择存在的原图文件夹。")
            return
        if not scan_only and (not output or not selected):
            QMessageBox.warning(self, "生成设置", "请选择输出文件夹，并至少勾选一个模型。")
            return
        self.control.paused.clear()
        self.control.stopped.clear()
        options = dict(source=source, output=output, product=self.product.currentText(), selected=selected,
                       face=self.face.currentData(), mode=self.mode.currentData(), limit=self.limit.value() or None)
        environment_name = self.environment.currentText().strip()
        if not environment_name:
            QMessageBox.warning(self, 'Conda 环境', '请选择或输入执行环境名称。')
            return
        self.worker = DatasetWorker(options, self.control, scan_only, self, environment_name=environment_name)
        self.worker.progress.connect(self._on_progress)
        self.worker.result.connect(self._on_result)
        self.worker.error.connect(self._on_error)
        self.worker.finished.connect(self._finished)
        self._set_busy(True, scan_only)
        self.progress_bar.setRange(0, 0)
        self.status.setText("正在扫描图片…" if scan_only else "正在准备模型与输出…")
        self.log.appendPlainText(self.status.text())
        self.worker.start()

    def _set_busy(self, busy, scan_only=False):
        self.settings.setEnabled(not busy)
        self.models_group.setEnabled(not busy)
        self.scan_button.setEnabled(not busy)
        self.start_button.setEnabled(not busy)
        self.preview_button.setEnabled(not busy)
        self.load_preview_button.setEnabled(not busy)
        self.pause_button.setEnabled(busy and not scan_only)
        self.stop_button.setEnabled(busy and not scan_only)

    def _toggle_pause(self):
        if self.control.paused.is_set():
            self.control.paused.clear()
            self.pause_button.setText("暂停")
            self.status.setText("继续处理中…")
        else:
            self.control.paused.set()
            self.pause_button.setText("继续")
            self.status.setText("已请求暂停，将在安全处理点等待。")

    def _stop(self):
        self.control.stopped.set()
        self.control.paused.clear()
        self.pause_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self.status.setText("已请求停止，正在完成当前安全处理点并保存进度…")

    def _on_progress(self, data):
        total = data.get("total", 0)
        done = data.get("current", data.get("done", data.get("processed", 0)))
        if isinstance(total, int) and total > 0 and isinstance(done, int):
            self.progress_bar.setRange(0, total)
            self.progress_bar.setValue(done)
        message = data.get("message") or json.dumps(data, ensure_ascii=False, default=str)
        self.log.appendPlainText(str(message))
        if not self.control.paused.is_set() and not self.control.stopped.is_set():
            self.status.setText(str(message))

    def _on_result(self, result):
        self.progress_bar.setRange(0, 100)
        if result.get("scan"):
            self.status.setText(f"扫描到 {result['count']} 张图片，合计 {result['bytes'] / 1024 ** 3:.2f} GiB。损坏图片在解码时检测。")
        else:
            self.status.setText("已停止，保留已完成结果。使用相同配置再次开始可续跑。" if self.control.stopped.is_set() else "生成结束，请查看各模型统计和输出报告。")
            training = result.get("export", {}).get("training_dataset", {}).get("directory")
            if result.get("export", {}).get("format") == "xanylabeling":
                training = result["export"]["directory"]
            if training and not self.control.stopped.is_set():
                self.status.setText(f"数据集已生成：{training}（datasets 为中间目录，无需用于训练）")
            self.progress_bar.setValue(0 if self.control.stopped.is_set() else 100)
        self.log.appendPlainText(json.dumps(result, ensure_ascii=False, indent=2, default=str))

    def _on_error(self, message):
        self.progress_bar.setRange(0, 100)
        self.status.setText("任务异常，请查看日志。")
        self.log.appendPlainText(message)

    def _finished(self):
        self._set_busy(False)
        self.pause_button.setText("暂停")
        if self.worker and not self.worker.scan_only:
            self._load_previews()

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
            self.log.appendPlainText("任务停止后请再次关闭窗口。")
            return True
        return super().eventFilter(watched, event)

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self._stop()
            event.ignore()
        else:
            event.accept()


def main():
    app = QApplication.instance() or QApplication(sys.argv)
    page = FieldDatasetPage()
    page.resize(1050, 900)
    page.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
