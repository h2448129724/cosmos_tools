from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

try:
    from .convert_xanylabeling import ConvertOptions, ConvertReport, convert
    from .generate_detect_train import TrainCodeOptions, generate_detect_train
except ImportError:
    from convert_xanylabeling import ConvertOptions, ConvertReport, convert
    from generate_detect_train import TrainCodeOptions, generate_detect_train


DEFAULT_YOLO = Path(r"D:\project\changrui\cosmos\tmp\CAB-F\D01-R_tail_dataset")
DEFAULT_XAL = DEFAULT_YOLO / "xanylabeling"


class ConvertWorker(QThread):
    logged = Signal(str)
    succeeded = Signal(object)
    failed = Signal(str)

    def __init__(self, direction: str, yolo_root: Path, xal_root: Path, options: ConvertOptions) -> None:
        super().__init__()
        self.direction = direction
        self.yolo_root = yolo_root
        self.xal_root = xal_root
        self.options = options

    def run(self) -> None:
        try:
            report = convert(self.direction, self.yolo_root, self.xal_root, self.options, log=self.logged.emit)
            self.succeeded.emit(report)
        except Exception as error:
            self.failed.emit(f"{error}\n{traceback.format_exc()}")


class ConvertWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("YOLO Detection ⇄ X-AnyLabeling / 训练代码")
        self.resize(860, 680)
        self.worker: ConvertWorker | None = None
        self.direction = "yolo_to_xal"

        self.stack = QStackedWidget()
        self.stack.addWidget(self._build_direction_page())
        self.stack.addWidget(self._build_path_page())
        self.setCentralWidget(self.stack)

    def _build_direction_page(self) -> QWidget:
        hint = QLabel("先选择转换方向，或直接生成检测训练脚本。目标可以是空目录。")
        hint.setWordWrap(True)
        yolo_to_xal = QPushButton("YOLO → X-AnyLabeling")
        xal_to_yolo = QPushButton("X-AnyLabeling → YOLO")
        gen_train = QPushButton("生成检测训练代码")
        yolo_to_xal.setMinimumHeight(48)
        xal_to_yolo.setMinimumHeight(48)
        gen_train.setMinimumHeight(48)
        yolo_to_xal.clicked.connect(lambda: self._choose_direction("yolo_to_xal"))
        xal_to_yolo.clicked.connect(lambda: self._choose_direction("xal_to_yolo"))
        gen_train.clicked.connect(lambda: self._choose_direction("gen_train"))
        layout = QVBoxLayout()
        layout.addStretch(1)
        layout.addWidget(hint)
        layout.addWidget(yolo_to_xal)
        layout.addWidget(xal_to_yolo)
        layout.addWidget(gen_train)
        layout.addStretch(2)
        page = QWidget()
        page.setLayout(layout)
        return page

    def _build_path_page(self) -> QWidget:
        self.direction_label = QLabel()
        self.source_edit = QLineEdit()
        self.dest_edit = QLineEdit()
        self.source_label = QLabel("源目录")
        self.dest_label = QLabel("目标目录")
        source_row = self._path_row(self.source_edit, "选择源目录")
        dest_row = self._path_row(self.dest_edit, "选择目标目录，可为空")
        self.overwrite_json = QCheckBox("覆盖已有 JSON")
        self.overwrite_labels = QCheckBox("覆盖已有 YOLO 标签")
        self.overwrite_labels.setChecked(True)
        self.copy_images = QCheckBox("复制图片")
        self.copy_images.setChecked(True)
        self.overwrite_images = QCheckBox("覆盖已有图片")
        self.overwrite_script = QCheckBox("覆盖已有训练脚本")
        self.dry_run = QCheckBox("只预览，不写盘")
        self.path_hint = QLabel(
            "目标目录可以为空。XAL→YOLO 时会向上查找 split_manifest.csv 恢复 train/val；找不到则写入 train。"
        )
        self.path_hint.setWordWrap(True)

        form = QFormLayout()
        form.addRow(self.source_label, source_row)
        form.addRow(self.dest_label, dest_row)
        box = QGroupBox("路径")
        box.setLayout(form)

        options = QHBoxLayout()
        for widget in (
            self.overwrite_json,
            self.overwrite_labels,
            self.copy_images,
            self.overwrite_images,
            self.overwrite_script,
            self.dry_run,
        ):
            options.addWidget(widget)
        options.addStretch(1)

        back = QPushButton("返回")
        back.clicked.connect(lambda: self.stack.setCurrentIndex(0))
        self.generate_button = QPushButton("生成训练代码")
        self.generate_button.clicked.connect(lambda: self._generate_train_code())
        self.run_button = QPushButton("开始转换")
        self.run_button.clicked.connect(self._start)
        buttons = QHBoxLayout()
        buttons.addWidget(back)
        buttons.addStretch(1)
        buttons.addWidget(self.generate_button)
        buttons.addWidget(self.run_button)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)

        layout = QVBoxLayout()
        layout.addWidget(self.direction_label)
        layout.addWidget(self.path_hint)
        layout.addWidget(box)
        layout.addLayout(options)
        layout.addLayout(buttons)
        layout.addWidget(self.log, 1)
        page = QWidget()
        page.setLayout(layout)
        return page

    def _path_row(self, edit: QLineEdit, title: str) -> QWidget:
        button = QPushButton("浏览…")
        button.clicked.connect(lambda: self._browse(edit, title))
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(edit, 1)
        row.addWidget(button)
        widget = QWidget()
        widget.setLayout(row)
        return widget

    def _browse(self, edit: QLineEdit, title: str) -> None:
        path = QFileDialog.getExistingDirectory(self, title, edit.text() or str(Path.home()))
        if path:
            edit.setText(path)

    def _choose_direction(self, direction: str) -> None:
        self.direction = direction
        to_xal = direction == "yolo_to_xal"
        gen_train = direction == "gen_train"
        self.dest_edit.setPlaceholderText("")
        if gen_train:
            self.direction_label.setText("当前动作：生成 YOLO Detection 训练代码")
            self.source_label.setText("YOLO 检测数据集目录")
            self.dest_label.setText("脚本输出目录（可空）")
            self.source_edit.setText(str(DEFAULT_YOLO if DEFAULT_YOLO.is_dir() else ""))
            self.dest_edit.clear()
            self.dest_edit.setPlaceholderText("留空则写到 data.yaml 旁")
            self.path_hint.setText(
                "选择已有 YOLO 检测目录（含 data.yaml）。脚本默认写在数据集旁，不会改写 images/labels。"
            )
            self.run_button.setVisible(False)
            self.generate_button.setDefault(True)
        elif to_xal:
            self.direction_label.setText("当前方向：YOLO → X-AnyLabeling")
            self.source_label.setText("YOLO 源目录")
            self.dest_label.setText("X-AnyLabeling 目标目录")
            self.source_edit.setText(str(DEFAULT_YOLO if DEFAULT_YOLO.is_dir() else ""))
            self.dest_edit.setText(str(DEFAULT_XAL if DEFAULT_XAL.is_dir() else ""))
            self.path_hint.setText("目标目录可以为空。转换完成后可直接生成检测训练脚本。")
            self.run_button.setVisible(True)
            self.run_button.setDefault(True)
        else:
            self.direction_label.setText("当前方向：X-AnyLabeling → YOLO")
            self.source_label.setText("X-AnyLabeling 源目录")
            self.dest_label.setText("YOLO 目标目录（可为空）")
            self.source_edit.setText(str(DEFAULT_XAL if DEFAULT_XAL.is_dir() else ""))
            self.dest_edit.clear()
            self.path_hint.setText(
                "目标目录可以为空。XAL→YOLO 时会向上查找 split_manifest.csv 恢复 train/val；"
                "找不到则写入 train。转换完成后可生成训练脚本。"
            )
            self.run_button.setVisible(True)
            self.run_button.setDefault(True)
        self.overwrite_json.setEnabled(to_xal)
        self.overwrite_labels.setEnabled(not to_xal and not gen_train)
        self.copy_images.setEnabled(not gen_train)
        self.overwrite_images.setEnabled(not gen_train)
        self.log.clear()
        self.stack.setCurrentIndex(1)

    def _options(self) -> ConvertOptions:
        return ConvertOptions(
            overwrite_json=self.overwrite_json.isChecked(),
            overwrite_labels=self.overwrite_labels.isChecked(),
            copy_images=self.copy_images.isChecked(),
            overwrite_images=self.overwrite_images.isChecked(),
            dry_run=self.dry_run.isChecked(),
        )

    def _current_yolo_root(self) -> Path | None:
        if self.direction == "xal_to_yolo":
            text = self.dest_edit.text().strip()
        else:
            text = self.source_edit.text().strip()
        return Path(text) if text else None

    def _train_output_dir(self) -> Path | None:
        if self.direction != "gen_train":
            return None
        text = self.dest_edit.text().strip()
        return Path(text) if text else None

    def _start(self) -> None:
        if self.direction == "gen_train":
            self._generate_train_code()
            return
        source_text = self.source_edit.text().strip()
        dest_text = self.dest_edit.text().strip()
        if not source_text:
            QMessageBox.warning(self, "缺少路径", "请选择源目录")
            return
        if not dest_text:
            QMessageBox.warning(self, "缺少路径", "请选择目标目录，空目录也可以")
            return
        source = Path(source_text)
        dest = Path(dest_text)
        if self.direction == "yolo_to_xal":
            yolo_root, xal_root = source, dest
        else:
            yolo_root, xal_root = dest, source
        if self.worker is not None and self.worker.isRunning():
            return
        self.log.clear()
        self.run_button.setEnabled(False)
        self.generate_button.setEnabled(False)
        self.worker = ConvertWorker(self.direction, yolo_root, xal_root, self._options())
        self.worker.logged.connect(self.log.appendPlainText)
        self.worker.succeeded.connect(self._finished)
        self.worker.failed.connect(self._failed)
        self.worker.start()

    def _generate_train_code(self, yolo_root: Path | None = None) -> None:
        yolo_root = yolo_root or self._current_yolo_root()
        if yolo_root is None:
            QMessageBox.warning(self, "缺少路径", "请选择 YOLO 检测数据集目录（含 data.yaml）")
            return
        output_dir = self._train_output_dir()
        try:
            report = generate_detect_train(
                yolo_root,
                output_dir=output_dir,
                options=TrainCodeOptions(
                    overwrite=self.overwrite_script.isChecked(),
                    dry_run=self.dry_run.isChecked(),
                ),
                log=self.log.appendPlainText,
            )
        except Exception as error:
            self.log.appendPlainText(f"{error}\n{traceback.format_exc()}")
            QMessageBox.critical(self, "生成失败", str(error).splitlines()[0])
            return
        self.log.appendPlainText(json.dumps(report.as_dict(), ensure_ascii=False, indent=2))
        if not report.ok:
            QMessageBox.warning(self, "生成失败", "未能生成训练脚本，详见日志")
            return
        paths = report.written or report.skipped
        commands = "\n".join(report.run_commands) or "conda run -n onnx-gpu python train_detect.py"
        if self.dry_run.isChecked():
            title = "预览训练代码"
            prefix = "将写入（未实际写盘）：\n"
        elif report.written:
            title = "已生成训练代码"
            prefix = "脚本：\n"
        else:
            title = "未写入新文件"
            prefix = "已有脚本，未覆盖：\n"
        QMessageBox.information(
            self,
            title,
            prefix
            + "\n".join(paths)
            + "\n\n运行（conda 环境 onnx-gpu）：\n"
            + commands
            + "\n\n可在脚本里改 DATA / EPOCHS / IMGSZ / BATCH / DEVICE 等超参。",
        )

    def _finished(self, report: ConvertReport) -> None:
        self.run_button.setEnabled(True)
        self.generate_button.setEnabled(True)
        self.log.appendPlainText(json.dumps(report.as_dict(), ensure_ascii=False, indent=2))
        if not report.ok:
            QMessageBox.warning(self, "完成但有错误", f"{len(report.errors)} 条错误，详见日志")
            return
        generate = QMessageBox.question(
            self,
            "完成",
            "转换完成。要现在生成检测训练脚本吗？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if generate == QMessageBox.StandardButton.Yes and self.worker is not None:
            self._generate_train_code(yolo_root=self.worker.yolo_root)

    def _failed(self, message: str) -> None:
        self.run_button.setEnabled(True)
        self.generate_button.setEnabled(True)
        self.log.appendPlainText(message)
        QMessageBox.critical(self, "转换失败", message.splitlines()[0])


def launch() -> None:
    app = QApplication.instance() or QApplication(sys.argv)
    window = ConvertWindow()
    window.show()
    app.exec()


def main() -> int:
    launch()
    return 0


if __name__ == "__main__":
    modules_root = Path(__file__).resolve().parent.parent
    if str(modules_root) not in sys.path:
        sys.path.insert(0, str(modules_root))
    from yolo.convert_xanylabeling_ui import main as package_main

    raise SystemExit(package_main())
