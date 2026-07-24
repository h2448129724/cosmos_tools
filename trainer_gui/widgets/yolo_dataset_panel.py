from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QGroupBox, QHBoxLayout, QPushButton, QTextEdit, QVBoxLayout, QWidget

from modules.yolo.dataset_inspector import YoloDatasetReport


class YoloDatasetPanel(QGroupBox):
    inspect_requested = Signal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__("YOLO 数据集体检", parent)
        self.setVisible(False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 12)
        layout.setSpacing(8)

        button_row = QHBoxLayout()
        self.inspect_button = QPushButton("检查数据集")
        self.inspect_button.setProperty("buttonRole", "secondary")
        self.inspect_button.clicked.connect(self.inspect_requested.emit)
        button_row.addWidget(self.inspect_button)
        button_row.addStretch(1)

        self.summary = QTextEdit()
        self.summary.setReadOnly(True)
        self.summary.setMinimumHeight(150)
        self.summary.setPlaceholderText("选择 data.yaml 后点击检查数据集。")

        layout.addLayout(button_row)
        layout.addWidget(self.summary)

    def reset(self) -> None:
        self.summary.setPlainText("选择 data.yaml 后点击检查数据集。")

    def set_error(self, message: str) -> None:
        self.summary.setPlainText(message)

    def set_report(self, report: YoloDatasetReport) -> None:
        lines = [
            "状态: " + ("通过" if report.ok else "需要处理"),
            f"data.yaml: {report.data_yaml}",
            f"数据根目录: {report.dataset_root}",
            f"train: {report.train_path or ''}",
            f"val: {report.val_path or ''}",
            "",
            f"训练图片: {report.train_image_count}",
            f"验证图片: {report.val_image_count}",
            f"标签文件: {report.label_file_count}",
            f"空标签文件: {report.empty_label_count}",
            f"标注框: {report.box_count}",
            f"缺失标签: {report.missing_label_count}",
            f"孤儿标签: {report.label_without_image_count}",
            f"非法标签行: {report.invalid_label_count}",
            "",
            "类别分布:",
        ]
        if report.class_names:
            for class_id, name in sorted(report.class_names.items()):
                lines.append(f"  {class_id} {name}: {report.class_counts.get(class_id, 0)}")
        else:
            lines.append("  未读取到 names")

        if report.errors:
            lines.extend(["", "错误:"])
            lines.extend(f"  {item}" for item in report.errors)
        if report.missing_label_samples:
            lines.extend(["", "缺失标签样例:"])
            lines.extend(f"  {item}" for item in report.missing_label_samples)
        if report.label_without_image_samples:
            lines.extend(["", "孤儿标签样例:"])
            lines.extend(f"  {item}" for item in report.label_without_image_samples)
        if report.invalid_label_samples:
            lines.extend(["", "非法标签样例:"])
            lines.extend(f"  {item}" for item in report.invalid_label_samples)

        self.summary.setPlainText("\n".join(lines))
