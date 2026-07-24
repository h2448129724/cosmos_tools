"""GUI tool for CAB-F dataset validation and export."""
from __future__ import annotations

import json
import os
from pathlib import Path

from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QFrame,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QScrollArea,
    QVBoxLayout,
    QWidget,
    QCheckBox,
)

from cabf import (
    export_master_to_model_a,
    export_master_to_model_b,
    summarize_validation,
    summarize_validation_findings,
    validate_master_dataset,
    write_json,
)
from shared.project_paths import repo_root
from .tools.base import make_page_header


DEFAULT_MASTER_IMAGE_DIR = ""
DEFAULT_MASTER_ANNOTATION_DIR = ""
DEFAULT_MODEL_A_OUTPUT_DIR = ""
DEFAULT_MODEL_B_OUTPUT_DIR = ""
DOCS_ROOT = repo_root() / "docs"
SOP_DOC_PATH = DOCS_ROOT / "CABF_DATASET_SOP.md"
SCHEMA_DOC_PATH = DOCS_ROOT / "CABF_MASTER_SCHEMA.md"


class CabfDatasetToolDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("CAB-F 数据集校验与导出")
        self.resize(880, 660)
        self._setup_ui()

    def _setup_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(8)
        scroll.setWidget(content)
        outer.addWidget(scroll)

        hero = make_page_header("数据集校验与导出", "统一处理母数据目录、校验报告输出，以及模型 A / 模型 B 的导出流程。")

        self.summary_label = QLabel("等待执行。")
        self.summary_label.setStyleSheet("color:#334155;font-size:12px;")
        hero.layout().addWidget(self.summary_label)
        layout.addWidget(hero)

        doc_row = QHBoxLayout()
        btn_open_sop = QPushButton("打开 SOP 文档")
        btn_open_sop.clicked.connect(lambda: self._open_file(SOP_DOC_PATH))
        btn_open_schema = QPushButton("打开 Schema 文档")
        btn_open_schema.clicked.connect(lambda: self._open_file(SCHEMA_DOC_PATH))
        doc_row.addWidget(btn_open_sop)
        doc_row.addWidget(btn_open_schema)
        doc_row.addStretch()
        layout.addLayout(doc_row)

        source_group = QGroupBox("母数据目录")
        source_form = QFormLayout(source_group)
        self.edit_image_dir = self._create_path_row(DEFAULT_MASTER_IMAGE_DIR, source_form, "图片目录", True)
        self.edit_annotation_dir = self._create_path_row(DEFAULT_MASTER_ANNOTATION_DIR, source_form, "标注目录", True)
        layout.addWidget(source_group)

        validate_group = QGroupBox("校验")
        validate_layout = QVBoxLayout(validate_group)
        validate_row = QHBoxLayout()
        self.edit_report_path = QLineEdit("")
        btn_report = QPushButton("选择报告路径")
        btn_report.clicked.connect(self._browse_report_path)
        validate_row.addWidget(QLabel("报告路径"))
        validate_row.addWidget(self.edit_report_path, 1)
        validate_row.addWidget(btn_report)
        validate_layout.addLayout(validate_row)

        self.check_show_samples = QCheckBox("在结果中包含逐样本问题详情")
        validate_layout.addWidget(self.check_show_samples)

        validate_btn_row = QHBoxLayout()
        self.btn_validate = QPushButton("校验当前母数据")
        self.btn_validate.clicked.connect(self._run_validate)
        validate_btn_row.addWidget(self.btn_validate)
        validate_btn_row.addStretch()
        validate_layout.addLayout(validate_btn_row)
        layout.addWidget(validate_group)

        export_a_group = QGroupBox("导出模型 A 训练集（点检测）")
        export_a_form = QFormLayout(export_a_group)
        self.edit_model_a_output_dir = self._create_path_row(DEFAULT_MODEL_A_OUTPUT_DIR, export_a_form, "导出目录", True)
        self.check_model_a_skip_empty = QCheckBox("跳过空标注样本")
        self.check_model_a_skip_empty.setChecked(True)
        export_a_form.addRow("", self.check_model_a_skip_empty)
        export_a_btn_row = QHBoxLayout()
        btn_export_a = QPushButton("导出模型 A")
        btn_export_a.clicked.connect(self._run_export_model_a)
        btn_open_model_a = QPushButton("打开输出目录")
        btn_open_model_a.clicked.connect(lambda: self._open_directory(self.edit_model_a_output_dir.text().strip()))
        export_a_btn_row.addWidget(btn_export_a)
        export_a_btn_row.addWidget(btn_open_model_a)
        export_a_btn_row.addStretch()
        export_a_form.addRow("", self._wrap_layout(export_a_btn_row))
        layout.addWidget(export_a_group)

        export_b_group = QGroupBox("导出模型 B 训练集（连边）")
        export_b_form = QFormLayout(export_b_group)
        self.edit_model_b_output_dir = self._create_path_row(DEFAULT_MODEL_B_OUTPUT_DIR, export_b_form, "导出目录", True)
        self.check_model_b_skip_empty = QCheckBox("跳过空标注样本")
        self.check_model_b_skip_empty.setChecked(True)
        export_b_form.addRow("", self.check_model_b_skip_empty)
        export_b_btn_row = QHBoxLayout()
        btn_export_b = QPushButton("导出模型 B")
        btn_export_b.clicked.connect(self._run_export_model_b)
        btn_open_model_b = QPushButton("打开输出目录")
        btn_open_model_b.clicked.connect(lambda: self._open_directory(self.edit_model_b_output_dir.text().strip()))
        export_b_btn_row.addWidget(btn_export_b)
        export_b_btn_row.addWidget(btn_open_model_b)
        export_b_btn_row.addStretch()
        export_b_form.addRow("", self._wrap_layout(export_b_btn_row))
        layout.addWidget(export_b_group)

        result_title = QLabel("执行结果")
        result_title.setStyleSheet("font-size:12px;color:#6B7280;font-weight:600;")
        layout.addWidget(result_title)
        self.output_text = QPlainTextEdit()
        self.output_text.setReadOnly(True)
        self.output_text.setFixedHeight(148)
        self.output_text.hide()
        self._result_panel = QWidget()
        self._result_layout = QVBoxLayout(self._result_panel)
        self._result_layout.setContentsMargins(0, 0, 0, 0)
        self._result_layout.setSpacing(6)
        layout.addWidget(self._result_panel, 1)

    def _create_path_row(self, default_value: str, form: QFormLayout, label: str, folder: bool) -> QLineEdit:
        container = QWidget()
        row = QHBoxLayout(container)
        row.setContentsMargins(0, 0, 0, 0)
        edit = QLineEdit(default_value)
        button = QPushButton("浏览")
        if folder:
            button.clicked.connect(lambda: self._browse_directory(edit))
        else:
            button.clicked.connect(lambda: self._browse_file(edit))
        row.addWidget(edit, 1)
        row.addWidget(button)
        form.addRow(label, container)
        return edit

    def _wrap_layout(self, layout: QHBoxLayout) -> QWidget:
        container = QWidget()
        container.setLayout(layout)
        return container

    def _browse_directory(self, target: QLineEdit):
        path = QFileDialog.getExistingDirectory(self, "选择文件夹", target.text().strip() or str(Path.cwd()))
        if path:
            target.setText(path)

    def _browse_file(self, target: QLineEdit):
        path, _ = QFileDialog.getSaveFileName(
            self,
            "选择文件",
            target.text().strip() or str(Path.cwd() / "cabf_validation_report.json"),
            "JSON Files (*.json)",
        )
        if path:
            target.setText(path)

    def _browse_report_path(self):
        self._browse_file(self.edit_report_path)

    def _open_directory(self, path: str):
        if not path:
            QMessageBox.warning(self, "提示", "请先填写输出目录。")
            return
        folder = Path(path)
        if not folder.exists():
            QMessageBox.warning(self, "提示", f"目录不存在: {folder}")
            return
        os.startfile(str(folder))

    def _open_file(self, path: str | Path):
        file_path = Path(path)
        if not file_path.exists():
            QMessageBox.warning(self, "提示", f"文件不存在: {file_path}")
            return
        os.startfile(str(file_path))

    def _append_output(self, text: str):
        self.output_text.show()
        self._result_panel.hide()
        current = self.output_text.toPlainText().strip()
        merged = f"{current}\n\n{text}".strip() if current else text
        self.output_text.setPlainText(merged)
        self.output_text.verticalScrollBar().setValue(self.output_text.verticalScrollBar().maximum())

    def _show_result_card(self, title: str, stats: list[tuple[str, str, str]],
                          details: str = ""):
        """Show a structured result card with stats and optional expandable details.

        stats: list of (label, value, color_style) tuples
        """
        self.output_text.hide()
        self._result_panel.show()

        # clear previous
        while self._result_layout.count():
            item = self._result_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # summary card
        card = QFrame()
        card.setStyleSheet("background:#ffffff;border:1px solid #cfd3d7;border-radius:3px;")
        card_lay = QVBoxLayout(card)
        card_lay.setContentsMargins(14, 12, 14, 12)
        card_lay.setSpacing(6)

        header = QLabel(title)
        header.setStyleSheet("color:#0f172a;font-size:15px;font-weight:700;")
        card_lay.addWidget(header)

        # stats row
        stats_row = QHBoxLayout()
        stats_row.setSpacing(10)
        for label, value, style in stats:
            stat_frame = QFrame()
            stat_frame.setStyleSheet("background:#f2f3f4;border:1px solid #dfe2e4;border-radius:2px;")
            stat_lay = QVBoxLayout(stat_frame)
            stat_lay.setContentsMargins(10, 7, 10, 7)
            stat_lay.setSpacing(2)
            val_lbl = QLabel(value)
            val_lbl.setStyleSheet(f"color:{style};font-size:16px;font-weight:700;")
            stat_lay.addWidget(val_lbl)
            lbl = QLabel(label)
            lbl.setStyleSheet("color:#6B7280;font-size:11px;font-weight:600;")
            stat_lay.addWidget(lbl)
            stats_row.addWidget(stat_frame)
        card_lay.addLayout(stats_row)

        if details:
            detail_box = QPlainTextEdit()
            detail_box.setReadOnly(True)
            detail_box.setPlainText(details)
            detail_box.setFixedHeight(140)
            card_lay.addWidget(detail_box)

        self._result_layout.addWidget(card)
        self._result_layout.addStretch()

    def _source_dirs(self) -> tuple[str, str]:
        image_dir = self.edit_image_dir.text().strip()
        annotation_dir = self.edit_annotation_dir.text().strip()
        if not image_dir or not annotation_dir:
            raise ValueError("请先填写母数据的图片目录和标注目录。")
        return image_dir, annotation_dir

    def _run_validate(self):
        try:
            image_dir, annotation_dir = self._source_dirs()
            report = validate_master_dataset(image_dir, annotation_dir)
            report_path = self.edit_report_path.text().strip()
            if report_path:
                if self.check_show_samples.isChecked():
                    write_json(report_path, report)
                else:
                    brief_report = dict(report)
                    brief_report.pop("samples", None)
                    write_json(report_path, brief_report)

            summary = summarize_validation(report)
            findings = summarize_validation_findings(report, include_details=self.check_show_samples.isChecked())
            total = report.get("total", 0)
            valid = report.get("valid", 0)
            errors = report.get("errors", 0)
            warnings = report.get("warnings", 0)
            self._show_result_card(
                "校验结果",
                [
                    ("总样本", str(total), "#111827"),
                    ("通过", str(valid), "#16A34A"),
                    ("错误", str(errors), "#DC2626"),
                    ("警告", str(warnings), "#D97706"),
                ],
                details=findings or summary,
            )
            if report_path:
                self._append_output(f"报告已保存: {report_path}")
            self.summary_label.setText("校验完成，可以根据结果继续修正数据或直接执行导出。")
        except Exception as exc:
            self.summary_label.setText("校验失败，请检查母数据目录与报告输出路径。")
            QMessageBox.critical(self, "校验失败", str(exc))

    def _run_export_model_a(self):
        try:
            image_dir, annotation_dir = self._source_dirs()
            result = export_master_to_model_a(
                image_dir=image_dir,
                annotation_dir=annotation_dir,
                output_dir=self.edit_model_a_output_dir.text().strip(),
                include_empty=not self.check_model_a_skip_empty.isChecked(),
            )
            exported = result.get("exported", 0) if isinstance(result, dict) else 0
            skipped = result.get("skipped", 0) if isinstance(result, dict) else 0
            self._show_result_card(
                "模型 A 导出结果",
                [
                    ("已导出", str(exported), "#16A34A"),
                    ("跳过", str(skipped), "#D97706"),
                ],
                details=json.dumps(result, ensure_ascii=False, indent=2) if isinstance(result, dict) else str(result),
            )
            self.summary_label.setText("模型 A 导出完成，已自动打开输出目录。")
            self._open_directory(self.edit_model_a_output_dir.text().strip())
        except Exception as exc:
            self.summary_label.setText("模型 A 导出失败，请检查目录与数据状态。")
            QMessageBox.critical(self, "导出失败", str(exc))

    def _run_export_model_b(self):
        try:
            image_dir, annotation_dir = self._source_dirs()
            result = export_master_to_model_b(
                image_dir=image_dir,
                annotation_dir=annotation_dir,
                output_dir=self.edit_model_b_output_dir.text().strip(),
                include_empty=not self.check_model_b_skip_empty.isChecked(),
            )
            exported = result.get("exported", 0) if isinstance(result, dict) else 0
            skipped = result.get("skipped", 0) if isinstance(result, dict) else 0
            self._show_result_card(
                "模型 B 导出结果",
                [
                    ("已导出", str(exported), "#16A34A"),
                    ("跳过", str(skipped), "#D97706"),
                ],
                details=json.dumps(result, ensure_ascii=False, indent=2) if isinstance(result, dict) else str(result),
            )
            self.summary_label.setText("模型 B 导出完成，已自动打开输出目录。")
            self._open_directory(self.edit_model_b_output_dir.text().strip())
        except Exception as exc:
            self.summary_label.setText("模型 B 导出失败，请检查目录与数据状态。")
            QMessageBox.critical(self, "导出失败", str(exc))
