"""GUI tool for CAB-F dataset validation and export.

The dialog deliberately keeps the Qt shell thin: validation and export are run in
an owned worker thread while the page itself only renders immutable results.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable

from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from cabf import (
    export_master_to_model_a,
    export_master_to_model_b,
    summarize_validation,
    summarize_validation_findings,
    validate_master_dataset,
    write_json,
)
from cosmos_toolbox.ui.primitives import (
    ActionBar,
    CollapsibleLogPanel,
    MetricGrid,
    PageHeader,
    PathField,
    SectionSurface,
    StatusBanner,
)
from cosmos_toolbox.ui.theme import StatusTone
from shared.project_paths import repo_root


DEFAULT_MASTER_IMAGE_DIR = ""
DEFAULT_MASTER_ANNOTATION_DIR = ""
DEFAULT_MODEL_A_OUTPUT_DIR = ""
DEFAULT_MODEL_B_OUTPUT_DIR = ""
DOCS_ROOT = repo_root() / "docs"
SOP_DOC_PATH = DOCS_ROOT / "CABF_DATASET_SOP.md"
SCHEMA_DOC_PATH = DOCS_ROOT / "CABF_MASTER_SCHEMA.md"


def _validate_and_write_report(
    image_dir: str,
    annotation_dir: str,
    report_path: str,
    include_samples: bool,
) -> dict[str, Any]:
    """Validate and persist the optional report inside the worker thread."""
    report = validate_master_dataset(image_dir, annotation_dir)
    if report_path:
        output = report if include_samples else {k: v for k, v in report.items() if k != "samples"}
        write_json(report_path, output)
    return report


class _DatasetWorker(QThread):
    """Run one CAB-F operation away from the GUI thread.

    The CAB-F functions do not currently expose cooperative progress, therefore
    the shell uses an indeterminate progress bar and a cancellation request.  A
    request is honoured before a result is published; the thread is always joined
    by ``closeEvent`` to avoid Qt teardown races.
    """

    succeeded = Signal(object)
    failed = Signal(object)
    cancelled = Signal()

    def __init__(self, operation: Callable[..., Any], *args: Any, parent: QWidget | None = None, **kwargs: Any) -> None:
        super().__init__(parent)
        self._operation = operation
        self._args = args
        self._kwargs = kwargs

    def run(self) -> None:  # noqa: D401 - Qt thread entry point
        try:
            result = self._operation(*self._args, **self._kwargs)
        except Exception as exc:  # pragma: no cover - exercised through dialog tests
            if self.isInterruptionRequested():
                self.cancelled.emit()
            else:
                self.failed.emit(exc)
            return
        if self.isInterruptionRequested():
            self.cancelled.emit()
        else:
            self.succeeded.emit(result)


class CabfDatasetToolDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("CAB-F 数据集校验与导出")
        self.resize(880, 660)
        self._worker: _DatasetWorker | None = None
        self._active_title = ""
        self._busy = False
        self._cancel_requested = False
        self._operation_controls: list[QWidget] = []
        self._setup_ui()

    def _setup_ui(self) -> None:
        shell = QVBoxLayout(self)
        shell.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        content = QWidget()
        outer = QVBoxLayout(content)
        outer.setContentsMargins(14, 14, 14, 14)
        outer.setSpacing(10)
        outer.setAlignment(Qt.AlignmentFlag.AlignTop)
        scroll.setWidget(content)
        shell.addWidget(scroll)

        hero = PageHeader("数据集校验与导出", "统一处理母数据目录、校验报告输出，以及模型 A / 模型 B 的导出流程。")
        self.summary_label = QLabel("等待执行。")
        self.summary_label.setWordWrap(True)
        self.summary_label.setAccessibleName("操作摘要")
        self.summary_label.setProperty("uiRole", "muted")
        hero.layout().itemAt(0).layout().addWidget(self.summary_label)
        outer.addWidget(hero)

        self.status_banner = StatusBanner("准备就绪", StatusTone.NEUTRAL)
        outer.addWidget(self.status_banner)

        doc_actions = ActionBar()
        self.btn_open_sop = self._button("打开 SOP 文档", "打开 CAB-F 数据集 SOP 文档")
        self.btn_open_sop.clicked.connect(lambda: self._open_file(SOP_DOC_PATH))
        self.btn_open_schema = self._button("打开 Schema 文档", "打开 CAB-F 母数据 Schema 文档")
        self.btn_open_schema.clicked.connect(lambda: self._open_file(SCHEMA_DOC_PATH))
        doc_actions.add_widget(self.btn_open_sop)
        doc_actions.add_widget(self.btn_open_schema)
        outer.addWidget(doc_actions)
        self._operation_controls.extend([self.btn_open_sop, self.btn_open_schema])

        source = SectionSurface("母数据目录", "图片和标注目录会在同一次校验或导出任务中使用。")
        self.edit_image_dir = self._create_path_row(DEFAULT_MASTER_IMAGE_DIR, source.body_layout, "图片目录", True)
        self.edit_annotation_dir = self._create_path_row(
            DEFAULT_MASTER_ANNOTATION_DIR, source.body_layout, "标注目录", True
        )
        outer.addWidget(source)

        validate = SectionSurface("校验", "可选地写出 JSON 报告；结构化结果会一直保留在页面上。")
        report_field = PathField("报告路径", placeholder="可选，例如 cabf_validation_report.json", browse_text="选择")
        report_field.browse_requested.connect(lambda: self._browse_file(report_field.line_edit))
        self.edit_report_path = report_field.line_edit
        validate.body_layout.addWidget(report_field)
        self.check_show_samples = QCheckBox("在结果中包含逐样本问题详情")
        self.check_show_samples.setAccessibleName("包含逐样本问题详情")
        validate.body_layout.addWidget(self.check_show_samples)
        validate_actions = ActionBar()
        self.btn_validate = self._button("校验当前母数据", "在后台校验当前母数据")
        self.btn_validate.setProperty("buttonRole", "primary")
        self.btn_validate.clicked.connect(self._run_validate)
        validate_actions.add_widget(self.btn_validate)
        validate.body_layout.addWidget(validate_actions)
        outer.addWidget(validate)
        self._operation_controls.extend(
            [self.edit_report_path, self.check_show_samples, self.btn_validate, report_field.browse_button]
        )

        (
            self.edit_model_a_output_dir,
            self.check_model_a_skip_empty,
            self.btn_export_a,
            self.btn_open_model_a,
        ) = self._build_export_surface(
            outer, "模型 A", "点检测", DEFAULT_MODEL_A_OUTPUT_DIR, self._run_export_model_a
        )
        (
            self.edit_model_b_output_dir,
            self.check_model_b_skip_empty,
            self.btn_export_b,
            self.btn_open_model_b,
        ) = self._build_export_surface(
            outer, "模型 B", "连边", DEFAULT_MODEL_B_OUTPUT_DIR, self._run_export_model_b
        )

        result = SectionSurface("执行结果", "每次任务的结构化统计与详细输出。")
        self._result_panel = QWidget()
        result_layout = QVBoxLayout(self._result_panel)
        result_layout.setContentsMargins(0, 0, 0, 0)
        result_layout.setSpacing(8)
        self._result_title = QLabel("尚未执行任务")
        self._result_title.setAccessibleName("结构化结果标题")
        self._result_title.setProperty("uiRole", "sectionTitle")
        result_layout.addWidget(self._result_title)
        self._metrics = MetricGrid()
        result_layout.addWidget(self._metrics)
        self._details_panel = CollapsibleLogPanel("结果详情", expanded=True)
        self._details_panel.log.setPlaceholderText("任务完成后显示结构化详情。")
        result_layout.addWidget(self._details_panel)
        result.body_layout.addWidget(self._result_panel)
        outer.addWidget(result)

        self._log_panel = CollapsibleLogPanel("运行日志", expanded=False)
        self.output_text = self._log_panel.log  # compatibility with the previous public-ish attribute
        outer.addWidget(self._log_panel)

        self._progress = QProgressBar()
        self._progress.setRange(0, 0)
        self._progress.setTextVisible(False)
        self._progress.setAccessibleName("任务进度")
        self._progress.hide()
        outer.addWidget(self._progress)
        self._cancel_button = QPushButton("取消任务")
        self._cancel_button.setAccessibleName("取消后台任务")
        self._cancel_button.setProperty("buttonRole", "danger")
        self._cancel_button.clicked.connect(self._cancel_operation)
        self._cancel_button.setEnabled(False)
        outer.addWidget(self._cancel_button)

    def _button(self, text: str, accessible_name: str) -> QPushButton:
        button = QPushButton(text)
        button.setAccessibleName(accessible_name)
        return button

    def _build_export_surface(
        self,
        outer: QVBoxLayout,
        model: str,
        kind: str,
        default_dir: str,
        callback: Callable[[], None],
    ) -> tuple[QLineEdit, QCheckBox, QPushButton, QPushButton]:
        surface = SectionSurface(f"导出{model}训练集（{kind}）")
        path_field = PathField("导出目录", placeholder="选择导出目录", browse_text="浏览", show_open=True)
        path_field.browse_requested.connect(lambda: self._browse_directory(path_field.line_edit))
        path_field.open_requested.connect(lambda: self._open_directory(path_field.line_edit.text().strip()))
        path_field.setText(default_dir)
        surface.body_layout.addWidget(path_field)
        skip_empty = QCheckBox("跳过空标注样本")
        skip_empty.setChecked(True)
        skip_empty.setAccessibleName(f"{model}跳过空标注样本")
        surface.body_layout.addWidget(skip_empty)
        actions = ActionBar()
        export_button = self._button(f"导出{model}", f"后台导出{model}训练集")
        export_button.setProperty("buttonRole", "primary")
        export_button.clicked.connect(callback)
        open_button = self._button("打开输出目录", f"打开{model}输出目录")
        open_button.clicked.connect(lambda: self._open_directory(path_field.line_edit.text().strip()))
        actions.add_widget(export_button)
        actions.add_widget(open_button)
        surface.body_layout.addWidget(actions)
        outer.addWidget(surface)
        self._operation_controls.extend(
            [
                path_field.line_edit,
                path_field.browse_button,
                path_field.open_button,
                skip_empty,
                export_button,
                open_button,
            ]
        )
        return path_field.line_edit, skip_empty, export_button, open_button

    def _create_path_row(self, default_value: str, form: QVBoxLayout, label: str, folder: bool | str) -> QLineEdit:
        field = PathField(label, placeholder="选择目录", browse_text="浏览")
        field.setText(default_value)
        if folder is False:
            field.browse_requested.connect(lambda: self._browse_file(field.line_edit))
        else:
            field.browse_requested.connect(lambda: self._browse_directory(field.line_edit))
        field.setAccessibleName(label if isinstance(folder, bool) else str(folder))
        if hasattr(form, "addWidget"):
            form.addWidget(field)
        else:  # legacy QFormLayout callers
            form.addRow(label, field)
        self._operation_controls.extend([field.line_edit, field.browse_button])
        return field.line_edit

    def _wrap_layout(self, layout: QHBoxLayout) -> QWidget:
        """Compatibility helper retained for older callers."""
        container = QWidget()
        container.setLayout(layout)
        return container

    def _browse_directory(self, target: QLineEdit) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择文件夹", target.text().strip() or str(Path.cwd()))
        if path:
            target.setText(path)

    def _browse_file(self, target: QLineEdit) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "选择文件",
            target.text().strip() or str(Path.cwd() / "cabf_validation_report.json"),
            "JSON Files (*.json)",
        )
        if path:
            target.setText(path)

    def _browse_report_path(self) -> None:
        """Compatibility slot for the previous report-path browse button."""
        self._browse_file(self.edit_report_path)

    def _open_directory(self, path: str) -> None:
        if not path:
            QMessageBox.warning(self, "提示", "请先填写输出目录。")
            return
        folder = Path(path)
        if not folder.exists():
            QMessageBox.warning(self, "提示", f"目录不存在: {folder}")
            return
        os.startfile(str(folder))

    def _open_file(self, path: str | Path) -> None:
        file_path = Path(path)
        if not file_path.exists():
            QMessageBox.warning(self, "提示", f"文件不存在: {file_path}")
            return
        os.startfile(str(file_path))

    def _append_output(self, text: str) -> None:
        """Append runtime output without hiding the structured result panel."""
        current = self.output_text.toPlainText().strip()
        merged = f"{current}\n\n{text}".strip() if current else text
        self.output_text.setPlainText(merged)
        self.output_text.verticalScrollBar().setValue(self.output_text.verticalScrollBar().maximum())

    def _show_result_card(self, title: str, stats: list[tuple[str, str, str]], details: str = "") -> None:
        self._result_panel.show()
        self._result_title.setText(title)
        self._metrics.set_metrics([(label, value) for label, value, _style in stats])
        self._details_panel.log.setPlainText(details)
        self._details_panel.setVisible(bool(details))

    def _source_dirs(self) -> tuple[str, str]:
        image_dir = self.edit_image_dir.text().strip()
        annotation_dir = self.edit_annotation_dir.text().strip()
        if not image_dir or not annotation_dir:
            raise ValueError("请先填写母数据的图片目录和标注目录。")
        return image_dir, annotation_dir

    def _start_operation(self, title: str, operation: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        if self._worker is not None and self._worker.isRunning():
            return
        self._set_busy(True, f"正在执行{title}…")
        worker = _DatasetWorker(operation, *args, **kwargs, parent=self)
        self._worker = worker
        self._active_title = title
        # Connect to bound QObject methods so Qt queues all UI updates onto the
        # dialog thread instead of executing a Python lambda in the worker thread.
        worker.succeeded.connect(self._on_worker_success)
        worker.failed.connect(self._on_worker_failure)
        worker.cancelled.connect(self._on_worker_cancelled)
        worker.finished.connect(self._worker_finished)
        worker.start()

    def _set_busy(self, busy: bool, message: str = "") -> None:
        self._busy = busy
        self._progress.setVisible(busy)
        self._cancel_button.setEnabled(busy)
        for widget in self._operation_controls:
            widget.setEnabled(not busy)
        if message:
            self.status_banner.set_status(message, StatusTone.RUNNING if busy else StatusTone.NEUTRAL)
            self.summary_label.setText(message)

    def _cancel_operation(self) -> None:
        worker = self._worker
        if worker is None or not worker.isRunning():
            return
        self._cancel_requested = True
        worker.requestInterruption()
        self._cancel_button.setEnabled(False)
        self.status_banner.set_status("已请求取消；正在安全收尾当前任务…", StatusTone.WARNING)
        self.summary_label.setText("已请求取消；当前 CAB-F 操作完成后将停止更新结果。")

    def _worker_finished(self) -> None:
        worker = self._worker
        self._worker = None
        if worker is not None:
            worker.deleteLater()

    def _on_worker_success(self, result: Any) -> None:
        self._operation_succeeded(self._active_title, result)

    def _on_worker_failure(self, exc: Exception) -> None:
        self._operation_failed(self._active_title, exc)

    def _on_worker_cancelled(self) -> None:
        self._operation_cancelled(self._active_title)

    def _operation_succeeded(self, title: str, result: Any) -> None:
        self._set_busy(False)
        self._cancel_requested = False
        self._render_operation_result(title, result)

    def _operation_failed(self, title: str, exc: Exception) -> None:
        self._set_busy(False)
        self._cancel_requested = False
        self.status_banner.set_status(f"{title}失败", StatusTone.DANGER)
        self.summary_label.setText(f"{title}失败，请检查目录与数据状态。")
        QMessageBox.critical(self, f"{title}失败", str(exc))

    def _operation_cancelled(self, title: str) -> None:
        self._set_busy(False)
        self._cancel_requested = False
        self.status_banner.set_status(f"{title}已取消", StatusTone.WARNING)
        self.summary_label.setText(f"{title}已取消，结构化结果保持不变。")
        self._append_output(f"{title}：取消请求已处理。")

    def _render_operation_result(self, title: str, result: Any) -> None:
        if title == "校验":
            self._render_validation_result(result)
            return
        model = "A" if "模型 A" in title else "B"
        exported = result.get("images_exported", 0) if isinstance(result, dict) else 0
        skipped = (
            result.get("samples_routed_to_error", 0) + result.get("skipped_empty_annotations", 0)
            if isinstance(result, dict)
            else 0
        )
        self._show_result_card(
            f"模型 {model} 导出结果",
            [("已导出", str(exported), "success"), ("跳过", str(skipped), "warning")],
            details=json.dumps(result, ensure_ascii=False, indent=2) if isinstance(result, dict) else str(result),
        )
        self._append_output(f"模型 {model} 导出完成。")
        self.status_banner.set_status(f"模型 {model} 导出完成", StatusTone.SUCCESS)
        self.summary_label.setText(f"模型 {model} 导出完成，可打开输出目录查看文件。")
        # Preserve the previous convenience behavior while keeping it outside
        # the worker (os.startfile is an imperative shell side effect).
        output_edit = self.edit_model_a_output_dir if model == "A" else self.edit_model_b_output_dir
        try:
            self._open_directory(output_edit.text().strip())
        except OSError:
            # Opening a directory is best-effort; the export result remains valid.
            pass

    def _render_validation_result(self, report: dict[str, Any]) -> None:
        report_path = self.edit_report_path.text().strip()
        summary = summarize_validation(report)
        findings = summarize_validation_findings(report, include_details=self.check_show_samples.isChecked())
        summary_counts = report.get("summary", {})
        total = summary_counts.get(
            "total_samples",
            summary_counts.get("paired_samples", 0)
            + summary_counts.get("missing_annotations", 0)
            + summary_counts.get("orphan_annotations", 0),
        )
        errors = summary_counts.get("samples_with_errors", 0)
        warnings = summary_counts.get("samples_with_warnings", 0)
        valid = max(total - errors, 0)
        self._show_result_card(
            "校验结果",
            [
                ("总样本", str(total), "neutral"),
                ("通过", str(valid), "success"),
                ("错误", str(errors), "danger"),
                ("警告", str(warnings), "warning"),
            ],
            details=findings or summary,
        )
        if report_path:
            self._append_output(f"报告已保存: {report_path}")
        self.status_banner.set_status("校验完成", StatusTone.SUCCESS if errors == 0 else StatusTone.WARNING)
        self.summary_label.setText("校验完成，可以根据结果继续修正数据或直接执行导出。")

    def _run_validate(self) -> None:
        try:
            image_dir, annotation_dir = self._source_dirs()
        except Exception as exc:
            self._operation_failed("校验", exc)
            return
        self._start_operation(
            "校验",
            _validate_and_write_report,
            image_dir,
            annotation_dir,
            self.edit_report_path.text().strip(),
            self.check_show_samples.isChecked(),
        )

    def _run_export_model_a(self) -> None:
        try:
            image_dir, annotation_dir = self._source_dirs()
            output_dir = self.edit_model_a_output_dir.text().strip()
            if not output_dir:
                raise ValueError("请先填写模型 A 导出目录。")
        except Exception as exc:
            self._operation_failed("模型 A 导出", exc)
            return
        self._start_operation(
            "模型 A 导出",
            export_master_to_model_a,
            image_dir=image_dir,
            annotation_dir=annotation_dir,
            output_dir=output_dir,
            include_empty=not self.check_model_a_skip_empty.isChecked(),
        )

    def _run_export_model_b(self) -> None:
        try:
            image_dir, annotation_dir = self._source_dirs()
            output_dir = self.edit_model_b_output_dir.text().strip()
            if not output_dir:
                raise ValueError("请先填写模型 B 导出目录。")
        except Exception as exc:
            self._operation_failed("模型 B 导出", exc)
            return
        self._start_operation(
            "模型 B 导出",
            export_master_to_model_b,
            image_dir=image_dir,
            annotation_dir=annotation_dir,
            output_dir=output_dir,
            include_empty=not self.check_model_b_skip_empty.isChecked(),
        )

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt interface
        worker = self._worker
        if worker is not None and worker.isRunning():
            worker.requestInterruption()
            worker.wait(10000)
        if worker is not None and worker.isRunning():
            event.ignore()
            return
        super().closeEvent(event)
