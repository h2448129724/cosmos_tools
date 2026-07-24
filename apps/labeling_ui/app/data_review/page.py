"""Compact, project-neutral dataset review page."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ..annotation.canvas import AnnotationCanvas
from ..annotation.document import AnnotationDocument
from ..annotation.layer_panel import LayerVisibilityPanel
from ..tools.base import set_primary
from .model import (
    ReviewItem,
    ReviewResult,
    ReviewSpec,
    build_review_result,
    collect_review_items,
    make_review_trash_dir,
    move_review_item,
)
from .preview import ImageOnlyPreviewAdapter, ReviewPreviewAdapter


def read_image_bgr(path: Path) -> np.ndarray:
    payload = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(payload, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"无法读取图片: {path}")
    return image


class ReviewCanvas(AnnotationCanvas):
    """Annotation canvas sized for an embedded workspace rather than a dialog."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(320, 240)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def set_document(self, image_or_document, document: AnnotationDocument | None = None) -> None:
        if document is None:
            super().set_document(image_or_document)
            return
        super().set_image(image_or_document, image_path=document.image_path)
        super().set_document(document)

    def change_zoom(self, delta_steps: float) -> None:
        self.scale = float(np.clip(self.scale * (1.15**delta_steps), 0.05, 20.0))
        self.update()

    def wheelEvent(self, event) -> None:
        if self.image_qimage is None:
            return
        self.change_zoom(1.0 if event.angleDelta().y() > 0 else -1.0)
        event.accept()


class DatasetReviewPage(QWidget):
    """Reusable review UI. Workflow-specific propagation belongs to the caller."""

    reviewCompleted = Signal(object)

    def __init__(
        self,
        parent=None,
        *,
        preview_adapter: ReviewPreviewAdapter | None = None,
        title: str = "数据审阅",
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(1280, 780)
        self.preview_adapter = preview_adapter or ImageOnlyPreviewAdapter()
        self.items: list[ReviewItem] = []
        self.current_index = -1
        self.current_item: ReviewItem | None = None
        self.spec: ReviewSpec | None = None
        self.trash_dir: Path | None = None
        self.accepted_count = 0
        self.removed_count = 0
        self._build_ui()
        self._connect_signals()

    @property
    def saved_count(self) -> int:
        return self.accepted_count

    @property
    def trash_count(self) -> int:
        return self.removed_count

    def configure_paths(
        self,
        *,
        mode: str = "unlabeled",
        image_dir: str = "",
        label_dir: str = "",
        save_dir: str = "",
        auto_load: bool = False,
    ) -> None:
        self.combo_mode.setCurrentIndex(1 if str(mode).lower() == "labeled" else 0)
        if image_dir:
            self.edit_image_dir.setText(image_dir)
        if label_dir:
            self.edit_label_dir.setText(label_dir)
        if save_dir:
            self.edit_save_dir.setText(save_dir)
        self._refresh_source_summary()
        if image_dir:
            self.source_panel.setVisible(False)
        if auto_load and image_dir:
            self.open_dataset()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 10, 12, 10)
        root.setSpacing(8)

        source_bar = QFrame()
        source_bar.setObjectName("hintPanel")
        source_layout = QHBoxLayout(source_bar)
        source_layout.setContentsMargins(10, 8, 10, 8)
        source_layout.setSpacing(8)
        source_layout.addWidget(QLabel("数据源"))
        self.source_summary = QLabel("尚未选择图片目录")
        self.source_summary.setTextInteractionFlags(Qt.TextSelectableByMouse)
        source_layout.addWidget(self.source_summary, 1)
        self.btn_toggle_source = QPushButton("设置")
        self.btn_toggle_source.setMaximumWidth(72)
        self.btn_load = QPushButton("加载 / 刷新")
        set_primary(self.btn_load)
        source_layout.addWidget(self.btn_toggle_source)
        source_layout.addWidget(self.btn_load)
        root.addWidget(source_bar)

        self.source_panel = QFrame()
        self.source_panel.setObjectName("card")
        source_grid = QGridLayout(self.source_panel)
        source_grid.setContentsMargins(10, 8, 10, 8)
        source_grid.setHorizontalSpacing(8)
        source_grid.setVerticalSpacing(6)
        self.combo_mode = QComboBox()
        self.combo_mode.addItem("全部图片（标注可选）", "unlabeled")
        self.combo_mode.addItem("仅显示有标注图片", "labeled")
        source_grid.addWidget(QLabel("范围"), 0, 0)
        source_grid.addWidget(self.combo_mode, 0, 1, 1, 2)
        self.edit_image_dir, self.btn_choose_image_dir = self._add_path_row(
            source_grid, 0, "图片目录", start_column=3
        )
        self.edit_label_dir, self.btn_choose_label_dir = self._add_path_row(source_grid, 1, "标注目录")
        self.edit_save_dir, self.btn_choose_save_dir = self._add_path_row(
            source_grid, 1, "保留目录", start_column=3
        )
        source_grid.setColumnStretch(1, 1)
        source_grid.setColumnStretch(4, 1)
        root.addWidget(self.source_panel)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)

        browser = QFrame()
        browser.setObjectName("card")
        browser.setMinimumWidth(260)
        browser.setMaximumWidth(340)
        browser_layout = QVBoxLayout(browser)
        browser_layout.setContentsMargins(10, 9, 10, 9)
        browser_layout.setSpacing(7)
        browser_title_row = QHBoxLayout()
        browser_title_row.addWidget(QLabel("样本"))
        self.lbl_index = QLabel("0 / 0")
        self.lbl_index.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        browser_title_row.addWidget(self.lbl_index, 1)
        browser_layout.addLayout(browser_title_row)
        self.file_list = QListWidget()
        self.file_list.setMinimumHeight(120)
        browser_layout.addWidget(self.file_list, 1)
        self.review_summary = QLabel("标注 -  ·  保留 0  ·  移除 0")
        self.review_summary.setWordWrap(True)
        self.review_summary.setStyleSheet("color:#475569;font-size:12px;")
        browser_layout.addWidget(self.review_summary)
        splitter.addWidget(browser)

        preview = QFrame()
        preview.setObjectName("card")
        preview_layout = QVBoxLayout(preview)
        preview_layout.setContentsMargins(10, 9, 10, 9)
        preview_layout.setSpacing(6)
        preview_header = QHBoxLayout()
        preview_header.addWidget(QLabel("预览"))
        self.lbl_current_name = QLabel("未加载图片")
        self.lbl_current_name.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.lbl_current_name.setTextInteractionFlags(Qt.TextSelectableByMouse)
        preview_header.addWidget(self.lbl_current_name, 1)
        preview_layout.addLayout(preview_header)
        self.canvas = ReviewCanvas()
        self.layer_panel = LayerVisibilityPanel()
        self.layer_panel.set_canvas(self.canvas)
        self.layer_panel.setVisible(False)
        preview_layout.addWidget(self.layer_panel)
        preview_layout.addWidget(self.canvas, 1)
        splitter.addWidget(preview)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([300, 900])
        root.addWidget(splitter, 1)

        action_bar = QFrame()
        action_layout = QHBoxLayout(action_bar)
        action_layout.setContentsMargins(0, 0, 0, 0)
        action_layout.setSpacing(8)
        self.btn_prev = QPushButton("上一张  A")
        self.btn_next = QPushButton("下一张  D")
        self.btn_accept = QPushButton("保留到目录  S")
        set_primary(self.btn_accept)
        self.btn_remove = QPushButton("移除  W")
        self.btn_complete = QPushButton("完成审阅")
        action_layout.addWidget(self.btn_prev)
        action_layout.addWidget(self.btn_next)
        action_layout.addStretch(1)
        action_layout.addWidget(self.btn_accept)
        action_layout.addWidget(self.btn_remove)
        action_layout.addWidget(self.btn_complete)
        root.addWidget(action_bar)

        self.status_label = QLabel("设置数据源后加载样本。")
        self.status_label.setStyleSheet("color:#64748b;font-size:12px;")
        root.addWidget(self.status_label)

        # Compatibility attributes used by older callers during migration.
        self.btn_save = self.btn_accept
        self.btn_trash = self.btn_remove
        self.btn_apply_flow = self.btn_complete
        self.lbl_label_state = QLabel("无")
        self.lbl_point_count = QLabel("0")
        self.lbl_edge_count = QLabel("0")
        self.lbl_saved_count = QLabel("0")
        self.lbl_trash_count = QLabel("0")

    @staticmethod
    def _add_path_row(
        layout: QGridLayout,
        row: int,
        title: str,
        *,
        start_column: int = 0,
    ) -> tuple[QLineEdit, QPushButton]:
        edit = QLineEdit()
        button = QPushButton("浏览")
        button.setMaximumWidth(72)
        layout.addWidget(QLabel(title), row, start_column)
        layout.addWidget(edit, row, start_column + 1)
        layout.addWidget(button, row, start_column + 2)
        return edit, button

    def _connect_signals(self) -> None:
        self.btn_toggle_source.clicked.connect(lambda: self.source_panel.setVisible(not self.source_panel.isVisible()))
        self.btn_choose_image_dir.clicked.connect(self.choose_image_dir)
        self.btn_choose_label_dir.clicked.connect(self.choose_label_dir)
        self.btn_choose_save_dir.clicked.connect(self.choose_save_dir)
        self.btn_load.clicked.connect(self.open_dataset)
        self.combo_mode.currentIndexChanged.connect(self._sync_mode_ui)
        self.file_list.currentRowChanged.connect(self.jump_to_index)
        self.btn_prev.clicked.connect(lambda: self.jump_to_index(self.current_index - 1))
        self.btn_next.clicked.connect(lambda: self.jump_to_index(self.current_index + 1))
        self.btn_accept.clicked.connect(self.move_current_to_save)
        self.btn_remove.clicked.connect(self.move_current_to_trash)
        self.btn_complete.clicked.connect(self.complete_review)
        self._sync_mode_ui()

    def _current_mode(self) -> str:
        return str(self.combo_mode.currentData() or "unlabeled")

    def _sync_mode_ui(self) -> None:
        self.edit_label_dir.setPlaceholderText(
            "必填：只加载能配对的标注" if self._current_mode() == "labeled" else "可选：存在同名标注时显示"
        )
        self._refresh_source_summary()

    def _refresh_source_summary(self) -> None:
        image_dir = self.edit_image_dir.text().strip() if hasattr(self, "edit_image_dir") else ""
        mode = "仅有标注" if hasattr(self, "combo_mode") and self._current_mode() == "labeled" else "全部图片"
        self.source_summary.setText(f"{image_dir or '尚未选择图片目录'}  ·  {mode}")

    def choose_image_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择图片目录", self.edit_image_dir.text().strip())
        if path:
            self.edit_image_dir.setText(path)
            self._refresh_source_summary()

    def choose_label_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择标注目录", self.edit_label_dir.text().strip())
        if path:
            self.edit_label_dir.setText(path)

    def choose_save_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择保留目录", self.edit_save_dir.text().strip())
        if path:
            self.edit_save_dir.setText(path)

    def _make_spec(self) -> ReviewSpec | None:
        image_text = self.edit_image_dir.text().strip()
        annotation_text = self.edit_label_dir.text().strip()
        accepted_text = self.edit_save_dir.text().strip()
        if not image_text:
            QMessageBox.warning(self, "提示", "请先选择图片目录。")
            return None
        image_dir = Path(image_text)
        if not image_dir.is_dir():
            QMessageBox.warning(self, "提示", "图片目录不存在。")
            return None
        annotation_dir = Path(annotation_text) if annotation_text else image_dir
        require_annotation = self._current_mode() == "labeled"
        if require_annotation and not annotation_dir.is_dir():
            QMessageBox.warning(self, "提示", "标注目录不存在。")
            return None
        if not annotation_dir.is_dir():
            annotation_dir = None
        return ReviewSpec(
            image_source=image_dir,
            annotation_source=annotation_dir,
            accepted_output=Path(accepted_text) if accepted_text else None,
            trash_root=Path.cwd(),
            require_annotation=require_annotation,
        )

    def open_dataset(self) -> None:
        spec = self._make_spec()
        if spec is None:
            return
        self.spec = spec
        self.items = collect_review_items(spec)
        self.current_index = -1
        self.current_item = None
        self.accepted_count = 0
        self.removed_count = 0
        self.trash_dir = None
        self.file_list.blockSignals(True)
        self.file_list.clear()
        for item in self.items:
            suffix = "" if item.has_annotation else "  [无标注]"
            QListWidgetItem(f"{item.image_path.name}{suffix}", self.file_list)
        self.file_list.blockSignals(False)
        self.source_panel.setVisible(False)
        self._refresh_source_summary()
        self._refresh_counts()
        if not self.items:
            self._clear_preview()
            self.status_label.setText("当前条件下没有可审阅的样本。")
            return
        annotated = sum(item.has_annotation for item in self.items)
        self.status_label.setText(f"已加载 {len(self.items)} 张图片，其中 {annotated} 张带标注。")
        self.jump_to_index(0)

    def jump_to_index(self, index: int) -> None:
        if not self.items:
            return
        index = int(np.clip(index, 0, len(self.items) - 1))
        if index == self.current_index and self.current_item is not None:
            return
        item = self.items[index]
        try:
            image = read_image_bgr(item.image_path)
            document = self.preview_adapter.load(item.image_path, item.annotation_path, image)
            summary = self.preview_adapter.summarize(document)
        except Exception as exc:
            QMessageBox.critical(self, "加载失败", str(exc))
            return
        self.current_index = index
        self.current_item = item
        self.file_list.blockSignals(True)
        self.file_list.setCurrentRow(index)
        self.file_list.blockSignals(False)
        self.canvas.set_document(image, document)
        self.layer_panel.set_canvas(self.canvas)
        self.layer_panel.setVisible(bool(document.layers))
        self.lbl_current_name.setText(item.image_path.name)
        self.lbl_label_state.setText("有" if item.has_annotation else "无")
        self.lbl_point_count.setText(summary.get("点", "0"))
        self.lbl_edge_count.setText(summary.get("边", "0"))
        self._refresh_counts(summary)
        detail = " · ".join(f"{key} {value}" for key, value in summary.items())
        self.status_label.setText(f"已加载 {item.image_path.name}" + (f"  ·  {detail}" if detail else ""))

    def _refresh_counts(self, annotation_summary: dict[str, str] | None = None) -> None:
        self.lbl_index.setText(f"{self.current_index + 1 if self.current_index >= 0 else 0} / {len(self.items)}")
        self.lbl_saved_count.setText(str(self.accepted_count))
        self.lbl_trash_count.setText(str(self.removed_count))
        if annotation_summary is None:
            annotation_text = "标注 -"
        else:
            annotation_text = " · ".join(f"{key} {value}" for key, value in annotation_summary.items()) or "仅图片"
        self.review_summary.setText(
            f"{annotation_text}  ·  保留 {self.accepted_count}  ·  移除 {self.removed_count}"
        )

    def _take_current_item(self) -> tuple[ReviewItem | None, int]:
        if self.current_index < 0 or self.current_index >= len(self.items):
            return None, -1
        index = self.current_index
        item = self.items.pop(index)
        self.file_list.blockSignals(True)
        self.file_list.takeItem(index)
        self.file_list.blockSignals(False)
        self.current_index = -1
        self.current_item = None
        return item, index

    def _restore_item(self, item: ReviewItem, index: int) -> None:
        self.items.insert(index, item)
        suffix = "" if item.has_annotation else "  [无标注]"
        self.file_list.blockSignals(True)
        self.file_list.insertItem(index, f"{item.image_path.name}{suffix}")
        self.file_list.blockSignals(False)
        self.jump_to_index(index)

    def _show_after_removal(self, index: int, message: str) -> None:
        if not self.items:
            self._clear_preview()
            self.status_label.setText(f"{message} 当前没有剩余样本。")
            self._refresh_counts()
            return
        self.jump_to_index(min(index, len(self.items) - 1))
        self.status_label.setText(message)

    def _clear_preview(self) -> None:
        self.current_index = -1
        self.current_item = None
        self.lbl_current_name.setText("未加载图片")
        self.lbl_label_state.setText("无")
        self.lbl_point_count.setText("0")
        self.lbl_edge_count.setText("0")
        self.canvas.image_bgr = None
        self.canvas.image_qimage = None
        self.canvas.update()
        self.layer_panel.setVisible(False)

    def move_current_to_save(self) -> None:
        if self.spec is None or self.spec.accepted_output is None:
            QMessageBox.warning(self, "提示", "请先在数据源设置中选择保留目录。")
            self.source_panel.setVisible(True)
            return
        item, index = self._take_current_item()
        if item is None:
            return
        try:
            moved_image, moved_annotation = move_review_item(item, self.spec.accepted_output)
        except Exception as exc:
            self._restore_item(item, index)
            QMessageBox.critical(self, "保留失败", str(exc))
            return
        self.accepted_count += 1
        names = [moved_image.name] + ([moved_annotation.name] if moved_annotation else [])
        self._show_after_removal(index, f"已保留: {', '.join(names)}")

    def move_current_to_trash(self) -> None:
        if self.spec is None:
            return
        item, index = self._take_current_item()
        if item is None:
            return
        if self.trash_dir is None:
            self.trash_dir = make_review_trash_dir(self.spec.trash_root or Path.cwd())
        try:
            moved_image, moved_annotation = move_review_item(item, self.trash_dir)
        except Exception as exc:
            self._restore_item(item, index)
            QMessageBox.critical(self, "移除失败", str(exc))
            return
        self.removed_count += 1
        names = [moved_image.name] + ([moved_annotation.name] if moved_annotation else [])
        self._show_after_removal(index, f"已移除: {', '.join(names)}")

    def current_result(self) -> ReviewResult | None:
        spec = self.spec or self._make_spec()
        if spec is None:
            return None
        return build_review_result(
            spec,
            accepted_count=self.accepted_count,
            removed_count=self.removed_count,
            remaining_count=len(self.items),
        )

    def complete_review(self) -> None:
        result = self.current_result()
        if result is None:
            return
        self.reviewCompleted.emit(result)
        self.status_label.setText(f"审阅结果已提交：{result.active_image_dir}")

    def apply_to_workflow(self) -> None:
        """Compatibility shim; new callers subscribe to reviewCompleted."""
        self.complete_review()

    def keyPressEvent(self, event) -> None:
        if isinstance(QApplication.focusWidget(), QLineEdit):
            super().keyPressEvent(event)
            return
        handlers = {
            Qt.Key_A: lambda: self.jump_to_index(self.current_index - 1),
            Qt.Key_D: lambda: self.jump_to_index(self.current_index + 1),
            Qt.Key_S: self.move_current_to_save,
            Qt.Key_W: self.move_current_to_trash,
        }
        handler = handlers.get(event.key())
        if handler is not None:
            handler()
            event.accept()
            return
        super().keyPressEvent(event)


class DatasetReviewDialog(QDialog):
    """Standalone compatibility shell around :class:`DatasetReviewPage`."""

    reviewCompleted = Signal(object)

    def __init__(
        self,
        parent=None,
        *,
        preview_adapter: ReviewPreviewAdapter | None = None,
        title: str = "数据审阅",
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(1280, 780)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.page = DatasetReviewPage(
            self,
            preview_adapter=preview_adapter,
            title=title,
        )
        self.page.reviewCompleted.connect(self.reviewCompleted.emit)
        layout.addWidget(self.page)

    def configure_paths(self, **kwargs) -> None:
        self.page.configure_paths(**kwargs)

    def open_dataset(self) -> None:
        self.page.open_dataset()

    def current_result(self) -> ReviewResult | None:
        return self.page.current_result()

    def complete_review(self) -> None:
        self.page.complete_review()

    def apply_to_workflow(self) -> None:
        self.page.apply_to_workflow()
