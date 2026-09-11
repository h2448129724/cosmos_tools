"""Compact, project-neutral dataset review page."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from img_tools.core.review_session import (
    ReviewSession,
    begin_review,
    clear_review_selection,
    plan_review_decision,
    select_review_index,
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QScrollArea,
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
    execute_review_intents,
    make_review_trash_dir,
)
from .preview import ImageOnlyPreviewAdapter, ReviewPreviewAdapter
from cosmos_toolbox.ui import ActionBar, PageHeader, PathField, SectionSurface, StatusBanner
from cosmos_toolbox.ui.primitives import FlowLayout


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
        self._session = ReviewSession()
        self.current_item: ReviewItem | None = None
        self.spec: ReviewSpec | None = None
        self.trash_dir: Path | None = None
        self._build_ui()
        self._connect_signals()

    @property
    def items(self) -> tuple[ReviewItem, ...]:
        return self._session.items

    @property
    def current_index(self) -> int:
        return self._session.current_index

    @property
    def accepted_count(self) -> int:
        return self._session.accepted_count

    @property
    def removed_count(self) -> int:
        return self._session.removed_count

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
        self.setProperty("ownsPageHeader", True)
        root = QVBoxLayout(self)
        root.setContentsMargins(14, 12, 14, 12)
        root.setSpacing(8)

        root.addWidget(PageHeader("数据审阅", "浏览图片样本，逐张决定保留或移除。"))

        source_bar = SectionSurface()
        source_bar.setAccessibleName("当前数据源")
        source_layout = source_bar.body_layout
        source_layout.setSpacing(8)
        source_layout.addWidget(QLabel("当前数据源"))
        self.source_summary = QLabel("尚未选择图片目录")
        self.source_summary.setAccessibleName("当前数据源")
        self.source_summary.setTextInteractionFlags(Qt.TextSelectableByMouse)
        source_layout.addWidget(self.source_summary, 1)
        self.btn_toggle_source = QPushButton("设置")
        self.btn_toggle_source.setMaximumWidth(72)
        self.btn_toggle_source.setAccessibleName("展开数据源设置")
        self.btn_toggle_source.setToolTip("展开或收起数据源路径设置")
        self.btn_load = QPushButton("加载 / 刷新")
        self.btn_load.setAccessibleName("加载或刷新样本")
        self.btn_load.setToolTip("按当前数据源加载样本")
        set_primary(self.btn_load)
        source_layout.addWidget(self.btn_toggle_source)
        source_layout.addWidget(self.btn_load)
        root.addWidget(source_bar)

        self.source_panel = SectionSurface()
        self.source_panel.setAccessibleName("数据源设置")
        source_layout = self.source_panel.body_layout
        mode_row = QHBoxLayout()
        mode_label = QLabel("范围")
        mode_label.setMinimumWidth(56)
        self.combo_mode = QComboBox()
        self.combo_mode.setAccessibleName("审阅范围")
        self.combo_mode.setToolTip("选择全部图片或仅显示带标注图片")
        self.combo_mode.addItem("全部图片（标注可选）", "unlabeled")
        self.combo_mode.addItem("仅显示有标注图片", "labeled")
        mode_row.addWidget(mode_label)
        mode_row.addWidget(self.combo_mode, 1)
        source_layout.addLayout(mode_row)
        self._image_path_field = PathField("图片目录", browse_text="浏览")
        self._label_path_field = PathField("标注目录", browse_text="浏览")
        self._save_path_field = PathField("保留目录", browse_text="浏览")
        self.edit_image_dir = self._image_path_field.line_edit
        self.edit_label_dir = self._label_path_field.line_edit
        self.edit_save_dir = self._save_path_field.line_edit
        self.btn_choose_image_dir = self._image_path_field.browse_button
        self.btn_choose_label_dir = self._label_path_field.browse_button
        self.btn_choose_save_dir = self._save_path_field.browse_button
        path_scroll = QScrollArea()
        path_scroll.setObjectName("sourcePathScroll")
        path_scroll.setWidgetResizable(True)
        path_scroll.setFrameShape(QFrame.NoFrame)
        path_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        path_scroll.setMaximumHeight(120)
        path_container = QWidget()
        path_flow = FlowLayout(path_container, h_spacing=8, v_spacing=6)
        path_scroll.setWidget(path_container)
        source_layout.addWidget(path_scroll)
        for field in (self._image_path_field, self._label_path_field, self._save_path_field):
            field.setMinimumWidth(240)
            field.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
            path_flow.addWidget(field)
        root.addWidget(self.source_panel)
        self.source_panel.setVisible(False)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)

        browser = SectionSurface()
        browser.setAccessibleName("样本浏览")
        browser.setMinimumWidth(220)
        browser.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        browser_layout = browser.body_layout
        browser_layout.setSpacing(7)
        browser_title_row = QHBoxLayout()
        browser_title_row.addWidget(QLabel("样本"))
        self.lbl_index = QLabel("0 / 0")
        self.lbl_index.setAccessibleName("样本进度")
        self.lbl_index.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        browser_title_row.addWidget(self.lbl_index, 1)
        browser_layout.addLayout(browser_title_row)
        self.file_list = QListWidget()
        self.file_list.setAccessibleName("样本文件列表")
        self.file_list.setToolTip("选择样本进行预览；可使用 A/D 切换")
        self.file_list.setMinimumHeight(120)
        browser_layout.addWidget(self.file_list, 1)
        self.review_summary = QLabel("标注 -  ·  保留 0  ·  移除 0")
        self.review_summary.setAccessibleName("审阅统计")
        self.review_summary.setWordWrap(True)
        browser_layout.addWidget(self.review_summary)
        splitter.addWidget(browser)

        preview = SectionSurface()
        preview.setAccessibleName("预览")
        preview_layout = preview.body_layout
        preview_layout.setSpacing(6)
        preview_header = QHBoxLayout()
        preview_header.addWidget(QLabel("预览"))
        self.lbl_current_name = QLabel("未加载图片")
        self.lbl_current_name.setAccessibleName("当前样本")
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

        action_bar = ActionBar()
        action_bar.setAccessibleName("审阅操作")
        self.btn_prev = QPushButton("上一张  A")
        self.btn_next = QPushButton("下一张  D")
        self.btn_accept = QPushButton("保留到目录  S")
        set_primary(self.btn_accept)
        self.btn_remove = QPushButton("移除  W")
        self.btn_complete = QPushButton("完成审阅")
        for button, tip in (
            (self.btn_prev, "上一张（快捷键 A）"),
            (self.btn_next, "下一张（快捷键 D）"),
            (self.btn_accept, "保留当前样本（快捷键 S）"),
            (self.btn_remove, "移除当前样本（快捷键 W）"),
            (self.btn_complete, "提交当前审阅结果"),
        ):
            button.setToolTip(tip)
            button.setAccessibleName(tip)
            action_bar.add_widget(button)
        root.addWidget(action_bar)

        self.status_banner = StatusBanner("设置数据源后加载样本。")
        self.status_label = self.status_banner.label
        root.addWidget(self.status_banner)

        # Compatibility attributes used by older callers during migration.
        self.btn_save = self.btn_accept
        self.btn_trash = self.btn_remove
        self.btn_apply_flow = self.btn_complete
        self.lbl_label_state = QLabel("无")
        self.lbl_point_count = QLabel("0")
        self.lbl_edge_count = QLabel("0")
        self.lbl_saved_count = QLabel("0")
        self.lbl_trash_count = QLabel("0")

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
        self._session = begin_review(collect_review_items(spec))
        self.current_item = None
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
        selected = select_review_index(self._session, index)
        if selected.current_index == self.current_index and self.current_item is not None:
            return
        item = selected.current_item
        assert item is not None
        try:
            image = read_image_bgr(item.image_path)
            document = self.preview_adapter.load(item.image_path, item.annotation_path, image)
            summary = self.preview_adapter.summarize(document)
        except Exception as exc:
            QMessageBox.critical(self, "加载失败", str(exc))
            return
        self._session = selected
        self.current_item = item
        self.file_list.blockSignals(True)
        self.file_list.setCurrentRow(selected.current_index)
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

    def _show_after_removal(self, index: int, message: str) -> None:
        if not self.items:
            self._clear_preview()
            self.status_label.setText(f"{message} 当前没有剩余样本。")
            self._refresh_counts()
            return
        self.jump_to_index(min(index, len(self.items) - 1))
        self.status_label.setText(message)

    def _clear_preview(self) -> None:
        self._session = clear_review_selection(self._session)
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
        transition = plan_review_decision(
            self._session,
            "accepted",
            self.spec.accepted_output,
            decision="accept",
        )
        if transition is None:
            return
        try:
            moved = execute_review_intents(transition.intents)
        except Exception as exc:
            QMessageBox.critical(self, "保留失败", str(exc))
            return
        self.file_list.blockSignals(True)
        self.file_list.takeItem(transition.removed_index)
        self.file_list.blockSignals(False)
        self._session = transition.session
        self.current_item = None
        self._show_after_removal(
            transition.removed_index,
            f"已保留: {', '.join(path.name for path in moved)}",
        )

    def move_current_to_trash(self) -> None:
        if self.spec is None:
            return
        if self.trash_dir is None:
            self.trash_dir = make_review_trash_dir(self.spec.trash_root or Path.cwd())
        transition = plan_review_decision(
            self._session,
            "removed",
            self.trash_dir,
            decision="remove",
        )
        if transition is None:
            return
        try:
            moved = execute_review_intents(transition.intents)
        except Exception as exc:
            QMessageBox.critical(self, "移除失败", str(exc))
            return
        self.file_list.blockSignals(True)
        self.file_list.takeItem(transition.removed_index)
        self.file_list.blockSignals(False)
        self._session = transition.session
        self.current_item = None
        self._show_after_removal(
            transition.removed_index,
            f"已移除: {', '.join(path.name for path in moved)}",
        )

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
