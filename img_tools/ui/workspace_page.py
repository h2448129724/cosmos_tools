"""Embeddable image workspace page.

The original image tool predates the toolbox shell and is intentionally kept
as :class:`img_tools.ui.main_window.MainWindow` for standalone launches.  This
module contains the small, shell-friendly surface used by the toolbox: image
selection, the interactive canvas/ROI seam, and compact task feedback.  Batch
operations remain discoverable through the toolbox capability catalogue rather
than adding another menu/dock chrome to an embedded page.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
    QApplication,
)

from cosmos_toolbox.ui import ActionBar, CollapsibleLogPanel, PageHeader, StatusBanner, set_ui_role
from img_tools.core.image_io import iter_image_files, read_image
from img_tools.core.models import Roi
from img_tools.ui.canvas import ImageCanvas


class ImageWorkspacePage(QWidget):
    """Native ``QWidget`` image surface mounted by the project workbench.

    This page deliberately does not own a menu bar, status bar, dock widget,
    or a second window.  ``MainWindow`` remains the compatibility shell for
    users launching the image tool directly; this class is the stable seam for
    the toolbox's embedded workspace.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("imageWorkspacePage")
        self._files: list[Path] = []
        self._current_path: Path | None = None
        self._current_image: np.ndarray | None = None
        self._project_context = None
        self._workspace_router = None
        self._workspace_key = "image"
        self._project_image_dir = ""
        self._rois: list[Roi] = []
        self._build_ui()

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 10, 12, 10)
        outer.setSpacing(8)

        header = PageHeader("素材与 ROI", "浏览项目图片、检查像素并在画布上配置 ROI。")
        outer.addWidget(header)

        action_bar = ActionBar()
        open_image = QPushButton("打开图片")
        open_image.clicked.connect(self._open_image)
        action_bar.add_widget(open_image)
        open_folder = QPushButton("打开目录")
        open_folder.clicked.connect(self._open_folder)
        action_bar.add_widget(open_folder)
        self.open_project_images = QPushButton("载入当前项目图片")
        self.open_project_images.setEnabled(False)
        self.open_project_images.clicked.connect(self._load_project_images)
        action_bar.add_widget(self.open_project_images)
        fit = QPushButton("适应窗口")
        fit.clicked.connect(self.canvas_fit)
        action_bar.add_widget(fit)
        self.roi_button = QPushButton("ROI 画布")
        self.roi_button.setCheckable(True)
        self.roi_button.toggled.connect(self._set_roi_mode)
        action_bar.add_widget(self.roi_button)
        outer.addWidget(action_bar)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self._build_file_panel())
        splitter.addWidget(self._build_canvas_panel())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([220, 720])
        outer.addWidget(splitter, 1)

        self.task_feedback = StatusBanner("就绪：打开图片后可直接在画布上拖拽 ROI。")
        self.task_feedback.setObjectName("imageTaskFeedback")
        outer.addWidget(self.task_feedback)
        self.log_panel = CollapsibleLogPanel("操作记录", expanded=False)
        self.log_box = self.log_panel.log
        self.log_box.setObjectName("imageTaskLog")
        self.log_box.setMaximumBlockCount(120)
        self.log_box.setMinimumHeight(56)
        self.log_box.setPlaceholderText("任务反馈")
        outer.addWidget(self.log_panel)

    def _build_file_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("imageFilePanel")
        set_ui_role(panel, "sectionSurface")
        panel.setMinimumWidth(170)
        panel.setMaximumWidth(260)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)
        title = QLabel("图片列表")
        set_ui_role(title, "sectionTitle")
        layout.addWidget(title)
        self.file_list = QListWidget()
        self.file_list.setObjectName("imageFileList")
        self.file_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.file_list.currentRowChanged.connect(self._select_file)
        layout.addWidget(self.file_list, 1)
        self.file_count = QLabel("0 张图片")
        set_ui_role(self.file_count, "muted")
        layout.addWidget(self.file_count)
        return panel

    def _build_canvas_panel(self) -> QWidget:
        panel = QWidget()
        panel.setObjectName("imageCanvasPanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.canvas = ImageCanvas()
        self.canvas.setMinimumSize(0, 0)
        self.canvas.pixelHovered.connect(self._show_pixel)
        self.canvas.coordinateClicked.connect(self._copy_coordinate)
        self.canvas.roiDrawn.connect(self._add_drawn_roi)
        self.canvas.roiMoved.connect(self._move_roi)
        layout.addWidget(self.canvas, 1)

        roi_form = QFrame()
        roi_form.setObjectName("imageRoiControls")
        roi_layout = QHBoxLayout(roi_form)
        roi_layout.setContentsMargins(0, 0, 0, 0)
        roi_layout.setSpacing(6)
        self.fixed_size_check = QCheckBox("固定尺寸")
        self.fixed_size_check.setAccessibleName("使用固定 ROI 尺寸")
        self.fixed_size_check.toggled.connect(self._sync_fixed_roi_size)
        roi_layout.addWidget(self.fixed_size_check)
        form = QFormLayout()
        self.width_box = self._spin_box(100)
        self.height_box = self._spin_box(100)
        self.width_box.setAccessibleName("ROI 宽度")
        self.height_box.setAccessibleName("ROI 高度")
        form.addRow("宽", self.width_box)
        form.addRow("高", self.height_box)
        self.width_box.valueChanged.connect(self._sync_fixed_roi_size)
        self.height_box.valueChanged.connect(self._sync_fixed_roi_size)
        roi_layout.addLayout(form)
        self.image_info = QLabel("未打开图片")
        self.image_info.setWordWrap(True)
        set_ui_role(self.image_info, "muted")
        roi_layout.addWidget(self.image_info, 1)
        layout.addWidget(roi_form)
        return panel

    @staticmethod
    def _spin_box(value: int) -> QSpinBox:
        box = QSpinBox()
        box.setRange(1, 1_000_000)
        box.setValue(value)
        box.setFixedWidth(80)
        return box

    def _open_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "打开图片", "", "图片 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp)")
        if path:
            self._set_files([Path(path)])

    def _open_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "打开图片目录", "")
        if folder:
            self._set_files(list(iter_image_files(folder)))

    def _load_project_images(self) -> None:
        if self._project_context is None:
            return
        state = self._project_context.state
        root = state.image_dir or state.dataset_root
        if root:
            self._set_files(list(iter_image_files(root, recursive=False)))

    def _set_files(self, files: list[Path]) -> None:
        self._files = files
        self.file_list.clear()
        for path in files:
            self.file_list.addItem(path.name)
        self.file_count.setText(f"{len(files)} 张图片")
        if files:
            self.file_list.setCurrentRow(0)
        else:
            self._current_path = None
            self._current_image = None
            self.canvas.set_image(None)
            self.image_info.setText("未打开图片")

    def _select_file(self, index: int) -> None:
        if not 0 <= index < len(self._files):
            return
        path = self._files[index]
        image = read_image(path)
        if image is None:
            self._append_log(f"无法读取图片：{path}")
            return
        self._current_path, self._current_image = path, image
        self.canvas.set_image(image)
        self.canvas.set_rois(self._rois)
        height, width = image.shape[:2]
        self.image_info.setText(f"{path.name} · {width}×{height}")
        if self._project_context is not None:
            self._project_context.update(selected_image=str(path))

    def bind_project_context(self, context) -> None:
        self._project_context = context
        self.apply_project_context(context.state)
        changed = getattr(context, "changed", None)
        if changed is not None:
            try:
                changed.connect(self.apply_project_context)
            except (AttributeError, TypeError):
                pass

    def apply_project_context(self, state) -> None:
        root = getattr(state, "image_dir", "") or getattr(state, "dataset_root", "")
        self._project_image_dir = str(root or "")
        self.open_project_images.setEnabled(bool(root))
        self.open_project_images.setToolTip(self._project_image_dir or "请先选择项目数据集")
        selected = Path(getattr(state, "selected_image", "")) if getattr(state, "selected_image", "") else None
        if selected and selected in self._files:
            self.file_list.setCurrentRow(self._files.index(selected))

    def bind_workspace_router(self, router, workspace_key: str = "image") -> None:
        self._workspace_router = router
        self._workspace_key = workspace_key

    def _set_roi_mode(self, enabled: bool) -> None:
        self.canvas.set_roi_mode(enabled)
        if enabled:
            self.task_feedback.set_status("ROI 模式：拖拽创建，框内拖动可调整位置。", "info")
        else:
            self.task_feedback.set_status("就绪：打开图片后可直接在画布上拖拽 ROI。", "neutral")

    def _sync_fixed_roi_size(self, *_args) -> None:
        size = (self.width_box.value(), self.height_box.value()) if self.fixed_size_check.isChecked() else None
        self.canvas.set_fixed_roi_size(size)

    def _add_drawn_roi(self, roi: Roi) -> None:
        self._rois.append(roi)
        self.canvas.set_rois(self._rois, len(self._rois) - 1)
        self._append_log(f"已添加 ROI：{roi.x},{roi.y} {roi.width}×{roi.height}")

    def _move_roi(self, index: int, roi: Roi) -> None:
        if 0 <= index < len(self._rois):
            self._rois[index] = roi
            self.canvas.set_rois(self._rois, index)

    def _show_pixel(self, x: int, y: int, values: tuple[int, ...]) -> None:
        self.image_info.setText(f"{self._current_path.name if self._current_path else '图片'} · ({x}, {y}) · {values}")

    def _copy_coordinate(self, x: int, y: int) -> None:
        app = QApplication.instance()
        if app is not None:
            app.clipboard().setText(f"({x}, {y})")
        self._append_log(f"坐标已复制：({x}, {y})")

    def canvas_fit(self) -> None:
        self.canvas.fit_to_view()
        self.canvas.update()

    def _append_log(self, message: str) -> None:
        self.log_box.appendPlainText(message)

    def shutdown(self) -> None:
        """Compatibility teardown hook used by :class:`PageRouter`."""
        self.canvas.set_roi_mode(False)

    def cancel(self) -> None:
        self.shutdown()


def workspace_page(parent: QWidget | None = None) -> ImageWorkspacePage:
    """Create the embeddable image page without constructing a main window."""

    return ImageWorkspacePage(parent)


__all__ = ["ImageWorkspacePage", "workspace_page"]
