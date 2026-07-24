"""Main window for the first inspector and precise-ROI workspace."""
from __future__ import annotations

from pathlib import Path
from datetime import datetime
import json

import numpy as np
from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtGui import QAction, QIcon, QKeySequence, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDockWidget,
    QDialog,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QLineEdit,
    QProgressBar,
    QDoubleSpinBox,
    QSplitter,
    QSpinBox,
    QStatusBar,
    QVBoxLayout,
    QWidget,
    QApplication,
)

from img_tools.core.crop import BatchCropResult, batch_crop, crop_image
from img_tools.core.image_io import iter_image_files, read_image, write_image
from img_tools.core.models import Roi
from img_tools.core.presets import load_roi_preset, save_roi_preset
from img_tools.core.tile import TileResult, batch_tile
from img_tools.core.tasking import TaskSummary, write_task_report
from img_tools.core.state import add_task_history, load_state, save_state
from img_tools.core.organize import batch_rename, classify_by_keywords, find_exact_duplicates, find_invalid_images, find_similar_images
from img_tools.core.filtering import ReviewMove, move_for_review, undo_review_move
from img_tools.core.labels import render_yolo_labels
from img_tools.core.mask import polygon_mask
from img_tools.core.enhance import enhance_image
from img_tools.core.pipeline import BatchPipelineResult, batch_pipeline, load_pipeline, run_pipeline, save_pipeline
from img_tools.core.preview import make_display_preview
from img_tools.core.transform import BatchTransformResult, batch_transform, transform_image
from img_tools.ui.canvas import ImageCanvas


class TaskWorker(QThread):
    """Shared cooperative-cancellation worker contract for every batch task."""
    progressChanged = Signal(int, int)
    completed = Signal(object)
    failed = Signal(str)


class BatchCropWorker(TaskWorker):
    def __init__(self, input_dir: Path, output_dir: Path, rois: list[Roi], reference_size: tuple[int, int], mode: str, recursive: bool, preserve_structure: bool, conflict_policy: str) -> None:
        super().__init__()
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.rois = rois
        self.reference_size = reference_size
        self.mode = mode
        self.recursive = recursive
        self.preserve_structure, self.conflict_policy = preserve_structure, conflict_policy

    def run(self) -> None:
        try:
            result = batch_crop(
                self.input_dir,
                self.output_dir,
                self.rois,
                reference_size=self.reference_size,
                coordinate_mode=self.mode,  # type: ignore[arg-type]
                recursive=self.recursive,
                preserve_structure=self.preserve_structure,
                conflict_policy=self.conflict_policy,  # type: ignore[arg-type]
                progress=lambda current, total: self.progressChanged.emit(current, total),
                should_cancel=self.isInterruptionRequested,
            )
            self.completed.emit(result)
        except Exception as error:  # UI boundary: show any unexpected processing failure.
            self.failed.emit(str(error))


class TileWorker(TaskWorker):
    def __init__(self, input_dir: Path, output_dir: Path, tile_width: int, tile_height: int, overlap: int, edge_policy: str, recursive: bool, preserve_structure: bool, conflict_policy: str) -> None:
        super().__init__()
        self.input_dir, self.output_dir = input_dir, output_dir
        self.tile_width, self.tile_height = tile_width, tile_height
        self.overlap, self.edge_policy, self.recursive = overlap, edge_policy, recursive
        self.preserve_structure, self.conflict_policy = preserve_structure, conflict_policy

    def run(self) -> None:
        try:
            result = batch_tile(
                self.input_dir,
                self.output_dir,
                self.tile_width,
                self.tile_height,
                overlap_percent=self.overlap,
                edge_policy=self.edge_policy,  # type: ignore[arg-type]
                recursive=self.recursive,
                preserve_structure=self.preserve_structure,
                conflict_policy=self.conflict_policy,  # type: ignore[arg-type]
                progress=lambda current, total: self.progressChanged.emit(current, total),
                should_cancel=self.isInterruptionRequested,
            )
            self.completed.emit(result)
        except Exception as error:
            self.failed.emit(str(error))


class TransformWorker(TaskWorker):
    def __init__(self, input_dir: Path, output_dir: Path, *, suffix: str, width: int | None, height: int | None, keep_aspect: bool, rotation: str, flip: str, grayscale: bool, recursive: bool, preserve_structure: bool, conflict_policy: str) -> None:
        super().__init__()
        self.input_dir, self.output_dir, self.suffix = input_dir, output_dir, suffix
        self.width, self.height, self.keep_aspect = width, height, keep_aspect
        self.rotation, self.flip, self.grayscale, self.recursive = rotation, flip, grayscale, recursive
        self.preserve_structure, self.conflict_policy = preserve_structure, conflict_policy

    def run(self) -> None:
        try:
            result = batch_transform(
                self.input_dir, self.output_dir, output_suffix=self.suffix, width=self.width, height=self.height,
                keep_aspect=self.keep_aspect, rotation=self.rotation, flip=self.flip, grayscale=self.grayscale,
                recursive=self.recursive, progress=lambda current, total: self.progressChanged.emit(current, total), should_cancel=self.isInterruptionRequested,
                preserve_structure=self.preserve_structure, conflict_policy=self.conflict_policy,  # type: ignore[arg-type]
            )
            self.completed.emit(result)
        except Exception as error:
            self.failed.emit(str(error))


class PipelineWorker(TaskWorker):
    def __init__(self, input_dir: Path, output_dir: Path, steps: list[dict], *, suffix: str, recursive: bool, preserve_structure: bool, conflict_policy: str) -> None:
        super().__init__()
        self.input_dir, self.output_dir, self.steps = input_dir, output_dir, steps
        self.suffix, self.recursive = suffix, recursive
        self.preserve_structure, self.conflict_policy = preserve_structure, conflict_policy

    def run(self) -> None:
        try:
            result = batch_pipeline(self.input_dir, self.output_dir, self.steps, output_suffix=self.suffix, recursive=self.recursive, preserve_structure=self.preserve_structure, conflict_policy=self.conflict_policy, progress=lambda current, total: self.progressChanged.emit(current, total), should_cancel=self.isInterruptionRequested)  # type: ignore[arg-type]
            self.completed.emit(result)
        except Exception as error:
            self.failed.emit(str(error))


class MainWindow(QMainWindow):
    """A focused first release: inspect pixels and create/crop exact ROIs."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("通用图像工具")
        self.resize(1460, 900)
        self._files: list[Path] = []
        self._state = load_state()
        self._current_path: Path | None = None
        self._current_image: np.ndarray | None = None
        self._reference_size: tuple[int, int] | None = None
        self._rois: list[Roi] = []
        self._syncing = False
        self._worker: BatchCropWorker | None = None
        self._tile_worker: TileWorker | None = None
        self._transform_worker: TransformWorker | None = None
        self._pipeline_worker: PipelineWorker | None = None
        self._active_worker: QThread | None = None
        self._review_history: list[ReviewMove] = []
        self._project_context = None
        self._project_image_dir = ""
        self._workspace_router = None
        self._workspace_key = "image"
        self._embedded_mode = False
        self._build_ui()
        self._build_actions()

    def bind_workspace_router(self, router, workspace_key: str = "image") -> None:
        """Route secondary views through the toolbox shell when embedded."""
        self._workspace_router = router
        self._workspace_key = workspace_key
        self._set_embedded_mode(True)

    def _set_embedded_mode(self, embedded: bool) -> None:
        self._embedded_mode = embedded
        for dock in self._workspace_docks():
            if embedded:
                dock.setFloating(False)
                dock.setFeatures(dock.features() & ~QDockWidget.DockWidgetFeature.DockWidgetFloatable)

    def _workspace_docks(self) -> tuple[QDockWidget, ...]:
        return (
            self.roi_dock,
            self.transform_dock,
            self.organize_dock,
            self.filter_dock,
            self.annotation_dock,
            self.enhance_dock,
            self.pipeline_dock,
        )

    def _build_ui(self) -> None:
        splitter = QSplitter()
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self._build_file_panel())
        splitter.addWidget(self._build_canvas_panel())
        splitter.setSizes([250, 1040])
        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.addWidget(splitter, 1)
        log_header = QHBoxLayout()
        log_header.addWidget(QLabel("<b>运行日志</b>"))
        log_header.addStretch()
        self.task_progress = QProgressBar()
        self.task_progress.setFixedWidth(180)
        self.task_progress.setVisible(False)
        log_header.addWidget(self.task_progress)
        self.cancel_task_button = QPushButton("取消任务")
        self.cancel_task_button.setEnabled(False)
        self.cancel_task_button.clicked.connect(self._cancel_active_task)
        log_header.addWidget(self.cancel_task_button)
        clear_log = QPushButton("清空")
        clear_log.clicked.connect(lambda: self.log_box.clear())
        log_header.addWidget(clear_log)
        root_layout.addLayout(log_header)
        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMaximumBlockCount(500)
        self.log_box.setFixedHeight(110)
        root_layout.addWidget(self.log_box)
        self.setCentralWidget(root)
        self.roi_dock = QDockWidget("ROI 工具", self)
        self.roi_dock.setObjectName("roiDock")
        self.roi_dock.setWidget(self._build_roi_panel())
        self.addDockWidget(Qt.RightDockWidgetArea, self.roi_dock)
        self.roi_dock.visibilityChanged.connect(self._on_roi_tool_visibility_changed)
        self.roi_dock.hide()
        self.transform_dock = QDockWidget("基础处理", self)
        self.transform_dock.setObjectName("transformDock")
        self.transform_dock.setWidget(self._build_transform_panel())
        self.addDockWidget(Qt.RightDockWidgetArea, self.transform_dock)
        self.transform_dock.hide()
        self.organize_dock = QDockWidget("数据整理", self)
        self.organize_dock.setObjectName("organizeDock")
        self.organize_dock.setWidget(self._build_organize_panel())
        self.addDockWidget(Qt.RightDockWidgetArea, self.organize_dock)
        self.organize_dock.hide()
        self.filter_dock = QDockWidget("人工筛选", self)
        self.filter_dock.setObjectName("filterDock")
        self.filter_dock.setWidget(self._build_filter_panel())
        self.addDockWidget(Qt.RightDockWidgetArea, self.filter_dock)
        self.filter_dock.hide()
        self.annotation_dock = QDockWidget("Mask 与标签", self)
        self.annotation_dock.setObjectName("annotationDock")
        self.annotation_dock.setWidget(self._build_annotation_panel())
        self.addDockWidget(Qt.RightDockWidgetArea, self.annotation_dock)
        self.annotation_dock.hide()
        self.enhance_dock = QDockWidget("图像增强", self)
        self.enhance_dock.setObjectName("enhanceDock")
        self.enhance_dock.setWidget(self._build_enhance_panel())
        self.addDockWidget(Qt.RightDockWidgetArea, self.enhance_dock)
        self.enhance_dock.hide()
        self.pipeline_dock = QDockWidget("处理流水线", self)
        self.pipeline_dock.setObjectName("pipelineDock")
        self.pipeline_dock.setWidget(self._build_pipeline_panel())
        self.addDockWidget(Qt.RightDockWidgetArea, self.pipeline_dock)
        self.pipeline_dock.hide()
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("就绪：打开图片后可点击设起点，或直接拖拽 ROI。")

    def _build_file_panel(self) -> QWidget:
        panel = QFrame()
        layout = QVBoxLayout(panel)
        layout.addWidget(QLabel("<b>图片列表</b>"))
        open_image = QPushButton("打开图片")
        open_image.clicked.connect(self._open_image)
        open_folder = QPushButton("打开文件夹")
        open_folder.clicked.connect(self._open_folder)
        self.open_project_images = QPushButton("载入当前项目图片")
        self.open_project_images.setEnabled(False)
        self.open_project_images.clicked.connect(self._load_project_images)
        layout.addWidget(open_image)
        layout.addWidget(open_folder)
        layout.addWidget(self.open_project_images)
        self.file_list = QListWidget()
        self.file_list.setIconSize(QPixmap(72, 72).size())
        self.file_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.file_list.currentRowChanged.connect(self._select_file)
        layout.addWidget(self.file_list, 1)
        self.file_count = QLabel("0 张图片")
        self.file_count.setStyleSheet("color:#667085")
        layout.addWidget(self.file_count)
        return panel

    def _build_canvas_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        toolbar = QHBoxLayout()
        fit = QPushButton("适应窗口")
        fit.clicked.connect(self.canvas_fit)
        toolbar.addWidget(fit)
        roi_tool = QPushButton("ROI 工具")
        roi_tool.clicked.connect(self._show_roi_tool)
        toolbar.addWidget(roi_tool)
        transform_tool = QPushButton("基础处理")
        transform_tool.clicked.connect(self._show_transform_tool)
        toolbar.addWidget(transform_tool)
        toolbar.addWidget(QLabel("滚轮缩放 · 中键平移 · 单击取起点 · 拖拽创建 ROI"))
        toolbar.addStretch()
        layout.addLayout(toolbar)
        self.canvas = ImageCanvas()
        self.canvas.pixelHovered.connect(self._show_pixel)
        self.canvas.coordinateClicked.connect(self._copy_coordinate)
        self.canvas.pointSelected.connect(self._set_origin_from_canvas)
        self.canvas.roiDrawn.connect(self._add_drawn_roi)
        layout.addWidget(self.canvas, 1)
        self.image_info = QLabel("未打开图片")
        self.image_info.setStyleSheet("color:#667085")
        layout.addWidget(self.image_info)
        return panel

    def _build_roi_panel(self) -> QWidget:
        panel = QFrame()
        layout = QVBoxLayout(panel)
        layout.addWidget(QLabel("<b>精确 ROI</b>"))
        layout.addWidget(QLabel("起点为左上角；裁剪范围不包含右、下边界。"))
        form = QFormLayout()
        self.x_box = self._spin_box()
        self.y_box = self._spin_box()
        self.width_box = self._spin_box(minimum=1, value=100)
        self.height_box = self._spin_box(minimum=1, value=100)
        form.addRow("起点 X", self.x_box)
        form.addRow("起点 Y", self.y_box)
        form.addRow("宽度", self.width_box)
        form.addRow("高度", self.height_box)
        layout.addLayout(form)
        self.roi_feedback = QLabel("终点：X2=100，Y2=100")
        self.roi_feedback.setStyleSheet("color:#0984e3")
        layout.addWidget(self.roi_feedback)
        for box in (self.x_box, self.y_box, self.width_box, self.height_box):
            box.valueChanged.connect(self._refresh_preview)
        buttons = QHBoxLayout()
        add_button = QPushButton("添加 ROI")
        add_button.clicked.connect(self._add_or_update_roi)
        update_button = QPushButton("更新选中")
        update_button.clicked.connect(lambda: self._add_or_update_roi(update=True))
        buttons.addWidget(add_button)
        buttons.addWidget(update_button)
        layout.addLayout(buttons)
        self.roi_list = QListWidget()
        self.roi_list.currentRowChanged.connect(self._select_roi)
        layout.addWidget(self.roi_list, 1)
        remove_button = QPushButton("删除选中 ROI")
        remove_button.clicked.connect(self._remove_roi)
        clear_button = QPushButton("清空全部 ROI")
        clear_button.clicked.connect(self._clear_rois)
        layout.addWidget(remove_button)
        layout.addWidget(clear_button)
        preset_buttons = QHBoxLayout()
        save_preset = QPushButton("保存预设")
        save_preset.clicked.connect(self._save_roi_preset)
        load_preset = QPushButton("加载预设")
        load_preset.clicked.connect(self._load_roi_preset)
        preset_buttons.addWidget(save_preset)
        preset_buttons.addWidget(load_preset)
        layout.addLayout(preset_buttons)
        layout.addWidget(QLabel("<b>导出</b>"))
        current_crop = QPushButton("导出当前 ROI")
        current_crop.clicked.connect(self._export_current_crop)
        layout.addWidget(current_crop)
        self.mapping_combo = QComboBox()
        self.mapping_combo.addItem("绝对像素坐标", "absolute")
        self.mapping_combo.addItem("按参考图等比例缩放", "scaled")
        layout.addWidget(self.mapping_combo)
        self.recursive_check = QCheckBox("递归处理子目录")
        layout.addWidget(self.recursive_check)
        self.preserve_structure_check = QCheckBox("保持原目录结构")
        self.preserve_structure_check.setChecked(True)
        layout.addWidget(self.preserve_structure_check)
        self.conflict_policy = QComboBox()
        self.conflict_policy.addItem("重名时自动改名", "rename")
        self.conflict_policy.addItem("重名时跳过", "skip")
        self.conflict_policy.addItem("重名时覆盖", "overwrite")
        layout.addWidget(self.conflict_policy)
        batch_button = QPushButton("批量裁剪目录")
        batch_button.clicked.connect(self._start_batch_crop)
        layout.addWidget(batch_button)
        self.batch_status = QLabel("")
        self.batch_status.setWordWrap(True)
        self.batch_status.setStyleSheet("color:#667085")
        layout.addWidget(self.batch_status)
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        layout.addWidget(line)
        layout.addWidget(QLabel("<b>自动切片</b>"))
        tile_form = QFormLayout()
        self.tile_width_box = self._spin_box(minimum=1, value=512)
        self.tile_height_box = self._spin_box(minimum=1, value=512)
        self.overlap_box = self._spin_box(minimum=0, value=0)
        self.overlap_box.setMaximum(99)
        self.edge_policy = QComboBox()
        self.edge_policy.addItem("边缘平移，保持完整块", "shift")
        self.edge_policy.addItem("丢弃不足块", "discard")
        self.edge_policy.addItem("保留不足块", "partial")
        self.edge_policy.addItem("补黑边到固定尺寸", "pad")
        tile_form.addRow("块宽", self.tile_width_box)
        tile_form.addRow("块高", self.tile_height_box)
        tile_form.addRow("重叠率 %", self.overlap_box)
        tile_form.addRow("边缘策略", self.edge_policy)
        layout.addLayout(tile_form)
        tile_button = QPushButton("自动切片目录")
        tile_button.clicked.connect(self._start_tile)
        layout.addWidget(tile_button)
        self.tile_status = QLabel("")
        self.tile_status.setWordWrap(True)
        self.tile_status.setStyleSheet("color:#667085")
        layout.addWidget(self.tile_status)
        return panel

    @staticmethod
    def _spin_box(*, minimum: int = 0, value: int = 0) -> QSpinBox:
        box = QSpinBox()
        box.setRange(minimum, 1_000_000)
        box.setValue(value)
        box.setSingleStep(1)
        return box

    def _build_actions(self) -> None:
        file_menu = self.menuBar().addMenu("文件")
        open_image = QAction("打开图片", self)
        open_image.setShortcut(QKeySequence.Open)
        open_image.triggered.connect(self._open_image)
        file_menu.addAction(open_image)
        open_folder = QAction("打开文件夹", self)
        open_folder.setShortcut("Ctrl+Shift+O")
        open_folder.triggered.connect(self._open_folder)
        file_menu.addAction(open_folder)
        view_menu = self.menuBar().addMenu("视图")
        fit = QAction("适应窗口", self)
        fit.setShortcut("F")
        fit.triggered.connect(self.canvas_fit)
        view_menu.addAction(fit)
        tools_menu = self.menuBar().addMenu("工具")
        roi_tool = QAction("ROI 工具", self)
        roi_tool.setShortcut("R")
        roi_tool.triggered.connect(self._show_roi_tool)
        tools_menu.addAction(roi_tool)
        transform_tool = QAction("基础处理", self)
        transform_tool.setShortcut("Ctrl+T")
        transform_tool.triggered.connect(self._show_transform_tool)
        tools_menu.addAction(transform_tool)
        organize_tool = QAction("数据整理", self)
        organize_tool.triggered.connect(self._show_organize_tool)
        tools_menu.addAction(organize_tool)
        filter_tool = QAction("人工筛选", self)
        filter_tool.triggered.connect(self._show_filter_tool)
        tools_menu.addAction(filter_tool)
        annotation_tool = QAction("Mask 与标签", self)
        annotation_tool.triggered.connect(self._show_annotation_tool)
        tools_menu.addAction(annotation_tool)
        enhance_tool = QAction("图像增强", self)
        enhance_tool.triggered.connect(self._show_enhance_tool)
        tools_menu.addAction(enhance_tool)
        pipeline_tool = QAction("处理流水线", self)
        pipeline_tool.triggered.connect(self._show_pipeline_tool)
        tools_menu.addAction(pipeline_tool)
        history_menu = self.menuBar().addMenu("历史")
        show_history = QAction("查看任务历史", self)
        show_history.triggered.connect(self._show_task_history)
        history_menu.addAction(show_history)

    def _build_transform_panel(self) -> QWidget:
        panel = QFrame()
        layout = QVBoxLayout(panel)
        layout.addWidget(QLabel("<b>基础处理</b>"))
        layout.addWidget(QLabel("调整尺寸、旋转、翻转和格式转换；原图不会被修改。"))
        form = QFormLayout()
        self.transform_width = self._spin_box(minimum=0, value=0)
        self.transform_height = self._spin_box(minimum=0, value=0)
        self.transform_keep_aspect = QCheckBox("保持比例")
        self.transform_keep_aspect.setChecked(True)
        self.transform_rotation = QComboBox()
        self.transform_rotation.addItem("不旋转", "none")
        self.transform_rotation.addItem("顺时针 90°", "cw90")
        self.transform_rotation.addItem("逆时针 90°", "ccw90")
        self.transform_rotation.addItem("旋转 180°", "180")
        self.transform_flip = QComboBox()
        self.transform_flip.addItem("不翻转", "none")
        self.transform_flip.addItem("水平翻转", "horizontal")
        self.transform_flip.addItem("垂直翻转", "vertical")
        self.transform_gray = QCheckBox("转为灰度图")
        self.transform_format = QComboBox()
        self.transform_format.addItem("PNG", ".png")
        self.transform_format.addItem("JPEG", ".jpg")
        self.transform_format.addItem("WebP", ".webp")
        form.addRow("目标宽（0=不指定）", self.transform_width)
        form.addRow("目标高（0=不指定）", self.transform_height)
        form.addRow("尺寸选项", self.transform_keep_aspect)
        form.addRow("旋转", self.transform_rotation)
        form.addRow("翻转", self.transform_flip)
        form.addRow("颜色", self.transform_gray)
        form.addRow("输出格式", self.transform_format)
        layout.addLayout(form)
        export = QPushButton("处理当前图片并导出")
        export.clicked.connect(self._export_transformed_image)
        layout.addWidget(export)
        self.transform_recursive = QCheckBox("批量时递归处理子目录")
        layout.addWidget(self.transform_recursive)
        self.transform_preserve_structure = QCheckBox("保持原目录结构")
        self.transform_preserve_structure.setChecked(True)
        layout.addWidget(self.transform_preserve_structure)
        self.transform_conflict_policy = QComboBox()
        self.transform_conflict_policy.addItems(["重名时自动改名", "重名时跳过", "重名时覆盖"])
        self.transform_conflict_policy.setItemData(0, "rename")
        self.transform_conflict_policy.setItemData(1, "skip")
        self.transform_conflict_policy.setItemData(2, "overwrite")
        layout.addWidget(self.transform_conflict_policy)
        batch = QPushButton("批量处理目录")
        batch.clicked.connect(self._start_batch_transform)
        layout.addWidget(batch)
        self.transform_status = QLabel("")
        self.transform_status.setWordWrap(True)
        self.transform_status.setStyleSheet("color:#667085")
        layout.addWidget(self.transform_status)
        layout.addStretch()
        return panel

    def _build_organize_panel(self) -> QWidget:
        panel = QFrame()
        layout = QVBoxLayout(panel)
        layout.addWidget(QLabel("<b>数据整理</b>"))
        layout.addWidget(QLabel("关键词用逗号分隔，按文件名归类。"))
        self.keyword_entry = QLineEdit()
        self.keyword_entry.setPlaceholderText("例如：left, right, defect")
        layout.addWidget(self.keyword_entry)
        self.keyword_mode = QComboBox()
        self.keyword_mode.addItem("复制到分类目录", "copy")
        self.keyword_mode.addItem("移动到分类目录", "move")
        layout.addWidget(self.keyword_mode)
        classify = QPushButton("按关键词分类目录")
        classify.clicked.connect(self._classify_keywords)
        layout.addWidget(classify)
        layout.addWidget(QLabel("<b>文件检查</b>"))
        invalid = QPushButton("扫描损坏图片")
        invalid.clicked.connect(self._scan_invalid_images)
        duplicates = QPushButton("扫描完全重复图片")
        duplicates.clicked.connect(self._scan_duplicates)
        similar = QPushButton("扫描视觉相似图片")
        similar.clicked.connect(self._scan_similar)
        layout.addWidget(invalid)
        layout.addWidget(duplicates)
        layout.addWidget(similar)
        layout.addWidget(QLabel("<b>批量重命名</b>"))
        self.rename_prefix = QLineEdit()
        self.rename_prefix.setPlaceholderText("例如：camera_a")
        layout.addWidget(self.rename_prefix)
        rename = QPushButton("按序号重命名目录")
        rename.clicked.connect(self._batch_rename)
        layout.addWidget(rename)
        layout.addStretch()
        return panel

    def _build_filter_panel(self) -> QWidget:
        panel = QFrame()
        layout = QVBoxLayout(panel)
        layout.addWidget(QLabel("<b>人工筛选</b>"))
        layout.addWidget(QLabel("对当前浏览图片执行移动；操作可撤销。"))
        self.keep_dir_entry = QLineEdit()
        self.reject_dir_entry = QLineEdit()
        for label, entry, callback in (("保留目录", self.keep_dir_entry, lambda: self._choose_review_dir(self.keep_dir_entry)), ("拒绝目录", self.reject_dir_entry, lambda: self._choose_review_dir(self.reject_dir_entry))):
            row = QHBoxLayout()
            row.addWidget(QLabel(label))
            row.addWidget(entry)
            button = QPushButton("选择")
            button.clicked.connect(callback)
            row.addWidget(button)
            layout.addLayout(row)
        keep = QPushButton("保留当前图片")
        keep.clicked.connect(lambda: self._review_current("keep"))
        reject = QPushButton("拒绝当前图片")
        reject.clicked.connect(lambda: self._review_current("reject"))
        undo = QPushButton("撤销上一步")
        undo.clicked.connect(self._undo_review)
        layout.addWidget(keep)
        layout.addWidget(reject)
        layout.addWidget(undo)
        layout.addStretch()
        return panel

    def _build_annotation_panel(self) -> QWidget:
        panel = QFrame()
        layout = QVBoxLayout(panel)
        layout.addWidget(QLabel("<b>多边形 Mask</b>"))
        layout.addWidget(QLabel("粘贴 JSON 坐标，例如：[[[10,10],[100,10],[10,100]]]") )
        self.polygon_entry = QPlainTextEdit()
        self.polygon_entry.setPlaceholderText("[[[10,10],[100,10],[10,100]]]")
        self.polygon_entry.setFixedHeight(90)
        layout.addWidget(self.polygon_entry)
        create_mask = QPushButton("按当前图片尺寸导出 Mask")
        create_mask.clicked.connect(self._export_polygon_mask)
        layout.addWidget(create_mask)
        layout.addWidget(QLabel("<b>YOLO 标签可视化</b>"))
        render_label = QPushButton("选择 YOLO 标签并显示")
        render_label.clicked.connect(self._render_yolo_label)
        layout.addWidget(render_label)
        layout.addStretch()
        return panel

    def _build_enhance_panel(self) -> QWidget:
        panel = QFrame()
        form = QFormLayout(panel)
        self.enhance_brightness = QSpinBox()
        self.enhance_brightness.setRange(-255, 255)
        self.enhance_contrast = QDoubleSpinBox()
        self.enhance_contrast.setRange(0.1, 4.0)
        self.enhance_contrast.setValue(1.0)
        self.enhance_gamma = QDoubleSpinBox()
        self.enhance_gamma.setRange(0.1, 4.0)
        self.enhance_gamma.setValue(1.0)
        self.enhance_blur = QSpinBox()
        self.enhance_blur.setRange(0, 20)
        self.enhance_threshold = QSpinBox()
        self.enhance_threshold.setRange(-1, 255)
        self.enhance_threshold.setValue(-1)
        self.enhance_sharpen = QCheckBox("锐化")
        form.addRow("亮度", self.enhance_brightness)
        form.addRow("对比度", self.enhance_contrast)
        form.addRow("Gamma", self.enhance_gamma)
        form.addRow("模糊半径", self.enhance_blur)
        form.addRow("阈值（-1=关闭）", self.enhance_threshold)
        form.addRow("选项", self.enhance_sharpen)
        export = QPushButton("增强当前图片并导出")
        export.clicked.connect(self._export_enhanced)
        form.addRow(export)
        return panel

    def _build_pipeline_panel(self) -> QWidget:
        panel = QFrame()
        layout = QVBoxLayout(panel)
        layout.addWidget(QLabel("<b>处理流水线</b>"))
        layout.addWidget(QLabel("JSON 步骤：transform 或 enhance。"))
        self.pipeline_entry = QPlainTextEdit()
        self.pipeline_entry.setPlainText('[\n  {"operation": "transform", "options": {"width": 1024}},\n  {"operation": "enhance", "options": {"brightness": 10}}\n]')
        layout.addWidget(self.pipeline_entry)
        controls = QHBoxLayout()
        load = QPushButton("加载")
        load.clicked.connect(self._load_pipeline)
        save = QPushButton("保存")
        save.clicked.connect(self._save_pipeline)
        run = QPushButton("执行并导出")
        run.clicked.connect(self._run_pipeline_export)
        batch = QPushButton("批量执行目录")
        batch.clicked.connect(self._start_batch_pipeline)
        controls.addWidget(load)
        controls.addWidget(save)
        controls.addWidget(run)
        controls.addWidget(batch)
        layout.addLayout(controls)
        return panel

    def _open_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "打开图片", self._recent_dir("images"), "图片 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp)")
        if path:
            self._remember_dir("images", str(Path(path).parent))
            self._set_files([Path(path)])

    def _open_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "打开图片目录", self._recent_dir("images"))
        if folder:
            self._remember_dir("images", folder)
            self._set_files(list(iter_image_files(folder)))

    def _load_project_images(self) -> None:
        if not self._project_image_dir:
            self.status.showMessage("当前项目尚未配置图片目录。", 5000)
            return
        directory = Path(self._project_image_dir)
        if not directory.is_dir():
            self.status.showMessage(f"项目图片目录不存在：{directory}", 6000)
            return
        self._remember_dir("images", str(directory))
        self._set_files(list(iter_image_files(directory)))
        self._append_log(f"已载入当前项目图片目录：{directory}")

    def _set_files(self, files: list[Path]) -> None:
        self._files = files
        self.file_list.blockSignals(True)
        self.file_list.clear()
        for path in files:
            item = QListWidgetItem(path.name)
            item.setToolTip(str(path))
            thumbnail = QPixmap(str(path))
            if not thumbnail.isNull():
                item.setIcon(QIcon(thumbnail.scaled(72, 72, Qt.KeepAspectRatio, Qt.SmoothTransformation)))
            self.file_list.addItem(item)
        self.file_list.blockSignals(False)
        self.file_count.setText(f"{len(files)} 张图片")
        if files:
            self.file_list.setCurrentRow(0)
        else:
            self.status.showMessage("该目录没有可打开的图片。", 5000)

    def _select_file(self, index: int) -> None:
        if not 0 <= index < len(self._files):
            return
        path = self._files[index]
        image = read_image(path)
        if image is None:
            QMessageBox.warning(self, "无法打开", f"无法读取图片：\n{path}")
            return
        self._current_path, self._current_image = path, image
        if self._project_context is not None:
            self._project_context.update(selected_image=str(path))
        self._rois.clear()
        self.roi_list.clear()
        height, width = image.shape[:2]
        self._reference_size = (width, height)
        self.canvas.set_image(make_display_preview(image), source_size=(width, height))
        self.canvas.set_roi_mode(self.roi_dock.isVisible())
        self.x_box.setRange(0, max(width - 1, 0))
        self.y_box.setRange(0, max(height - 1, 0))
        self.width_box.setMaximum(max(width * 4, 1))
        self.height_box.setMaximum(max(height * 4, 1))
        self._syncing = True
        self.x_box.setValue(0)
        self.y_box.setValue(0)
        self.width_box.setValue(width)
        self.height_box.setValue(height)
        self._syncing = False
        channels = 1 if image.ndim == 2 else image.shape[2]
        self.image_info.setText(f"{path.name}  |  {width} × {height}  |  {channels} 通道  |  {image.dtype}")
        self._append_log(f"已打开图片：{path.name}（{width} × {height}）")
        if self.roi_dock.isVisible():
            self._refresh_preview()

    def bind_project_context(self, context) -> None:
        self._project_context = context
        self.apply_project_context(context.state)

    def apply_project_context(self, state) -> None:
        self._project_image_dir = state.image_dir or state.dataset_root
        enabled = bool(self._project_image_dir)
        self.open_project_images.setEnabled(enabled)
        self.open_project_images.setToolTip(self._project_image_dir if enabled else "请先在顶部选择项目数据集")
        if enabled:
            self.status.showMessage(f"当前项目图片目录：{self._project_image_dir}", 5000)

    def _show_pixel(self, x: int, y: int, values: tuple[int, ...]) -> None:
        if len(values) == 1:
            text = f"坐标：({x}, {y})   灰度：{values[0]}"
        elif len(values) == 3:
            b, g, r = values
            text = f"坐标：({x}, {y})   RGB：({r}, {g}, {b})"
        else:
            b, g, r, a = values
            text = f"坐标：({x}, {y})   RGBA：({r}, {g}, {b}, {a})"
        self.status.showMessage(text)

    def _set_origin_from_canvas(self, x: int, y: int) -> None:
        self._syncing = True
        self.x_box.setValue(x)
        self.y_box.setValue(y)
        self._syncing = False
        self._refresh_preview()
        self.status.showMessage(f"ROI 起点已设为 ({x}, {y})；请调整宽度和高度后添加。", 5000)

    def _copy_coordinate(self, x: int, y: int) -> None:
        QApplication.clipboard().setText(f"({x}, {y})")
        self.status.showMessage(f"坐标已复制：({x}, {y})", 3000)
        self._append_log(f"坐标已复制：({x}, {y})")

    def _append_log(self, message: str) -> None:
        self.log_box.appendPlainText(f"[{datetime.now():%H:%M:%S}] {message}")

    def _roi_from_form(self, *, name: str = "ROI") -> Roi:
        return Roi(self.x_box.value(), self.y_box.value(), self.width_box.value(), self.height_box.value(), name)

    def _refresh_preview(self) -> None:
        if self._syncing:
            return
        if not self.roi_dock.isVisible():
            self.canvas.set_preview_roi(None)
            return
        roi = self._roi_from_form()
        self.roi_feedback.setText(f"终点：X2={roi.x2}，Y2={roi.y2}；尺寸：{roi.width} × {roi.height}")
        selected = self.roi_list.currentRow()
        if 0 <= selected < len(self._rois) and self._rois[selected].as_xywh() == roi.as_xywh():
            self.canvas.set_preview_roi(None)
        else:
            self.canvas.set_preview_roi(roi)

    def _add_drawn_roi(self, roi: Roi) -> None:
        self._syncing = True
        self.x_box.setValue(roi.x)
        self.y_box.setValue(roi.y)
        self.width_box.setValue(roi.width)
        self.height_box.setValue(roi.height)
        self._syncing = False
        self._add_or_update_roi()

    def _add_or_update_roi(self, *, update: bool = False) -> None:
        selected = self.roi_list.currentRow()
        name = self._rois[selected].name if update and 0 <= selected < len(self._rois) else f"ROI {len(self._rois) + 1}"
        roi = self._roi_from_form(name=name)
        if update:
            if not 0 <= selected < len(self._rois):
                self.status.showMessage("请先选中要更新的 ROI。", 4000)
                return
            self._rois[selected] = roi
        else:
            self._rois.append(roi)
            selected = len(self._rois) - 1
        self._refresh_roi_list(selected)

    def _refresh_roi_list(self, selected: int = -1) -> None:
        self.roi_list.blockSignals(True)
        self.roi_list.clear()
        for index, roi in enumerate(self._rois, start=1):
            self.roi_list.addItem(f"{index}. ({roi.x}, {roi.y})  {roi.width} × {roi.height}")
        self.roi_list.setCurrentRow(selected)
        self.roi_list.blockSignals(False)
        self.canvas.set_rois(self._rois, selected)
        self.canvas.set_preview_roi(self._roi_from_form())

    def _select_roi(self, index: int) -> None:
        if not 0 <= index < len(self._rois):
            self.canvas.set_rois(self._rois, -1)
            return
        roi = self._rois[index]
        self._syncing = True
        self.x_box.setValue(roi.x)
        self.y_box.setValue(roi.y)
        self.width_box.setValue(roi.width)
        self.height_box.setValue(roi.height)
        self._syncing = False
        self.canvas.set_rois(self._rois, index)
        self._refresh_preview()

    def _remove_roi(self) -> None:
        index = self.roi_list.currentRow()
        if 0 <= index < len(self._rois):
            self._rois.pop(index)
            self._refresh_roi_list(min(index, len(self._rois) - 1))

    def _clear_rois(self) -> None:
        self._rois.clear()
        self._refresh_roi_list()

    def _save_roi_preset(self) -> None:
        if not self._rois:
            self.status.showMessage("没有可保存的 ROI。", 4000)
            return
        path, _ = QFileDialog.getSaveFileName(self, "保存 ROI 预设", "roi_preset.json", "JSON (*.json)")
        if not path:
            return
        try:
            save_roi_preset(path, self._rois, reference_size=self._reference_size)
            self._append_log(f"已保存 ROI 预设：{Path(path).name}")
            self.status.showMessage("ROI 预设已保存。", 5000)
        except (ValueError, OSError) as error:
            QMessageBox.warning(self, "无法保存预设", str(error))

    def _load_roi_preset(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "加载 ROI 预设", "", "JSON (*.json)")
        if not path:
            return
        try:
            rois, reference = load_roi_preset(path)
            self._rois = rois
            if reference:
                self._reference_size = reference
            self._refresh_roi_list(0 if rois else -1)
            self._append_log(f"已加载 ROI 预设：{Path(path).name}（{len(rois)} 个 ROI）")
            self.status.showMessage(f"已加载 {len(rois)} 个 ROI。", 5000)
        except (ValueError, OSError, KeyError, TypeError) as error:
            QMessageBox.warning(self, "无法加载预设", str(error))

    def canvas_fit(self) -> None:
        self.canvas.fit_to_view()
        self.canvas.update()

    def _selected_or_form_roi(self) -> Roi:
        index = self.roi_list.currentRow()
        return self._rois[index] if 0 <= index < len(self._rois) else self._roi_from_form()

    def _export_current_crop(self) -> None:
        if self._current_image is None or self._current_path is None:
            self.status.showMessage("请先打开图片。", 4000)
            return
        path, _ = QFileDialog.getSaveFileName(self, "保存裁剪结果", str(self._current_path.with_name(f"{self._current_path.stem}_roi.png")), "PNG (*.png);;JPEG (*.jpg *.jpeg);;WebP (*.webp)")
        if not path:
            return
        try:
            cropped, effective = crop_image(self._current_image, self._selected_or_form_roi())
            write_image(path, cropped)
            self.status.showMessage(f"已导出 {effective.width} × {effective.height}：{path}", 6000)
        except (ValueError, OSError) as error:
            QMessageBox.warning(self, "无法导出", str(error))

    def _start_batch_crop(self) -> None:
        if not self._rois or self._reference_size is None:
            QMessageBox.information(self, "需要 ROI", "请先打开参考图片并至少添加一个 ROI。")
            return
        input_dir = QFileDialog.getExistingDirectory(self, "选择待裁剪图片目录")
        if not input_dir:
            return
        output_dir = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if not output_dir:
            return
        self._worker = BatchCropWorker(
            Path(input_dir),
            Path(output_dir),
            list(self._rois),
            self._reference_size,
            str(self.mapping_combo.currentData()),
            self.recursive_check.isChecked(),
            self.preserve_structure_check.isChecked(), str(self.conflict_policy.currentData()),
        )
        self._worker.progressChanged.connect(self._on_batch_progress)
        self._worker.completed.connect(self._on_batch_completed)
        self._worker.failed.connect(self._on_batch_failed)
        self.batch_status.setText("正在批量裁剪…")
        self._append_log(f"开始批量裁剪：{input_dir} → {output_dir}，ROI {len(self._rois)} 个")
        self._worker.start()
        self._begin_task_progress(self._worker)

    def _on_batch_progress(self, current: int, total: int) -> None:
        self.batch_status.setText(f"正在处理：{current} / {total}")
        self._update_task_progress(current, total)

    def _on_batch_completed(self, result: BatchCropResult) -> None:
        worker = self._worker
        details = f"{'已取消：' if result.cancelled else '完成：'}处理 {result.processed_files} 张，写入 {result.written_files} 张。"
        if result.errors:
            details += f"失败 {len(result.errors)} 项。"
            self.batch_status.setText(details + "\n" + "\n".join(result.errors[:3]))
        else:
            self.batch_status.setText(details)
        self.status.showMessage(details, 8000)
        self._append_log(details)
        if worker:
            write_task_report(worker.output_dir, TaskSummary("crop", result.processed_files, result.written_files, result.errors, result.cancelled), parameters={"coordinate_mode": worker.mode, "roi_count": len(worker.rois)})
        self._worker = None
        self._record_task("crop", str(worker.output_dir) if worker else "", details)
        self._finish_task_progress()

    def _on_batch_failed(self, message: str) -> None:
        self.batch_status.setText(f"批量任务失败：{message}")
        self._append_log(f"批量裁剪失败：{message}")
        self._worker = None
        self._finish_task_progress()

    def _start_tile(self) -> None:
        input_dir = QFileDialog.getExistingDirectory(self, "选择待切片图片目录")
        if not input_dir:
            return
        output_dir = QFileDialog.getExistingDirectory(self, "选择切片输出目录")
        if not output_dir:
            return
        self._tile_worker = TileWorker(
            Path(input_dir), Path(output_dir), self.tile_width_box.value(), self.tile_height_box.value(),
            self.overlap_box.value(), str(self.edge_policy.currentData()), self.recursive_check.isChecked(), self.preserve_structure_check.isChecked(), str(self.conflict_policy.currentData()),
        )
        self._tile_worker.progressChanged.connect(self._on_tile_progress)
        self._tile_worker.completed.connect(self._on_tile_completed)
        self._tile_worker.failed.connect(self._on_tile_failed)
        self.tile_status.setText("正在自动切片…")
        self._append_log(f"开始自动切片：{input_dir} → {output_dir}")
        self._tile_worker.start()
        self._begin_task_progress(self._tile_worker)

    def _on_tile_progress(self, current: int, total: int) -> None:
        self.tile_status.setText(f"正在切片：{current} / {total}")
        self._update_task_progress(current, total)

    def _on_tile_completed(self, result: TileResult) -> None:
        worker = self._tile_worker
        text = f"{'已取消：' if result.cancelled else '完成：'}处理 {result.processed_files} 张，写入 {result.written_files} 块。"
        if result.errors:
            text += f"失败 {len(result.errors)} 项。"
        self.tile_status.setText(text)
        self.status.showMessage(text, 8000)
        self._append_log(text)
        if worker:
            write_task_report(worker.output_dir, TaskSummary("tile", result.processed_files, result.written_files, result.errors, result.cancelled), parameters={"tile_width": worker.tile_width, "tile_height": worker.tile_height, "overlap": worker.overlap, "edge_policy": worker.edge_policy})
        self._tile_worker = None
        self._record_task("tile", str(worker.output_dir) if worker else "", text)
        self._finish_task_progress()

    def _on_tile_failed(self, message: str) -> None:
        self.tile_status.setText(f"自动切片失败：{message}")
        self._append_log(f"自动切片失败：{message}")
        self._tile_worker = None
        self._finish_task_progress()

    def _show_roi_tool(self) -> None:
        self.roi_dock.show()
        self.roi_dock.raise_()
        self._on_roi_tool_visibility_changed(True)

    def _on_roi_tool_visibility_changed(self, visible: bool) -> None:
        self.canvas.set_roi_mode(visible)
        if visible:
            self._refresh_preview()
            self.status.showMessage("ROI 工具已打开：单击设起点，拖拽创建 ROI。", 5000)
        else:
            self.canvas.set_preview_roi(None)

    def _show_transform_tool(self) -> None:
        self.transform_dock.show()
        self.transform_dock.raise_()

    def _show_organize_tool(self) -> None:
        self.organize_dock.show()
        self.organize_dock.raise_()

    def _show_filter_tool(self) -> None:
        self.filter_dock.show()
        self.filter_dock.raise_()

    def _show_annotation_tool(self) -> None:
        self.annotation_dock.show()
        self.annotation_dock.raise_()

    def _show_enhance_tool(self) -> None:
        self.enhance_dock.show()
        self.enhance_dock.raise_()

    def _show_pipeline_tool(self) -> None:
        self.pipeline_dock.show()
        self.pipeline_dock.raise_()

    def _load_pipeline(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "加载处理流水线", "", "JSON (*.json)")
        if path:
            try:
                self.pipeline_entry.setPlainText(json.dumps(load_pipeline(path), ensure_ascii=False, indent=2))
            except (OSError, ValueError, json.JSONDecodeError) as error:
                QMessageBox.warning(self, "加载失败", str(error))

    def _save_pipeline(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "保存处理流水线", "pipeline.json", "JSON (*.json)")
        if path:
            try:
                save_pipeline(path, json.loads(self.pipeline_entry.toPlainText()))
            except (OSError, ValueError, json.JSONDecodeError) as error:
                QMessageBox.warning(self, "保存失败", str(error))

    def _run_pipeline_export(self) -> None:
        if self._current_image is None or self._current_path is None:
            self.status.showMessage("请先打开图片。", 4000)
            return
        path, _ = QFileDialog.getSaveFileName(self, "导出流水线结果", str(self._current_path.with_name(f"{self._current_path.stem}_pipeline.png")), "PNG (*.png);;JPEG (*.jpg)")
        if path:
            try:
                write_image(path, run_pipeline(self._current_image, json.loads(self.pipeline_entry.toPlainText())))
                self._append_log(f"已导出流水线结果：{Path(path).name}")
            except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
                QMessageBox.warning(self, "流水线执行失败", str(error))

    def _start_batch_pipeline(self) -> None:
        try:
            steps = json.loads(self.pipeline_entry.toPlainText())
            if not isinstance(steps, list):
                raise ValueError("流水线必须是步骤数组")
        except (ValueError, json.JSONDecodeError) as error:
            QMessageBox.warning(self, "流水线无效", str(error))
            return
        source = QFileDialog.getExistingDirectory(self, "选择待处理图片目录", self._recent_dir("pipeline"))
        if not source:
            return
        output = QFileDialog.getExistingDirectory(self, "选择流水线输出目录", self._recent_dir("pipeline"))
        if not output:
            return
        self._pipeline_worker = PipelineWorker(Path(source), Path(output), steps, suffix=".png", recursive=True, preserve_structure=True, conflict_policy="rename")
        self._pipeline_worker.progressChanged.connect(self._on_pipeline_progress)
        self._pipeline_worker.completed.connect(self._on_pipeline_completed)
        self._pipeline_worker.failed.connect(self._on_pipeline_failed)
        self._pipeline_worker.start()
        self._begin_task_progress(self._pipeline_worker)
        self._append_log(f"开始批量流水线：{source} → {output}")
        self._remember_dir("pipeline", source)

    def _on_pipeline_progress(self, current: int, total: int) -> None:
        self._update_task_progress(current, total)

    def _on_pipeline_completed(self, result: BatchPipelineResult) -> None:
        worker = self._pipeline_worker
        text = f"{'已取消：' if result.cancelled else '完成：'}流水线处理 {result.processed_files} 张，导出 {result.written_files} 张。"
        self._append_log(text)
        if worker:
            write_task_report(worker.output_dir, TaskSummary("pipeline", result.processed_files, result.written_files, result.errors, result.cancelled), parameters={"steps": worker.steps})
            self._record_task("pipeline", str(worker.output_dir), text)
        self._pipeline_worker = None
        self._finish_task_progress()

    def _on_pipeline_failed(self, message: str) -> None:
        self._append_log(f"批量流水线失败：{message}")
        self._pipeline_worker = None
        self._finish_task_progress()

    def _export_enhanced(self) -> None:
        if self._current_image is None or self._current_path is None:
            self.status.showMessage("请先打开图片。", 4000)
            return
        path, _ = QFileDialog.getSaveFileName(self, "导出增强结果", str(self._current_path.with_name(f"{self._current_path.stem}_enhanced.png")), "PNG (*.png);;JPEG (*.jpg)")
        if not path:
            return
        try:
            result = enhance_image(self._current_image, brightness=self.enhance_brightness.value(), contrast=self.enhance_contrast.value(), gamma=self.enhance_gamma.value(), blur_radius=self.enhance_blur.value(), sharpen=self.enhance_sharpen.isChecked(), threshold=None if self.enhance_threshold.value() < 0 else self.enhance_threshold.value())
            write_image(path, result)
            self._append_log(f"已导出增强结果：{Path(path).name}")
        except (ValueError, OSError) as error:
            QMessageBox.warning(self, "增强失败", str(error))

    def _export_polygon_mask(self) -> None:
        if self._current_image is None or self._current_path is None:
            self.status.showMessage("请先打开图片。", 4000)
            return
        try:
            polygons = [[(int(point[0]), int(point[1])) for point in polygon] for polygon in json.loads(self.polygon_entry.toPlainText())]
            height, width = self._current_image.shape[:2]
            mask = polygon_mask(width, height, polygons)
            path, _ = QFileDialog.getSaveFileName(self, "保存 Mask", str(self._current_path.with_name(f"{self._current_path.stem}_mask.png")), "PNG (*.png)")
            if path:
                write_image(path, mask)
                self._append_log(f"已导出多边形 Mask：{Path(path).name}")
        except (ValueError, TypeError, IndexError, json.JSONDecodeError) as error:
            QMessageBox.warning(self, "Mask 坐标无效", str(error))

    def _render_yolo_label(self) -> None:
        if self._current_image is None:
            self.status.showMessage("请先打开图片。", 4000)
            return
        path, _ = QFileDialog.getOpenFileName(self, "选择 YOLO 标签", "", "TXT (*.txt)")
        if not path:
            return
        try:
            result = render_yolo_labels(self._current_image, path)
            self.canvas.set_image(result)
            self._append_log(f"已渲染 YOLO 标签：{Path(path).name}")
        except (OSError, ValueError) as error:
            QMessageBox.warning(self, "标签渲染失败", str(error))

    def _choose_review_dir(self, entry: QLineEdit) -> None:
        directory = QFileDialog.getExistingDirectory(self, "选择输出目录", entry.text() or self._recent_dir("review"))
        if directory:
            entry.setText(directory)
            self._remember_dir("review", directory)

    def _review_current(self, decision: str) -> None:
        if self._current_path is None:
            self.status.showMessage("请先打开待筛选图片。", 4000)
            return
        directory = self.keep_dir_entry.text().strip() if decision == "keep" else self.reject_dir_entry.text().strip()
        if not directory:
            self.status.showMessage("请先选择对应的输出目录。", 4000)
            return
        try:
            move = move_for_review(self._current_path, directory, decision)
            if move is None:
                self.status.showMessage("目标已有同名文件，已跳过。", 4000)
                return
            self._review_history.append(move)
            self._append_log(f"人工筛选：{decision} {move.source.name}")
            self._set_files([path for path in self._files if path != move.source])
        except OSError as error:
            QMessageBox.warning(self, "筛选失败", str(error))

    def _undo_review(self) -> None:
        if not self._review_history:
            self.status.showMessage("没有可撤销的筛选操作。", 4000)
            return
        move = self._review_history.pop()
        try:
            restored = undo_review_move(move)
            self._files.append(restored)
            self._files.sort()
            self._set_files(self._files)
            self._append_log(f"已撤销筛选：{restored.name}")
        except OSError as error:
            QMessageBox.warning(self, "撤销失败", str(error))

    def _classify_keywords(self) -> None:
        keywords = [part.strip() for part in self.keyword_entry.text().split(",") if part.strip()]
        if not keywords:
            self.status.showMessage("请先输入至少一个关键词。", 4000)
            return
        source = QFileDialog.getExistingDirectory(self, "选择待分类目录", self._recent_dir("organize"))
        if not source:
            return
        target = QFileDialog.getExistingDirectory(self, "选择分类输出目录", self._recent_dir("organize"))
        if not target:
            return
        try:
            result = classify_by_keywords(source, target, keywords, mode=str(self.keyword_mode.currentData()))  # type: ignore[arg-type]
            text = f"关键词分类完成：处理 {result.processed_files}，归类 {result.changed_files}，未匹配 {result.unmatched_files}。"
            self._append_log(text)
            self.status.showMessage(text, 7000)
            self._remember_dir("organize", source)
        except (ValueError, OSError) as error:
            QMessageBox.warning(self, "分类失败", str(error))

    def _scan_invalid_images(self) -> None:
        source = QFileDialog.getExistingDirectory(self, "选择待扫描目录", self._recent_dir("organize"))
        if not source:
            return
        invalid = find_invalid_images(source, recursive=True)
        text = "未发现损坏图片。" if not invalid else f"发现 {len(invalid)} 个无法读取的图片：\n" + "\n".join(str(path) for path in invalid[:20])
        QMessageBox.information(self, "损坏图片扫描", text)
        self._append_log(f"损坏图片扫描：{len(invalid)} 个异常文件。")

    def _scan_duplicates(self) -> None:
        source = QFileDialog.getExistingDirectory(self, "选择待扫描目录", self._recent_dir("organize"))
        if not source:
            return
        groups = find_exact_duplicates(source, recursive=True)
        text = "未发现完全重复图片。" if not groups else f"发现 {len(groups)} 组完全重复图片（共 {sum(len(group) for group in groups)} 个文件）。"
        QMessageBox.information(self, "重复图片扫描", text)
        self._append_log(f"重复图片扫描：{len(groups)} 组。")

    def _scan_similar(self) -> None:
        source = QFileDialog.getExistingDirectory(self, "选择待扫描目录", self._recent_dir("organize"))
        if not source:
            return
        groups = find_similar_images(source, recursive=True)
        text = "未发现视觉相似图片。" if not groups else f"发现 {len(groups)} 组视觉相似图片（平均哈希）。"
        QMessageBox.information(self, "相似图片扫描", text)
        self._append_log(f"相似图片扫描：{len(groups)} 组。")

    def _batch_rename(self) -> None:
        prefix = self.rename_prefix.text().strip()
        source = QFileDialog.getExistingDirectory(self, "选择待重命名目录", self._recent_dir("organize"))
        if not source or not prefix:
            return
        if QMessageBox.question(self, "确认重命名", "将直接修改原文件名，是否继续？") != QMessageBox.StandardButton.Yes:
            return
        try:
            result = batch_rename(source, prefix)
            text = f"批量重命名完成：{len(result)} 个文件。"
            self._append_log(text)
            self.status.showMessage(text, 7000)
        except (ValueError, OSError) as error:
            QMessageBox.warning(self, "重命名失败", str(error))

    def _export_transformed_image(self) -> None:
        if self._current_image is None or self._current_path is None:
            self.status.showMessage("请先打开图片。", 4000)
            return
        suffix = str(self.transform_format.currentData())
        default = self._current_path.with_name(f"{self._current_path.stem}_processed{suffix}")
        path, _ = QFileDialog.getSaveFileName(self, "导出处理结果", str(default), "图片 (*.png *.jpg *.jpeg *.webp)")
        if not path:
            return
        try:
            result = transform_image(
                self._current_image,
                width=self.transform_width.value() or None,
                height=self.transform_height.value() or None,
                keep_aspect=self.transform_keep_aspect.isChecked(),
                rotation=str(self.transform_rotation.currentData()),  # type: ignore[arg-type]
                flip=str(self.transform_flip.currentData()),  # type: ignore[arg-type]
                grayscale=self.transform_gray.isChecked(),
            )
            write_image(path, result)
            self.status.showMessage(f"已导出处理结果：{Path(path).name}", 6000)
            self._append_log(f"已导出基础处理结果：{Path(path).name}")
        except (ValueError, OSError) as error:
            QMessageBox.warning(self, "无法导出", str(error))

    def _start_batch_transform(self) -> None:
        input_dir = QFileDialog.getExistingDirectory(self, "选择待处理图片目录")
        if not input_dir:
            return
        output_dir = QFileDialog.getExistingDirectory(self, "选择处理结果输出目录")
        if not output_dir:
            return
        self._transform_worker = TransformWorker(
            Path(input_dir), Path(output_dir), suffix=str(self.transform_format.currentData()),
            width=self.transform_width.value() or None, height=self.transform_height.value() or None,
            keep_aspect=self.transform_keep_aspect.isChecked(), rotation=str(self.transform_rotation.currentData()),
            flip=str(self.transform_flip.currentData()), grayscale=self.transform_gray.isChecked(),
            recursive=self.transform_recursive.isChecked(),
            preserve_structure=self.transform_preserve_structure.isChecked(), conflict_policy=str(self.transform_conflict_policy.currentData()),
        )
        self._transform_worker.progressChanged.connect(self._on_transform_progress)
        self._transform_worker.completed.connect(self._on_transform_completed)
        self._transform_worker.failed.connect(self._on_transform_failed)
        self.transform_status.setText("正在批量处理…")
        self._append_log(f"开始批量基础处理：{input_dir} → {output_dir}")
        self._transform_worker.start()
        self._begin_task_progress(self._transform_worker)

    def _on_transform_progress(self, current: int, total: int) -> None:
        self.transform_status.setText(f"正在处理：{current} / {total}")
        self._update_task_progress(current, total)

    def _on_transform_completed(self, result: BatchTransformResult) -> None:
        worker = self._transform_worker
        text = f"{'已取消：' if result.cancelled else '完成：'}处理 {result.processed_files} 张，导出 {result.written_files} 张。"
        if result.errors:
            text += f"失败 {len(result.errors)} 项。"
        self.transform_status.setText(text)
        self.status.showMessage(text, 8000)
        self._append_log(text)
        if worker:
            write_task_report(worker.output_dir, TaskSummary("transform", result.processed_files, result.written_files, result.errors, result.cancelled), parameters={"suffix": worker.suffix, "width": worker.width, "height": worker.height, "rotation": worker.rotation, "flip": worker.flip, "grayscale": worker.grayscale})
        self._transform_worker = None
        self._record_task("transform", str(worker.output_dir) if worker else "", text)
        self._finish_task_progress()

    def _on_transform_failed(self, message: str) -> None:
        self.transform_status.setText(f"批量处理失败：{message}")
        self._append_log(f"批量基础处理失败：{message}")
        self._transform_worker = None
        self._finish_task_progress()

    def _begin_task_progress(self, worker: QThread) -> None:
        self._active_worker = worker
        self.task_progress.setRange(0, 0)
        self.task_progress.setVisible(True)
        self.cancel_task_button.setEnabled(True)

    def _update_task_progress(self, current: int, total: int) -> None:
        self.task_progress.setRange(0, max(total, 1))
        self.task_progress.setValue(current)

    def _finish_task_progress(self) -> None:
        self._active_worker = None
        self.task_progress.setVisible(False)
        self.cancel_task_button.setEnabled(False)

    def _cancel_active_task(self) -> None:
        if self._active_worker is not None:
            self._active_worker.requestInterruption()
            self.cancel_task_button.setEnabled(False)
            self._append_log("已请求取消当前任务，将在完成当前图片后停止。")

    def _recent_dir(self, key: str) -> str:
        recent = self._state.get("recent_dirs", {})
        value = recent.get(key, "") if isinstance(recent, dict) else ""
        return value if value and Path(value).exists() else ""

    def _remember_dir(self, key: str, value: str) -> None:
        self._state.setdefault("recent_dirs", {})[key] = value
        save_state(self._state)

    def _record_task(self, task_type: str, output_dir: str, summary: str) -> None:
        if not output_dir:
            return
        add_task_history(self._state, task_type=task_type, output_dir=output_dir, summary=summary)
        save_state(self._state)

    def _show_task_history(self) -> None:
        page = self._build_task_history_page()
        if self._workspace_router is not None:
            self._workspace_router.open_page(
                page,
                title="任务历史",
                source_key=self._workspace_key,
                subtitle="查看图像处理、裁剪与流水线任务记录。",
            )
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("任务历史")
        dialog.resize(760, 420)
        layout = QVBoxLayout(dialog)
        layout.addWidget(page)
        dialog.exec()

    def _build_task_history_page(self) -> QWidget:
        page = QWidget()
        page.setObjectName("imageTaskHistoryPage")
        layout = QVBoxLayout(page)
        items = QListWidget()
        items.setObjectName("imageTaskHistoryList")
        for item in self._state.get("task_history", []):
            items.addItem(f"[{item.get('at', '')}] {item.get('task_type', '')}  {item.get('summary', '')}\n{item.get('output_dir', '')}")
        if not items.count():
            items.addItem("尚无已完成任务。")
        layout.addWidget(items)
        return page

    def shutdown(self) -> None:
        """Request cooperative cancellation and join the active batch worker."""
        worker = self._active_worker
        if worker is None or not worker.isRunning():
            return
        worker.requestInterruption()
        worker.wait()
