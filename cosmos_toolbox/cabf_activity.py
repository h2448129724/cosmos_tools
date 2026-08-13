from __future__ import annotations

import math
from pathlib import Path
from uuid import uuid4

import cv2
from PySide6.QtCore import QSize, QThread, Qt, Signal
from PySide6.QtGui import QImageReader, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from apps.labeling_ui.app.preview_widget import ZoomableLabel, cv2_to_qpixmap

from .cabf_config import (
    CabfConfigDocument,
    CabfConfigError,
    CabfTemplateGenerator,
    RoiField,
    TemplateGenerationResult,
    write_reference,
    write_template,
)
from .capabilities import CapabilityRuntime
from .paths import COSMOS_ROOT
from .project_context import ProjectState
from .task_center import TaskStatus


_MAX_PREVIEW_DIMENSION = 4096
_MEBIBYTE = 1024 * 1024


def _load_preview_pixmap(path: str | Path) -> tuple[QPixmap, tuple[int, int]]:
    reader = QImageReader(str(path))
    reader.setAutoTransform(True)
    source_size = reader.size()
    if not source_size.isValid():
        raise CabfConfigError(f"无法读取图片尺寸：{path}（{reader.errorString()}）")
    width, height = source_size.width(), source_size.height()
    scale = min(1.0, _MAX_PREVIEW_DIMENSION / max(width, height))
    if scale < 1.0:
        reader.setScaledSize(QSize(max(1, round(width * scale)), max(1, round(height * scale))))
    previous_limit = QImageReader.allocationLimit()
    required_limit = math.ceil(width * height * 4 / _MEBIBYTE)
    limit_changed = previous_limit > 0 and required_limit > previous_limit
    if limit_changed:
        # Some Qt image handlers check the unscaled 32-bit source size before
        # honouring setScaledSize(). Keep the exception local to this bounded
        # preview read instead of disabling the process-wide safety limit.
        QImageReader.setAllocationLimit(required_limit)
    try:
        image = reader.read()
    finally:
        if limit_changed:
            QImageReader.setAllocationLimit(previous_limit)
    if image.isNull():
        raise CabfConfigError(f"无法读取图片：{path}（{reader.errorString()}）")
    return QPixmap.fromImage(image), (width, height)


def _template_preview_pixmap(image) -> QPixmap:
    height, width = image.shape[:2]
    scale = min(1.0, _MAX_PREVIEW_DIMENSION / max(width, height))
    preview = image
    if scale < 1.0:
        preview = cv2.resize(
            image,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return cv2_to_qpixmap(preview)


class _TemplateWorker(QThread):
    completed = Signal(object)

    def __init__(self, generator: CabfTemplateGenerator, image_path: str, parent=None) -> None:
        super().__init__(parent)
        self.generator = generator
        self.image_path = image_path

    def run(self) -> None:
        try:
            self.completed.emit(self.generator.generate_from_path(self.image_path))
        except Exception as exc:  # pragma: no cover - exercised through the UI slot
            self.completed.emit(exc)


def _card(title: str, subtitle: str = "") -> tuple[QFrame, QVBoxLayout]:
    frame = QFrame()
    frame.setObjectName("cabfCard")
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(16, 14, 16, 14)
    layout.setSpacing(9)
    title_label = QLabel(title)
    title_label.setObjectName("cabfCardTitle")
    layout.addWidget(title_label)
    if subtitle:
        subtitle_label = QLabel(subtitle)
        subtitle_label.setObjectName("cabfCardSubtitle")
        subtitle_label.setWordWrap(True)
        layout.addWidget(subtitle_label)
    return frame, layout


class CabfConfigActivity(QWidget):
    """Native CAB-F configuration, reference-image, template and ROI workspace."""

    capability_key = "cabf.config_studio"

    def __init__(self, runtime: CapabilityRuntime, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.runtime = runtime
        self.document: CabfConfigDocument | None = None
        self._source_paths = {"top": "", "bottom": ""}
        self._source_sizes: dict[str, tuple[int, int] | None] = {"top": None, "bottom": None}
        self._reference_paths = {"top": "", "bottom": ""}
        self._reference_sizes: dict[str, tuple[int, int] | None] = {"top": None, "bottom": None}
        self._generated_templates: dict[str, TemplateGenerationResult | None] = {"top": None, "bottom": None}
        self._visible_fields: list[RoiField] = []
        self._rect_map: list[tuple[RoiField, int]] = []
        self._canvas_to_source_scale = (1.0, 1.0)
        self._syncing_canvas = False
        self._generator = CabfTemplateGenerator()
        self._worker: _TemplateWorker | None = None
        self._task_id = ""
        self._task_side = ""
        self._applying_project_context = False
        self._build_ui()
        self.apply_project_context(self.runtime.project_context.state)

    @property
    def side(self) -> str:
        return str(self.side_combo.currentData() or "top")

    @property
    def current_field(self) -> RoiField | None:
        row = self.field_list.currentRow()
        if 0 <= row < len(self._visible_fields):
            return self._visible_fields[row]
        return None

    def apply_project_context(self, state: ProjectState) -> None:
        """Restore CAB-F-specific state from the shared project context."""
        if self._applying_project_context:
            return
        self._applying_project_context = True
        try:
            for side in ("top", "bottom"):
                source = str(getattr(state, f"cabf_source_{side}", "") or "")
                if source != self._source_paths[side]:
                    self._source_paths[side] = source
                    self._source_sizes[side] = None
                reference = str(getattr(state, f"cabf_reference_{side}", "") or "")
                if reference != self._reference_paths[side]:
                    self._reference_paths[side] = reference
                    self._reference_sizes[side] = None

            config_path = str(state.cabf_config_path or "")
            if config_path:
                self.config_edit.setText(config_path)
                loaded_path = str(self.document.path) if self.document is not None else ""
                if Path(config_path).is_file() and str(Path(config_path).resolve()) != loaded_path:
                    self._load_config()
                    return
            self._refresh_side_state()
        finally:
            self._applying_project_context = False

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        context_card, context_layout = _card(
            "CAB-F 配置上下文",
            "先选择产品 YAML 和初始原图；工具会同时生成全尺寸校准基准图和半尺寸匹配模板。",
        )
        config_row = QHBoxLayout()
        config_row.setSpacing(8)
        self.config_edit = QLineEdit()
        self.config_edit.setPlaceholderText("选择 conf/CAB-F 下的产品配置")
        config_row.addWidget(self.config_edit, 1)
        browse_config = QPushButton("选择配置")
        browse_config.clicked.connect(self._pick_config)
        config_row.addWidget(browse_config)
        load_config = QPushButton("加载")
        load_config.setProperty("buttonRole", "primary")
        load_config.clicked.connect(self._load_config)
        config_row.addWidget(load_config)
        context_layout.addLayout(config_row)

        context_controls = QHBoxLayout()
        context_controls.setSpacing(10)
        self.product_badge = QLabel("尚未加载配置")
        self.product_badge.setObjectName("cabfBadge")
        context_controls.addWidget(self.product_badge)
        context_controls.addWidget(QLabel("侧别"))
        self.side_combo = QComboBox()
        self.side_combo.addItem("TOP", "top")
        self.side_combo.addItem("BOTTOM", "bottom")
        self.side_combo.currentIndexChanged.connect(self._switch_side)
        context_controls.addWidget(self.side_combo)
        self.source_edit = QLineEdit()
        self.source_edit.setReadOnly(True)
        self.source_edit.setPlaceholderText("请选择当前侧别的初始原图")
        context_controls.addWidget(self.source_edit, 1)
        choose_source = QPushButton("选择初始原图")
        choose_source.clicked.connect(self._pick_source)
        context_controls.addWidget(choose_source)
        context_layout.addLayout(context_controls)
        layout.addWidget(context_card)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(7)

        fields_card, fields_layout = _card("ROI 字段", "按 TOP/BOTTOM 自动筛选 YAML 中的 roi / rois。")
        self.field_list = QListWidget()
        self.field_list.currentRowChanged.connect(self._select_field)
        fields_layout.addWidget(self.field_list, 1)
        self.field_summary = QLabel("请先加载配置。")
        self.field_summary.setObjectName("cabfMuted")
        self.field_summary.setWordWrap(True)
        fields_layout.addWidget(self.field_summary)
        splitter.addWidget(fields_card)

        canvas_card, canvas_layout = _card("统一画布")
        canvas_toolbar = QHBoxLayout()
        canvas_toolbar.addWidget(QLabel("图层"))
        self.view_combo = QComboBox()
        self.view_combo.addItem("初始原图（输入）", "source")
        self.view_combo.addItem("校准基准图 + YAML ROI", "reference")
        self.view_combo.addItem("匹配模板", "template")
        self.view_combo.currentIndexChanged.connect(self._rebuild_canvas)
        canvas_toolbar.addWidget(self.view_combo)
        fit_button = QPushButton("适应窗口")
        fit_button.clicked.connect(self._fit_canvas)
        canvas_toolbar.addWidget(fit_button)
        canvas_toolbar.addStretch(1)
        self.canvas_badge = QLabel("未选择原图")
        self.canvas_badge.setObjectName("cabfBadge")
        canvas_toolbar.addWidget(self.canvas_badge)
        canvas_layout.addLayout(canvas_toolbar)
        self.preview = ZoomableLabel()
        self.preview.setObjectName("cabfCanvas")
        self.preview.rectsChanged.connect(self._sync_rects_from_canvas)
        self.preview.rectSelectionChanged.connect(self._select_field_from_canvas)
        canvas_layout.addWidget(self.preview, 1)
        self.canvas_meta = QLabel("选择初始原图后生成两种 CAB-F 图像。")
        self.canvas_meta.setObjectName("cabfMuted")
        self.canvas_meta.setWordWrap(True)
        canvas_layout.addWidget(self.canvas_meta)
        splitter.addWidget(canvas_card)

        inspector_card, inspector_layout = _card("配置检查", "修改发生在内存中，点击保存后才写回 YAML。")
        template_form = QFormLayout()
        template_form.setContentsMargins(0, 0, 0, 0)
        self.model_label = QLabel("由 assets/config/backend_config.yaml 自动读取")
        self.model_label.setWordWrap(True)
        self.model_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.template_path_label = QLabel("未读取")
        self.template_path_label.setWordWrap(True)
        self.template_path_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.reference_path_label = QLabel("尚未生成")
        self.reference_path_label.setWordWrap(True)
        self.reference_path_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        template_form.addRow("胶体模型", self.model_label)
        template_form.addRow("校准基准图", self.reference_path_label)
        template_form.addRow("匹配模板", self.template_path_label)
        inspector_layout.addLayout(template_form)

        self.image_purpose_label = QLabel(
            "校准基准图：全尺寸彩色标准坐标图，YAML ROI 在此图上查看和编辑。\n"
            "匹配模板：半尺寸二值胶体图，生产推理用它计算现场图的平移与旋转，不承载 ROI。"
        )
        self.image_purpose_label.setObjectName("cabfMuted")
        self.image_purpose_label.setWordWrap(True)
        inspector_layout.addWidget(self.image_purpose_label)

        template_actions = QHBoxLayout()
        self.generate_button = QPushButton("生成两种图")
        self.generate_button.setProperty("buttonRole", "primary")
        self.generate_button.clicked.connect(self._generate_template)
        self.save_reference_button = QPushButton("保存校准基准图")
        self.save_reference_button.clicked.connect(self._save_reference)
        self.save_reference_button.setEnabled(False)
        self.save_template_button = QPushButton("保存匹配模板")
        self.save_template_button.clicked.connect(self._save_template)
        self.save_template_button.setEnabled(False)
        template_actions.addWidget(self.generate_button)
        template_actions.addWidget(self.save_reference_button)
        template_actions.addWidget(self.save_template_button)
        inspector_layout.addLayout(template_actions)
        self.update_config_check = QCheckBox("保存后更新 inspection.match_template")
        self.update_config_check.setChecked(True)
        inspector_layout.addWidget(self.update_config_check)

        roi_title = QLabel("当前字段坐标")
        roi_title.setObjectName("cabfSectionTitle")
        inspector_layout.addWidget(roi_title)
        self.roi_list = QListWidget()
        self.roi_list.currentRowChanged.connect(self._select_roi)
        inspector_layout.addWidget(self.roi_list, 1)
        roi_actions = QHBoxLayout()
        self.edit_roi_button = QPushButton("启用画布编辑")
        self.edit_roi_button.setCheckable(True)
        self.edit_roi_button.toggled.connect(self._toggle_roi_editing)
        delete_roi = QPushButton("删除末个")
        delete_roi.clicked.connect(self._delete_last_current_roi)
        clear_roi = QPushButton("清空字段")
        clear_roi.setProperty("buttonRole", "danger")
        clear_roi.clicked.connect(self._clear_current_roi)
        roi_actions.addWidget(self.edit_roi_button)
        roi_actions.addWidget(delete_roi)
        roi_actions.addWidget(clear_roi)
        inspector_layout.addLayout(roi_actions)
        self.save_roi_button = QPushButton("保存 ROI 到当前 YAML")
        self.save_roi_button.clicked.connect(self._save_rois)
        self.save_roi_button.setEnabled(False)
        inspector_layout.addWidget(self.save_roi_button)

        self.validation_label = QLabel("等待配置与原图。")
        self.validation_label.setObjectName("cabfValidation")
        self.validation_label.setWordWrap(True)
        inspector_layout.addWidget(self.validation_label)
        splitter.addWidget(inspector_card)

        splitter.setSizes([260, 790, 360])
        layout.addWidget(splitter, 1)

        self.status_label = QLabel("准备就绪。")
        self.status_label.setObjectName("cabfStatus")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

    def _pick_config(self) -> None:
        start = self.config_edit.text().strip() or str(COSMOS_ROOT / "conf" / "CAB-F")
        path, _ = QFileDialog.getOpenFileName(self, "选择 CAB-F 配置", start, "YAML (*.yaml *.yml)")
        if path:
            self.config_edit.setText(path)
            self._load_config()

    def _load_config(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            self.status_label.setText("模板生成期间不能切换配置。")
            return
        try:
            self.document = CabfConfigDocument.load(self.config_edit.text().strip())
        except Exception as exc:
            self._show_error("配置加载失败", exc)
            return
        self.product_badge.setText(self.document.project_name)
        self._generated_templates = {"top": None, "bottom": None}
        self._refresh_side_state()
        self.status_label.setText(f"已加载配置：{self.document.path}")
        if not self._applying_project_context:
            updates = {"cabf_config_path": str(self.document.path)}
            for side in ("top", "bottom"):
                template = self.document.template_path(side)
                if template is not None:
                    updates[f"cabf_template_{side}"] = str(template)
            self.runtime.project_context.update(**updates)

    def _pick_source(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            self.status_label.setText("图像生成期间不能更换初始原图。")
            return
        current = self._source_paths[self.side]
        start = str(Path(current).parent) if current else str(COSMOS_ROOT)
        path, _ = QFileDialog.getOpenFileName(
            self,
            f"选择 {self.side.upper()} 初始原图",
            start,
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp)",
        )
        if not path:
            return
        try:
            pixmap, (width, height) = _load_preview_pixmap(path)
        except Exception as exc:
            self._show_error("初始原图加载失败", exc)
            return
        self._source_paths[self.side] = str(Path(path).resolve())
        self._source_sizes[self.side] = (width, height)
        self._reference_paths[self.side] = ""
        self._reference_sizes[self.side] = None
        self.runtime.project_context.update(
            **{
                f"cabf_source_{self.side}": self._source_paths[self.side],
                f"cabf_reference_{self.side}": "",
            }
        )
        self.source_edit.setText(self._source_paths[self.side])
        self.reference_path_label.setText("尚未生成")
        self._generated_templates[self.side] = None
        self.view_combo.setCurrentIndex(0)
        self._set_source_canvas(pixmap)
        self._refresh_validation()
        self.status_label.setText(f"已选择 {self.side.upper()} 初始原图：{Path(path).name}；请生成两种图。")

    def _switch_side(self) -> None:
        self.edit_roi_button.setChecked(False)
        self._refresh_side_state()

    def _refresh_side_state(self) -> None:
        self.source_edit.setText(self._source_paths[self.side])
        self.reference_path_label.setText(self._reference_paths[self.side] or "尚未生成")
        generated = self._generated_templates[self.side]
        self.save_reference_button.setEnabled(generated is not None and generated.calibrated_image is not None)
        self.save_template_button.setEnabled(generated is not None)
        if self.document is None:
            self.field_list.clear()
            self.template_path_label.setText("未读取")
            self._rebuild_canvas()
            return
        template_path = self.document.template_path(self.side)
        self.template_path_label.setText(str(template_path) if template_path else "未配置")
        self._visible_fields = self.document.roi_fields(self.side)
        self.field_list.blockSignals(True)
        self.field_list.clear()
        for field in self._visible_fields:
            self.field_list.addItem(field.display_name)
        self.field_list.blockSignals(False)
        self.field_summary.setText(f"{self.side.upper()} 共 {len(self._visible_fields)} 个 ROI 字段。")
        if self._visible_fields:
            self.field_list.setCurrentRow(0)
        else:
            self.roi_list.clear()
        self._rebuild_canvas()
        self._refresh_validation()

    def _select_field(self, _row: int) -> None:
        self._refresh_roi_list()
        self._select_current_field_rect()

    def _refresh_roi_list(self) -> None:
        self.roi_list.blockSignals(True)
        self.roi_list.clear()
        field = self.current_field
        if self.document is not None and field is not None:
            for index, (x1, y1, x2, y2) in enumerate(self.document.rects(field), start=1):
                self.roi_list.addItem(f"ROI {index}  ·  ({x1}, {y1}) → ({x2}, {y2})")
        self.roi_list.blockSignals(False)

    def _set_source_canvas(self, pixmap: QPixmap | None = None) -> None:
        path = self._source_paths[self.side]
        if not path:
            self._set_canvas_pixmap(None)
            self.canvas_badge.setText("未选择初始图")
            self.canvas_meta.setText("请先选择当前侧别的初始原图。")
            return
        if pixmap is None:
            try:
                pixmap, (width, height) = _load_preview_pixmap(path)
            except Exception as exc:
                self._set_canvas_pixmap(None)
                self.canvas_meta.setText(str(exc))
                return
            self._source_sizes[self.side] = (width, height)
        width, height = self._source_sizes[self.side] or (0, 0)
        self._set_canvas_pixmap(pixmap, logical_size=(width, height))
        self.canvas_badge.setText(f"初始图 {width}×{height}")
        self.canvas_meta.setText(f"{self.side.upper()} 未校准输入图；该图不承载 YAML ROI · {path}")

    def _set_reference_canvas(self, pixmap: QPixmap | None = None) -> None:
        result = self._generated_templates[self.side]
        if pixmap is None and result is not None and result.calibrated_image is not None:
            pixmap = _template_preview_pixmap(result.calibrated_image)
            self._reference_sizes[self.side] = result.source_size
        path = self._reference_paths[self.side]
        if pixmap is None and not path:
            self._set_canvas_pixmap(None)
            self.canvas_badge.setText("未生成基准图")
            self.canvas_meta.setText("请先从初始原图生成全尺寸校准基准图。")
            return
        if pixmap is None:
            try:
                pixmap, (width, height) = _load_preview_pixmap(path)
            except Exception as exc:
                self._set_canvas_pixmap(None)
                self.canvas_meta.setText(str(exc))
                return
            self._reference_sizes[self.side] = (width, height)
        logical_size = self._reference_sizes[self.side]
        self._set_canvas_pixmap(pixmap, logical_size=logical_size)
        self._rebuild_roi_rects()
        width, height = self._reference_sizes[self.side] or (0, 0)
        self.canvas_badge.setText(f"原图 {width}×{height}")
        preview_note = ""
        if pixmap.width() != width or pixmap.height() != height:
            preview_note = f" · 画布预览 {pixmap.width()}×{pixmap.height()}，坐标自动映射到原图"
        source_text = path or "尚未保存"
        self.canvas_meta.setText(
            f"{self.side.upper()} 校准后全尺寸标准坐标系{preview_note} · YAML ROI 仅属于此图 · {source_text}"
        )

    def _set_canvas_pixmap(self, pixmap: QPixmap | None, *, logical_size: tuple[int, int] | None = None) -> None:
        self._syncing_canvas = True
        self.preview.set_pixmap(pixmap or cv2_to_qpixmap(None))
        self._syncing_canvas = False
        if pixmap is None or pixmap.isNull() or logical_size is None:
            self._canvas_to_source_scale = (1.0, 1.0)
        else:
            self._canvas_to_source_scale = (
                logical_size[0] / max(pixmap.width(), 1),
                logical_size[1] / max(pixmap.height(), 1),
            )

    def _rebuild_canvas(self) -> None:
        view = str(self.view_combo.currentData())
        if view == "source":
            self.edit_roi_button.setChecked(False)
            self._set_source_canvas()
            return
        if view == "template":
            self.edit_roi_button.setChecked(False)
            result = self._generated_templates[self.side]
            if result is not None:
                self._set_canvas_pixmap(_template_preview_pixmap(result.image))
                width, height = result.template_size
                self.canvas_badge.setText(f"新模板 {width}×{height}")
                self.canvas_meta.setText(
                    "半尺寸二值匹配模板：生产推理用它计算现场图的平移与旋转；不叠加 YAML ROI。"
                )
                return
            template_path = self.document.template_path(self.side) if self.document else None
            if template_path and template_path.exists():
                try:
                    pixmap, (width, height) = _load_preview_pixmap(template_path)
                except Exception as exc:
                    self._set_canvas_pixmap(None)
                    self.canvas_meta.setText(str(exc))
                    return
                self._set_canvas_pixmap(pixmap)
                self.canvas_badge.setText(f"配置模板 {width}×{height}")
                self.canvas_meta.setText(f"半尺寸二值匹配模板；不承载 YAML ROI · {template_path}")
                return
            self._set_canvas_pixmap(None)
            self.canvas_badge.setText("无匹配模板")
            self.canvas_meta.setText("当前侧别尚未生成或配置匹配模板。")
            return
        self._set_reference_canvas()
        self._refresh_validation()

    def _rebuild_roi_rects(self) -> None:
        self._rect_map = []
        rects: list[tuple[int, int, int, int]] = []
        if self.document is not None:
            scale_x, scale_y = self._canvas_to_source_scale
            for field in self._visible_fields:
                for index, rect in enumerate(self.document.rects(field)):
                    rects.append(self._source_to_canvas_rect(rect, scale_x, scale_y))
                    self._rect_map.append((field, index))
        self._syncing_canvas = True
        self.preview._rects_image = rects
        self.preview.set_selected_rect_index(-1)
        self.preview.update()
        self._syncing_canvas = False
        self._select_current_field_rect()

    def _select_current_field_rect(self) -> None:
        field = self.current_field
        index = next((idx for idx, (mapped, _roi) in enumerate(self._rect_map) if mapped == field), -1)
        self.preview.set_selected_rect_index(index)

    def _select_roi(self, roi_index: int) -> None:
        field = self.current_field
        preview_index = next(
            (idx for idx, (mapped, mapped_roi) in enumerate(self._rect_map) if mapped == field and mapped_roi == roi_index),
            -1,
        )
        self.preview.set_selected_rect_index(preview_index)

    def _select_field_from_canvas(self, preview_index: int) -> None:
        if self._syncing_canvas or not (0 <= preview_index < len(self._rect_map)):
            return
        field, roi_index = self._rect_map[preview_index]
        try:
            field_row = self._visible_fields.index(field)
        except ValueError:
            return
        self.field_list.setCurrentRow(field_row)
        self.roi_list.setCurrentRow(roi_index)

    def _sync_rects_from_canvas(self) -> None:
        if self._syncing_canvas or self.document is None or str(self.view_combo.currentData()) != "reference":
            return
        canvas_rects = self.preview.get_roi_rects()
        if len(canvas_rects) == len(self._rect_map) + 1:
            field = self.current_field
            if field is None or (not field.is_multi and self.document.rects(field)):
                self.status_label.setText("当前字段只允许一个 ROI，已撤销新建区域。")
                self._rebuild_roi_rects()
                return
            self._rect_map.append((field, len(self.document.rects(field))))
        elif len(canvas_rects) == len(self._rect_map) - 1:
            self._rect_map = self._rect_map[:-1]
        elif len(canvas_rects) != len(self._rect_map):
            self._rebuild_roi_rects()
            return
        scale_x, scale_y = self._canvas_to_source_scale
        rects = []
        for canvas_rect, (field, roi_index) in zip(canvas_rects, self._rect_map):
            existing = self.document.rects(field)
            if roi_index < len(existing) and canvas_rect == self._source_to_canvas_rect(
                existing[roi_index], scale_x, scale_y
            ):
                rects.append(existing[roi_index])
            else:
                x1, y1, x2, y2 = canvas_rect
                rects.append(
                    (
                        round(x1 * scale_x),
                        round(y1 * scale_y),
                        round(x2 * scale_x),
                        round(y2 * scale_y),
                    )
                )
        grouped = {field.path_key: [] for field in self._visible_fields}
        for rect, (field, _index) in zip(rects, self._rect_map):
            grouped[field.path_key].append(tuple(int(value) for value in rect))
        for field in self._visible_fields:
            self.document.set_rects(field, grouped[field.path_key])
        self._rebuild_roi_rects()
        self._refresh_roi_list()
        self._refresh_validation()
        self.status_label.setText("ROI 已修改；尚未写回 YAML。")

    @staticmethod
    def _source_to_canvas_rect(
        rect: tuple[int, int, int, int], scale_x: float, scale_y: float
    ) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = rect
        return (
            round(x1 / scale_x),
            round(y1 / scale_y),
            round(x2 / scale_x),
            round(y2 / scale_y),
        )

    def _toggle_roi_editing(self, enabled: bool) -> None:
        result = self._generated_templates[self.side]
        has_reference = bool(self._reference_paths[self.side]) or bool(
            result is not None and result.calibrated_image is not None
        )
        if enabled and (not has_reference or self.document is None):
            self.edit_roi_button.blockSignals(True)
            self.edit_roi_button.setChecked(False)
            self.edit_roi_button.blockSignals(False)
            self.status_label.setText("请先加载配置并生成校准基准图。")
            return
        if enabled and str(self.view_combo.currentData()) != "reference":
            self.view_combo.setCurrentIndex(self.view_combo.findData("reference"))
        self.preview.rect_select_mode = enabled
        self.edit_roi_button.setText("结束画布编辑" if enabled else "启用画布编辑")

    def _delete_last_current_roi(self) -> None:
        if self.document is None or self.current_field is None:
            return
        rects = self.document.rects(self.current_field)
        if rects:
            self.document.set_rects(self.current_field, rects[:-1])
            self._rebuild_roi_rects()
            self._refresh_roi_list()
            self._refresh_validation()

    def _clear_current_roi(self) -> None:
        if self.document is None or self.current_field is None:
            return
        self.document.set_rects(self.current_field, [])
        self._rebuild_roi_rects()
        self._refresh_roi_list()
        self._refresh_validation()

    def _save_rois(self) -> None:
        if self.document is None:
            return
        size = self._reference_sizes[self.side]
        if size and self.document.validate_rois(self.side, size):
            self.status_label.setText("存在越界或无效 ROI，请修正后再保存。")
            return
        try:
            self.document.save_rois()
        except Exception as exc:
            self._show_error("ROI 保存失败", exc)
            return
        self.status_label.setText(f"ROI 已保存：{self.document.path}")

    def _generate_template(self) -> None:
        image_path = self._source_paths[self.side]
        if not image_path:
            self.status_label.setText("请先选择当前侧别的初始原图。")
            return
        if self._worker is not None and self._worker.isRunning():
            return
        self._task_id = f"cabf-template-{uuid4().hex[:10]}"
        self._task_side = self.side
        center = self.runtime.task_center
        center.start(self._task_id, f"生成 {self._task_side.upper()} CAB-F 校准基准图与匹配模板", self.capability_key)
        center.log(self._task_id, f"初始原图：{image_path}")
        center.log(self._task_id, "模型来源：assets/config/backend_config.yaml → cab_f.glue_segment.path")
        self.generate_button.setEnabled(False)
        self.status_label.setText("正在生成全尺寸校准基准图和半尺寸匹配模板；日志已进入任务中心。")
        self._worker = _TemplateWorker(self._generator, image_path, self)
        self._worker.completed.connect(self._template_completed)
        self._worker.finished.connect(self._template_thread_finished)
        self._worker.start()

    def _template_completed(self, result: object) -> None:
        center = self.runtime.task_center
        if isinstance(result, Exception):
            center.log(self._task_id, str(result), "stderr")
            center.finish(self._task_id, TaskStatus.FAILED)
            self.status_label.setText(f"模板生成失败：{result}")
            return
        assert isinstance(result, TemplateGenerationResult)
        if result.calibrated_image is None:
            error = CabfConfigError("生成结果缺少全尺寸校准基准图")
            center.log(self._task_id, str(error), "stderr")
            center.finish(self._task_id, TaskStatus.FAILED)
            self.status_label.setText(str(error))
            return
        target_side = self._task_side
        self._generated_templates[target_side] = result
        self._reference_sizes[target_side] = result.source_size
        self.model_label.setText(result.model_path)
        center.log(self._task_id, f"模板尺寸：{result.template_size[0]}×{result.template_size[1]}")
        center.log(self._task_id, f"校准基准图尺寸：{result.source_size[0]}×{result.source_size[1]}")
        center.log(self._task_id, f"全尺寸校准偏移（行, 列）：{result.calibration_offset}")
        center.finish(self._task_id, TaskStatus.SUCCESS)
        if self.side == target_side:
            self.save_template_button.setEnabled(True)
            self.save_reference_button.setEnabled(True)
            self.view_combo.setCurrentIndex(self.view_combo.findData("reference"))
            self._rebuild_canvas()
            self._refresh_validation()
            self.status_label.setText("两种图已生成；请分别保存校准基准图和匹配模板。")
        else:
            self.status_label.setText(f"{target_side.upper()} 两种图已生成；切回该侧别后可以预览和保存。")

    def _template_thread_finished(self) -> None:
        self.generate_button.setEnabled(True)
        worker = self._worker
        self._worker = None
        if worker is not None:
            worker.deleteLater()

    def _save_template(self) -> None:
        result = self._generated_templates[self.side]
        if result is None:
            return
        configured = self.document.template_path(self.side) if self.document else None
        start = str(configured or (COSMOS_ROOT / "conf" / f"CAB_F_template_{self.side}.png"))
        path, _ = QFileDialog.getSaveFileName(self, "保存 CAB-F 匹配模板", start, "PNG (*.png)")
        if not path:
            return
        try:
            saved = write_template(path, result.image)
            if self.update_config_check.isChecked():
                if self.document is None:
                    raise CabfConfigError("尚未加载 CAB-F 配置，无法更新 match_template")
                self.document.update_template_path(self.side, saved)
                self.template_path_label.setText(str(saved))
        except Exception as exc:
            self._show_error("模板保存失败", exc)
            return
        self.runtime.project_context.update(**{f"cabf_template_{self.side}": str(saved)})
        self.status_label.setText(f"匹配模板已保存：{saved}")

    def _save_reference(self) -> None:
        result = self._generated_templates[self.side]
        if result is None or result.calibrated_image is None:
            return
        configured = self._reference_paths[self.side]
        start = str(configured or (COSMOS_ROOT / "conf" / f"CAB_F_reference_{self.side}.png"))
        path, _ = QFileDialog.getSaveFileName(self, "保存 CAB-F 全尺寸校准基准图", start, "PNG (*.png)")
        if not path:
            return
        try:
            saved = write_reference(path, result.calibrated_image)
        except Exception as exc:
            self._show_error("校准基准图保存失败", exc)
            return
        self._reference_paths[self.side] = str(saved)
        self._reference_sizes[self.side] = result.source_size
        self.reference_path_label.setText(str(saved))
        self.runtime.project_context.update(**{f"cabf_reference_{self.side}": str(saved)})
        self.status_label.setText(f"全尺寸校准基准图已保存：{saved}")

    def _refresh_validation(self) -> None:
        if self.document is None:
            self.validation_label.setText("等待 CAB-F 配置。")
            self.save_roi_button.setEnabled(False)
            return
        size = self._reference_sizes[self.side]
        if size is None:
            self.validation_label.setText("已读取配置；生成或加载校准基准图后检查 YAML ROI 边界。")
            self.save_roi_button.setEnabled(False)
            return
        issues = self.document.validate_rois(self.side, size)
        template_warning = ""
        template_path = self.document.template_path(self.side)
        if template_path and template_path.exists():
            reader = QImageReader(str(template_path))
            template_size = reader.size()
            if template_size.isValid():
                expected_width = round(size[0] * 0.5)
                expected_height = round(size[1] * 0.5)
                if abs(template_size.width() - expected_width) > 1 or abs(template_size.height() - expected_height) > 1:
                    template_warning = (
                        f"配置模板为 {template_size.width()}×{template_size.height()}，"
                        f"原图半尺寸应约为 {expected_width}×{expected_height}。"
                    )
        roi_count = sum(len(self.document.rects(field)) for field in self._visible_fields)
        if issues or template_warning:
            preview = "；".join(f"{item.field}: {item.message}" for item in issues[:3])
            suffix = f" 等 {len(issues)} 项" if len(issues) > 3 else ""
            details = "；".join(item for item in (preview + suffix, template_warning) if item)
            self.validation_label.setText(f"发现 {len(issues) + bool(template_warning)} 个问题：{details}")
            self.validation_label.setProperty("level", "warning")
            self.save_roi_button.setEnabled(not issues)
        else:
            width, height = size
            self.validation_label.setText(f"检查通过：{roi_count} 个 ROI 均位于原图 {width}×{height} 内。")
            self.validation_label.setProperty("level", "success")
            self.save_roi_button.setEnabled(True)
        self.validation_label.style().unpolish(self.validation_label)
        self.validation_label.style().polish(self.validation_label)

    def _fit_canvas(self) -> None:
        self.preview.fit_to_view()
        self.preview.update()

    def _show_error(self, title: str, exc: Exception) -> None:
        self.status_label.setText(f"{title}：{exc}")
        QMessageBox.critical(self, title, str(exc))

    def shutdown(self) -> None:
        """Wait for an in-flight ONNX task before the owning Qt window is destroyed."""
        worker = self._worker
        if worker is None or not worker.isRunning():
            return
        worker.requestInterruption()
        worker.wait()
        task = self.runtime.task_center.get(self._task_id)
        if task is not None and task.status == TaskStatus.RUNNING:
            self.runtime.task_center.finish(self._task_id, TaskStatus.STOPPED)
