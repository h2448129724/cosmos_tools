from __future__ import annotations

import math
from pathlib import Path

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QImageReader, QPixmap, QResizeEvent
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QLabel,
    QLineEdit,
    QLayout,
    QListWidget,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from apps.labeling_ui.app.preview_widget import ZoomableLabel, cv2_to_qpixmap

from .cabf_config import CabfConfigDocument, CabfConfigError, RoiField
from .capabilities import CapabilityRuntime
from .paths import COSMOS_ROOT
from .project_context import ProjectState
from .ui.primitives import ActionBar, SectionSurface, StatusBanner


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


def _card(title: str, subtitle: str = "") -> tuple[QFrame, QVBoxLayout]:
    surface = SectionSurface(title, subtitle)
    return surface, surface.body_layout


class CabfConfigActivity(QWidget):
    """CAB-F YAML ROI editor on the full-size calibration reference image.

    Reference images and match templates are produced by the production tools;
    this page only views them and edits ROI coordinates.
    """

    capability_key = "cabf.config_studio"

    def __init__(self, runtime: CapabilityRuntime, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.runtime = runtime
        self.document: CabfConfigDocument | None = None
        self._reference_paths = {"top": "", "bottom": ""}
        self._reference_sizes: dict[str, tuple[int, int] | None] = {"top": None, "bottom": None}
        self._visible_fields: list[RoiField] = []
        self._rect_map: list[tuple[RoiField, int]] = []
        self._canvas_to_source_scale = (1.0, 1.0)
        self._syncing_canvas = False
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
            "选择产品 YAML，再为每个侧别选择全尺寸校准基准图；ROI 坐标基于该图。",
        )
        config_row = ActionBar()
        self.config_edit = QLineEdit()
        self.config_edit.setPlaceholderText("选择 conf/CAB-F 下的产品配置")
        self.config_edit.setAccessibleName("CAB-F 产品配置")
        self.config_edit.setMinimumWidth(280)
        config_row.add_widget(self.config_edit)
        browse_config = QPushButton("选择配置")
        browse_config.clicked.connect(self._pick_config)
        config_row.add_widget(browse_config)
        load_config = QPushButton("加载")
        load_config.setProperty("buttonRole", "primary")
        load_config.clicked.connect(self._load_config)
        config_row.add_widget(load_config)
        context_layout.addWidget(config_row)

        context_controls = ActionBar()
        self.product_badge = QLabel("尚未加载配置")
        self.product_badge.setObjectName("cabfBadge")
        context_controls.add_widget(self.product_badge)
        context_controls.add_widget(QLabel("侧别"))
        self.side_combo = QComboBox()
        self.side_combo.addItem("TOP", "top")
        self.side_combo.addItem("BOTTOM", "bottom")
        self.side_combo.currentIndexChanged.connect(self._switch_side)
        self.side_combo.setAccessibleName("CAB-F 侧别")
        context_controls.add_widget(self.side_combo)
        self.source_edit = QLineEdit()
        self.source_edit.setReadOnly(True)
        self.source_edit.setPlaceholderText("请选择当前侧别的校准基准图")
        self.source_edit.setAccessibleName("当前侧别校准基准图")
        self.source_edit.setMinimumWidth(280)
        context_controls.add_widget(self.source_edit)
        choose_source = QPushButton("选择校准基准图")
        choose_source.clicked.connect(self._pick_reference)
        context_controls.add_widget(choose_source)
        context_layout.addWidget(context_controls)
        layout.addWidget(context_card)

        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.setChildrenCollapsible(True)
        self.splitter.setHandleWidth(7)

        fields_card, fields_layout = _card("ROI 字段", "按 TOP/BOTTOM 自动筛选 YAML 中的 roi / rois。")
        self.field_list = QListWidget()
        self.field_list.currentRowChanged.connect(self._select_field)
        fields_layout.addWidget(self.field_list, 1)
        self.field_summary = QLabel("请先加载配置。")
        self.field_summary.setObjectName("cabfMuted")
        self.field_summary.setWordWrap(True)
        fields_layout.addWidget(self.field_summary)
        self.fields_panel = fields_card
        self.splitter.addWidget(fields_card)

        canvas_card, canvas_layout = _card("统一画布")
        canvas_toolbar = ActionBar()
        self.fields_toggle = QPushButton("ROI 字段")
        self.fields_toggle.setCheckable(True)
        self.fields_toggle.setAccessibleName("显示 ROI 字段面板")
        self.fields_toggle.toggled.connect(lambda visible: self._show_compact_panel("fields", visible))
        self.fields_toggle.hide()
        canvas_toolbar.add_widget(self.fields_toggle)
        self.inspector_toggle = QPushButton("配置检查")
        self.inspector_toggle.setCheckable(True)
        self.inspector_toggle.setAccessibleName("显示配置检查面板")
        self.inspector_toggle.toggled.connect(
            lambda visible: self._show_compact_panel("inspector", visible)
        )
        self.inspector_toggle.hide()
        canvas_toolbar.add_widget(self.inspector_toggle)
        canvas_toolbar.add_widget(QLabel("图层"))
        self.view_combo = QComboBox()
        self.view_combo.addItem("校准基准图 + YAML ROI", "reference")
        self.view_combo.addItem("匹配模板", "template")
        self.view_combo.currentIndexChanged.connect(self._rebuild_canvas)
        self.view_combo.setAccessibleName("画布图层")
        canvas_toolbar.add_widget(self.view_combo)
        fit_button = QPushButton("适应窗口")
        fit_button.clicked.connect(self._fit_canvas)
        canvas_toolbar.add_widget(fit_button)
        self.canvas_badge = QLabel("未选择基准图")
        self.canvas_badge.setObjectName("cabfBadge")
        canvas_toolbar.add_widget(self.canvas_badge)
        canvas_layout.addWidget(canvas_toolbar)
        self.preview = ZoomableLabel()
        self.preview.setObjectName("cabfCanvas")
        self.preview.rectsChanged.connect(self._sync_rects_from_canvas)
        self.preview.rectSelectionChanged.connect(self._select_field_from_canvas)
        canvas_layout.addWidget(self.preview, 1)
        self.canvas_meta = QLabel("选择校准基准图后在画布中查看和编辑 ROI。")
        self.canvas_meta.setObjectName("cabfMuted")
        self.canvas_meta.setWordWrap(True)
        canvas_layout.addWidget(self.canvas_meta)
        self.canvas_panel = canvas_card
        self.splitter.addWidget(canvas_card)

        inspector_card, inspector_surface_layout = _card(
            "配置检查",
            "修改发生在内存中，点击保存后才写回 YAML。",
        )
        self.inspector_scroll = QScrollArea()
        self.inspector_scroll.setObjectName("cabfInspectorScroll")
        self.inspector_scroll.setWidgetResizable(True)
        self.inspector_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.inspector_scroll.setFrameShape(QFrame.Shape.NoFrame)
        inspector_content = QWidget()
        inspector_layout = QVBoxLayout(inspector_content)
        inspector_layout.setContentsMargins(0, 0, 0, 0)
        inspector_layout.setSpacing(8)
        inspector_layout.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
        self.inspector_scroll.setWidget(inspector_content)
        inspector_surface_layout.addWidget(self.inspector_scroll, 1)
        template_form = QFormLayout()
        template_form.setContentsMargins(0, 0, 0, 0)
        self.template_path_label = QLabel("未读取")
        self.template_path_label.setWordWrap(True)
        self.template_path_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.reference_path_label = QLabel("尚未选择")
        self.reference_path_label.setWordWrap(True)
        self.reference_path_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        for value_label in (
            self.template_path_label,
            self.reference_path_label,
        ):
            value_label.setMinimumWidth(0)
            value_label.setSizePolicy(
                QSizePolicy.Policy.Ignored,
                QSizePolicy.Policy.Preferred,
            )
        template_form.addRow("校准基准图", self.reference_path_label)
        template_form.addRow("匹配模板", self.template_path_label)
        inspector_layout.addLayout(template_form)

        self.image_purpose_label = QLabel(
            "校准基准图：全尺寸彩色标准坐标图，YAML ROI 在此图上查看和编辑。\n"
            "匹配模板：半尺寸二值胶体图，只读预览，不承载 ROI。"
        )
        self.image_purpose_label.setObjectName("cabfMuted")
        self.image_purpose_label.setWordWrap(True)
        inspector_layout.addWidget(self.image_purpose_label)

        roi_title = QLabel("当前字段坐标")
        roi_title.setObjectName("cabfSectionTitle")
        inspector_layout.addWidget(roi_title)
        self.roi_list = QListWidget()
        self.roi_list.setMinimumHeight(96)
        self.roi_list.currentRowChanged.connect(self._select_roi)
        inspector_layout.addWidget(self.roi_list, 1)
        roi_actions = ActionBar()
        self.edit_roi_button = QPushButton("启用画布编辑")
        self.edit_roi_button.setCheckable(True)
        self.edit_roi_button.toggled.connect(self._toggle_roi_editing)
        delete_roi = QPushButton("删除末个")
        delete_roi.clicked.connect(self._delete_last_current_roi)
        clear_roi = QPushButton("清空字段")
        clear_roi.setProperty("buttonRole", "danger")
        clear_roi.clicked.connect(self._clear_current_roi)
        roi_actions.add_widget(self.edit_roi_button)
        roi_actions.add_widget(delete_roi)
        roi_actions.add_widget(clear_roi)
        inspector_layout.addWidget(roi_actions)
        self.save_roi_button = QPushButton("保存 ROI 到当前 YAML")
        self.save_roi_button.clicked.connect(self._save_rois)
        self.save_roi_button.setEnabled(False)
        inspector_layout.addWidget(self.save_roi_button)

        self.validation_label = QLabel("等待配置与校准基准图。")
        self.validation_label.setObjectName("cabfValidation")
        self.validation_label.setWordWrap(True)
        inspector_layout.addWidget(self.validation_label)
        self.inspector_panel = inspector_card
        self.splitter.addWidget(inspector_card)

        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 4)
        self.splitter.setStretchFactor(2, 2)
        self.splitter.setSizes([260, 790, 360])
        layout.addWidget(self.splitter, 1)

        self.status_banner = StatusBanner("准备就绪。")
        self.status_label = self.status_banner.label
        layout.addWidget(self.status_banner)
        self._compact_mode = False

    def _show_compact_panel(self, panel: str, visible: bool) -> None:
        if not self._compact_mode:
            return
        if panel == "fields":
            self.fields_panel.setVisible(visible)
            if visible:
                blocked = self.inspector_toggle.blockSignals(True)
                self.inspector_toggle.setChecked(False)
                self.inspector_toggle.blockSignals(blocked)
                self.inspector_panel.hide()
        else:
            self.inspector_panel.setVisible(visible)
            if visible:
                blocked = self.fields_toggle.blockSignals(True)
                self.fields_toggle.setChecked(False)
                self.fields_toggle.blockSignals(blocked)
                self.fields_panel.hide()

    def _set_compact_mode(self, compact: bool) -> None:
        if compact == self._compact_mode:
            return
        self._compact_mode = compact
        self.fields_toggle.setVisible(compact)
        self.inspector_toggle.setVisible(compact)
        if compact:
            self.fields_panel.hide()
            self.inspector_panel.hide()
            self.fields_toggle.setChecked(False)
            self.inspector_toggle.setChecked(False)
        else:
            self.fields_panel.show()
            self.inspector_panel.show()
            self.splitter.setSizes([260, 790, 360])

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self._set_compact_mode(event.size().width() < 1180)

    def _pick_config(self) -> None:
        start = self.config_edit.text().strip() or str(COSMOS_ROOT / "conf" / "CAB-F")
        path, _ = QFileDialog.getOpenFileName(self, "选择 CAB-F 配置", start, "YAML (*.yaml *.yml)")
        if path:
            self.config_edit.setText(path)
            self._load_config()

    def _load_config(self) -> None:
        try:
            self.document = CabfConfigDocument.load(self.config_edit.text().strip())
        except Exception as exc:
            self._show_error("配置加载失败", exc)
            return
        self.product_badge.setText(self.document.project_name)
        self._refresh_side_state()
        self.status_label.setText(f"已加载配置：{self.document.path}")
        if not self._applying_project_context:
            updates = {"cabf_config_path": str(self.document.path)}
            for side in ("top", "bottom"):
                template = self.document.template_path(side)
                if template is not None:
                    updates[f"cabf_template_{side}"] = str(template)
            self.runtime.project_context.update(**updates)

    def _pick_reference(self) -> None:
        current = self._reference_paths[self.side]
        start = str(Path(current).parent) if current else str(COSMOS_ROOT / "conf")
        path, _ = QFileDialog.getOpenFileName(
            self,
            f"选择 {self.side.upper()} 校准基准图",
            start,
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp)",
        )
        if not path:
            return
        try:
            pixmap, (width, height) = _load_preview_pixmap(path)
        except Exception as exc:
            self._show_error("校准基准图加载失败", exc)
            return
        self._reference_paths[self.side] = str(Path(path).resolve())
        self._reference_sizes[self.side] = (width, height)
        self.runtime.project_context.update(**{f"cabf_reference_{self.side}": self._reference_paths[self.side]})
        self.source_edit.setText(self._reference_paths[self.side])
        self.reference_path_label.setText(self._reference_paths[self.side])
        self.view_combo.setCurrentIndex(self.view_combo.findData("reference"))
        self._set_reference_canvas(pixmap)
        self._refresh_validation()
        self.status_label.setText(f"已选择 {self.side.upper()} 校准基准图：{Path(path).name}")

    def _switch_side(self) -> None:
        self.edit_roi_button.setChecked(False)
        self._refresh_side_state()

    def _refresh_side_state(self) -> None:
        self.source_edit.setText(self._reference_paths[self.side])
        self.reference_path_label.setText(self._reference_paths[self.side] or "尚未选择")
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

    def _set_reference_canvas(self, pixmap: QPixmap | None = None) -> None:
        path = self._reference_paths[self.side]
        if pixmap is None and not path:
            self._set_canvas_pixmap(None)
            self.canvas_badge.setText("未选择基准图")
            self.canvas_meta.setText("请先选择当前侧别的全尺寸校准基准图。")
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
        self.canvas_meta.setText(
            f"{self.side.upper()} 校准后全尺寸标准坐标系{preview_note} · YAML ROI 仅属于此图 · {path}"
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
        if view == "template":
            self.edit_roi_button.setChecked(False)
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
            self.canvas_meta.setText("当前侧别的 YAML 没有配置匹配模板，或模板文件不存在。")
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
        has_reference = bool(self._reference_paths[self.side])
        if enabled and (not has_reference or self.document is None):
            self.edit_roi_button.blockSignals(True)
            self.edit_roi_button.setChecked(False)
            self.edit_roi_button.blockSignals(False)
            self.status_label.setText("请先加载配置并选择校准基准图。")
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

    def _refresh_validation(self) -> None:
        if self.document is None:
            self.validation_label.setText("等待 CAB-F 配置。")
            self.save_roi_button.setEnabled(False)
            return
        size = self._reference_sizes[self.side]
        if size is None:
            self.validation_label.setText("已读取配置；选择校准基准图后检查 YAML ROI 边界。")
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
