"""YAML ROI 配置可视化编辑工具。"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
)
from cabf.roi import (
    RoiField,
    collect_roi_fields,
    get_value as _get_value,
    normalize_rects,
    patch_roi_text,
    set_rects,
)
try:
    from ruamel.yaml import YAML
except ImportError:  # pragma: no cover - depends on local env
    YAML = None
    import yaml as pyyaml
else:
    pyyaml = None

from apps.data_tools.processing.image_io import read_image
from cosmos_toolbox.ui.primitives import ActionBar, set_ui_role
from cosmos_toolbox.ui.theme import status_badge_stylesheet
from ..preview_widget import ZoomableLabel, cv2_to_qpixmap
from .base import BaseToolPage, make_card, make_page_header, set_primary


if YAML is not None:
    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.width = 4096
else:
    yaml = None


def _normalize_rects(value: Any) -> list[tuple[int, int, int, int]]:
    return normalize_rects(value, allow_boolean=True)


def _write_rects_back(field: RoiField, data: Any, rects: list[tuple[int, int, int, int]]) -> None:
    set_rects(data, field, rects, reject_multiple=False)


def _collect_roi_fields(
    node: Any,
    path_parts: tuple[Any, ...] = (),
    cab_f_config: bool | None = None,
) -> list[RoiField]:
    if cab_f_config is None:
        inspection = node.get("inspection", {}) if isinstance(node, dict) else {}
        cab_f_config = str(inspection.get("project", "")).upper() == "CAB-F"
    if cab_f_config:
        return collect_roi_fields(node, path_parts)
    # This page predates CAB-F and accepts both spellings for other projects.
    return collect_roi_fields(
        node,
        path_parts,
        roi_keys=("roi", "rois"),
        empty_roi_is_multi=False,
        allow_boolean=True,
        side_case_sensitive=True,
    )


def _find_loader_dir(config_data: Any, side: str) -> str:
    loaders = config_data.get("image_loader", []) if isinstance(config_data, dict) else []
    for loader in loaders:
        if not isinstance(loader, dict):
            continue
        if str(loader.get("loader_name", "")).lower() == side:
            return str(loader.get("load_dir", "") or "")
    return ""


def _find_sample_images(config_data: Any, side: str) -> list[str]:
    loaders = config_data.get("image_loader", []) if isinstance(config_data, dict) else []
    for loader in loaders:
        if not isinstance(loader, dict):
            continue
        if str(loader.get("loader_name", "")).lower() != side:
            continue
        load_dir = Path(str(loader.get("load_dir", "") or ""))
        pattern = str(loader.get("glob_pattern", "*") or "*")
        if load_dir.exists():
            return [str(path) for path in sorted(load_dir.glob(pattern)) if path.is_file()]
    return []


def _load_yaml_file(path: Path) -> Any:
    if yaml is not None:
        with path.open("r", encoding="utf-8") as fh:
            return yaml.load(fh)
    with path.open("r", encoding="utf-8") as fh:
        return pyyaml.safe_load(fh)


def _patch_yaml_roi_text(original_text: str, data: Any, fields: list[RoiField]) -> str:
    return patch_roi_text(original_text, data, fields, allow_boolean=True)


def _dump_yaml_file(path: Path, data: Any, roi_fields: list[RoiField], original_text: str | None) -> str:
    if original_text:
        patched = _patch_yaml_roi_text(original_text, data, roi_fields)
        with path.open("w", encoding="utf-8") as fh:
            fh.write(patched)
        return patched

    if yaml is not None:
        with path.open("w", encoding="utf-8") as fh:
            yaml.dump(data, fh)
        return path.read_text(encoding="utf-8")

    with path.open("w", encoding="utf-8") as fh:
        pyyaml.safe_dump(data, fh, allow_unicode=True, sort_keys=False)
    return path.read_text(encoding="utf-8")


class RoiConfigEditorPage(BaseToolPage):
    tool_key = "roi_config_editor"
    tool_title = "ROI配置编辑"
    tool_nav_title = "ROI配置"
    tool_icon = "▣"
    tool_summary = "读取 YAML 配置中的 ROI 字段，在样本图上可视化查看、创建和调整区域。"
    tool_tags = ("YAML", "ROI编辑", "可视化配置")

    def __init__(self, main_window, parent=None):
        super().__init__(main_window, parent)
        self._config_path: Path | None = None
        self._config_data = None
        self._original_yaml_text = ""
        self._roi_fields: list[RoiField] = []
        self._visible_field_indices: list[int] = []
        self._current_field: RoiField | None = None
        self._current_field_index = -1
        self._sample_image_paths: dict[str, str] = {"top": "", "bottom": ""}
        self._manual_sample_override: dict[str, bool] = {"top": False, "bottom": False}
        self._current_image_side = "top"
        self._left_panel_visible = True
        self._left_panel_width = 420
        self._suspend_rect_sync = False
        self._display_rect_map: list[tuple[int, int]] = []
        self._build_ui()

    def _build_ui(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 14, 14, 14)
        lay.setSpacing(8)

        lay.addWidget(make_page_header("ROI配置编辑", "读取 YAML 并直接调整图上的 ROI。"))

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(6)
        self._splitter = splitter

        left = make_card()
        self._left_panel = left
        left_lay = QVBoxLayout(left)
        left_lay.setContentsMargins(16, 16, 16, 16)
        left_lay.setSpacing(10)

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QFrame.NoFrame)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        left_scroll.setWidget(left)
        self._left_panel_container = left_scroll

        config_card = make_card()
        config_card_lay = QVBoxLayout(config_card)
        config_card_lay.setContentsMargins(14, 14, 14, 14)
        config_card_lay.setSpacing(8)
        config_title = QLabel("配置加载区")
        set_ui_role(config_title, "sectionTitle")
        config_card_lay.addWidget(config_title)

        config_desc = QLabel("选择配置与当前编辑图片。图片只会在你手动选择时变更。")
        config_desc.setWordWrap(True)
        set_ui_role(config_desc, "muted")
        config_card_lay.addWidget(config_desc)

        cfg_box = QFrame()
        cfg_form = QFormLayout(cfg_box)
        cfg_form.setContentsMargins(0, 0, 0, 0)
        self._config_entry = QLineEdit("")
        self._config_entry.setAccessibleName("配置文件路径")
        self._config_entry.setPlaceholderText("选择 YAML 配置文件")
        self._side_combo = QComboBox()
        self._side_combo.setAccessibleName("样本侧别")
        self._side_combo.setToolTip("切换 TOP/BOTTOM 样本及对应 ROI")
        self._side_combo.addItem("TOP", "top")
        self._side_combo.addItem("BOTTOM", "bottom")
        cfg_form.addRow("配置文件", self._config_entry)
        cfg_form.addRow("样本侧别", self._side_combo)
        config_card_lay.addWidget(cfg_box)

        image_state_card = QFrame()
        image_state_card.setObjectName("hintPanel")
        set_ui_role(image_state_card, "hint")
        image_state_card.setAccessibleName("当前图片状态")
        image_state_lay = QVBoxLayout(image_state_card)
        image_state_lay.setContentsMargins(12, 10, 12, 10)
        image_state_lay.setSpacing(4)
        image_state_label = QLabel("当前图片")
        set_ui_role(image_state_label, "sectionTitle")
        image_state_lay.addWidget(image_state_label)
        badge_row = QHBoxLayout()
        badge_row.setSpacing(8)
        self._side_badge = QLabel("TOP")
        self._side_badge.setAccessibleName("当前样本侧别")
        self._side_badge.setStyleSheet(status_badge_stylesheet("info"))
        self._select_badge = QLabel("未选择")
        self._select_badge.setAccessibleName("样本图片选择状态")
        self._select_badge.setStyleSheet(status_badge_stylesheet("neutral"))
        badge_row.addWidget(self._side_badge)
        badge_row.addWidget(self._select_badge)
        badge_row.addStretch(1)
        image_state_lay.addLayout(badge_row)
        self._sample_image_bar = QLabel("未选择图片")
        self._sample_image_bar.setWordWrap(True)
        set_ui_role(self._sample_image_bar, "sectionTitle")
        image_state_lay.addWidget(self._sample_image_bar)
        self._sample_image_meta = QLabel("请选择一张图片作为当前编辑底图。")
        self._sample_image_meta.setWordWrap(True)
        set_ui_role(self._sample_image_meta, "muted")
        image_state_lay.addWidget(self._sample_image_meta)
        config_card_lay.addWidget(image_state_card)

        cfg_btns = ActionBar()
        cfg_btns.setAccessibleName("配置操作工具栏")
        btn_cfg = QPushButton("浏览配置")
        btn_cfg.setAccessibleName("浏览 YAML 配置")
        btn_cfg.setToolTip("选择要编辑的 YAML 配置文件")
        btn_cfg.clicked.connect(self._pick_config)
        btn_sample_image = QPushButton("选择图片")
        btn_sample_image.setAccessibleName("选择样本图片")
        btn_sample_image.setToolTip("选择当前侧别的样本底图")
        btn_sample_image.clicked.connect(self._pick_sample_image)
        btn_clear_image = QPushButton("清除当前图片")
        btn_clear_image.setAccessibleName("清除当前样本图片")
        btn_clear_image.clicked.connect(self._clear_sample_image)
        btn_load = QPushButton("加载配置")
        btn_load.setAccessibleName("加载 YAML 配置")
        btn_load.setToolTip("读取配置并发现可编辑 ROI 字段")
        set_primary(btn_load)
        btn_load.clicked.connect(self._load_config)
        for widget in (btn_cfg, btn_sample_image, btn_clear_image, btn_load):
            cfg_btns.add_widget(widget)
        config_card_lay.addWidget(cfg_btns)
        left_lay.addWidget(config_card)

        editor_card = make_card()
        editor_card_lay = QVBoxLayout(editor_card)
        editor_card_lay.setContentsMargins(14, 14, 14, 14)
        editor_card_lay.setSpacing(8)
        editor_title = QLabel("ROI 编辑区")
        set_ui_role(editor_title, "sectionTitle")
        editor_card_lay.addWidget(editor_title)

        editor_desc = QLabel("选择字段后，在右侧直接编辑 ROI。")
        editor_desc.setWordWrap(True)
        set_ui_role(editor_desc, "muted")
        editor_card_lay.addWidget(editor_desc)

        self._field_list = QListWidget()
        self._field_list.setAccessibleName("ROI 字段列表")
        self._field_list.setToolTip("选择要在预览画布中编辑的 ROI 字段")
        self._field_list.currentRowChanged.connect(self._on_field_selected)
        field_section = QFrame()
        set_ui_role(field_section, "sectionSurface")
        field_section.setProperty("nested", True)
        field_section_lay = QVBoxLayout(field_section)
        field_section_lay.setContentsMargins(0, 0, 0, 0)
        field_section_lay.setSpacing(8)
        field_title = QLabel("ROI 字段")
        set_ui_role(field_title, "sectionTitle")
        field_section_lay.addWidget(field_title)
        field_section_lay.addWidget(self._field_list, 1)
        editor_card_lay.addWidget(field_section, 1)

        self._field_meta = QLabel("等待加载配置。")
        self._field_meta.setWordWrap(True)
        set_ui_role(self._field_meta, "muted")
        editor_card_lay.addWidget(self._field_meta)

        self._edit_help = QLabel("框内拖动可移动，拖四角可缩放。")
        self._edit_help.setWordWrap(True)
        set_ui_role(self._edit_help, "muted")
        editor_card_lay.addWidget(self._edit_help)

        roi_btns = ActionBar()
        roi_btns.setAccessibleName("ROI 编辑工具栏")
        btn_start = QPushButton("开始框选")
        btn_start.setAccessibleName("开始 ROI 框选")
        btn_start.setToolTip("在右侧图像上拖动创建 ROI")
        set_primary(btn_start)
        btn_start.clicked.connect(self._start_select)
        btn_undo = QPushButton("撤销上一个")
        btn_undo.setAccessibleName("撤销上一个 ROI")
        btn_undo.clicked.connect(self._undo_rect)
        btn_clear = QPushButton("清空当前字段")
        btn_clear.setAccessibleName("清空当前字段 ROI")
        btn_clear.clicked.connect(self._clear_rects)
        for widget in (btn_start, btn_undo, btn_clear):
            roi_btns.add_widget(widget)
        editor_card_lay.addWidget(roi_btns)

        self._roi_list = QListWidget()
        self._roi_list.setAccessibleName("当前字段 ROI 列表")
        self._roi_list.currentRowChanged.connect(self._preview_select_row)
        roi_section = QFrame()
        set_ui_role(roi_section, "sectionSurface")
        roi_section.setProperty("nested", True)
        roi_section_lay = QVBoxLayout(roi_section)
        roi_section_lay.setContentsMargins(0, 0, 0, 0)
        roi_section_lay.setSpacing(8)
        roi_title = QLabel("当前字段 ROI")
        set_ui_role(roi_title, "sectionTitle")
        roi_section_lay.addWidget(roi_title)
        roi_section_lay.addWidget(self._roi_list)
        editor_card_lay.addWidget(roi_section)

        io_btns = ActionBar()
        io_btns.setAccessibleName("配置保存工具栏")
        btn_save_field = QPushButton("保存当前字段到配置")
        btn_save_field.setAccessibleName("保存当前字段到配置")
        btn_save_field.clicked.connect(self._save_current_field)
        btn_save_all = QPushButton("写回 YAML")
        btn_save_all.setAccessibleName("写回 YAML 配置")
        btn_save_all.setToolTip("将当前 ROI 修改写回 YAML")
        set_primary(btn_save_all)
        btn_save_all.clicked.connect(self._save_yaml)
        for widget in (btn_save_field, btn_save_all):
            io_btns.add_widget(widget)
        editor_card_lay.addWidget(io_btns)
        left_lay.addWidget(editor_card, 1)

        splitter.addWidget(left_scroll)

        right = make_card()
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(16, 16, 16, 16)
        right_lay.setSpacing(8)

        topbar = ActionBar()
        topbar.setAccessibleName("ROI 预览操作工具栏")
        self._btn_toggle_sidebar = QPushButton("收起侧栏")
        self._btn_toggle_sidebar.setAccessibleName("收起或展开 ROI 侧栏")
        self._btn_toggle_sidebar.setToolTip("切换 ROI 配置侧栏")
        self._btn_toggle_sidebar.clicked.connect(self._toggle_left_panel)
        topbar.add_widget(self._btn_toggle_sidebar)
        self._btn_save_quick = QPushButton("保存配置")
        self._btn_save_quick.setAccessibleName("保存 ROI 配置")
        self._btn_save_quick.setToolTip("将当前 ROI 修改写回 YAML")
        set_primary(self._btn_save_quick)
        self._btn_save_quick.clicked.connect(self._save_yaml)
        topbar.add_widget(self._btn_save_quick)
        self._image_meta = QLabel("当前样本：未加载")
        set_ui_role(self._image_meta, "muted")
        self._image_meta.setAccessibleName("当前样本信息")
        self._image_meta.setWordWrap(True)
        topbar.add_widget(self._image_meta)
        right_lay.addWidget(topbar)

        self._preview = ZoomableLabel()
        self._preview.setAccessibleName("ROI 编辑预览画布")
        self._preview.setToolTip("拖动 ROI 可移动，拖四角可缩放")
        self._preview.rectSelected.connect(self._on_rects_changed)
        self._preview.rectsChanged.connect(self._on_rects_changed)
        self._preview.rectSelectionChanged.connect(self._sync_selected_roi_row)
        right_lay.addWidget(self._preview, 1)

        self._status = QLabel("请先加载 YAML 配置。")
        set_ui_role(self._status, "muted")
        self._status.setAccessibleName("ROI 配置状态")
        right_lay.addWidget(self._status)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        lay.addWidget(splitter, 1)
        self._side_combo.currentIndexChanged.connect(self._on_side_changed)

    def _pick_config(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择 YAML 配置", self._config_entry.text().strip() or ".", "YAML (*.yaml *.yml)")
        if path:
            self._config_entry.setText(path)

    def _load_config(self):
        path_text = self._config_entry.text().strip()
        if not path_text:
            QMessageBox.warning(self, "提示", "请先选择 YAML 配置文件。")
            return
        path = Path(path_text)
        if not path.exists():
            QMessageBox.warning(self, "提示", "配置文件不存在。")
            return
        try:
            self._original_yaml_text = path.read_text(encoding="utf-8")
            self._config_data = _load_yaml_file(path)
        except Exception as exc:
            QMessageBox.critical(self, "加载失败", str(exc))
            return

        self._config_path = path
        self._roi_fields = _collect_roi_fields(self._config_data)
        for side in ("top", "bottom"):
            if not self._manual_sample_override[side]:
                found = _find_sample_images(self._config_data, side)
                self._sample_image_paths[side] = found[0] if found else ""
        self._refresh_field_list()
        self._sync_sample_path_entry()

        self._field_list.blockSignals(True)
        self._field_list.blockSignals(False)

        if not self._roi_fields:
            self._field_meta.setText("没有识别到可编辑的 ROI 字段。")
            self._status.setText("当前 YAML 中没有找到 `roi` 或 `rois` 结构。")
            return

        self._field_meta.setText(f"共识别到 {len(self._roi_fields)} 个 ROI 字段。")
        if yaml is None:
            self._status.setText("配置已加载。保存时会优先只替换 ROI 文本块，尽量不改动其他注释和排版。")
        else:
            self._status.setText("配置已加载，选择左侧 ROI 字段后即可在样本图上调整。")
        if self._field_list.count() > 0:
            self._field_list.setCurrentRow(0)

    def _on_field_selected(self, row: int):
        if row < 0 or row >= len(self._visible_field_indices):
            self._current_field = None
            self._current_field_index = -1
            return
        self._current_field_index = self._visible_field_indices[row]
        self._current_field = self._roi_fields[self._current_field_index]
        self._load_current_field_rects()

    def _load_current_image(self):
        image_path = self._sample_image_paths.get(self._current_image_side, "")
        self._sync_sample_path_entry()
        if not image_path:
            self._suspend_rect_sync = True
            self._preview.set_pixmap(cv2_to_qpixmap(None))
            self._suspend_rect_sync = False
            self._image_meta.setText(f"当前图片：{self._current_image_side} 未选择")
            return
        img = read_image(image_path)
        self._suspend_rect_sync = True
        self._preview.set_pixmap(cv2_to_qpixmap(img), preserve_rects=False)
        self._suspend_rect_sync = False
        height, width = img.shape[:2]
        self._image_meta.setText(f"当前图片：{Path(image_path).name}  ({width} x {height})")

    def _sync_side_combo(self):
        idx = 0 if self._current_image_side == "top" else 1
        self._side_combo.blockSignals(True)
        self._side_combo.setCurrentIndex(idx)
        self._side_combo.blockSignals(False)
        self._side_badge.setText(self._current_image_side.upper())

    def _sync_sample_path_entry(self):
        path = self._sample_image_paths.get(self._current_image_side, "")
        if path:
            image_path = Path(path)
            self._sample_image_bar.setText(image_path.name)
            self._sample_image_bar.setToolTip(path)
            self._sample_image_meta.setText(f"{self._current_image_side.upper()} · {path}")
            self._select_badge.setText("已选择")
            self._select_badge.setStyleSheet(status_badge_stylesheet("success"))
        else:
            self._sample_image_bar.setText("未选择图片")
            self._sample_image_bar.setToolTip("")
            self._sample_image_meta.setText("请选择一张图片作为当前编辑底图。")
            self._select_badge.setText("未选择")
            self._select_badge.setStyleSheet(status_badge_stylesheet("neutral"))

    def _refresh_field_list(self):
        side = self._current_image_side
        self._visible_field_indices = [idx for idx, field in enumerate(self._roi_fields) if field.side == side]
        self._field_list.blockSignals(True)
        self._field_list.clear()
        for field_index in self._visible_field_indices:
            field = self._roi_fields[field_index]
            item = QListWidgetItem(field.display_name)
            item.setData(Qt.UserRole, field.path_key)
            self._field_list.addItem(item)
        self._field_list.blockSignals(False)

    def _on_side_changed(self):
        side = str(self._side_combo.currentData() or "top")
        self._current_image_side = side
        self._side_badge.setText(side.upper())
        self._refresh_field_list()
        self._load_current_image()
        if self._field_list.count() > 0:
            self._field_list.setCurrentRow(0)
        else:
            self._preview.clear_all_rects()
            self._field_meta.setText("当前侧别没有可编辑的 ROI 字段。")
            self._status.setText(f"当前为 {side.upper()}，请先选择图片或切换到有 ROI 的侧别。")

    def _pick_sample_image(self):
        current_path = self._sample_image_paths.get(self._current_image_side, "")
        current_dir = str(Path(current_path).parent) if current_path else "."
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择样本图片",
            current_dir,
            "Images (*.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff)",
        )
        if not path:
            return
        side = self._current_image_side
        self._manual_sample_override[side] = True
        image_path = Path(path)
        self._sample_image_paths[side] = str(image_path)
        self._sync_sample_path_entry()
        self._load_current_image()
        self._restore_current_field_rects()
        self._status.setText(f"已指定 {side.upper()} 样本图：{image_path.name}")

    def _clear_sample_image(self):
        side = self._current_image_side
        self._manual_sample_override[side] = True
        self._sample_image_paths[side] = ""
        self._sync_sample_path_entry()
        self._load_current_image()
        self._status.setText(f"已清除 {side.upper()} 当前图片。")

    def _load_current_field_rects(self):
        if self._current_field is None or self._config_data is None:
            return
        rects = _normalize_rects(_get_value(self._config_data, self._current_field.path_parts))
        self._rebuild_preview_from_config(selected_field_index=self._current_field_index, selected_roi_index=0 if rects else -1)
        self._preview.rect_select_mode = True
        mode_text = "多区域" if self._current_field.is_multi else "单区域"
        self._field_meta.setText(f"{self._current_field.display_name}\n归属：{self._current_field.side} · {mode_text}")
        self._status.setText(f"已加载 {self._current_field.display_name}，当前 {len(rects)} 个 ROI。框内拖动可移动，拖四角可缩放。")

    def _apply_rects_to_preview(self, rects: list[tuple[int, int, int, int]], rect_map: list[tuple[int, int]], selected_preview_index: int):
        self._suspend_rect_sync = True
        self._preview.clear_all_rects()
        self._preview._rects_image = list(rects)
        self._display_rect_map = list(rect_map)
        self._preview.set_selected_rect_index(selected_preview_index if rects else -1)
        self._suspend_rect_sync = False
        self._refresh_current_roi_list()
        self._preview.update()

    def _refresh_current_roi_list(self):
        if self._current_field is None or self._config_data is None:
            self._roi_list.blockSignals(True)
            self._roi_list.clear()
            self._roi_list.blockSignals(False)
            return
        current_rects = _normalize_rects(_get_value(self._config_data, self._current_field.path_parts))
        self._roi_list.blockSignals(True)
        self._roi_list.clear()
        for idx, (x1, y1, x2, y2) in enumerate(current_rects, start=1):
            self._roi_list.addItem(f"roi_{idx}: ({x1}, {y1}) -> ({x2}, {y2})")
        self._roi_list.blockSignals(False)

    def _rebuild_preview_from_config(self, *, selected_field_index: int | None = None, selected_roi_index: int = 0):
        if self._config_data is None:
            return
        rects: list[tuple[int, int, int, int]] = []
        rect_map: list[tuple[int, int]] = []
        selected_preview_index = -1
        for field_index, field in enumerate(self._roi_fields):
            if field.side != self._current_image_side:
                continue
            field_rects = _normalize_rects(_get_value(self._config_data, field.path_parts))
            for roi_index, rect in enumerate(field_rects):
                if selected_field_index == field_index and selected_roi_index == roi_index:
                    selected_preview_index = len(rects)
                rects.append(rect)
                rect_map.append((field_index, roi_index))
        self._apply_rects_to_preview(rects, rect_map, selected_preview_index)

    def _restore_current_field_rects(self):
        if self._current_field is None or self._config_data is None:
            return
        rects = _normalize_rects(_get_value(self._config_data, self._current_field.path_parts))
        self._rebuild_preview_from_config(selected_field_index=self._current_field_index, selected_roi_index=0 if rects else -1)

    def _start_select(self):
        self._preview.rect_select_mode = True
        self._status.setText("ROI 编辑已开启：框内拖动可移动，拖四角可缩放，空白处拖拽可新建。")

    def _undo_rect(self):
        self._preview.remove_last_rect()

    def _clear_rects(self):
        self._preview.clear_all_rects()
        self._status.setText("当前字段的 ROI 已清空。")

    def _preview_select_row(self, row: int):
        if self._current_field is None or self._current_field_index < 0:
            return
        preview_index = -1
        for idx, (field_index, roi_index) in enumerate(self._display_rect_map):
            if field_index == self._current_field_index and roi_index == row:
                preview_index = idx
                break
        self._preview.set_selected_rect_index(preview_index)

    def _sync_selected_roi_row(self, index: int):
        if index < 0 or index >= len(self._display_rect_map):
            self._roi_list.blockSignals(True)
            self._roi_list.setCurrentRow(-1)
            self._roi_list.blockSignals(False)
            return
        field_index, roi_index = self._display_rect_map[index]
        if field_index != self._current_field_index:
            visible_row = self._visible_field_indices.index(field_index) if field_index in self._visible_field_indices else -1
            self._field_list.blockSignals(True)
            self._field_list.setCurrentRow(visible_row)
            self._field_list.blockSignals(False)
            self._current_field_index = field_index
            self._current_field = self._roi_fields[field_index]
        mode_text = "多区域" if self._current_field.is_multi else "单区域"
        self._field_meta.setText(f"{self._current_field.display_name}\n归属：{self._current_field.side} · {mode_text}")
        self._refresh_current_roi_list()
        self._roi_list.blockSignals(True)
        self._roi_list.setCurrentRow(roi_index)
        self._roi_list.blockSignals(False)

    def _on_rects_changed(self, *args):
        if self._suspend_rect_sync:
            return
        rects = self._preview.get_roi_rects()
        if self._config_data is None or self._current_field is None or self._current_field_index < 0:
            return

        rect_map = list(self._display_rect_map)
        current_field_rects = _normalize_rects(_get_value(self._config_data, self._current_field.path_parts))
        if len(rects) == len(rect_map) + 1:
            rect_map.append((self._current_field_index, len(current_field_rects)))
        elif len(rects) == len(rect_map) - 1 and rect_map:
            removed_field_index, removed_roi_index = rect_map.pop()
            removed_field = self._roi_fields[removed_field_index]
            removed_rects = _normalize_rects(_get_value(self._config_data, removed_field.path_parts))
            if 0 <= removed_roi_index < len(removed_rects):
                removed_rects.pop(removed_roi_index)
                _write_rects_back(removed_field, self._config_data, removed_rects)

        grouped: dict[int, list[tuple[int, int, int, int]]] = {}
        for rect, (field_index, _roi_index) in zip(rects, rect_map):
            grouped.setdefault(field_index, []).append(rect)

        for field_index, field in enumerate(self._roi_fields):
            if field.side != self._current_image_side:
                continue
            field_rects = grouped.get(field_index, [])
            if not field.is_multi and len(field_rects) > 1:
                field_rects = field_rects[:1]
            _write_rects_back(field, self._config_data, field_rects)

        current_rects = _normalize_rects(_get_value(self._config_data, self._current_field.path_parts))
        self._rebuild_preview_from_config(selected_field_index=self._current_field_index, selected_roi_index=0 if current_rects else -1)
        self._status.setText(f"{self._current_field.display_name} 已更新，当前 {len(current_rects)} 个 ROI。")

    def _save_current_field(self):
        if self._current_field is None or self._config_data is None:
            QMessageBox.warning(self, "提示", "请先加载配置并选择一个 ROI 字段。")
            return
        rects = self._preview.get_roi_rects()
        _write_rects_back(self._current_field, self._config_data, rects)
        self._status.setText(f"已写入内存配置：{self._current_field.display_name}")

    def _save_yaml(self):
        if self._config_path is None or self._config_data is None:
            QMessageBox.warning(self, "提示", "请先加载配置。")
            return
        try:
            self._original_yaml_text = _dump_yaml_file(
                self._config_path,
                self._config_data,
                self._roi_fields,
                self._original_yaml_text,
            )
        except Exception as exc:
            QMessageBox.critical(self, "保存失败", str(exc))
            return
        self._status.setText(f"已保存到 {self._config_path.name}")
        self._mw.show_status("ROI 配置已保存")

    def _toggle_left_panel(self):
        if self._left_panel_visible:
            sizes = self._splitter.sizes()
            if sizes and sizes[0] > 0:
                self._left_panel_width = sizes[0]
            self._left_panel_container.hide()
            self._splitter.setSizes([0, 1])
            self._btn_toggle_sidebar.setText("展开侧栏")
            self._left_panel_visible = False
        else:
            self._left_panel_container.show()
            self._splitter.setSizes([self._left_panel_width, 960])
            self._btn_toggle_sidebar.setText("收起侧栏")
            self._left_panel_visible = True

    def keyPressEvent(self, event):
        super().keyPressEvent(event)
