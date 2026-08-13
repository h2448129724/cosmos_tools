"""YAML ROI 配置可视化编辑工具。"""
from __future__ import annotations

from dataclasses import dataclass
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
try:
    from ruamel.yaml import YAML
except ImportError:  # pragma: no cover - depends on local env
    YAML = None
    import yaml as pyyaml
else:
    pyyaml = None

from apps.data_tools.processing.image_io import read_image
from ..preview_widget import ZoomableLabel, cv2_to_qpixmap
from .base import BaseToolPage, make_card, make_page_header, set_primary


if YAML is not None:
    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.width = 4096
else:
    yaml = None


@dataclass
class RoiField:
    path_parts: tuple[Any, ...]
    display_name: str
    side: str
    is_multi: bool

    @property
    def path_key(self) -> str:
        return ".".join(str(part) for part in self.path_parts)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float))


def _is_rect_list(value: Any) -> bool:
    return isinstance(value, list) and len(value) == 4 and all(_is_number(v) for v in value)


def _is_multi_rect_list(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and all(_is_rect_list(item) for item in value)


def _normalize_rects(value: Any) -> list[tuple[int, int, int, int]]:
    if _is_rect_list(value):
        x1, y1, x2, y2 = value
        return [(int(x1), int(y1), int(x2), int(y2))]
    if _is_multi_rect_list(value):
        return [tuple(int(v) for v in item) for item in value]
    return []


def _write_rects_back(field: RoiField, data: Any, rects: list[tuple[int, int, int, int]]) -> None:
    target = data
    for part in field.path_parts[:-1]:
        target = target[part]
    if field.is_multi:
        target[field.path_parts[-1]] = [[int(v) for v in rect] for rect in rects]
    else:
        if not rects:
            target[field.path_parts[-1]] = []
        else:
            target[field.path_parts[-1]] = [int(v) for v in rects[0]]


def _get_value(data: Any, path_parts: tuple[Any, ...]) -> Any:
    target = data
    for part in path_parts:
        target = target[part]
    return target


def _detect_side(path_parts: tuple[Any, ...]) -> str:
    if "top" in path_parts:
        return "top"
    if "bottom" in path_parts:
        return "bottom"
    return "unknown"


def _collect_roi_fields(
    node: Any,
    path_parts: tuple[Any, ...] = (),
    cab_f_config: bool | None = None,
) -> list[RoiField]:
    if cab_f_config is None:
        inspection = node.get("inspection", {}) if isinstance(node, dict) else {}
        cab_f_config = str(inspection.get("project", "")).upper() == "CAB-F"
    fields: list[RoiField] = []
    if isinstance(node, dict):
        for key, value in node.items():
            next_path = path_parts + (key,)
            key_str = str(key).lower()
            is_roi_key = key_str == "roi" if cab_f_config else key_str in {"roi", "rois"}
            is_empty_multi_roi = cab_f_config and key_str == "roi" and isinstance(value, list) and not value
            if is_roi_key and (
                _is_rect_list(value) or _is_multi_rect_list(value) or is_empty_multi_roi
            ):
                side = _detect_side(next_path)
                display = ".".join(str(part) for part in next_path)
                fields.append(RoiField(next_path, display, side, _is_multi_rect_list(value) or is_empty_multi_roi))
            fields.extend(_collect_roi_fields(value, next_path, cab_f_config))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            fields.extend(_collect_roi_fields(value, path_parts + (index,), cab_f_config))
    return fields


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


def _format_roi_value(field: RoiField, rects: list[tuple[int, int, int, int]]) -> str:
    if field.is_multi:
        inner = ", ".join(f"[{x1}, {y1}, {x2}, {y2}]" for x1, y1, x2, y2 in rects)
        return f"[{inner}]"
    if not rects:
        return "[]"
    x1, y1, x2, y2 = rects[0]
    return f"[{x1}, {y1}, {x2}, {y2}]"


def _line_indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _find_yaml_key_line(lines: list[str], path_parts: tuple[Any, ...]) -> int:
    if not path_parts or any(not isinstance(part, str) for part in path_parts):
        return -1

    start = 0
    end = len(lines)
    parent_indent = -1

    for depth, key in enumerate(path_parts):
        found_index = -1
        found_indent = -1
        expected_prefix = f"{key}:"
        for idx in range(start, end):
            stripped = lines[idx].strip()
            if not stripped or stripped.startswith("#"):
                continue
            if not stripped.startswith(expected_prefix):
                continue
            indent = _line_indent(lines[idx])
            if indent <= parent_indent:
                continue
            if found_index == -1 or indent < found_indent:
                found_index = idx
                found_indent = indent
        if found_index == -1:
            return -1
        if depth == len(path_parts) - 1:
            return found_index

        child_start = found_index + 1
        child_end = len(lines)
        for idx in range(child_start, len(lines)):
            stripped = lines[idx].strip()
            if not stripped or stripped.startswith("#"):
                continue
            indent = _line_indent(lines[idx])
            if indent <= found_indent:
                child_end = idx
                break
        start = child_start
        end = child_end
        parent_indent = found_indent

    return -1


def _replace_roi_line(lines: list[str], line_index: int, field: RoiField, rects: list[tuple[int, int, int, int]]) -> None:
    line = lines[line_index]
    newline = "\n" if line.endswith("\n") else ""
    body = line[:-1] if newline else line
    comment = ""
    hash_index = body.find("#")
    if hash_index >= 0:
        comment = body[hash_index:]
        body = body[:hash_index].rstrip()
    prefix, _, _ = body.partition(":")
    replaced = f"{prefix}: {_format_roi_value(field, rects)}"
    if comment:
        replaced = f"{replaced}  {comment}"
    lines[line_index] = replaced + newline


def _patch_yaml_roi_text(original_text: str, data: Any, fields: list[RoiField]) -> str:
    lines = original_text.splitlines(keepends=True)
    for field in fields:
        line_index = _find_yaml_key_line(lines, field.path_parts)
        if line_index < 0:
            continue
        rects = _normalize_rects(_get_value(data, field.path_parts))
        _replace_roi_line(lines, line_index, field, rects)
    return "".join(lines)


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
        config_title.setStyleSheet("color:#0f172a;font-size:15px;font-weight:700;")
        config_card_lay.addWidget(config_title)

        config_desc = QLabel("选择配置与当前编辑图片。图片只会在你手动选择时变更。")
        config_desc.setWordWrap(True)
        config_desc.setStyleSheet("color:#64748b;")
        config_card_lay.addWidget(config_desc)

        cfg_box = QFrame()
        cfg_form = QFormLayout(cfg_box)
        cfg_form.setContentsMargins(0, 0, 0, 0)
        self._config_entry = QLineEdit("")
        self._side_combo = QComboBox()
        self._side_combo.addItem("TOP", "top")
        self._side_combo.addItem("BOTTOM", "bottom")
        cfg_form.addRow("配置文件", self._config_entry)
        cfg_form.addRow("样本侧别", self._side_combo)
        config_card_lay.addWidget(cfg_box)

        image_state_card = QFrame()
        image_state_card.setObjectName("hintPanel")
        image_state_lay = QVBoxLayout(image_state_card)
        image_state_lay.setContentsMargins(12, 10, 12, 10)
        image_state_lay.setSpacing(4)
        image_state_label = QLabel("当前图片")
        image_state_label.setStyleSheet("color:#0f172a;font-weight:700;")
        image_state_lay.addWidget(image_state_label)
        badge_row = QHBoxLayout()
        badge_row.setSpacing(8)
        self._side_badge = QLabel("TOP")
        self._side_badge.setStyleSheet("background:#D8E3EE;color:#28577F;border:1px solid #C5D4E1;border-radius:2px;padding:2px 7px;font-weight:700;")
        self._select_badge = QLabel("未选择")
        self._select_badge.setStyleSheet("background:#E9EBED;color:#555D64;border:1px solid #CFD3D7;border-radius:2px;padding:2px 7px;font-weight:700;")
        badge_row.addWidget(self._side_badge)
        badge_row.addWidget(self._select_badge)
        badge_row.addStretch(1)
        image_state_lay.addLayout(badge_row)
        self._sample_image_bar = QLabel("未选择图片")
        self._sample_image_bar.setWordWrap(True)
        self._sample_image_bar.setStyleSheet("color:#0f172a;font-weight:700;")
        image_state_lay.addWidget(self._sample_image_bar)
        self._sample_image_meta = QLabel("请选择一张图片作为当前编辑底图。")
        self._sample_image_meta.setWordWrap(True)
        self._sample_image_meta.setStyleSheet("color:#64748b;")
        image_state_lay.addWidget(self._sample_image_meta)
        config_card_lay.addWidget(image_state_card)

        cfg_btns = QHBoxLayout()
        btn_cfg = QPushButton("浏览配置")
        btn_cfg.setMinimumHeight(36)
        btn_cfg.clicked.connect(self._pick_config)
        btn_sample_image = QPushButton("选择图片")
        btn_sample_image.setMinimumHeight(36)
        btn_sample_image.clicked.connect(self._pick_sample_image)
        btn_clear_image = QPushButton("清除当前图片")
        btn_clear_image.setMinimumHeight(36)
        btn_clear_image.clicked.connect(self._clear_sample_image)
        btn_load = QPushButton("加载配置")
        btn_load.setMinimumHeight(36)
        set_primary(btn_load)
        btn_load.clicked.connect(self._load_config)
        cfg_btns.addWidget(btn_cfg)
        cfg_btns.addWidget(btn_sample_image)
        cfg_btns.addWidget(btn_clear_image)
        cfg_btns.addWidget(btn_load)
        config_card_lay.addLayout(cfg_btns)
        left_lay.addWidget(config_card)

        editor_card = make_card()
        editor_card_lay = QVBoxLayout(editor_card)
        editor_card_lay.setContentsMargins(14, 14, 14, 14)
        editor_card_lay.setSpacing(8)
        editor_title = QLabel("ROI 编辑区")
        editor_title.setStyleSheet("color:#0f172a;font-size:15px;font-weight:700;")
        editor_card_lay.addWidget(editor_title)

        editor_desc = QLabel("选择字段后，在右侧直接编辑 ROI。")
        editor_desc.setWordWrap(True)
        editor_desc.setStyleSheet("color:#64748b;")
        editor_card_lay.addWidget(editor_desc)

        self._field_list = QListWidget()
        self._field_list.currentRowChanged.connect(self._on_field_selected)
        field_section = QFrame()
        field_section_lay = QVBoxLayout(field_section)
        field_section_lay.setContentsMargins(0, 0, 0, 0)
        field_section_lay.setSpacing(8)
        field_title = QLabel("ROI 字段")
        field_title.setStyleSheet("color:#0f172a;font-weight:700;")
        field_section_lay.addWidget(field_title)
        field_section_lay.addWidget(self._field_list, 1)
        editor_card_lay.addWidget(field_section, 1)

        self._field_meta = QLabel("等待加载配置。")
        self._field_meta.setWordWrap(True)
        self._field_meta.setStyleSheet("color:#64748b;")
        editor_card_lay.addWidget(self._field_meta)

        self._edit_help = QLabel("框内拖动可移动，拖四角可缩放。")
        self._edit_help.setWordWrap(True)
        self._edit_help.setStyleSheet("color:#64748b;")
        editor_card_lay.addWidget(self._edit_help)

        roi_btns = QHBoxLayout()
        btn_start = QPushButton("开始框选")
        btn_start.setMinimumHeight(36)
        set_primary(btn_start)
        btn_start.clicked.connect(self._start_select)
        btn_undo = QPushButton("撤销上一个")
        btn_undo.setMinimumHeight(36)
        btn_undo.clicked.connect(self._undo_rect)
        btn_clear = QPushButton("清空当前字段")
        btn_clear.setMinimumHeight(36)
        btn_clear.clicked.connect(self._clear_rects)
        roi_btns.addWidget(btn_start)
        roi_btns.addWidget(btn_undo)
        roi_btns.addWidget(btn_clear)
        editor_card_lay.addLayout(roi_btns)

        self._roi_list = QListWidget()
        self._roi_list.setMinimumHeight(96)
        self._roi_list.setMaximumHeight(156)
        self._roi_list.currentRowChanged.connect(self._preview_select_row)
        roi_section = QFrame()
        roi_section_lay = QVBoxLayout(roi_section)
        roi_section_lay.setContentsMargins(0, 0, 0, 0)
        roi_section_lay.setSpacing(8)
        roi_title = QLabel("当前字段 ROI")
        roi_title.setStyleSheet("color:#0f172a;font-weight:700;")
        roi_section_lay.addWidget(roi_title)
        roi_section_lay.addWidget(self._roi_list)
        editor_card_lay.addWidget(roi_section)

        io_btns = QHBoxLayout()
        btn_save_field = QPushButton("保存当前字段到配置")
        btn_save_field.setMinimumHeight(36)
        btn_save_field.clicked.connect(self._save_current_field)
        btn_save_all = QPushButton("写回 YAML")
        btn_save_all.setMinimumHeight(36)
        set_primary(btn_save_all)
        btn_save_all.clicked.connect(self._save_yaml)
        io_btns.addWidget(btn_save_field)
        io_btns.addWidget(btn_save_all)
        editor_card_lay.addLayout(io_btns)
        left_lay.addWidget(editor_card, 1)

        splitter.addWidget(left_scroll)

        right = make_card()
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(16, 16, 16, 16)
        right_lay.setSpacing(8)

        topbar = QHBoxLayout()
        self._btn_toggle_sidebar = QPushButton("收起侧栏")
        self._btn_toggle_sidebar.setMinimumHeight(36)
        self._btn_toggle_sidebar.clicked.connect(self._toggle_left_panel)
        topbar.addWidget(self._btn_toggle_sidebar)
        self._btn_save_quick = QPushButton("保存配置")
        self._btn_save_quick.setMinimumHeight(36)
        set_primary(self._btn_save_quick)
        self._btn_save_quick.clicked.connect(self._save_yaml)
        topbar.addWidget(self._btn_save_quick)
        self._image_meta = QLabel("当前样本：未加载")
        self._image_meta.setStyleSheet("color:#475569;")
        self._image_meta.setWordWrap(True)
        topbar.addWidget(self._image_meta, 1)
        right_lay.addLayout(topbar)

        self._preview = ZoomableLabel()
        self._preview.setMinimumHeight(240)
        self._preview.rectSelected.connect(self._on_rects_changed)
        self._preview.rectsChanged.connect(self._on_rects_changed)
        self._preview.rectSelectionChanged.connect(self._sync_selected_roi_row)
        right_lay.addWidget(self._preview, 1)

        self._status = QLabel("请先加载 YAML 配置。")
        self._status.setStyleSheet("color:#64748b;")
        right_lay.addWidget(self._status)

        splitter.addWidget(right)
        splitter.setSizes([420, 960])
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
            self._select_badge.setStyleSheet("background:#E3EEE7;color:#356848;border:1px solid #C8D8CE;border-radius:2px;padding:2px 7px;font-weight:700;")
        else:
            self._sample_image_bar.setText("未选择图片")
            self._sample_image_bar.setToolTip("")
            self._sample_image_meta.setText("请选择一张图片作为当前编辑底图。")
            self._select_badge.setText("未选择")
            self._select_badge.setStyleSheet("background:#E9EBED;color:#555D64;border:1px solid #CFD3D7;border-radius:2px;padding:2px 7px;font-weight:700;")

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
