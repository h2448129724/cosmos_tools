"""逐张裁剪工具页面。"""

from __future__ import annotations

import os

from PySide6.QtCore import QEvent, Qt, QTimer
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
)

from apps.data_tools.processing.batch_crop import crop_single_image
from apps.data_tools.processing.image_io import read_image
from cosmos_toolbox.ui.primitives import PathField, set_ui_role
from img_tools.core.crop_plan import CropRegion, allocate_crop_names, build_image_crop_plan
from img_tools.core.fixed_roi import resolve_fixed_boxes, resize_fixed_box

from ..preview_widget import ZoomableLabel
from .base import BaseToolPage, make_log_box, make_page_header, set_primary


class SingleCropPage(BaseToolPage):
    tool_key = "single_crop"
    tool_title = "逐张裁剪"
    tool_nav_title = "逐张裁剪"
    tool_icon = "▢"
    tool_summary = "按固定尺寸逐张调整 ROI 位置，并只裁剪当前图片。"
    tool_tags = ("ROI 校正", "固定尺寸", "单张输出")

    def __init__(self, main_window, parent=None):
        super().__init__(main_window, parent)
        self._ref_w = 0
        self._ref_h = 0
        self._files: list[str] = []
        self._current_file_index = -1
        self._boxes: dict[str, tuple[tuple[int, int, int, int], ...]] = {}
        self._cropped: set[str] = set()
        self._showing = False
        self._user_zoomed = False
        self._build_ui()
        self._preview.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._preview.installEventFilter(self)
        self._preview.set_fixed_rect_size(self._requested_size())
        for box in (self._fixed_width, self._fixed_height):
            editor = box.lineEdit()
            if editor is not None:
                editor.installEventFilter(self)

    def _action_button(self, text: str, accessible: str, slot, *, primary: bool = False) -> QPushButton:
        button = QPushButton(text)
        button.setAccessibleName(accessible)
        button.setMinimumHeight(36)
        button.setMinimumWidth(96)
        button.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        button.clicked.connect(slot)
        if primary:
            set_primary(button)
        return button

    def _build_ui(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 10, 12, 10)
        lay.setSpacing(8)

        lay.addWidget(make_page_header(
            "逐张裁剪",
            "一张图可以框多个区域。每次裁剪都会写成新文件，不会覆盖。",
        ))

        path_row = QHBoxLayout()
        path_row.setSpacing(12)
        input_field = PathField("图片目录", placeholder="选择包含图片的目录")
        input_field.browse_requested.connect(self._pick_input)
        self._in_entry = input_field.line_edit
        output_field = PathField("输出目录", placeholder="裁剪结果写入此目录")
        output_field.browse_requested.connect(lambda: self._pick_dir(self._out_entry))
        self._out_entry = output_field.line_edit
        path_row.addWidget(input_field, 1)
        path_row.addWidget(output_field, 1)
        load_button = self._action_button("加载目录", "加载图片目录", self._load_folder, primary=True)
        path_row.addWidget(load_button, 0, Qt.AlignmentFlag.AlignBottom)
        lay.addLayout(path_row)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(8)

        left = QFrame()
        left.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        left_lay = QVBoxLayout(left)
        left_lay.setContentsMargins(0, 4, 8, 0)
        left_lay.setSpacing(8)

        heading = QHBoxLayout()
        heading.setSpacing(12)
        self._current_name = QLabel("当前图片：未加载")
        set_ui_role(self._current_name, "sectionTitle")
        self._current_name.setAccessibleName("当前图片")
        self._current_name.setWordWrap(True)
        self._zoom_label = QLabel("100%")
        self._zoom_label.setAccessibleName("当前缩放比例")
        set_ui_role(self._zoom_label, "sectionTitle")
        heading.addWidget(self._current_name, 1)
        heading.addWidget(self._zoom_label, 0, Qt.AlignmentFlag.AlignRight)
        left_lay.addLayout(heading)

        zoom_row = QHBoxLayout()
        zoom_row.setSpacing(8)
        self._zoom_out_button = self._action_button("缩小", "缩小预览", self._zoom_out)
        self._zoom_in_button = self._action_button("放大", "放大预览", self._zoom_in)
        self._fit_button = self._action_button("适应窗口", "适应窗口", self._fit_preview)
        self._actual_button = self._action_button("1:1", "按原始像素显示", self._zoom_actual)
        for button in (self._zoom_out_button, self._zoom_in_button, self._fit_button, self._actual_button):
            zoom_row.addWidget(button)
        zoom_row.addStretch(1)
        zoom_hint = QLabel("滚轮缩放，中键拖动画布，A / D 切图")
        set_ui_role(zoom_hint, "muted")
        zoom_row.addWidget(zoom_hint)
        left_lay.addLayout(zoom_row)

        self._preview = ZoomableLabel()
        self._preview.setMinimumSize(560, 460)
        self._preview.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._preview.setAccessibleName("图片预览与 ROI 画布")
        self._preview.setAutoFillBackground(True)
        preview_palette = self._preview.palette()
        preview_palette.setColor(QPalette.ColorRole.Window, QColor("#d7dbde"))
        self._preview.setPalette(preview_palette)
        self._preview.rectsChanged.connect(self._on_rects_changed)
        self._preview.zoomChanged.connect(self._update_zoom_label)
        left_lay.addWidget(self._preview, 1)

        self._preview_meta = QLabel("加载图片目录后，点击图片放置 ROI。")
        set_ui_role(self._preview_meta, "muted")
        self._preview_meta.setAccessibleName("预览信息")
        self._preview_meta.setWordWrap(True)
        left_lay.addWidget(self._preview_meta)

        self._box_info = QLabel("当前框：未放置")
        self._box_info.setWordWrap(True)
        self._box_info.setAccessibleName("当前 ROI")
        set_ui_role(self._box_info, "muted")
        left_lay.addWidget(self._box_info)

        right = QFrame()
        right.setMinimumWidth(292)
        right.setMaximumWidth(360)
        right.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(8, 4, 0, 0)
        right_lay.setSpacing(8)

        self._task_current = QLabel("当前图片：0 / 0")
        self._task_size = QLabel("图片尺寸：-")
        self._task_state = QLabel("状态：待加载目录")
        for label in (self._task_current, self._task_size, self._task_state):
            set_ui_role(label, "muted")
            label.setWordWrap(True)
            right_lay.addWidget(label)

        size_row = QHBoxLayout()
        size_row.setSpacing(8)
        self._fixed_width = QSpinBox()
        self._fixed_height = QSpinBox()
        for box in (self._fixed_width, self._fixed_height):
            box.setRange(1, 1_000_000)
            box.setValue(512)
            box.setMinimumHeight(36)
            box.setMinimumWidth(108)
            box.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._fixed_width.setAccessibleName("ROI 宽度")
        self._fixed_height.setAccessibleName("ROI 高度")
        width_label = QLabel("宽")
        height_label = QLabel("高")
        size_row.addWidget(width_label)
        size_row.addWidget(self._fixed_width, 1)
        size_row.addWidget(height_label)
        size_row.addWidget(self._fixed_height, 1)
        right_lay.addLayout(size_row)
        size_hint = QLabel("改数值只调整选中的框，并保持它的中心。点击图片可继续追加区域。")
        size_hint.setWordWrap(True)
        set_ui_role(size_hint, "muted")
        right_lay.addWidget(size_hint)
        edit_row = QHBoxLayout()
        edit_row.setSpacing(8)
        undo_button = self._action_button("撤销上一个", "撤销上一个 ROI", self._undo_last)
        delete_button = self._action_button("删除选中", "删除选中 ROI", self._delete_selected)
        edit_row.addWidget(undo_button, 1)
        edit_row.addWidget(delete_button, 1)
        right_lay.addLayout(edit_row)
        self._fixed_width.valueChanged.connect(self._on_size_changed)
        self._fixed_height.valueChanged.connect(self._on_size_changed)

        nav_row = QHBoxLayout()
        nav_row.setSpacing(8)
        self._prev_button = self._action_button("上一张", "上一张图片", lambda: self._show_offset(-1))
        self._next_button = self._action_button("下一张", "下一张图片", lambda: self._show_offset(1))
        nav_row.addWidget(self._prev_button, 1)
        nav_row.addWidget(self._next_button, 1)
        right_lay.addLayout(nav_row)

        self._crop_button = self._action_button("裁剪当前图片", "裁剪当前图片", lambda: self._crop_current(advance=False))
        self._advance_button = self._action_button(
            "裁剪并下一张",
            "裁剪并下一张",
            lambda: self._crop_current(advance=True),
            primary=True,
        )
        for button in (self._crop_button, self._advance_button):
            button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            right_lay.addWidget(button)

        self._summary = QLabel("等待加载图片目录。")
        self._summary.setWordWrap(True)
        set_ui_role(self._summary, "muted")
        self._summary.setAccessibleName("裁剪摘要")
        right_lay.addWidget(self._summary)

        list_label = QLabel("图片列表")
        set_ui_role(list_label, "sectionTitle")
        right_lay.addWidget(list_label)
        self._file_list = QListWidget()
        self._file_list.setAccessibleName("图片文件列表")
        self._file_list.setMinimumHeight(160)
        self._file_list.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._file_list.currentRowChanged.connect(self._show_file_at)
        right_lay.addWidget(self._file_list, 1)

        self._log = make_log_box("运行日志...", height=88)
        self._log.setMinimumHeight(72)
        self._log.setMaximumHeight(120)
        right_lay.addWidget(self._log)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setSizes([920, 320])
        lay.addWidget(splitter, 1)
        self._update_task_panel()
        self._update_zoom_label()

    def eventFilter(self, watched, event):  # noqa: N802 - Qt interface
        if event.type() == QEvent.Type.KeyPress and event.key() in (Qt.Key.Key_A, Qt.Key.Key_D):
            self._show_offset(-1 if event.key() == Qt.Key.Key_A else 1)
            return True
        return super().eventFilter(watched, event)

    def keyPressEvent(self, event):  # noqa: N802 - Qt interface
        if event.key() == Qt.Key.Key_A:
            self._show_offset(-1)
            event.accept()
            return
        if event.key() == Qt.Key.Key_D:
            self._show_offset(1)
            event.accept()
            return
        super().keyPressEvent(event)

    def _pick_dir(self, entry: QLineEdit):
        directory = QFileDialog.getExistingDirectory(self._mw, "选择目录", entry.text() or ".")
        if directory:
            entry.setText(directory)

    def _pick_input(self):
        self._pick_dir(self._in_entry)

    def _load_folder(self):
        input_dir = self._in_entry.text().strip()
        if not input_dir or not os.path.isdir(input_dir):
            QMessageBox.warning(self._mw, "提示", "请先选择有效的图片目录")
            return
        from apps.data_tools.common.helpers import get_image_files

        files = get_image_files(input_dir)
        if not files:
            QMessageBox.warning(self._mw, "提示", "目录中没有图片文件")
            return
        self._files = files
        self._boxes.clear()
        self._cropped.clear()
        self._current_file_index = -1
        self._ref_w = 0
        self._ref_h = 0
        self._file_list.blockSignals(True)
        self._file_list.clear()
        for path in self._files:
            self._file_list.addItem(self._file_label(path))
        self._file_list.blockSignals(False)
        self._show_file_at(0)

    def _show_file_at(self, index: int):
        if not self._files or self._showing or index < 0 or index >= len(self._files):
            return
        previous = None
        if 0 <= self._current_file_index < len(self._files):
            self._save_current_boxes()
            previous = self._boxes.get(self._files[self._current_file_index])
        path = self._files[index]
        img = read_image(path)
        if img is None:
            QMessageBox.warning(self._mw, "提示", f"无法读取图片: {os.path.basename(path)}")
            return

        self._showing = True
        try:
            self._current_file_index = index
            self._ref_h, self._ref_w = img.shape[:2]
            from ..preview_widget import cv2_to_qpixmap

            self._preview.set_pixmap(cv2_to_qpixmap(img), preserve_view=previous is not None)
            self._preview.rect_select_mode = True
            self._preview.set_fixed_rect_size(self._requested_size())
            if previous is None:
                QTimer.singleShot(0, self._fit_preview)
            boxes = resolve_fixed_boxes(
                saved=self._boxes.get(path),
                previous=previous,
                image_size=(self._ref_w, self._ref_h),
            )
            if boxes:
                self._preview.set_roi_rects(list(boxes))
            if path not in self._boxes and boxes:
                self._boxes[path] = boxes
            self._file_list.blockSignals(True)
            self._file_list.setCurrentRow(index)
            self._file_list.blockSignals(False)
            self._current_name.setText(f"当前图片：{os.path.basename(path)}")
            self._preview_meta.setText(
                f"当前第 {index + 1} / {len(self._files)} 张，图片尺寸 {self._ref_w} × {self._ref_h}。"
            )
            self._summary.setText(
                f"当前 {index + 1}/{len(self._files)}：{os.path.basename(path)}。"
                " 点击追加固定框，一张图可以放多个区域。"
            )
            self._log.appendPlainText(f"已加载: {os.path.basename(path)} ({self._ref_w}x{self._ref_h})")
        finally:
            self._showing = False
        self._update_box_label()
        self._update_task_panel()
        self._preview.setFocus()

    def _show_offset(self, delta: int):
        if not self._files or self._showing:
            return
        current = 0 if self._current_file_index < 0 else self._current_file_index
        next_index = max(0, min(len(self._files) - 1, current + delta))
        if next_index != self._current_file_index:
            self._show_file_at(next_index)

    def _zoom_in(self):
        self._user_zoomed = True
        self._preview.zoom_by(1.25)

    def _zoom_out(self):
        self._user_zoomed = True
        self._preview.zoom_by(0.8)

    def _zoom_actual(self):
        self._user_zoomed = True
        self._preview.set_zoom_factor(1.0)

    def _fit_preview(self):
        self._user_zoomed = False
        self._preview.fit_to_view(allow_upscale=True)
        self._update_zoom_label()

    def _update_zoom_label(self, *_args):
        percent = int(round(self._preview.zoom_factor() * 100))
        self._zoom_label.setText(f"{percent}%")

    def _requested_size(self) -> tuple[int, int]:
        return self._fixed_width.value(), self._fixed_height.value()

    def _on_size_changed(self, *_args):
        size = self._requested_size()
        self._preview.set_fixed_rect_size(size)
        if self._showing or self._ref_w <= 0 or self._ref_h <= 0:
            return
        rects = self._preview.get_roi_rects()
        if not rects:
            return
        index = self._preview.selected_rect_index()
        if index < 0 or index >= len(rects):
            index = len(rects) - 1
        resized = resize_fixed_box(rects[index], size, (self._ref_w, self._ref_h))
        if resized == rects[index]:
            self._update_box_label()
            return
        rects[index] = resized
        self._preview.set_roi_rects(rects, selected=index)

    def _on_rects_changed(self):
        if self._showing:
            return
        self._save_current_boxes()
        self._update_box_label()
        self._update_task_panel()

    def _save_current_boxes(self):
        if self._current_file_index < 0 or not self._files:
            return
        path = self._files[self._current_file_index]
        rects = tuple(self._preview.get_roi_rects())
        if rects or path in self._boxes:
            self._boxes[path] = rects

    def _undo_last(self):
        self._preview.remove_last_rect()

    def _delete_selected(self):
        rects = self._preview.get_roi_rects()
        index = self._preview.selected_rect_index()
        if not rects or index < 0 or index >= len(rects):
            return
        del rects[index]
        selected = min(index, len(rects) - 1) if rects else -1
        self._preview.set_roi_rects(rects, selected=selected)

    def _crop_current(self, *, advance: bool):
        output_dir = self._out_entry.text().strip()
        if self._current_file_index < 0 or not self._files:
            QMessageBox.warning(self._mw, "提示", "请先加载图片目录并选择当前图片")
            return
        if not output_dir:
            QMessageBox.warning(self._mw, "提示", "请先选择输出目录")
            return
        rects = self._preview.get_roi_rects()
        if not rects:
            QMessageBox.warning(self._mw, "提示", "请先在图片上放置 ROI")
            return
        if self._ref_w <= 0 or self._ref_h <= 0:
            QMessageBox.warning(self._mw, "提示", "请先加载当前图片")
            return

        path = self._files[self._current_file_index]
        self._boxes[path] = tuple(rects)
        plan = build_image_crop_plan(
            source_name=path,
            image_size=(self._ref_w, self._ref_h),
            regions=(
                CropRegion(x1, y1, x2, y2, f"roi_{index}")
                for index, (x1, y1, x2, y2) in enumerate(rects, start=1)
            ),
            reference_size=(self._ref_w, self._ref_h),
            coordinate_mode="absolute",
            scale_semantics="edges",
            suffix_mode="preserve",
        )
        if not plan.operations:
            reason = plan.rejections[0].reason if plan.rejections else "ROI 无效"
            QMessageBox.warning(self._mw, "提示", reason)
            return
        stem, suffix = os.path.splitext(os.path.basename(path))
        taken = set(os.listdir(output_dir)) if os.path.isdir(output_dir) else set()
        names = allocate_crop_names(stem, suffix, len(plan.operations), taken)
        try:
            total = crop_single_image(
                path,
                rects,
                self._ref_w,
                self._ref_h,
                output_dir,
                coordinate_mode="absolute",
                output_names=list(names),
            )
        except Exception as exc:
            QMessageBox.critical(self._mw, "裁剪失败", str(exc))
            return
        if total <= 0:
            QMessageBox.warning(self._mw, "裁剪失败", f"没有写出裁剪结果: {os.path.basename(path)}")
            return

        details: list[str] = []
        clipped = False
        for operation, name in zip(plan.operations, names, strict=True):
            box = operation.box
            actual_w = box[2] - box[0]
            actual_h = box[3] - box[1]
            details.append(f"{name}（{actual_w} x {actual_h}）")
            source = rects[operation.region_index - 1]
            source_w = source[2] - source[0]
            source_h = source[3] - source[1]
            if (actual_w, actual_h) != (source_w, source_h):
                clipped = True
        message = f"裁剪完成: {os.path.basename(path)} -> " + "，".join(details)
        if clipped:
            message += "，部分结果小于框选范围，已夹到图片内"
        if plan.rejections:
            message += "；未写出：" + "；".join(item.reason for item in plan.rejections)
        self._log.appendPlainText(message)
        self._summary.setText(message)
        self._cropped.add(path)
        self._refresh_file_label(self._current_file_index)
        self._update_task_panel("状态：单张裁剪完成")
        status = getattr(self._mw, "show_status", None)
        if callable(status):
            status("当前图片裁剪完成")
        if advance:
            self._show_offset(1)

    def _file_label(self, path: str) -> str:
        name = os.path.basename(path)
        if path in self._cropped:
            return f"已裁 {name}"
        return name

    def _refresh_file_label(self, index: int):
        if index < 0 or index >= len(self._files):
            return
        item = self._file_list.item(index)
        if item is not None:
            item.setText(self._file_label(self._files[index]))

    def _update_box_label(self):
        requested = self._requested_size()
        rects = self._preview.get_roi_rects()
        if not rects:
            self._box_info.setText(
                f"当前没有区域。点击图片追加 {requested[0]} × {requested[1]}，可以放多个。"
            )
            return
        selected = self._preview.selected_rect_index()
        parts = [f"共 {len(rects)} 个区域"]
        for index, (x1, y1, x2, y2) in enumerate(rects):
            marker = "选中 " if index == selected else ""
            parts.append(f"{marker}{index + 1}: ({x1}, {y1}) {x2 - x1} × {y2 - y1}")
        self._box_info.setText("；".join(parts))

    def _update_task_panel(self, state_text: str | None = None):
        total = len(self._files)
        current = self._current_file_index + 1 if self._current_file_index >= 0 else 0
        self._task_current.setText(f"当前图片：{current} / {total}")
        if self._ref_w and self._ref_h:
            self._task_size.setText(f"图片尺寸：{self._ref_w} × {self._ref_h}")
        else:
            self._task_size.setText("图片尺寸：-")
        if state_text is None:
            if not total:
                state_text = "状态：待加载目录"
            elif not self._preview.get_roi_rects():
                state_text = "状态：可放置 ROI"
            elif self._out_entry.text().strip():
                state_text = "状态：可裁剪"
            else:
                state_text = "状态：已放置 ROI"
        self._task_state.setText(state_text)
