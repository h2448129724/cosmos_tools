"""批量裁剪工具页面。"""

from __future__ import annotations

import os

from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QSizePolicy,
    QVBoxLayout,
)
from PySide6.QtCore import Qt

from apps.data_tools.processing.image_io import read_image
from apps.data_tools.processing.batch_crop import batch_crop, crop_single_image
from ..preview_widget import ZoomableLabel
from cosmos_toolbox.ui.primitives import ActionBar, PathField, SectionSurface, set_ui_role
from .base import BaseToolPage, make_log_box, make_log_card, make_page_header, set_primary


class BatchCropPage(BaseToolPage):
    tool_key = "batch_crop"
    tool_title = "批量裁剪"
    tool_nav_title = "批量裁剪"
    tool_icon = "◈"
    tool_summary = "在参考图上框选一个或多个 ROI，并按比例批量裁剪整批图片。"
    tool_tags = ("ROI 采集", "批量输出", "比例映射")

    def __init__(self, main_window, parent=None):
        super().__init__(main_window, parent)
        self._ref_w = 0
        self._ref_h = 0
        self._files: list[str] = []
        self._current_file_index = -1
        self._build_ui()

    def _build_ui(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 16, 16, 16)
        lay.setSpacing(10)

        lay.addWidget(make_page_header("批量裁剪", "框选 ROI 后验证单张，再执行整批输出。"))

        settings_card = SectionSurface("输入与输出", "选择图片目录和输出位置，然后加载图片。")
        settings_lay = settings_card.body_layout
        input_field = PathField("图片目录", placeholder="选择包含图片的目录")
        input_field.browse_requested.connect(self._pick_input)
        self._in_entry = input_field.line_edit
        output_field = PathField("输出目录", placeholder="可选；留空则使用输入目录")
        output_field.browse_requested.connect(lambda: self._pick_dir(self._out_entry))
        self._out_entry = output_field.line_edit
        settings_lay.addWidget(input_field)
        settings_lay.addWidget(output_field)
        settings_actions = ActionBar()
        b_load = QPushButton("加载目录")
        b_load.setAccessibleName("加载图片目录")
        b_load.clicked.connect(self._load_first)
        set_primary(b_load)
        settings_actions.add_widget(b_load)
        settings_lay.addWidget(settings_actions)
        lay.addWidget(settings_card)

        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_splitter.setChildrenCollapsible(False)
        main_splitter.setHandleWidth(3)

        # left: preview-first workspace
        left = SectionSurface("预览与 ROI", "在预览中框选 ROI；已有 ROI 可在框内拖动。")
        left_lay = left.body_layout
        # Keep the canvas an even width at the desktop breakpoint so image
        # coordinate rounding remains stable for ROI placement.
        left_lay.setContentsMargins(17, 18, 18, 18)
        left_lay.setSpacing(10)

        self._current_name = QLabel("当前图片：未加载")
        set_ui_role(self._current_name, "sectionTitle")
        self._current_name.setAccessibleName("当前图片")
        self._current_name.setWordWrap(True)
        self._nav_hint = QLabel("A / D 快速切图")
        set_ui_role(self._nav_hint, "muted")
        left_lay.addWidget(self._current_name)
        left_lay.addWidget(self._nav_hint)

        self._preview_meta = QLabel("加载图片目录后可开始框选 ROI。")
        set_ui_role(self._preview_meta, "muted")
        self._preview_meta.setAccessibleName("预览信息")
        left_lay.addWidget(self._preview_meta)

        self._preview = ZoomableLabel()
        self._preview.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._preview.setAccessibleName("图片预览与 ROI 画布")
        self._preview.rectSelected.connect(self._on_rect_selected)
        self._preview.rectsChanged.connect(self._refresh_roi_list)
        self._preview.rectSelectionChanged.connect(self._sync_selected_roi_row)
        left_lay.addWidget(self._preview, 1)

        nav_btns = ActionBar()
        b_prev = QPushButton("上一张(A)")
        b_prev.setAccessibleName("上一张图片")
        b_prev.clicked.connect(lambda: self._show_offset(-1))
        b_next = QPushButton("下一张(D)")
        b_next.setAccessibleName("下一张图片")
        b_next.clicked.connect(lambda: self._show_offset(+1))
        nav_btns.add_widget(b_prev)
        nav_btns.add_widget(b_next)
        left_lay.addWidget(nav_btns)

        roi_btns = ActionBar()
        b_start = QPushButton("开始框选")
        b_start.setAccessibleName("开始框选 ROI")
        set_primary(b_start)
        b_start.clicked.connect(self._start_select)
        b_undo = QPushButton("撤销上一个")
        b_undo.setAccessibleName("撤销上一个 ROI")
        b_undo.clicked.connect(self._undo_rect)
        b_clear = QPushButton("清除所有")
        b_clear.setAccessibleName("清除所有 ROI")
        b_clear.clicked.connect(self._clear_rects)
        roi_btns.add_widget(b_start)
        roi_btns.add_widget(b_undo)
        roi_btns.add_widget(b_clear)
        left_lay.addWidget(roi_btns)

        fixed_size_row = ActionBar()
        self._fixed_size_check = QCheckBox("固定尺寸")
        self._fixed_size_check.setAccessibleName("固定 ROI 尺寸")
        self._fixed_size_check.setToolTip("开启后，新建 ROI 始终使用指定宽高；已有 ROI 仍可在框内拖动。")
        self._fixed_width = QSpinBox()
        self._fixed_height = QSpinBox()
        for box in (self._fixed_width, self._fixed_height):
            box.setRange(1, 1_000_000)
            box.setValue(256)
        self._fixed_width.setAccessibleName("固定 ROI 宽度")
        self._fixed_height.setAccessibleName("固定 ROI 高度")
        fixed_size_row.add_widget(self._fixed_size_check)
        width_label = QLabel("宽")
        height_label = QLabel("高")
        width_label.setAccessibleName("固定 ROI 宽度")
        height_label.setAccessibleName("固定 ROI 高度")
        fixed_size_row.add_widget(width_label)
        fixed_size_row.add_widget(self._fixed_width)
        fixed_size_row.add_widget(height_label)
        fixed_size_row.add_widget(self._fixed_height)
        move_hint = QLabel("框内拖动可移动 ROI")
        set_ui_role(move_hint, "muted")
        fixed_size_row.add_widget(move_hint)
        left_lay.addWidget(fixed_size_row)
        self._fixed_size_check.toggled.connect(self._sync_fixed_size)
        self._fixed_width.valueChanged.connect(self._sync_fixed_size)
        self._fixed_height.valueChanged.connect(self._sync_fixed_size)

        self._roi_list = QListWidget()
        self._roi_list.setAccessibleName("ROI 列表")
        self._roi_list.setMaximumHeight(96)
        self._roi_list.currentRowChanged.connect(self._preview.set_selected_rect_index)
        left_lay.addWidget(self._roi_list)

        main_splitter.addWidget(left)

        # right: status + actions + list
        right = SectionSurface("当前任务", "任务状态、文件列表和执行操作。")
        right_lay = right.body_layout
        right_lay.setContentsMargins(16, 16, 16, 16)
        right_lay.setSpacing(10)
        right.setMinimumWidth(250)
        right.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)

        self._task_current = QLabel("当前图片：0 / 0")
        self._task_size = QLabel("图片尺寸：-")
        self._task_roi = QLabel("ROI 数量：0")
        self._task_state = QLabel("状态：待加载目录")
        for label in (self._task_current, self._task_size, self._task_roi, self._task_state):
            set_ui_role(label, "muted")
            right_lay.addWidget(label)

        self._summary = QLabel("等待加载图片目录。")
        self._summary.setWordWrap(True)
        set_ui_role(self._summary, "muted")
        self._summary.setAccessibleName("裁剪摘要")
        right_lay.addWidget(self._summary)

        action_row = ActionBar()
        b_single = QPushButton("裁剪当前图片")
        b_single.setAccessibleName("裁剪当前图片")
        b_single.clicked.connect(self._run_single)
        action_row.add_widget(b_single)
        right_lay.addWidget(action_row)

        b_run = QPushButton("执行批量裁剪")
        b_run.setAccessibleName("执行批量裁剪")
        set_primary(b_run)
        b_run.clicked.connect(self._run)
        right_lay.addWidget(b_run)

        self._file_list_toggle = QPushButton("图片列表 0/0 v")
        self._file_list_toggle.setAccessibleName("展开或收起图片列表")
        self._file_list_toggle.setCheckable(True)
        self._file_list_toggle.setChecked(True)
        self._file_list_toggle.clicked.connect(self._toggle_file_list)
        right_lay.addWidget(self._file_list_toggle)

        self._file_list = QListWidget()
        self._file_list.setAccessibleName("图片文件列表")
        self._file_list.setMaximumHeight(156)
        self._file_list.currentRowChanged.connect(self._show_file_at)
        right_lay.addWidget(self._file_list)
        right_lay.addStretch(1)

        main_splitter.addWidget(right)
        main_splitter.setStretchFactor(0, 3)
        main_splitter.setStretchFactor(1, 1)
        main_splitter.setSizes([760, 300])
        lay.addWidget(main_splitter, 1)

        self._log = make_log_box("运行日志...")
        lay.addWidget(make_log_card(self._log))
        self._update_task_panel()

    def _pick_dir(self, entry: QLineEdit):
        d = QFileDialog.getExistingDirectory(self._mw, "选择目录", entry.text() or ".")
        if d:
            entry.setText(d)

    def _pick_input(self):
        self._pick_dir(self._in_entry)

    def _load_first(self):
        input_dir = self._in_entry.text().strip()
        if not input_dir or not os.path.isdir(input_dir):
            QMessageBox.warning(self._mw, "提示", "请先选择有效的图片目录")
            return
        from apps.data_tools.common.helpers import get_image_files

        self._files = get_image_files(input_dir)
        if not self._files:
            QMessageBox.warning(self._mw, "提示", "目录中没有图片文件")
            return
        self._file_list.blockSignals(True)
        self._file_list.clear()
        for path in self._files:
            self._file_list.addItem(os.path.basename(path))
        self._file_list.blockSignals(False)
        self._file_list_toggle.setText(f"图片列表 1/{len(self._files)} v")
        self._show_file_at(0)

    def _show_file_at(self, index: int):
        if not self._files:
            return
        if index < 0 or index >= len(self._files):
            return
        path = self._files[index]
        img = read_image(path)
        if img is None:
            QMessageBox.warning(self._mw, "提示", f"无法读取图片: {os.path.basename(path)}")
            return
        self._current_file_index = index
        self._ref_h, self._ref_w = img.shape[:2]
        self._fixed_width.setMaximum(max(1, self._ref_w))
        self._fixed_height.setMaximum(max(1, self._ref_h))
        from ..preview_widget import cv2_to_qpixmap

        self._preview.set_pixmap(cv2_to_qpixmap(img), preserve_rects=True)
        self._file_list.blockSignals(True)
        self._file_list.setCurrentRow(index)
        self._file_list.blockSignals(False)
        self._current_name.setText(f"当前图片：{os.path.basename(path)}")
        self._preview_meta.setText(
            f"当前第 {index + 1} / {len(self._files)} 张，参考尺寸 {self._ref_w} × {self._ref_h}。"
        )
        self._file_list_toggle.setText(
            f"图片列表 {index + 1}/{len(self._files)} {'v' if self._file_list.isVisible() else '>'}"
        )
        self._log.appendPlainText(f"已加载: {os.path.basename(path)} ({self._ref_w}x{self._ref_h})")
        self._summary.setText(
            f"当前 {index + 1}/{len(self._files)}：{self._ref_w} x {self._ref_h}。"
            f" 已记录 {self._roi_list.count()} 个 ROI，可单张验证或批量裁剪。"
        )
        self._update_task_panel()

    def _show_offset(self, delta: int):
        if not self._files:
            return
        next_index = 0 if self._current_file_index < 0 else self._current_file_index + delta
        next_index = max(0, min(len(self._files) - 1, next_index))
        if next_index != self._current_file_index:
            self._show_file_at(next_index)

    def _start_select(self):
        self._sync_fixed_size()
        self._preview.rect_select_mode = True

    def _sync_fixed_size(self, *_args):
        size = (
            (self._fixed_width.value(), self._fixed_height.value())
            if self._fixed_size_check.isChecked()
            else None
        )
        self._preview.set_fixed_rect_size(size)

    def _on_rect_selected(self, x1, y1, x2, y2):
        self._refresh_roi_list()
        width = max(0, x2 - x1)
        height = max(0, y2 - y1)
        if self._files and self._current_file_index >= 0:
            self._summary.setText(
                f"当前 {self._current_file_index + 1}/{len(self._files)}，已记录 {self._roi_list.count()} 个 ROI。"
                f" 最新尺寸：{width} x {height}。可继续追加、单张验证或直接批量裁剪。"
            )
        else:
            self._summary.setText(
                f"已记录 {self._roi_list.count()} 个 ROI。最新尺寸：{width} x {height}。可继续追加或直接执行裁剪。"
            )
        self._update_task_panel()

    def _refresh_roi_list(self):
        rects = self._preview.get_roi_rects()
        selected_row = self._roi_list.currentRow()
        self._roi_list.blockSignals(True)
        self._roi_list.clear()
        for idx, (x1, y1, x2, y2) in enumerate(rects, start=1):
            width = max(0, x2 - x1)
            height = max(0, y2 - y1)
            self._roi_list.addItem(f"roi_{idx}: ({x1}, {y1}) -> ({x2}, {y2})  |  {width} x {height}")
        if rects:
            target_row = min(max(selected_row, 0), len(rects) - 1)
            self._roi_list.setCurrentRow(target_row)
        else:
            self._roi_list.setCurrentRow(-1)
        self._roi_list.blockSignals(False)
        self._update_task_panel()

    def _sync_selected_roi_row(self, index: int):
        self._roi_list.blockSignals(True)
        self._roi_list.setCurrentRow(index if index >= 0 else -1)
        self._roi_list.blockSignals(False)

    def _undo_rect(self):
        if self._roi_list.count() > 0:
            self._preview.remove_last_rect()

    def _clear_rects(self):
        self._preview.clear_all_rects()
        self._summary.setText("ROI 已清空，请重新框选。")
        self._update_task_panel()

    def _get_rects(self) -> list[tuple[int, int, int, int]]:
        return self._preview.get_roi_rects()

    def _run(self):
        if self.worker_running():
            return
        input_dir = self._in_entry.text().strip()
        output_dir = self._out_entry.text().strip()
        rects = self._get_rects()
        if not input_dir or not os.path.isdir(input_dir):
            QMessageBox.warning(self._mw, "提示", "请先选择有效的图片目录")
            return
        if not output_dir:
            QMessageBox.warning(self._mw, "提示", "请先选择输出目录")
            return
        if not rects:
            QMessageBox.warning(self._mw, "提示", "请先框选至少一个ROI区域")
            return
        if self._ref_w == 0 or self._ref_h == 0:
            QMessageBox.warning(self._mw, "提示", "请先加载首图")
            return
        self._log.clear()
        self._log.appendPlainText(f"裁剪 {len(rects)} 个ROI...")
        self._summary.setText(f"开始处理：{len(rects)} 个 ROI，将按参考尺寸 {self._ref_w} x {self._ref_h} 映射。")
        self._update_task_panel("状态：批量处理中")
        self.run_background(
            batch_crop,
            input_dir,
            rects,
            self._ref_w,
            self._ref_h,
            output_dir,
            on_result=self._on_done,
        )

    def _run_single(self):
        output_dir = self._out_entry.text().strip()
        rects = self._get_rects()
        if self._current_file_index < 0 or not self._files:
            QMessageBox.warning(self._mw, "提示", "请先加载图片目录并选择当前图片")
            return
        if not output_dir:
            QMessageBox.warning(self._mw, "提示", "请先选择输出目录")
            return
        if not rects:
            QMessageBox.warning(self._mw, "提示", "请先框选至少一个ROI区域")
            return
        if self._ref_w == 0 or self._ref_h == 0:
            QMessageBox.warning(self._mw, "提示", "请先加载当前图片")
            return
        current_path = self._files[self._current_file_index]
        try:
            total = crop_single_image(current_path, rects, self._ref_w, self._ref_h, output_dir)
        except Exception as exc:
            QMessageBox.critical(self._mw, "裁剪失败", str(exc))
            return
        self._log.appendPlainText(f"单张裁剪完成: {os.path.basename(current_path)} -> {total} 张")
        self._summary.setText(f"单张裁剪完成：{os.path.basename(current_path)} 共输出 {total} 张。")
        self._update_task_panel("状态：单张验证完成")
        self._mw.show_status("当前图片裁剪完成")

    def _on_done(self, result):
        if isinstance(result, Exception):
            self._log.appendPlainText(f"错误: {result}")
            self._summary.setText("裁剪失败，请检查图片目录、输出目录和 ROI 是否有效。")
            self._update_task_panel("状态：裁剪失败")
        else:
            self._log.appendPlainText(f"完成！共裁剪 {result} 张图片")
            self._summary.setText(f"裁剪完成：累计输出 {result} 张裁剪结果。")
            self._update_task_panel("状态：批量裁剪完成")
        self._mw.show_status("批量裁剪完成")

    def _toggle_file_list(self):
        visible = self._file_list_toggle.isChecked()
        self._file_list.setVisible(visible)
        prefix = f"图片列表 {max(self._current_file_index + 1, 0)}/{len(self._files)}"
        self._file_list_toggle.setText(f"{prefix} {'v' if visible else '>'}")

    def _update_task_panel(self, state_text: str | None = None):
        total = len(self._files)
        current = self._current_file_index + 1 if self._current_file_index >= 0 else 0
        self._task_current.setText(f"当前图片：{current} / {total}")
        if self._ref_w and self._ref_h:
            self._task_size.setText(f"图片尺寸：{self._ref_w} × {self._ref_h}")
        else:
            self._task_size.setText("图片尺寸：-")
        roi_count = len(self._get_rects())
        self._task_roi.setText(f"ROI 数量：{roi_count}")
        if state_text is None:
            if not total:
                state_text = "状态：待加载目录"
            elif roi_count == 0:
                state_text = "状态：可框选"
            elif self._out_entry.text().strip():
                state_text = "状态：可裁剪"
            else:
                state_text = "状态：已框选"
        self._task_state.setText(state_text)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_A:
            self._show_offset(-1)
            event.accept()
            return
        if event.key() == Qt.Key_D:
            self._show_offset(+1)
            event.accept()
            return
        super().keyPressEvent(event)
