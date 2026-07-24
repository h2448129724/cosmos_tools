"""批量裁剪工具页面。"""

from __future__ import annotations

import os

from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)
from PySide6.QtCore import Qt

from apps.data_tools.processing.image_io import read_image
from apps.data_tools.processing.batch_crop import batch_crop, crop_single_image
from ..preview_widget import ZoomableLabel
from .base import BaseToolPage, FuncWorker, make_card, make_log_box, make_log_card, make_page_header, set_primary


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

        settings_card = make_card()
        settings_lay = QVBoxLayout(settings_card)
        settings_lay.setContentsMargins(18, 16, 18, 16)
        settings_lay.setSpacing(12)

        in_row = QHBoxLayout()
        in_row.setSpacing(10)
        in_label = QLabel("图片目录")
        in_label.setFixedWidth(84)
        in_row.addWidget(in_label)
        self._in_entry = QLineEdit()
        in_row.addWidget(self._in_entry, 1)
        b_in = QPushButton("浏览")
        b_in.setFixedWidth(72)
        b_in.clicked.connect(self._pick_input)
        in_row.addWidget(b_in)
        b_load = QPushButton("加载目录")
        b_load.setFixedWidth(92)
        b_load.clicked.connect(self._load_first)
        set_primary(b_load)
        in_row.addWidget(b_load)
        settings_lay.addLayout(in_row)

        out_row = QHBoxLayout()
        out_row.setSpacing(10)
        out_label = QLabel("输出目录")
        out_label.setFixedWidth(84)
        out_row.addWidget(out_label)
        self._out_entry = QLineEdit()
        out_row.addWidget(self._out_entry, 1)
        b_out = QPushButton("浏览")
        b_out.setFixedWidth(72)
        b_out.clicked.connect(lambda: self._pick_dir(self._out_entry))
        out_row.addWidget(b_out)
        settings_lay.addLayout(out_row)
        lay.addWidget(settings_card)

        main_row = QHBoxLayout()
        main_row.setSpacing(12)

        # left: preview-first workspace
        left = make_card()
        left_lay = QVBoxLayout(left)
        left_lay.setContentsMargins(18, 18, 18, 18)
        left_lay.setSpacing(10)

        info_row = QHBoxLayout()
        self._current_name = QLabel("当前图片：未加载")
        self._current_name.setStyleSheet("color:#0f172a;font-size:15px;font-weight:700;")
        self._current_name.setWordWrap(True)
        info_row.addWidget(self._current_name, 1)
        self._nav_hint = QLabel("A / D 快速切图")
        self._nav_hint.setStyleSheet("color:#64748b;")
        info_row.addWidget(self._nav_hint)
        left_lay.addLayout(info_row)

        self._preview_meta = QLabel("加载图片目录后可开始框选 ROI。")
        self._preview_meta.setStyleSheet("color:#64748b;")
        left_lay.addWidget(self._preview_meta)

        self._preview = ZoomableLabel()
        self._preview.setMinimumHeight(420)
        self._preview.rectSelected.connect(self._on_rect_selected)
        self._preview.rectsChanged.connect(self._refresh_roi_list)
        self._preview.rectSelectionChanged.connect(self._sync_selected_roi_row)
        left_lay.addWidget(self._preview, 1)

        nav_btns = QHBoxLayout()
        nav_btns.setSpacing(8)
        b_prev = QPushButton("上一张(A)")
        b_prev.clicked.connect(lambda: self._show_offset(-1))
        b_next = QPushButton("下一张(D)")
        b_next.clicked.connect(lambda: self._show_offset(+1))
        nav_btns.addWidget(b_prev)
        nav_btns.addWidget(b_next)
        nav_btns.addStretch(1)
        left_lay.addLayout(nav_btns)

        roi_btns = QHBoxLayout()
        roi_btns.setSpacing(8)
        b_start = QPushButton("开始框选")
        set_primary(b_start)
        b_start.clicked.connect(self._start_select)
        b_undo = QPushButton("撤销上一个")
        b_undo.clicked.connect(self._undo_rect)
        b_clear = QPushButton("清除所有")
        b_clear.clicked.connect(self._clear_rects)
        roi_btns.addWidget(b_start)
        roi_btns.addWidget(b_undo)
        roi_btns.addWidget(b_clear)
        roi_btns.addStretch(1)
        left_lay.addLayout(roi_btns)

        self._roi_list = QListWidget()
        self._roi_list.setMaximumHeight(96)
        self._roi_list.currentRowChanged.connect(self._preview.set_selected_rect_index)
        left_lay.addWidget(self._roi_list)

        main_row.addWidget(left, 1)

        # right: status + actions + list
        right = make_card()
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(16, 16, 16, 16)
        right_lay.setSpacing(10)
        right.setFixedWidth(320)

        status_title = QLabel("当前任务")
        status_title.setStyleSheet("color:#111827;font-size:15px;font-weight:700;")
        right_lay.addWidget(status_title)

        self._task_current = QLabel("当前图片：0 / 0")
        self._task_size = QLabel("图片尺寸：-")
        self._task_roi = QLabel("ROI 数量：0")
        self._task_state = QLabel("状态：待加载目录")
        for label in (self._task_current, self._task_size, self._task_roi, self._task_state):
            label.setStyleSheet("color:#475569;")
            right_lay.addWidget(label)

        self._summary = QLabel("等待加载图片目录。")
        self._summary.setWordWrap(True)
        self._summary.setStyleSheet("color:#64748b;")
        right_lay.addWidget(self._summary)

        action_row = QHBoxLayout()
        action_row.setSpacing(8)
        b_single = QPushButton("裁剪当前图片")
        b_single.clicked.connect(self._run_single)
        action_row.addWidget(b_single)
        right_lay.addLayout(action_row)

        b_run = QPushButton("执行批量裁剪")
        set_primary(b_run)
        b_run.clicked.connect(self._run)
        right_lay.addWidget(b_run)

        self._file_list_toggle = QPushButton("图片列表 0/0 v")
        self._file_list_toggle.setCheckable(True)
        self._file_list_toggle.setChecked(True)
        self._file_list_toggle.clicked.connect(self._toggle_file_list)
        right_lay.addWidget(self._file_list_toggle)

        self._file_list = QListWidget()
        self._file_list.setMaximumHeight(156)
        self._file_list.currentRowChanged.connect(self._show_file_at)
        right_lay.addWidget(self._file_list)
        right_lay.addStretch(1)

        main_row.addWidget(right, 0)
        lay.addLayout(main_row, 1)

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
        self._preview.rect_select_mode = True

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
        self._worker = FuncWorker(batch_crop, input_dir, rects, self._ref_w, self._ref_h, output_dir)
        self._worker.finished.connect(self._on_done)
        self._worker.start()

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
