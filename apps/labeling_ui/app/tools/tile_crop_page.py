"""自动裁剪工具页面。"""

from __future__ import annotations

import os

import cv2
from PySide6.QtWidgets import (
    QFileDialog,
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from apps.data_tools.processing.image_io import read_image
from apps.data_tools.processing.auto_tile_crop import compute_tile_grid, batch_tile_crop
from ..preview_widget import ZoomableLabel, cv2_to_qpixmap
from .base import BaseToolPage, FuncWorker, make_card, make_log_box, make_log_card, make_page_header, set_primary


class AutoTileCropPage(BaseToolPage):
    tool_key = "auto_tile_crop"
    tool_title = "自动裁剪"
    tool_nav_title = "自动裁剪"
    tool_icon = "◈"
    tool_summary = "按固定尺寸自动切块，适合训练样本铺切和大图拆分。"
    tool_tags = ("自动切块", "训练准备", "大图拆分")

    def __init__(self, main_window, parent=None):
        super().__init__(main_window, parent)
        self._build_ui()

    def _build_ui(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 16, 16, 16)
        lay.setSpacing(10)

        lay.addWidget(make_page_header("自动裁剪", "按固定尺寸自动切割成多个小块。"))

        settings_card = make_card()
        settings_lay = QVBoxLayout(settings_card)
        settings_lay.setContentsMargins(18, 16, 18, 16)
        settings_lay.setSpacing(10)

        self._in_entry, r1 = self._dir_row("输入目录")
        self._out_entry, r2 = self._dir_row("输出目录")
        settings_lay.addLayout(r1)
        settings_lay.addLayout(r2)

        size_row = QHBoxLayout()
        size_row.setSpacing(8)
        size_row.addWidget(QLabel("宽度"))
        self._spin_w = QSpinBox()
        self._spin_w.setRange(16, 8192)
        self._spin_w.setValue(256)
        self._spin_w.setSuffix(" px")
        size_row.addWidget(self._spin_w)
        size_row.addWidget(QLabel("高度"))
        self._spin_h = QSpinBox()
        self._spin_h.setRange(16, 8192)
        self._spin_h.setValue(256)
        self._spin_h.setSuffix(" px")
        size_row.addWidget(self._spin_h)
        self._cb_overlap = QCheckBox("允许重叠补边（保持固定切块尺寸）")
        self._cb_overlap.setChecked(True)
        size_row.addWidget(self._cb_overlap)
        size_row.addStretch(1)
        settings_lay.addLayout(size_row)
        lay.addWidget(settings_card)

        work_card = make_card()
        card_lay = QVBoxLayout(work_card)
        card_lay.setContentsMargins(18, 16, 18, 16)
        card_lay.setSpacing(10)
        preview_row = QHBoxLayout()
        preview_row.setSpacing(8)
        b_preview = QPushButton("加载预览")
        b_preview.clicked.connect(self._preview)
        b_run = QPushButton("执行裁剪")
        set_primary(b_run)
        b_run.clicked.connect(self._run)
        preview_row.addWidget(b_preview)
        preview_row.addWidget(b_run)
        preview_row.addStretch(1)
        card_lay.addLayout(preview_row)

        self._preview_label = ZoomableLabel()
        self._preview_label.setMinimumHeight(240)
        self._preview_label.setMaximumHeight(360)
        card_lay.addWidget(self._preview_label, 1)
        self._info_label = QLabel("")
        self._info_label.setStyleSheet("color:#64748b;")
        card_lay.addWidget(self._info_label)
        lay.addWidget(work_card, 1)

        self._log = make_log_box("运行日志...")
        lay.addWidget(make_log_card(self._log))

    def _dir_row(self, label: str) -> tuple:
        row = QHBoxLayout()
        row.setSpacing(10)
        lbl = QLabel(label)
        lbl.setFixedWidth(88)
        row.addWidget(lbl)
        entry = QLineEdit()
        row.addWidget(entry, 1)
        btn = QPushButton("浏览")
        btn.setFixedWidth(72)
        btn.clicked.connect(lambda: self._pick_dir(entry))
        row.addWidget(btn)
        return entry, row

    def _pick_dir(self, entry: QLineEdit):
        d = QFileDialog.getExistingDirectory(self._mw, "选择目录", entry.text() or ".")
        if d:
            entry.setText(d)

    def _preview(self):
        input_dir = self._in_entry.text().strip()
        if not input_dir or not os.path.isdir(input_dir):
            QMessageBox.warning(self._mw, "提示", "请先选择有效的输入目录")
            return
        from apps.data_tools.common.helpers import get_image_files

        files = get_image_files(input_dir)
        if not files:
            QMessageBox.warning(self._mw, "提示", "目录中没有图片文件")
            return
        img = read_image(files[0])
        if img is None:
            QMessageBox.warning(self._mw, "提示", "无法读取图片")
            return

        tile_w = self._spin_w.value()
        tile_h = self._spin_h.value()
        h, w = img.shape[:2]
        tiles = compute_tile_grid(w, h, tile_w, tile_h, allow_overlap=self._cb_overlap.isChecked())

        # draw grid on copy
        preview = img.copy()
        for x, y, tw, th, r, c in tiles:
            cv2.rectangle(preview, (x, y), (x + tw, y + th), (0, 255, 0), 2)
            cv2.putText(preview, f"{r},{c}", (x + 4, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        self._preview_label.set_pixmap(cv2_to_qpixmap(preview))
        rows = tiles[-1][4] + 1 if tiles else 0
        cols = max((t[5] for t in tiles), default=0) + 1 if tiles else 0
        self._info_label.setText(
            f"{os.path.basename(files[0])}: {w}×{h} → 预计产出 {len(tiles)} 块 ({rows}行 × {cols}列)"
        )

    def _run(self):
        input_dir = self._in_entry.text().strip()
        output_dir = self._out_entry.text().strip()
        tile_w = self._spin_w.value()
        tile_h = self._spin_h.value()
        if not input_dir or not os.path.isdir(input_dir):
            QMessageBox.warning(self._mw, "提示", "请先选择有效的输入目录")
            return
        if not output_dir:
            QMessageBox.warning(self._mw, "提示", "请先选择输出目录")
            return
        self._log.clear()
        overlap_text = "重叠补边" if self._cb_overlap.isChecked() else "边缘保留原始尺寸"
        self._log.appendPlainText(f"开始裁剪: {tile_w}×{tile_h}，模式：{overlap_text}")
        self._info_label.setText(f"准备执行：切块尺寸 {tile_w} x {tile_h}，模式：{overlap_text}")
        self._worker = FuncWorker(batch_tile_crop, input_dir, output_dir, tile_w, tile_h, self._cb_overlap.isChecked())
        self._worker.finished.connect(self._on_done)
        self._worker.start()

    def _on_done(self, result):
        if isinstance(result, Exception):
            self._log.appendPlainText(f"错误: {result}")
            self._info_label.setText("裁剪失败，请检查输入目录、输出目录和切块尺寸。")
        else:
            self._log.appendPlainText(f"完成！共生成 {result} 块")
            self._info_label.setText(f"裁剪完成：累计生成 {result} 个切块。")
        self._mw.show_status("自动裁剪完成")
