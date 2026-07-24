"""Inner-side mask creation tool."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from apps.data_tools.processing.image_io import read_image
from ..preview_widget import ZoomableLabel, cv2_to_qpixmap
from .base import BaseToolPage, make_card, make_page_header, set_primary


def _write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, data = cv2.imencode(path.suffix, image)
    if not ok:
        raise RuntimeError(f"cannot write image: {path}")
    data.tofile(str(path))


class InnerMaskPage(BaseToolPage):
    tool_key = "inner_mask"
    tool_title = "内侧 Mask 制作"
    tool_nav_title = "内侧Mask"
    tool_icon = "◒"
    tool_summary = "在图片上绘制内侧区域，导出同尺寸单通道 mask。"
    tool_tags = ("Mask", "内外侧", "多边形")

    def __init__(self, main_window, parent=None):
        super().__init__(main_window, parent)
        self._image_path: Path | None = None
        self._image: np.ndarray | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(8)

        root.addWidget(make_page_header("内侧 Mask 制作", "绘制内侧多边形并保存为白色内侧、黑色背景的 mask。"))

        controls = make_card()
        controls_lay = QVBoxLayout(controls)
        controls_lay.setContentsMargins(14, 14, 14, 14)
        controls_lay.setSpacing(10)

        image_row = QHBoxLayout()
        image_row.addWidget(QLabel("图片"))
        self._image_entry = QLineEdit("")
        image_row.addWidget(self._image_entry, 1)
        btn_pick_image = QPushButton("选择图片")
        btn_pick_image.clicked.connect(self._pick_image)
        image_row.addWidget(btn_pick_image)
        btn_load = QPushButton("加载")
        set_primary(btn_load)
        btn_load.clicked.connect(self._load_image)
        image_row.addWidget(btn_load)
        controls_lay.addLayout(image_row)

        output_row = QHBoxLayout()
        output_row.addWidget(QLabel("输出"))
        self._output_entry = QLineEdit("")
        output_row.addWidget(self._output_entry, 1)
        btn_pick_output = QPushButton("保存到")
        btn_pick_output.clicked.connect(self._pick_output)
        output_row.addWidget(btn_pick_output)
        controls_lay.addLayout(output_row)

        button_row = QHBoxLayout()
        btn_draw = QPushButton("开始绘制内侧")
        set_primary(btn_draw)
        btn_draw.clicked.connect(self._start_draw)
        button_row.addWidget(btn_draw)
        btn_undo = QPushButton("撤销上一个")
        btn_undo.clicked.connect(self._undo_polygon)
        button_row.addWidget(btn_undo)
        btn_clear = QPushButton("清空")
        btn_clear.clicked.connect(self._clear_polygons)
        button_row.addWidget(btn_clear)
        btn_save = QPushButton("保存 Mask")
        set_primary(btn_save)
        btn_save.clicked.connect(self._save_mask)
        button_row.addWidget(btn_save)
        button_row.addStretch(1)
        controls_lay.addLayout(button_row)

        self._status = QLabel("请选择图片。")
        self._status.setStyleSheet("color:#64748b;")
        controls_lay.addWidget(self._status)
        root.addWidget(controls)

        preview_card = make_card()
        preview_lay = QVBoxLayout(preview_card)
        preview_lay.setContentsMargins(12, 12, 12, 12)
        preview_lay.setSpacing(8)
        self._preview = ZoomableLabel()
        self._preview.setMinimumHeight(420)
        self._preview.polygonClosed.connect(self._on_polygon_closed)
        preview_lay.addWidget(self._preview, 1)
        hint = QLabel("左键逐点绘制，双击或点击起点闭合；可连续绘制多个内侧区域。")
        hint.setStyleSheet("color:#64748b;")
        preview_lay.addWidget(hint)
        root.addWidget(preview_card, 1)

    def _pick_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择图片",
            self._image_entry.text().strip() or ".",
            "Images (*.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff)",
        )
        if path:
            self._image_entry.setText(path)
            if not self._output_entry.text().strip():
                image_path = Path(path)
                self._output_entry.setText(str(image_path.with_name(f"{image_path.stem}_inner_mask.png")))

    def _pick_output(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "保存内侧 mask",
            self._output_entry.text().strip() or ".",
            "PNG (*.png);;Bitmap (*.bmp);;TIFF (*.tif *.tiff)",
        )
        if path:
            self._output_entry.setText(path)

    def _load_image(self) -> None:
        path = Path(self._image_entry.text().strip())
        if not path.exists():
            QMessageBox.warning(self, "提示", "图片不存在。")
            return
        try:
            image = read_image(str(path))
        except Exception as exc:
            QMessageBox.critical(self, "加载失败", str(exc))
            return

        self._image_path = path
        self._image = image
        self._preview.set_pixmap(cv2_to_qpixmap(image))
        self._preview.polygon_mode = True
        h, w = image.shape[:2]
        self._status.setText(f"已加载 {path.name}，尺寸 {w} x {h}。")

    def _start_draw(self) -> None:
        if self._image is None:
            QMessageBox.warning(self, "提示", "请先加载图片。")
            return
        if not self._preview.polygon_mode:
            self._preview.polygon_mode = True
        self._status.setText("绘制模式已开启。")

    def _undo_polygon(self) -> None:
        self._preview.remove_last_polygon()
        self._update_count()

    def _clear_polygons(self) -> None:
        self._preview.clear_polygon()
        self._update_count()

    def _on_polygon_closed(self, _points) -> None:
        self._update_count()

    def _update_count(self) -> None:
        count = len(self._preview._polygons_list)
        self._status.setText(f"当前内侧区域：{count} 个。")

    def _save_mask(self) -> None:
        if self._image is None:
            QMessageBox.warning(self, "提示", "请先加载图片。")
            return
        output_text = self._output_entry.text().strip()
        if not output_text:
            QMessageBox.warning(self, "提示", "请选择输出路径。")
            return
        output = Path(output_text)
        if not output.suffix:
            output = output.with_suffix(".png")
        polygons = self._preview._polygons_list
        if not polygons:
            QMessageBox.warning(self, "提示", "请至少绘制一个内侧区域。")
            return

        h, w = self._image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        for poly in polygons:
            pts = np.asarray(poly, dtype=np.int32).reshape(-1, 1, 2)
            if len(pts) >= 3:
                cv2.fillPoly(mask, [pts], 255)

        try:
            _write_image(output, mask)
        except Exception as exc:
            QMessageBox.critical(self, "保存失败", str(exc))
            return

        self._status.setText(f"已保存：{output}")
        if hasattr(self._mw, "show_status"):
            self._mw.show_status("内侧 mask 已保存")
