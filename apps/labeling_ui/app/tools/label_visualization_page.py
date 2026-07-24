"""Generic annotation label visualization page."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from cabf import IMAGE_SUFFIXES, read_image_bgr

from ..annotation.adapters.cabf import CabfAnnotationAdapter
from ..annotation.canvas import AnnotationCanvas
from ..annotation.layer_panel import LayerVisibilityPanel
from .base import BaseToolPage, make_card, make_page_header, set_primary


@dataclass(frozen=True)
class VisualizationItem:
    image_path: Path
    label_path: Path | None


def collect_visualization_items(image_dir: Path, label_dir: Path | None = None) -> list[VisualizationItem]:
    label_root = label_dir or image_dir
    items: list[VisualizationItem] = []
    for path in sorted(image_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        label_path = label_root / f"{path.stem}.json"
        items.append(VisualizationItem(image_path=path, label_path=label_path if label_path.exists() else None))
    return items


class LabelVisualizationPage(BaseToolPage):
    tool_key = "label_visualization"
    tool_title = "标签可视化"
    tool_nav_title = "标签可视化"
    tool_icon = "V"
    tool_summary = "加载图片与同名标注文件，按点、线、ROI、多边形图层查看标签。"
    tool_tags = ("标签", "可视化", "图层")

    def __init__(self, main_window, parent=None):
        super().__init__(main_window, parent)
        self.items: list[VisualizationItem] = []
        self.current_index = -1
        self.adapter = CabfAnnotationAdapter()
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(18, 16, 18, 16)
        root.setSpacing(12)

        root.addWidget(make_page_header("标签可视化", "选择图片目录和标签目录，按图层查看点、线、ROI 和多边形。", "通用工具"))

        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter, 1)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(10)

        form_card = make_card()
        form_layout = QVBoxLayout(form_card)
        form_layout.setContentsMargins(14, 12, 14, 12)
        form_layout.setSpacing(8)

        self.edit_image_dir, image_row = self._path_row("图片目录")
        self.edit_label_dir, label_row = self._path_row("标签目录")
        form_layout.addLayout(image_row)
        form_layout.addLayout(label_row)

        format_row = QHBoxLayout()
        format_label = QLabel("标注格式")
        format_label.setFixedWidth(72)
        self.combo_format = QComboBox()
        self.combo_format.addItem("自动识别 / CAB-F", "cabf")
        format_row.addWidget(format_label)
        format_row.addWidget(self.combo_format, 1)
        form_layout.addLayout(format_row)

        button_row = QHBoxLayout()
        self.btn_choose_image = QPushButton("选择图片")
        self.btn_choose_label = QPushButton("选择标签")
        self.btn_load = set_primary(QPushButton("加载"))
        button_row.addWidget(self.btn_choose_image)
        button_row.addWidget(self.btn_choose_label)
        button_row.addWidget(self.btn_load)
        form_layout.addLayout(button_row)
        left_layout.addWidget(form_card)

        self.file_list = QListWidget()
        self.file_list.setMinimumWidth(260)
        left_layout.addWidget(self.file_list, 1)

        self.status_label = QLabel("请选择图片目录和标签目录。")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("color:#475569;font-size:12px;")
        left_layout.addWidget(self.status_label)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)

        layer_card = QFrame()
        layer_layout = QVBoxLayout(layer_card)
        layer_layout.setContentsMargins(0, 0, 0, 0)
        self.layer_panel = LayerVisibilityPanel()
        layer_layout.addWidget(self.layer_panel)
        right_layout.addWidget(layer_card)

        self.canvas = AnnotationCanvas()
        self.layer_panel.set_canvas(self.canvas)
        right_layout.addWidget(self.canvas, 1)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setSizes([320, 900])

        self.btn_choose_image.clicked.connect(self.choose_image_dir)
        self.btn_choose_label.clicked.connect(self.choose_label_dir)
        self.btn_load.clicked.connect(self.load_from_entries)
        self.file_list.currentRowChanged.connect(self.show_item)

    def _path_row(self, label: str) -> tuple[QLineEdit, QHBoxLayout]:
        row = QHBoxLayout()
        lbl = QLabel(label)
        lbl.setFixedWidth(72)
        edit = QLineEdit()
        row.addWidget(lbl)
        row.addWidget(edit, 1)
        return edit, row

    def choose_image_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择图片目录", self.edit_image_dir.text().strip())
        if path:
            self.edit_image_dir.setText(path)

    def choose_label_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择标签目录", self.edit_label_dir.text().strip())
        if path:
            self.edit_label_dir.setText(path)

    def load_from_entries(self) -> None:
        image_text = self.edit_image_dir.text().strip()
        label_text = self.edit_label_dir.text().strip()
        if not image_text:
            QMessageBox.warning(self, "提示", "请先选择图片目录。")
            return
        self.load_paths(Path(image_text), Path(label_text) if label_text else None)

    def load_paths(self, image_dir: Path, label_dir: Path | None = None) -> None:
        if not image_dir.exists():
            raise FileNotFoundError(f"图片目录不存在: {image_dir}")
        if label_dir is not None and not label_dir.exists():
            raise FileNotFoundError(f"标签目录不存在: {label_dir}")
        self.edit_image_dir.setText(str(image_dir))
        self.edit_label_dir.setText(str(label_dir or ""))
        self.items = collect_visualization_items(image_dir, label_dir)
        self.file_list.blockSignals(True)
        self.file_list.clear()
        for item in self.items:
            suffix = "" if item.label_path is not None else " [无标签]"
            QListWidgetItem(f"{item.image_path.name}{suffix}", self.file_list)
        self.file_list.blockSignals(False)
        if not self.items:
            self.current_index = -1
            self.status_label.setText("未找到可视化图片。")
            return
        self.file_list.setCurrentRow(0)
        self.show_item(0)

    def show_item(self, index: int) -> None:
        if index < 0 or index >= len(self.items):
            return
        self.current_index = index
        item = self.items[index]
        image = read_image_bgr(item.image_path)
        document = self.adapter.load(item.image_path, item.label_path)
        self.canvas.set_document(document)
        self.canvas.set_image(image, image_path=str(item.image_path))
        self.layer_panel.rebuild()
        points = len(document.get_layer("points").shapes) if document.get_layer("points") else 0
        edges = len(document.get_layer("edges").shapes) if document.get_layer("edges") else 0
        roi = len(document.get_layer("roi").shapes) if document.get_layer("roi") else 0
        segments = len(document.get_layer("segments").shapes) if document.get_layer("segments") else 0
        label_state = "有标签" if item.label_path is not None else "无标签"
        self.status_label.setText(
            f"{item.image_path.name} - {label_state}；点 {points} / 线 {edges} / ROI {roi} / 多边形 {segments}"
        )
