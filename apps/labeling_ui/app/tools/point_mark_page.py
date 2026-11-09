"""图片点位标注与坐标导出页面。

读入一张图片，滚轮缩放，左键按顺序标点，最后把有序坐标导出成一个二层
列表。画布只负责把输入事件翻译成图片坐标，坐标顺序和导出格式由
``apps.point_marking.point_core`` 唯一决定。
"""
from __future__ import annotations

from pathlib import Path

import cv2
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QFileDialog,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from apps.data_tools.processing.image_io import read_image
from apps.point_marking.point_core import (
    Point,
    append_point,
    build_snapshot,
    clamp_point,
    describe_points,
    parse_export_text,
    remove_point,
    render_export_text,
)
from cosmos_toolbox.ui.primitives import ActionBar, set_ui_role

from ..preview_widget import cv2_to_qpixmap
from .base import BaseToolPage, make_card, make_page_header, set_primary


# The preview never exceeds this edge; marked coordinates always stay in the
# full-resolution source image.
MAX_DISPLAY_EDGE = 4096

IMAGE_FILTER = "图片文件 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp);;所有文件 (*.*)"


class PointMarkCanvas(QWidget):
    """Canvas that zooms with the wheel and reports left-clicked image points."""

    pointPicked = Signal(int, int)
    pointRemoved = Signal(int)
    cursorMoved = Signal(int, int)
    zoomChanged = Signal(float)
    imageDropped = Signal(str)

    MIN_ZOOM = 0.05
    MAX_ZOOM = 32.0

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._pixmap = QPixmap()
        self._source_size = (0, 0)
        self._display_ratio = 1.0
        self._points: list[Point] = []
        self._selected_index = -1
        self._zoom = 1.0
        self._offset = QPointF()
        self._pan_origin: QPointF | None = None
        self._pan_offset = QPointF()
        self.setMinimumSize(360, 280)
        self.setMouseTracking(True)
        self.setAcceptDrops(True)
        self.setCursor(Qt.CursorShape.CrossCursor)

    # ------------------------------------------------------------------ state
    @property
    def has_image(self) -> bool:
        return not self._pixmap.isNull()

    @property
    def source_size(self) -> tuple[int, int]:
        return self._source_size

    @property
    def zoom(self) -> float:
        return self._zoom

    def set_image(
        self,
        pixmap: QPixmap,
        *,
        source_size: tuple[int, int],
        display_ratio: float = 1.0,
    ) -> None:
        """Show a preview pixmap while keeping the source image coordinate space."""

        self._pixmap = pixmap
        self._source_size = source_size
        self._display_ratio = display_ratio if display_ratio > 0 else 1.0
        self._points.clear()
        self._selected_index = -1
        self.fit_to_view()
        self.update()

    def set_points(self, points: tuple[Point, ...]) -> None:
        self._points = list(points)
        if self._selected_index >= len(self._points):
            self._selected_index = -1
        self.update()

    def set_selected_index(self, index: int) -> None:
        self._selected_index = index
        self.update()

    def fit_to_view(self) -> None:
        if self._pixmap.isNull() or self.width() <= 0 or self.height() <= 0:
            return
        self._zoom = min(
            (self.width() - 24) / self._pixmap.width(),
            (self.height() - 24) / self._pixmap.height(),
            1.0,
        )
        self._zoom = max(self.MIN_ZOOM, self._zoom)
        self._offset = QPointF()
        self.zoomChanged.emit(self._zoom)

    def zoom_by(self, factor: float) -> None:
        """Zoom around the canvas centre."""

        if self._pixmap.isNull():
            return
        self._apply_zoom(self._zoom * factor, self.width() / 2, self.height() / 2)

    def reset_zoom(self) -> None:
        if self._pixmap.isNull():
            return
        self._zoom = 1.0
        self._offset = QPointF()
        self.zoomChanged.emit(self._zoom)
        self.update()

    # --------------------------------------------------------------- geometry
    def _image_top_left(self) -> QPointF:
        return QPointF(
            (self.width() - self._pixmap.width() * self._zoom) / 2 + self._offset.x(),
            (self.height() - self._pixmap.height() * self._zoom) / 2 + self._offset.y(),
        )

    def _screen_to_source(self, position: QPointF) -> Point | None:
        if self._pixmap.isNull():
            return None
        top_left = self._image_top_left()
        display_x = (position.x() - top_left.x()) / self._zoom
        display_y = (position.y() - top_left.y()) / self._zoom
        if not (0 <= display_x < self._pixmap.width() and 0 <= display_y < self._pixmap.height()):
            return None
        return int(round(display_x * self._display_ratio)), int(round(display_y * self._display_ratio))

    def _source_to_screen(self, point: Point) -> QPointF:
        top_left = self._image_top_left()
        display_x = point[0] / self._display_ratio
        display_y = point[1] / self._display_ratio
        return QPointF(
            top_left.x() + display_x * self._zoom,
            top_left.y() + display_y * self._zoom,
        )

    def _apply_zoom(self, target_zoom: float, anchor_x: float, anchor_y: float) -> None:
        if self._pixmap.isNull():
            return
        zoom = max(self.MIN_ZOOM, min(self.MAX_ZOOM, target_zoom))
        top_left = self._image_top_left()
        display_x = (anchor_x - top_left.x()) / self._zoom
        display_y = (anchor_y - top_left.y()) / self._zoom
        self._zoom = zoom
        self._offset = QPointF(
            (anchor_x - display_x * zoom) - (self.width() - self._pixmap.width() * zoom) / 2,
            (anchor_y - display_y * zoom) - (self.height() - self._pixmap.height() * zoom) / 2,
        )
        self.zoomChanged.emit(self._zoom)
        self.update()

    def _point_at(self, position: QPointF) -> int:
        """Return the index of the marker under the cursor, or -1."""

        best_index = -1
        best_distance = 10.0
        for index, point in enumerate(self._points):
            screen = self._source_to_screen(point)
            distance = (screen.x() - position.x()) ** 2 + (screen.y() - position.y()) ** 2
            if distance <= best_distance * best_distance:
                best_distance = distance**0.5
                best_index = index
        return best_index

    # ----------------------------------------------------------------- events
    def paintEvent(self, event) -> None:  # noqa: N802 - Qt method name
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#eef1f5"))
        if self._pixmap.isNull():
            painter.setPen(QColor("#7b8794"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "打开一张图片开始标点")
            return

        top_left = self._image_top_left()
        target = QRectF(
            top_left.x(),
            top_left.y(),
            self._pixmap.width() * self._zoom,
            self._pixmap.height() * self._zoom,
        )
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, self._zoom < 1)
        painter.drawPixmap(target, self._pixmap, QRectF(self._pixmap.rect()))
        painter.setPen(QPen(QColor("#c3cad3"), 1))
        painter.drawRect(target)

        for index, point in enumerate(self._points):
            screen = self._source_to_screen(point)
            active = index == self._selected_index
            color = QColor("#d9480f") if active else QColor("#1f6feb")
            painter.setPen(QPen(color, 2))
            painter.setBrush(QColor(color.red(), color.green(), color.blue(), 60))
            painter.drawEllipse(screen, 6, 6)
            painter.drawLine(int(screen.x()) - 10, int(screen.y()), int(screen.x()) + 10, int(screen.y()))
            painter.drawLine(int(screen.x()), int(screen.y()) - 10, int(screen.x()), int(screen.y()) + 10)
            painter.setPen(QPen(color))
            painter.drawText(QPointF(screen.x() + 9, screen.y() - 9), str(index + 1))

    def wheelEvent(self, event) -> None:  # noqa: N802 - Qt method name
        if self._pixmap.isNull():
            return
        delta = event.angleDelta().y()
        if delta == 0:
            return
        factor = 1.2 if delta > 0 else 1 / 1.2
        position = event.position()
        self._apply_zoom(self._zoom * factor, position.x(), position.y())

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt method name
        if event.button() == Qt.MouseButton.MiddleButton:
            self._pan_origin = event.position()
            self._pan_offset = QPointF(self._offset)
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return
        if event.button() == Qt.MouseButton.LeftButton:
            point = self._screen_to_source(event.position())
            if point is not None:
                self.pointPicked.emit(point[0], point[1])
            return
        if event.button() == Qt.MouseButton.RightButton:
            index = self._point_at(event.position())
            if index >= 0:
                self.pointRemoved.emit(index)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 - Qt method name
        if self._pan_origin is not None:
            self._offset = self._pan_offset + event.position() - self._pan_origin
            self.update()
            return
        point = self._screen_to_source(event.position())
        if point is not None:
            self.cursorMoved.emit(point[0], point[1])

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 - Qt method name
        if event.button() == Qt.MouseButton.MiddleButton:
            self._pan_origin = None
            self.setCursor(Qt.CursorShape.CrossCursor)

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt method name
        super().resizeEvent(event)
        if self._pixmap.isNull():
            return
        if self._zoom <= 1.0 and self._offset.isNull():
            self.fit_to_view()

    # ------------------------------------------------------------------ drops
    def dragEnterEvent(self, event) -> None:  # noqa: N802 - Qt method name
        if self._local_image_path(event) is not None:
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:  # noqa: N802 - Qt method name
        path = self._local_image_path(event)
        if path is None:
            return
        event.acceptProposedAction()
        self.imageDropped.emit(path)

    @staticmethod
    def _local_image_path(event) -> str | None:
        mime = event.mimeData()
        if not mime.hasUrls():
            return None
        for url in mime.urls():
            if not url.isLocalFile():
                continue
            path = Path(url.toLocalFile())
            if path.is_file() and path.suffix.lower() in {
                ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp",
            }:
                return str(path)
        return None


class PointMarkPage(BaseToolPage):
    """Mark ordered points on an image and export the two-level coordinate list."""

    tool_key = "point_mark"
    tool_title = "点位标注导出"
    tool_nav_title = "点位标注"
    tool_icon = "✛"
    tool_summary = "读取图片，滚轮缩放，左键按顺序标点，导出二维坐标列表。"
    tool_tags = ("标点", "坐标", "导出", "JSON")

    def __init__(self, main_window, parent=None):
        super().__init__(main_window, parent)
        self._points: tuple[Point, ...] = ()
        self._history: list[tuple[Point, ...]] = []
        self._image_path: Path | None = None
        self._project_image_dir: Path | None = None
        self._selected_index = -1
        self._build_ui()

    def configure_paths(self, image_dir: str = "") -> None:
        """Use the project image directory as the default browsing location."""

        if image_dir:
            self._project_image_dir = Path(image_dir)

    # -------------------------------------------------------------------- UI
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(8)

        layout.addWidget(
            make_page_header(
                "点位标注导出",
                "打开一张图片，滚轮缩放后左键标点；点按标注顺序保存，导出为一个二层坐标列表。",
            )
        )

        self._open_button = QPushButton("打开图片")
        self._fit_button = QPushButton("适配窗口")
        self._zoom_in_button = QPushButton("放大")
        self._zoom_out_button = QPushButton("缩小")
        self._reset_zoom_button = QPushButton("原始比例")
        self._undo_button = QPushButton("撤销")
        self._delete_button = QPushButton("删除选中")
        self._clear_button = QPushButton("清空")
        self._import_button = QPushButton("导入坐标")
        self._copy_button = QPushButton("复制")
        self._export_button = set_primary(QPushButton("导出坐标 JSON"))

        action_bar = ActionBar(
            [
                self._open_button,
                self._fit_button,
                self._zoom_in_button,
                self._zoom_out_button,
                self._reset_zoom_button,
                self._undo_button,
                self._delete_button,
                self._clear_button,
                self._import_button,
                self._copy_button,
                self._export_button,
            ]
        )
        layout.addWidget(action_bar)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(6)

        canvas_card = make_card()
        canvas_layout = QVBoxLayout(canvas_card)
        canvas_layout.setContentsMargins(10, 10, 10, 10)
        canvas_layout.setSpacing(6)
        self._canvas = PointMarkCanvas()
        canvas_layout.addWidget(self._canvas, 1)
        hint = QLabel("左键标点 · 滚轮缩放 · 中键拖动平移 · 右键删除光标下的点")
        hint.setWordWrap(True)
        set_ui_role(hint, "muted")
        canvas_layout.addWidget(hint)
        self._cursor_label = QLabel("")
        set_ui_role(self._cursor_label, "muted")
        canvas_layout.addWidget(self._cursor_label)
        splitter.addWidget(canvas_card)

        splitter.addWidget(self._build_side_panel())
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        layout.addWidget(splitter, 1)

        self._status = QLabel("就绪")
        set_ui_role(self._status, "muted")
        layout.addWidget(self._status)

        self._connect_actions()
        self._refresh()

    def _build_side_panel(self) -> QWidget:
        panel = make_card()
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(12, 12, 12, 12)
        panel_layout.setSpacing(8)

        self._info_label = QLabel("未打开图片")
        self._info_label.setWordWrap(True)
        set_ui_role(self._info_label, "sectionTitle")
        panel_layout.addWidget(self._info_label)

        self._count_label = QLabel("")
        set_ui_role(self._count_label, "muted")
        panel_layout.addWidget(self._count_label)

        self._table = QTableWidget(0, 3)
        self._table.setHorizontalHeaderLabels(["序号", "X", "Y"])
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.verticalHeader().setVisible(False)
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        panel_layout.addWidget(self._table, 1)

        preview_title = QLabel("导出内容（二层列表）")
        set_ui_role(preview_title, "sectionTitle")
        panel_layout.addWidget(preview_title)

        self._preview = QPlainTextEdit()
        self._preview.setReadOnly(True)
        self._preview.setProperty("uiRole", "logViewer")
        self._preview.setMinimumHeight(140)
        panel_layout.addWidget(self._preview)

        return panel

    def _connect_actions(self) -> None:
        self._open_button.clicked.connect(self._open_image)
        self._fit_button.clicked.connect(self._canvas.fit_to_view)
        self._zoom_in_button.clicked.connect(lambda: self._canvas.zoom_by(1.2))
        self._zoom_out_button.clicked.connect(lambda: self._canvas.zoom_by(1 / 1.2))
        self._reset_zoom_button.clicked.connect(self._canvas.reset_zoom)
        self._undo_button.clicked.connect(self._undo)
        self._delete_button.clicked.connect(self._delete_selected)
        self._clear_button.clicked.connect(self._clear_points)
        self._import_button.clicked.connect(self._import_points)
        self._copy_button.clicked.connect(self._copy_export_text)
        self._export_button.clicked.connect(self._export_points)

        self._canvas.pointPicked.connect(self._add_point)
        self._canvas.pointRemoved.connect(self._remove_point_at)
        self._canvas.imageDropped.connect(self._load_image)
        self._canvas.cursorMoved.connect(self._show_cursor_position)
        self._canvas.zoomChanged.connect(self._show_zoom)
        self._table.itemSelectionChanged.connect(self._on_selection_changed)

    # ------------------------------------------------------------------ image
    def _open_image(self) -> None:
        start_dir = ""
        if self._image_path is not None:
            start_dir = str(self._image_path.parent)
        elif self._project_image_dir is not None:
            start_dir = str(self._project_image_dir)
        path, _ = QFileDialog.getOpenFileName(self, "选择图片", start_dir, IMAGE_FILTER)
        if path:
            self._load_image(path)

    def _load_image(self, path: str) -> None:
        image = read_image(path, flags=cv2.IMREAD_COLOR)
        if image is None:
            self._status.setText(f"无法读取图片：{path}")
            return
        height, width = image.shape[:2]
        display_ratio = 1.0
        display = image
        longest = max(width, height)
        if longest > MAX_DISPLAY_EDGE:
            display_ratio = longest / MAX_DISPLAY_EDGE
            display = cv2.resize(
                image,
                (max(1, int(width / display_ratio)), max(1, int(height / display_ratio))),
                interpolation=cv2.INTER_AREA,
            )
        pixmap = cv2_to_qpixmap(display)
        if pixmap.isNull():
            self._status.setText(f"无法渲染图片：{path}")
            return
        self._image_path = Path(path)
        self._points = ()
        self._history.clear()
        self._selected_index = -1
        self._canvas.set_image(pixmap, source_size=(width, height), display_ratio=display_ratio)
        self._refresh()
        self._status.setText(f"已打开 {self._image_path.name}（{width}×{height}）")

    # ----------------------------------------------------------------- points
    def _add_point(self, x: int, y: int) -> None:
        if not self._canvas.has_image:
            return
        width, height = self._canvas.source_size
        point = clamp_point((int(x), int(y)), width, height)
        self._push_history()
        self._points = append_point(self._points, point[0], point[1])
        self._selected_index = len(self._points) - 1
        self._refresh()

    def _remove_point_at(self, index: int) -> None:
        if not 0 <= index < len(self._points):
            return
        self._push_history()
        self._points = remove_point(self._points, index)
        self._selected_index = -1
        self._refresh()

    def _delete_selected(self) -> None:
        if self._selected_index < 0:
            self._status.setText("先在列表或画布中选中一个点")
            return
        self._remove_point_at(self._selected_index)

    def _clear_points(self) -> None:
        if not self._points:
            return
        self._push_history()
        self._points = ()
        self._selected_index = -1
        self._refresh()
        self._status.setText("已清空所有标注点")

    def _undo(self) -> None:
        if not self._history:
            self._status.setText("没有可撤销的操作")
            return
        self._points = self._history.pop()
        self._selected_index = -1
        self._refresh()
        self._status.setText("已撤销上一次标注操作")

    def _push_history(self) -> None:
        self._history.append(self._points)

    def _on_selection_changed(self) -> None:
        rows = self._table.selectionModel().selectedRows() if self._table.selectionModel() else []
        index = rows[0].row() if rows else -1
        self._selected_index = index
        self._canvas.set_selected_index(index)

    # ----------------------------------------------------------------- export
    def _export_points(self) -> None:
        if not self._points:
            QMessageBox.information(self, "导出坐标", "还没有标注任何点。")
            return
        default_name = "points.json"
        start_dir = ""
        if self._image_path is not None:
            start_dir = str(self._image_path.parent)
            default_name = f"{self._image_path.stem}.points.json"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "导出坐标",
            str(Path(start_dir) / default_name) if start_dir else default_name,
            "JSON 文件 (*.json);;所有文件 (*.*)",
        )
        if not path:
            return
        try:
            Path(path).write_text(self._export_text() + "\n", encoding="utf-8")
        except OSError as exc:
            QMessageBox.warning(self, "导出坐标", f"写入失败：{exc}")
            self._status.setText(f"导出失败：{exc}")
            return
        self._status.setText(f"已导出 {len(self._points)} 个点：{path}")

    def _import_points(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "导入坐标", "", "JSON 文件 (*.json);;所有文件 (*.*)")
        if not path:
            return
        try:
            text = Path(path).read_text(encoding="utf-8")
        except OSError as exc:
            QMessageBox.warning(self, "导入坐标", f"读取失败：{exc}")
            return
        points, error = parse_export_text(text)
        if error:
            QMessageBox.warning(self, "导入坐标", error)
            self._status.setText(f"导入失败：{error}")
            return
        self._push_history()
        self._points = points
        self._selected_index = -1
        self._refresh()
        self._status.setText(f"已导入 {len(self._points)} 个点：{path}")

    def _copy_export_text(self) -> None:
        if not self._points:
            QMessageBox.information(self, "复制", "还没有标注任何点。")
            return
        QApplication.clipboard().setText(self._export_text())
        self._status.setText("已复制到剪贴板")

    def _export_text(self) -> str:
        return render_export_text(build_snapshot(self._points).payload)

    # ---------------------------------------------------------------- refresh
    def _refresh(self) -> None:
        self._canvas.set_points(self._points)
        self._table.setRowCount(len(self._points))
        for row, (x, y) in enumerate(self._points):
            for column, value in enumerate((row + 1, x, y)):
                item = QTableWidgetItem(str(value))
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self._table.setItem(row, column, item)
        self._preview.setPlainText(self._export_text())
        self._count_label.setText(describe_points(self._points))
        self._undo_button.setEnabled(bool(self._history))
        self._delete_button.setEnabled(bool(self._points))
        self._clear_button.setEnabled(bool(self._points))
        self._copy_button.setEnabled(bool(self._points))
        self._export_button.setEnabled(bool(self._points))
        if self._image_path is not None:
            width, height = self._canvas.source_size
            self._info_label.setText(f"{self._image_path.name}\n{width} × {height}")
        else:
            self._info_label.setText("未打开图片")

    def _show_cursor_position(self, x: int, y: int) -> None:
        self._cursor_label.setText(f"光标位置 ({x}, {y})")

    def _show_zoom(self, zoom: float) -> None:
        if self._image_path is None:
            return
        width, height = self._canvas.source_size
        self._status.setText(f"{self._image_path.name} · {width}×{height} · 缩放 {zoom * 100:.0f}%")

    # --------------------------------------------------------------- teardown
    def shutdown(self) -> None:
        super().shutdown()
        self._points = ()
        self._history.clear()
        self._image_path = None
