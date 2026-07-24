"""Image preview widget with zoom, pan, coordinate picker, and polygon ROI drawing."""
import json
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                                QPushButton, QTabWidget, QApplication, QFrame)
from PySide6.QtCore import Qt, Signal, QPointF, QRectF, QTimer
from PySide6.QtGui import (QPixmap, QImage, QWheelEvent, QMouseEvent, QPainter,
                           QPen, QColor, QBrush, QPolygonF)
import cv2
import numpy as np


def cv2_to_qpixmap(img: np.ndarray | None) -> QPixmap:
    """Convert OpenCV image (BGR) to QPixmap for display."""
    if img is None:
        return QPixmap()
    if len(img.shape) == 2:
        h, w = img.shape
        data = img.tobytes()
        qimg = QImage(data, w, h, w, QImage.Format_Grayscale8)
    elif img.shape[2] == 4:
        h, w, _ = img.shape
        rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        data = rgb.tobytes()
        qimg = QImage(data, w, h, w * 4, QImage.Format_RGBA8888)
    else:
        h, w, _ = img.shape
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        data = rgb.tobytes()
        qimg = QImage(data, w, h, w * 3, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


class ZoomableLabel(QWidget):
    """Label that supports zoom, pan, pixel picking, and polygon ROI drawing."""
    zoomChanged = Signal(float)
    pixelClicked = Signal(int, int)
    pixelMoved = Signal(int, int)
    polygonPointAdded = Signal(int, int)
    polygonClosed = Signal(list)
    rectSelected = Signal(int, int, int, int)  # x1, y1, x2, y2
    rectsChanged = Signal()
    rectSelectionChanged = Signal(int)

    _CLOSE_DIST = 10  # screen pixels to detect click-near-first-point

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixmap = None
        self._zoom = 1.0
        self._offset_x = 0
        self._offset_y = 0
        self._dragging = False
        self._drag_start = None
        self._base_offset = (0, 0)

        # Picker mode
        self._picker_mode = False
        self._marker_point = None

        # Polygon mode
        self._polygon_mode = False
        self._polygon_image_pts = []    # current polygon image coords (source of truth)
        self._polygon_closed = False
        self._polygons_list = []        # completed polygons: list of list of (x, y)

        # Rect select mode
        self._rect_select_mode = False
        self._rect_drawing = False
        self._rect_start_screen = None
        self._rect_start_image = None
        self._rect_current_screen = None
        self._rects_image = []   # list of (x1, y1, x2, y2) image coords — source of truth
        self._rect_move_index = -1
        self._rect_move_anchor = None
        self._rect_move_origin = None
        self._rect_resize_index = -1
        self._rect_resize_corner = ""
        self._rect_selected_index = -1
        self._rect_resize_origin = None

        self.setMinimumSize(220, 160)
        self.setMouseTracking(True)

    # ---- Picker mode ----
    @property
    def picker_mode(self):
        return self._picker_mode

    @picker_mode.setter
    def picker_mode(self, value):
        self._picker_mode = value
        if value:
            self.setCursor(Qt.CrossCursor)
            self._marker_point = None
        else:
            self.setCursor(Qt.ArrowCursor)
            self._marker_point = None
        self.update()

    # ---- Polygon mode ----
    @property
    def polygon_mode(self):
        return self._polygon_mode

    @polygon_mode.setter
    def polygon_mode(self, value):
        self._polygon_mode = value
        if value:
            self.setCursor(Qt.CrossCursor)
        else:
            self.setCursor(Qt.ArrowCursor)
        self._polygon_image_pts.clear()
        self._polygon_closed = False
        self._polygons_list.clear()
        self.update()

    def clear_polygon(self):
        self._polygon_image_pts.clear()
        self._polygon_closed = False
        self._polygons_list.clear()
        self.update()

    def remove_last_polygon(self):
        if self._polygon_image_pts:
            self._polygon_image_pts.clear()
            self._polygon_closed = False
        elif self._polygons_list:
            self._polygons_list.pop()
        self.update()

    # ---- Rect select mode ----
    @property
    def rect_select_mode(self):
        return self._rect_select_mode

    @rect_select_mode.setter
    def rect_select_mode(self, value):
        self._rect_select_mode = value
        if value:
            self.setCursor(Qt.CrossCursor)
        else:
            self.setCursor(Qt.ArrowCursor)
        self._rect_drawing = False
        self.update()

    def clear_all_rects(self):
        self._rects_image.clear()
        self._set_selected_rect_index(-1)
        self.rectsChanged.emit()
        self.update()

    def remove_last_rect(self):
        if self._rects_image:
            self._rects_image.pop()
            if self._rect_selected_index >= len(self._rects_image):
                self._set_selected_rect_index(len(self._rects_image) - 1)
            self.rectsChanged.emit()
            self.update()

    def get_roi_rects(self) -> list[tuple[int, int, int, int]]:
        """Return accumulated ROI rectangles in image coordinates."""
        return list(self._rects_image)

    def set_selected_rect_index(self, index: int) -> None:
        if index < 0 or index >= len(self._rects_image):
            self._set_selected_rect_index(-1)
        else:
            self._set_selected_rect_index(index)
        self.update()

    # ---- Geometry ----
    def _screen_to_image(self, sx, sy):
        if self._pixmap is None or self._pixmap.isNull():
            return None
        pw, ph = self.width(), self.height()
        scaled_w = self._pixmap.width() * self._zoom
        scaled_h = self._pixmap.height() * self._zoom
        img_x = int((sx - (pw - scaled_w) / 2 - self._offset_x) / self._zoom)
        img_y = int((sy - (ph - scaled_h) / 2 - self._offset_y) / self._zoom)
        iw, ih = self._pixmap.width(), self._pixmap.height()
        if 0 <= img_x < iw and 0 <= img_y < ih:
            return img_x, img_y
        return None

    def _image_to_screen(self, ix, iy):
        if self._pixmap is None or self._pixmap.isNull():
            return 0, 0
        pw, ph = self.width(), self.height()
        scaled_w = self._pixmap.width() * self._zoom
        scaled_h = self._pixmap.height() * self._zoom
        sx = ix * self._zoom + (pw - scaled_w) / 2 + self._offset_x
        sy = iy * self._zoom + (ph - scaled_h) / 2 + self._offset_y
        return sx, sy

    def _image_rect_to_screen(self, ix, iy, iw, ih):
        sx, sy = self._image_to_screen(ix, iy)
        sw = iw * self._zoom
        sh = ih * self._zoom
        return sx, sy, sw, sh

    def set_pixmap(self, pixmap, *, preserve_rects: bool = False):
        self._pixmap = pixmap
        self._zoom = 1.0
        self._offset_x = 0
        self._offset_y = 0
        self._marker_point = None
        if not preserve_rects:
            self._rects_image.clear()
            self._set_selected_rect_index(-1)
            self.rectsChanged.emit()
        self.fit_to_view()
        self.update()

    def _find_rect_at(self, ix: int, iy: int) -> int:
        for index in range(len(self._rects_image) - 1, -1, -1):
            x1, y1, x2, y2 = self._rects_image[index]
            if x1 <= ix <= x2 and y1 <= iy <= y2:
                return index
        return -1

    def _set_selected_rect_index(self, index: int) -> None:
        if self._rect_selected_index == index:
            return
        self._rect_selected_index = index
        self.rectSelectionChanged.emit(index)

    def _find_rect_corner_at(self, sx: float, sy: float) -> tuple[int, str]:
        handle_radius = 8
        corners = ("tl", "tr", "br", "bl")
        for index in range(len(self._rects_image) - 1, -1, -1):
            x1, y1, x2, y2 = self._rects_image[index]
            handle_points = {
                "tl": self._image_to_screen(x1, y1),
                "tr": self._image_to_screen(x2, y1),
                "br": self._image_to_screen(x2, y2),
                "bl": self._image_to_screen(x1, y2),
            }
            for corner in corners:
                hx, hy = handle_points[corner]
                if (sx - hx) ** 2 + (sy - hy) ** 2 <= handle_radius ** 2:
                    return index, corner
        return -1, ""

    def _clamp_rect_to_image(self, rect: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        if self._pixmap is None or self._pixmap.isNull():
            return rect
        x1, y1, x2, y2 = rect
        width = x2 - x1
        height = y2 - y1
        max_x1 = max(self._pixmap.width() - width, 0)
        max_y1 = max(self._pixmap.height() - height, 0)
        x1 = max(0, min(x1, max_x1))
        y1 = max(0, min(y1, max_y1))
        return x1, y1, x1 + width, y1 + height

    def _normalize_rect(self, rect: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = rect
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        return x1, y1, x2, y2

    def _resize_rect_to_corner(
        self,
        origin: tuple[int, int, int, int],
        corner: str,
        ix: int,
        iy: int,
    ) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = origin
        min_size = 4
        if self._pixmap is not None and not self._pixmap.isNull():
            ix = max(0, min(ix, self._pixmap.width() - 1))
            iy = max(0, min(iy, self._pixmap.height() - 1))
        if corner == "tl":
            x1, y1 = ix, iy
            x1 = min(x1, x2 - min_size)
            y1 = min(y1, y2 - min_size)
        elif corner == "tr":
            x2, y1 = ix, iy
            x2 = max(x2, x1 + min_size)
            y1 = min(y1, y2 - min_size)
        elif corner == "br":
            x2, y2 = ix, iy
            x2 = max(x2, x1 + min_size)
            y2 = max(y2, y1 + min_size)
        elif corner == "bl":
            x1, y2 = ix, iy
            x1 = min(x1, x2 - min_size)
            y2 = max(y2, y1 + min_size)
        return self._normalize_rect((x1, y1, x2, y2))

    def fit_to_view(self):
        if self._pixmap is None or self._pixmap.isNull():
            return
        pw, ph = self.width() - 20, self.height() - 20
        iw, ih = self._pixmap.width(), self._pixmap.height()
        if iw <= 0 or ih <= 0:
            return
        self._zoom = min(pw / iw, ph / ih, 1.0)
        self._offset_x = 0
        self._offset_y = 0
        self.zoomChanged.emit(self._zoom)

    # ---- Painting ----
    def paintEvent(self, event):
        super().paintEvent(event)
        if self._pixmap is None or self._pixmap.isNull():
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        pw = self.width()
        ph = self.height()
        scaled_w = self._pixmap.width() * self._zoom
        scaled_h = self._pixmap.height() * self._zoom
        x = int((pw - scaled_w) // 2 + self._offset_x)
        y = int((ph - scaled_h) // 2 + self._offset_y)
        target = QRectF(x, y, scaled_w, scaled_h)
        source = QRectF(0, 0, self._pixmap.width(), self._pixmap.height())
        painter.drawPixmap(target, self._pixmap, source)

        # Crosshair in picker mode
        if self._picker_mode and self._marker_point is not None:
            mx, my = self._marker_point
            painter.setPen(QPen(QColor(255, 0, 0), 1))
            painter.drawLine(QPointF(x, my), QPointF(x + scaled_w, my))
            painter.drawLine(QPointF(mx, y), QPointF(mx, y + scaled_h))
            painter.setPen(QPen(QColor(255, 0, 0), 3))
            painter.drawPoint(QPointF(mx, my))

        # Polygon ROI in polygon mode — draw completed polygons
        poly_colors = [
            QColor(0, 255, 0),
            QColor(0, 200, 255),
            QColor(255, 200, 0),
            QColor(200, 0, 255),
            QColor(255, 100, 100),
        ]
        if self._polygon_mode:
            for pi, poly_pts in enumerate(self._polygons_list):
                if len(poly_pts) < 3:
                    continue
                color = poly_colors[pi % len(poly_colors)]
                screen_pts = [self._image_to_screen(ix, iy) for ix, iy in poly_pts]
                painter.setPen(QPen(color, 2))
                painter.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), 30)))
                polygon = QPolygonF([QPointF(sx, sy) for sx, sy in screen_pts])
                painter.drawPolygon(polygon)
                # Point markers
                painter.setPen(QPen(QColor(255, 255, 0), 4))
                for px, py in screen_pts:
                    painter.drawPoint(QPointF(px, py))
                # Index label
                painter.setPen(QPen(color, 1))
                painter.drawText(QPointF(screen_pts[0][0] + 3, screen_pts[0][1] + 14), str(pi + 1))

            # Draw current in-progress polygon
            if len(self._polygon_image_pts) > 0:
                screen_pts = [self._image_to_screen(ix, iy) for ix, iy in self._polygon_image_pts]
                color = poly_colors[len(self._polygons_list) % len(poly_colors)]
                # Lines
                painter.setPen(QPen(color, 2))
                for i in range(len(screen_pts) - 1):
                    painter.drawLine(QPointF(*screen_pts[i]),
                                     QPointF(*screen_pts[i + 1]))
                if self._polygon_closed and len(screen_pts) >= 3:
                    painter.drawLine(QPointF(*screen_pts[-1]),
                                     QPointF(*screen_pts[0]))
                # Point markers
                painter.setPen(QPen(QColor(255, 255, 0), 6))
                for px, py in screen_pts:
                    painter.drawPoint(QPointF(px, py))
                # Highlight first point for closing
                if len(screen_pts) >= 3 and not self._polygon_closed:
                    painter.setPen(QPen(QColor(255, 0, 255), 8))
                    painter.drawPoint(QPointF(*screen_pts[0]))

        # Rect selection - draw all accumulated rects (image coords -> screen coords)
        colors = [
            (QColor(0, 120, 215), QColor(0, 120, 215, 40)),    # blue
            (QColor(255, 140, 0), QColor(255, 140, 0, 40)),     # orange
            (QColor(0, 200, 0), QColor(0, 200, 0, 40)),         # green
            (QColor(200, 0, 200), QColor(200, 0, 200, 40)),     # magenta
            (QColor(200, 200, 0), QColor(200, 200, 0, 40)),     # yellow
        ]
        for i, (x1, y1, x2, y2) in enumerate(self._rects_image):
            pen_color, brush_color = colors[i % len(colors)]
            selected = i == self._rect_selected_index
            painter.setPen(QPen(pen_color, 3 if selected else 2, Qt.SolidLine if selected else Qt.DashLine))
            painter.setBrush(QBrush(brush_color))
            rx, ry = self._image_to_screen(x1, y1)
            rw = (x2 - x1) * self._zoom
            rh = (y2 - y1) * self._zoom
            painter.drawRect(QRectF(rx, ry, rw, rh))
            painter.setPen(QPen(pen_color, 1))
            painter.drawText(QPointF(rx + 3, ry + 14), str(i + 1))
            if selected:
                handle_size = 8
                painter.setBrush(QBrush(QColor(255, 255, 255)))
                painter.setPen(QPen(pen_color, 2))
                for hx, hy in (
                    self._image_to_screen(x1, y1),
                    self._image_to_screen(x2, y1),
                    self._image_to_screen(x2, y2),
                    self._image_to_screen(x1, y2),
                ):
                    painter.drawRect(QRectF(hx - handle_size / 2, hy - handle_size / 2, handle_size, handle_size))
        # Draw current in-progress rect (screen coords, while dragging)
        if self._rect_select_mode and self._rect_drawing and self._rect_start_screen and self._rect_current_screen:
            sx, sy = self._rect_start_screen
            ex, ey = self._rect_current_screen
            rx, ry, rw, rh = min(sx, ex), min(sy, ey), abs(ex - sx), abs(ey - sy)
            painter.setPen(QPen(QColor(0, 120, 215), 2, Qt.DashLine))
            painter.setBrush(QBrush(QColor(0, 120, 215, 40)))
            painter.drawRect(QRectF(rx, ry, rw, rh))

    # ---- Mouse events ----
    def wheelEvent(self, event: QWheelEvent):
        delta = event.angleDelta().y()
        factor = 1.15 if delta > 0 else 1 / 1.15
        self._zoom = max(0.01, min(10.0, self._zoom * factor))
        self.zoomChanged.emit(self._zoom)
        self.update()

    def mousePressEvent(self, event: QMouseEvent):
        pos = event.position()

        # Middle button: always pan
        if event.button() == Qt.MiddleButton:
            self._dragging = True
            self._drag_start = pos.toPoint()
            self._base_offset = (self._offset_x, self._offset_y)
            self.setCursor(Qt.ClosedHandCursor)
            return

        if event.button() != Qt.LeftButton:
            return

        if self._rect_select_mode:
            img_coord = self._screen_to_image(pos.x(), pos.y())
            if img_coord:
                rect_index, corner = self._find_rect_corner_at(pos.x(), pos.y())
                if rect_index >= 0:
                    self._set_selected_rect_index(rect_index)
                    self._rect_resize_index = rect_index
                    self._rect_resize_corner = corner
                    self._rect_resize_origin = self._rects_image[rect_index]
                    self.setCursor(Qt.SizeFDiagCursor if corner in ("tl", "br") else Qt.SizeBDiagCursor)
                    return
                rect_index = self._find_rect_at(*img_coord)
                if rect_index >= 0:
                    self._set_selected_rect_index(rect_index)
                    self._rect_move_index = rect_index
                    self._rect_move_anchor = img_coord
                    self._rect_move_origin = self._rects_image[rect_index]
                    self.setCursor(Qt.SizeAllCursor)
                else:
                    self._set_selected_rect_index(-1)
                    self._rect_drawing = True
                    self._rect_start_screen = (pos.x(), pos.y())
                    self._rect_start_image = img_coord
                    self._rect_current_screen = (pos.x(), pos.y())
            return

        if self._polygon_mode:
            self._on_polygon_click(pos)
            return

        if self._picker_mode:
            img_coord = self._screen_to_image(pos.x(), pos.y())
            if img_coord:
                self._marker_point = (pos.x(), pos.y())
                self.pixelClicked.emit(img_coord[0], img_coord[1])
                self.update()
            return

        # Left button default: pan
        self._dragging = True
        self._drag_start = pos.toPoint()
        self._base_offset = (self._offset_x, self._offset_y)
        self.setCursor(Qt.ClosedHandCursor)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._rect_resize_index >= 0 and self._rect_resize_origin and self._rect_resize_corner:
            pos = event.position()
            img_coord = self._screen_to_image(pos.x(), pos.y())
            if img_coord:
                self._rects_image[self._rect_resize_index] = self._resize_rect_to_corner(
                    self._rect_resize_origin,
                    self._rect_resize_corner,
                    img_coord[0],
                    img_coord[1],
                )
                self.update()
            return
        if self._rect_move_index >= 0 and self._rect_move_anchor and self._rect_move_origin:
            pos = event.position()
            img_coord = self._screen_to_image(pos.x(), pos.y())
            if img_coord:
                dx = img_coord[0] - self._rect_move_anchor[0]
                dy = img_coord[1] - self._rect_move_anchor[1]
                x1, y1, x2, y2 = self._rect_move_origin
                moved = self._clamp_rect_to_image((x1 + dx, y1 + dy, x2 + dx, y2 + dy))
                self._rects_image[self._rect_move_index] = moved
                self.update()
            return
        if self._rect_drawing:
            pos = event.position()
            self._rect_current_screen = (pos.x(), pos.y())
            self.update()
            return
        if self._dragging:
            pos = event.position().toPoint()
            dx = pos.x() - self._drag_start.x()
            dy = pos.y() - self._drag_start.y()
            self._offset_x = self._base_offset[0] + dx
            self._offset_y = self._base_offset[1] + dy
            self.update()
        elif self._picker_mode:
            pos = event.position()
            img_coord = self._screen_to_image(pos.x(), pos.y())
            if img_coord:
                self.pixelMoved.emit(img_coord[0], img_coord[1])

    def mouseReleaseEvent(self, event: QMouseEvent):
        if self._rect_resize_index >= 0:
            self._rect_resize_index = -1
            self._rect_resize_corner = ""
            self._rect_resize_origin = None
            if self._rect_select_mode:
                self.setCursor(Qt.CrossCursor)
            else:
                self.setCursor(Qt.ArrowCursor)
            self.rectsChanged.emit()
            self.update()
            return
        if self._rect_move_index >= 0:
            self._rect_move_index = -1
            self._rect_move_anchor = None
            self._rect_move_origin = None
            if self._rect_select_mode:
                self.setCursor(Qt.CrossCursor)
            else:
                self.setCursor(Qt.ArrowCursor)
            self.rectsChanged.emit()
            self.update()
            return
        if self._rect_drawing and self._rect_start_image:
            pos = event.position()
            end_coord = self._screen_to_image(pos.x(), pos.y())
            if end_coord:
                ix1, iy1 = self._rect_start_image
                ix2, iy2 = end_coord
                x1, y1 = min(ix1, ix2), min(iy1, iy2)
                x2, y2 = max(ix1, ix2), max(iy1, iy2)
                if x2 > x1 and y2 > y1:
                    self._rects_image.append((x1, y1, x2, y2))
                    self._set_selected_rect_index(len(self._rects_image) - 1)
                    self.rectSelected.emit(x1, y1, x2, y2)
                    self.rectsChanged.emit()
            self._rect_drawing = False
            self.update()
            return
        self._dragging = False
        if self._picker_mode or self._polygon_mode or self._rect_select_mode:
            self.setCursor(Qt.CrossCursor)
        else:
            self.setCursor(Qt.ArrowCursor)

    def _finish_polygon(self):
        """Save current polygon to list and reset for next one."""
        if len(self._polygon_image_pts) >= 3:
            self._polygons_list.append(list(self._polygon_image_pts))
            self.polygonClosed.emit(list(self._polygon_image_pts))
        self._polygon_image_pts.clear()
        self._polygon_closed = False
        self.update()

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        if self._polygon_mode and len(self._polygon_image_pts) >= 3:
            self._finish_polygon()

    def _on_polygon_click(self, pos):
        img_coord = self._screen_to_image(pos.x(), pos.y())
        if img_coord is None:
            return
        # Close if near first point
        if len(self._polygon_image_pts) >= 3:
            sx, sy = self._image_to_screen(*self._polygon_image_pts[0])
            dist = ((pos.x() - sx) ** 2 + (pos.y() - sy) ** 2) ** 0.5
            if dist < self._CLOSE_DIST:
                self._finish_polygon()
                return
        self._polygon_image_pts.append(img_coord)
        self.polygonPointAdded.emit(img_coord[0], img_coord[1])
        self.update()


class PreviewPanel(QWidget):
    """Preview panel with before/after tabs, coordinate picker, and polygon ROI drawing."""
    imageDropped = Signal(str)
    polygonCreated = Signal(list)
    roiSelected = Signal(int, int, int, int)  # x1, y1, x2, y2
    saveAllRois = Signal()  # signal to save all accumulated ROIs

    def __init__(self, parent=None):
        super().__init__(parent)
        self._picker_active = False
        self._picked_points = []
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addLayout(self._build_zoom_toolbar())
        layout.addWidget(self._build_extract_tools_panel())
        layout.addWidget(self._build_tabs())
        self.current_image = None
        self._connect_preview_signals()

    def _build_zoom_toolbar(self):
        toolbar = QHBoxLayout()
        toolbar.setSpacing(10)
        self.btn_fit = QPushButton("适应窗口")
        self.btn_fit.clicked.connect(self._on_fit)
        self.btn_zoom_in = QPushButton("+")
        self.btn_zoom_in.setMinimumWidth(44)
        self.btn_zoom_in.clicked.connect(self._on_zoom_in)
        self.btn_zoom_out = QPushButton("-")
        self.btn_zoom_out.setMinimumWidth(44)
        self.btn_zoom_out.clicked.connect(self._on_zoom_out)
        self.zoom_label = QLabel("100%")
        self.zoom_label.setStyleSheet("font-size:13px;color:#6B7280;font-weight:600;")

        toolbar.addWidget(self.btn_fit)
        toolbar.addWidget(self.btn_zoom_in)
        toolbar.addWidget(self.btn_zoom_out)
        toolbar.addWidget(self.zoom_label)
        toolbar.addStretch()
        return toolbar

    def _build_extract_tools_panel(self):
        panel = QFrame()
        panel.setObjectName("extractToolsPanel")
        panel.setStyleSheet(
            "QFrame#extractToolsPanel {"
            "background: rgba(0, 0, 0, 0.03);"
            "border: 1px solid rgba(0, 0, 0, 0.08);"
            "border-radius: 3px;"
            "}"
        )
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(6)

        header = QHBoxLayout()
        title = QLabel("采集工具")
        title_font = title.font()
        title_font.setBold(True)
        title.setFont(title_font)
        self.extract_mode_label = QLabel("当前模式：未启用")
        self.extract_mode_label.setStyleSheet("color: #666666; font-weight: bold;")
        header.addWidget(title)
        header.addStretch()
        header.addWidget(self.extract_mode_label)
        layout.addLayout(header)

        help_label = QLabel("坐标拾取用于采样点位；多边形用于轮廓记录；ROI 用于框选区域并可回填到支持 x/y/w/h 的功能参数。")
        help_label.setWordWrap(True)
        help_label.setStyleSheet("color: #666666; font-size: 12px;")
        help_label.setMaximumHeight(34)
        layout.addWidget(help_label)

        layout.addLayout(self._build_picker_toolbar())
        layout.addLayout(self._build_polygon_toolbar())
        layout.addLayout(self._build_roi_toolbar())

        self.roi_apply_label = QLabel("")
        self.roi_apply_label.setWordWrap(True)
        self.roi_apply_label.setStyleSheet("color: #666666; font-size: 12px;")
        layout.addWidget(self.roi_apply_label)
        return panel

    def _build_picker_toolbar(self):
        tools = QHBoxLayout()
        tools.setSpacing(6)

        section = QLabel("坐标")
        section.setStyleSheet("font-weight: bold; color: #c0392b; min-width: 42px;")

        self.btn_picker = QPushButton("开始拾取")
        self.btn_picker.setObjectName("btnPicker")
        self.btn_picker.setCheckable(True)
        self.btn_picker.toggled.connect(self._on_picker_toggled)

        self.btn_copy_all = QPushButton("复制所有点")
        self.btn_copy_all.setMinimumWidth(112)
        self.btn_copy_all.clicked.connect(self._on_copy_all_points)

        self.btn_clear_points = QPushButton("清除")
        self.btn_clear_points.setMinimumWidth(80)
        self.btn_clear_points.clicked.connect(self._on_clear_points)

        self.coord_label = QLabel("")
        self.coord_label.setStyleSheet("color: #c0392b; font-weight: bold;")

        self.points_label = QLabel("")
        self.points_label.setStyleSheet("color: #c0392b; font-size: 11px;")

        tools.addWidget(section)
        tools.addWidget(self.btn_picker)
        tools.addWidget(self.btn_copy_all)
        tools.addWidget(self.btn_clear_points)
        tools.addWidget(self.coord_label)
        tools.addWidget(self.points_label)
        tools.addStretch()
        return tools

    def _build_polygon_toolbar(self):
        poly_bar = QHBoxLayout()
        poly_bar.setSpacing(6)

        section = QLabel("多边形")
        section.setStyleSheet("font-weight: bold; color: #27ae60; min-width: 56px;")

        self.btn_polygon = QPushButton("开始绘制")
        self.btn_polygon.setObjectName("btnPolygon")
        self.btn_polygon.setCheckable(True)
        self.btn_polygon.toggled.connect(self._on_polygon_toggled)

        self.btn_undo_polygon = QPushButton("撤销上一个")
        self.btn_undo_polygon.clicked.connect(self._on_undo_polygon)

        self.btn_clear_polygon = QPushButton("清除所有")
        self.btn_clear_polygon.clicked.connect(self._on_clear_polygon)

        self.btn_copy_polygon = QPushButton("复制多边形坐标")
        self.btn_copy_polygon.clicked.connect(self._on_copy_polygon)

        self.polygon_info = QLabel("")
        self.polygon_info.setStyleSheet("color: #27ae60; font-weight: bold;")

        poly_bar.addWidget(section)
        poly_bar.addWidget(self.btn_polygon)
        poly_bar.addWidget(self.btn_undo_polygon)
        poly_bar.addWidget(self.btn_clear_polygon)
        poly_bar.addWidget(self.btn_copy_polygon)
        poly_bar.addWidget(self.polygon_info)
        poly_bar.addStretch()
        return poly_bar

    def _build_roi_toolbar(self):
        roi_bar = QHBoxLayout()
        roi_bar.setSpacing(6)

        section = QLabel("ROI")
        section.setStyleSheet("font-weight: bold; color: #0078d7; min-width: 42px;")

        self.btn_rect_select = QPushButton("开始框选")
        self.btn_rect_select.setObjectName("btnRectSelect")
        self.btn_rect_select.setCheckable(True)
        self.btn_rect_select.toggled.connect(self._on_rect_select_toggled)

        self.btn_undo_rect = QPushButton("撤销上一个")
        self.btn_undo_rect.clicked.connect(self._on_undo_rect)

        self.btn_clear_rect = QPushButton("清除所有")
        self.btn_clear_rect.clicked.connect(self._on_clear_rect)

        self.btn_copy_rois = QPushButton("复制所有ROI坐标")
        self.btn_copy_rois.clicked.connect(self._on_copy_rois)

        self.btn_save_rois = QPushButton("保存所有ROI")
        self.btn_save_rois.setStyleSheet("QPushButton { background-color: #0078d7; color: white; font-weight: bold; }")
        self.btn_save_rois.clicked.connect(self._on_save_rois)

        self.roi_info = QLabel("")
        self.roi_info.setStyleSheet("color: #0078d7; font-weight: bold;")

        roi_bar.addWidget(section)
        roi_bar.addWidget(self.btn_rect_select)
        roi_bar.addWidget(self.btn_undo_rect)
        roi_bar.addWidget(self.btn_clear_rect)
        roi_bar.addWidget(self.btn_copy_rois)
        roi_bar.addWidget(self.btn_save_rois)
        roi_bar.addWidget(self.roi_info)
        roi_bar.addStretch()
        return roi_bar

    def _build_tabs(self):
        self.tabs = QTabWidget()
        self.original_view = ZoomableLabel()
        self.result_view = ZoomableLabel()
        self.tabs.addTab(self.original_view, "原图")
        self.tabs.addTab(self.result_view, "处理后")
        return self.tabs

    def _connect_preview_signals(self):
        for view in (self.original_view, self.result_view):
            view.pixelClicked.connect(self._on_pixel_clicked)
            view.pixelMoved.connect(self._on_pixel_moved)
            view.polygonPointAdded.connect(self._on_polygon_point_added)
            view.polygonClosed.connect(self._on_polygon_closed_signal)
            view.rectSelected.connect(self._on_rect_selected)

    def set_original(self, img: np.ndarray | None) -> None:
        self.current_image = img
        self.original_view.set_pixmap(cv2_to_qpixmap(img))

    def set_result(self, img: np.ndarray | None) -> None:
        self.result_view.set_pixmap(cv2_to_qpixmap(img))
        self.tabs.setCurrentIndex(1)

    # ---- Picker callbacks ----
    def _on_picker_toggled(self, checked):
        if checked and self.btn_polygon.isChecked():
            self.btn_polygon.setChecked(False)
        if checked and self.btn_rect_select.isChecked():
            self.btn_rect_select.setChecked(False)
        self._picker_active = checked
        self.original_view.picker_mode = checked
        self.result_view.picker_mode = checked
        if not checked:
            self.coord_label.setText("")
            self._picked_points.clear()
            self.points_label.setText("")
        self._update_extract_mode_hint()

    def _on_pixel_clicked(self, x, y):
        self._picked_points.append((x, y))
        self.coord_label.setText(f"X={x}  Y={y}")
        self._update_points_label()
        clipboard = QApplication.clipboard()
        if clipboard:
            clipboard.setText(f"({x}, {y})")

    def _on_pixel_moved(self, x, y):
        if self._picker_active:
            self.coord_label.setText(f"X={x}  Y={y}")

    def _update_points_label(self):
        if not self._picked_points:
            self.points_label.setText("")
            return
        recent = self._picked_points[-5:]
        text = " | ".join(f"({x},{y})" for x, y in recent)
        if len(self._picked_points) > 5:
            text = f"... | {text}"
        self.points_label.setText(f"已采集{len(self._picked_points)}点: {text}")

    def _on_copy_all_points(self):
        if not self._picked_points:
            return
        clipboard = QApplication.clipboard()
        if clipboard:
            clipboard.setText(json.dumps(self._picked_points))

    def _on_clear_points(self):
        self._picked_points.clear()
        self.points_label.setText("")
        self.coord_label.setText("")

    # ---- Polygon callbacks ----
    def _on_polygon_toggled(self, checked):
        if checked and self._picker_active:
            self.btn_picker.setChecked(False)
        if checked and self.btn_rect_select.isChecked():
            self.btn_rect_select.setChecked(False)
        self.original_view.polygon_mode = checked
        self.result_view.polygon_mode = checked
        if not checked:
            self.polygon_info.setText("")
        self._update_extract_mode_hint()

    def _on_polygon_point_added(self, x, y):
        view = self.original_view if self.tabs.currentIndex() == 0 else self.result_view
        cur_pts = len(view._polygon_image_pts)
        total = len(view._polygons_list)
        self.polygon_info.setText(f"多边形{total + 1}: {cur_pts} 个点 (双击或点击起点关闭)")

    def _on_polygon_closed_signal(self, points):
        view = self.original_view if self.tabs.currentIndex() == 0 else self.result_view
        total = len(view._polygons_list)
        self.polygon_info.setText(f"已完成 {total} 个多边形 (可继续绘制)")
        self.polygonCreated.emit(points)

    def _on_undo_polygon(self):
        self.original_view.remove_last_polygon()
        self.result_view.remove_last_polygon()
        self._update_polygon_info()

    def _on_clear_polygon(self):
        self.original_view.clear_polygon()
        self.result_view.clear_polygon()
        self.polygon_info.setText("")

    def _on_copy_polygon(self):
        view = self.original_view if self.tabs.currentIndex() == 0 else self.result_view
        all_polys = view._polygons_list
        if not all_polys:
            return
        clipboard = QApplication.clipboard()
        if clipboard:
            clipboard.setText(json.dumps(all_polys))
        self.polygon_info.setText(f"已复制 {len(all_polys)} 个多边形坐标")

    def _update_polygon_info(self):
        view = self.original_view if self.tabs.currentIndex() == 0 else self.result_view
        total = len(view._polygons_list)
        cur = len(view._polygon_image_pts)
        if total > 0:
            self.polygon_info.setText(f"已完成 {total} 个多边形" +
                                      (f"，绘制中: {cur} 个点" if cur > 0 else ""))
        elif cur > 0:
            self.polygon_info.setText(f"多边形1: {cur} 个点 (双击或点击起点关闭)")
        else:
            self.polygon_info.setText("")

    # ---- Rect select callbacks ----
    def _on_rect_select_toggled(self, checked):
        if checked and self._picker_active:
            self.btn_picker.setChecked(False)
        if checked and self.btn_polygon.isChecked():
            self.btn_polygon.setChecked(False)
        self.original_view.rect_select_mode = checked
        self.result_view.rect_select_mode = checked
        if not checked:
            self._update_roi_info()
        self._update_extract_mode_hint()

    def _on_rect_selected(self, x1, y1, x2, y2):
        self._update_roi_info()
        self.roiSelected.emit(x1, y1, x2, y2)

    def _update_roi_info(self):
        view = self.original_view if self.tabs.currentIndex() == 0 else self.result_view
        count = len(view._rects_image)
        if count > 0:
            rects_text = " | ".join(f"({x1},{y1}-{x2},{y2})" for x1, y1, x2, y2 in view._rects_image[-3:])
            if count > 3:
                rects_text = f"... | {rects_text}"
            self.roi_info.setText(f"已框选 {count} 个ROI: {rects_text}")
        else:
            self.roi_info.setText("")

    def _on_undo_rect(self):
        self.original_view.remove_last_rect()
        self.result_view.remove_last_rect()
        self._update_roi_info()

    def _on_clear_rect(self):
        self.original_view.rect_select_mode = False
        self.result_view.rect_select_mode = False
        self.original_view.clear_all_rects()
        self.result_view.clear_all_rects()
        self.btn_rect_select.setChecked(False)
        self.roi_info.setText("")
        self.roi_apply_label.setText("")

    def _on_copy_rois(self):
        view = self.original_view if self.tabs.currentIndex() == 0 else self.result_view
        rects = view._rects_image
        if not rects:
            return
        clipboard = QApplication.clipboard()
        if clipboard:
            clipboard.setText(json.dumps(rects))
        self.roi_info.setText(f"已复制 {len(rects)} 个ROI坐标")

    def _on_save_rois(self):
        self.saveAllRois.emit()

    def get_roi_rects(self) -> list[tuple[int, int, int, int]]:
        """Return list of (x1, y1, x2, y2) in image coordinates."""
        return list(self.original_view._rects_image)

    def set_roi_apply_feedback(self, text: str, *, level: str = "info") -> None:
        colors = {
            "info": "#666666",
            "success": "#1f7a1f",
            "warning": "#a15c00",
        }
        self.roi_apply_label.setStyleSheet(f"color: {colors.get(level, '#666666')}; font-size: 12px;")
        self.roi_apply_label.setText(text)

    def _update_extract_mode_hint(self) -> None:
        if self.btn_picker.isChecked():
            self.extract_mode_label.setText("当前模式：坐标拾取")
            self.extract_mode_label.setStyleSheet("color: #c0392b; font-weight: bold;")
        elif self.btn_polygon.isChecked():
            self.extract_mode_label.setText("当前模式：多边形绘制")
            self.extract_mode_label.setStyleSheet("color: #27ae60; font-weight: bold;")
        elif self.btn_rect_select.isChecked():
            self.extract_mode_label.setText("当前模式：ROI 框选")
            self.extract_mode_label.setStyleSheet("color: #0078d7; font-weight: bold;")
        else:
            self.extract_mode_label.setText("当前模式：未启用")
            self.extract_mode_label.setStyleSheet("color: #666666; font-weight: bold;")

    # ---- Zoom helpers ----
    def _on_fit(self):
        current = self.tabs.currentWidget()
        if isinstance(current, ZoomableLabel):
            current.fit_to_view()
            self._update_zoom_label()

    def _on_zoom_in(self):
        current = self.tabs.currentWidget()
        if isinstance(current, ZoomableLabel):
            current._zoom *= 1.25
            current.update()
            self._update_zoom_label()

    def _on_zoom_out(self):
        current = self.tabs.currentWidget()
        if isinstance(current, ZoomableLabel):
            current._zoom *= 0.8
            current.update()
            self._update_zoom_label()

    def _update_zoom_label(self):
        current = self.tabs.currentWidget()
        if isinstance(current, ZoomableLabel):
            self.zoom_label.setText(f"{int(current._zoom * 100)}%")

    def _schedule_fit_to_view(self):
        if not hasattr(self, "_resize_timer"):
            self._resize_timer = QTimer(self)
            self._resize_timer.setSingleShot(True)
            self._resize_timer.timeout.connect(self._do_fit_to_view)
        self._resize_timer.start(100)

    def _do_fit_to_view(self):
        self.original_view.fit_to_view()
        self.result_view.fit_to_view()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._schedule_fit_to_view()
