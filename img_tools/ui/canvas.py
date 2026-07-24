"""Interactive image canvas used by the inspector workspace."""
from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QMouseEvent, QPainter, QPen, QPixmap, QWheelEvent
from PySide6.QtWidgets import QWidget

from img_tools.core.models import Roi


def _to_pixmap(image: np.ndarray) -> QPixmap:
    if image.ndim == 2:
        height, width = image.shape
        qimage = QImage(image.data, width, height, image.strides[0], QImage.Format_Grayscale8)
    elif image.shape[2] == 4:
        rgba = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
        height, width = rgba.shape[:2]
        qimage = QImage(rgba.data, width, height, rgba.strides[0], QImage.Format_RGBA8888)
    else:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]
        qimage = QImage(rgb.data, width, height, rgb.strides[0], QImage.Format_RGB888)
    return QPixmap.fromImage(qimage.copy())


class ImageCanvas(QWidget):
    """Canvas with pixel inspection, middle-button pan, zoom and drag ROI creation."""

    pixelHovered = Signal(int, int, object)
    coordinateClicked = Signal(int, int)
    pointSelected = Signal(int, int)
    roiDrawn = Signal(object)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._image: np.ndarray | None = None
        self._pixmap = QPixmap()
        self._source_size: tuple[int, int] | None = None
        self._zoom = 1.0
        self._offset = QPointF()
        self._rois: list[Roi] = []
        self._preview_roi: Roi | None = None
        self._active_roi = -1
        self._roi_mode = False
        self._drag_origin: tuple[int, int] | None = None
        self._drag_current: tuple[int, int] | None = None
        self._pressed_coordinate: tuple[int, int] | None = None
        self._pan_origin: QPointF | None = None
        self._pan_offset = QPointF()
        self.setMinimumSize(480, 360)
        self.setMouseTracking(True)
        self.setAutoFillBackground(False)

    @property
    def image_size(self) -> tuple[int, int] | None:
        return self._source_size

    @property
    def roi_mode(self) -> bool:
        return self._roi_mode

    @property
    def preview_roi(self) -> Roi | None:
        return self._preview_roi

    def set_roi_mode(self, enabled: bool) -> None:
        """Enable point/drag ROI capture; normal view mode remains inspection-only."""
        self._roi_mode = enabled
        if not enabled:
            self._drag_origin = None
            self._drag_current = None
            self._preview_roi = None
        self.setCursor(Qt.CrossCursor if enabled else Qt.ArrowCursor)
        self.update()

    def set_image(self, image: np.ndarray | None, *, source_size: tuple[int, int] | None = None) -> None:
        self._image = image
        self._pixmap = _to_pixmap(image) if image is not None else QPixmap()
        self._source_size = source_size or (image.shape[1], image.shape[0]) if image is not None else None
        self._rois.clear()
        self._preview_roi = None
        self._active_roi = -1
        self.fit_to_view()
        self.update()

    def set_rois(self, rois: Iterable[Roi], active_index: int = -1) -> None:
        self._rois = list(rois)
        self._active_roi = active_index
        self.update()

    def set_preview_roi(self, roi: Roi | None) -> None:
        self._preview_roi = roi
        self.update()

    def fit_to_view(self) -> None:
        if self._pixmap.isNull() or self.width() <= 0 or self.height() <= 0:
            return
        self._zoom = min((self.width() - 32) / self._pixmap.width(), (self.height() - 32) / self._pixmap.height(), 1.0)
        self._zoom = max(0.01, self._zoom)
        self._offset = QPointF()

    def _image_top_left(self) -> QPointF:
        return QPointF(
            (self.width() - self._pixmap.width() * self._zoom) / 2 + self._offset.x(),
            (self.height() - self._pixmap.height() * self._zoom) / 2 + self._offset.y(),
        )

    def _screen_to_image(self, point: QPointF) -> tuple[int, int] | None:
        if self._pixmap.isNull():
            return None
        top_left = self._image_top_left()
        x = int((point.x() - top_left.x()) / self._zoom)
        y = int((point.y() - top_left.y()) / self._zoom)
        if 0 <= x < self._pixmap.width() and 0 <= y < self._pixmap.height() and self._source_size:
            source_w, source_h = self._source_size
            x, y = int(x * source_w / self._pixmap.width()), int(y * source_h / self._pixmap.height())
            return x, y
        return None

    def _image_to_screen(self, x: int, y: int) -> QPointF:
        top_left = self._image_top_left()
        source_w, source_h = self._source_size or (self._pixmap.width(), self._pixmap.height())
        return QPointF(top_left.x() + x * self._pixmap.width() / source_w * self._zoom, top_left.y() + y * self._pixmap.height() / source_h * self._zoom)

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt method name
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#171b22"))
        if self._pixmap.isNull():
            painter.setPen(QColor("#9aa4b2"))
            painter.drawText(self.rect(), Qt.AlignCenter, "打开一张图片或一个图片目录")
            return

        top_left = self._image_top_left()
        target = QRectF(top_left.x(), top_left.y(), self._pixmap.width() * self._zoom, self._pixmap.height() * self._zoom)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, self._zoom < 1)
        painter.drawPixmap(target, self._pixmap, QRectF(self._pixmap.rect()))
        for index, roi in enumerate(self._rois):
            self._draw_roi(painter, roi, QColor("#00c2ff") if index == self._active_roi else QColor("#ffb020"), str(index + 1), index == self._active_roi)
        if self._preview_roi is not None:
            self._draw_roi(painter, self._preview_roi, QColor("#00e096"), "预览", False, dashed=True)
        if self._drag_origin is not None and self._drag_current is not None:
            x1, y1 = self._drag_origin
            x2, y2 = self._drag_current
            roi = Roi(min(x1, x2), min(y1, y2), max(1, abs(x2 - x1)), max(1, abs(y2 - y1)), "拖拽")
            self._draw_roi(painter, roi, QColor("#4dabf7"), "新 ROI", False, dashed=True)

    def _draw_roi(self, painter: QPainter, roi: Roi, color: QColor, label: str, active: bool, *, dashed: bool = False) -> None:
        point = self._image_to_screen(roi.x, roi.y)
        bottom_right = self._image_to_screen(roi.x2, roi.y2)
        rect = QRectF(point.x(), point.y(), bottom_right.x() - point.x(), bottom_right.y() - point.y())
        pen = QPen(color, 2.5 if active else 1.5, Qt.DashLine if dashed else Qt.SolidLine)
        painter.setPen(pen)
        painter.setBrush(QColor(color.red(), color.green(), color.blue(), 28))
        painter.drawRect(rect)
        painter.setPen(QPen(color))
        # Keep the label away from the ROI origin, where point coordinates are inspected.
        label_offset = QPointF(4, -6) if point.y() > 22 else QPointF(4, 18)
        painter.drawText(point + label_offset, f"{label}: {roi.x},{roi.y} {roi.width}×{roi.height}")

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MiddleButton:
            self._pan_origin = event.position()
            self._pan_offset = QPointF(self._offset)
            self.setCursor(Qt.ClosedHandCursor)
            return
        if event.button() == Qt.LeftButton:
            coordinate = self._screen_to_image(event.position())
            self._pressed_coordinate = coordinate
            if not self._roi_mode:
                return
            if coordinate is not None:
                self._drag_origin = coordinate
                self._drag_current = coordinate
            return

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._pan_origin is not None:
            self._offset = self._pan_offset + event.position() - self._pan_origin
            self.update()
            return
        coordinate = self._screen_to_image(event.position())
        if coordinate is not None and self._image is not None:
            x, y = coordinate
            source_w, source_h = self._source_size or (self._image.shape[1], self._image.shape[0])
            preview_x = min(self._image.shape[1] - 1, int(x * self._image.shape[1] / source_w))
            preview_y = min(self._image.shape[0] - 1, int(y * self._image.shape[0] / source_h))
            pixel = self._image[preview_y, preview_x]
            values = tuple(int(value) for value in (pixel if np.ndim(pixel) else [pixel]))
            self.pixelHovered.emit(x, y, values)
        if self._drag_origin is not None and coordinate is not None:
            self._drag_current = coordinate
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MiddleButton and self._pan_origin is not None:
            self._pan_origin = None
            self.setCursor(Qt.ArrowCursor)
            return
        if event.button() != Qt.LeftButton:
            return
        if not self._roi_mode:
            if self._pressed_coordinate is not None:
                self.coordinateClicked.emit(*self._pressed_coordinate)
            self._pressed_coordinate = None
            return
        if self._drag_origin is None:
            self._pressed_coordinate = None
            return
        origin, current = self._drag_origin, self._drag_current or self._drag_origin
        self._drag_origin = None
        self._drag_current = None
        dx, dy = abs(current[0] - origin[0]), abs(current[1] - origin[1])
        if dx < 3 and dy < 3:
            self.coordinateClicked.emit(*origin)
            self.pointSelected.emit(*origin)
        else:
            roi = Roi(min(origin[0], current[0]), min(origin[1], current[1]), dx, dy, "ROI")
            self.roiDrawn.emit(roi)
        self._pressed_coordinate = None
        self.update()

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802
        if self._pixmap.isNull():
            return
        factor = 1.2 if event.angleDelta().y() > 0 else 1 / 1.2
        self._zoom = max(0.02, min(16.0, self._zoom * factor))
        self.update()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        if self._pixmap.isNull():
            return
        if self._zoom == 1.0 and self._offset.isNull():
            self.fit_to_view()
