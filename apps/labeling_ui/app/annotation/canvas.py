from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np
from PySide6.QtCore import QPoint, QRect, Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

from .document import AnnotationDocument, AnnotationLayer, AnnotationShape, ImageSize


class AnnotationCanvas(QWidget):
    """Generic image annotation canvas backed by AnnotationDocument."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.document = AnnotationDocument(
            image_path="",
            image_size=ImageSize(width=0, height=0),
            layers=[],
        )
        self.image_bgr: np.ndarray | None = None
        self.image_qimage: QImage | None = None
        self.scale = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self._pan_anchor: QPoint | None = None
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMinimumSize(560, 420)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def set_document(self, document: AnnotationDocument) -> None:
        self.document = document
        self.update()

    def set_image(self, image_bgr: np.ndarray, *, image_path: str = "") -> None:
        self.image_bgr = image_bgr
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        height, width = image_rgb.shape[:2]
        bytes_per_line = width * 3
        self.image_qimage = QImage(image_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888).copy()
        if image_path:
            self.document.image_path = image_path
        self.document.image_size = ImageSize(width=width, height=height)
        self.fit_view()

    def visible_layer_keys(self) -> list[str]:
        return [layer.key for layer in self.document.layers if layer.visible]

    def visible_layers(self) -> list[AnnotationLayer]:
        return [layer for layer in self.document.layers if layer.visible]

    def set_layer_visible(self, key: str, visible: bool) -> None:
        layer = self._require_layer(key)
        layer.visible = bool(visible)
        self.update()

    def set_layer_editable(self, key: str, editable: bool) -> None:
        layer = self._require_layer(key)
        layer.editable = bool(editable)

    def add_point(self, layer_key: str, x: float, y: float, *, attrs: dict | None = None) -> AnnotationShape:
        layer = self._require_layer(layer_key)
        if layer.shape_type != "point":
            raise ValueError(f"Layer {layer_key!r} is not a point layer")
        if not layer.editable:
            raise ValueError(f"Layer {layer_key!r} is not editable")
        shape = AnnotationShape(
            id=self._next_shape_id(layer),
            geometry={"x": float(x), "y": float(y)},
            attrs=dict(attrs or {}),
        )
        layer.shapes.append(shape)
        self.update()
        return shape

    def iter_visible_shapes(self) -> Iterable[tuple[AnnotationLayer, AnnotationShape]]:
        for layer in self.document.layers:
            if not layer.visible:
                continue
            for shape in layer.shapes:
                yield layer, shape

    def fit_view(self) -> None:
        if self.image_qimage is None or self.image_qimage.isNull():
            return
        view_w = max(self.width(), 1)
        view_h = max(self.height(), 1)
        self.scale = min(view_w / max(self.image_qimage.width(), 1), view_h / max(self.image_qimage.height(), 1))
        self.scale = max(self.scale, 0.05)
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.update()

    def _require_layer(self, key: str) -> AnnotationLayer:
        layer = self.document.get_layer(key)
        if layer is None:
            raise KeyError(key)
        return layer

    @staticmethod
    def _next_shape_id(layer: AnnotationLayer) -> str:
        numeric_ids = [int(shape.id) for shape in layer.shapes if str(shape.id).isdigit()]
        return str(max(numeric_ids, default=-1) + 1)

    def _canvas_to_image(self, pos: QPoint) -> tuple[float, float]:
        return (
            self.pan_x + pos.x() / max(self.scale, 1e-6),
            self.pan_y + pos.y() / max(self.scale, 1e-6),
        )

    def _image_to_canvas(self, x: float, y: float) -> tuple[int, int]:
        return (
            int(round((x - self.pan_x) * self.scale)),
            int(round((y - self.pan_y) * self.scale)),
        )

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#111111"))
        if self.image_qimage is None or self.image_qimage.isNull():
            painter.setPen(QColor("#9CA3AF"))
            painter.drawText(self.rect(), Qt.AlignCenter, "请先加载图片")
            return

        source = QRect(
            int(round(self.pan_x)),
            int(round(self.pan_y)),
            int(round(self.width() / max(self.scale, 1e-6))),
            int(round(self.height() / max(self.scale, 1e-6))),
        )
        painter.drawImage(self.rect(), self.image_qimage, source)
        for layer, shape in self.iter_visible_shapes():
            self._paint_shape(painter, layer, shape)

    def _paint_shape(self, painter: QPainter, layer: AnnotationLayer, shape: AnnotationShape) -> None:
        color = QColor(layer.style.color)
        painter.setPen(QPen(color, 2))
        painter.setBrush(color)
        geometry = shape.geometry
        if layer.shape_type == "point":
            x, y = self._image_to_canvas(float(geometry.get("x", 0)), float(geometry.get("y", 0)))
            painter.drawEllipse(QPoint(x, y), 4, 4)
        elif layer.shape_type == "line":
            points = geometry.get("points", [])
            if isinstance(points, list) and len(points) >= 2:
                a = points[0]
                b = points[1]
                ax, ay = self._image_to_canvas(float(a[0]), float(a[1]))
                bx, by = self._image_to_canvas(float(b[0]), float(b[1]))
                painter.drawLine(ax, ay, bx, by)
        elif layer.shape_type == "box":
            x1, y1 = self._image_to_canvas(float(geometry.get("x1", 0)), float(geometry.get("y1", 0)))
            x2, y2 = self._image_to_canvas(float(geometry.get("x2", 0)), float(geometry.get("y2", 0)))
            painter.drawRect(min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))
        elif layer.shape_type == "polygon":
            points = geometry.get("points", [])
            if isinstance(points, list) and len(points) >= 2:
                canvas_points = [self._image_to_canvas(float(point[0]), float(point[1])) for point in points]
                for current, nxt in zip(canvas_points, canvas_points[1:] + canvas_points[:1]):
                    painter.drawLine(current[0], current[1], nxt[0], nxt[1])

    def mousePressEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self._pan_anchor = event.pos()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._pan_anchor is not None:
            dx = event.pos().x() - self._pan_anchor.x()
            dy = event.pos().y() - self._pan_anchor.y()
            self.pan_x -= dx / max(self.scale, 1e-6)
            self.pan_y -= dy / max(self.scale, 1e-6)
            self._pan_anchor = event.pos()
            self.update()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self._pan_anchor = None
            event.accept()
            return
        super().mouseReleaseEvent(event)
