"""CAB-F stitch point editor dialog integrated into img_tools."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import QPoint, QRect, Qt, QThread, Signal
from PySide6.QtGui import QAction, QColor, QFont, QImage, QPainter, QPen, QWheelEvent
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QDoubleSpinBox,
    QSizePolicy,
    QSplitter,
    QToolBar,
    QVBoxLayout,
    QWidget,
)
from .autosave import AutoSaveStatusController
from cabf import make_empty_master_annotation, normalize_points_for_editor, read_image_bgr, write_json


DEFAULT_COSMOS_ROOT = Path("")
DEFAULT_DETECTOR_SCRIPT = DEFAULT_COSMOS_ROOT / "algo" / "cab_f" / "sew_point_detector.py"
DEFAULT_MODEL_PATH = DEFAULT_COSMOS_ROOT / "assets" / "weights" / "cab_f" / "sew_point_detector.onnx"


class DetectorRunner:
    """Lazy loader for the external CAB-F stitch detector."""

    def __init__(self):
        self._detector_class = None
        self._detector = None
        self._cache_key = None

    def _load_detector_class(self, detector_script_path: str):
        detector_path = Path(detector_script_path)
        spec = importlib.util.spec_from_file_location("cab_f_sew_point_detector_in_img_tools", detector_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"无法加载检测器文件: {detector_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.SewPointDetector

    def detect(self, image_bgr: np.ndarray, detector_script_path: str, model_path: str, conf: float):
        cache_key = (str(detector_script_path), str(model_path), float(conf))
        if self._detector is None or self._cache_key != cache_key:
            self._detector_class = self._load_detector_class(detector_script_path)
            self._detector = self._detector_class({"path": str(model_path), "conf": float(conf)})
            self._cache_key = cache_key

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        result = self._detector.evaluate(image_rgb)
        raw_points = result.get("points", [])
        return [
            {
                "id": idx,
                "x": float(point[0]),
                "y": float(point[1]),
                "score": float(point[2]) if len(point) >= 3 else 1.0,
                "source": "model",
            }
            for idx, point in enumerate(raw_points)
        ]


class PointCanvas(QWidget):
    """Interactive image canvas for point add/move/delete."""

    pointSelectionChanged = Signal(object)
    pointCountChanged = Signal(int)
    statusMessage = Signal(str)
    annotationModified = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMinimumSize(520, 380)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.image_bgr: Optional[np.ndarray] = None
        self.image_rgb: Optional[np.ndarray] = None
        self.image_qimage: Optional[QImage] = None
        self.image_path: str = ""
        self.points: list[dict] = []
        self.selected_point_id: Optional[int] = None
        self.mode = "select"

        self.scale = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0

        self._drag_point_id: Optional[int] = None
        self._pan_anchor: Optional[QPoint] = None

    def set_mode(self, mode: str):
        self.mode = mode
        self.update()

    def set_image(self, image_bgr: np.ndarray, image_path: str = ""):
        self.image_bgr = image_bgr
        self.image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        height, width = self.image_rgb.shape[:2]
        bytes_per_line = width * 3
        self.image_qimage = QImage(self.image_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888).copy()
        self.image_path = image_path
        self.points = []
        self.selected_point_id = None
        self.fit_view()
        self.pointCountChanged.emit(len(self.points))

    def set_points(self, points: list[dict]):
        self.points = normalize_points_for_editor(points)
        self.selected_point_id = None
        self.pointCountChanged.emit(len(self.points))
        self.update()

    def fit_view(self):
        if self.image_qimage is None or self.width() <= 0 or self.height() <= 0:
            return
        image_w = self.image_qimage.width()
        image_h = self.image_qimage.height()
        self.scale = min(self.width() / max(image_w, 1), self.height() / max(image_h, 1))
        self.scale = max(self.scale, 0.05)
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.update()

    def _clamp_pan(self):
        if self.image_qimage is None:
            return
        view_w = self.width() / max(self.scale, 1e-6)
        view_h = self.height() / max(self.scale, 1e-6)
        max_pan_x = max(self.image_qimage.width() - view_w, 0.0)
        max_pan_y = max(self.image_qimage.height() - view_h, 0.0)
        self.pan_x = float(np.clip(self.pan_x, 0.0, max_pan_x))
        self.pan_y = float(np.clip(self.pan_y, 0.0, max_pan_y))

    def _canvas_to_image(self, pos: QPoint):
        return (
            self.pan_x + pos.x() / max(self.scale, 1e-6),
            self.pan_y + pos.y() / max(self.scale, 1e-6),
        )

    def _image_to_canvas(self, x: float, y: float):
        return (
            int(round((x - self.pan_x) * self.scale)),
            int(round((y - self.pan_y) * self.scale)),
        )

    def _nearest_point(self, image_x: float, image_y: float, max_screen_dist: float = 14.0):
        if not self.points:
            return None
        pts = np.asarray([[point["x"], point["y"]] for point in self.points], dtype=np.float32)
        click = np.asarray([image_x, image_y], dtype=np.float32)
        distances = np.linalg.norm(pts - click[None, :], axis=1)
        idx = int(np.argmin(distances))
        max_dist = max_screen_dist / max(self.scale, 1e-6)
        if float(distances[idx]) > max_dist:
            return None
        return self.points[idx]

    def _next_point_id(self) -> int:
        if not self.points:
            return 0
        return max(int(point["id"]) for point in self.points) + 1

    def delete_selected_point(self):
        if self.selected_point_id is None:
            return
        self.points = [point for point in self.points if int(point["id"]) != int(self.selected_point_id)]
        self.statusMessage.emit(f"已删除点 {self.selected_point_id}")
        self.selected_point_id = None
        self.pointSelectionChanged.emit(None)
        self.pointCountChanged.emit(len(self.points))
        self.annotationModified.emit()
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#111111"))

        if self.image_qimage is None:
            painter.setPen(QColor("#9CA3AF"))
            painter.setFont(QFont("", 14))
            painter.drawText(self.rect(), Qt.AlignCenter, "请先选择图片")
            return

        self._clamp_pan()
        src_rect = QRect(
            int(self.pan_x),
            int(self.pan_y),
            int(np.ceil(self.width() / max(self.scale, 1e-6))),
            int(np.ceil(self.height() / max(self.scale, 1e-6))),
        )
        painter.drawImage(self.rect(), self.image_qimage, src_rect)

        font = painter.font()
        font.setPointSize(9)
        painter.setFont(font)

        for point in self.points:
            cx, cy = self._image_to_canvas(point["x"], point["y"])
            is_selected = int(point["id"]) == self.selected_point_id
            radius = 6 if is_selected else 4
            fill = QColor(0, 220, 255) if is_selected else QColor(80, 255, 80)
            painter.setPen(QPen(QColor(20, 20, 20), 1))
            painter.setBrush(fill)
            painter.drawEllipse(QPoint(cx, cy), radius, radius)
            painter.setPen(QColor("#ffffff"))
            painter.drawText(cx + 6, cy - 6, str(point["id"]))

        header = [
            f"mode={self.mode} points={len(self.points)} selected={self.selected_point_id}",
            f"zoom={self.scale:.2f}",
        ]
        painter.setPen(QColor(80, 255, 80))
        y = 22
        for line in header:
            painter.drawText(10, y, line)
            y += 20

    def mousePressEvent(self, event):
        if self.image_qimage is None:
            return
        if event.button() == Qt.RightButton:
            self._pan_anchor = event.pos()
            return
        if event.button() != Qt.LeftButton:
            return

        image_x, image_y = self._canvas_to_image(event.position().toPoint())
        if self.mode == "add":
            new_point = {
                "id": self._next_point_id(),
                "x": float(image_x),
                "y": float(image_y),
                "score": 1.0,
                "source": "manual",
            }
            self.points.append(new_point)
            self.selected_point_id = int(new_point["id"])
            self.pointSelectionChanged.emit(new_point)
            self.pointCountChanged.emit(len(self.points))
            self.statusMessage.emit(f"已新增点 {new_point['id']}")
            self.annotationModified.emit()
            self.update()
            return

        nearest = self._nearest_point(image_x, image_y)
        if nearest is None:
            self.selected_point_id = None
            self.pointSelectionChanged.emit(None)
            self.update()
            return

        self.selected_point_id = int(nearest["id"])
        self.pointSelectionChanged.emit(nearest)
        if self.mode == "delete":
            self.delete_selected_point()
            return
        if self.mode == "move":
            self._drag_point_id = int(nearest["id"])
        self.update()

    def mouseMoveEvent(self, event):
        if self.image_qimage is None:
            return
        if self._pan_anchor is not None:
            delta = event.pos() - self._pan_anchor
            self.pan_x -= delta.x() / max(self.scale, 1e-6)
            self.pan_y -= delta.y() / max(self.scale, 1e-6)
            self._pan_anchor = event.pos()
            self.update()
            return
        if self._drag_point_id is None or self.mode != "move":
            return
        image_x, image_y = self._canvas_to_image(event.position().toPoint())
        for point in self.points:
            if int(point["id"]) == int(self._drag_point_id):
                point["x"] = float(image_x)
                point["y"] = float(image_y)
                point["source"] = "manual"
                self.pointSelectionChanged.emit(point)
                self.annotationModified.emit()
                break
        self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.RightButton:
            self._pan_anchor = None
        if event.button() == Qt.LeftButton:
            self._drag_point_id = None

    def wheelEvent(self, event: QWheelEvent):
        if self.image_qimage is None:
            return
        old_scale = self.scale
        factor = 1.15 if event.angleDelta().y() > 0 else 1.0 / 1.15
        new_scale = float(np.clip(old_scale * factor, 0.05, 20.0))
        if abs(new_scale - old_scale) < 1e-6:
            return
        image_x, image_y = self._canvas_to_image(event.position().toPoint())
        self.scale = new_scale
        self.pan_x = image_x - event.position().x() / max(self.scale, 1e-6)
        self.pan_y = image_y - event.position().y() / max(self.scale, 1e-6)
        self.update()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.image_qimage is not None:
            self._clamp_pan()


class StitchPointEditorDialog(QDialog):
    """Integrated CAB-F stitch point editor dialog."""

    def __init__(self, parent=None, image: Optional[np.ndarray] = None, image_path: str = ""):
        super().__init__(parent)
        self.setWindowTitle("CAB-F 针点编辑器")
        self.resize(1420, 880)
        self._left_panel_visible = True
        self._left_panel_width = 340
        self.detector_runner = DetectorRunner()
        self.image_bgr: Optional[np.ndarray] = None
        self.image_path = image_path
        self._has_unsaved_changes = False

        self._build_ui()
        self._connect_signals()

        if image is not None:
            self.set_image(image, image_path)

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        topbar = QHBoxLayout()
        topbar.setSpacing(8)
        self.btn_toggle_sidebar = QPushButton("收起侧栏")
        self.btn_toggle_sidebar.clicked.connect(self._toggle_left_panel)
        topbar.addWidget(self.btn_toggle_sidebar)
        topbar.addStretch(1)
        root.addLayout(topbar)

        self.splitter = QSplitter(Qt.Horizontal)
        root.addWidget(self.splitter)

        self.left_panel = QWidget()
        self.left_panel.setMinimumWidth(260)
        self.left_panel.setMaximumWidth(380)
        left_layout = QVBoxLayout(self.left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(10)

        form_box = QFrame()
        form_layout = QFormLayout(form_box)

        self.edit_image = QLineEdit()
        self.edit_detector_script = QLineEdit(str(DEFAULT_DETECTOR_SCRIPT))
        self.edit_model = QLineEdit(str(DEFAULT_MODEL_PATH))
        self.edit_output = QLineEdit("")
        self.spin_conf = QDoubleSpinBox()
        self.spin_conf.setRange(0.01, 1.0)
        self.spin_conf.setDecimals(3)
        self.spin_conf.setSingleStep(0.05)
        self.spin_conf.setValue(0.5)

        form_layout.addRow("图片", self.edit_image)
        form_layout.addRow("检测脚本", self.edit_detector_script)
        form_layout.addRow("模型", self.edit_model)
        form_layout.addRow("置信度", self.spin_conf)
        form_layout.addRow("输出JSON", self.edit_output)
        left_layout.addWidget(form_box)

        row1 = QHBoxLayout()
        row1.setSpacing(8)
        self.btn_choose_image = QPushButton("选择图片")
        self.btn_use_current = QPushButton("使用当前图")
        self.btn_choose_script = QPushButton("选择脚本")
        row1.addWidget(self.btn_choose_image)
        row1.addWidget(self.btn_use_current)
        row1.addWidget(self.btn_choose_script)
        left_layout.addLayout(row1)

        row2 = QHBoxLayout()
        row2.setSpacing(8)
        self.btn_choose_model = QPushButton("选择模型")
        self.btn_detect = QPushButton("模型出点")
        self.btn_clear = QPushButton("清空点")
        row2.addWidget(self.btn_choose_model)
        row2.addWidget(self.btn_detect)
        row2.addWidget(self.btn_clear)
        left_layout.addLayout(row2)

        row3 = QHBoxLayout()
        row3.setSpacing(8)
        self.btn_choose_output = QPushButton("选择输出")
        self.btn_save = QPushButton("保存JSON")
        row3.addWidget(self.btn_choose_output)
        row3.addWidget(self.btn_save)
        left_layout.addLayout(row3)

        self.toolbar = QToolBar()
        self.toolbar.setMovable(False)
        self.action_group = []
        for text, mode in [
            ("选择", "select"),
            ("新增", "add"),
            ("移动", "move"),
            ("删除", "delete"),
        ]:
            action = QAction(text, self)
            action.setCheckable(True)
            action.triggered.connect(lambda checked=False, m=mode: self._set_mode(m))
            self.toolbar.addAction(action)
            self.action_group.append((action, mode))
        self.action_group[0][0].setChecked(True)
        left_layout.addWidget(self.toolbar)

        info_box = QFrame()
        info_layout = QFormLayout(info_box)
        self.lbl_point_count = QLabel("0")
        self.lbl_selected = QLabel("-")
        self.lbl_xy = QLabel("-")
        info_layout.addRow("点数", self.lbl_point_count)
        info_layout.addRow("选中点", self.lbl_selected)
        info_layout.addRow("坐标", self.lbl_xy)
        left_layout.addWidget(info_box)

        self.lbl_help = QLabel(
            "快捷说明\n"
            "新增补点 / 移动拖拽 / 删除误检 / 滚轮缩放 / 右键平移 / 保存后继续流程"
        )
        self.lbl_help.setWordWrap(True)
        self.lbl_help.setStyleSheet("color:#6B7280;font-size:12px;")
        left_layout.addWidget(self.lbl_help)
        left_layout.addStretch(1)

        self.status_label = QLabel("请选择图片，然后点击“模型出点”。")
        left_layout.addWidget(self.status_label)
        self.save_status_label = QLabel("未修改")
        left_layout.addWidget(self.save_status_label)

        self.canvas = PointCanvas()
        self.splitter.addWidget(self.left_panel)
        self.splitter.addWidget(self.canvas)
        self.splitter.setSizes([340, 1080])
        self.save_state = AutoSaveStatusController(self.save_status_label, self.btn_save)

    def _connect_signals(self):
        self.btn_choose_image.clicked.connect(self.choose_image)
        self.btn_use_current.clicked.connect(self.use_current_image)
        self.btn_choose_script.clicked.connect(self.choose_detector_script)
        self.btn_choose_model.clicked.connect(self.choose_model)
        self.btn_choose_output.clicked.connect(self.choose_output)
        self.btn_detect.clicked.connect(self.detect_points)
        self.btn_clear.clicked.connect(self.clear_points)
        self.btn_save.clicked.connect(self.save_json)
        self.canvas.pointSelectionChanged.connect(self._on_point_selection_changed)
        self.canvas.pointCountChanged.connect(lambda count: self.lbl_point_count.setText(str(count)))
        self.canvas.statusMessage.connect(self.status_label.setText)
        self.canvas.annotationModified.connect(self._on_annotation_modified)

    def _toggle_left_panel(self):
        if self._left_panel_visible:
            self._left_panel_width = max(self.splitter.sizes()[0], 260)
            self.left_panel.hide()
            self.splitter.setSizes([0, 1])
            self.btn_toggle_sidebar.setText("展开侧栏")
            self._left_panel_visible = False
            return

        self.left_panel.show()
        self.splitter.setSizes([self._left_panel_width, max(self.width() - self._left_panel_width, 1)])
        self.btn_toggle_sidebar.setText("收起侧栏")
        self._left_panel_visible = True

    def _set_mode(self, mode: str):
        for action, action_mode in self.action_group:
            action.setChecked(action_mode == mode)
        self.canvas.set_mode(mode)
        self.status_label.setText(f"当前模式: {mode}")

    def _on_annotation_modified(self):
        self._has_unsaved_changes = True
        self.save_state.mark_dirty("未保存修改")
        self._auto_save_annotation()

    def set_image(self, image_bgr: np.ndarray, image_path: str = ""):
        self.image_bgr = image_bgr.copy()
        self.image_path = image_path
        self.canvas.set_image(self.image_bgr, image_path=image_path)
        self._has_unsaved_changes = False
        if image_path:
            self.edit_image.setText(image_path)
        if not self.edit_output.text().strip() and image_path:
            image_path_obj = Path(image_path)
            self.edit_output.setText(str(image_path_obj.with_name(f"{image_path_obj.stem}_points_anno.json")))
        self.status_label.setText("图片已加载。")
        self.save_state.mark_pristine("已加载，未修改")

    def use_current_image(self):
        parent = self.parent()
        if parent is None or getattr(parent, "_current_image", None) is None:
            QMessageBox.information(self, "提示", "主窗口当前没有已加载图片。")
            return
        self.set_image(parent._current_image, getattr(parent, "_current_file", "") or "")

    def choose_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择图片",
            str(Path.cwd()),
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All Files (*)",
        )
        if not path:
            return
        try:
            self.set_image(read_image_bgr(path), path)
        except Exception as exc:
            QMessageBox.critical(self, "打开图片失败", str(exc))

    def choose_detector_script(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择 sew_point_detector.py", str(DEFAULT_DETECTOR_SCRIPT.parent), "Python (*.py)")
        if path:
            self.edit_detector_script.setText(path)

    def choose_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择 ONNX 模型", str(DEFAULT_MODEL_PATH.parent), "ONNX (*.onnx);;All Files (*)")
        if path:
            self.edit_model.setText(path)

    def choose_output(self):
        init = self.edit_output.text().strip()
        path, _ = QFileDialog.getSaveFileName(
            self,
            "保存 JSON",
            init or str(Path.cwd() / "stitch_points_anno.json"),
            "JSON (*.json)",
        )
        if path:
            self.edit_output.setText(path)
            self._auto_save_annotation()

    def clear_points(self):
        self.canvas.set_points([])
        self.status_label.setText("已清空当前点。")
        self._has_unsaved_changes = True
        self.save_state.mark_dirty("未保存修改")
        self._auto_save_annotation()

    def detect_points(self):
        if self.image_bgr is None:
            QMessageBox.warning(self, "提示", "请先选择图片。")
            return
        detector_script = self.edit_detector_script.text().strip()
        model_path = self.edit_model.text().strip()
        if not detector_script or not Path(detector_script).exists():
            QMessageBox.warning(self, "提示", "请先选择有效的 sew_point_detector.py。")
            return
        if not model_path or not Path(model_path).exists():
            QMessageBox.warning(self, "提示", "请先选择有效的 ONNX 模型。")
            return

        self.status_label.setText("模型推理中，请稍候...")
        self.btn_detect.setEnabled(False)

        image_bgr = self.image_bgr
        conf = self.spin_conf.value()
        runner = self.detector_runner

        class _DetectWorker(QThread):
            done = Signal(object)
            error = Signal(str)

            def __init__(self, img, ds, mp, c):
                super().__init__()
                self._img = img
                self._ds = ds
                self._mp = mp
                self._c = c

            def run(self):
                try:
                    pts = runner.detect(self._img, detector_script_path=self._ds,
                                        model_path=self._mp, conf=self._c)
                    self.done.emit(pts)
                except Exception as exc:
                    self.error.emit(str(exc))

        self._detect_worker = _DetectWorker(image_bgr, detector_script, model_path, conf)
        self._detect_worker.done.connect(self._on_detect_done)
        self._detect_worker.error.connect(self._on_detect_error)
        self._detect_worker.start()

    def _on_detect_done(self, points):
        self.canvas.set_points(points)
        self.status_label.setText(f"模型出点完成，共 {len(points)} 个点。")
        self.btn_detect.setEnabled(True)
        self._has_unsaved_changes = True
        self.save_state.mark_dirty("未保存修改")
        self._auto_save_annotation()

    def _on_detect_error(self, msg):
        QMessageBox.critical(self, "模型出点失败", msg)
        self.status_label.setText("模型出点失败。")
        self.btn_detect.setEnabled(True)

    def _on_point_selection_changed(self, point):
        if point is None:
            self.lbl_selected.setText("-")
            self.lbl_xy.setText("-")
            return
        self.lbl_selected.setText(str(point["id"]))
        self.lbl_xy.setText(f"({point['x']:.1f}, {point['y']:.1f})")

    def save_json(self):
        self._save_annotation(silent=False)

    def _save_annotation(self, silent: bool = True) -> bool:
        if self.image_bgr is None:
            if not silent:
                QMessageBox.warning(self, "提示", "没有可保存的图片上下文。")
            return False
        output_path = self.edit_output.text().strip()
        if not output_path:
            self.save_state.mark_dirty("未保存：请先设置输出路径")
            if not silent:
                QMessageBox.warning(self, "提示", "请先设置输出 JSON 路径。")
            return False

        sample_id = Path(output_path).stem
        annotation = make_empty_master_annotation(
            image_path=self.edit_image.text().strip(),
            width=int(self.image_bgr.shape[1]),
            height=int(self.image_bgr.shape[0]),
            sample_id=sample_id,
        )
        annotation["points"] = [
            {
                "id": int(point["id"]),
                "x": float(point["x"]),
                "y": float(point["y"]),
                "score": float(point.get("score", 1.0)),
                "source": point.get("source", "manual"),
            }
            for point in sorted(self.canvas.points, key=lambda item: int(item["id"]))
        ]
        annotation["segments"] = []
        annotation["metadata"] = {
            "source": "labeling_ui_point_annotation_dialog",
            "detector_script": self.edit_detector_script.text().strip(),
            "model_path": self.edit_model.text().strip(),
            "conf": float(self.spin_conf.value()),
            "point_count": len(self.canvas.points),
        }
        self.save_state.mark_saving(auto=silent)
        try:
            write_json(output_path, annotation)
        except Exception as exc:
            self.save_state.mark_error(f"保存失败：{exc}")
            if not silent:
                QMessageBox.critical(self, "保存失败", str(exc))
            return False
        self._has_unsaved_changes = False
        self.save_state.mark_saved(output_path, auto=silent)
        self.status_label.setText(f"已保存: {output_path}")
        return True

    def _auto_save_annotation(self):
        if self.edit_output.text().strip():
            self._save_annotation(silent=True)

    def shutdown(self) -> None:
        worker = getattr(self, "_detect_worker", None)
        if worker is None or not worker.isRunning():
            return
        worker.requestInterruption()
        worker.wait()
