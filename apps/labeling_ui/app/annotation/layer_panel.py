from __future__ import annotations

from PySide6.QtWidgets import QCheckBox, QHBoxLayout, QLabel, QWidget

from .canvas import AnnotationCanvas


class LayerVisibilityPanel(QWidget):
    """Checkbox panel for top-level annotation layer visibility."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._canvas: AnnotationCanvas | None = None
        self._checkboxes: dict[str, QCheckBox] = {}
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(8)
        label = QLabel("显示")
        label.setStyleSheet("color:#64748b;font-size:12px;font-weight:600;")
        self._layout.addWidget(label)
        self._layout.addStretch(1)

    def set_canvas(self, canvas: AnnotationCanvas) -> None:
        self._canvas = canvas
        self.rebuild()

    def rebuild(self) -> None:
        while self._layout.count() > 1:
            item = self._layout.takeAt(1)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._checkboxes.clear()
        if self._canvas is None:
            self._layout.addStretch(1)
            return

        for layer in self._canvas.document.layers:
            checkbox = QCheckBox(layer.title or layer.key)
            checkbox.setChecked(layer.visible)
            checkbox.setToolTip(layer.source_field)
            checkbox.toggled.connect(lambda checked, key=layer.key: self._on_layer_toggled(key, checked))
            self._checkboxes[layer.key] = checkbox
            self._layout.addWidget(checkbox)
        self._layout.addStretch(1)

    def checkbox_for(self, layer_key: str) -> QCheckBox:
        return self._checkboxes[layer_key]

    def _on_layer_toggled(self, layer_key: str, checked: bool) -> None:
        if self._canvas is not None:
            self._canvas.set_layer_visible(layer_key, checked)
