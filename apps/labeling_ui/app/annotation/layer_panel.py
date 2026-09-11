from __future__ import annotations

from PySide6.QtWidgets import QCheckBox, QHBoxLayout, QLabel, QWidget

from .canvas import AnnotationCanvas
from cosmos_toolbox.ui.primitives import set_ui_role


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
        set_ui_role(label, "fieldLabel")
        label.setAccessibleName("图层可见性")
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
            checkbox.setAccessibleName(f"显示图层：{layer.title or layer.key}")
            checkbox.setToolTip(layer.source_field or f"切换图层 {layer.title or layer.key} 的可见性")
            checkbox.toggled.connect(lambda checked, key=layer.key: self._on_layer_toggled(key, checked))
            self._checkboxes[layer.key] = checkbox
            self._layout.addWidget(checkbox)
        self._layout.addStretch(1)

    def checkbox_for(self, layer_key: str) -> QCheckBox:
        return self._checkboxes[layer_key]

    def _on_layer_toggled(self, layer_key: str, checked: bool) -> None:
        if self._canvas is not None:
            self._canvas.set_layer_visible(layer_key, checked)
