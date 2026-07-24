import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from apps.labeling_ui.app.annotation.canvas import AnnotationCanvas
from apps.labeling_ui.app.annotation.document import AnnotationDocument, AnnotationLayer, ImageSize
from apps.labeling_ui.app.annotation.layer_panel import LayerVisibilityPanel


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_layer_panel_toggles_canvas_layer_visibility(qapp):
    canvas = AnnotationCanvas()
    canvas.set_document(
        AnnotationDocument(
            image_path="sample.png",
            image_size=ImageSize(width=20, height=10),
            layers=[
                AnnotationLayer(key="points", title="点", shape_type="point", source_field="points"),
                AnnotationLayer(key="edges", title="线", shape_type="line", source_field="edges"),
            ],
        )
    )
    panel = LayerVisibilityPanel()

    panel.set_canvas(canvas)
    panel.checkbox_for("edges").setChecked(False)

    assert canvas.visible_layer_keys() == ["points"]
