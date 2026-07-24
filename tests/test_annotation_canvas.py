import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from apps.labeling_ui.app.annotation.canvas import AnnotationCanvas
from apps.labeling_ui.app.annotation.document import (
    AnnotationDocument,
    AnnotationLayer,
    ImageSize,
)


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _document() -> AnnotationDocument:
    return AnnotationDocument(
        image_path="sample.png",
        image_size=ImageSize(width=100, height=80),
        layers=[
            AnnotationLayer(key="points", title="Points", shape_type="point", source_field="points"),
            AnnotationLayer(key="edges", title="Edges", shape_type="line", source_field="edges"),
            AnnotationLayer(key="roi", title="ROI", shape_type="box", source_field="roi", visible=False),
        ],
    )


def test_canvas_controls_top_level_layer_visibility(qapp):
    canvas = AnnotationCanvas()
    canvas.set_document(_document())

    assert canvas.visible_layer_keys() == ["points", "edges"]

    canvas.set_layer_visible("edges", False)
    canvas.set_layer_visible("roi", True)

    assert canvas.visible_layer_keys() == ["points", "roi"]


def test_canvas_adds_points_to_editable_point_layer(qapp):
    canvas = AnnotationCanvas()
    canvas.set_document(_document())

    shape = canvas.add_point("points", 12.5, 20.0)

    layer = canvas.document.get_layer("points")
    assert shape.id == "0"
    assert shape.geometry == {"x": 12.5, "y": 20.0}
    assert layer.shapes == [shape]


def test_canvas_rejects_edits_to_non_editable_layer(qapp):
    document = _document()
    document.get_layer("points").editable = False
    canvas = AnnotationCanvas()
    canvas.set_document(document)

    with pytest.raises(ValueError, match="not editable"):
        canvas.add_point("points", 1, 2)
