import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtWidgets import QApplication

from apps.labeling_ui.app.annotation.canvas import AnnotationCanvas
from apps.labeling_ui.app.point_filter_dialog import (
    PointFilterCanvas,
    StitchPointFilterDialog,
    load_annotation_document_from_json,
)


def test_point_filter_loads_annotation_document_from_cabf_json(tmp_path):
    image_path = tmp_path / "sample.png"
    json_path = tmp_path / "sample.json"
    json_path.write_text(
        json.dumps(
            {
                "sample_id": "sample",
                "image_path": "sample.png",
                "image_size": {"width": 200, "height": 120},
                "points": [
                    {"id": 0, "x": 10, "y": 20},
                    {"id": 1, "x": 30, "y": 40},
                ],
                "edges": [
                    {"edge_id": "edge_0001", "src": 0, "dst": 1},
                ],
            }
        ),
        encoding="utf-8",
    )

    document = load_annotation_document_from_json(json_path, image_path=image_path)

    assert document.get_layer("points").shapes[0].geometry == {"x": 10.0, "y": 20.0}
    assert document.get_layer("edges").shapes[0].geometry["refs"] == ["0", "1"]


def test_point_filter_canvas_uses_generic_annotation_canvas(tmp_path):
    _app = QApplication.instance() or QApplication([])
    image_path = tmp_path / "sample.png"
    json_path = tmp_path / "sample.json"
    json_path.write_text(
        json.dumps(
            {
                "sample_id": "sample",
                "image_size": {"width": 20, "height": 10},
                "points": [{"id": 0, "x": 2, "y": 3}],
                "edges": [],
            }
        ),
        encoding="utf-8",
    )
    document = load_annotation_document_from_json(json_path, image_path=image_path)
    canvas = PointFilterCanvas()

    canvas.set_document(np.zeros((10, 20, 3), dtype=np.uint8), document)
    canvas.set_layer_visible("points", False)

    assert isinstance(canvas, AnnotationCanvas)
    assert canvas.document is document
    assert canvas.visible_layer_keys() == ["edges"]


def test_point_filter_apply_uses_input_dir_when_no_samples_saved(tmp_path):
    _app = QApplication.instance() or QApplication([])
    image_dir = tmp_path / "pending_filter"
    save_dir = tmp_path / "filtered_keep"
    image_dir.mkdir()
    dialog = StitchPointFilterDialog()
    emitted: list[str] = []
    dialog.applyRequested.connect(emitted.append)
    dialog.configure_paths(
        mode="unlabeled",
        image_dir=str(image_dir),
        label_dir="",
        save_dir=str(save_dir),
        auto_load=False,
    )

    dialog.apply_to_workflow()

    assert emitted == [str(image_dir)]
