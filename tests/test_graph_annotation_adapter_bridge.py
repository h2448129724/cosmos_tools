from pathlib import Path

import numpy as np
from PySide6.QtWidgets import QApplication

from apps.labeling_ui.app.graph_annotation_dialog import (
    EdgeAnnotationCanvas,
    FolderItem,
    StitchGraphEditorDialog,
    build_editor_document_from_payload,
    build_editor_payload_from_state,
    extract_editor_state_from_document,
)


def get_qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_graph_editor_builds_annotation_document_from_canvas_state():
    payload = build_editor_payload_from_state(
        image_path=Path("sample.png"),
        width=100,
        height=80,
        sample_id="sample",
        points=[
            {"id": 0, "x": 10, "y": 20, "score": 1.0, "source": "manual"},
            {"id": 1, "x": 30, "y": 40, "score": 0.9, "source": "model"},
        ],
        edges=[
            {"edge_id": "edge_0001", "src": 0, "dst": 1, "label": 1, "source": "manual"},
        ],
        origin_json="pred.json",
    )

    assert payload["points"][0]["x"] == 10.0
    assert payload["edges"][0]["src"] == 0
    assert payload["edges"][0]["dst"] == 1
    assert payload["metadata"]["origin_json"] == "pred.json"


def test_graph_editor_builds_document_from_loaded_payload():
    document = build_editor_document_from_payload(
        payload={
            "sample_id": "sample",
            "image_path": "sample.png",
            "image_size": {"width": 50, "height": 40},
            "points": [{"id": 0, "x": 7, "y": 8}],
            "edges": [],
        },
        image_path=Path("sample.png"),
    )

    assert document.image_size.width == 50
    assert document.get_layer("points").shapes[0].geometry == {"x": 7.0, "y": 8.0}


def test_graph_editor_extracts_canvas_state_from_document():
    document = build_editor_document_from_payload(
        payload={
            "sample_id": "sample",
            "image_path": "sample.png",
            "image_size": {"width": 50, "height": 40},
            "points": [
                {"id": 0, "x": 7, "y": 8, "score": 0.9, "source": "model"},
                {"id": 1, "x": 17, "y": 18, "score": 1.0, "source": "manual"},
            ],
            "edges": [{"edge_id": "edge_0001", "src": 0, "dst": 1, "label": 1, "source": "manual"}],
        },
        image_path=Path("sample.png"),
    )

    points, edges = extract_editor_state_from_document(document)

    assert points[0] == {"id": 0, "x": 7.0, "y": 8.0, "score": 0.9, "source": "model"}
    assert edges[0]["src"] == 0
    assert edges[0]["dst"] == 1


def test_graph_canvas_tracks_point_and_edge_layer_visibility():
    get_qapp()
    canvas = EdgeAnnotationCanvas()

    canvas.set_visible_layers({"points"})

    assert canvas.is_layer_visible("points") is True
    assert canvas.is_layer_visible("edges") is False

    canvas.set_layer_visible("edges", True)

    assert canvas.is_layer_visible("edges") is True

    canvas.set_overlay_visible(False)

    assert canvas.is_layer_visible("points") is False
    assert canvas.is_layer_visible("edges") is False


def test_graph_editor_layer_buttons_toggle_canvas_layers():
    get_qapp()
    dialog = StitchGraphEditorDialog()

    dialog.btn_toggle_edges.setChecked(False)

    assert dialog.canvas.is_layer_visible("points") is True
    assert dialog.canvas.is_layer_visible("edges") is False

    dialog.btn_toggle_points.setChecked(False)

    assert dialog.canvas.is_layer_visible("points") is False


def test_graph_editor_keeps_current_annotation_document(tmp_path, monkeypatch):
    get_qapp()
    image_path = tmp_path / "sample.png"
    dialog = StitchGraphEditorDialog()
    dialog.folder_items = [FolderItem(image_path=image_path, source_json_path=None)]

    dialog._on_jump_done(
        np.zeros((40, 50, 3), dtype=np.uint8),
        {
            "sample_id": "sample",
            "image_path": str(image_path),
            "image_size": {"width": 50, "height": 40},
            "points": [{"id": 0, "x": 7, "y": 8}],
            "edges": [],
        },
        dialog.folder_items[0],
        0,
    )

    assert dialog.current_document is not None
    assert dialog.current_document.get_layer("points").shapes[0].geometry == {"x": 7.0, "y": 8.0}

    dialog.current_index = 0
    monkeypatch.setattr(dialog, "jump_to_index", lambda index: None)
    dialog.reload_current_item()

    assert dialog.current_document is None
