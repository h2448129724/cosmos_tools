from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtWidgets import QApplication

from apps.labeling_ui.app.main_window import MainWindow
from apps.labeling_ui.app.tools.label_visualization_page import (
    LabelVisualizationPage,
    collect_visualization_items,
)


def get_qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def write_image(path: Path) -> None:
    image = np.zeros((20, 30, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    path.write_bytes(encoded.tobytes())


def test_label_visualization_page_is_registered_as_generic_tool():
    get_qapp()

    window = MainWindow()

    page_titles = [window._tool_list.item(i).text() for i in range(window._tool_list.count())]
    assert "标签可视化" in page_titles


def test_collect_visualization_items_pairs_same_stem_labels(tmp_path):
    image_dir = tmp_path / "images"
    label_dir = tmp_path / "labels"
    image_dir.mkdir()
    label_dir.mkdir()
    write_image(image_dir / "sample.png")
    (label_dir / "sample.json").write_text("{}", encoding="utf-8")

    items = collect_visualization_items(image_dir, label_dir)

    assert len(items) == 1
    assert items[0].image_path == image_dir / "sample.png"
    assert items[0].label_path == label_dir / "sample.json"


def test_label_visualization_loads_annotation_document(tmp_path):
    get_qapp()
    image_dir = tmp_path / "images"
    label_dir = tmp_path / "labels"
    image_dir.mkdir()
    label_dir.mkdir()
    image_path = image_dir / "sample.png"
    label_path = label_dir / "sample.json"
    write_image(image_path)
    label_path.write_text(
        json.dumps(
            {
                "sample_id": "sample",
                "image_path": str(image_path),
                "image_size": {"width": 30, "height": 20},
                "points": [{"id": 0, "x": 5, "y": 6}],
                "edges": [],
            }
        ),
        encoding="utf-8",
    )
    page = LabelVisualizationPage(main_window=None)

    page.load_paths(image_dir, label_dir)

    assert page.canvas.document.image_path == str(image_path)
    assert page.canvas.document.get_layer("points").shapes[0].geometry == {"x": 5.0, "y": 6.0}
    assert "点 1" in page.status_label.text()
