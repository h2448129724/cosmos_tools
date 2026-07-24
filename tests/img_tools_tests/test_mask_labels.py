import numpy as np

from img_tools.core.labels import render_yolo_labels
from img_tools.core.mask import polygon_mask


def test_polygon_mask_fills_inside_area():
    mask = polygon_mask(10, 10, [[(1, 1), (8, 1), (1, 8)]])
    assert mask[2, 2] == 255
    assert mask[9, 9] == 0


def test_yolo_renderer_draws_box(tmp_path):
    label = tmp_path / "sample.txt"
    label.write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
    result = render_yolo_labels(np.zeros((20, 20, 3), dtype=np.uint8), label)
    assert result.sum() > 0
