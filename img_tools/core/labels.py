"""Minimal YOLO detection/segmentation label parser and renderer."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def render_yolo_labels(image: np.ndarray, label_path: str | Path, *, class_names: list[str] | None = None) -> np.ndarray:
    """Render normalized YOLO boxes or polygons onto a copy of an image."""
    result = image.copy()
    height, width = result.shape[:2]
    for raw in Path(label_path).read_text(encoding="utf-8").splitlines():
        values = raw.split()
        if len(values) < 5:
            continue
        class_id, coords = int(float(values[0])), list(map(float, values[1:]))
        color = (0, 220, 0)
        name = class_names[class_id] if class_names and 0 <= class_id < len(class_names) else str(class_id)
        if len(coords) == 4:
            cx, cy, box_w, box_h = coords
            x1, y1 = round((cx - box_w / 2) * width), round((cy - box_h / 2) * height)
            x2, y2 = round((cx + box_w / 2) * width), round((cy + box_h / 2) * height)
            cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
            cv2.putText(result, name, (x1, max(14, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        elif len(coords) >= 6 and len(coords) % 2 == 0:
            points = np.asarray([(round(coords[index] * width), round(coords[index + 1] * height)) for index in range(0, len(coords), 2)], dtype=np.int32)
            cv2.polylines(result, [points], True, color, 2)
            cv2.putText(result, name, tuple(points[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return result
