from __future__ import annotations

import os

from .constants import MASTER_SCHEMA_VERSION, POINT_LABEL_ALIASES


def make_empty_master_annotation(image_path: str, width: int, height: int, sample_id: str) -> dict:
    return {
        "schema_version": MASTER_SCHEMA_VERSION,
        "sample_id": sample_id,
        "image_path": image_path,
        "image_size": {"width": int(width), "height": int(height)},
        "roi": None,
        "spacing_hint": None,
        "points": [],
        "edges": [],
        "segments": [],
        "metadata": {},
    }


def is_labelme_point_annotation(data: dict) -> bool:
    if not isinstance(data, dict):
        return False
    if "shapes" not in data:
        return False
    return isinstance(data.get("shapes"), list)


def load_labelme_points(data: dict) -> list[dict]:
    shapes = data.get("shapes", [])
    points = []
    for shape in shapes:
        if not isinstance(shape, dict):
            continue
        if shape.get("shape_type") != "point":
            continue
        if str(shape.get("label", "")).strip().lower() not in POINT_LABEL_ALIASES:
            continue
        raw_points = shape.get("points", [])
        if not raw_points or len(raw_points[0]) < 2:
            continue
        xy = raw_points[0]
        points.append(
            {
                "x": float(xy[0]),
                "y": float(xy[1]),
                "score": float(shape.get("score", 1.0) or 1.0),
                "source": "labelme_point",
            }
        )
    return points


def convert_labelme_to_master(data: dict, image_path: str, sample_id: str) -> dict:
    width = int(data.get("imageWidth", 0) or 256)
    height = int(data.get("imageHeight", 0) or 256)
    annotation = make_empty_master_annotation(image_path=image_path, width=width, height=height, sample_id=sample_id)
    points = load_labelme_points(data)
    annotation["points"] = [
        {
            "id": idx,
            "x": float(point["x"]),
            "y": float(point["y"]),
            "score": float(point.get("score", 1.0)),
            "source": point.get("source", "labelme_point"),
        }
        for idx, point in enumerate(points)
    ]
    annotation["metadata"] = {
        "source": "labelme_point",
        "origin_format": "labelme",
    }
    return annotation


def master_to_labelme(master_annotation: dict) -> dict:
    image_path = str(master_annotation.get("image_path", ""))
    width = int(master_annotation.get("image_size", {}).get("width", 256) or 256)
    height = int(master_annotation.get("image_size", {}).get("height", 256) or 256)
    shapes = []
    for point in master_annotation.get("points", []):
        shapes.append(
            {
                "label": "sew",
                "points": [[float(point["x"]), float(point["y"])]],
                "group_id": None,
                "description": "",
                "shape_type": "point",
                "flags": {},
                "score": float(point.get("score", 1.0)),
            }
        )
    return {
        "version": "5.0.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": os.path.basename(image_path),
        "imageData": None,
        "imageHeight": height,
        "imageWidth": width,
    }
