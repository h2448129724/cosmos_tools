"""Compatibility entry point for dataset review.

The reusable implementation lives in :mod:`data_review`.  CAB-F contributes
only its annotation preview adapter here.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Signal

from .annotation.adapters.cabf import CabfAnnotationAdapter
from .annotation.document import AnnotationDocument
from .data_review import CabfPointEdgePreviewAdapter, DatasetReviewDialog, ReviewCanvas
from .data_review.model import (
    ReviewItem,
    ReviewSpec,
    collect_review_items,
    make_review_trash_dir,
    move_file_safe,
    move_review_item,
)


FilterItem = ReviewItem
PointFilterCanvas = ReviewCanvas


def collect_filter_items(
    image_dir: Path,
    label_dir: Path | None = None,
    *,
    require_label: bool = True,
) -> list[ReviewItem]:
    return collect_review_items(
        ReviewSpec(
            image_source=image_dir,
            annotation_source=label_dir,
            require_annotation=require_label,
        )
    )


def make_trash_dir(project_root: Path) -> Path:
    return make_review_trash_dir(project_root)


def move_item_files(item: ReviewItem, dest_dir: Path) -> tuple[Path, Path | None]:
    return move_review_item(item, dest_dir)


def load_annotation_document_from_json(
    json_path: Path,
    *,
    image_path: Path | None = None,
) -> AnnotationDocument:
    image_path = image_path or json_path.with_suffix(".png")
    return CabfAnnotationAdapter().load(image_path, json_path)


def _points_edges_from_document(document: AnnotationDocument) -> tuple[list[dict], list[dict]]:
    point_layer = document.get_layer("points")
    edge_layer = document.get_layer("edges")
    points = [
        {
            "id": int(shape.id) if str(shape.id).isdigit() else index,
            "x": float(shape.geometry.get("x", 0)),
            "y": float(shape.geometry.get("y", 0)),
            **shape.attrs,
        }
        for index, shape in enumerate(point_layer.shapes if point_layer is not None else [])
    ]
    edges: list[dict] = []
    for index, shape in enumerate(edge_layer.shapes if edge_layer is not None else []):
        refs = shape.geometry.get("refs", [])
        if not isinstance(refs, list) or len(refs) < 2:
            continue
        edges.append(
            {
                "edge_id": shape.id or f"edge_{index + 1:04d}",
                "src": int(refs[0]),
                "dst": int(refs[1]),
                **shape.attrs,
            }
        )
    return points, edges


def load_annotation_from_json(json_path: Path) -> tuple[list[dict], list[dict]]:
    return _points_edges_from_document(load_annotation_document_from_json(json_path))


class StitchPointFilterDialog(DatasetReviewDialog):
    """Thin CAB-F adapter retained for existing launchers and scripts."""

    applyRequested = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(
            parent,
            preview_adapter=CabfPointEdgePreviewAdapter(),
            title="CAB-F 数据审阅",
        )
        self.reviewCompleted.connect(lambda result: self.applyRequested.emit(str(result.active_image_dir)))


__all__ = [
    "FilterItem",
    "PointFilterCanvas",
    "StitchPointFilterDialog",
    "collect_filter_items",
    "load_annotation_document_from_json",
    "load_annotation_from_json",
    "make_trash_dir",
    "move_file_safe",
    "move_item_files",
]
