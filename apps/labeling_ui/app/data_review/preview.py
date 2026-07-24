"""Preview adapter seam for the generic dataset review page."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np

from ..annotation.adapters.cabf import CabfAnnotationAdapter
from ..annotation.document import AnnotationDocument, ImageSize


class ReviewPreviewAdapter(Protocol):
    display_name: str

    def load(
        self,
        image_path: Path,
        annotation_path: Path | None,
        image_bgr: np.ndarray,
    ) -> AnnotationDocument:
        ...

    def summarize(self, document: AnnotationDocument) -> dict[str, str]:
        ...


class ImageOnlyPreviewAdapter:
    """Default adapter: review images without assuming an annotation format."""

    display_name = "图片"

    def load(
        self,
        image_path: Path,
        annotation_path: Path | None,
        image_bgr: np.ndarray,
    ) -> AnnotationDocument:
        height, width = image_bgr.shape[:2]
        return AnnotationDocument(
            image_path=str(image_path),
            image_size=ImageSize(width=width, height=height),
            layers=[],
        )

    def summarize(self, document: AnnotationDocument) -> dict[str, str]:
        return {}


class CabfPointEdgePreviewAdapter:
    """CAB-F extension that adds master point/edge overlays to generic review."""

    display_name = "CAB-F 点边"

    def __init__(self) -> None:
        self._adapter = CabfAnnotationAdapter()

    def load(
        self,
        image_path: Path,
        annotation_path: Path | None,
        image_bgr: np.ndarray,
    ) -> AnnotationDocument:
        if annotation_path is not None:
            return self._adapter.load(image_path, annotation_path)
        height, width = image_bgr.shape[:2]
        return self._adapter.new_empty(image_path, width=width, height=height)

    def summarize(self, document: AnnotationDocument) -> dict[str, str]:
        points = document.get_layer("points")
        edges = document.get_layer("edges")
        return {
            "点": str(len(points.shapes) if points is not None else 0),
            "边": str(len(edges.shapes) if edges is not None else 0),
        }
