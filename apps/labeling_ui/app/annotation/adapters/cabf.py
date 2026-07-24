from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from cabf import (
    MASTER_SCHEMA_VERSION,
    make_empty_master_annotation,
    normalize_master_annotation,
    read_json,
    write_json,
)

from ..document import AnnotationDocument, AnnotationLayer, AnnotationShape, ImageSize, LayerStyle


class CabfAnnotationAdapter:
    format_key = "cabf"
    display_name = "CAB-F Master Annotation"

    def can_load(self, path: Path) -> bool:
        return path.suffix.lower() == ".json"

    def default_layers(self) -> list[AnnotationLayer]:
        return [
            AnnotationLayer(
                key="points",
                title="点",
                shape_type="point",
                source_field="points",
                visible=True,
                editable=True,
                style=LayerStyle(color="#50ff50"),
            ),
            AnnotationLayer(
                key="edges",
                title="线",
                shape_type="line",
                source_field="edges",
                visible=True,
                editable=True,
                style=LayerStyle(color="#ffb400"),
            ),
            AnnotationLayer(
                key="roi",
                title="ROI",
                shape_type="box",
                source_field="roi",
                visible=False,
                editable=True,
                style=LayerStyle(color="#0078d7"),
            ),
            AnnotationLayer(
                key="segments",
                title="多边形",
                shape_type="polygon",
                source_field="segments",
                visible=False,
                editable=True,
                style=LayerStyle(color="#27ae60"),
            ),
        ]

    def new_empty(self, image_path: Path, *, width: int = 256, height: int = 256) -> AnnotationDocument:
        return AnnotationDocument(
            image_path=str(image_path),
            image_size=ImageSize(width=int(width), height=int(height)),
            layers=self.default_layers(),
            metadata={
                "cabf": {},
                "cabf_master": {
                    "schema_version": MASTER_SCHEMA_VERSION,
                    "sample_id": image_path.stem,
                    "spacing_hint": None,
                },
            },
        )

    def load(self, image_path: Path, annotation_path: Path | None) -> AnnotationDocument:
        if annotation_path is None or not annotation_path.exists():
            return self.new_empty(image_path)

        raw = read_json(annotation_path)
        return self.from_master_dict(raw, image_path=image_path, sample_id=annotation_path.stem)

    def from_master_dict(self, raw: dict, *, image_path: Path, sample_id: str | None = None) -> AnnotationDocument:
        normalized, _issues = normalize_master_annotation(
            raw,
            sample_id=sample_id or image_path.stem,
            image_path=str(image_path),
        )
        width = int(normalized.get("image_size", {}).get("width", 256) or 256)
        height = int(normalized.get("image_size", {}).get("height", 256) or 256)
        document = self.new_empty(image_path, width=width, height=height)
        document.metadata["cabf"] = deepcopy(normalized.get("metadata", {}))
        document.metadata["cabf_master"] = {
            "schema_version": normalized.get("schema_version", MASTER_SCHEMA_VERSION),
            "sample_id": normalized.get("sample_id", image_path.stem),
            "spacing_hint": normalized.get("spacing_hint"),
        }

        self._replace_layer_shapes(document, "points", self._point_shapes(normalized.get("points", [])))
        self._replace_layer_shapes(document, "edges", self._edge_shapes(normalized.get("edges", []), normalized.get("points", [])))
        self._replace_layer_shapes(document, "roi", self._roi_shapes(normalized.get("roi")))
        self._replace_layer_shapes(document, "segments", self._segment_shapes(normalized.get("segments", [])))
        return document

    def dump(self, document: AnnotationDocument, output_path: Path) -> None:
        write_json(output_path, self.to_master_dict(document, sample_id=output_path.stem))

    def to_master_dict(self, document: AnnotationDocument, *, sample_id: str | None = None) -> dict:
        master_meta = document.metadata.get("cabf_master", {}) if isinstance(document.metadata, dict) else {}
        payload = make_empty_master_annotation(
            image_path=document.image_path,
            width=document.image_size.width,
            height=document.image_size.height,
            sample_id=str(master_meta.get("sample_id") or sample_id or Path(document.image_path).stem),
        )
        payload["schema_version"] = str(master_meta.get("schema_version") or MASTER_SCHEMA_VERSION)
        payload["spacing_hint"] = master_meta.get("spacing_hint")
        payload["points"] = self._dump_points(document.get_layer("points"))
        payload["edges"] = self._dump_edges(document.get_layer("edges"))
        payload["roi"] = self._dump_roi(document.get_layer("roi"))
        payload["segments"] = self._dump_segments(document.get_layer("segments"))
        metadata = deepcopy(document.metadata.get("cabf", {})) if isinstance(document.metadata.get("cabf", {}), dict) else {}
        metadata["source"] = "generic_annotation_adapter"
        metadata["point_count"] = len(payload["points"])
        metadata["edge_count"] = len(payload["edges"])
        payload["metadata"] = metadata
        return payload

    def point_shape(self, *, point_id: int | str, x: float, y: float, attrs: dict[str, Any] | None = None) -> AnnotationShape:
        return AnnotationShape(
            id=str(point_id),
            geometry={"x": float(x), "y": float(y)},
            attrs=dict(attrs or {}),
        )

    def edge_shape(self, *, edge_id: str, src: int | str, dst: int | str, attrs: dict[str, Any] | None = None) -> AnnotationShape:
        src_text = str(src)
        dst_text = str(dst)
        return AnnotationShape(
            id=str(edge_id),
            geometry={"refs": [src_text, dst_text], "points": []},
            attrs=dict(attrs or {}),
        )

    def box_shape(self, *, shape_id: str, x1: float, y1: float, x2: float, y2: float, attrs: dict[str, Any] | None = None) -> AnnotationShape:
        return AnnotationShape(
            id=str(shape_id),
            geometry={"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2)},
            attrs=dict(attrs or {}),
        )

    @staticmethod
    def _replace_layer_shapes(document: AnnotationDocument, key: str, shapes: list[AnnotationShape]) -> None:
        layer = document.get_layer(key)
        if layer is not None:
            layer.shapes = shapes

    def _point_shapes(self, points: list[dict]) -> list[AnnotationShape]:
        return [
            self.point_shape(
                point_id=point.get("id", index),
                x=point.get("x", 0),
                y=point.get("y", 0),
                attrs={
                    key: value
                    for key, value in point.items()
                    if key not in {"id", "x", "y"}
                },
            )
            for index, point in enumerate(points)
            if isinstance(point, dict)
        ]

    def _edge_shapes(self, edges: list[dict], points: list[dict]) -> list[AnnotationShape]:
        point_lookup = {str(point.get("id")): point for point in points if isinstance(point, dict)}
        shapes: list[AnnotationShape] = []
        for index, edge in enumerate(edges):
            if not isinstance(edge, dict):
                continue
            src = str(edge.get("src", ""))
            dst = str(edge.get("dst", ""))
            geometry: dict[str, Any] = {"refs": [src, dst], "points": []}
            if src in point_lookup and dst in point_lookup:
                geometry["points"] = [
                    [float(point_lookup[src]["x"]), float(point_lookup[src]["y"])],
                    [float(point_lookup[dst]["x"]), float(point_lookup[dst]["y"])],
                ]
            shapes.append(
                AnnotationShape(
                    id=str(edge.get("edge_id") or f"edge_{index + 1:04d}"),
                    geometry=geometry,
                    attrs={key: value for key, value in edge.items() if key not in {"edge_id", "src", "dst"}},
                )
            )
        return shapes

    def _roi_shapes(self, roi: Any) -> list[AnnotationShape]:
        if isinstance(roi, list) and len(roi) == 4 and all(isinstance(value, (int, float)) for value in roi):
            return [self.box_shape(shape_id="roi", x1=roi[0], y1=roi[1], x2=roi[2], y2=roi[3])]
        return []

    @staticmethod
    def _segment_shapes(segments: list[Any]) -> list[AnnotationShape]:
        shapes: list[AnnotationShape] = []
        for index, segment in enumerate(segments):
            if not isinstance(segment, dict):
                continue
            raw_points = segment.get("points", [])
            if not isinstance(raw_points, list):
                continue
            points = [
                [float(point[0]), float(point[1])]
                for point in raw_points
                if isinstance(point, (list, tuple)) and len(point) >= 2
            ]
            shapes.append(
                AnnotationShape(
                    id=str(segment.get("id") or f"segment_{index + 1:04d}"),
                    geometry={"points": points},
                    attrs={key: value for key, value in segment.items() if key not in {"id", "points"}},
                )
            )
        return shapes

    @staticmethod
    def _dump_points(layer: AnnotationLayer | None) -> list[dict]:
        if layer is None:
            return []
        points = []
        for index, shape in enumerate(layer.shapes):
            geometry = shape.geometry
            point_id = int(shape.id) if str(shape.id).isdigit() else index
            points.append(
                {
                    "id": point_id,
                    "x": float(geometry.get("x", 0)),
                    "y": float(geometry.get("y", 0)),
                    "score": float(shape.attrs.get("score", 1.0)),
                    "source": shape.attrs.get("source", "manual"),
                }
            )
        return sorted(points, key=lambda item: int(item["id"]))

    @staticmethod
    def _dump_edges(layer: AnnotationLayer | None) -> list[dict]:
        if layer is None:
            return []
        edges = []
        for index, shape in enumerate(layer.shapes):
            refs = shape.geometry.get("refs", [])
            if not isinstance(refs, list) or len(refs) < 2:
                continue
            edges.append(
                {
                    "edge_id": shape.id or f"edge_{index + 1:04d}",
                    "src": int(refs[0]),
                    "dst": int(refs[1]),
                    "label": int(shape.attrs.get("label", 1)),
                    "source": shape.attrs.get("source", "manual"),
                }
            )
        return edges

    @staticmethod
    def _dump_roi(layer: AnnotationLayer | None) -> list[float] | None:
        if layer is None or not layer.shapes:
            return None
        geometry = layer.shapes[0].geometry
        return [
            float(geometry.get("x1", 0)),
            float(geometry.get("y1", 0)),
            float(geometry.get("x2", 0)),
            float(geometry.get("y2", 0)),
        ]

    @staticmethod
    def _dump_segments(layer: AnnotationLayer | None) -> list[dict]:
        if layer is None:
            return []
        segments = []
        for shape in layer.shapes:
            item = dict(shape.attrs)
            item["id"] = shape.id
            item["points"] = shape.geometry.get("points", [])
            segments.append(item)
        return segments
