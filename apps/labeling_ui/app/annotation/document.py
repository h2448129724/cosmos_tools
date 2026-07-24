from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


ShapeType = Literal["point", "line", "box", "polygon"]


@dataclass
class ImageSize:
    width: int
    height: int


@dataclass
class LayerStyle:
    color: str = "#50ff50"


@dataclass
class AnnotationShape:
    id: str
    geometry: dict[str, Any]
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass
class AnnotationLayer:
    key: str
    title: str
    shape_type: ShapeType
    source_field: str
    shapes: list[AnnotationShape] = field(default_factory=list)
    visible: bool = True
    editable: bool = True
    style: LayerStyle = field(default_factory=LayerStyle)


@dataclass
class AnnotationDocument:
    image_path: str
    image_size: ImageSize
    layers: list[AnnotationLayer] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_layer(self, key: str) -> AnnotationLayer | None:
        for layer in self.layers:
            if layer.key == key:
                return layer
        return None
