from __future__ import annotations

from pathlib import Path
from typing import Protocol

from ..document import AnnotationDocument, AnnotationLayer


class AnnotationAdapter(Protocol):
    format_key: str
    display_name: str

    def can_load(self, path: Path) -> bool:
        ...

    def load(self, image_path: Path, annotation_path: Path | None) -> AnnotationDocument:
        ...

    def dump(self, document: AnnotationDocument, output_path: Path) -> None:
        ...

    def default_layers(self) -> list[AnnotationLayer]:
        ...
