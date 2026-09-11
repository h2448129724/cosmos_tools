"""Imperative annotation-source discovery used by CAB-F shells."""
from __future__ import annotations

import json
from pathlib import Path

from .annotation_source_core import (
    AnnotationDirectoryFacts,
    AnnotationFileFacts,
    count_labelme_points,
    facts_from_json,
    resolve_annotation_source,
)


def _count_labelme_points(data: dict) -> int:
    return count_labelme_points(data)


def _scan_directory(path_text: str) -> AnnotationDirectoryFacts:
    path = Path(path_text)
    if not path.is_dir():
        return AnnotationDirectoryFacts(path_text, False)
    files: list[AnnotationFileFacts] = []
    for json_path in path.glob("*.json"):
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError, OSError):
            files.append(AnnotationFileFacts(parseable=False))
            continue
        files.append(facts_from_json(data))
    return AnnotationDirectoryFacts(path_text, True, tuple(files))


def _has_usable_point_json(path_text: str) -> bool:
    """Compatibility helper retained for existing callers/tests."""

    from .annotation_source_core import directory_is_eligible

    return directory_is_eligible(_scan_directory(path_text))


def resolve_edge_annotation_dir(cfg: dict, requested: str) -> tuple[str, str | None]:
    """Choose the first usable edge input, preserving CAB-F fallback rules."""
    requested_facts = _scan_directory(requested) if requested else None
    master_path = str(cfg.get("master_annotations_dir", "") or "")
    prediction_path = str(cfg.get("point_predictions_dir", "") or "")
    resolution = resolve_annotation_source(
        requested=requested_facts,
        master=_scan_directory(master_path) if master_path else None,
        prediction=_scan_directory(prediction_path) if prediction_path else None,
    )
    return resolution.as_tuple()


__all__ = ["resolve_edge_annotation_dir"]
