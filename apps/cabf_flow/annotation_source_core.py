"""Pure decisions for choosing CAB-F point annotation sources.

The shell in :mod:`annotation_source` is intentionally limited to directory
scanning and JSON decoding.  This module receives immutable facts and decides
whether a candidate is usable and which fallback wins.  Keeping this policy
free of ``Path`` and JSON I/O makes the requested/master/prediction precedence
easy to test and reuse from non-CLI callers.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

from shared.cabf_common.cabf.constants import POINT_LABEL_ALIASES

__all__ = [
    "AnnotationFileFacts",
    "AnnotationDirectoryFacts",
    "AnnotationResolution",
    "canonical_point_label",
    "is_point_label",
    "count_labelme_points",
    "facts_from_json",
    "directory_is_eligible",
    "resolve_annotation_source",
]


def canonical_point_label(label: object) -> str | None:
    """Return the canonical label for a point annotation, if recognised.

    ``keypoint`` is a historical alias for ``sew`` and must follow the same
    path as the canonical spelling used by the CAB-F schema.
    """

    normalised = str(label or "").strip().lower()
    return "sew" if normalised in POINT_LABEL_ALIASES else None


def is_point_label(label: object) -> bool:
    return canonical_point_label(label) is not None


def _has_xy(value: object) -> bool:
    if isinstance(value, Mapping):
        coordinates = (value.get("x"), value.get("y"))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if len(value) < 2:
            return False
        coordinates = (value[0], value[1])
    else:
        return False
    try:
        return all(math.isfinite(float(coordinate)) for coordinate in coordinates)
    except (TypeError, ValueError, OverflowError):
        return False


def _valid_point_count(values: object) -> int:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
        return 0
    return sum(1 for value in values if _has_xy(value))


def count_labelme_points(data: Mapping[str, Any] | object) -> int:
    """Count valid LabelMe point shapes using the shared label aliases."""

    if not isinstance(data, Mapping):
        return 0
    shapes = data.get("shapes", [])
    if not isinstance(shapes, Sequence) or isinstance(shapes, (str, bytes, bytearray)):
        return 0
    count = 0
    for shape in shapes:
        if not isinstance(shape, Mapping) or shape.get("shape_type") != "point":
            continue
        if not is_point_label(shape.get("label", "")):
            continue
        raw_points = shape.get("points", [])
        if not isinstance(raw_points, Sequence) or isinstance(raw_points, (str, bytes, bytearray)):
            continue
        if raw_points and _has_xy(raw_points[0]):
            count += 1
    return count


@dataclass(frozen=True, slots=True)
class AnnotationFileFacts:
    """Facts extracted from one JSON file by the imperative shell."""

    point_count: int = 0
    labelme_point_count: int = 0
    parseable: bool = True

    @property
    def usable_point_count(self) -> int:
        return max(0, int(self.point_count), int(self.labelme_point_count))


def facts_from_json(data: object) -> AnnotationFileFacts:
    """Build facts from decoded JSON without touching the filesystem."""

    if not isinstance(data, Mapping):
        return AnnotationFileFacts(parseable=False)
    return AnnotationFileFacts(
        point_count=_valid_point_count(data.get("points", [])),
        labelme_point_count=count_labelme_points(data),
    )


@dataclass(frozen=True, slots=True)
class AnnotationDirectoryFacts:
    """Directory scan facts supplied by the shell."""

    path: str
    is_directory: bool = False
    files: tuple[AnnotationFileFacts, ...] = ()


def directory_is_eligible(candidate: AnnotationDirectoryFacts | None) -> bool:
    if candidate is None or not candidate.is_directory:
        return False
    return any(file.parseable and file.usable_point_count >= 2 for file in candidate.files)


@dataclass(frozen=True, slots=True)
class AnnotationResolution:
    path: str
    note: str | None = None

    def as_tuple(self) -> tuple[str, str | None]:
        return self.path, self.note


def resolve_annotation_source(
    *,
    requested: AnnotationDirectoryFacts | None,
    master: AnnotationDirectoryFacts | None,
    prediction: AnnotationDirectoryFacts | None,
) -> AnnotationResolution:
    """Apply requested → master → prediction fallback policy.

    An existing usable requested directory wins.  Otherwise the configured
    master directory is preferred over point predictions, regardless of
    whether the requested path exists; an existing but empty requested path
    is retained only when no fallback is usable.
    """

    requested_eligible = directory_is_eligible(requested)
    if requested_eligible:
        return AnnotationResolution(requested.path)

    if directory_is_eligible(master) and master is not None:
        if requested is not None and requested.path and requested.path != master.path:
            return AnnotationResolution(
                master.path,
                f"annotation_dir fallback: {requested.path} -> {master.path}",
            )
        return AnnotationResolution(master.path)
    if directory_is_eligible(prediction) and prediction is not None:
        source = (master.path if master is not None and master.path else requested.path if requested is not None else "")
        return AnnotationResolution(
            prediction.path,
            f"annotation_dir fallback: {source} -> {prediction.path}",
        )
    return AnnotationResolution(
        requested.path if requested is not None and requested.path else master.path if master is not None else prediction.path if prediction is not None else "",
    )
