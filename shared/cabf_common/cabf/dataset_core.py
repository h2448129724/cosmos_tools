"""Pure CAB-F dataset assessment and export decision logic.

This module deliberately knows nothing about filesystems, image decoders, Qt,
or output operations.  The filesystem façade in :mod:`cabf.dataset` supplies
the facts collected from those effects and applies the returned decisions.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping

from .constants import MASTER_SCHEMA_VERSION
from .normalize import normalize_master_annotation
from .schema import master_to_labelme


class ExportDisposition(str, Enum):
    """The operation selected by an export decision."""

    EXPORT = "EXPORT"
    ERROR = "ERROR"
    SKIP = "SKIP"


@dataclass(frozen=True)
class SampleFacts:
    """Identity and source names for one paired dataset sample."""

    sample_id: str
    image_path: str = ""
    json_path: str = ""

    @property
    def annotation_path(self) -> str:
        """Compatibility alias for callers that call the JSON path annotation path."""

        return self.json_path


@dataclass(frozen=True)
class ImageSize:
    width: int
    height: int


@dataclass(frozen=True)
class SampleAssessment:
    """Normalized facts and deterministic validation findings for one sample."""

    sample_facts: SampleFacts
    normalized: dict[str, Any] | None
    point_count: int
    edge_count: int
    errors: tuple[str, ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def sample_id(self) -> str:
        return self.sample_facts.sample_id

    @property
    def image_path(self) -> str:
        return self.sample_facts.image_path

    @property
    def json_path(self) -> str:
        return self.sample_facts.json_path

    @property
    def counts(self) -> dict[str, int]:
        return {"point_count": self.point_count, "edge_count": self.edge_count}

    def to_report(self) -> dict[str, Any]:
        """Return the legacy sample-report dictionary used by dataset.py."""

        return {
            "sample_id": self.sample_id,
            "image_path": self.image_path,
            "json_path": self.json_path,
            "point_count": self.point_count,
            "edge_count": self.edge_count,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class ExportDecision:
    """Pure export outcome, including payload and facts for the shell operation."""

    disposition: ExportDisposition
    payload: dict[str, Any] | None
    operation_facts: dict[str, str]
    reasons: tuple[str, ...] = field(default_factory=tuple)

    @property
    def action(self) -> str:
        """String alias useful to callers that do not import the enum."""

        return self.disposition.value

    @property
    def operation(self) -> dict[str, str]:
        return self.operation_facts

    def to_dict(self) -> dict[str, Any]:
        return {
            "disposition": self.disposition.value,
            "payload": self.payload,
            "operation_facts": dict(self.operation_facts),
            "reasons": list(self.reasons),
        }


def _error_text(value: object) -> str:
    return str(value) if isinstance(value, BaseException) else str(value)


def _coerce_facts(sample_facts: SampleFacts | Mapping[str, Any] | str) -> SampleFacts:
    if isinstance(sample_facts, SampleFacts):
        return sample_facts
    if isinstance(sample_facts, Mapping):
        return SampleFacts(
            sample_id=str(sample_facts.get("sample_id", "")),
            image_path=str(sample_facts.get("image_path", "")),
            json_path=str(sample_facts.get("json_path", sample_facts.get("annotation_path", ""))),
        )
    return SampleFacts(sample_id=str(sample_facts))


def _coerce_size(actual_size: ImageSize | tuple[int, int] | Mapping[str, Any] | None) -> ImageSize | None:
    if actual_size is None:
        return None
    if isinstance(actual_size, ImageSize):
        return actual_size
    if isinstance(actual_size, Mapping):
        return ImageSize(int(actual_size["width"]), int(actual_size["height"]))
    return ImageSize(int(actual_size[0]), int(actual_size[1]))


def assess_sample(
    sample_facts: SampleFacts | Mapping[str, Any] | str,
    *,
    raw_annotation: Any = None,
    annotation_read_error: object | None = None,
    annotation_missing: bool = False,
    actual_size: ImageSize | tuple[int, int] | Mapping[str, Any] | None = None,
    image_read_error: object | None = None,
) -> SampleAssessment:
    """Normalize and assess one sample from explicitly supplied side-effect facts.

    Read errors are supplied by the shell rather than raised here.  The order
    matches the historical implementation: missing/read errors short-circuit
    normalization, then size and graph warnings are calculated.
    """

    facts = _coerce_facts(sample_facts)
    errors: list[str] = []
    warnings: list[str] = []
    if annotation_missing:
        errors.append("缺少标注文件")
        return SampleAssessment(facts, None, 0, 0, tuple(errors), tuple(warnings))
    if annotation_read_error is not None:
        errors.append(f"JSON 读取失败: {_error_text(annotation_read_error)}")
        return SampleAssessment(facts, None, 0, 0, tuple(errors), tuple(warnings))
    if image_read_error is not None:
        errors.append(f"图片读取失败: {_error_text(image_read_error)}")
        return SampleAssessment(facts, None, 0, 0, tuple(errors), tuple(warnings))

    # Keep the historical contract: malformed annotation structures that make
    # normalization itself raise propagate to the caller.  Recoverable schema
    # findings remain in ``normalize_issues`` and are represented as errors.
    normalized, normalize_issues = normalize_master_annotation(
        raw_annotation,
        sample_id=facts.sample_id,
        image_path=_basename(facts.image_path),
    )

    errors.extend(normalize_issues)
    try:
        size = _coerce_size(actual_size)
    except (TypeError, ValueError, IndexError, KeyError) as exc:
        errors.append(f"图片尺寸无效: {exc}")
        size = None
    ann_size = normalized.get("image_size", {})
    ann_w = int(ann_size.get("width", 0) or 0)
    ann_h = int(ann_size.get("height", 0) or 0)
    if size is not None and (ann_w != size.width or ann_h != size.height):
        warnings.append(f"image_size 与实际图片不一致: 标注=({ann_w},{ann_h}) 实际=({size.width},{size.height})")

    points = normalized.get("points", [])
    edges = normalized.get("edges", [])
    degree: Counter[int] = Counter()
    for edge in edges:
        degree[int(edge["src"])] += 1
        degree[int(edge["dst"])] += 1
    overflow_points = sorted(point_id for point_id, degree_value in degree.items() if degree_value > 2)
    if overflow_points:
        warnings.append(f"{len(overflow_points)} 个点的度数超过 2: {overflow_points[:10]}")

    return SampleAssessment(
        facts,
        normalized,
        len(points),
        len(edges),
        tuple(errors),
        tuple(warnings),
    )


def _basename(path: str) -> str:
    """Get a source basename without touching the filesystem."""

    candidate = str(path).rsplit("/", 1)[-1]
    return candidate.rsplit("\\", 1)[-1]


def _operation_facts(assessment: SampleAssessment, *, annotation_name: str) -> dict[str, str]:
    return {
        "image_source": assessment.image_path,
        "annotation_source": assessment.json_path,
        "image_name": _basename(assessment.image_path),
        "annotation_name": annotation_name,
    }


def _decision(assessment: SampleAssessment, *, model: str, include_empty: bool) -> ExportDecision:
    if include_empty is False and not assessment.errors and assessment.point_count == 0:
        facts = _operation_facts(assessment, annotation_name=f"{assessment.sample_id}.json")
        return ExportDecision(ExportDisposition.SKIP, None, facts, ("空标注已跳过",))
    if model == "A":
        invalid = bool(assessment.errors) or assessment.point_count == 0
    else:
        invalid = bool(assessment.errors) or assessment.point_count <= 1 or assessment.edge_count == 0
    facts = _operation_facts(assessment, annotation_name=f"{assessment.sample_id}.json")
    if invalid or assessment.normalized is None:
        reasons = assessment.errors
        if not reasons:
            if model == "A":
                reasons = ("点数为 0",)
            elif assessment.point_count <= 1:
                reasons = ("点数不超过 1",)
            else:
                reasons = ("边数为 0",)
        return ExportDecision(ExportDisposition.ERROR, None, facts, tuple(reasons))
    payload = master_to_labelme(assessment.normalized) if model == "A" else assessment.normalized
    return ExportDecision(ExportDisposition.EXPORT, payload, facts)


def aggregate_validation_report(
    *,
    image_dir: str,
    annotation_dir: str,
    image_sample_ids: Iterable[str],
    annotation_sample_ids: Iterable[str],
    assessments: Iterable[SampleAssessment],
) -> dict[str, Any]:
    """Aggregate supplied sample facts into the legacy validation report.

    Directory scanning and image/JSON decoding remain shell concerns; every
    dataset-level count, histogram, and report list is derived here.
    """

    image_ids = {str(sample_id) for sample_id in image_sample_ids}
    annotation_ids = {str(sample_id) for sample_id in annotation_sample_ids}
    missing_annotations = sorted(image_ids - annotation_ids)
    orphan_annotations = sorted(annotation_ids - image_ids)
    ordered_assessments = sorted(assessments, key=lambda item: item.sample_id)
    sample_reports = [assessment.to_report() for assessment in ordered_assessments]

    stats: Counter[str] = Counter()
    point_count_hist: Counter[int] = Counter()
    edge_count_hist: Counter[int] = Counter()
    for sample_report in sample_reports:
        errors = sample_report.get("errors", [])
        warnings = sample_report.get("warnings", [])
        if any(str(message).startswith("JSON 读取失败:") for message in errors):
            stats["invalid_json"] += 1
        if any(str(message).startswith("图片读取失败:") for message in errors):
            stats["invalid_image"] += 1
        if any("度数超过 2" in str(message) for message in warnings):
            stats["degree_overflow_samples"] += 1

        point_count = int(sample_report.get("point_count", 0) or 0)
        edge_count = int(sample_report.get("edge_count", 0) or 0)
        point_count_hist[min(point_count, 20)] += 1
        edge_count_hist[min(edge_count, 20)] += 1
        if point_count == 0:
            stats["zero_point_samples"] += 1
        if point_count == 1:
            stats["one_point_samples"] += 1
        if edge_count == 0:
            stats["zero_edge_samples"] += 1
        if edge_count > 0:
            stats["samples_with_edges"] += 1
        if point_count > 0:
            stats["samples_with_points"] += 1

    paired_samples = len(image_ids & annotation_ids)
    paired_errors = sum(1 for item in sample_reports if item.get("errors"))
    paired_warnings = sum(1 for item in sample_reports if item.get("warnings"))
    total_samples = paired_samples + len(missing_annotations) + len(orphan_annotations)
    stats.update(
        {
            "num_images": len(image_ids),
            "num_annotations": len(annotation_ids),
            "paired_samples": paired_samples,
            "missing_annotations": len(missing_annotations),
            "orphan_annotations": len(orphan_annotations),
            "samples_with_errors": paired_errors + len(missing_annotations) + len(orphan_annotations),
            "samples_with_warnings": paired_warnings,
            "total_samples": total_samples,
        }
    )
    zero_point_samples = sorted(item["sample_id"] for item in sample_reports if item.get("point_count") == 0)
    one_point_samples = sorted(item["sample_id"] for item in sample_reports if item.get("point_count") == 1)
    zero_edge_samples = sorted(item["sample_id"] for item in sample_reports if item.get("edge_count") == 0)
    error_samples = sorted(item["sample_id"] for item in sample_reports if item.get("errors"))
    warning_samples = sorted(item["sample_id"] for item in sample_reports if item.get("warnings"))

    return {
        "schema_version": MASTER_SCHEMA_VERSION,
        "image_dir": str(image_dir),
        "annotation_dir": str(annotation_dir),
        "summary": dict(stats),
        "missing_annotations": missing_annotations,
        "orphan_annotations": orphan_annotations,
        "zero_point_samples": zero_point_samples,
        "one_point_samples": one_point_samples,
        "zero_edge_samples": zero_edge_samples,
        "error_samples": error_samples,
        "warning_samples": warning_samples,
        "point_count_histogram": dict(point_count_hist),
        "edge_count_histogram": dict(edge_count_hist),
        "samples": sample_reports,
    }


def decide_model_a_export(assessment: SampleAssessment, *, include_empty: bool = True) -> ExportDecision:
    return _decision(assessment, model="A", include_empty=include_empty)


def decide_model_b_export(assessment: SampleAssessment, *, include_empty: bool = True) -> ExportDecision:
    return _decision(assessment, model="B", include_empty=include_empty)


# Short aliases make the pure API pleasant to discover while retaining the
# explicit names used by callers that distinguish model A and model B.
model_a_export_decision = decide_model_a_export
model_b_export_decision = decide_model_b_export


__all__ = [
    "ExportDecision",
    "ExportDisposition",
    "ImageSize",
    "SampleAssessment",
    "SampleFacts",
    "assess_sample",
    "aggregate_validation_report",
    "decide_model_a_export",
    "decide_model_b_export",
    "model_a_export_decision",
    "model_b_export_decision",
]
