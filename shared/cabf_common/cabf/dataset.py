from __future__ import annotations

import shutil
from collections import Counter, defaultdict
from pathlib import Path

from .dataset_core import (
    SampleAssessment,
    SampleFacts,
    aggregate_validation_report,
    assess_sample,
    decide_model_a_export,
    decide_model_b_export,
)
from .io import iter_image_files, iter_json_files, read_image_size, read_json, write_json


def collect_stem_maps(image_dir: str | Path, annotation_dir: str | Path) -> tuple[dict[str, Path], dict[str, Path]]:
    image_map = {path.stem: path for path in iter_image_files(image_dir)}
    json_map = {path.stem: path for path in iter_json_files(annotation_dir)}
    return image_map, json_map


def _inspect_master_sample(stem: str, image_path: Path, json_path: Path | None) -> SampleAssessment:
    if json_path is None or not json_path.exists():
        return assess_sample(
            SampleFacts(stem, str(image_path), str(json_path) if json_path else ""),
            annotation_missing=True,
        )

    try:
        raw = read_json(json_path)
    except Exception as exc:
        return assess_sample(
            SampleFacts(stem, str(image_path), str(json_path)),
            annotation_read_error=exc,
        )

    try:
        image_w, image_h = read_image_size(image_path)
    except Exception as exc:
        return assess_sample(
            SampleFacts(stem, str(image_path), str(json_path)),
            raw_annotation=raw,
            image_read_error=exc,
        )

    return assess_sample(
        SampleFacts(stem, str(image_path), str(json_path)),
        raw_annotation=raw,
        actual_size=(image_w, image_h),
    )


def validate_master_dataset(image_dir: str | Path, annotation_dir: str | Path) -> dict:
    image_map, json_map = collect_stem_maps(image_dir, annotation_dir)
    assessments: list[SampleAssessment] = []

    for stem in sorted(set(image_map) & set(json_map)):
        image_path = image_map[stem]
        json_path = json_map[stem]
        assessments.append(_inspect_master_sample(stem, image_path, json_path))

    return aggregate_validation_report(
        image_dir=str(image_dir),
        annotation_dir=str(annotation_dir),
        image_sample_ids=image_map,
        annotation_sample_ids=json_map,
        assessments=assessments,
    )


def _copy_image(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _resolve_export_dirs(output_dir: str | Path) -> tuple[Path, Path, Path]:
    output_root = Path(output_dir)
    return output_root / "images", output_root / "annotations", output_root / "error"


def _route_sample_to_error(image_path: Path, json_path: Path | None, error_dir: Path) -> None:
    _copy_image(image_path, error_dir / image_path.name)
    if json_path and json_path.exists():
        _copy_image(json_path, error_dir / json_path.name)


def export_master_to_model_a(
    image_dir: str | Path,
    annotation_dir: str | Path,
    output_dir: str | Path,
    *,
    include_empty: bool = True,
) -> dict:
    image_map, json_map = collect_stem_maps(image_dir, annotation_dir)
    output_image_dir, output_annotation_dir, error_dir = _resolve_export_dirs(output_dir)
    report = Counter()

    for stem, image_path in sorted(image_map.items()):
        json_path = json_map.get(stem)
        assessment = _inspect_master_sample(stem, image_path, json_path)
        decision = decide_model_a_export(assessment, include_empty=include_empty)
        if decision.action == "ERROR":
            _route_sample_to_error(image_path, json_path, error_dir)
            report["samples_routed_to_error"] += 1
            continue
        if decision.action == "SKIP":
            report["skipped_empty_annotations"] += 1
            continue
        assert decision.payload is not None
        _copy_image(image_path, output_image_dir / image_path.name)
        report["images_exported"] += 1
        write_json(output_annotation_dir / f"{stem}.json", decision.payload)
        report["annotations_exported"] += 1

    result = dict(report)
    result.update(
        {
            "output_dir": str(Path(output_dir)),
            "images_dir": str(output_image_dir),
            "annotations_dir": str(output_annotation_dir),
            "error_dir": str(error_dir),
        }
    )
    return result


def export_master_to_model_b(
    image_dir: str | Path,
    annotation_dir: str | Path,
    output_dir: str | Path,
    *,
    include_empty: bool = True,
) -> dict:
    image_map, json_map = collect_stem_maps(image_dir, annotation_dir)
    output_image_dir, output_annotation_dir, error_dir = _resolve_export_dirs(output_dir)
    report = Counter()

    for stem, image_path in sorted(image_map.items()):
        json_path = json_map.get(stem)
        assessment = _inspect_master_sample(stem, image_path, json_path)
        decision = decide_model_b_export(assessment, include_empty=include_empty)
        if decision.action == "ERROR":
            _route_sample_to_error(image_path, json_path, error_dir)
            report["samples_routed_to_error"] += 1
            continue
        if decision.action == "SKIP":
            report["skipped_empty_annotations"] += 1
            continue
        assert decision.payload is not None
        _copy_image(image_path, output_image_dir / image_path.name)
        report["images_exported"] += 1
        write_json(output_annotation_dir / f"{stem}.json", decision.payload)
        report["annotations_exported"] += 1

    result = dict(report)
    result.update(
        {
            "output_dir": str(Path(output_dir)),
            "images_dir": str(output_image_dir),
            "annotations_dir": str(output_annotation_dir),
            "error_dir": str(error_dir),
        }
    )
    return result


def summarize_validation(report: dict) -> str:
    summary = report.get("summary", {})
    lines = [
        f"images={summary.get('num_images', 0)} annotations={summary.get('num_annotations', 0)} paired={summary.get('paired_samples', 0)}",
        f"missing_annotations={summary.get('missing_annotations', 0)} orphan_annotations={summary.get('orphan_annotations', 0)}",
        f"zero_point={summary.get('zero_point_samples', 0)} one_point={summary.get('one_point_samples', 0)} zero_edge={summary.get('zero_edge_samples', 0)}",
        f"samples_with_errors={summary.get('samples_with_errors', 0)} samples_with_warnings={summary.get('samples_with_warnings', 0)}",
    ]
    return "\n".join(lines)


def summarize_validation_findings(report: dict, *, include_details: bool = False) -> str:
    sample_map = {item.get("sample_id"): item for item in report.get("samples", []) if item.get("sample_id")}
    sample_types: defaultdict[str, list[str]] = defaultdict(list)
    category_labels = [
        ("missing_annotations", "missing_annotation"),
        ("orphan_annotations", "orphan_annotation"),
        ("error_samples", "error"),
        ("warning_samples", "warning"),
        ("zero_point_samples", "zero_point"),
        ("one_point_samples", "one_point"),
        ("zero_edge_samples", "zero_edge"),
    ]
    for key, label in category_labels:
        for sample_id in report.get(key, []):
            sample_types[sample_id].append(label)

    lines: list[str] = []
    for sample_id in sorted(sample_types):
        labels = ", ".join(dict.fromkeys(sample_types[sample_id]))
        lines.append(f"[{sample_id}] {labels}")
        if not include_details:
            continue
        sample = sample_map.get(sample_id, {})
        if sample:
            lines.append(f"  counts: points={sample.get('point_count', 0)} edges={sample.get('edge_count', 0)}")
            for issue in sample.get("errors", []):
                lines.append(f"  error: {issue}")
            for warning in sample.get("warnings", []):
                lines.append(f"  warning: {warning}")
        elif sample_id in report.get("missing_annotations", []):
            lines.append("  error: 缺少标注文件")
        elif sample_id in report.get("orphan_annotations", []):
            lines.append("  error: 缺少对应图片文件")
    return "\n".join(lines)
