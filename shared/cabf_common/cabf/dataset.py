from __future__ import annotations

import shutil
from collections import Counter, defaultdict
from pathlib import Path

from .constants import MASTER_SCHEMA_VERSION
from .io import iter_image_files, iter_json_files, read_image_size, read_json, write_json
from .normalize import normalize_master_annotation
from .schema import master_to_labelme


def collect_stem_maps(image_dir: str | Path, annotation_dir: str | Path) -> tuple[dict[str, Path], dict[str, Path]]:
    image_map = {path.stem: path for path in iter_image_files(image_dir)}
    json_map = {path.stem: path for path in iter_json_files(annotation_dir)}
    return image_map, json_map


def _inspect_master_sample(stem: str, image_path: Path, json_path: Path | None) -> tuple[dict, dict | None]:
    if json_path is None or not json_path.exists():
        return (
            {
                "sample_id": stem,
                "image_path": str(image_path),
                "json_path": str(json_path) if json_path else "",
                "point_count": 0,
                "edge_count": 0,
                "errors": ["缺少标注文件"],
                "warnings": [],
            },
            None,
        )

    try:
        raw = read_json(json_path)
    except Exception as exc:
        return (
            {
                "sample_id": stem,
                "image_path": str(image_path),
                "json_path": str(json_path),
                "point_count": 0,
                "edge_count": 0,
                "errors": [f"JSON 读取失败: {exc}"],
                "warnings": [],
            },
            None,
        )

    try:
        image_w, image_h = read_image_size(image_path)
    except Exception as exc:
        return (
            {
                "sample_id": stem,
                "image_path": str(image_path),
                "json_path": str(json_path),
                "point_count": 0,
                "edge_count": 0,
                "errors": [f"图片读取失败: {exc}"],
                "warnings": [],
            },
            None,
        )

    issues: list[str] = []
    warnings: list[str] = []
    normalized, normalize_issues = normalize_master_annotation(raw, sample_id=stem, image_path=image_path.name)
    issues.extend(normalize_issues)

    ann_w = int(normalized.get("image_size", {}).get("width", 0) or 0)
    ann_h = int(normalized.get("image_size", {}).get("height", 0) or 0)
    if ann_w != image_w or ann_h != image_h:
        warnings.append(f"image_size 与实际图片不一致: 标注=({ann_w},{ann_h}) 实际=({image_w},{image_h})")

    points = normalized["points"]
    edges = normalized["edges"]
    degree = Counter()
    for edge in edges:
        degree[int(edge["src"])] += 1
        degree[int(edge["dst"])] += 1
    overflow_points = sorted([point_id for point_id, deg in degree.items() if deg > 2])
    if overflow_points:
        warnings.append(f"{len(overflow_points)} 个点的度数超过 2: {overflow_points[:10]}")

    return (
        {
            "sample_id": stem,
            "image_path": str(image_path),
            "json_path": str(json_path),
            "point_count": len(points),
            "edge_count": len(edges),
            "errors": issues,
            "warnings": warnings,
        },
        normalized,
    )


def validate_master_dataset(image_dir: str | Path, annotation_dir: str | Path) -> dict:
    image_map, json_map = collect_stem_maps(image_dir, annotation_dir)
    missing_annotations = sorted(set(image_map) - set(json_map))
    orphan_annotations = sorted(set(json_map) - set(image_map))
    sample_reports = []
    stats = Counter()
    point_count_hist = Counter()
    edge_count_hist = Counter()

    for stem in sorted(set(image_map) & set(json_map)):
        image_path = image_map[stem]
        json_path = json_map[stem]
        sample_report, _ = _inspect_master_sample(stem, image_path, json_path)
        if any(msg.startswith("JSON 读取失败:") for msg in sample_report.get("errors", [])):
            stats["invalid_json"] += 1
        if any(msg.startswith("图片读取失败:") for msg in sample_report.get("errors", [])):
            stats["invalid_image"] += 1
        if sample_report.get("warnings") and any("度数超过 2" in msg for msg in sample_report["warnings"]):
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

        sample_reports.append(sample_report)

    stats["num_images"] = len(image_map)
    stats["num_annotations"] = len(json_map)
    stats["paired_samples"] = len(set(image_map) & set(json_map))
    stats["missing_annotations"] = len(missing_annotations)
    stats["orphan_annotations"] = len(orphan_annotations)
    stats["samples_with_errors"] = sum(1 for item in sample_reports if item.get("errors"))
    stats["samples_with_warnings"] = sum(1 for item in sample_reports if item.get("warnings"))
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
        sample_report, master = _inspect_master_sample(stem, image_path, json_path)
        if sample_report.get("errors") or sample_report.get("point_count", 0) == 0:
            _route_sample_to_error(image_path, json_path, error_dir)
            report["samples_routed_to_error"] += 1
            continue
        assert master is not None
        if not include_empty and not master.get("points"):
            report["skipped_empty_annotations"] += 1
            continue
        _copy_image(image_path, output_image_dir / image_path.name)
        report["images_exported"] += 1
        labelme = master_to_labelme(master)
        write_json(output_annotation_dir / f"{stem}.json", labelme)
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
        sample_report, master = _inspect_master_sample(stem, image_path, json_path)
        if sample_report.get("errors") or sample_report.get("point_count", 0) <= 1 or sample_report.get("edge_count", 0) == 0:
            _route_sample_to_error(image_path, json_path, error_dir)
            report["samples_routed_to_error"] += 1
            continue
        assert master is not None
        if not include_empty and not master.get("points"):
            report["skipped_empty_annotations"] += 1
            continue
        _copy_image(image_path, output_image_dir / image_path.name)
        report["images_exported"] += 1
        write_json(output_annotation_dir / f"{stem}.json", master)
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
