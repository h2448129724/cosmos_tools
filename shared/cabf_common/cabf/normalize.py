from __future__ import annotations

from pathlib import Path

from .constants import MASTER_SCHEMA_VERSION
from .schema import convert_labelme_to_master, is_labelme_point_annotation, make_empty_master_annotation


def _normalize_points(raw_points: list[dict]) -> tuple[list[dict], dict[int, int], list[str]]:
    issues: list[str] = []
    normalized = []
    used_ids: set[int] = set()
    old_to_new: dict[int, int] = {}
    next_id = 0

    for idx, point in enumerate(raw_points):
        if not isinstance(point, dict):
            issues.append(f"point[{idx}] 不是对象，已跳过")
            continue
        try:
            old_id = int(point.get("id", idx))
            x = float(point["x"])
            y = float(point["y"])
        except (ValueError, TypeError, KeyError):
            issues.append(f"point[{idx}] 缺少有效的 id/x/y，已跳过")
            continue
        if old_id in used_ids:
            issues.append(f"point id {old_id} 重复，已重排")
            old_id = next_id
        while old_id in used_ids:
            old_id += 1
        used_ids.add(old_id)
        old_to_new[int(point.get("id", idx))] = old_id
        normalized.append(
            {
                "id": old_id,
                "x": x,
                "y": y,
                "score": float(point.get("score", 1.0)),
                "source": point.get("source", "manual"),
            }
        )
        next_id = max(next_id, old_id + 1)

    normalized.sort(key=lambda item: int(item["id"]))
    final_points = []
    for idx, point in enumerate(normalized):
        final_points.append({**point, "id": idx})
        old_to_new[int(point["id"])] = idx
    return final_points, old_to_new, issues


def _normalize_edges(raw_edges: list[dict], id_remap: dict[int, int]) -> tuple[list[dict], list[str]]:
    issues: list[str] = []
    normalized = []
    seen: set[tuple[int, int]] = set()

    for idx, edge in enumerate(raw_edges):
        if not isinstance(edge, dict):
            issues.append(f"edge[{idx}] 不是对象，已跳过")
            continue
        try:
            src = int(edge["src"])
            dst = int(edge["dst"])
        except (ValueError, TypeError, KeyError):
            issues.append(f"edge[{idx}] 缺少有效的 src/dst，已跳过")
            continue
        if src not in id_remap or dst not in id_remap:
            issues.append(f"edge[{idx}] 引用了不存在的点 {src}/{dst}，已跳过")
            continue
        src = id_remap[src]
        dst = id_remap[dst]
        if src == dst:
            issues.append(f"edge[{idx}] 是自环，已跳过")
            continue
        key = tuple(sorted((src, dst)))
        if key in seen:
            issues.append(f"edge[{idx}] 与已有边重复，已去重")
            continue
        seen.add(key)
        normalized.append(
            {
                "edge_id": str(edge.get("edge_id") or f"edge_{len(normalized) + 1:04d}"),
                "src": int(key[0]),
                "dst": int(key[1]),
                "label": int(edge.get("label", 1)),
                "source": edge.get("source", "manual"),
            }
        )
    return normalized, issues


def normalize_points_for_editor(points: list[dict]) -> list[dict]:
    normalized = []
    for idx, point in enumerate(points):
        normalized.append(
            {
                "id": int(point.get("id", idx)),
                "x": float(point["x"]),
                "y": float(point["y"]),
                "score": float(point.get("score", 1.0)),
                "source": point.get("source", "manual"),
            }
        )
    return normalized


def normalize_edges_for_editor(edges: list[dict]) -> list[dict]:
    normalized = []
    seen = set()
    for edge in edges:
        src = int(edge.get("src", -1))
        dst = int(edge.get("dst", -1))
        if src < 0 or dst < 0 or src == dst:
            continue
        key = tuple(sorted((src, dst)))
        if key in seen:
            continue
        seen.add(key)
        normalized.append(
            {
                "edge_id": str(edge.get("edge_id") or f"edge_{len(normalized) + 1:04d}"),
                "src": key[0],
                "dst": key[1],
                "label": int(edge.get("label", 1)),
                "source": edge.get("source", "manual"),
            }
        )
    return normalized


def normalize_master_annotation(data: dict, *, sample_id: str | None = None, image_path: str | None = None) -> tuple[dict, list[str]]:
    issues: list[str] = []
    if is_labelme_point_annotation(data):
        sample_id = sample_id or Path(str(image_path or data.get("imagePath", ""))).stem or "sample"
        data = convert_labelme_to_master(data, image_path=image_path or str(data.get("imagePath", "")), sample_id=sample_id)

    if not isinstance(data, dict):
        raise ValueError("标注内容不是有效 JSON 对象")

    width = int(data.get("image_size", {}).get("width", 0) or 256)
    height = int(data.get("image_size", {}).get("height", 0) or 256)
    normalized = make_empty_master_annotation(
        image_path=str(image_path or data.get("image_path", "")),
        width=width,
        height=height,
        sample_id=str(sample_id or data.get("sample_id", Path(str(image_path or "sample")).stem)),
    )
    normalized["schema_version"] = str(data.get("schema_version", MASTER_SCHEMA_VERSION) or MASTER_SCHEMA_VERSION)
    normalized["roi"] = data.get("roi")
    normalized["spacing_hint"] = data.get("spacing_hint")
    normalized["segments"] = data.get("segments", []) if isinstance(data.get("segments", []), list) else []
    normalized["metadata"] = data.get("metadata", {}) if isinstance(data.get("metadata", {}), dict) else {}

    points, id_remap, point_issues = _normalize_points(data.get("points", []) if isinstance(data.get("points", []), list) else [])
    edges, edge_issues = _normalize_edges(data.get("edges", []) if isinstance(data.get("edges", []), list) else [], id_remap)
    issues.extend(point_issues)
    issues.extend(edge_issues)
    normalized["points"] = points
    normalized["edges"] = edges
    normalized["schema_version"] = MASTER_SCHEMA_VERSION
    if not normalized["image_path"]:
        normalized["image_path"] = f"{normalized['sample_id']}.png"
        issues.append("image_path 缺失，已回填为 sample_id.png")
    return normalized, issues
