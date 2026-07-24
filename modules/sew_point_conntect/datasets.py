from __future__ import annotations

import glob
import json
import os
import random
from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


JSON_GLOB = "*.json"


@dataclass
class GraphSample:
    sample_id: str
    json_path: str
    image_path: str
    width: int
    height: int
    spacing: float
    positive_edge_count: int
    candidate_positive_covered: int
    point_source: str
    point_ids: list[int]
    node_x: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    edge_patch: torch.Tensor
    edge_y: torch.Tensor

    @property
    def node_dim(self) -> int:
        return int(self.node_x.shape[1]) if self.node_x.ndim == 2 else 0

    @property
    def edge_dim(self) -> int:
        return int(self.edge_attr.shape[1]) if self.edge_attr.ndim == 2 else 0

    @property
    def patch_shape(self) -> tuple[int, ...]:
        return tuple(self.edge_patch.shape)


class GraphDataset(Dataset):
    def __init__(self, samples: list[GraphSample]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_graphs(batch: list[GraphSample]) -> list[GraphSample]:
    return batch


def collect_json_files(annotation_dir: str) -> list[str]:
    if not os.path.isdir(annotation_dir):
        raise FileNotFoundError(f"标注目录不存在: {annotation_dir}")
    pattern = os.path.join(annotation_dir, JSON_GLOB)
    files = []
    for path in sorted(glob.glob(pattern)):
        name = os.path.basename(path)
        if name.startswith("."):
            continue
        files.append(path)
    return files


def load_annotation(json_path: str) -> dict:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_image(image_path: str) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"无法读取图片: {image_path}")
    return image


def resolve_image_path(annotation: dict, json_path: str, image_dir: str) -> str:
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"图片目录不存在: {image_dir}")

    raw = str(annotation.get("image_path", "")).strip()
    candidates = []
    if raw:
        basename = os.path.basename(raw)
        if basename:
            candidates.append(os.path.join(image_dir, basename))
    stem = os.path.splitext(os.path.basename(json_path))[0]
    for ext in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"):
        candidates.append(os.path.join(image_dir, stem + ext))
    for path in candidates:
        if path and os.path.exists(path):
            return path
    raise FileNotFoundError(f"无法在图片目录 {image_dir} 中找到与 {json_path} 对应的图片。")


def estimate_spacing(xy: np.ndarray, default_spacing: float = 28.0) -> float:
    if len(xy) < 2:
        return float(default_spacing)
    diff = xy[:, None, :] - xy[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(dist, np.inf)
    nn = np.min(dist, axis=1)
    valid = nn[np.isfinite(nn)]
    if len(valid) == 0:
        return float(default_spacing)
    q1, q3 = np.percentile(valid, [25, 75])
    iqr = max(float(q3 - q1), 1e-6)
    keep = valid[(valid >= q1 - 1.5 * iqr) & (valid <= q3 + 1.5 * iqr)]
    if len(keep) == 0:
        keep = valid
    return float(np.median(keep))


def _build_positive_edge_set(annotation: dict, id_to_index: dict[int, int]) -> set[tuple[int, int]]:
    positives: set[tuple[int, int]] = set()
    for edge in annotation.get("edges", []):
        try:
            src_id = int(edge["src"])
            dst_id = int(edge["dst"])
        except Exception:
            continue
        if src_id == dst_id:
            continue
        if src_id not in id_to_index or dst_id not in id_to_index:
            continue
        src = id_to_index[src_id]
        dst = id_to_index[dst_id]
        positives.add(tuple(sorted((src, dst))))
    return positives


def _extract_edge_patch(
    image_bgr: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    spacing: float,
    patch_width: int,
    patch_height: int,
) -> np.ndarray:
    center = (p1 + p2) * 0.5
    dx = float(p2[0] - p1[0])
    dy = float(p2[1] - p1[1])
    length = float(np.hypot(dx, dy))
    if length < 1e-6:
        length = 1.0
    ux = dx / length
    uy = dy / length
    vx = -uy
    vy = ux

    half_len = 0.5 * (length + max(spacing, 1.0))
    half_w = 0.75 * max(spacing, 1.0)

    src = np.asarray(
        [
            [center[0] - half_len * ux - half_w * vx, center[1] - half_len * uy - half_w * vy],
            [center[0] + half_len * ux - half_w * vx, center[1] + half_len * uy - half_w * vy],
            [center[0] - half_len * ux + half_w * vx, center[1] - half_len * uy + half_w * vy],
        ],
        dtype=np.float32,
    )
    dst = np.asarray(
        [
            [0.0, 0.0],
            [patch_width - 1.0, 0.0],
            [0.0, patch_height - 1.0],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getAffineTransform(src, dst)
    patch = cv2.warpAffine(
        image_bgr,
        matrix,
        (patch_width, patch_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    patch = patch.astype(np.float32) / 255.0
    patch = np.transpose(patch, (2, 0, 1))
    return patch


def _build_candidate_edges(
    dist: np.ndarray,
    spacing: float,
    k_neighbors: int,
    radius_multiplier: float,
) -> list[tuple[int, int]]:
    n = int(dist.shape[0])
    radius = float(radius_multiplier * spacing)
    edges: set[tuple[int, int]] = set()

    for i in range(n):
        order = np.argsort(dist[i])
        taken = 0
        for j in order:
            if i == j:
                continue
            d = float(dist[i, j])
            if not np.isfinite(d):
                continue
            if d > radius and taken >= k_neighbors:
                break
            if d <= radius or taken < k_neighbors:
                edges.add(tuple(sorted((i, int(j)))))
                taken += 1
            if taken >= k_neighbors and d > radius:
                break

    if not edges and n >= 2:
        for i in range(n - 1):
            edges.add((i, i + 1))

    return sorted(edges)


def build_graph_sample(
    annotation: dict,
    json_path: str,
    image_dir: str,
    image_bgr: np.ndarray | None = None,
    k_neighbors: int = 8,
    radius_multiplier: float = 2.5,
    default_spacing: float = 28.0,
    patch_width: int = 96,
    patch_height: int = 24,
) -> GraphSample | None:
    raw_points = annotation.get("points", [])
    if len(raw_points) < 2:
        return None

    points = []
    for idx, point in enumerate(raw_points):
        points.append(
            {
                "id": int(point.get("id", idx)),
                "x": float(point["x"]),
                "y": float(point["y"]),
                "score": float(point.get("score", 1.0)),
            }
        )

    points = sorted(points, key=lambda item: int(item["id"]))
    id_to_index = {int(point["id"]): idx for idx, point in enumerate(points)}

    xy = np.asarray([[point["x"], point["y"]] for point in points], dtype=np.float32)
    score = np.asarray([point["score"] for point in points], dtype=np.float32)
    width = int(annotation.get("image_size", {}).get("width", 256) or 256)
    height = int(annotation.get("image_size", {}).get("height", 256) or 256)
    image_path = str(annotation.get("image_path", "")).strip()
    if image_bgr is None:
        image_path = resolve_image_path(annotation, json_path, image_dir)
        if image_path and os.path.exists(image_path):
            image_bgr = read_image(image_path)
    else:
        image_bgr = np.ascontiguousarray(image_bgr)

    diff = xy[:, None, :] - xy[None, :, :]
    dist = np.linalg.norm(diff, axis=-1).astype(np.float32)
    np.fill_diagonal(dist, np.inf)
    spacing = estimate_spacing(xy, default_spacing=default_spacing)

    positive_edges = _build_positive_edge_set(annotation, id_to_index)
    candidate_edges_by_rule = _build_candidate_edges(
        dist=dist,
        spacing=spacing,
        k_neighbors=k_neighbors,
        radius_multiplier=radius_multiplier,
    )
    candidate_positive_covered = len(set(candidate_edges_by_rule) & positive_edges)
    candidate_edges = sorted(set(candidate_edges_by_rule) | positive_edges)
    if not candidate_edges:
        return None

    nn = np.min(dist, axis=1)
    nn[~np.isfinite(nn)] = spacing
    valid_neighbor_count = np.sum(np.isfinite(dist), axis=1).astype(np.float32)
    mean3 = []
    count_r1 = []
    count_r2 = []
    for i in range(len(points)):
        row = dist[i][np.isfinite(dist[i])]
        if len(row) == 0:
            mean3.append(spacing)
        else:
            mean3.append(float(np.mean(np.sort(row)[: min(3, len(row))])))
        count_r1.append(float(np.sum(row <= 1.2 * spacing)))
        count_r2.append(float(np.sum(row <= 2.2 * spacing)))

    mean3 = np.asarray(mean3, dtype=np.float32)
    count_r1 = np.asarray(count_r1, dtype=np.float32)
    count_r2 = np.asarray(count_r2, dtype=np.float32)

    node_features = np.stack(
        [
            xy[:, 0] / max(width, 1),
            xy[:, 1] / max(height, 1),
            score,
            nn / max(spacing, 1e-6),
            mean3 / max(spacing, 1e-6),
            count_r1 / np.maximum(valid_neighbor_count, 1.0),
            count_r2 / np.maximum(valid_neighbor_count, 1.0),
            valid_neighbor_count / max(len(points) - 1, 1),
        ],
        axis=1,
    ).astype(np.float32)

    edge_index = []
    edge_attr = []
    edge_patch = []
    edge_y = []
    for src, dst in candidate_edges:
        dx = float(xy[dst, 0] - xy[src, 0])
        dy = float(xy[dst, 1] - xy[src, 1])
        d = float(np.hypot(dx, dy))
        edge_index.append((src, dst))
        edge_attr.append(
            [
                dx / max(spacing, 1e-6),
                dy / max(spacing, 1e-6),
                d / max(spacing, 1e-6),
                abs(dx) / max(width, 1),
                abs(dy) / max(height, 1),
                score[src],
                score[dst],
                abs(score[src] - score[dst]),
                count_r1[src] / max(float(valid_neighbor_count[src]), 1.0),
                count_r1[dst] / max(float(valid_neighbor_count[dst]), 1.0),
            ]
        )
        if image_bgr is not None:
            patch = _extract_edge_patch(
                image_bgr=image_bgr,
                p1=xy[src],
                p2=xy[dst],
                spacing=spacing,
                patch_width=patch_width,
                patch_height=patch_height,
            )
        else:
            patch = np.zeros((3, patch_height, patch_width), dtype=np.float32)
        edge_patch.append(patch)
        edge_y.append(1.0 if tuple(sorted((src, dst))) in positive_edges else 0.0)

    sample_id = str(annotation.get("sample_id") or os.path.splitext(os.path.basename(json_path))[0])

    return GraphSample(
        sample_id=sample_id,
        json_path=json_path,
        image_path=image_path,
        width=width,
        height=height,
        spacing=float(spacing),
        positive_edge_count=len(positive_edges),
        candidate_positive_covered=candidate_positive_covered,
        point_source=str(annotation.get("metadata", {}).get("point_source", "gt")),
        point_ids=[int(point["id"]) for point in points],
        node_x=torch.from_numpy(node_features),
        edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
        edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
        edge_patch=torch.tensor(np.stack(edge_patch, axis=0), dtype=torch.float32),
        edge_y=torch.tensor(edge_y, dtype=torch.float32),
    )


def _copy_annotation_with_points(annotation: dict, points: list[dict], point_source: str) -> dict:
    copied = dict(annotation)
    copied["points"] = [dict(item) for item in points]
    copied["edges"] = [dict(item) for item in annotation.get("edges", [])]
    metadata = dict(annotation.get("metadata", {}))
    metadata["point_source"] = point_source
    metadata["point_count"] = len(points)
    copied["metadata"] = metadata
    return copied


def _normalize_gt_points(annotation: dict) -> list[dict]:
    points = []
    for idx, point in enumerate(annotation.get("points", [])):
        points.append(
            {
                "id": int(point.get("id", idx)),
                "x": float(point["x"]),
                "y": float(point["y"]),
                "score": float(point.get("score", 1.0)),
                "source": str(point.get("source", "gt")),
            }
        )
    points.sort(key=lambda item: int(item["id"]))
    return points


def _point_match_radius(spacing: float, factor: float = 0.45, minimum: float = 4.0) -> float:
    return max(minimum, float(spacing) * factor)


def _deduplicate_points_by_id(points: list[dict]) -> list[dict]:
    best_by_id: dict[int, dict] = {}
    for point in points:
        point_id = int(point["id"])
        prev = best_by_id.get(point_id)
        if prev is None or float(point.get("score", 0.0)) > float(prev.get("score", 0.0)):
            best_by_id[point_id] = dict(point)
    return [best_by_id[key] for key in sorted(best_by_id)]


def match_points_by_geometry(
    gt_points: list[dict],
    predicted_points: list[tuple[int, int, float]] | list[tuple[float, float, float]],
    match_radius: float | None = None,
    default_spacing: float = 28.0,
) -> tuple[list[tuple[int, int, float]], list[int | None]]:
    gt_points = _normalize_gt_points({"points": gt_points})
    if gt_points:
        gt_xy = np.asarray([[point["x"], point["y"]] for point in gt_points], dtype=np.float32)
        spacing = estimate_spacing(gt_xy, default_spacing=default_spacing)
    else:
        spacing = float(default_spacing)
    radius = _point_match_radius(spacing) if match_radius is None else float(match_radius)

    matched_gt_ids: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    matched_ids: list[int | None] = []
    ordered_predictions = sorted(predicted_points, key=lambda item: float(item[2]), reverse=True)
    for x, y, score in ordered_predictions:
        pred_xy = np.asarray([float(x), float(y)], dtype=np.float32)
        best_point = None
        best_dist = None
        for gt_point in gt_points:
            gt_id = int(gt_point["id"])
            if gt_id in matched_gt_ids:
                continue
            dist = float(np.linalg.norm(pred_xy - np.asarray([gt_point["x"], gt_point["y"]], dtype=np.float32)))
            if dist > radius:
                continue
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_point = gt_point
        gt_id = None
        if best_point is not None:
            gt_id = int(best_point["id"])
            matched_gt_ids.add(gt_id)
        matches.append((int(round(float(x))), int(round(float(y))), float(score)))
        matched_ids.append(gt_id)
    return matches, matched_ids


def build_perturbed_point_annotation(
    annotation: dict,
    rng: random.Random,
    jitter_std_ratio: float = 0.12,
    drop_prob: float = 0.05,
    spurious_prob: float = 0.10,
) -> dict:
    gt_points = _normalize_gt_points(annotation)
    if not gt_points:
        return _copy_annotation_with_points(annotation, [], "perturb")

    width = int(annotation.get("image_size", {}).get("width", 256) or 256)
    height = int(annotation.get("image_size", {}).get("height", 256) or 256)
    xy = np.asarray([[point["x"], point["y"]] for point in gt_points], dtype=np.float32)
    spacing = estimate_spacing(xy)
    jitter_std = max(1.0, spacing * float(jitter_std_ratio))

    perturbed_points: list[dict] = []
    for point in gt_points:
        if len(gt_points) > 2 and rng.random() < drop_prob:
            continue
        x = float(np.clip(point["x"] + rng.gauss(0.0, jitter_std), 0.0, max(width - 1, 0)))
        y = float(np.clip(point["y"] + rng.gauss(0.0, jitter_std), 0.0, max(height - 1, 0)))
        score = float(np.clip(point.get("score", 1.0) - abs(rng.gauss(0.0, 0.08)), 0.2, 1.0))
        perturbed_points.append(
            {
                "id": int(point["id"]),
                "x": x,
                "y": y,
                "score": score,
                "source": "gt_perturbed",
            }
        )

    next_id = max((int(point["id"]) for point in gt_points), default=-1) + 1
    spurious_count = 0
    for point in gt_points:
        if rng.random() >= spurious_prob:
            continue
        dx = rng.uniform(-1.0, 1.0) * spacing
        dy = rng.uniform(-1.0, 1.0) * spacing
        perturbed_points.append(
            {
                "id": next_id + spurious_count,
                "x": float(np.clip(point["x"] + dx, 0.0, max(width - 1, 0))),
                "y": float(np.clip(point["y"] + dy, 0.0, max(height - 1, 0))),
                "score": float(rng.uniform(0.2, 0.8)),
                "source": "spurious",
            }
        )
        spurious_count += 1

    perturbed_points = _deduplicate_points_by_id(perturbed_points)
    return _copy_annotation_with_points(annotation, perturbed_points, "perturb")


def build_predicted_point_annotation(
    annotation: dict,
    predicted_points: list[tuple[int, int, float]] | list[tuple[float, float, float]],
) -> dict:
    gt_points = _normalize_gt_points(annotation)
    width = int(annotation.get("image_size", {}).get("width", 256) or 256)
    height = int(annotation.get("image_size", {}).get("height", 256) or 256)

    if not gt_points:
        point_items = [
            {
                "id": idx,
                "x": float(x),
                "y": float(y),
                "score": float(score),
                "source": "stage1_pred",
            }
            for idx, (x, y, score) in enumerate(predicted_points)
        ]
        return _copy_annotation_with_points(annotation, point_items, "predicted")

    _, matched_ids = match_points_by_geometry(gt_points, predicted_points)
    point_items: list[dict] = []
    next_id = max((int(point["id"]) for point in gt_points), default=-1) + 1
    extra_id = 0

    ordered_predictions = sorted(predicted_points, key=lambda item: float(item[2]), reverse=True)
    for (x, y, score), matched_gt_id in zip(ordered_predictions, matched_ids):
        if matched_gt_id is None:
            point_id = next_id + extra_id
            extra_id += 1
        else:
            point_id = int(matched_gt_id)

        point_items.append(
            {
                "id": int(point_id),
                "x": float(np.clip(x, 0.0, max(width - 1, 0))),
                "y": float(np.clip(y, 0.0, max(height - 1, 0))),
                "score": float(score),
                "source": "stage1_pred",
            }
        )

    point_items = _deduplicate_points_by_id(point_items)
    return _copy_annotation_with_points(annotation, point_items, "predicted")


def build_raw_predicted_point_annotation(
    annotation: dict,
    predicted_points: list[tuple[int, int, float]] | list[tuple[float, float, float]],
) -> dict:
    width = int(annotation.get("image_size", {}).get("width", 256) or 256)
    height = int(annotation.get("image_size", {}).get("height", 256) or 256)
    point_items = []
    ordered_predictions = sorted(predicted_points, key=lambda item: float(item[2]), reverse=True)
    for idx, (x, y, score) in enumerate(ordered_predictions):
        point_items.append(
            {
                "id": idx,
                "x": float(np.clip(x, 0.0, max(width - 1, 0))),
                "y": float(np.clip(y, 0.0, max(height - 1, 0))),
                "score": float(score),
                "source": "stage1_pred_raw",
            }
        )
    return _copy_annotation_with_points(annotation, point_items, "predicted_raw")


def build_point_source_annotations(
    annotation: dict,
    image_bgr: np.ndarray | None = None,
    point_source: str = "gt",
    stage1_detector=None,
    rng: random.Random | None = None,
    jitter_std_ratio: float = 0.12,
    drop_prob: float = 0.05,
    spurious_prob: float = 0.10,
    stage1_use_tta: bool = True,
) -> list[dict]:
    rng = rng or random.Random(42)
    variants: list[dict] = []

    if point_source in {"gt", "mixed"}:
        variants.append(_copy_annotation_with_points(annotation, _normalize_gt_points(annotation), "gt"))

    if point_source in {"perturb", "mixed"}:
        variants.append(
            build_perturbed_point_annotation(
                annotation,
                rng=rng,
                jitter_std_ratio=jitter_std_ratio,
                drop_prob=drop_prob,
                spurious_prob=spurious_prob,
            )
        )

    if point_source in {"predicted", "mixed"} and stage1_detector is not None and image_bgr is not None:
        predicted_points = stage1_detector.detect_numpy(image_bgr, use_tta=stage1_use_tta)
        variants.append(build_predicted_point_annotation(annotation, predicted_points))

    if not variants:
        variants.append(_copy_annotation_with_points(annotation, _normalize_gt_points(annotation), "gt"))

    return variants


def build_graph_samples(
    annotation_dir: str,
    image_dir: str,
    k_neighbors: int = 8,
    radius_multiplier: float = 2.5,
    default_spacing: float = 28.0,
    patch_width: int = 96,
    patch_height: int = 24,
    point_source: str = "gt",
    stage1_detector=None,
    perturb_jitter_std_ratio: float = 0.12,
    perturb_drop_prob: float = 0.05,
    perturb_spurious_prob: float = 0.10,
    stage1_use_tta: bool = True,
    random_seed: int = 42,
) -> list[GraphSample]:
    return build_graph_samples_from_json_paths(
        json_paths=collect_json_files(annotation_dir),
        image_dir=image_dir,
        k_neighbors=k_neighbors,
        radius_multiplier=radius_multiplier,
        default_spacing=default_spacing,
        patch_width=patch_width,
        patch_height=patch_height,
        point_source=point_source,
        stage1_detector=stage1_detector,
        perturb_jitter_std_ratio=perturb_jitter_std_ratio,
        perturb_drop_prob=perturb_drop_prob,
        perturb_spurious_prob=perturb_spurious_prob,
        stage1_use_tta=stage1_use_tta,
        random_seed=random_seed,
    )


def build_graph_samples_from_json_paths(
    json_paths: list[str],
    image_dir: str,
    k_neighbors: int = 8,
    radius_multiplier: float = 2.5,
    default_spacing: float = 28.0,
    patch_width: int = 96,
    patch_height: int = 24,
    point_source: str = "gt",
    stage1_detector=None,
    perturb_jitter_std_ratio: float = 0.12,
    perturb_drop_prob: float = 0.05,
    perturb_spurious_prob: float = 0.10,
    stage1_use_tta: bool = True,
    random_seed: int = 42,
) -> list[GraphSample]:
    samples: list[GraphSample] = []
    rng = random.Random(random_seed)
    for json_path in json_paths:
        annotation = load_annotation(json_path)
        image_path = resolve_image_path(annotation, json_path, image_dir)
        image_bgr = read_image(image_path)
        point_variants = build_point_source_annotations(
            annotation=annotation,
            image_bgr=image_bgr,
            point_source=point_source,
            stage1_detector=stage1_detector,
            rng=rng,
            jitter_std_ratio=perturb_jitter_std_ratio,
            drop_prob=perturb_drop_prob,
            spurious_prob=perturb_spurious_prob,
            stage1_use_tta=stage1_use_tta,
        )
        for variant_annotation in point_variants:
            variant_annotation["image_path"] = image_path
            sample = build_graph_sample(
                annotation=variant_annotation,
                json_path=json_path,
                image_dir=image_dir,
                image_bgr=image_bgr,
                k_neighbors=k_neighbors,
                radius_multiplier=radius_multiplier,
                default_spacing=default_spacing,
                patch_width=patch_width,
                patch_height=patch_height,
            )
            if sample is not None:
                samples.append(sample)
    return samples


def split_json_paths(
    json_paths: list[str],
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    json_paths = list(json_paths)
    rng = random.Random(seed)
    rng.shuffle(json_paths)
    if len(json_paths) <= 1:
        return json_paths, json_paths
    n_val = max(1, int(round(len(json_paths) * val_ratio)))
    n_val = min(n_val, len(json_paths) - 1)
    val_paths = json_paths[:n_val]
    train_paths = json_paths[n_val:]
    return train_paths, val_paths


def split_samples(samples: list[GraphSample], val_ratio: float = 0.2, seed: int = 42) -> tuple[list[GraphSample], list[GraphSample]]:
    samples = list(samples)
    rng = random.Random(seed)
    rng.shuffle(samples)
    if len(samples) <= 1:
        return samples, samples
    n_val = max(1, int(round(len(samples) * val_ratio)))
    n_val = min(n_val, len(samples) - 1)
    val_samples = samples[:n_val]
    train_samples = samples[n_val:]
    return train_samples, val_samples


def summarize_samples(samples: Iterable[GraphSample]) -> dict:
    samples = list(samples)
    if not samples:
        return {
            "num_samples": 0,
            "avg_nodes": 0.0,
            "avg_edges": 0.0,
            "pos_edges": 0,
            "neg_edges": 0,
            "pos_ratio": 0.0,
            "candidate_edge_recall": 0.0,
        }
    num_nodes = sum(int(sample.node_x.shape[0]) for sample in samples)
    num_edges = sum(int(sample.edge_y.numel()) for sample in samples)
    pos_edges = sum(int(sample.edge_y.sum().item()) for sample in samples)
    neg_edges = num_edges - pos_edges
    gt_positive_edges = sum(int(sample.positive_edge_count) for sample in samples)
    candidate_positive_covered = sum(int(sample.candidate_positive_covered) for sample in samples)
    return {
        "num_samples": len(samples),
        "avg_nodes": num_nodes / len(samples),
        "avg_edges": num_edges / len(samples),
        "pos_edges": pos_edges,
        "neg_edges": neg_edges,
        "pos_ratio": pos_edges / max(num_edges, 1),
        "candidate_edge_recall": candidate_positive_covered / max(gt_positive_edges, 1),
    }
