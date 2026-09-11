"""Pure scoring and post-processing for stitch-edge inference.

This module intentionally knows nothing about files, torch, images, or Qt.  The
imperative adapters build graph samples and provide model scores, then delegate
the deterministic edge policy here.
"""

from __future__ import annotations

from collections import deque
from typing import Iterable, Mapping, Sequence


POSTPROCESS_PRESETS = {
    "conservative": {
        "max_degree": 2,
        "max_small_cycle_length": 5,
        "continuity_weight": 0.30,
        "cycle_penalty": 0.55,
    },
    "balanced": {
        "max_degree": 2,
        "max_small_cycle_length": 4,
        "continuity_weight": 0.20,
        "cycle_penalty": 0.35,
    },
    "aggressive": {
        "max_degree": 2,
        "max_small_cycle_length": 3,
        "continuity_weight": 0.10,
        "cycle_penalty": 0.15,
    },
}


def resolve_threshold(checkpoint: Mapping[str, object], override: float | None = None) -> float:
    """Resolve CLI threshold over checkpoint threshold over historical default."""

    return float(checkpoint.get("threshold", 0.5) if override is None else override)


def resolve_postprocess_params(
    preset: str = "balanced",
    max_degree: int | None = None,
    max_small_cycle_length: int | None = None,
    continuity_weight: float | None = None,
    cycle_penalty: float | None = None,
) -> dict:
    if preset not in POSTPROCESS_PRESETS:
        raise ValueError(f"未知 postprocess preset: {preset}")
    params = dict(POSTPROCESS_PRESETS[preset])
    if max_degree is not None:
        params["max_degree"] = int(max_degree)
    if max_small_cycle_length is not None:
        params["max_small_cycle_length"] = int(max_small_cycle_length)
    if continuity_weight is not None:
        params["continuity_weight"] = float(continuity_weight)
    if cycle_penalty is not None:
        params["cycle_penalty"] = float(cycle_penalty)
    return params


def _continuity_bonus(node_id: int, other_id: int, adjacency, point_xy) -> float:
    neighbors = adjacency.get(node_id, set())
    if len(neighbors) != 1:
        return 0.0
    existing_id = next(iter(neighbors))
    if existing_id not in point_xy or node_id not in point_xy or other_id not in point_xy:
        return 0.0
    x0, y0 = point_xy[node_id]
    x1, y1 = point_xy[existing_id]
    x2, y2 = point_xy[other_id]
    v1 = (x1 - x0, y1 - y0)
    v2 = (x2 - x0, y2 - y0)
    n1 = (v1[0] ** 2 + v1[1] ** 2) ** 0.5
    n2 = (v2[0] ** 2 + v2[1] ** 2) ** 0.5
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    cosine = (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)
    cosine = max(-1.0, min(1.0, cosine))
    return 0.5 * (1.0 - cosine)


def _shortest_path_len(adjacency, src: int, dst: int, max_depth: int) -> int | None:
    if src == dst:
        return 0
    queue = deque([(src, 0)])
    visited = {src}
    while queue:
        node, depth = queue.popleft()
        if depth >= max_depth:
            continue
        for nbr in adjacency.get(node, set()):
            if nbr == dst:
                return depth + 1
            if nbr in visited:
                continue
            visited.add(nbr)
            queue.append((nbr, depth + 1))
    return None


def apply_max_degree_constraint(
    predicted_edges: list[dict],
    point_xy: dict[int, tuple[float, float]] | None = None,
    max_degree: int = 2,
    max_small_cycle_length: int = 4,
    continuity_weight: float = 0.20,
    cycle_penalty: float = 0.35,
) -> list[dict]:
    """Greedily enforce degree/cycle constraints, preserving historical order."""

    if max_degree <= 0:
        return []
    remaining = [dict(edge) for edge in predicted_edges]
    kept = []
    degree: dict[int, int] = {}
    adjacency: dict[int, set[int]] = {}

    while remaining:
        best_idx = None
        best_score = None
        for idx, edge in enumerate(remaining):
            src = int(edge["src"])
            dst = int(edge["dst"])
            if degree.get(src, 0) >= max_degree or degree.get(dst, 0) >= max_degree:
                continue
            score = float(edge.get("score", 0.0))
            if point_xy is not None:
                score += continuity_weight * _continuity_bonus(src, dst, adjacency, point_xy)
                score += continuity_weight * _continuity_bonus(dst, src, adjacency, point_xy)
            if max_small_cycle_length >= 3:
                path_len = _shortest_path_len(adjacency, src, dst, max_small_cycle_length - 1)
                if path_len is not None:
                    cycle_len = path_len + 1
                    if cycle_len <= max_small_cycle_length:
                        score -= cycle_penalty * (max_small_cycle_length + 1 - cycle_len)
            if best_score is None or score > best_score:
                best_score = score
                best_idx = idx
        if best_idx is None:
            break
        edge = remaining.pop(best_idx)
        src = int(edge["src"])
        dst = int(edge["dst"])
        if degree.get(src, 0) >= max_degree or degree.get(dst, 0) >= max_degree:
            continue
        if max_small_cycle_length >= 3:
            path_len = _shortest_path_len(adjacency, src, dst, max_small_cycle_length - 1)
            if path_len is not None and path_len + 1 <= max_small_cycle_length:
                continue
        kept.append(edge)
        adjacency.setdefault(src, set()).add(dst)
        adjacency.setdefault(dst, set()).add(src)
        degree[src] = degree.get(src, 0) + 1
        degree[dst] = degree.get(dst, 0) + 1

    kept.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
    for idx, edge in enumerate(kept, start=1):
        edge["edge_id"] = f"pred_edge_{idx:04d}"
    return kept


def score_to_edge_results(
    probabilities: Sequence[float],
    edge_index: Iterable[Sequence[int]],
    point_ids: Sequence[int],
    threshold: float,
    point_xy: dict[int, tuple[float, float]] | None = None,
    postprocess_params: Mapping[str, object] | None = None,
) -> list[dict]:
    """Convert model probabilities to the public edge schema and apply policy."""

    predicted_edges: list[dict] = []
    for idx, pair in enumerate(edge_index):
        src, dst = int(pair[0]), int(pair[1])
        score = float(probabilities[idx])
        if score < threshold:
            continue
        predicted_edges.append(
            {
                "edge_id": f"pred_edge_{len(predicted_edges) + 1:04d}",
                "src": int(point_ids[src]),
                "dst": int(point_ids[dst]),
                "score": score,
                "label": 1,
                "source": "gnn_predict",
            }
        )
    params = dict(postprocess_params or resolve_postprocess_params())
    return apply_max_degree_constraint(predicted_edges, point_xy=point_xy, **params)

