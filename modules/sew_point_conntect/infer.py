from __future__ import annotations

import argparse
import json
import os
from collections import deque

import torch

from .datasets import build_graph_sample, load_annotation
from .model_registry import DEFAULT_MODEL, get_model


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


def _build_point_xy(annotation: dict) -> dict[int, tuple[float, float]]:
    point_xy = {}
    for idx, point in enumerate(annotation.get("points", [])):
        point_id = int(point.get("id", idx))
        point_xy[point_id] = (float(point["x"]), float(point["y"]))
    return point_xy


def _continuity_bonus(
    node_id: int,
    other_id: int,
    adjacency: dict[int, set[int]],
    point_xy: dict[int, tuple[float, float]],
) -> float:
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
    cos = (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)
    cos = max(-1.0, min(1.0, cos))
    return 0.5 * (1.0 - cos)


def _shortest_path_len(adjacency: dict[int, set[int]], src: int, dst: int, max_depth: int) -> int | None:
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


def predict_edges(
    json_path: str,
    image_path: str,
    ckpt_path: str,
    threshold: float | None = None,
    postprocess_preset: str = "balanced",
    max_degree: int | None = None,
    max_small_cycle_length: int | None = None,
    continuity_weight: float | None = None,
    cycle_penalty: float | None = None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(ckpt_path, map_location=device)
    threshold = float(checkpoint.get("threshold", 0.5) if threshold is None else threshold)

    annotation = load_annotation(json_path)
    annotation = dict(annotation)
    annotation["image_path"] = image_path
    args = checkpoint.get("args", {})
    sample = build_graph_sample(
        annotation=annotation,
        json_path=json_path,
        image_dir=os.path.dirname(image_path),
        k_neighbors=int(args.get("k_neighbors", 8)),
        radius_multiplier=float(args.get("radius_multiplier", 2.5)),
        default_spacing=float(args.get("default_spacing", 28.0)),
        patch_width=int(args.get("patch_width", 96)),
        patch_height=int(args.get("patch_height", 24)),
    )
    if sample is None:
        raise RuntimeError("当前样本没有足够点构图。")

    model = get_model(
        str(checkpoint.get("model_key") or args.get("model_name") or DEFAULT_MODEL),
        node_dim=int(checkpoint["node_dim"]),
        edge_dim=int(checkpoint["edge_dim"]),
        hidden_dim=int(args.get("hidden_dim", 128)),
        num_layers=int(args.get("num_layers", 3)),
        dropout=float(args.get("dropout", 0.1)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    print(f"[INFO] 连线推理设备: {device}")

    with torch.no_grad():
        node_x = sample.node_x.to(device)
        edge_index = sample.edge_index.to(device)
        edge_attr = sample.edge_attr.to(device)
        edge_patch = sample.edge_patch.to(device)
        logits = model(node_x, edge_index, edge_attr, edge_patch)
        probs = torch.sigmoid(logits).cpu().numpy()

    predicted_edges = []
    edge_index = sample.edge_index.t().cpu().numpy()
    for idx, (src, dst) in enumerate(edge_index):
        score = float(probs[idx])
        if score >= threshold:
            src_id = int(sample.point_ids[int(src)])
            dst_id = int(sample.point_ids[int(dst)])
            predicted_edges.append(
                {
                    "edge_id": f"pred_edge_{len(predicted_edges) + 1:04d}",
                    "src": src_id,
                    "dst": dst_id,
                    "score": score,
                    "label": 1,
                    "source": "gnn_predict",
                }
            )
    point_xy = _build_point_xy(annotation)
    postprocess_params = resolve_postprocess_params(
        preset=postprocess_preset,
        max_degree=max_degree,
        max_small_cycle_length=max_small_cycle_length,
        continuity_weight=continuity_weight,
        cycle_penalty=cycle_penalty,
    )
    return apply_max_degree_constraint(
        predicted_edges,
        point_xy=point_xy,
        max_degree=postprocess_params["max_degree"],
        max_small_cycle_length=postprocess_params["max_small_cycle_length"],
        continuity_weight=postprocess_params["continuity_weight"],
        cycle_penalty=postprocess_params["cycle_penalty"],
    )


def main():
    parser = argparse.ArgumentParser(description="Run stitch-point edge prediction on one JSON.")
    parser.add_argument("--json_path", type=str, required=True)
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, default=os.path.join(os.path.dirname(__file__), "checkpoints", "best.pth"))
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--postprocess_preset", type=str, default="balanced", choices=sorted(POSTPROCESS_PRESETS))
    parser.add_argument("--max_degree", type=int, default=None)
    parser.add_argument("--max_small_cycle_length", type=int, default=None)
    parser.add_argument("--continuity_weight", type=float, default=None)
    parser.add_argument("--cycle_penalty", type=float, default=None)
    parser.add_argument("--save_path", type=str, default="")
    args = parser.parse_args()

    predicted_edges = predict_edges(
        args.json_path,
        args.image_path,
        args.model_path,
        threshold=args.threshold,
        postprocess_preset=args.postprocess_preset,
        max_degree=args.max_degree,
        max_small_cycle_length=args.max_small_cycle_length,
        continuity_weight=args.continuity_weight,
        cycle_penalty=args.cycle_penalty,
    )
    print(f"Predicted edges: {len(predicted_edges)}")

    if args.save_path:
        annotation = load_annotation(args.json_path)
        annotation["predicted_edges"] = predicted_edges
        with open(args.save_path, "w", encoding="utf-8") as f:
            json.dump(annotation, f, ensure_ascii=False, indent=2)
        print(f"Saved to {args.save_path}")
    else:
        for edge in predicted_edges[:20]:
            print(edge)


if __name__ == "__main__":
    main()
