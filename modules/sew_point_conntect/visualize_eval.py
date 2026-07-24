from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .infer import predict_edges


def _imread(path: str) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"无法读取图片: {path}")
    return image


def _imwrite(path: str, image: np.ndarray):
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    suffix = path_obj.suffix or ".png"
    ok, buf = cv2.imencode(suffix, image)
    if not ok:
        raise RuntimeError(f"无法保存图片: {path}")
    buf.tofile(str(path_obj))


def load_annotation(json_path: str) -> dict:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_image(annotation: dict, json_path: str) -> tuple[np.ndarray, str]:
    candidates = []
    image_path = str(annotation.get("image_path", "")).strip()
    if image_path:
        candidates.append(Path(image_path))
    json_parent = Path(json_path).parent
    if image_path:
        candidates.append(json_parent / image_path)
    candidates.append(Path(json_path).with_suffix(".png"))

    for candidate in candidates:
        if candidate.exists():
            return _imread(str(candidate)), str(candidate)

    width = int(annotation.get("image_size", {}).get("width", 256) or 256)
    height = int(annotation.get("image_size", {}).get("height", 256) or 256)
    blank = np.full((height, width, 3), 24, dtype=np.uint8)
    return blank, ""


def normalize_points(annotation: dict) -> dict[int, tuple[int, int]]:
    point_map: dict[int, tuple[int, int]] = {}
    for idx, point in enumerate(annotation.get("points", [])):
        point_id = int(point.get("id", idx))
        point_map[point_id] = (int(round(float(point["x"]))), int(round(float(point["y"]))))
    return point_map


def normalize_edge_set(edges: list[dict]) -> set[tuple[int, int]]:
    edge_set: set[tuple[int, int]] = set()
    for edge in edges:
        try:
            src = int(edge["src"])
            dst = int(edge["dst"])
        except Exception:
            continue
        if src == dst:
            continue
        edge_set.add(tuple(sorted((src, dst))))
    return edge_set


def draw_points(image: np.ndarray, point_map: dict[int, tuple[int, int]]):
    for point_id, (x, y) in point_map.items():
        cv2.circle(image, (x, y), 4, (80, 255, 80), -1, lineType=cv2.LINE_AA)
        cv2.circle(image, (x, y), 5, (20, 20, 20), 1, lineType=cv2.LINE_AA)
        cv2.putText(
            image,
            str(point_id),
            (x + 6, y - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
            lineType=cv2.LINE_AA,
        )


def draw_edges(
    image: np.ndarray,
    edge_set: set[tuple[int, int]],
    point_map: dict[int, tuple[int, int]],
    color: tuple[int, int, int],
    thickness: int = 2,
):
    for src, dst in sorted(edge_set):
        if src not in point_map or dst not in point_map:
            continue
        cv2.line(image, point_map[src], point_map[dst], color, thickness, lineType=cv2.LINE_AA)


def draw_legend(image: np.ndarray):
    items = [
        ("GT only (FN)", (255, 0, 0)),
        ("Pred only (FP)", (0, 0, 255)),
        ("Pred & GT (TP)", (0, 220, 255)),
        ("Points", (80, 255, 80)),
    ]
    x0, y0 = 12, 18
    for idx, (label, color) in enumerate(items):
        y = y0 + idx * 22
        cv2.line(image, (x0, y), (x0 + 22, y), color, 2, lineType=cv2.LINE_AA)
        cv2.putText(
            image,
            label,
            (x0 + 30, y + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            lineType=cv2.LINE_AA,
        )


def build_visualization(json_path: str, image_path: str, model_path: str, threshold: float | None = None) -> tuple[np.ndarray, dict]:
    annotation = load_annotation(json_path)
    annotation = dict(annotation)
    annotation["image_path"] = image_path
    image, resolved_image_path = resolve_image(annotation, json_path)
    point_map = normalize_points(annotation)

    pred_edges_raw = predict_edges(json_path, image_path, model_path, threshold=threshold)
    pred_edges = normalize_edge_set(pred_edges_raw)
    gt_edges = normalize_edge_set(annotation.get("edges", []))

    tp_edges = pred_edges & gt_edges
    fp_edges = pred_edges - gt_edges
    fn_edges = gt_edges - pred_edges

    vis = image.copy()
    draw_edges(vis, fn_edges, point_map, (255, 0, 0), thickness=2)
    draw_edges(vis, fp_edges, point_map, (0, 0, 255), thickness=2)
    draw_edges(vis, tp_edges, point_map, (0, 220, 255), thickness=3)
    draw_points(vis, point_map)
    draw_legend(vis)

    metrics = {
        "json_path": json_path,
        "image_path": resolved_image_path,
        "num_points": len(point_map),
        "gt_edges": len(gt_edges),
        "pred_edges": len(pred_edges),
        "tp": len(tp_edges),
        "fp": len(fp_edges),
        "fn": len(fn_edges),
        "precision": len(tp_edges) / max(len(tp_edges) + len(fp_edges), 1),
        "recall": len(tp_edges) / max(len(tp_edges) + len(fn_edges), 1),
    }
    return vis, metrics


def default_output_path(json_path: str, out_dir: str = "") -> str:
    json_obj = Path(json_path)
    base_dir = Path(out_dir) if out_dir else json_obj.parent
    return str(base_dir / f"{json_obj.stem}_edge_eval.png")


def main():

    parser = argparse.ArgumentParser(description="Visualize predicted stitch edges on image.")
    parser.add_argument("--json_path", type=str, required=True)
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--save_path", type=str, default="")
    parser.add_argument("--out_dir", type=str, default="")
    args = parser.parse_args()

    vis, metrics = build_visualization(args.json_path, args.image_path, args.model_path, threshold=args.threshold)
    save_path = args.save_path or default_output_path(args.json_path, args.out_dir)
    _imwrite(save_path, vis)

    print(f"Saved visualization: {save_path}")
    print(
        f"points={metrics['num_points']} gt_edges={metrics['gt_edges']} pred_edges={metrics['pred_edges']} "
        f"tp={metrics['tp']} fp={metrics['fp']} fn={metrics['fn']} "
        f"precision={metrics['precision']:.4f} recall={metrics['recall']:.4f}"
    )


if __name__ == "__main__":
    main()
