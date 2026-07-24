# Run from the toolbox root: python -m sew_point_conntect.batch_predict --help
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from .datasets import collect_json_files, load_annotation
from .infer import POSTPROCESS_PRESETS, apply_max_degree_constraint, resolve_postprocess_params
# cabf is imported lazily inside the functions that need it so this entry point
# stays importable without cabf on the path (direct `python -m` execution),
# matching sew_point.utils. cabf is on path in every real run — trainer_gui
# injects shared/cabf_common into PYTHONPATH before launching modules.


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


def save_json(path: Path, data: dict):
    from cabf import write_json
    write_json(path, data)


def load_labelme_points(annotation: dict) -> list[dict]:
    shapes = annotation.get("shapes", []) if isinstance(annotation, dict) else []
    points = []
    next_id = 0
    for shape in shapes:
        if not isinstance(shape, dict):
            continue
        if shape.get("shape_type") != "point":
            continue
        if str(shape.get("label", "")).strip() != "sew":
            continue
        raw_points = shape.get("points", [])
        if not raw_points or len(raw_points[0]) < 2:
            continue
        xy = raw_points[0]
        points.append(
            {
                "id": next_id,
                "x": float(xy[0]),
                "y": float(xy[1]),
                "score": float(shape["score"]) if shape.get("score") is not None else 1.0,
                "source": "labelme_point",
            }
        )
        next_id += 1
    return points


def convert_annotation_to_point_schema(annotation: dict, json_path: Path, image_dir: Path | None) -> dict:
    if isinstance(annotation, dict) and "points" in annotation:
        converted = dict(annotation)
        converted["points"] = list(annotation.get("points", []))
        converted["edges"] = list(annotation.get("edges", []))
        converted["image_path"] = resolve_image_path(annotation, json_path, image_dir)
        return converted

    width = int(annotation.get("imageWidth", 0) or 256)
    height = int(annotation.get("imageHeight", 0) or 256)
    image_path = resolve_image_path(annotation, json_path, image_dir)
    from cabf import MASTER_SCHEMA_VERSION, make_empty_master_annotation
    converted = make_empty_master_annotation(
        image_path=image_path,
        width=width,
        height=height,
        sample_id=json_path.stem,
    )
    converted["schema_version"] = MASTER_SCHEMA_VERSION
    converted["points"] = load_labelme_points(annotation)
    converted["metadata"] = {
        "source": "labelme_point_folder",
        "origin_json": str(json_path),
    }
    return converted


def resolve_image_path(annotation: dict, json_path: Path, image_dir: Path | None) -> str:
    from cabf import IMAGE_SUFFIX_ORDER
    candidates: list[Path] = []
    raw = str(annotation.get("image_path", "")).strip()
    if raw:
        candidates.append(Path(raw))
        candidates.append(json_path.parent / raw)
    if image_dir is not None:
        for ext in IMAGE_SUFFIX_ORDER:
            candidates.append(image_dir / f"{json_path.stem}{ext}")
    for ext in IMAGE_SUFFIX_ORDER:
        candidates.append(json_path.with_suffix(ext))
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return raw


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


def build_edge_metrics(annotation: dict, predicted_edges: list[dict]) -> dict:
    pred_edges = normalize_edge_set(predicted_edges)
    gt_edges = normalize_edge_set(annotation.get("edges", []))
    tp = len(pred_edges & gt_edges)
    fp = len(pred_edges - gt_edges)
    fn = len(gt_edges - pred_edges)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return {
        "gt_edges": len(gt_edges),
        "pred_edges": len(pred_edges),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def save_metrics_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_id",
        "json_name",
        "gt_edges",
        "pred_edges",
        "tp",
        "fp",
        "fn",
        "precision",
        "recall",
        "f1",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def draw_visualization(image: np.ndarray, annotation: dict, predicted_edges: list[dict], compare_with_gt: bool = True) -> np.ndarray:
    vis = image.copy()
    point_map = normalize_points(annotation)
    pred_edges = normalize_edge_set(predicted_edges)
    gt_edges = normalize_edge_set(annotation.get("edges", [])) if compare_with_gt else set()

    if gt_edges:
        tp_edges = pred_edges & gt_edges
        fp_edges = pred_edges - gt_edges
        fn_edges = gt_edges - pred_edges
        edge_groups = [
            (fn_edges, (255, 0, 0), 2),
            (fp_edges, (0, 0, 255), 2),
            (tp_edges, (0, 220, 255), 3),
        ]
        legend = [
            ("GT only (FN)", (255, 0, 0)),
            ("Pred only (FP)", (0, 0, 255)),
            ("Pred & GT (TP)", (0, 220, 255)),
        ]
    else:
        edge_groups = [(pred_edges, (0, 220, 255), 2)]
        legend = [("Pred edges", (0, 220, 255))]

    for edge_set, color, thickness in edge_groups:
        for src, dst in sorted(edge_set):
            if src not in point_map or dst not in point_map:
                continue
            cv2.line(vis, point_map[src], point_map[dst], color, thickness, lineType=cv2.LINE_AA)

    for point_id, (x, y) in point_map.items():
        cv2.circle(vis, (x, y), 4, (80, 255, 80), -1, lineType=cv2.LINE_AA)
        cv2.circle(vis, (x, y), 5, (20, 20, 20), 1, lineType=cv2.LINE_AA)
        cv2.putText(vis, str(point_id), (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    y = 18
    for label, color in legend + [("Points", (80, 255, 80))]:
        cv2.line(vis, (12, y), (34, y), color, 2, lineType=cv2.LINE_AA)
        cv2.putText(vis, label, (42, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        y += 22

    return vis


def batch_predict(
    image_dir: str,
    annotation_dir: str,
    model_path: str,
    output_annotation_dir: str,
    vis_dir: str = "",
    threshold: float | None = None,
    compare_with_gt: bool = True,
    postprocess_preset: str = "balanced",
    max_degree: int | None = None,
    max_small_cycle_length: int | None = None,
    continuity_weight: float | None = None,
    cycle_penalty: float | None = None,
):
    annotation_dir = Path(annotation_dir)
    image_root = Path(image_dir)
    output_dir = Path(output_annotation_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_root = Path(vis_dir) if vis_dir else None
    if vis_root is not None:
        vis_root.mkdir(parents=True, exist_ok=True)
    postprocess_params = resolve_postprocess_params(
        preset=postprocess_preset,
        max_degree=max_degree,
        max_small_cycle_length=max_small_cycle_length,
        continuity_weight=continuity_weight,
        cycle_penalty=cycle_penalty,
    )

    json_files = collect_json_files(str(annotation_dir))
    summary = {
        "num_json": len(json_files),
        "saved_annotations": 0,
        "saved_vis": 0,
        "skipped_empty_points": 0,
        "errors": 0,
        "evaluated_samples": 0,
        "tp": 0,
        "fp": 0,
        "fn": 0,
    }
    metrics_rows: list[dict] = []

    for json_file in json_files:
        json_path = Path(json_file)
        try:
            raw_annotation = load_annotation(str(json_path))
            annotation = convert_annotation_to_point_schema(raw_annotation, json_path, image_root)
            if len(annotation.get("points", [])) < 2:
                summary["skipped_empty_points"] += 1
                continue

            temp_input = dict(annotation)
            predicted_edges = predict_edges_from_annotation(
                temp_input,
                str(json_path),
                str(image_root),
                model_path,
                threshold=threshold,
                max_degree=postprocess_params["max_degree"],
                max_small_cycle_length=postprocess_params["max_small_cycle_length"],
                continuity_weight=postprocess_params["continuity_weight"],
                cycle_penalty=postprocess_params["cycle_penalty"],
            )
            output_annotation = dict(annotation)
            output_annotation["predicted_edges"] = predicted_edges
            output_annotation["edges"] = predicted_edges
            output_annotation["image_path"] = resolve_image_path(annotation, json_path, image_root)

            metadata = dict(output_annotation.get("metadata", {}))
            metadata["source"] = "sew_point_conntect_batch_predict"
            metadata["predicted_edge_count"] = len(predicted_edges)
            metadata["model_path"] = str(model_path)
            if compare_with_gt:
                metrics = build_edge_metrics(annotation, predicted_edges)
                metadata.update(
                    {
                        "gt_edge_count": metrics["gt_edges"],
                        "tp": metrics["tp"],
                        "fp": metrics["fp"],
                        "fn": metrics["fn"],
                        "precision": metrics["precision"],
                        "recall": metrics["recall"],
                        "f1": metrics["f1"],
                    }
                )
                summary["evaluated_samples"] += 1
                summary["tp"] += metrics["tp"]
                summary["fp"] += metrics["fp"]
                summary["fn"] += metrics["fn"]
                metrics_rows.append(
                    {
                        "sample_id": annotation.get("sample_id", json_path.stem),
                        "json_name": json_path.name,
                        **metrics,
                    }
                )
            output_annotation["metadata"] = metadata

            preset_name = str(postprocess_preset).strip()
            metadata["postprocess_preset"] = preset_name
            out_json = output_dir / f"{json_path.stem}.json"
            save_json(out_json, output_annotation)
            summary["saved_annotations"] += 1

            if vis_root is not None:
                image_path = output_annotation.get("image_path", "")
                if image_path and Path(image_path).exists():
                    image = _imread(image_path)
                    vis = draw_visualization(image, annotation, predicted_edges, compare_with_gt=compare_with_gt)
                    _imwrite(str(vis_root / f"{json_path.stem}_{preset_name}.png"), vis)
                    summary["saved_vis"] += 1
        except Exception as exc:
            summary["errors"] += 1
            print(f"[ERROR] {json_path.name}: {exc}")

    if compare_with_gt and summary["evaluated_samples"] > 0:
        summary["precision"] = summary["tp"] / max(summary["tp"] + summary["fp"], 1)
        summary["recall"] = summary["tp"] / max(summary["tp"] + summary["fn"], 1)
        summary["f1"] = 2 * summary["precision"] * summary["recall"] / max(summary["precision"] + summary["recall"], 1e-8)
        save_metrics_csv(output_dir / "metrics.csv", metrics_rows)
        save_json(output_dir / "metrics_summary.json", summary)

    return summary


def predict_edges_from_annotation(
    annotation: dict,
    json_path: str,
    image_dir: str,
    model_path: str,
    image_bgr: np.ndarray | None = None,
    threshold: float | None = None,
    max_degree: int | None = None,
    max_small_cycle_length: int | None = None,
    continuity_weight: float | None = None,
    cycle_penalty: float | None = None,
):
    from .datasets import build_graph_sample
    import torch
    from .model_registry import DEFAULT_MODEL, get_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)
    threshold = float(checkpoint.get("threshold", 0.5) if threshold is None else threshold)
    args = checkpoint.get("args", {})
    sample = build_graph_sample(
        annotation=annotation,
        json_path=json_path,
        image_dir=image_dir,
        image_bgr=image_bgr,
        k_neighbors=int(args.get("k_neighbors", 8)),
        radius_multiplier=float(args.get("radius_multiplier", 2.5)),
        default_spacing=float(args.get("default_spacing", 28.0)),
        patch_width=int(args.get("patch_width", 96)),
        patch_height=int(args.get("patch_height", 24)),
    )
    if sample is None:
        return []

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
    point_xy = normalize_points(annotation)
    params = resolve_postprocess_params(
        preset="balanced",
        max_degree=max_degree,
        max_small_cycle_length=max_small_cycle_length,
        continuity_weight=continuity_weight,
        cycle_penalty=cycle_penalty,
    )
    return apply_max_degree_constraint(
        predicted_edges,
        point_xy=point_xy,
        max_degree=params["max_degree"],
        max_small_cycle_length=params["max_small_cycle_length"],
        continuity_weight=params["continuity_weight"],
        cycle_penalty=params["cycle_penalty"],
    )


def main():
    parser = argparse.ArgumentParser(description="Batch predict stitch edges from point annotations and export annotations / optional vis.")
    parser.add_argument("--image_dir", type=str, required=True, help="Image folder.")
    parser.add_argument("--annotation_dir", type=str, required=True, help="Point-annotation folder or dataset root containing annotations.")
    parser.add_argument("--model_path", type=str, required=True, help="Pretrained checkpoint path.")
    parser.add_argument("--output_annotation_dir", type=str, required=True, help="Directory to save predicted annotations.")
    parser.add_argument("--vis_dir", type=str, default="", help="Optional directory for visualization output.")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--postprocess_preset", type=str, default="balanced", choices=sorted(POSTPROCESS_PRESETS))
    parser.add_argument("--max_degree", type=int, default=None)
    parser.add_argument("--max_small_cycle_length", type=int, default=None)
    parser.add_argument("--continuity_weight", type=float, default=None)
    parser.add_argument("--cycle_penalty", type=float, default=None)
    parser.add_argument("--no_compare_gt", action="store_true", help="If set, vis only shows predicted edges.")
    args = parser.parse_args()

    summary = batch_predict(
        image_dir=args.image_dir,
        annotation_dir=args.annotation_dir,
        model_path=args.model_path,
        output_annotation_dir=args.output_annotation_dir,
        vis_dir=args.vis_dir,
        threshold=args.threshold,
        compare_with_gt=not args.no_compare_gt,
        postprocess_preset=args.postprocess_preset,
        max_degree=args.max_degree,
        max_small_cycle_length=args.max_small_cycle_length,
        continuity_weight=args.continuity_weight,
        cycle_penalty=args.cycle_penalty,
    )
    print(summary)


if __name__ == "__main__":
    main()
