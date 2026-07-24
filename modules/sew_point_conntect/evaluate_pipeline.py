from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from sew_point.inference import KeypointDetector

from .batch_predict import build_edge_metrics
from .datasets import (
    build_graph_sample,
    build_raw_predicted_point_annotation,
    collect_json_files,
    estimate_spacing,
    load_annotation,
    match_points_by_geometry,
    read_image,
    resolve_image_path,
)
from .infer import apply_max_degree_constraint, resolve_postprocess_params
from .model_registry import DEFAULT_MODEL, get_model


def compute_point_metrics(gt_annotation: dict, predicted_point_annotation: dict) -> dict:
    gt_points = list(gt_annotation.get("points", []))
    predicted_points = [
        (float(point["x"]), float(point["y"]), float(point.get("score", 1.0)))
        for point in predicted_point_annotation.get("points", [])
    ]
    _, matched_ids = match_points_by_geometry(gt_points, predicted_points)
    matched = sum(1 for item in matched_ids if item is not None)
    pred_points = list(predicted_point_annotation.get("points", []))
    pred_count = len(pred_points)
    gt_count = len(gt_points)
    tp = matched
    fp = max(pred_count - matched, 0)
    fn = max(gt_count - matched, 0)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return {
        "gt_points": gt_count,
        "pred_points": pred_count,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        }


def build_graph_metrics_from_matches(gt_annotation: dict, predicted_edges: list[dict], predicted_point_annotation: dict) -> dict:
    predicted_points = [
        (float(point["x"]), float(point["y"]), float(point.get("score", 1.0)))
        for point in predicted_point_annotation.get("points", [])
    ]
    _, matched_ids = match_points_by_geometry(gt_annotation.get("points", []), predicted_points)
    point_id_to_gt_id = {
        int(point.get("id", idx)): matched_ids[idx]
        for idx, point in enumerate(predicted_point_annotation.get("points", []))
    }

    lifted_edges = []
    seen_edges: set[tuple[int, int]] = set()
    for edge in predicted_edges:
        src_pred = int(edge["src"])
        dst_pred = int(edge["dst"])
        src_gt = point_id_to_gt_id.get(src_pred)
        dst_gt = point_id_to_gt_id.get(dst_pred)
        if src_gt is None or dst_gt is None or src_gt == dst_gt:
            continue
        norm_edge = tuple(sorted((int(src_gt), int(dst_gt))))
        if norm_edge in seen_edges:
            continue
        seen_edges.add(norm_edge)
        lifted_edges.append(
            {
                "src": norm_edge[0],
                "dst": norm_edge[1],
                "score": float(edge.get("score", 0.0)),
            }
        )

    return build_edge_metrics(gt_annotation, lifted_edges)


class EdgePredictor:
    def __init__(self, model_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint = torch.load(model_path, map_location=self.device)
        args = self.checkpoint.get("args", {})
        self.threshold = float(self.checkpoint.get("threshold", 0.5))
        self.args = args
        self.model = get_model(
            str(self.checkpoint.get("model_key") or args.get("model_name") or DEFAULT_MODEL),
            node_dim=int(self.checkpoint["node_dim"]),
            edge_dim=int(self.checkpoint["edge_dim"]),
            hidden_dim=int(args.get("hidden_dim", 128)),
            num_layers=int(args.get("num_layers", 3)),
            dropout=float(args.get("dropout", 0.1)),
        ).to(self.device)
        self.model.load_state_dict(self.checkpoint["model_state"])
        self.model.eval()

    def predict_edges(
        self,
        annotation: dict,
        json_path: str,
        image_dir: str,
        image_bgr,
        threshold: float | None = None,
        postprocess_preset: str = "balanced",
        max_degree: int | None = None,
        max_small_cycle_length: int | None = None,
        continuity_weight: float | None = None,
        cycle_penalty: float | None = None,
    ) -> list[dict]:
        sample = build_graph_sample(
            annotation=annotation,
            json_path=json_path,
            image_dir=image_dir,
            image_bgr=image_bgr,
            k_neighbors=int(self.args.get("k_neighbors", 8)),
            radius_multiplier=float(self.args.get("radius_multiplier", 2.5)),
            default_spacing=float(self.args.get("default_spacing", 28.0)),
            patch_width=int(self.args.get("patch_width", 96)),
            patch_height=int(self.args.get("patch_height", 24)),
        )
        if sample is None:
            return []

        edge_threshold = self.threshold if threshold is None else float(threshold)
        with torch.no_grad():
            logits = self.model(
                sample.node_x.to(self.device),
                sample.edge_index.to(self.device),
                sample.edge_attr.to(self.device),
                sample.edge_patch.to(self.device),
            )
            probs = torch.sigmoid(logits).cpu().numpy()

        predicted_edges = []
        edge_index = sample.edge_index.t().cpu().numpy()
        for idx, (src, dst) in enumerate(edge_index):
            score = float(probs[idx])
            if score < edge_threshold:
                continue
            predicted_edges.append(
                {
                    "edge_id": f"pred_edge_{len(predicted_edges) + 1:04d}",
                    "src": int(sample.point_ids[int(src)]),
                    "dst": int(sample.point_ids[int(dst)]),
                    "score": score,
                    "label": 1,
                    "source": "gnn_predict",
                }
            )
        point_xy = {
            int(point.get("id", idx)): (float(point["x"]), float(point["y"]))
            for idx, point in enumerate(annotation.get("points", []))
        }
        params = resolve_postprocess_params(
            preset=postprocess_preset,
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


def summarize_counts(tp: int, fp: int, fn: int) -> dict:
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def save_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_pipeline(args) -> dict:
    detector = KeypointDetector(
        model_path=args.stage1_model_path,
        threshold=args.stage1_threshold,
        cluster_dist=args.stage1_cluster_dist,
    )
    edge_predictor = EdgePredictor(args.stage2_model_path)

    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    json_files = collect_json_files(args.annotation_dir)
    sample_rows: list[dict] = []
    point_tp = point_fp = point_fn = 0
    edge_tp = edge_fp = edge_fn = 0

    for json_path in json_files:
        gt_annotation = load_annotation(json_path)
        image_path = resolve_image_path(gt_annotation, json_path, args.image_dir)
        image_bgr = read_image(image_path)

        predicted_points = detector.detect_numpy(image_bgr, use_tta=not args.stage1_disable_tta)
        predicted_point_annotation = build_raw_predicted_point_annotation(gt_annotation, predicted_points)
        predicted_point_annotation["image_path"] = image_path

        point_metrics = compute_point_metrics(gt_annotation, predicted_point_annotation)
        predicted_edges = edge_predictor.predict_edges(
            annotation=predicted_point_annotation,
            json_path=json_path,
            image_dir=args.image_dir,
            image_bgr=image_bgr,
            threshold=args.stage2_threshold,
            postprocess_preset=args.stage2_postprocess_preset,
            max_degree=args.stage2_max_degree,
            max_small_cycle_length=args.stage2_max_small_cycle_length,
            continuity_weight=args.stage2_continuity_weight,
            cycle_penalty=args.stage2_cycle_penalty,
        )
        edge_metrics = build_graph_metrics_from_matches(gt_annotation, predicted_edges, predicted_point_annotation)

        gt_points = gt_annotation.get("points", [])
        if gt_points:
            spacing = estimate_spacing(
                torch.tensor([[float(point["x"]), float(point["y"])] for point in gt_points], dtype=torch.float32).numpy()
            )
        else:
            spacing = 0.0

        point_tp += int(point_metrics["tp"])
        point_fp += int(point_metrics["fp"])
        point_fn += int(point_metrics["fn"])
        edge_tp += int(edge_metrics["tp"])
        edge_fp += int(edge_metrics["fp"])
        edge_fn += int(edge_metrics["fn"])

        row = {
            "sample_id": gt_annotation.get("sample_id", Path(json_path).stem),
            "json_name": Path(json_path).name,
            "gt_points": point_metrics["gt_points"],
            "pred_points": point_metrics["pred_points"],
            "point_f1": point_metrics["f1"],
            "gt_edges": edge_metrics["gt_edges"],
            "pred_edges": edge_metrics["pred_edges"],
            "graph_f1": edge_metrics["f1"],
            "graph_precision": edge_metrics["precision"],
            "graph_recall": edge_metrics["recall"],
            "estimated_spacing": spacing,
        }
        sample_rows.append(row)

        if output_dir is not None:
            combined = dict(predicted_point_annotation)
            combined["predicted_edges"] = predicted_edges
            combined["edges"] = predicted_edges
            combined["metadata"] = {
                **dict(predicted_point_annotation.get("metadata", {})),
                "stage1_model_path": args.stage1_model_path,
                "stage2_model_path": args.stage2_model_path,
                "point_f1": point_metrics["f1"],
                "graph_f1": edge_metrics["f1"],
            }
            with (output_dir / f"{Path(json_path).stem}.json").open("w", encoding="utf-8") as handle:
                json.dump(combined, handle, ensure_ascii=False, indent=2)

    point_summary = summarize_counts(point_tp, point_fp, point_fn)
    edge_summary = summarize_counts(edge_tp, edge_fp, edge_fn)
    summary = {
        "num_samples": len(sample_rows),
        "point_metrics": point_summary,
        "graph_metrics": edge_summary,
    }

    if output_dir is not None:
        save_csv(output_dir / "pipeline_metrics.csv", sample_rows)
        with (output_dir / "pipeline_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)

    return summary


def build_argparser():
    parser = argparse.ArgumentParser(description="Evaluate full stitch-graph pipeline: stage1 points + stage2 edges.")
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--annotation_dir", type=str, required=True)
    parser.add_argument("--stage1_model_path", type=str, required=True)
    parser.add_argument("--stage2_model_path", type=str, required=True)
    parser.add_argument("--stage1_threshold", type=float, default=0.5)
    parser.add_argument("--stage1_cluster_dist", type=float, default=3.0)
    parser.add_argument("--stage1_disable_tta", action="store_true")
    parser.add_argument("--stage2_threshold", type=float, default=None)
    parser.add_argument("--stage2_postprocess_preset", type=str, default="balanced", choices=["conservative", "balanced", "aggressive"])
    parser.add_argument("--stage2_max_degree", type=int, default=None)
    parser.add_argument("--stage2_max_small_cycle_length", type=int, default=None)
    parser.add_argument("--stage2_continuity_weight", type=float, default=None)
    parser.add_argument("--stage2_cycle_penalty", type=float, default=None)
    parser.add_argument("--output_dir", type=str, default="")
    return parser


def main():
    args = build_argparser().parse_args()
    summary = evaluate_pipeline(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
