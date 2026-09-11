from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from sew_point.inference import KeypointDetector

from .datasets import (
    build_graph_sample,
    build_raw_predicted_point_annotation,
    collect_json_files,
    estimate_spacing,
    load_annotation,
    read_image,
    resolve_image_path,
)
from .evaluation_core import graph_metrics_from_matches, point_metrics, summarize_counts as core_summarize_counts
from .runtime import TorchConnectorRuntime


def compute_point_metrics(gt_annotation: dict, predicted_point_annotation: dict) -> dict:
    predicted_points = [
        (float(point["x"]), float(point["y"]), float(point.get("score", 1.0)))
        for point in predicted_point_annotation.get("points", [])
    ]
    return point_metrics(gt_annotation.get("points", []), predicted_points)


def build_graph_metrics_from_matches(gt_annotation: dict, predicted_edges: list[dict], predicted_point_annotation: dict) -> dict:
    return graph_metrics_from_matches(
        gt_annotation.get("points", []),
        gt_annotation.get("edges", []),
        predicted_point_annotation.get("points", []),
        predicted_edges,
    )


class EdgePredictor:
    def __init__(self, model_path: str):
        self.runtime = TorchConnectorRuntime(model_path)
        self.device = self.runtime.device
        self.checkpoint = self.runtime.checkpoint
        self.threshold = self.runtime.threshold
        self.args = self.runtime.args

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
        return self.runtime.predict_edges(
            sample,
            annotation,
            threshold=edge_threshold,
            postprocess_preset=postprocess_preset,
            max_degree=max_degree,
            max_small_cycle_length=max_small_cycle_length,
            continuity_weight=continuity_weight,
            cycle_penalty=cycle_penalty,
        )


def summarize_counts(tp: int, fp: int, fn: int) -> dict:
    return core_summarize_counts(tp, fp, fn)


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
