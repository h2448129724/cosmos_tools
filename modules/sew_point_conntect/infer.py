from __future__ import annotations

import argparse
import json
import os

from .datasets import build_graph_sample, load_annotation
from .inference_core import POSTPROCESS_PRESETS, apply_max_degree_constraint, resolve_postprocess_params
from .runtime import TorchConnectorRuntime

__all__ = ["POSTPROCESS_PRESETS", "apply_max_degree_constraint", "resolve_postprocess_params", "predict_edges"]


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
    runtime = TorchConnectorRuntime(ckpt_path)
    annotation = load_annotation(json_path)
    annotation = dict(annotation)
    annotation["image_path"] = image_path
    args = runtime.args
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
    print(f"[INFO] 连线推理设备: {runtime.device}")
    return runtime.predict_edges(
        sample,
        annotation,
        threshold=threshold,
        postprocess_preset=postprocess_preset,
        max_degree=max_degree,
        max_small_cycle_length=max_small_cycle_length,
        continuity_weight=continuity_weight,
        cycle_penalty=cycle_penalty,
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
