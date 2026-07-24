from __future__ import annotations

import argparse
from pathlib import Path

from .batch_predict import (
    _imread,
    _imwrite,
    convert_annotation_to_point_schema,
    draw_visualization,
    predict_edges_from_annotation,
    resolve_image_path,
)
from .datasets import collect_json_files, load_annotation
from .infer import POSTPROCESS_PRESETS, resolve_postprocess_params


def visualize_sequence(
    image_dir: str,
    annotation_dir: str,
    model_path: str,
    out_dir: str,
    threshold: float | None = None,
    compare_with_gt: bool = True,
    postprocess_preset: str = "balanced",
    max_degree: int | None = None,
    max_small_cycle_length: int | None = None,
    continuity_weight: float | None = None,
    cycle_penalty: float | None = None,
):
    image_root = Path(image_dir)
    annotation_root = Path(annotation_dir)
    output_root = Path(out_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    postprocess_params = resolve_postprocess_params(
        preset=postprocess_preset,
        max_degree=max_degree,
        max_small_cycle_length=max_small_cycle_length,
        continuity_weight=continuity_weight,
        cycle_penalty=cycle_penalty,
    )

    json_files = collect_json_files(str(annotation_root))
    summary = {
        "num_json": len(json_files),
        "saved_visualizations": 0,
        "skipped_empty_points": 0,
        "errors": 0,
    }

    total = len(json_files)
    for index, json_file in enumerate(json_files, start=1):
        json_path = Path(json_file)
        try:
            raw_annotation = load_annotation(str(json_path))
            annotation = convert_annotation_to_point_schema(raw_annotation, json_path, image_root)
            if len(annotation.get("points", [])) < 2:
                summary["skipped_empty_points"] += 1
                print(f"顺序可视化 step {index:03d}/{total:03d} skip={json_path.name} reason=not_enough_points")
                continue

            image_path = resolve_image_path(annotation, json_path, image_root)
            if not image_path or not Path(image_path).exists():
                raise FileNotFoundError(f"未找到对应图片: {json_path.name}")

            predicted_edges = predict_edges_from_annotation(
                annotation=annotation,
                json_path=str(json_path),
                image_dir=str(image_root),
                model_path=model_path,
                threshold=threshold,
                max_degree=postprocess_params["max_degree"],
                max_small_cycle_length=postprocess_params["max_small_cycle_length"],
                continuity_weight=postprocess_params["continuity_weight"],
                cycle_penalty=postprocess_params["cycle_penalty"],
            )
            image = _imread(image_path)
            vis = draw_visualization(image, annotation, predicted_edges, compare_with_gt=compare_with_gt)

            out_path = output_root / f"{json_path.stem}_{postprocess_preset}.png"
            _imwrite(str(out_path), vis)
            summary["saved_visualizations"] += 1
            print(
                f"顺序可视化 step {index:03d}/{total:03d} "
                f"file={json_path.name} pred_edges={len(predicted_edges)} output={out_path.name}"
            )
        except Exception as exc:
            summary["errors"] += 1
            print(f"[ERROR] 顺序可视化 {json_path.name}: {exc}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Sequentially visualize stitch-edge predictions for a folder of annotations.")
    parser.add_argument("--image_dir", type=str, required=True, help="Image folder.")
    parser.add_argument("--annotation_dir", type=str, required=True, help="Annotation folder.")
    parser.add_argument("--model_path", type=str, required=True, help="Pretrained checkpoint path.")
    parser.add_argument("--out_dir", type=str, required=True, help="Directory to save visualization images.")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--postprocess_preset", type=str, default="balanced", choices=sorted(POSTPROCESS_PRESETS))
    parser.add_argument("--max_degree", type=int, default=None)
    parser.add_argument("--max_small_cycle_length", type=int, default=None)
    parser.add_argument("--continuity_weight", type=float, default=None)
    parser.add_argument("--cycle_penalty", type=float, default=None)
    parser.add_argument("--no_compare_gt", action="store_true", help="If set, vis only shows predicted edges.")
    args = parser.parse_args()

    summary = visualize_sequence(
        image_dir=args.image_dir,
        annotation_dir=args.annotation_dir,
        model_path=args.model_path,
        out_dir=args.out_dir,
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
