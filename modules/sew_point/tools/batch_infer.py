"""
批量推理小图（256x256 裁片）并输出母格式或 LabelMe JSON。

用法：
    python -m sew_point.tools.batch_infer --input_dir ./crops --output_dir ./crops_pred
"""

import argparse
import os
import shutil
from pathlib import Path

# ruff: noqa: E402  -- direct-script bootstrap must run before project imports

# Bootstrap: allow direct execution of ``python sew_point/tools/batch_infer.py``.
# No-op when launched via ``python -m sew_point.tools.batch_infer``
# (which expects PYTHONPATH=modules;shared/cabf_common).
try:
    from ._bootstrap import ensure_module_paths
except ImportError:
    from _bootstrap import ensure_module_paths

ensure_module_paths(__file__)

from sew_point.inference_onnx import KeypointDetectorONNX
from sew_point.utils import _imread, export_labelme_json, export_master_json, IMG_EXTS


def main():
    parser = argparse.ArgumentParser(
        description="批量推理小图并输出 CAB-F 母格式或 LabelMe JSON。")
    parser.add_argument("--input_dir", type=str, required=True, help="输入图片目录")
    parser.add_argument("--output_dir", type=str, default=None, help="输出目录，默认=input_dir")
    parser.add_argument("--model", type=str, default=None, help="ONNX 模型路径")
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--cluster_dist", type=float, default=3.0)
    parser.add_argument("--device", type=str, default="cpu", help="cuda/cpu")
    parser.add_argument("--label", type=str, default="sew", help="Labelme 点标签")
    parser.add_argument(
        "--output_format",
        type=str,
        default="master",
        choices=("master", "labelme"),
        help="输出 JSON 格式：master=CAB-F 母格式，labelme=兼容旧点标注格式",
    )
    parser.add_argument("--overwrite-json", action="store_true", help="覆盖已有 JSON（不备份）")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir

    if not input_dir.exists():
        raise FileNotFoundError(f"目录不存在: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = args.model
    if model_path is None:
        model_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "checkpoints", "best.onnx")

    detector = KeypointDetectorONNX(
        model_path=model_path,
        device=args.device,
        threshold=args.threshold,
        cluster_dist=args.cluster_dist,
    )

    images = sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    )
    if not images:
        print(f"[WARN] 未找到图片: {input_dir}")
        return

    print(f"[INFO] Input: {input_dir}  Output: {output_dir}  Images: {len(images)}")

    total_points = 0
    for i, image_path in enumerate(images, 1):
        stem = image_path.stem
        json_path = output_dir / f"{stem}.json"

        points = detector.detect(str(image_path), use_tta=False)

        img = _imread(str(image_path))
        if img is None:
            print(f"[ERROR] 无法读取: {image_path}")
            continue
        h, w = img.shape[:2]

        if not args.overwrite_json and json_path.exists():
            backup = json_path.with_suffix(json_path.suffix + ".bak")
            shutil.copy2(json_path, backup)

        if args.output_format == "master":
            export_master_json(
                str(image_path),
                h,
                w,
                points,
                str(json_path),
                sample_id=stem,
                metadata={
                    "model_path": str(model_path),
                    "threshold": float(args.threshold),
                    "cluster_dist": float(args.cluster_dist),
                },
            )
        else:
            export_labelme_json(
                str(image_path),
                h,
                w,
                points,
                str(json_path),
                label_name=args.label,
            )
        total_points += len(points)
        print(
            f"[{i}/{len(images)}] {image_path.name}: {len(points)} pts -> "
            f"{json_path.name} ({args.output_format})"
        )

    print(f"[DONE] {len(images)} images, {total_points} points")


if __name__ == "__main__":
    main()
