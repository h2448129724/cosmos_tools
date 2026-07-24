from __future__ import annotations

import argparse
from pathlib import Path


TASK_CHOICES = ["detect", "segment", "classify", "pose"]


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="YOLO predict: run inference on images, videos, or directories."
    )
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Input path: a single image, a directory of images, or a video file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to YOLO .pt model weights, for example best.pt or last.pt.",
    )
    parser.add_argument("--task", choices=TASK_CHOICES, default="detect", help="YOLO task type.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold.")
    parser.add_argument("--device", default=None, help="Inference device, for example 0, cpu, or cuda:0.")
    parser.add_argument("--save", action="store_true", default=True, help="Save prediction images.")
    parser.add_argument("--save_txt", action="store_true", help="Save results as .txt labels.")
    parser.add_argument("--save_conf", action="store_true", help="Include confidence scores in saved txt.")
    parser.add_argument("--save_crop", action="store_true", help="Save cropped prediction regions.")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("predict_output"),
        help="Output directory for prediction results.",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()

    source_path = args.source.resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"Source path not found: {source_path}")

    model_path = Path(args.model).resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("ultralytics is not installed. Install it with: python -m pip install ultralytics") from exc

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"YOLO task      : {args.task}")
    print(f"YOLO model     : {model_path}")
    print(f"Source path    : {source_path}")
    print(f"Image size     : {args.imgsz}")
    print(f"Conf threshold : {args.conf}")
    print(f"IoU threshold  : {args.iou}")
    print(f"Output dir     : {output_dir}")
    print("=" * 60)

    model = YOLO(str(model_path))
    predict_kwargs = {
        "source": str(source_path),
        "task": args.task,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou": args.iou,
        "project": str(output_dir.parent),
        "name": output_dir.name,
        "save": args.save,
        "save_txt": args.save_txt,
        "save_conf": args.save_conf,
        "save_crop": args.save_crop,
    }
    if args.device is not None:
        predict_kwargs["device"] = args.device

    results = model.predict(**predict_kwargs)

    print("-" * 60)
    print(f"Prediction complete. {len(results)} image(s) processed.")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
