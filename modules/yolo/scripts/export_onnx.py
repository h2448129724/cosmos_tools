from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ..action_plan import YoloActionPlan
#  python -m yolo.scripts.export_onnx yolo/PullTail/runs/pulltail_yolo11n-2/weights/best.pt --output yolo/PullTail/runs/pulltail_yolo11n-2/weights/best.onnx

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a YOLO .pt model to ONNX format.")
    parser.add_argument(
        "--model",
        required=True,
        help="Path to the .pt model file, for example best.pt or last.pt.",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for the ONNX model input.")
    parser.add_argument("--opset", type=int, default=11, help="ONNX opset version.")
    parser.add_argument("--simplify", action="store_true", help="Simplify the exported ONNX model.")
    parser.add_argument("--dynamic", action="store_true", help="Enable dynamic input shapes.")
    parser.add_argument("--half", action="store_true", help="Export in FP16 half-precision.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output ONNX file path. Defaults to the same directory and name as the input model.",
    )
    parser.add_argument("--device", default=None, help="Export device, for example cpu or cuda:0.")
    return parser.parse_args()


def move_export_result(result: str | Path, out_path: Path) -> None:
    """Move an exported ONNX file, including across Windows drive letters."""
    shutil.move(str(Path(result)), str(out_path))


def main() -> None:
    args = parse_args()

    model_path = Path(args.model).resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "ultralytics is not installed. Install it with: python -m pip install ultralytics"
        ) from exc

    model = YOLO(str(model_path))

    plan = YoloActionPlan(
        "export_onnx",
        model=model_path,
        imgsz=args.imgsz,
        opset=args.opset,
        simplify=args.simplify,
        dynamic=args.dynamic,
        half=args.half,
        device=args.device,
        output=args.output,
    )
    result = model.export(**plan.export_kwargs())

    intent = plan.materialization_intent()
    if intent.move_result and intent.target is not None:
        out_path = intent.target.resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        move_export_result(result, out_path)
        print(f"Exported ONNX model to: {out_path}")
    else:
        print(f"Exported ONNX model to: {result}")


if __name__ == "__main__":
    main()
