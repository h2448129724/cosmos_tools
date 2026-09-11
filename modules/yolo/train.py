from __future__ import annotations

import argparse
from pathlib import Path

from .action_plan import YoloActionPlan


TASK_CHOICES = ["detect", "segment", "classify", "pose"]
SERIES_CHOICES = ["yolo11", "yolo26"]
SIZE_CHOICES = ["n", "s", "m", "l", "x"]


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ultralytics YOLO training schema for detect, segment, classify, and pose tasks."
    )
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Dataset path. Use data.yaml for detect/segment, or a dataset root for classify.",
    )
    parser.add_argument("--task", choices=TASK_CHOICES, default="detect", help="YOLO task type.")
    parser.add_argument(
        "--model_series",
        choices=SERIES_CHOICES,
        default="yolo11",
        help="Official Ultralytics model family. GUI command generation uses the yolo CLI.",
    )
    parser.add_argument("--model_size", choices=SIZE_CHOICES, default="n", help="YOLO model size.")
    parser.add_argument(
        "--custom_model",
        type=str,
        default="",
        help="Optional custom .pt path. If provided, it overrides model_series/model_size/task naming.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="0", help="Training device, for example 0, cpu, or cuda:0.")
    parser.add_argument(
        "--save_dir",
        type=Path,
        default=Path("checkpoints/yolo/runs"),
        help="Final run output directory. The wrapper maps this into Ultralytics project/name fields.",
    )
    parser.add_argument("--run_name", default="", help="Optional run name override. Leave empty to auto-generate one.")
    parser.add_argument("--workers", type=int, default=0, help="Use 0 on Windows to avoid DataLoader issues.")
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache", action="store_true", help="Enable Ultralytics dataset cache.")
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision training.")
    return parser


def resolve_model_name(task: str, model_series: str, model_size: str, custom_model: str) -> str:
    return YoloActionPlan(
        "train",
        task=task,
        model_series=model_series,
        model_size=model_size,
        custom_model=custom_model,
    ).model_name


def split_output_target(save_dir: Path, run_name: str) -> tuple[Path, str]:
    plan = YoloActionPlan("train", save_dir=save_dir.resolve(), run_name=run_name)
    intent = plan.materialization_intent()
    assert intent.project is not None and intent.name is not None
    return intent.project, intent.name


def train(args: argparse.Namespace) -> None:
    data_path = args.data.resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {data_path}")

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("ultralytics is not installed. Install it with: python -m pip install ultralytics") from exc

    plan = YoloActionPlan(
        "train",
        task=args.task,
        model_series=args.model_series,
        model_size=args.model_size,
        custom_model=args.custom_model,
        data=data_path,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        save_dir=args.save_dir.resolve(),
        run_name=args.run_name,
        workers=args.workers,
        patience=args.patience,
        seed=args.seed,
        cache=args.cache,
        amp=args.amp,
    )
    intent = plan.materialization_intent()
    assert intent.project is not None and intent.name is not None
    project_dir, run_name = intent.project, intent.name
    model_name = plan.model_name

    print("=" * 60)
    print(f"YOLO task      : {args.task}")
    print(f"YOLO model     : {model_name}")
    print(f"Dataset path   : {data_path}")
    print(f"Project dir    : {project_dir}")
    print(f"Run name       : {run_name}")
    print("=" * 60)

    model = YOLO(model_name)
    model.train(**plan.train_kwargs())


def main() -> None:
    train(build_argparser().parse_args())


if __name__ == "__main__":
    main()
