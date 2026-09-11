from __future__ import annotations

import argparse
from datetime import datetime
import json
import shlex
import subprocess
import sys
from pathlib import Path

from shared.project_paths import repo_root

from .annotation_source import resolve_edge_annotation_dir
from .config_model import load_config
from .plan import (
    CabfStepPlan,
    build_export_model_a_plan,
    build_export_model_b_plan,
    build_predict_edges_plan,
    build_predict_points_plan,
    build_post_point_workflow_plan,
    build_train_model_a_plan,
    build_train_model_b_plan,
    build_validate_plan,
)


SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_DIR / "config" / "default_paths.json"
LOG_DIR = SCRIPT_DIR / "logs"
REPO_ROOT = repo_root()


def quote_args(args: list[str]) -> str:
    return " ".join(shlex.quote(str(arg)) for arg in args)


def append_log(line: str) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{datetime.now():%Y%m%d}.log"
    with log_path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def run_command(args: list[str], cwd: str, dry_run: bool) -> int:
    timestamp = f"{datetime.now():%Y-%m-%d %H:%M:%S}"
    cwd_line = f"[{timestamp}] [cwd] {cwd}"
    cmd_line = f"[{timestamp}] [cmd] {quote_args(args)}"
    print(cwd_line)
    print(cmd_line)
    append_log(cwd_line)
    append_log(cmd_line)
    if dry_run:
        append_log(f"[{timestamp}] [result] dry-run")
        return 0
    completed = subprocess.run(args, cwd=cwd, check=False)
    append_log(f"[{timestamp}] [result] exit={completed.returncode}")
    return int(completed.returncode)


def _run_plan(step: CabfStepPlan, dry_run: bool) -> int:
    return run_command(list(step.argv), cwd=step.cwd, dry_run=dry_run)


def ensure_value(value: str, field_name: str) -> None:
    if not str(value).strip():
        raise ValueError(f"配置缺失: {field_name}")


_resolve_edge_annotation_dir = resolve_edge_annotation_dir


def cmd_predict_points(cfg: dict, args: argparse.Namespace) -> int:
    model = args.model or cfg["weights"]["sew_point_onnx"]
    step = build_predict_points_plan(
        python_executable=sys.executable,
        modules_root=cfg["train_model_modules_root"],
        images_dir=args.image_dir or cfg["master_images_dir"],
        output_dir=args.output_dir or cfg["point_predictions_dir"],
        threshold=args.threshold,
        distance_threshold=args.distance_threshold,
        model=model,
    )
    return _run_plan(step, args.dry_run)


def cmd_predict_edges(cfg: dict, args: argparse.Namespace) -> int:
    model = args.model or cfg["weights"]["sew_point_connector_pth"]
    if not args.dry_run:
        ensure_value(model, "weights.sew_point_connector_pth")
    annotation_dir, note = _resolve_edge_annotation_dir(cfg, args.annotation_dir or cfg["master_annotations_dir"])
    step = build_predict_edges_plan(
        python_executable=sys.executable,
        modules_root=cfg["train_model_modules_root"],
        images_dir=args.image_dir or cfg["master_images_dir"],
        annotation_dir=annotation_dir,
        output_dir=args.output_dir or cfg["edge_predictions_dir"],
        model=model,
        postprocess_preset=args.postprocess_preset,
        no_compare_gt=args.no_compare_gt,
    )
    if note:
        timestamp = f"{datetime.now():%Y-%m-%d %H:%M:%S}"
        print(f"[{timestamp}] [info] {note}")
        append_log(f"[{timestamp}] [info] {note}")
    return _run_plan(step, args.dry_run)


def cmd_validate(cfg: dict, args: argparse.Namespace) -> int:
    step = build_validate_plan(
        python_executable=sys.executable,
        repo_root=str(REPO_ROOT),
        images_dir=args.image_dir or cfg["master_images_dir"],
        annotation_dir=args.annotation_dir or cfg["master_annotations_dir"],
        report_path=args.report_path,
        show_samples=args.show_samples,
    )
    return _run_plan(step, args.dry_run)


def cmd_export(cfg: dict, args: argparse.Namespace) -> int:
    image_dir = args.image_dir or cfg["master_images_dir"]
    annotation_dir = args.annotation_dir or cfg["master_annotations_dir"]
    model_a_output = args.model_a_output or cfg["model_a_export_root"]
    model_b_output = args.model_b_output or cfg["model_b_export_root"]
    steps = [
        build_export_model_a_plan(
            python_executable=sys.executable, repo_root=str(REPO_ROOT), images_dir=image_dir,
            annotation_dir=annotation_dir, output_dir=model_a_output,
        ),
        build_export_model_b_plan(
            python_executable=sys.executable, repo_root=str(REPO_ROOT), images_dir=image_dir,
            annotation_dir=annotation_dir, output_dir=model_b_output,
        ),
    ]
    for step in steps:
        code = _run_plan(step, args.dry_run)
        if code != 0:
            return code
    return 0


def cmd_train(cfg: dict, args: argparse.Namespace) -> int:
    steps = [
        build_train_model_a_plan(
            python_executable=sys.executable, modules_root=cfg["train_model_modules_root"],
            images_dir=args.model_a_images or str(Path(cfg["model_a_export_root"]) / "images"),
            annotation_dir=args.model_a_annotations or str(Path(cfg["model_a_export_root"]) / "annotations"),
            output_dir=args.model_a_out or cfg["outputs"]["sew_point_train_out"],
        ),
        build_train_model_b_plan(
            python_executable=sys.executable, modules_root=cfg["train_model_modules_root"],
            images_dir=args.model_b_images or str(Path(cfg["model_b_export_root"]) / "images"),
            annotation_dir=args.model_b_annotations or str(Path(cfg["model_b_export_root"]) / "annotations"),
            output_dir=args.model_b_out or cfg["outputs"]["sew_point_conntect_train_out"],
        ),
    ]
    for step in steps:
        code = _run_plan(step, args.dry_run)
        if code != 0:
            return code
    return 0


def cmd_show_config(cfg: dict, _: argparse.Namespace) -> int:
    print(json.dumps(cfg, ensure_ascii=False, indent=2))
    return 0


def cmd_doctor(cfg: dict, _: argparse.Namespace) -> int:
    checks = [
        ("repo_root", cfg["repo_root"]),
        ("train_model_modules_root", cfg["train_model_modules_root"]),
        ("dataset_root", cfg["dataset_root"]),
        ("master_images_dir", cfg["master_images_dir"]),
        ("master_annotations_dir", cfg["master_annotations_dir"]),
    ]
    for name, value in checks:
        path = Path(value)
        label = "OK" if str(value).strip() and path.exists() else "EMPTY_OR_MISSING"
        print(f"[{label}] {name}: {value}")
    for name, value in cfg.get("weights", {}).items():
        label = "OK" if str(value).strip() and Path(value).exists() else "EMPTY_OR_MISSING"
        print(f"[{label}] weights.{name}: {value}")
    return 0


def cmd_init_config(_: dict, args: argparse.Namespace) -> int:
    src = CONFIG_PATH
    dst = Path(args.output)
    if dst.exists() and not args.force:
        raise FileExistsError(f"配置文件已存在: {dst}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"[WRITTEN] {dst}")
    return 0


def cmd_pipeline(cfg: dict, args: argparse.Namespace) -> int:
    if not args.dry_run:
        ensure_value(cfg["weights"]["sew_point_onnx"], "weights.sew_point_onnx")
        ensure_value(cfg["weights"]["sew_point_connector_pth"], "weights.sew_point_connector_pth")

    point_output_dir = args.point_output_dir or cfg["point_predictions_dir"]
    edge_output_dir = args.edge_output_dir or cfg["edge_predictions_dir"]
    validate_annotation_dir = args.validate_annotation_dir or edge_output_dir
    export_annotation_dir = args.export_annotation_dir or edge_output_dir
    point_step = build_predict_points_plan(
        python_executable=sys.executable,
        modules_root=cfg["train_model_modules_root"],
        images_dir=args.image_dir or cfg["master_images_dir"],
        model=args.point_model or cfg["weights"]["sew_point_onnx"],
        output_dir=point_output_dir,
        threshold=args.point_threshold,
        distance_threshold=args.point_distance_threshold,
    )
    print(f"\n=== {point_step.kind.value} ===")
    code = _run_plan(point_step, args.dry_run)
    if code != 0:
        return code

    requested_edge_annotation_dir = args.edge_annotation_dir or point_output_dir
    edge_annotation_dir, note = _resolve_edge_annotation_dir(cfg, requested_edge_annotation_dir)
    if note:
        timestamp = f"{datetime.now():%Y-%m-%d %H:%M:%S}"
        print(f"[{timestamp}] [info] {note}")
        append_log(f"[{timestamp}] [info] {note}")
    tail = build_post_point_workflow_plan(
        python_executable=sys.executable,
        repo_root=str(REPO_ROOT),
        modules_root=cfg["train_model_modules_root"],
        images_dir=args.image_dir or cfg["master_images_dir"],
        point_annotation_dir=edge_annotation_dir,
        edge_output_dir=edge_output_dir,
        edge_model=args.edge_model or cfg["weights"]["sew_point_connector_pth"],
        postprocess_preset=args.postprocess_preset,
        no_compare_gt=args.no_compare_gt,
        validate_annotation_dir=validate_annotation_dir,
        export_annotation_dir=export_annotation_dir,
        report_path=args.report_path,
        show_samples=args.show_samples,
        model_a_output=args.model_a_output or cfg["model_a_export_root"],
        model_b_output=args.model_b_output or cfg["model_b_export_root"],
        include_train=args.include_train,
        model_a_images=args.model_a_images or str(Path(cfg["model_a_export_root"]) / "images"),
        model_a_annotations=args.model_a_annotations or str(Path(cfg["model_a_export_root"]) / "annotations"),
        model_a_out=args.model_a_out or cfg["outputs"]["sew_point_train_out"],
        model_b_images=args.model_b_images or str(Path(cfg["model_b_export_root"]) / "images"),
        model_b_annotations=args.model_b_annotations or str(Path(cfg["model_b_export_root"]) / "annotations"),
        model_b_out=args.model_b_out or cfg["outputs"]["sew_point_conntect_train_out"],
    )
    for step in tail:
        print(f"\n=== {step.kind.value} ===")
        code = _run_plan(step, args.dry_run)
        if code != 0:
            return code
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CAB-F monorepo workflow runner.")
    parser.add_argument("--config", default=str(CONFIG_PATH), help="Path to workflow config JSON.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_points = sub.add_parser("predict-points", help="Run sew_point batch inference in master format.")
    p_points.add_argument("--image-dir", default="")
    p_points.add_argument("--output-dir", default="")
    p_points.add_argument("--model", default="")
    p_points.add_argument("--threshold", type=float, default=0.3)
    p_points.add_argument("--distance-threshold", "--cluster_dist", dest="distance_threshold", type=float, default=0.0)
    p_points.add_argument("--dry-run", action="store_true")
    p_points.set_defaults(func=cmd_predict_points)

    p_edges = sub.add_parser("predict-edges", help="Run sew_point_conntect batch edge prediction.")
    p_edges.add_argument("--image-dir", default="")
    p_edges.add_argument("--annotation-dir", default="")
    p_edges.add_argument("--output-dir", default="")
    p_edges.add_argument("--model", default="")
    p_edges.add_argument("--postprocess-preset", default="balanced", choices=("aggressive", "balanced", "conservative"))
    p_edges.add_argument("--no-compare-gt", action="store_true")
    p_edges.add_argument("--dry-run", action="store_true")
    p_edges.set_defaults(func=cmd_predict_edges)

    p_validate = sub.add_parser("validate", help="Validate CAB-F master annotations.")
    p_validate.add_argument("--image-dir", default="")
    p_validate.add_argument("--annotation-dir", default="")
    p_validate.add_argument("--report-path", default="")
    p_validate.add_argument("--show-samples", action="store_true")
    p_validate.add_argument("--dry-run", action="store_true")
    p_validate.set_defaults(func=cmd_validate)

    p_export = sub.add_parser("export", help="Export model A and model B datasets.")
    p_export.add_argument("--image-dir", default="")
    p_export.add_argument("--annotation-dir", default="")
    p_export.add_argument("--model-a-output", default="")
    p_export.add_argument("--model-b-output", default="")
    p_export.add_argument("--dry-run", action="store_true")
    p_export.set_defaults(func=cmd_export)

    p_train = sub.add_parser("train", help="Launch model A and model B training.")
    p_train.add_argument("--model-a-images", default="")
    p_train.add_argument("--model-a-annotations", default="")
    p_train.add_argument("--model-a-out", default="")
    p_train.add_argument("--model-b-images", default="")
    p_train.add_argument("--model-b-annotations", default="")
    p_train.add_argument("--model-b-out", default="")
    p_train.add_argument("--dry-run", action="store_true")
    p_train.set_defaults(func=cmd_train)

    p_pipeline = sub.add_parser("pipeline", help="Run the standard CAB-F workflow sequence.")
    p_pipeline.add_argument("--image-dir", default="")
    p_pipeline.add_argument("--point-output-dir", default="")
    p_pipeline.add_argument("--point-model", default="")
    p_pipeline.add_argument("--point-threshold", type=float, default=0.3)
    p_pipeline.add_argument("--point-distance-threshold", type=float, default=0.0)
    p_pipeline.add_argument("--edge-annotation-dir", default="")
    p_pipeline.add_argument("--edge-output-dir", default="")
    p_pipeline.add_argument("--edge-model", default="")
    p_pipeline.add_argument("--postprocess-preset", default="balanced", choices=("aggressive", "balanced", "conservative"))
    p_pipeline.add_argument("--no-compare-gt", action="store_true")
    p_pipeline.add_argument("--validate-annotation-dir", default="")
    p_pipeline.add_argument("--export-annotation-dir", default="")
    p_pipeline.add_argument("--report-path", default="")
    p_pipeline.add_argument("--show-samples", action="store_true")
    p_pipeline.add_argument("--model-a-output", default="")
    p_pipeline.add_argument("--model-b-output", default="")
    p_pipeline.add_argument("--include-train", action="store_true")
    p_pipeline.add_argument("--model-a-images", default="")
    p_pipeline.add_argument("--model-a-annotations", default="")
    p_pipeline.add_argument("--model-a-out", default="")
    p_pipeline.add_argument("--model-b-images", default="")
    p_pipeline.add_argument("--model-b-annotations", default="")
    p_pipeline.add_argument("--model-b-out", default="")
    p_pipeline.add_argument("--dry-run", action="store_true")
    p_pipeline.set_defaults(func=cmd_pipeline)

    p_doctor = sub.add_parser("doctor", help="Check key roots and configured weights.")
    p_doctor.set_defaults(func=cmd_doctor)

    p_cfg = sub.add_parser("show-config", help="Print resolved workflow config.")
    p_cfg.set_defaults(func=cmd_show_config)

    p_init = sub.add_parser("init-config", help="Write a default workflow config template.")
    p_init.add_argument("--output", default=str(CONFIG_PATH))
    p_init.add_argument("--force", action="store_true")
    p_init.set_defaults(func=cmd_init_config)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    cfg = load_config(Path(args.config))
    return int(args.func(cfg, args))


if __name__ == "__main__":
    raise SystemExit(main())
