from __future__ import annotations

import argparse
from datetime import datetime
import json
import shlex
import subprocess
import sys
from pathlib import Path

from shared.project_paths import repo_root

from .config_model import DEFAULT_CONFIG, load_config, save_config
from .plan import (
    CabfStepPlan,
    build_export_model_a_plan,
    build_export_model_b_plan,
    build_validate_plan,
    build_workflow_plan,
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


def cmd_show_config(cfg: dict, _: argparse.Namespace) -> int:
    print(json.dumps(cfg, ensure_ascii=False, indent=2))
    return 0


def cmd_doctor(cfg: dict, _: argparse.Namespace) -> int:
    checks = [
        ("repo_root", cfg["repo_root"]),
        ("dataset_root", cfg["dataset_root"]),
        ("master_images_dir", cfg["master_images_dir"]),
        ("master_annotations_dir", cfg["master_annotations_dir"]),
    ]
    for name, value in checks:
        path = Path(value)
        label = "OK" if str(value).strip() and path.exists() else "EMPTY_OR_MISSING"
        print(f"[{label}] {name}: {value}")
    return 0


def cmd_init_config(_: dict, args: argparse.Namespace) -> int:
    dst = Path(args.output)
    if dst.exists() and not args.force:
        raise FileExistsError(f"配置文件已存在: {dst}")
    save_config(dst, dict(DEFAULT_CONFIG))
    print(f"[WRITTEN] {dst}")
    return 0


def cmd_pipeline(cfg: dict, args: argparse.Namespace) -> int:
    plan = build_workflow_plan(
        python_executable=sys.executable,
        repo_root=str(REPO_ROOT),
        images_dir=args.image_dir or cfg["master_images_dir"],
        annotation_dir=args.annotation_dir or cfg["master_annotations_dir"],
        model_a_output=args.model_a_output or cfg["model_a_export_root"],
        model_b_output=args.model_b_output or cfg["model_b_export_root"],
        report_path=args.report_path,
        show_samples=args.show_samples,
    )
    for step in plan:
        code = _run_plan(step, args.dry_run)
        if code != 0:
            return code
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CAB-F monorepo workflow runner.")
    parser.add_argument("--config", default=str(CONFIG_PATH), help="Path to workflow config JSON.")
    sub = parser.add_subparsers(dest="command", required=True)

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

    p_pipeline = sub.add_parser("pipeline", help="Validate annotations, then export both datasets.")
    p_pipeline.add_argument("--image-dir", default="")
    p_pipeline.add_argument("--annotation-dir", default="")
    p_pipeline.add_argument("--report-path", default="")
    p_pipeline.add_argument("--show-samples", action="store_true")
    p_pipeline.add_argument("--model-a-output", default="")
    p_pipeline.add_argument("--model-b-output", default="")
    p_pipeline.add_argument("--dry-run", action="store_true")
    p_pipeline.set_defaults(func=cmd_pipeline)

    p_doctor = sub.add_parser("doctor", help="Check key dataset roots.")
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
