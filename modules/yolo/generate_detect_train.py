from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    from .convert_xanylabeling import ConvertError, discover_yolo_datasets
except ImportError:
    from convert_xanylabeling import ConvertError, discover_yolo_datasets


LogFn = Callable[[str], None]
MEDIA_DIR_NAMES = {"images", "labels"}
SPLIT_NAMES = {"train", "val", "test"}

# 与 modules/yolo/train.py 默认值对齐（仅 detect）。
DEFAULT_MODEL_SERIES = "yolo11"
DEFAULT_MODEL_SIZE = "n"
DEFAULT_EPOCHS = 100
DEFAULT_IMGSZ = 1024
DEFAULT_BATCH = 4
DEFAULT_DEVICE = "0"
DEFAULT_WORKERS = 0
DEFAULT_PATIENCE = 30
DEFAULT_SEED = 42
DEFAULT_CONDA_ENV = "onnx-gpu"
DEFAULT_RUN_NAME = "detect"
SCRIPT_NAME = "train_detect.py"


@dataclass
class TrainCodeOptions:
    model_series: str = DEFAULT_MODEL_SERIES
    model_size: str = DEFAULT_MODEL_SIZE
    custom_model: str = ""
    epochs: int = DEFAULT_EPOCHS
    imgsz: int = DEFAULT_IMGSZ
    batch: int = DEFAULT_BATCH
    device: str = DEFAULT_DEVICE
    workers: int = DEFAULT_WORKERS
    patience: int = DEFAULT_PATIENCE
    seed: int = DEFAULT_SEED
    cache: bool = False
    amp: bool = False
    conda_env: str = DEFAULT_CONDA_ENV
    run_name: str = DEFAULT_RUN_NAME
    overwrite: bool = False
    dry_run: bool = False


@dataclass
class TrainCodeReport:
    datasets: list[str] = field(default_factory=list)
    written: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    run_commands: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors and bool(self.written or self.skipped)

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "datasets": self.datasets,
            "written": self.written,
            "skipped": self.skipped,
            "run_commands": self.run_commands,
            "warnings": self.warnings,
            "errors": self.errors,
        }


def _log(log: LogFn | None, message: str) -> None:
    if log is not None:
        log(message)


def ensure_absolute_data_yaml_path(yaml_path: Path, *, dry_run: bool = False) -> bool:
    """把 data.yaml 里的 path: . 改成绝对路径，避免 Ultralytics 按 CWD 解析失败。"""

    text = yaml_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    root = yaml_path.parent.resolve().as_posix()
    changed = False
    out: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("path:"):
            raw = stripped[5:].strip().strip("'\"")
            if raw in {".", "./", ""} or not Path(raw).is_absolute():
                out.append(f"path: {root}")
                changed = True
                continue
        out.append(line)
    if not changed:
        return False
    if dry_run:
        return True
    ending = "\n" if text.endswith("\n") else ""
    yaml_path.write_text("\n".join(out) + ending, encoding="utf-8")
    return True


def _py_literal(value: object) -> str:
    return repr(str(value))


def _path_literal(path: Path) -> str:
    return "Path(" + _py_literal(path) + ")"


def detect_model_name(model_series: str, model_size: str, custom_model: str) -> str:
    if custom_model.strip():
        return custom_model.strip()
    return f"{model_series}{model_size}.pt"


def default_save_dir(dataset_dir: Path) -> Path:
    return dataset_dir / "runs" / DEFAULT_RUN_NAME


def script_filename(yolo_root: Path, dataset_dir: Path, dataset_count: int) -> str:
    if dataset_count <= 1:
        return SCRIPT_NAME
    try:
        relative = dataset_dir.resolve().relative_to(yolo_root.resolve())
    except ValueError:
        relative = Path(dataset_dir.name)
    if str(relative) in {".", ""}:
        return SCRIPT_NAME
    safe = "_".join(part for part in relative.parts if part not in {".", ""})
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in safe).strip("_")
    return f"train_detect_{safe or dataset_dir.name}.py"


def is_unsafe_output_dir(path: Path) -> bool:
    parts = [part.lower() for part in path.resolve().parts]
    for index, part in enumerate(parts):
        if part in MEDIA_DIR_NAMES:
            if index + 1 < len(parts) and parts[index + 1] in SPLIT_NAMES:
                return True
            if index == len(parts) - 1:
                return True
    return False


def run_command(script_path: Path, conda_env: str) -> str:
    return f'conda run -n {conda_env} python "{script_path}"'


def render_detect_train_script(
    *,
    data_yaml: Path,
    save_dir: Path,
    options: TrainCodeOptions,
    script_path: Path | None = None,
) -> str:
    data_yaml = data_yaml.resolve()
    save_dir = save_dir.resolve()
    script_name = script_path.name if script_path is not None else SCRIPT_NAME
    model_name = detect_model_name(options.model_series, options.model_size, options.custom_model)
    return "\n".join(
        [
            '"""YOLO Detection 训练脚本（Ultralytics）。',
            "",
            f"Conda 环境: {options.conda_env}",
            "运行示例:",
            f"  conda run -n {options.conda_env} python {script_name}",
            "",
            "只做目标检测。不要改 TASK。",
            "本脚本只读取 data.yaml / images / labels，不会改写标注或图片。",
            '"""',
            "from __future__ import annotations",
            "",
            "from pathlib import Path",
            "",
            "",
            f"CONDA_ENV = {_py_literal(options.conda_env)}",
            f"DATA = {_path_literal(data_yaml)}",
            f"TASK = {_py_literal('detect')}",
            f"MODEL_SERIES = {_py_literal(options.model_series)}",
            f"MODEL_SIZE = {_py_literal(options.model_size)}",
            f"CUSTOM_MODEL = {_py_literal(options.custom_model)}  # 当前权重: {model_name}",
            f"EPOCHS = {int(options.epochs)}",
            f"IMGSZ = {int(options.imgsz)}",
            f"BATCH = {int(options.batch)}",
            f"DEVICE = {_py_literal(options.device)}",
            f"WORKERS = {int(options.workers)}",
            f"PATIENCE = {int(options.patience)}",
            f"SEED = {int(options.seed)}",
            f"CACHE = {options.cache!s}",
            f"AMP = {options.amp!s}",
            f"SAVE_DIR = {_path_literal(save_dir)}",
            f"RUN_NAME = {_py_literal(options.run_name)}",
            "",
            "",
            "def model_name() -> str:",
            "    if CUSTOM_MODEL.strip():",
            "        return CUSTOM_MODEL.strip()",
            '    return f"{MODEL_SERIES}{MODEL_SIZE}.pt"',
            "",
            "",
            "def main() -> None:",
            "    try:",
            "        from ultralytics import YOLO",
            "    except ImportError as exc:",
            '        raise SystemExit(',
            f'            "未安装 ultralytics。请先激活 conda 环境 {options.conda_env}，再运行本脚本。"',
            "        ) from exc",
            "",
            "    data_path = DATA.resolve()",
            "    if not data_path.is_file():",
            '        raise FileNotFoundError(f"找不到 data.yaml: {data_path}")',
            "",
            "    save_dir = SAVE_DIR.resolve()",
            '    project = save_dir.parent',
            '    name = RUN_NAME.strip() or save_dir.name',
            '    weights = model_name()',
            '    print("=" * 60)',
            "    print(f\"conda env : {CONDA_ENV}\")",
            "    print(f\"task      : {TASK}\")",
            "    print(f\"model     : {weights}\")",
            "    print(f\"data      : {data_path}\")",
            "    print(f\"project   : {project}\")",
            "    print(f\"name      : {name}\")",
            '    print("=" * 60)',
            "    if TASK != \"detect\":",
            '        raise SystemExit("本脚本只支持 YOLO Detection")',
            "    model = YOLO(weights)",
            "    model.train(",
            "        data=str(data_path),",
            "        task=TASK,",
            "        epochs=EPOCHS,",
            "        imgsz=IMGSZ,",
            "        batch=BATCH,",
            "        device=DEVICE,",
            "        project=str(project),",
            "        name=name,",
            "        workers=WORKERS,",
            "        patience=PATIENCE,",
            "        seed=SEED,",
            "        cache=CACHE,",
            "        amp=AMP,",
            "    )",
            "",
            "",
            'if __name__ == "__main__":',
            "    main()",
            "",
        ]
    )


def generate_detect_train(
    yolo_root: Path,
    output_dir: Path | None = None,
    options: TrainCodeOptions | None = None,
    log: LogFn | None = None,
) -> TrainCodeReport:
    options = options or TrainCodeOptions()
    report = TrainCodeReport()
    yolo_root = yolo_root.resolve()
    datasets = discover_yolo_datasets(yolo_root)
    if output_dir is not None:
        output_dir = output_dir.resolve()
        if is_unsafe_output_dir(output_dir):
            raise ConvertError(f"拒绝写入 images/labels: {output_dir}")
    _log(log, f"发现 {len(datasets)} 个 YOLO 检测数据集")
    for dataset_dir in datasets:
        yaml_path = dataset_dir / "data.yaml"
        relative = "." if dataset_dir == yolo_root else str(dataset_dir.relative_to(yolo_root)).replace("\\", "/")
        report.datasets.append(relative)
        if not (dataset_dir / "images" / "train").is_dir():
            report.warnings.append(f"{relative}: 缺少 images/train")
        if yaml_path.is_file() and ensure_absolute_data_yaml_path(yaml_path, dry_run=options.dry_run):
            msg = f"{relative}: 已将 data.yaml 的 path 改为绝对路径"
            if options.dry_run:
                msg += "（dry-run，未写盘）"
            report.warnings.append(msg)
            _log(log, msg)
        target_dir = dataset_dir if output_dir is None else output_dir
        if is_unsafe_output_dir(target_dir):
            report.errors.append(f"{relative}: 拒绝写入 images/labels: {target_dir}")
            continue
        script_path = target_dir / script_filename(yolo_root, dataset_dir, len(datasets))
        content = render_detect_train_script(
            data_yaml=yaml_path,
            save_dir=default_save_dir(dataset_dir),
            options=options,
            script_path=script_path,
        )
        command = run_command(script_path, options.conda_env)
        if script_path.exists():
            existing = script_path.read_text(encoding="utf-8")
            if existing == content:
                report.skipped.append(str(script_path))
                report.run_commands.append(command)
                _log(log, f"已存在且内容相同，跳过: {script_path}")
                continue
            if not options.overwrite:
                report.skipped.append(str(script_path))
                report.warnings.append(f"已存在，未覆盖: {script_path}")
                _log(log, f"已存在，未覆盖: {script_path}")
                continue
        if options.dry_run:
            report.written.append(str(script_path))
            report.run_commands.append(command)
            _log(log, f"dry-run 将写入: {script_path}")
            _log(log, f"data.yaml: {yaml_path}")
            continue
        target_dir.mkdir(parents=True, exist_ok=True)
        script_path.write_text(content, encoding="utf-8")
        report.written.append(str(script_path))
        report.run_commands.append(command)
        _log(log, f"已写入: {script_path}")
        _log(log, f"data.yaml: {yaml_path}")
        _log(log, f"运行: {command}")
    if not report.written and not report.skipped and not report.errors:
        report.errors.append("没有生成任何训练脚本")
    return report
