from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - ultralytics normally installs PyYAML
    yaml = None


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class YoloDatasetReport:
    data_yaml: Path
    dataset_root: Path
    train_path: Path | None = None
    val_path: Path | None = None
    class_names: dict[int, str] = field(default_factory=dict)
    train_image_count: int = 0
    val_image_count: int = 0
    label_file_count: int = 0
    empty_label_count: int = 0
    box_count: int = 0
    class_counts: dict[int, int] = field(default_factory=dict)
    missing_label_count: int = 0
    label_without_image_count: int = 0
    invalid_label_count: int = 0
    missing_label_samples: list[str] = field(default_factory=list)
    label_without_image_samples: list[str] = field(default_factory=list)
    invalid_label_samples: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors and self.missing_label_count == 0 and self.label_without_image_count == 0 and self.invalid_label_count == 0

    @property
    def total_image_count(self) -> int:
        return self.train_image_count + self.val_image_count


def inspect_yolo_dataset(data_yaml: str | Path) -> YoloDatasetReport:
    yaml_path = Path(data_yaml).expanduser().resolve()
    dataset_root = yaml_path.parent
    report = YoloDatasetReport(data_yaml=yaml_path, dataset_root=dataset_root)

    if not yaml_path.exists():
        report.errors.append(f"data.yaml 不存在: {yaml_path}")
        return report

    try:
        config = _load_yaml(yaml_path)
    except Exception as exc:  # noqa: BLE001 - shown directly in GUI
        report.errors.append(f"data.yaml 解析失败: {exc}")
        return report

    root_value = config.get("path")
    if root_value:
        root_path = Path(str(root_value)).expanduser()
        dataset_root = root_path if root_path.is_absolute() else (yaml_path.parent / root_path)
        dataset_root = dataset_root.resolve()
        report.dataset_root = dataset_root

    report.class_names = _parse_names(config.get("names"))
    train_path = _resolve_split_path(config.get("train"), dataset_root)
    val_path = _resolve_split_path(config.get("val"), dataset_root)
    report.train_path = train_path
    report.val_path = val_path

    image_to_label: dict[Path, Path] = {}
    label_to_image: dict[Path, Path] = {}
    if train_path is not None:
        train_images = _collect_images(train_path)
        report.train_image_count = len(train_images)
        image_to_label.update({image: _label_path_for_image(image, train_path) for image in train_images})
    if val_path is not None and val_path != train_path:
        val_images = _collect_images(val_path)
        report.val_image_count = len(val_images)
        image_to_label.update({image: _label_path_for_image(image, val_path) for image in val_images})
    elif val_path is not None:
        report.val_image_count = report.train_image_count

    for image_path, label_path in image_to_label.items():
        label_to_image[label_path] = image_path
        if not label_path.exists():
            report.missing_label_count += 1
            _append_sample(report.missing_label_samples, image_path.stem)
            continue
        _inspect_label_file(label_path, report)

    split_label_dirs = {label.parent for label in image_to_label.values()}
    for label_dir in sorted(split_label_dirs):
        if not label_dir.exists():
            continue
        for label_path in sorted(label_dir.rglob("*.txt")):
            if label_path not in label_to_image:
                report.label_without_image_count += 1
                _append_sample(report.label_without_image_samples, label_path.stem)

    return report


def _load_yaml(path: Path) -> dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML 未安装，无法解析 data.yaml")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("顶层结构必须是 YAML 对象")
    return data


def _parse_names(raw_names: Any) -> dict[int, str]:
    if isinstance(raw_names, list):
        return {idx: str(name) for idx, name in enumerate(raw_names)}
    if isinstance(raw_names, dict):
        parsed: dict[int, str] = {}
        for key, value in raw_names.items():
            try:
                parsed[int(key)] = str(value)
            except (TypeError, ValueError):
                continue
        return parsed
    return {}


def _resolve_split_path(raw_path: Any, dataset_root: Path) -> Path | None:
    if raw_path in (None, ""):
        return None
    if isinstance(raw_path, list):
        raw_path = raw_path[0] if raw_path else None
    if raw_path in (None, ""):
        return None
    path = Path(str(raw_path)).expanduser()
    return path.resolve() if path.is_absolute() else (dataset_root / path).resolve()


def _collect_images(path: Path) -> list[Path]:
    if not path.exists():
        return []
    if path.is_file():
        return [line_path for line_path in _read_image_list(path) if line_path.exists()]
    return [p.resolve() for p in sorted(path.rglob("*")) if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]


def _read_image_list(path: Path) -> list[Path]:
    result: list[Path] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw:
            continue
        item = Path(raw).expanduser()
        result.append(item.resolve() if item.is_absolute() else (path.parent / item).resolve())
    return result


def _label_path_for_image(image_path: Path, split_image_root: Path) -> Path:
    try:
        relative = image_path.relative_to(split_image_root)
    except ValueError:
        relative = Path(image_path.name)
    parts = list(split_image_root.parts)
    if "images" in parts:
        index = len(parts) - 1 - parts[::-1].index("images")
        label_root = Path(*parts[:index], "labels", *parts[index + 1:])
    else:
        label_root = split_image_root.parent / "labels" / split_image_root.name
    return (label_root / relative).with_suffix(".txt").resolve()


def _inspect_label_file(label_path: Path, report: YoloDatasetReport) -> None:
    report.label_file_count += 1
    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        report.empty_label_count += 1
        return
    for line_no, line in enumerate(text.splitlines(), 1):
        parts = line.split()
        if len(parts) != 5:
            _mark_invalid(report, label_path, line_no, line)
            continue
        try:
            class_id = int(parts[0])
            values = [float(item) for item in parts[1:]]
        except ValueError:
            _mark_invalid(report, label_path, line_no, line)
            continue
        if class_id not in report.class_names or any(value < 0.0 or value > 1.0 for value in values) or values[2] <= 0 or values[3] <= 0:
            _mark_invalid(report, label_path, line_no, line)
            continue
        report.box_count += 1
        report.class_counts[class_id] = report.class_counts.get(class_id, 0) + 1


def _mark_invalid(report: YoloDatasetReport, label_path: Path, line_no: int, line: str) -> None:
    report.invalid_label_count += 1
    _append_sample(report.invalid_label_samples, f"{label_path.name}:{line_no}: {line}")


def _append_sample(samples: list[str], value: str, limit: int = 8) -> None:
    if len(samples) < limit:
        samples.append(value)
