from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
SPLITS = ("train", "val")
XAL_VERSION = "2.5.4"
MANIFEST_FIELDS = ["dataset", "face", "split", "image", "json", "shape_count"]
LogFn = Callable[[str], None]


class ConvertError(ValueError):
    """转换输入或标注无法安全处理。"""


@dataclass
class ConvertOptions:
    overwrite_json: bool = False
    overwrite_labels: bool = True
    copy_images: bool = True
    overwrite_images: bool = False
    dry_run: bool = False
    unmatched_split: str | None = None


@dataclass
class ConvertReport:
    direction: str
    datasets: list[str] = field(default_factory=list)
    images: int = 0
    json_written: int = 0
    json_skipped: int = 0
    labels_written: int = 0
    labels_skipped: int = 0
    images_copied: int = 0
    images_skipped: int = 0
    empty_labels: int = 0
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def as_dict(self) -> dict[str, Any]:
        return {
            "direction": self.direction,
            "ok": self.ok,
            "datasets": self.datasets,
            "images": self.images,
            "json_written": self.json_written,
            "json_skipped": self.json_skipped,
            "labels_written": self.labels_written,
            "labels_skipped": self.labels_skipped,
            "images_copied": self.images_copied,
            "images_skipped": self.images_skipped,
            "empty_labels": self.empty_labels,
            "warnings": self.warnings,
            "errors": self.errors,
        }


def _log(log: LogFn | None, message: str) -> None:
    if log is not None:
        log(message)


def image_size(path: Path) -> tuple[int, int]:
    try:
        from PIL import Image

        with Image.open(path) as image:
            return int(image.size[0]), int(image.size[1])
    except ImportError:
        pass
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ConvertError(f"无法读取图像尺寸: {path}")
    height, width = image.shape[:2]
    return int(width), int(height)


def parse_names(raw_names: Any) -> list[str]:
    if isinstance(raw_names, list):
        return [str(name) for name in raw_names]
    if isinstance(raw_names, dict):
        items: list[tuple[int, str]] = []
        for key, value in raw_names.items():
            items.append((int(key), str(value)))
        return [name for _, name in sorted(items)]
    raise ConvertError("data.yaml 的 names 必须是列表或 id→名称映射")


def load_yaml_names(yaml_path: Path) -> list[str]:
    try:
        import yaml
    except ImportError as exc:
        raise ConvertError("需要 PyYAML 才能读取 data.yaml") from exc
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ConvertError(f"data.yaml 无效: {yaml_path}")
    return parse_names(data.get("names"))


def read_classes_txt(path: Path) -> list[str]:
    names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not names:
        raise ConvertError(f"classes.txt 为空: {path}")
    return names


def write_classes_txt(path: Path, names: list[str], *, dry_run: bool) -> None:
    content = "\n".join(names) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") == content:
        return
    if dry_run:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_data_yaml(path: Path, names: list[str], *, dry_run: bool) -> None:
    # Ultralytics 会把相对 path 按当前工作目录解析，因此写入 data.yaml 所在目录的绝对路径。
    dataset_root = path.parent.resolve().as_posix()
    lines = [
        f"path: {dataset_root}",
        "train: images/train",
        "val: images/val",
        "names:",
        *(f"  {index}: {name}" for index, name in enumerate(names)),
        "",
    ]
    content = "\n".join(lines)
    if path.exists() and path.read_text(encoding="utf-8") == content:
        return
    if dry_run:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def discover_yolo_datasets(root: Path) -> list[Path]:
    root = root.resolve()
    if not root.is_dir():
        raise ConvertError(f"YOLO 目录不存在: {root}")
    if (root / "data.yaml").is_file():
        return [root]
    found = []
    for yaml_path in sorted(root.rglob("data.yaml")):
        if "xanylabeling" in yaml_path.parts:
            continue
        found.append(yaml_path.parent)
    if not found:
        raise ConvertError(f"未找到 data.yaml: {root}")
    return found


def discover_xal_datasets(root: Path) -> list[Path]:
    root = root.resolve()
    if not root.is_dir():
        raise ConvertError(f"X-AnyLabeling 目录不存在: {root}")
    if (root / "classes.txt").is_file():
        return [root]
    found = [path.parent for path in sorted(root.rglob("classes.txt"))]
    if not found:
        raise ConvertError(f"未找到 classes.txt: {root}")
    return found


def dataset_key(root: Path, dataset_dir: Path) -> Path:
    try:
        relative = dataset_dir.resolve().relative_to(root.resolve())
    except ValueError:
        return Path(dataset_dir.name)
    return Path(".") if str(relative) == "." else relative


def split_key(relative: Path) -> tuple[str, str]:
    parts = relative.parts
    if relative == Path(".") or not parts:
        return ".", ""
    if len(parts) == 1:
        return parts[0], ""
    return parts[0], str(Path(*parts[1:]).as_posix())


def collect_images(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def yolo_to_xyxy(cx: float, cy: float, box_w: float, box_h: float, width: int, height: int) -> tuple[float, float, float, float]:
    x1 = max(0.0, (cx - box_w / 2.0) * width)
    y1 = max(0.0, (cy - box_h / 2.0) * height)
    x2 = min(float(width - 1), (cx + box_w / 2.0) * width)
    y2 = min(float(height - 1), (cy + box_h / 2.0) * height)
    if x2 <= x1 or y2 <= y1:
        raise ConvertError("YOLO 框转换后无效")
    return x1, y1, x2, y2


def xyxy_to_yolo(x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> tuple[float, float, float, float]:
    left = max(0.0, min(x1, x2))
    top = max(0.0, min(y1, y2))
    x_right = min(float(width), max(x1, x2))
    y_bottom = min(float(height), max(y1, y2))
    box_w = x_right - left
    box_h = y_bottom - top
    if box_w <= 0 or box_h <= 0:
        raise ConvertError("矩形框无效")
    return (
        ((left + x_right) / 2.0) / width,
        ((top + y_bottom) / 2.0) / height,
        box_w / width,
        box_h / height,
    )


def rectangle_shape(label: str, x1: float, y1: float, x2: float, y2: float) -> dict[str, Any]:
    return {
        "label": label,
        "points": [[float(x1), float(y1)], [float(x2), float(y2)]],
        "group_id": None,
        "description": "",
        "difficult": False,
        "shape_type": "rectangle",
        "flags": {},
        "attributes": {},
    }


def annotation_payload(image_path: Path, width: int, height: int, shapes: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "version": XAL_VERSION,
        "flags": {},
        "shapes": shapes,
        "imagePath": image_path.name,
        "imageData": None,
        "imageHeight": int(height),
        "imageWidth": int(width),
    }


def read_yolo_shapes(label_path: Path, names: list[str], width: int, height: int) -> list[dict[str, Any]]:
    if not label_path.exists():
        raise ConvertError(f"缺少 YOLO 标签: {label_path}")
    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    shapes = []
    for line_number, line in enumerate(text.splitlines(), 1):
        parts = line.split()
        if len(parts) != 5:
            raise ConvertError(f"无效 YOLO 标签 {label_path}:{line_number}: {line}")
        class_id = int(parts[0])
        if class_id < 0 or class_id >= len(names):
            raise ConvertError(f"类别越界 {label_path}:{line_number}: {class_id}")
        cx, cy, box_w, box_h = (float(value) for value in parts[1:])
        if any(value < 0.0 or value > 1.0 for value in (cx, cy, box_w, box_h)) or box_w <= 0 or box_h <= 0:
            raise ConvertError(f"坐标越界 {label_path}:{line_number}: {line}")
        x1, y1, x2, y2 = yolo_to_xyxy(cx, cy, box_w, box_h, width, height)
        shapes.append(rectangle_shape(names[class_id], x1, y1, x2, y2))
    return shapes


def shape_to_yolo_line(shape: dict[str, Any], names: list[str], width: int, height: int) -> str:
    label = str(shape.get("label", ""))
    if label not in names:
        raise ConvertError(f"未知类别: {label}")
    shape_type = str(shape.get("shape_type") or "rectangle")
    points = shape.get("points") or []
    # X-AnyLabeling 2.x 用两点对角；4.x 常用四角点。检测训练都取轴对齐外接框。
    if shape_type not in {"rectangle", "rotation"} or len(points) < 2:
        raise ConvertError(f"仅支持检测矩形，收到 {shape_type} / {len(points)} 点")
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    cx, cy, box_w, box_h = xyxy_to_yolo(min(xs), min(ys), max(xs), max(ys), width, height)
    class_id = names.index(label)
    return f"{class_id} {cx:.6f} {cy:.6f} {box_w:.6f} {box_h:.6f}"


def copy_image(source: Path, target: Path, options: ConvertOptions, report: ConvertReport) -> None:
    if target.exists() and not options.overwrite_images:
        report.images_skipped += 1
        return
    if not options.copy_images and target.exists():
        report.images_skipped += 1
        return
    if not options.copy_images and not target.exists():
        raise ConvertError(f"目标图片不存在且未启用复制: {target}")
    if options.dry_run:
        report.images_copied += 1
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    report.images_copied += 1


def write_json(path: Path, payload: dict[str, Any], options: ConvertOptions, report: ConvertReport) -> None:
    if path.exists() and not options.overwrite_json:
        report.json_skipped += 1
        return
    if options.dry_run:
        report.json_written += 1
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report.json_written += 1


def write_label(path: Path, lines: list[str], options: ConvertOptions, report: ConvertReport) -> None:
    if path.exists() and not options.overwrite_labels:
        report.labels_skipped += 1
        return
    if options.dry_run:
        report.labels_written += 1
        if not lines:
            report.empty_labels += 1
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    report.labels_written += 1
    if not lines:
        report.empty_labels += 1


def read_split_manifest(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    result: dict[str, str] = {}
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            image = Path(str(row.get("image", ""))).name
            split = str(row.get("split", "")).strip()
            if image and split in SPLITS:
                result[image] = split
                result[Path(image).stem] = split
            json_name = Path(str(row.get("json", ""))).name
            if json_name and split in SPLITS:
                result[json_name] = split
                result[Path(json_name).stem] = split
    return result


def collect_manifest_splits(*roots: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    seen: set[Path] = set()
    for root in roots:
        current = root.resolve()
        for _ in range(8):
            path = current / "split_manifest.csv"
            if path not in seen and path.is_file():
                seen.add(path)
                for key, split in read_split_manifest(path).items():
                    result.setdefault(key, split)
            if current.parent == current:
                break
            current = current.parent
    return result


def resolve_split(
    *,
    dataset_dir: Path,
    stem: str,
    image_name: str,
    manifest_splits: dict[str, str],
    options: ConvertOptions,
    report: ConvertReport,
    source: Path,
) -> tuple[str, Path]:
    located = locate_yolo_image(dataset_dir, stem)
    if located is not None:
        return located
    split = (
        manifest_splits.get(image_name)
        or manifest_splits.get(stem)
        or options.unmatched_split
        or "train"
    )
    if split not in SPLITS:
        raise ConvertError("无法确定 train/val")
    if stem not in manifest_splits and image_name not in manifest_splits and options.unmatched_split not in SPLITS:
        report.warnings.append(f"{source.name}: 空目录或缺少划分记录，写入 {split}")
    suffix = Path(image_name).suffix or ".png"
    return split, dataset_dir / "images" / split / f"{stem}{suffix}"


def write_split_manifest(path: Path, rows: list[dict[str, str]], *, replace_keys: set[tuple[str, str]], dry_run: bool) -> None:
    existing: list[dict[str, str]] = []
    if path.exists():
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            existing = [
                row
                for row in csv.DictReader(handle)
                if (row.get("dataset", ""), row.get("face", "")) not in replace_keys
            ]
    merged = existing + rows
    if dry_run:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(merged)


def locate_yolo_image(dataset_dir: Path, stem: str) -> tuple[str, Path] | None:
    hits = []
    for split in SPLITS:
        for suffix in IMAGE_SUFFIXES:
            candidate = dataset_dir / "images" / split / f"{stem}{suffix}"
            if candidate.is_file():
                hits.append((split, candidate))
    if len(hits) > 1:
        raise ConvertError(f"{stem} 同时存在于 train 和 val: {dataset_dir}")
    return hits[0] if hits else None


def yolo_to_xal(yolo_root: Path, xal_root: Path, options: ConvertOptions, log: LogFn | None = None) -> ConvertReport:
    report = ConvertReport(direction="yolo_to_xal")
    if not xal_root.exists() and not options.dry_run:
        xal_root.mkdir(parents=True, exist_ok=True)
        _log(log, f"已创建空 X-AnyLabeling 目录: {xal_root}")
    datasets = discover_yolo_datasets(yolo_root)
    single = (yolo_root / "data.yaml").is_file()
    manifest_rows: list[dict[str, str]] = []
    replace_keys: set[tuple[str, str]] = set()
    _log(log, f"发现 {len(datasets)} 个 YOLO 检测数据集")
    for dataset_dir in datasets:
        names = load_yaml_names(dataset_dir / "data.yaml")
        relative = Path(".") if single else dataset_key(yolo_root, dataset_dir)
        target_dir = xal_root if single else xal_root / relative
        dataset_name, face = split_key(relative)
        replace_keys.add((dataset_name, face))
        report.datasets.append(str(relative).replace("\\", "/"))
        write_classes_txt(target_dir / "classes.txt", names, dry_run=options.dry_run)
        image_count = 0
        for split in SPLITS:
            for image_path in collect_images(dataset_dir / "images" / split):
                image_count += 1
                report.images += 1
                width, height = image_size(image_path)
                label_path = dataset_dir / "labels" / split / f"{image_path.stem}.txt"
                try:
                    shapes = read_yolo_shapes(label_path, names, width, height)
                    target_image = target_dir / image_path.name
                    target_json = target_dir / f"{image_path.stem}.json"
                    copy_image(image_path, target_image, options, report)
                    write_json(
                        target_json,
                        annotation_payload(target_image, width, height, shapes),
                        options,
                        report,
                    )
                    manifest_rows.append(
                        {
                            "dataset": dataset_name,
                            "face": face,
                            "split": split,
                            "image": str(target_image if not options.dry_run else target_dir / image_path.name),
                            "json": str(target_json),
                            "shape_count": str(len(shapes)),
                        }
                    )
                except Exception as error:
                    report.errors.append(f"{image_path}: {error}")
        _log(log, f"{relative}: {image_count} 张")
    write_split_manifest(xal_root / "split_manifest.csv", manifest_rows, replace_keys=replace_keys, dry_run=options.dry_run)
    return report


def xal_to_yolo(yolo_root: Path, xal_root: Path, options: ConvertOptions, log: LogFn | None = None) -> ConvertReport:
    report = ConvertReport(direction="xal_to_yolo")
    yolo_root = yolo_root.resolve()
    xal_root = xal_root.resolve()
    if not yolo_root.exists():
        if options.dry_run:
            _log(log, f"目标 YOLO 目录不存在，将创建: {yolo_root}")
        else:
            yolo_root.mkdir(parents=True, exist_ok=True)
            _log(log, f"已创建空 YOLO 目录: {yolo_root}")
    yolo_exists = yolo_root.is_dir() and ((yolo_root / "data.yaml").is_file() or any(yolo_root.rglob("data.yaml")))
    xal_datasets = discover_xal_datasets(xal_root)
    single_xal = (xal_root / "classes.txt").is_file()
    single_yolo = (yolo_root / "data.yaml").is_file()
    manifest_splits = collect_manifest_splits(xal_root, *xal_datasets)
    _log(log, f"发现 {len(xal_datasets)} 个 X-AnyLabeling 检测目录")
    if not yolo_exists:
        _log(log, "目标是空目录：优先使用 split_manifest，否则写入 train")
    for xal_dir in xal_datasets:
        relative = Path(".") if single_xal else dataset_key(xal_root, xal_dir)
        dataset_dir = yolo_root if (single_xal or single_yolo) and len(xal_datasets) == 1 else yolo_root / relative
        names = read_classes_txt(xal_dir / "classes.txt")
        yaml_path = dataset_dir / "data.yaml"
        if yaml_path.exists():
            yolo_names = load_yaml_names(yaml_path)
            if yolo_names != names:
                report.errors.append(f"{relative}: classes.txt 与 data.yaml 类别不一致: {names} vs {yolo_names}")
                continue
        else:
            write_data_yaml(yaml_path, names, dry_run=options.dry_run)
        report.datasets.append(str(relative).replace("\\", "/"))
        json_files = sorted(path for path in xal_dir.glob("*.json") if path.name != "classes.txt")
        local_manifest = collect_manifest_splits(xal_dir)
        merged_splits = {**manifest_splits, **local_manifest}
        for json_path in json_files:
            try:
                data = json.loads(json_path.read_text(encoding="utf-8"))
                image_name = str(data.get("imagePath") or f"{json_path.stem}.png")
                image_path = xal_dir / Path(image_name).name
                if not image_path.is_file():
                    matches = [path for path in collect_images(xal_dir) if path.stem == json_path.stem]
                    if not matches:
                        raise ConvertError("JSON 没有配对图片")
                    image_path = matches[0]
                width = int(data.get("imageWidth") or 0)
                height = int(data.get("imageHeight") or 0)
                if width <= 0 or height <= 0:
                    width, height = image_size(image_path)
                split, dest_image = resolve_split(
                    dataset_dir=dataset_dir,
                    stem=json_path.stem,
                    image_name=image_path.name,
                    manifest_splits=merged_splits,
                    options=options,
                    report=report,
                    source=json_path,
                )
                report.images += 1
                copy_image(image_path, dest_image, options, report)
                lines = [shape_to_yolo_line(shape, names, width, height) for shape in data.get("shapes", [])]
                write_label(dataset_dir / "labels" / split / f"{json_path.stem}.txt", lines, options, report)
            except Exception as error:
                report.errors.append(f"{json_path}: {error}")
        _log(log, f"{relative}: {len(json_files)} 个 JSON")
    if not yolo_exists and not options.dry_run:
        _log(log, f"已创建 YOLO 目录: {yolo_root}")
    return report


def convert(direction: str, yolo_root: Path, xal_root: Path, options: ConvertOptions, log: LogFn | None = None) -> ConvertReport:
    if direction == "yolo_to_xal":
        report = yolo_to_xal(yolo_root, xal_root, options, log)
    elif direction == "xal_to_yolo":
        report = xal_to_yolo(yolo_root, xal_root, options, log)
    else:
        raise ConvertError(f"未知方向: {direction}")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="YOLO Detection 与 X-AnyLabeling 矩形标注互转，并可生成检测训练脚本")
    parser.add_argument("direction", choices=["yolo-to-xal", "xal-to-yolo", "ui", "gen-train"])
    parser.add_argument("--yolo", type=Path, help="YOLO 检测数据集目录，含 data.yaml 或若干子数据集")
    parser.add_argument("--xal", type=Path, help="X-AnyLabeling 目录")
    parser.add_argument("--output", type=Path, help="gen-train 时训练脚本输出目录，默认写到各数据集 data.yaml 旁")
    parser.add_argument("--overwrite-json", action="store_true", help="YOLO→XAL 时覆盖已有 JSON")
    parser.add_argument("--no-overwrite-labels", action="store_true", help="XAL→YOLO 时不覆盖已有 txt")
    parser.add_argument("--no-copy-images", action="store_true")
    parser.add_argument("--overwrite-images", action="store_true")
    parser.add_argument("--overwrite-script", action="store_true", help="gen-train 时覆盖已有 train_detect.py")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--unmatched-split", choices=list(SPLITS), help="XAL 中找不到对应 YOLO 图片时写入该划分")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.direction == "ui":
        try:
            from .convert_xanylabeling_ui import launch
        except ImportError:
            from convert_xanylabeling_ui import launch

        launch()
        return 0
    if args.direction == "gen-train":
        if args.yolo is None:
            parser.error("gen-train 需要 --yolo")
        try:
            from .generate_detect_train import TrainCodeOptions, generate_detect_train
        except ImportError:
            from generate_detect_train import TrainCodeOptions, generate_detect_train

        try:
            report = generate_detect_train(
                args.yolo,
                output_dir=args.output,
                options=TrainCodeOptions(overwrite=bool(args.overwrite_script), dry_run=bool(args.dry_run)),
                log=print,
            )
        except ConvertError as error:
            print(json.dumps({"ok": False, "errors": [str(error)]}, ensure_ascii=False, indent=2))
            return 1
        print(json.dumps(report.as_dict(), ensure_ascii=False, indent=2))
        return 0 if report.ok else 1
    if args.yolo is None or args.xal is None:
        parser.error("需要 --yolo 和 --xal")
    options = ConvertOptions(
        overwrite_json=bool(args.overwrite_json),
        overwrite_labels=not bool(args.no_overwrite_labels),
        copy_images=not bool(args.no_copy_images),
        overwrite_images=bool(args.overwrite_images),
        dry_run=bool(args.dry_run),
        unmatched_split=args.unmatched_split,
    )
    direction = "yolo_to_xal" if args.direction == "yolo-to-xal" else "xal_to_yolo"
    report = convert(direction, args.yolo, args.xal, options, log=print)
    print(json.dumps(report.as_dict(), ensure_ascii=False, indent=2))
    return 0 if report.ok else 1


if __name__ == "__main__":
    import sys

    modules_root = Path(__file__).resolve().parent.parent
    if str(modules_root) not in sys.path:
        sys.path.insert(0, str(modules_root))
    from yolo.convert_xanylabeling import main as package_main

    raise SystemExit(package_main())
