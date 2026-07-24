from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def yolo_box(points: list[list[float]], width: int, height: int) -> tuple[float, float, float, float]:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    x_min, x_max = max(0.0, min(xs)), min(float(width), max(xs))
    y_min, y_max = max(0.0, min(ys)), min(float(height), max(ys))
    box_w = max(0.0, x_max - x_min)
    box_h = max(0.0, y_max - y_min)
    x_center = x_min + box_w / 2.0
    y_center = y_min + box_h / 2.0
    return x_center / width, y_center / height, box_w / width, box_h / height


def load_labelme(json_path: Path, class_to_id: dict[str, int]) -> tuple[list[str], int, int]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    width = int(data["imageWidth"])
    height = int(data["imageHeight"])
    lines: list[str] = []
    for shape in data.get("shapes", []):
        label = shape.get("label", "")
        if label not in class_to_id:
            raise ValueError(f"Unknown label {label!r} in {json_path}")
        points = shape.get("points", [])
        if len(points) < 2:
            continue
        x, y, w, h = yolo_box(points, width, height)
        if w <= 0 or h <= 0:
            continue
        lines.append(f"{class_to_id[label]} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
    return lines, width, height


def collect_items(src_dir: Path) -> tuple[list[Path], list[str]]:
    images = sorted(
        path
        for path in src_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    labels = sorted(
        {
            shape.get("label", "")
            for json_path in src_dir.rglob("*.json")
            for shape in json.loads(json_path.read_text(encoding="utf-8")).get("shapes", [])
            if shape.get("label")
        }
    )
    return images, labels


def write_yaml(path: Path, root: Path, names: list[str]) -> None:
    rel_root = root.as_posix()
    lines = [
        f"path: {rel_root}",
        "train: images/train",
        "val: images/val",
        "names:",
    ]
    lines.extend(f"  {idx}: {name}" for idx, name in enumerate(names))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def prepare(src_dir: Path, out_dir: Path, val_ratio: float, seed: int) -> None:
    images, names = collect_items(src_dir)
    if not images:
        raise ValueError(f"No images found in {src_dir}")
    if not names:
        raise ValueError(f"No labels found in {src_dir}")

    class_to_id = {name: idx for idx, name in enumerate(names)}
    rng = random.Random(seed)
    shuffled = images[:]
    rng.shuffle(shuffled)
    val_count = max(1, round(len(shuffled) * val_ratio))
    val_images = set(shuffled[:val_count])

    for split in ("train", "val"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    missing_json: list[Path] = []
    for image_path in images:
        split = "val" if image_path in val_images else "train"
        dest_image = out_dir / "images" / split / image_path.name
        shutil.copy2(image_path, dest_image)

        json_path = image_path.with_suffix(".json")
        if json_path.exists():
            label_lines, _, _ = load_labelme(json_path, class_to_id)
        else:
            label_lines = []
            missing_json.append(image_path)
        dest_label = out_dir / "labels" / split / f"{image_path.stem}.txt"
        dest_label.write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")

    write_yaml(out_dir / "data.yaml", out_dir.resolve(), names)

    print(f"Prepared {len(images)} images at {out_dir}")
    print(f"Classes: {', '.join(names)}")
    print(f"Train: {len(images) - len(val_images)}, Val: {len(val_images)}")
    if missing_json:
        print("Images without JSON were written with empty labels:")
        for path in missing_json:
            print(f"  {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert PullTail Labelme JSON annotations to YOLO format.")
    parser.add_argument("--src-dir", type=Path, required=True,
                        help="LabelMe 标注源目录（包含图片和 JSON）")
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="YOLO 格式输出目录")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    prepare(args.src_dir, args.out_dir, args.val_ratio, args.seed)


if __name__ == "__main__":
    main()
