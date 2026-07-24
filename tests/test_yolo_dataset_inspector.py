from __future__ import annotations

from pathlib import Path

from PIL import Image

from modules.yolo.dataset_inspector import inspect_yolo_dataset


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), "white").save(path)


def test_inspect_yolo_dataset_counts_images_labels_empty_and_classes(tmp_path: Path):
    root = tmp_path / "dataset"
    train_images = root / "images" / "train"
    val_images = root / "images" / "val"
    train_labels = root / "labels" / "train"
    val_labels = root / "labels" / "val"
    _write_image(train_images / "a.png")
    _write_image(train_images / "b.jpg")
    _write_image(val_images / "c.png")
    train_labels.mkdir(parents=True)
    val_labels.mkdir(parents=True)
    (train_labels / "a.txt").write_text("0 0.500000 0.500000 0.250000 0.250000\n", encoding="utf-8")
    (train_labels / "b.txt").write_text("", encoding="utf-8")
    (val_labels / "c.txt").write_text("1 0.500000 0.500000 0.500000 0.500000\n", encoding="utf-8")
    data_yaml = root / "data.yaml"
    data_yaml.write_text(
        "path: .\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: non_stitch_defect\n"
        "  1: stitch_defect\n",
        encoding="utf-8",
    )

    report = inspect_yolo_dataset(data_yaml)

    assert report.ok
    assert report.dataset_root == root
    assert report.train_image_count == 2
    assert report.val_image_count == 1
    assert report.label_file_count == 3
    assert report.empty_label_count == 1
    assert report.box_count == 2
    assert report.class_counts == {0: 1, 1: 1}
    assert report.missing_label_count == 0
    assert report.label_without_image_count == 0
    assert report.invalid_label_count == 0
    assert report.class_names == {0: "non_stitch_defect", 1: "stitch_defect"}


def test_inspect_yolo_dataset_reports_missing_and_invalid_labels(tmp_path: Path):
    root = tmp_path / "dataset"
    _write_image(root / "images" / "train" / "a.png")
    _write_image(root / "images" / "train" / "missing_label.png")
    labels = root / "labels" / "train"
    labels.mkdir(parents=True)
    (labels / "a.txt").write_text("2 1.200000 0.500000 0.250000 0.250000\n", encoding="utf-8")
    (labels / "orphan.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
    data_yaml = root / "data.yaml"
    data_yaml.write_text(
        f"path: {root.as_posix()}\n"
        "train: images/train\n"
        "val: images/train\n"
        "names:\n"
        "  0: defect\n",
        encoding="utf-8",
    )

    report = inspect_yolo_dataset(data_yaml)

    assert not report.ok
    assert report.missing_label_count == 1
    assert report.label_without_image_count == 1
    assert report.invalid_label_count == 1
    assert report.class_counts == {}
    assert "missing_label" in report.missing_label_samples[0]
    assert "orphan" in report.label_without_image_samples[0]
