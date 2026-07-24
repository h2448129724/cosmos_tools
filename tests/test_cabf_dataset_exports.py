from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from cabf import (
    export_master_to_model_a,
    export_master_to_model_b,
    make_empty_master_annotation,
    read_json,
    validate_master_dataset,
)


def _write_image(path: Path, width: int = 32, height: int = 24) -> None:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    path.write_bytes(encoded.tobytes())


def test_validate_master_dataset_summary(tmp_path: Path):
    image_dir = tmp_path / "images"
    annotation_dir = tmp_path / "annotations"
    image_dir.mkdir()
    annotation_dir.mkdir()

    _write_image(image_dir / "sample_ok.png")
    _write_image(image_dir / "sample_missing.png")

    ann = make_empty_master_annotation("sample_ok.png", 32, 24, "sample_ok")
    ann["points"] = [
        {"id": 0, "x": 8.0, "y": 10.0, "score": 1.0, "source": "manual"},
        {"id": 1, "x": 20.0, "y": 12.0, "score": 1.0, "source": "manual"},
    ]
    ann["edges"] = [{"edge_id": "edge_0001", "src": 0, "dst": 1, "label": 1, "source": "manual"}]
    (annotation_dir / "sample_ok.json").write_text(__import__("json").dumps(ann, ensure_ascii=False, indent=2), encoding="utf-8")

    report = validate_master_dataset(image_dir, annotation_dir)
    assert report["summary"]["num_images"] == 2
    assert report["summary"]["paired_samples"] == 1
    assert report["summary"]["missing_annotations"] == 1
    assert report["missing_annotations"] == ["sample_missing"]


def test_export_model_a_and_b(tmp_path: Path):
    image_dir = tmp_path / "images"
    annotation_dir = tmp_path / "annotations"
    output_a = tmp_path / "model_a_export"
    output_b = tmp_path / "model_b_export"
    image_dir.mkdir()
    annotation_dir.mkdir()

    _write_image(image_dir / "sample.png")

    ann = make_empty_master_annotation("sample.png", 32, 24, "sample")
    ann["points"] = [
        {"id": 0, "x": 8.0, "y": 10.0, "score": 1.0, "source": "manual"},
        {"id": 1, "x": 20.0, "y": 12.0, "score": 1.0, "source": "manual"},
    ]
    ann["edges"] = [{"edge_id": "edge_0001", "src": 0, "dst": 1, "label": 1, "source": "manual"}]
    (annotation_dir / "sample.json").write_text(__import__("json").dumps(ann, ensure_ascii=False, indent=2), encoding="utf-8")

    result_a = export_master_to_model_a(image_dir, annotation_dir, output_a)
    result_b = export_master_to_model_b(image_dir, annotation_dir, output_b)

    assert result_a["images_exported"] == 1
    assert result_a["annotations_exported"] == 1
    assert (output_a / "images" / "sample.png").exists()
    labelme = read_json(output_a / "annotations" / "sample.json")
    assert labelme["imageWidth"] == 32
    assert len(labelme["shapes"]) == 2

    assert result_b["images_exported"] == 1
    assert result_b["annotations_exported"] == 1
    assert (output_b / "images" / "sample.png").exists()
    master = read_json(output_b / "annotations" / "sample.json")
    assert len(master["points"]) == 2
    assert len(master["edges"]) == 1
