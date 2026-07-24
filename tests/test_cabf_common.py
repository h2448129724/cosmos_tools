"""共享层 cabf_common 边界测试。

覆盖：缺文件、损坏 JSON、尺寸不一致、重复 ID、无效边引用、
自环、重复边、Model A/B error 分流、LabelMe ↔ master 互转。
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from cabf.constants import MASTER_SCHEMA_VERSION
from cabf.dataset import (
    export_master_to_model_a,
    export_master_to_model_b,
    summarize_validation,
    summarize_validation_findings,
    validate_master_dataset,
)
from cabf.io import iter_image_files, iter_json_files, read_image_bgr, read_json, write_image
from cabf.normalize import normalize_edges_for_editor, normalize_master_annotation, normalize_points_for_editor
from cabf.schema import (
    convert_labelme_to_master,
    is_labelme_point_annotation,
    load_labelme_points,
    make_empty_master_annotation,
    master_to_labelme,
)


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def _write_image(path: Path, width: int = 32, height: int = 24) -> None:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    path.write_bytes(encoded.tobytes())


def _write_ann(path: Path, ann: dict) -> None:
    path.write_text(json.dumps(ann, ensure_ascii=False, indent=2), encoding="utf-8")


def _base_ann(sample_id: str = "s1", width: int = 32, height: int = 24) -> dict:
    return make_empty_master_annotation(f"{sample_id}.png", width, height, sample_id)


# ---------------------------------------------------------------------------
# 1. 缺文件场景
# ---------------------------------------------------------------------------


class TestMissingFiles:
    def test_iter_image_files_missing_dir(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="图片目录不存在"):
            iter_image_files(tmp_path / "nope")

    def test_iter_json_files_missing_dir(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="标注目录不存在"):
            iter_json_files(tmp_path / "nope")

    def test_read_json_missing_file(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            read_json(tmp_path / "missing.json")

    def test_read_image_size_missing_file(self, tmp_path: Path):
        from cabf.io import read_image_size
        with pytest.raises(FileNotFoundError):
            read_image_size(tmp_path / "missing.png")

    def test_validate_dataset_missing_annotation(self, tmp_path: Path):
        img_dir = tmp_path / "images"
        ann_dir = tmp_path / "annotations"
        img_dir.mkdir()
        ann_dir.mkdir()
        _write_image(img_dir / "sample.png")

        report = validate_master_dataset(img_dir, ann_dir)
        assert report["summary"]["missing_annotations"] == 1
        assert "sample" in report["missing_annotations"]


# ---------------------------------------------------------------------------
# 1b. Unicode-safe image read/write primitives
# ---------------------------------------------------------------------------


class TestImageIoPrimitives:
    def test_read_image_bgr_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            read_image_bgr(tmp_path / "missing.png")

    def test_write_then_read_roundtrip(self, tmp_path: Path):
        original = np.zeros((24, 32, 3), dtype=np.uint8)
        original[:, :, 1] = 123  # green channel
        target = tmp_path / "out" / "img.png"
        write_image(target, original)
        assert target.exists()
        decoded = read_image_bgr(target)
        assert decoded.shape == (24, 32, 3)
        assert np.array_equal(decoded[:, :, 1], np.full((24, 32), 123, dtype=np.uint8))

    def test_write_image_creates_parent_dirs(self, tmp_path: Path):
        nested = tmp_path / "deep" / "nested" / "dir" / "x.bmp"
        write_image(nested, np.zeros((8, 8, 3), dtype=np.uint8))
        assert nested.exists()

    def test_write_image_rejects_unknown_extension(self, tmp_path: Path):
        with pytest.raises(ValueError):
            write_image(tmp_path / "bad.xyz", np.zeros((4, 4, 3), dtype=np.uint8))


# ---------------------------------------------------------------------------
# 2. 损坏 JSON
# ---------------------------------------------------------------------------


class TestCorruptJson:
    def test_read_json_invalid(self, tmp_path: Path):
        bad = tmp_path / "bad.json"
        bad.write_text("{invalid json", encoding="utf-8")
        with pytest.raises(json.JSONDecodeError):
            read_json(bad)

    def test_validate_dataset_corrupt_json(self, tmp_path: Path):
        img_dir = tmp_path / "images"
        ann_dir = tmp_path / "annotations"
        img_dir.mkdir()
        ann_dir.mkdir()
        _write_image(img_dir / "broken.png")
        (ann_dir / "broken.json").write_text("{bad", encoding="utf-8")

        report = validate_master_dataset(img_dir, ann_dir)
        assert report["summary"]["invalid_json"] == 1


# ---------------------------------------------------------------------------
# 3. 图片尺寸 vs 标注尺寸不一致
# ---------------------------------------------------------------------------


class TestSizeMismatch:
    def test_validate_flags_size_mismatch(self, tmp_path: Path):
        img_dir = tmp_path / "images"
        ann_dir = tmp_path / "annotations"
        img_dir.mkdir()
        ann_dir.mkdir()

        # 实际图片 32x24，标注写 100x100
        _write_image(img_dir / "mismatch.png", width=32, height=24)
        ann = _base_ann("mismatch", width=100, height=100)
        _write_ann(ann_dir / "mismatch.json", ann)

        report = validate_master_dataset(img_dir, ann_dir)
        sample = report["samples"][0]
        assert any("不一致" in w for w in sample["warnings"])


# ---------------------------------------------------------------------------
# 4. 点 ID 重复、边引用不存在、自环、重复边
# ---------------------------------------------------------------------------


class TestNormalizeEdgeCases:
    def test_duplicate_point_ids_reassigned(self):
        raw = {
            "points": [
                {"id": 0, "x": 1, "y": 2},
                {"id": 0, "x": 3, "y": 4},  # 重复 id
            ],
            "edges": [],
        }
        ann, issues = normalize_master_annotation(raw, sample_id="dup")
        assert any("重复" in i for i in issues)
        ids = [p["id"] for p in ann["points"]]
        assert len(set(ids)) == len(ids), "点 ID 重复后应被重排"

    def test_edge_references_nonexistent_point(self):
        raw = {
            "points": [{"id": 0, "x": 1, "y": 2}],
            "edges": [{"src": 0, "dst": 99}],  # 99 不存在
        }
        ann, issues = normalize_master_annotation(raw, sample_id="bad_edge")
        assert any("不存在" in i for i in issues)
        assert len(ann["edges"]) == 0

    def test_self_loop_edge_removed(self):
        raw = {
            "points": [{"id": 0, "x": 1, "y": 2}, {"id": 1, "x": 3, "y": 4}],
            "edges": [{"src": 0, "dst": 0}],  # 自环
        }
        ann, issues = normalize_master_annotation(raw, sample_id="self_loop")
        assert any("自环" in i for i in issues)
        assert len(ann["edges"]) == 0

    def test_duplicate_edge_deduped(self):
        raw = {
            "points": [{"id": 0, "x": 1, "y": 2}, {"id": 1, "x": 3, "y": 4}],
            "edges": [
                {"src": 0, "dst": 1},
                {"src": 1, "dst": 0},  # 与上条等价（无向）
            ],
        }
        ann, issues = normalize_master_annotation(raw, sample_id="dup_edge")
        assert any("重复" in i for i in issues)
        assert len(ann["edges"]) == 1

    def test_normalize_points_for_editor_assigns_ids(self):
        result = normalize_points_for_editor([{"x": 1, "y": 2}, {"x": 3, "y": 4}])
        assert result[0]["id"] == 0
        assert result[1]["id"] == 1

    def test_normalize_edges_for_editor_skips_self_loop(self):
        result = normalize_edges_for_editor([{"src": 0, "dst": 0}])
        assert len(result) == 0


# ---------------------------------------------------------------------------
# 5. Model A / Model B export 的 error 分流
# ---------------------------------------------------------------------------


class TestExportErrorRouting:
    def _setup_dataset(self, tmp_path: Path) -> tuple[Path, Path]:
        img_dir = tmp_path / "images"
        ann_dir = tmp_path / "annotations"
        img_dir.mkdir()
        ann_dir.mkdir()
        return img_dir, ann_dir

    def test_model_a_routes_empty_to_error(self, tmp_path: Path):
        """无点的样本应被路由到 error 目录。"""
        img_dir, ann_dir = self._setup_dataset(tmp_path)
        _write_image(img_dir / "empty.png")
        _write_ann(ann_dir / "empty.json", _base_ann("empty"))

        result = export_master_to_model_a(img_dir, ann_dir, tmp_path / "export_a")
        assert result["samples_routed_to_error"] == 1
        assert (tmp_path / "export_a" / "error" / "empty.png").exists()

    def test_model_b_routes_single_point_to_error(self, tmp_path: Path):
        """只有 1 个点（无法构成边）的样本应被路由到 error。"""
        img_dir, ann_dir = self._setup_dataset(tmp_path)
        _write_image(img_dir / "one.png")
        ann = _base_ann("one")
        ann["points"] = [{"id": 0, "x": 1, "y": 2}]
        _write_ann(ann_dir / "one.json", ann)

        result = export_master_to_model_b(img_dir, ann_dir, tmp_path / "export_b")
        assert result["samples_routed_to_error"] == 1
        assert (tmp_path / "export_b" / "error" / "one.png").exists()

    def test_model_b_routes_no_edges_to_error(self, tmp_path: Path):
        """有 2 个点但无边也应路由到 error。"""
        img_dir, ann_dir = self._setup_dataset(tmp_path)
        _write_image(img_dir / "no_edges.png")
        ann = _base_ann("no_edges")
        ann["points"] = [{"id": 0, "x": 1, "y": 2}, {"id": 1, "x": 3, "y": 4}]
        _write_ann(ann_dir / "no_edges.json", ann)

        result = export_master_to_model_b(img_dir, ann_dir, tmp_path / "export_b2")
        assert result["samples_routed_to_error"] == 1

    def test_model_a_routes_corrupt_json_to_error(self, tmp_path: Path):
        """损坏 JSON 应被路由到 error。"""
        img_dir, ann_dir = self._setup_dataset(tmp_path)
        _write_image(img_dir / "bad.png")
        (ann_dir / "bad.json").write_text("{bad", encoding="utf-8")

        result = export_master_to_model_a(img_dir, ann_dir, tmp_path / "export_a_bad")
        assert result["samples_routed_to_error"] == 1

    def test_model_a_exports_good_sample(self, tmp_path: Path):
        """正常样本应正确导出。"""
        img_dir, ann_dir = self._setup_dataset(tmp_path)
        _write_image(img_dir / "good.png")
        ann = _base_ann("good")
        ann["points"] = [{"id": 0, "x": 1, "y": 2}, {"id": 1, "x": 3, "y": 4}]
        _write_ann(ann_dir / "good.json", ann)

        result = export_master_to_model_a(img_dir, ann_dir, tmp_path / "export_a_good")
        assert result["images_exported"] == 1
        assert result["annotations_exported"] == 1


# ---------------------------------------------------------------------------
# 6. LabelMe ↔ master 格式互转
# ---------------------------------------------------------------------------


class TestLabelMeRoundtrip:
    def test_is_labelme_identifies_shapes(self):
        assert is_labelme_point_annotation({"shapes": []}) is True
        assert is_labelme_point_annotation({"points": []}) is False
        assert is_labelme_point_annotation("string") is False

    def test_load_labelme_points_extracts_sew_points(self):
        data = {
            "shapes": [
                {"shape_type": "point", "label": "sew", "points": [[10, 20]]},
                {"shape_type": "point", "label": "other", "points": [[30, 40]]},
                {"shape_type": "rectangle", "label": "sew", "points": [[0, 0], [100, 100]]},
            ],
        }
        points = load_labelme_points(data)
        assert len(points) == 1
        assert points[0]["x"] == 10.0

    def test_convert_labelme_to_master(self):
        data = {
            "imageWidth": 640,
            "imageHeight": 480,
            "shapes": [
                {"shape_type": "point", "label": "sew", "points": [[100, 200]]},
                {"shape_type": "point", "label": "keypoint", "points": [[300, 400]]},
            ],
        }
        master = convert_labelme_to_master(data, "test.png", "test")
        assert master["image_size"] == {"width": 640, "height": 480}
        assert len(master["points"]) == 2
        assert master["metadata"]["source"] == "labelme_point"

    def test_master_to_labelme_roundtrip(self):
        data = {
            "imageWidth": 320,
            "imageHeight": 240,
            "shapes": [
                {"shape_type": "point", "label": "sew", "points": [[10, 20]]},
                {"shape_type": "point", "label": "sew", "points": [[30, 40]]},
            ],
        }
        master = convert_labelme_to_master(data, "rt.png", "rt")
        labelme = master_to_labelme(master)
        assert labelme["imageWidth"] == 320
        assert labelme["imageHeight"] == 240
        assert len(labelme["shapes"]) == 2
        for shape in labelme["shapes"]:
            assert shape["shape_type"] == "point"
            assert shape["label"] == "sew"

    def test_normalize_master_detects_labelme_and_converts(self):
        labelme = {
            "imageWidth": 100,
            "imageHeight": 100,
            "shapes": [{"shape_type": "point", "label": "sew", "points": [[5, 15]]}],
        }
        ann, issues = normalize_master_annotation(labelme, sample_id="auto", image_path="auto.png")
        assert ann["schema_version"] == MASTER_SCHEMA_VERSION
        assert len(ann["points"]) == 1
        assert ann["points"][0]["x"] == 5.0


# ---------------------------------------------------------------------------
# 7. 汇总输出
# ---------------------------------------------------------------------------


class TestSummarize:
    def test_summarize_validation_empty_dirs(self, tmp_path: Path):
        """空目录应返回全零汇总。"""
        img_dir = tmp_path / "images_empty"
        ann_dir = tmp_path / "annotations_empty"
        img_dir.mkdir()
        ann_dir.mkdir()
        report = validate_master_dataset(img_dir, ann_dir)
        assert report["summary"]["num_images"] == 0
        assert report["summary"]["paired_samples"] == 0

    def test_summarize_validation_format(self, tmp_path: Path):
        img_dir = tmp_path / "empty_images"
        ann_dir = tmp_path / "empty_annotations"
        img_dir.mkdir()
        ann_dir.mkdir()
        report = validate_master_dataset(img_dir, ann_dir)
        text = summarize_validation(report)
        assert "images=0" in text

    def test_summarize_validation_text(self, tmp_path: Path):
        img_dir = tmp_path / "images"
        ann_dir = tmp_path / "annotations"
        img_dir.mkdir()
        ann_dir.mkdir()
        _write_image(img_dir / "s1.png")
        ann = _base_ann("s1")
        ann["points"] = [{"id": 0, "x": 1, "y": 2}, {"id": 1, "x": 3, "y": 4}]
        _write_ann(ann_dir / "s1.json", ann)

        report = validate_master_dataset(img_dir, ann_dir)
        text = summarize_validation(report)
        assert "images=1" in text
        assert "paired=1" in text

    def test_summarize_findings_with_details(self, tmp_path: Path):
        img_dir = tmp_path / "images"
        ann_dir = tmp_path / "annotations"
        img_dir.mkdir()
        ann_dir.mkdir()
        _write_image(img_dir / "good.png")
        _write_image(img_dir / "missing_ann.png")
        ann = _base_ann("good")
        ann["points"] = [{"id": 0, "x": 1, "y": 2}]
        _write_ann(ann_dir / "good.json", ann)

        report = validate_master_dataset(img_dir, ann_dir)
        text = summarize_validation_findings(report, include_details=True)
        assert "missing_ann" in text
        assert "missing_annotation" in text
