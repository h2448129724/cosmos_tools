from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from PySide6.QtCore import QCoreApplication
from PySide6.QtWidgets import QFileDialog

from cosmos_toolbox import cabf_activity
from cosmos_toolbox.cabf_activity import CabfConfigActivity
from cosmos_toolbox.cabf_config import (
    CabfConfigDocument,
    CabfConfigError,
    CabfTemplateGenerator,
    TemplateGenerationResult,
    resolve_backend_glue_model_path,
)
from cosmos_toolbox.capabilities import ActivityStage
from cosmos_toolbox.default_capabilities import build_capability_catalog
from cosmos_toolbox.project_context import ProjectContext
from cosmos_toolbox.project_session import ProjectSession
from cosmos_toolbox.task_center import TaskCenter
from cosmos_toolbox.task_center import TaskStatus


def _yaml(template_path: str = "conf/old_top.png") -> str:
    return f"""image_loader: []
inspection:
  project: CAB-F
  product: TEST
  conf:
    top:
      locating_hole:
        roi: [10, 20, 110, 120]  # keep roi comment
      decode:
        roi: [[200, 210, 260, 270]]
    bottom:
      hook:
        roi: [30, 40, 130, 140]
  match_template: # 0 top, 1 bottom
    - path: "{template_path}"
    - path: "conf/old_bottom.png"
  max_overlap: 0.2
"""


def test_cabf_document_reads_sides_and_preserves_yaml_when_saving(tmp_path: Path):
    config_path = tmp_path / "cabf.yaml"
    original = _yaml()
    config_path.write_text(original, encoding="utf-8")
    document = CabfConfigDocument.load(config_path)

    assert document.project_name == "CAB-F / TEST"
    assert [field.display_name for field in document.roi_fields("top")] == [
        "inspection.conf.top.locating_hole.roi",
        "inspection.conf.top.decode.roi",
    ]
    field = document.roi_fields("top")[0]
    document.set_rects(field, [(11, 22, 111, 122)])
    document.save_rois()

    saved = config_path.read_text(encoding="utf-8")
    assert "roi: [11, 22, 111, 122]  # keep roi comment" in saved
    assert "max_overlap: 0.2" in saved
    assert "old_bottom.png" in saved
    assert config_path.with_suffix(".yaml.bak").read_text(encoding="utf-8") == original


def test_cabf_document_discovers_empty_multi_roi_field(tmp_path: Path):
    config_path = tmp_path / "cabf.yaml"
    config_path.write_text(
        """inspection:
  project: CAB-F
  conf:
    bottom:
      dense_stitch:
        roi: []
""",
        encoding="utf-8",
    )

    document = CabfConfigDocument.load(config_path)
    fields = document.roi_fields("bottom")

    assert [field.display_name for field in fields] == [
        "inspection.conf.bottom.dense_stitch.roi"
    ]
    assert fields[0].is_multi is True
    assert document.rects(fields[0]) == []

    document.set_rects(fields[0], [(10, 20, 110, 120), (200, 210, 260, 270)])
    document.save_rois()

    saved = config_path.read_text(encoding="utf-8")
    assert "roi: [[10, 20, 110, 120], [200, 210, 260, 270]]" in saved


def test_cabf_document_does_not_accept_legacy_rois_field(tmp_path: Path):
    config_path = tmp_path / "cabf.yaml"
    config_path.write_text(
        """inspection:
  project: CAB-F
  conf:
    top:
      decode_image:
        rois: [10, 20, 110, 120]
""",
        encoding="utf-8",
    )

    document = CabfConfigDocument.load(config_path)

    assert document.roi_fields("top") == []


def test_cabf_document_updates_only_selected_template_and_keeps_comments(tmp_path: Path):
    config_path = tmp_path / "cabf.yaml"
    original = _yaml()
    config_path.write_text(original, encoding="utf-8")
    output = tmp_path / "new bottom.png"
    document = CabfConfigDocument.load(config_path)

    value = document.update_template_path("bottom", output)

    saved = config_path.read_text(encoding="utf-8")
    assert value == output.resolve().as_posix()
    assert 'path: "conf/old_top.png"' in saved
    assert f'path: "{output.resolve().as_posix()}"' in saved
    assert "match_template: # 0 top, 1 bottom" in saved
    assert config_path.with_suffix(".yaml.bak").read_text(encoding="utf-8") == original


def test_cabf_document_validates_roi_against_original_reference_size(tmp_path: Path):
    config_path = tmp_path / "cabf.yaml"
    config_path.write_text(_yaml(), encoding="utf-8")
    document = CabfConfigDocument.load(config_path)

    assert document.validate_rois("top", (500, 500)) == []
    issues = document.validate_rois("top", (100, 100))
    assert len(issues) == 2
    assert all("超出基准图" in issue.message for issue in issues)


def test_template_generator_loads_backend_model_lazily_and_reuses_components():
    calls: list[object] = []

    class Extractor:
        def __init__(self, path):
            calls.append(("extractor", path))

        def glue_extract(self, image):
            calls.append(("pixel", tuple(image[0, 0])))
            return np.zeros(image.shape[:2], dtype=np.uint8)

    class Matcher:
        def __init__(self):
            calls.append("matcher")

        def center_and_refine_template_with_offset(self, image):
            return image + 255, (1, -2)

    def model_path():
        calls.append("model_path")
        return "D:/models/glue.onnx"

    generator = CabfTemplateGenerator(model_path, Extractor, Matcher)
    assert calls == []
    source = np.zeros((8, 12, 3), dtype=np.uint8)
    source[:] = (10, 20, 30)  # BGR file input

    first = generator.generate(source)
    second = generator.generate(source)

    assert first.source_size == (12, 8)
    assert first.template_size == (6, 4)
    assert first.calibrated_image is not None
    assert first.calibrated_image.shape == source.shape
    assert first.calibration_offset == (2, -4)
    assert first.image.dtype == np.uint8
    assert ("pixel", (10, 20, 30)) in calls  # production GlueExtractor receives BGR
    assert calls.count(("extractor", "D:/models/glue.onnx")) == 1
    assert calls.count("matcher") == 1
    assert second.model_path == "D:/models/glue.onnx"


def test_template_generator_rejects_existing_binary_template_before_loading_model():
    calls: list[str] = []
    generator = CabfTemplateGenerator(lambda: calls.append("model") or "unused.onnx")
    source = np.full((20, 30, 3), 255, dtype=np.uint8)
    source[5:15, 8:22] = 0

    with pytest.raises(CabfConfigError, match="黑白二值匹配模板"):
        generator.generate(source)

    assert calls == []


def test_template_generator_reports_empty_glue_segmentation():
    class Extractor:
        def __init__(self, _path):
            pass

        def glue_extract(self, image):
            return np.full(image.shape[:2], 255, dtype=np.uint8)

    class Matcher:
        def center_and_refine_template_with_offset(self, image):
            raise AssertionError("empty masks must be rejected before matching")

    source = np.zeros((8, 12, 3), dtype=np.uint8)
    source[:] = (10, 20, 30)
    generator = CabfTemplateGenerator(lambda: "model.onnx", Extractor, Matcher)

    with pytest.raises(CabfConfigError, match="胶体分割未检测到前景"):
        generator.generate(source)


def test_backend_model_resolution_rejects_unresolved_registry_reference(tmp_path: Path):
    with pytest.raises(CabfConfigError, match="尚未解析"):
        resolve_backend_glue_model_path(
            lambda: {"cab_f": {"glue_segment": {"path": "@cab_f/glue_segment.onnx#stable"}}}
        )

    model = tmp_path / "glue.onnx"
    model.write_bytes(b"model")
    assert resolve_backend_glue_model_path(
        lambda: {"cab_f": {"glue_segment": {"path": str(model)}}}
    ) == str(model.resolve())


class _Runtime:
    def __init__(self, tmp_path: Path):
        self.task_center = TaskCenter()
        self.project_context = ProjectContext(tmp_path / "workspace.json")
        self.project_session = ProjectSession(tmp_path / "session.json")

    def open_capability(self, _key: str) -> None:
        pass


def test_native_cabf_activity_uses_calibrated_reference_for_roi_coordinates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    config_path = tmp_path / "cabf.yaml"
    config_path.write_text(_yaml(), encoding="utf-8")
    image_path = tmp_path / "reference.png"
    image = np.zeros((400, 800, 3), dtype=np.uint8)
    assert cv2.imwrite(str(image_path), image)
    monkeypatch.setattr(cabf_activity, "_MAX_PREVIEW_DIMENSION", 200)
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args, **kwargs: (str(image_path), ""))

    runtime = _Runtime(tmp_path)
    page = CabfConfigActivity(runtime)
    page.config_edit.setText(str(config_path))
    page._load_config()

    assert runtime.project_context.state.cabf_config_path == str(config_path.resolve())
    assert page._source_paths == {"top": "", "bottom": ""}
    page._pick_source()
    QCoreApplication.processEvents()

    assert runtime.project_context.state.cabf_source_top == str(image_path.resolve())
    assert runtime.project_context.state.cabf_reference_top == ""
    assert page._source_sizes["top"] == (800, 400)
    assert page.preview._pixmap.width() == 200
    assert page._canvas_to_source_scale == (4.0, 4.0)
    assert not page.save_roi_button.isEnabled()

    page._generated_templates["top"] = TemplateGenerationResult(
        image=np.zeros((200, 400), dtype=np.uint8),
        model_path="D:/cache/glue.onnx",
        source_size=(800, 400),
        template_size=(400, 200),
        calibrated_image=image,
        calibration_offset=(0, 0),
    )
    page._reference_sizes["top"] = (800, 400)
    page.view_combo.setCurrentIndex(page.view_combo.findData("reference"))
    page._rebuild_canvas()
    page._refresh_validation()

    assert page.document is not None
    assert page.document.rects(page.document.roi_fields("top")[0]) == [(10, 20, 110, 120)]
    assert page.save_roi_button.isEnabled()

    restored = CabfConfigActivity(runtime)
    assert restored.document is not None
    assert restored.document.path == config_path.resolve()
    assert restored._source_paths["top"] == str(image_path.resolve())
    assert restored._reference_paths["top"] == ""

    page.preview._rects_image[0] = (3, 5, 30, 31)
    page.preview.rectsChanged.emit()
    fields = page.document.roi_fields("top")
    assert page.document.rects(fields[0]) == [(12, 20, 120, 124)]
    assert page.document.rects(fields[1]) == [(200, 210, 260, 270)]  # untouched ROI is not quantized


def test_default_catalog_exposes_native_cabf_config_studio():
    catalog = build_capability_catalog((), include_project_capabilities=True)
    capability = catalog.get("cabf.config_studio")

    assert capability.page_factory is not None
    assert capability.workspace_key is None
    assert capability in catalog.search("基准图")
    assert capability.title == "CAB-F 基准图与模板生成"
    assert capability.stage == ActivityStage.DATA


def test_native_activity_explains_the_two_generated_images(tmp_path: Path):
    page = CabfConfigActivity(_Runtime(tmp_path))

    purpose = page.image_purpose_label.text()
    assert "校准基准图" in purpose
    assert "YAML ROI" in purpose
    assert "匹配模板" in purpose
    assert "平移与旋转" in purpose
    assert "不承载 ROI" in purpose


def test_native_activity_reports_template_generation_to_task_center(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    image_path = tmp_path / "reference.png"
    assert cv2.imwrite(str(image_path), np.zeros((40, 80, 3), dtype=np.uint8))
    runtime = _Runtime(tmp_path)
    page = CabfConfigActivity(runtime)

    class Generator:
        def generate_from_path(self, path):
            assert path == str(image_path)
            return TemplateGenerationResult(
                image=np.zeros((20, 40), dtype=np.uint8),
                model_path="D:/cache/glue.onnx",
                source_size=(80, 40),
                template_size=(40, 20),
                calibrated_image=np.zeros((40, 80, 3), dtype=np.uint8),
                calibration_offset=(4, -6),
            )

    page._generator = Generator()
    page._source_paths["top"] = str(image_path)
    page._source_sizes["top"] = (80, 40)
    page._generate_template()
    assert page._worker is not None
    assert page._worker.wait(5000)
    QCoreApplication.processEvents()

    task = runtime.task_center.get(page._task_id)
    assert task is not None
    assert task.status == TaskStatus.SUCCESS
    assert page._generated_templates["top"] is not None
    assert page.model_label.text() == "D:/cache/glue.onnx"
    assert page.view_combo.currentData() == "reference"

    output = tmp_path / "top-template.png"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args, **kwargs: (str(output), ""))
    page.update_config_check.setChecked(False)
    page._save_template()
    assert runtime.project_context.state.cabf_template_top == str(output.resolve())

    reference_output = tmp_path / "top-reference.png"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args, **kwargs: (str(reference_output), ""))
    page._save_reference()
    assert runtime.project_context.state.cabf_reference_top == str(reference_output.resolve())
