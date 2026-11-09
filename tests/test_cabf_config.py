"""CAB-F YAML document tests: read-only template paths and ROI round trips."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml


TOOLBOX = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLBOX))
sys.path.insert(0, str(TOOLBOX / "shared" / "cabf_common"))

from cosmos_toolbox.cabf_config import CabfConfigDocument, CabfConfigError  # noqa: E402


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "product.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_load_rejects_yaml_without_inspection(tmp_path: Path) -> None:
    with pytest.raises(CabfConfigError):
        CabfConfigDocument.load(_write(tmp_path, "other: 1\n"))


@pytest.mark.parametrize(
    "templates",
    ['["top.png", "bottom.png"]', "[{path: top.png, score: 0.5}, {path: bottom.png}]"],
)
def test_template_path_reads_string_and_mapping_entries(tmp_path: Path, templates: str) -> None:
    (tmp_path / "bottom.png").write_bytes(b"")
    document = CabfConfigDocument.load(_write(tmp_path, f"inspection:\n  match_template: {templates}\n"))
    assert document.template_path("bottom") == (tmp_path / "bottom.png").resolve()


def test_missing_template_entry_is_none(tmp_path: Path) -> None:
    document = CabfConfigDocument.load(_write(tmp_path, "inspection:\n  match_template: [top.png]\n"))
    assert document.template_path("bottom") is None


def test_save_rois_preserves_comments_and_writes_new_coordinates(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "inspection:\n"
        "  project: CAB-F  # keep me\n"
        "  top:\n"
        "    glue:\n"
        "      roi: [10, 20, 30, 40]\n",
    )
    document = CabfConfigDocument.load(path)
    (field,) = document.roi_fields("top")
    document.set_rects(field, [(11, 21, 31, 41)])
    assert not document.validate_rois("top", (100, 100))
    document.save_rois()

    saved = path.read_text(encoding="utf-8")
    assert "# keep me" in saved
    assert yaml.safe_load(saved)["inspection"]["top"]["glue"]["roi"] == [11, 21, 31, 41]
