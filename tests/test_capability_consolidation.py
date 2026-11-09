"""Navigation exposes one entry per job, and duplicate entries hand off correctly."""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from cosmos_toolbox.capability_catalog import WorkspaceCapabilityFact, plan_default_capabilities  # noqa: E402


def _visible_keys():
    plan = plan_default_capabilities(
        (WorkspaceCapabilityFact("image", "图片浏览", ""),), include_project_capabilities=True
    )
    return {item.spec.key for item in plan if item.spec.visible}


def test_merged_entries_are_gone_and_replacements_exist():
    keys = _visible_keys()
    for removed in ("data.image_filter", "cabf.dataset_export", "cabf.db_ng_export", "data.transfer",
                    "cabf.point_filter", "labeling"):
        assert removed not in keys
    for kept in ("data.sample_review", "cabf.workflow", "data.ng_extract", "cabf.config_studio"):
        assert kept in keys


@pytest.fixture
def window(tmp_path, monkeypatch):
    from PySide6.QtWidgets import QApplication

    QApplication.instance() or QApplication([])
    from apps.labeling_ui.app.tools import workflow_page
    from cosmos_toolbox.app import ToolboxWindow
    from cosmos_toolbox.project_context import ProjectContext

    monkeypatch.setattr(workflow_page, "CONFIG_PATH", tmp_path / "flow.json")
    shell = ToolboxWindow(context=ProjectContext(tmp_path / "state.json"))
    yield shell
    shell.close()


def test_generic_roi_editor_hands_cabf_yaml_to_cabf_page(window, tmp_path):
    config = tmp_path / "D01-L.yaml"
    config.write_text("inspection:\n  project: CAB-F\n  top:\n    glue:\n      roi: [1, 2, 3, 4]\n", encoding="utf-8")
    window.navigate("data.roi_editor")
    editor = window.stack.currentWidget()
    editor._config_entry.setText(str(config))
    editor._load_config()
    assert window._current_capability_key == "cabf.config_studio"
    assert window.context.state.cabf_config_path == str(config.resolve())


def test_workflow_and_ng_pages_mount_directly(window):
    window.navigate("cabf.workflow")
    assert type(window.stack.currentWidget()).__name__ == "CabfWorkflowActivity"
    window.navigate("data.ng_extract")
    page = window.stack.currentWidget()
    assert page.sources.count() == 2
