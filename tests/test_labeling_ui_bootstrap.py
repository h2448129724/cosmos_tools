from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_PAGE = REPO_ROOT / "apps" / "labeling_ui" / "app" / "tools" / "workflow_page.py"
DATASET_EXPORT_DIALOG = REPO_ROOT / "apps" / "labeling_ui" / "app" / "dataset_export_dialog.py"
UI_CABF_CONSUMERS = [
    REPO_ROOT / "apps" / "labeling_ui" / "app" / "annotation" / "adapters" / "cabf.py",
    REPO_ROOT / "apps" / "labeling_ui" / "app" / "tools" / "label_visualization_page.py",
    REPO_ROOT / "apps" / "labeling_ui" / "app" / "dataset_export_dialog.py",
    REPO_ROOT / "apps" / "labeling_ui" / "app" / "graph_annotation_dialog.py",
    REPO_ROOT / "apps" / "labeling_ui" / "app" / "point_annotation_dialog.py",
    REPO_ROOT / "apps" / "labeling_ui" / "app" / "point_filter_dialog.py",
]


def test_labeling_ui_import_smoke():
    import apps.labeling_ui.app.main_window  # noqa: F401
    import apps.labeling_ui.app.tools.workflow_page  # noqa: F401


def test_labeling_ui_script_exists():
    script = REPO_ROOT / "scripts" / "labeling_ui.py"
    assert script.exists()


def test_workflow_page_uses_config_model_paths():
    source = WORKFLOW_PAGE.read_text(encoding="utf-8")

    assert "Path(__file__).resolve().parents" not in source


def test_workflow_dataset_root_paths_have_config_fields():
    from apps.cabf_flow.config_model import DEFAULT_CONFIG, FIELDS

    field_keys = {key for key, _label, _kind in FIELDS}
    expected_auto_path_keys = {
        "img_tools_root",
        "dataset_root",
        "train_model_modules_root",
        "pending_filter_dir",
        "filtered_keep_dir",
        "master_images_dir",
        "master_annotations_dir",
        "point_predictions_dir",
        "edge_predictions_dir",
        "model_a_export_root",
        "model_b_export_root",
    }

    assert expected_auto_path_keys <= field_keys
    assert "pending_filter_dir" in DEFAULT_CONFIG
    assert "filtered_keep_dir" in DEFAULT_CONFIG


def test_workflow_dataset_root_keeps_train_output_paths_after_sync(tmp_path: Path):
    from apps.labeling_ui.app.main_window import MainWindow
    from apps.labeling_ui.app.tools.workflow_page import StitchWorkflowPage

    _app = QApplication.instance() or QApplication([])
    config_path = tmp_path / "workflow.json"
    window = MainWindow(config_path=config_path)
    page = StitchWorkflowPage(window)
    dataset_root = tmp_path / "dataset"
    expected_point_out = str(dataset_root / "runs" / "sew_point_train")
    expected_edge_out = str(dataset_root / "runs" / "sew_point_conntect_train")

    page._apply_dataset_root_paths(dataset_root)
    page._sync_form()
    page._go_to_step(8)

    assert window.config_data["outputs"]["sew_point_train_out"] == expected_point_out
    assert window.config_data["outputs"]["sew_point_conntect_train_out"] == expected_edge_out
    assert page._tr_a.text() == expected_point_out
    assert page._tr_b.text() == expected_edge_out


def test_dataset_export_dialog_uses_repo_docs_root():
    from apps.labeling_ui.app import dataset_export_dialog

    source = DATASET_EXPORT_DIALOG.read_text(encoding="utf-8")

    assert dataset_export_dialog.DOCS_ROOT == REPO_ROOT / "docs"
    assert "Path(__file__).resolve().parents" not in source


def test_first_ui_cabf_consumers_import_shared_package_directly():
    offenders = []
    for path in UI_CABF_CONSUMERS:
        source = path.read_text(encoding="utf-8")
        if "apps.data_tools.processing.cabf_shared" in source:
            offenders.append(str(path.relative_to(REPO_ROOT)))

    assert not offenders, "UI CAB-F consumers should import cabf directly: " + ", ".join(offenders)
