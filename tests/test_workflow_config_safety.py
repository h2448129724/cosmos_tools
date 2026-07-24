from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication


REPO_ROOT = Path(__file__).resolve().parents[1]
RENDER_SCRIPT = REPO_ROOT / "scripts" / "render_capability_gallery.py"


def _qapp() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_workflow_page_navigation_updates_memory_without_writing_config(tmp_path: Path):
    from apps.labeling_ui.app.main_window import MainWindow
    from apps.labeling_ui.app.tools.workflow_page import StitchWorkflowPage

    _qapp()
    config_path = tmp_path / "workflow.json"
    original = json.dumps(
        {"dataset_root": "D:/real/dataset", "weights": {"sew_point_onnx": "model.onnx"}},
        ensure_ascii=False,
        indent=2,
    ).encode("utf-8")
    config_path.write_bytes(original)
    window = MainWindow(config_path=config_path)

    page = StitchWorkflowPage(window)
    page._cfg_entries["dataset_root"].setText("D:/edited/in-memory")
    page._go_to_step(1)
    page._go_to_step(8)

    assert window.config_data["dataset_root"] == "D:/edited/in-memory"
    assert config_path.read_bytes() == original

    page._sync_form()
    assert config_path.read_bytes() != original


def test_visual_qa_runtime_uses_isolated_config_and_trainer_settings(tmp_path: Path):
    from apps.cabf_flow.flow import CONFIG_PATH
    from apps.labeling_ui.app import main_window as labeling_main_window
    from scripts import render_capability_gallery
    from trainer_gui import main_window as trainer_main_window

    real_config = Path(CONFIG_PATH)
    real_settings_manager = trainer_main_window.SettingsManager
    real_labeling_config = labeling_main_window.CONFIG_PATH

    with render_capability_gallery._isolated_runtime_state(tmp_path) as state:
        assert labeling_main_window.CONFIG_PATH == state.config_path
        assert state.config_path != real_config
        assert state.config_path.read_bytes() == real_config.read_bytes()
        assert trainer_main_window.SettingsManager(REPO_ROOT).settings_path == state.settings_path

    assert labeling_main_window.CONFIG_PATH == real_labeling_config
    assert trainer_main_window.SettingsManager is real_settings_manager


def test_workflow_page_shutdown_cancels_and_joins_active_worker(tmp_path: Path):
    from apps.labeling_ui.app.main_window import MainWindow
    from apps.labeling_ui.app.tools.workflow_page import StitchWorkflowPage

    _qapp()
    page = StitchWorkflowPage(MainWindow(config_path=tmp_path / "workflow.json"))
    calls: list[object] = []

    class Worker:
        def isRunning(self):
            return True

        def cancel(self):
            calls.append("cancel")

        def wait(self, timeout):
            calls.append(("wait", timeout))
            return True

    page._worker = Worker()
    page.shutdown()

    assert calls == ["cancel", ("wait", 10000)]


def test_render_gallery_cli_imports_work_from_script_and_module():
    commands = (
        [sys.executable, str(RENDER_SCRIPT), "--help"],
        [sys.executable, "-m", "scripts.render_capability_gallery", "--help"],
    )
    for command in commands:
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert "Render every Toolbox capability" in result.stdout
