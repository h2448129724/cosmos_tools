from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINER_GUI_APP = REPO_ROOT / "trainer_gui" / "app.py"
CABF_CONFIG_MODEL = REPO_ROOT / "apps" / "cabf_flow" / "config_model.py"
CABF_FLOW = REPO_ROOT / "apps" / "cabf_flow" / "flow.py"


def test_shared_project_paths_owns_repo_root():
    from shared.project_paths import repo_root

    assert repo_root() == REPO_ROOT


def test_repo_root_consumers_use_shared_project_paths():
    trainer_source = TRAINER_GUI_APP.read_text(encoding="utf-8")
    cabf_source = CABF_CONFIG_MODEL.read_text(encoding="utf-8")
    flow_source = CABF_FLOW.read_text(encoding="utf-8")

    assert "from shared.project_paths import repo_root" in trainer_source
    assert "from shared.project_paths import repo_root" in cabf_source
    assert "from shared.project_paths import repo_root" in flow_source
    assert "Path(__file__).resolve" not in trainer_source
    assert "Path(__file__).resolve" not in cabf_source
    assert "SCRIPT_DIR.parents" not in flow_source
