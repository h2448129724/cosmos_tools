from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER_SCRIPT = REPO_ROOT / "scripts" / "cabf_launcher.py"


def test_launcher_script_exists():
    assert (REPO_ROOT / "scripts" / "cabf_launcher.py").exists()
    assert (REPO_ROOT / "scripts" / "cabf_launcher.ps1").exists()


def test_launcher_import_smoke():
    import scripts.cabf_launcher  # noqa: F401


def test_launcher_uses_shared_bootstrap_helper():
    source = LAUNCHER_SCRIPT.read_text(encoding="utf-8")

    assert "python_env" in source
    assert "__import__(\"os\")" not in source


def test_bootstrap_python_env_prepends_repo_and_shared(monkeypatch):
    from scripts._bootstrap import REPO_ROOT as bootstrap_root
    from scripts._bootstrap import SHARED_ROOT, python_env

    monkeypatch.setenv("PYTHONPATH", "existing")
    env = python_env()
    parts = env["PYTHONPATH"].split(os.pathsep)

    assert bootstrap_root == REPO_ROOT
    assert parts[:2] == [str(REPO_ROOT), str(SHARED_ROOT)]
    assert parts[-1] == "existing"
