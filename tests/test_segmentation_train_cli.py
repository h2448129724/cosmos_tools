from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_train_script_help_works_outside_toolbox_directory(tmp_path: Path) -> None:
    toolbox_root = Path(__file__).resolve().parents[1]
    script = toolbox_root / "modules" / "segmentation" / "train.py"

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--img_dir" in result.stdout
