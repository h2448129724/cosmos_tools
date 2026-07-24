"""Small, versioned local state store for settings and task history."""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any


def default_state_path() -> Path:
    base = Path(os.environ.get("APPDATA") or Path.home()) / "通用图像工具"
    return base / "state.json"


def load_state(path: str | Path | None = None) -> dict[str, Any]:
    target = Path(path) if path else default_state_path()
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) and data.get("version") == 1 else _empty_state()
    except (OSError, json.JSONDecodeError):
        return _empty_state()


def save_state(state: dict[str, Any], path: str | Path | None = None) -> None:
    target = Path(path) if path else default_state_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(target)


def add_task_history(state: dict[str, Any], *, task_type: str, output_dir: str, summary: str) -> None:
    history = state.setdefault("task_history", [])
    if not isinstance(history, list):
        history = state["task_history"] = []
    history.insert(0, {"at": datetime.now().isoformat(timespec="seconds"), "task_type": task_type, "output_dir": output_dir, "summary": summary})
    del history[50:]


def _empty_state() -> dict[str, Any]:
    return {"version": 1, "recent_dirs": {}, "task_history": []}
