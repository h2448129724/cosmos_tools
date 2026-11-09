from __future__ import annotations

import json
from pathlib import Path

from shared.project_paths import repo_root


def cabf_flow_script() -> Path:
    return repo_root() / "scripts" / "cabf_flow.py"


FIELDS = [
    ("repo_root", "Cosmos 工具箱根目录", "dir"),
    ("dataset_root", "数据集根目录", "dir"),
    ("pending_filter_dir", "待筛选目录", "dir"),
    ("filtered_keep_dir", "筛选保留目录", "dir"),
    ("master_images_dir", "母图目录", "dir"),
    ("master_annotations_dir", "母标签目录", "dir"),
    ("model_a_export_root", "点数据集导出目录", "dir"),
    ("model_b_export_root", "连边数据集导出目录", "dir"),
]

DEFAULT_CONFIG = {
    "repo_root": str(repo_root()),
    "dataset_root": "",
    "pending_filter_dir": "",
    "filtered_keep_dir": "",
    "master_images_dir": "",
    "master_annotations_dir": "",
    "model_a_export_root": "",
    "model_b_export_root": "",
}

# Keys written by the removed prediction/training steps; dropped on load and save.
_LEGACY_KEYS = (
    "img_tools_root", "train_model_modules_root", "point_predictions_dir", "edge_predictions_dir",
    "weights", "predict", "outputs",
)


def _deep_merge(defaults: dict, overrides: dict) -> dict:
    result = dict(defaults)
    for key, value in overrides.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        elif value in ("", None):
            result[key] = result.get(key)
        else:
            result[key] = value
    return result


def apply_defaults(data: dict) -> dict:
    data = {key: value for key, value in data.items() if key not in _LEGACY_KEYS}
    return _deep_merge(DEFAULT_CONFIG, data)


def load_config(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = {}
    return apply_defaults(data)


def save_config(path: Path, data: dict) -> None:
    data = {key: value for key, value in data.items() if key not in _LEGACY_KEYS}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def get_nested(data: dict, dotted_key: str) -> str:
    current = data
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    value = current.get(parts[-1], "")
    return str(value)


def set_nested(data: dict, dotted_key: str, value: str) -> None:
    current = data
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = value
