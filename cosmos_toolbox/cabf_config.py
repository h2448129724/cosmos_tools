from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import yaml
from cabf.roi import (
    CabfRoiDocument,
    RoiCardinalityError,
    RoiField,
    RoiIssue,
    collect_roi_fields as collect_roi_fields,
    normalize_rects as normalize_rects,
    _find_yaml_key_line as _find_yaml_key_line,
    _line_indent as _line_indent,
)

from .paths import COSMOS_ROOT


class CabfConfigError(ValueError):
    """Raised when a CAB-F configuration cannot support the requested action."""


def _atomic_write_with_backup(path: Path, text: str) -> Path | None:
    """Atomically replace a production config and keep its previous bytes."""
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    backup = path.with_suffix(path.suffix + ".bak") if path.exists() else None
    backup_tmp: Path | None = None
    output_tmp: Path | None = None
    try:
        if backup is not None:
            handle, name = tempfile.mkstemp(prefix=f".{backup.name}.", suffix=".tmp", dir=path.parent)
            os.close(handle)
            backup_tmp = Path(name)
            shutil.copy2(path, backup_tmp)
            os.replace(backup_tmp, backup)
            backup_tmp = None

        handle, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        output_tmp = Path(name)
        with os.fdopen(handle, "w", encoding="utf-8", newline="") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(output_tmp, path)
        output_tmp = None
        return backup
    finally:
        for temporary in (backup_tmp, output_tmp):
            if temporary is not None:
                temporary.unlink(missing_ok=True)


def resolve_config_path(value: str, config_path: Path, cosmos_root: Path = COSMOS_ROOT) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    candidates = (cosmos_root / path, config_path.parent / path)
    return next((candidate.resolve() for candidate in candidates if candidate.exists()), candidates[0].resolve())


def path_for_config(path: Path, cosmos_root: Path = COSMOS_ROOT) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(cosmos_root.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


class CabfConfigDocument:
    """YAML persistence shell around the pure CAB-F ROI document."""

    SIDE_INDEX = {"top": 0, "bottom": 1}

    def __init__(self, path: Path, data: dict[str, Any], original_text: str) -> None:
        self.path = path.resolve()
        self.data = data
        self.original_text = original_text
        self.roi_document = CabfRoiDocument(data)

    @classmethod
    def load(cls, path: str | Path) -> "CabfConfigDocument":
        resolved = Path(path).resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"配置文件不存在：{resolved}")
        text = resolved.read_text(encoding="utf-8")
        data = yaml.safe_load(text)
        if not isinstance(data, dict) or not isinstance(data.get("inspection"), dict):
            raise CabfConfigError("不是有效的 CAB-F 配置：缺少 inspection")
        return cls(resolved, data, text)

    @property
    def project_name(self) -> str:
        inspection = self.data.get("inspection", {})
        project = str(inspection.get("project", "CAB-F"))
        product = str(inspection.get("product", ""))
        return " / ".join(item for item in (project, product) if item)

    def roi_fields(self, side: str) -> list[RoiField]:
        return self.roi_document.fields(side)

    def rects(self, field: RoiField) -> list[tuple[int, int, int, int]]:
        return self.roi_document.rects(field)

    def set_rects(self, field: RoiField, rects: list[tuple[int, int, int, int]]) -> None:
        try:
            self.roi_document.set_rects(field, rects)
        except RoiCardinalityError as exc:
            raise CabfConfigError(f"{field.display_name} 是单 ROI 字段") from exc

    def template_value(self, side: str) -> str:
        index = self.SIDE_INDEX[side]
        templates = self.data.get("inspection", {}).get("match_template", [])
        if not isinstance(templates, list) or index >= len(templates):
            return ""
        value = templates[index]
        if isinstance(value, dict):
            return str(value.get("path", "") or "")
        return str(value or "")

    def template_path(self, side: str) -> Path | None:
        value = self.template_value(side)
        return resolve_config_path(value, self.path) if value else None

    def validate_rois(self, side: str, image_size: tuple[int, int]) -> list[RoiIssue]:
        return self.roi_document.validate(
            side,
            calibrated_reference_size=image_size,
        )

    def save_rois(self) -> None:
        patched = self.roi_document.patch_text(self.original_text)
        _atomic_write_with_backup(self.path, patched)
        self.original_text = patched
