from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

from PySide6.QtCore import QObject, Signal

from .paths import TOOLBOX_ROOT


@dataclass(frozen=True, slots=True)
class ProjectState:
    project_name: str = ""
    dataset_root: str = ""
    image_dir: str = ""
    annotation_dir: str = ""
    cabf_config_path: str = ""
    cabf_source_top: str = ""
    cabf_source_bottom: str = ""
    cabf_reference_top: str = ""
    cabf_reference_bottom: str = ""
    cabf_template_top: str = ""
    cabf_template_bottom: str = ""
    pipeline_config_path: str = ""
    pipeline_backend_config_path: str = ""
    pipeline_input_manifest_path: str = ""
    model_path: str = ""
    output_root: str = ""
    selected_image: str = ""
    runtime_profile: str = "onnx-gpu"


class ProjectContext(QObject):
    """One persisted source of truth shared by every toolbox workspace."""

    changed = Signal(object)

    def __init__(self, state_path: str | Path | None = None, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.state_path = Path(state_path) if state_path else TOOLBOX_ROOT / "artifacts" / "project_workspace.json"
        self._state = self._load()

    @property
    def state(self) -> ProjectState:
        return self._state

    def update(self, **changes: object) -> ProjectState:
        known = {field.name for field in fields(ProjectState)}
        unknown = set(changes) - known
        if unknown:
            raise KeyError(f"Unknown project context fields: {', '.join(sorted(unknown))}")

        normalized = {
            key: self._normalize_value(key, value)
            for key, value in changes.items()
        }
        updated = replace(self._state, **normalized)
        if updated == self._state:
            return self._state

        self._state = updated
        self.save()
        self.changed.emit(self._state)
        return self._state

    def set_dataset_root(self, root: str | Path) -> ProjectState:
        dataset = Path(root).expanduser().resolve()
        image_dir = self._first_directory(dataset, ("master_images", "images", "image")) or dataset
        annotation_dir = self._first_directory(dataset, ("master_annotations", "annotations", "labels"))
        output_root = self._first_directory(dataset, ("outputs", "runs", "artifacts"))

        changes: dict[str, object] = {
            "project_name": self._state.project_name or dataset.name,
            "dataset_root": dataset,
            "image_dir": image_dir,
        }
        if annotation_dir is not None:
            changes["annotation_dir"] = annotation_dir
        elif not self._state.annotation_dir:
            changes["annotation_dir"] = dataset / "annotations"
        if output_root is not None:
            changes["output_root"] = output_root
        elif not self._state.output_root:
            changes["output_root"] = dataset / "outputs"
        return self.update(**changes)

    def clear(self) -> ProjectState:
        self._state = ProjectState()
        self.save()
        self.changed.emit(self._state)
        return self._state

    def save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"schema_version": 2, **asdict(self._state)}
        temporary = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.state_path)

    def _load(self) -> ProjectState:
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return ProjectState()
        if not isinstance(payload, dict) or payload.get("schema_version") not in {1, 2}:
            return ProjectState()
        if payload.get("schema_version") == 1:
            payload["cabf_source_top"] = self._normalize_value(
                "cabf_source_top", payload.get("cabf_reference_top", "")
            )
            payload["cabf_source_bottom"] = self._normalize_value(
                "cabf_source_bottom", payload.get("cabf_reference_bottom", "")
            )
            payload["cabf_reference_top"] = ""
            payload["cabf_reference_bottom"] = ""
        values = {
            field.name: str(payload.get(field.name, field.default) or "")
            for field in fields(ProjectState)
        }
        values["runtime_profile"] = values["runtime_profile"] or "onnx-gpu"
        return ProjectState(**values)

    @staticmethod
    def _normalize_value(key: str, value: object) -> str:
        text = str(value or "").strip()
        if not text or key in {"project_name", "runtime_profile"}:
            return text
        return str(Path(text).expanduser().resolve())

    @staticmethod
    def _first_directory(root: Path, names: tuple[str, ...]) -> Path | None:
        return next((root / name for name in names if (root / name).is_dir()), None)
