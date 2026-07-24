from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

from PySide6.QtCore import QObject, Signal

from .paths import TOOLBOX_ROOT


class ArtifactKind(StrEnum):
    DATASET = "dataset"
    ANNOTATIONS = "annotations"
    MODEL = "model"
    RUN_OUTPUT = "run_output"
    EXPORT = "export"


@dataclass(frozen=True, slots=True)
class ProjectArtifact:
    artifact_id: str
    kind: ArtifactKind
    name: str
    path: str
    created_at: str
    source_capability: str = ""
    source_task_id: str = ""
    inputs: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


class ProjectSession(QObject):
    """Persist project artifacts and their lineage behind one small interface."""

    changed = Signal()

    def __init__(self, state_path: str | Path | None = None, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.state_path = Path(state_path) if state_path else TOOLBOX_ROOT / "artifacts" / "project_session.json"
        self._artifacts: list[ProjectArtifact] = self._load()

    @property
    def artifacts(self) -> tuple[ProjectArtifact, ...]:
        return tuple(self._artifacts)

    def recent(self, limit: int = 8, kind: ArtifactKind | None = None) -> tuple[ProjectArtifact, ...]:
        items: Iterable[ProjectArtifact] = self._artifacts
        if kind is not None:
            items = (item for item in items if item.kind == kind)
        return tuple(list(items)[-max(0, limit):][::-1])

    def register_artifact(
        self,
        *,
        kind: ArtifactKind,
        name: str,
        path: str | Path,
        source_capability: str = "",
        source_task_id: str = "",
        inputs: Iterable[str] = (),
        metadata: dict[str, Any] | None = None,
    ) -> ProjectArtifact:
        normalized_path = str(Path(path).expanduser().resolve())
        artifact = ProjectArtifact(
            artifact_id=f"artifact-{uuid4().hex[:12]}",
            kind=kind,
            name=name.strip() or Path(normalized_path).name,
            path=normalized_path,
            created_at=datetime.now().isoformat(timespec="seconds"),
            source_capability=source_capability,
            source_task_id=source_task_id,
            inputs=tuple(inputs),
            metadata=dict(metadata or {}),
        )
        self._artifacts.append(artifact)
        self.save()
        self.changed.emit()
        return artifact

    def clear(self) -> None:
        self._artifacts.clear()
        self.save()
        self.changed.emit()

    def save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "artifacts": [
                {**asdict(item), "kind": item.kind.value, "inputs": list(item.inputs)}
                for item in self._artifacts
            ],
        }
        temporary = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.state_path)

    def _load(self) -> list[ProjectArtifact]:
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        if not isinstance(payload, dict) or payload.get("schema_version") != 1:
            return []
        artifacts: list[ProjectArtifact] = []
        for raw in payload.get("artifacts", []):
            if not isinstance(raw, dict):
                continue
            try:
                artifacts.append(
                    ProjectArtifact(
                        artifact_id=str(raw["artifact_id"]),
                        kind=ArtifactKind(str(raw["kind"])),
                        name=str(raw["name"]),
                        path=str(raw["path"]),
                        created_at=str(raw["created_at"]),
                        source_capability=str(raw.get("source_capability") or ""),
                        source_task_id=str(raw.get("source_task_id") or ""),
                        inputs=tuple(str(item) for item in raw.get("inputs", [])),
                        metadata=dict(raw.get("metadata") or {}),
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue
        return artifacts
