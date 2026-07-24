from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Callable, Iterable, Protocol

from PySide6.QtWidgets import QWidget


class ActivityStage(StrEnum):
    PROJECT = "project"
    DATA = "data"
    ANNOTATION = "annotation"
    TRAINING = "training"
    EVALUATION = "evaluation"
    ARTIFACTS = "artifacts"
    CUSTOM = "custom"


STAGE_TITLES: dict[ActivityStage, str] = {
    ActivityStage.PROJECT: "项目",
    ActivityStage.DATA: "数据准备",
    ActivityStage.ANNOTATION: "标注与复核",
    ActivityStage.TRAINING: "训练与推理",
    ActivityStage.EVALUATION: "评估",
    ActivityStage.ARTIFACTS: "产物",
    ActivityStage.CUSTOM: "定制功能",
}


class CapabilityRuntime(Protocol):
    project_context: object
    project_session: object
    page_router: object
    task_center: object

    def open_capability(self, key: str) -> None: ...

    def workspace(self, key: str) -> QWidget | None: ...


PageFactory = Callable[[CapabilityRuntime, QWidget], QWidget]
WorkspaceActivator = Callable[[QWidget, CapabilityRuntime], None]


@dataclass(frozen=True, slots=True)
class Capability:
    """One discoverable toolbox activity.

    Native activities provide ``page_factory``. Migrating activities point at a
    ``workspace_key`` and can optionally perform one focused activation after
    the legacy workspace has been created.
    """

    key: str
    title: str
    description: str
    stage: ActivityStage
    order: int = 100
    page_factory: PageFactory | None = None
    workspace_key: str | None = None
    activate: WorkspaceActivator | None = None
    keywords: tuple[str, ...] = ()
    visible: bool = True

    def __post_init__(self) -> None:
        if not self.key.strip():
            raise ValueError("Capability key cannot be empty")
        if (self.page_factory is None) == (self.workspace_key is None):
            raise ValueError("Capability must provide exactly one of page_factory or workspace_key")

    @property
    def cache_key(self) -> str:
        return self.workspace_key or self.key


class CapabilityCatalog:
    """Ordered source of truth for navigation, search, and activity creation."""

    def __init__(self, capabilities: Iterable[Capability] = ()) -> None:
        self._items: dict[str, Capability] = {}
        for capability in capabilities:
            self.register(capability)

    def register(self, capability: Capability) -> None:
        if capability.key in self._items:
            raise KeyError(f"Capability already registered: {capability.key}")
        self._items[capability.key] = capability

    def get(self, key: str) -> Capability:
        try:
            return self._items[key]
        except KeyError as exc:
            raise KeyError(f"Unknown capability: {key}") from exc

    def find(self, key: str) -> Capability | None:
        return self._items.get(key)

    def visible(self) -> tuple[Capability, ...]:
        return tuple(sorted((item for item in self._items.values() if item.visible), key=self._sort_key))

    def grouped(self) -> tuple[tuple[ActivityStage, tuple[Capability, ...]], ...]:
        groups: list[tuple[ActivityStage, tuple[Capability, ...]]] = []
        for stage in ActivityStage:
            items = tuple(item for item in self.visible() if item.stage == stage)
            if items:
                groups.append((stage, items))
        return tuple(groups)

    def search(self, query: str) -> tuple[Capability, ...]:
        needle = query.strip().casefold()
        if not needle:
            return self.visible()
        return tuple(
            item
            for item in self.visible()
            if needle in " ".join((item.title, item.description, *item.keywords)).casefold()
        )

    @staticmethod
    def _sort_key(item: Capability) -> tuple[int, int, str]:
        return (list(ActivityStage).index(item.stage), item.order, item.title.casefold())
