from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Iterable, Protocol

from .capability_catalog import (
    ActivityStage,
    CapabilityProjection,
    CapabilitySpec,
    STAGE_TITLES as STAGE_TITLES,
)

if TYPE_CHECKING:
    from PySide6.QtWidgets import QWidget
    from shared.conda_runtime import CondaEnvManager

    from .page_router import PageRouter
    from .project_context import ProjectContext
    from .project_session import ProjectSession
    from .task_center import TaskCenter


class CapabilityRuntime(Protocol):
    window: "QWidget"
    project_context: "ProjectContext"
    project_session: "ProjectSession"
    page_router: "PageRouter"
    task_center: "TaskCenter"
    conda_manager: "CondaEnvManager"

    def open_capability(self, key: str) -> None: ...

    def workspace(self, key: str) -> "QWidget | None": ...


PageFactory = Callable[[CapabilityRuntime, "QWidget"], "QWidget"]
WorkspaceActivator = Callable[["QWidget", CapabilityRuntime], None]


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
        self.to_spec()
        if (self.page_factory is None) == (self.workspace_key is None):
            raise ValueError("Capability must provide exactly one of page_factory or workspace_key")

    @property
    def cache_key(self) -> str:
        return self.workspace_key or self.key

    def to_spec(self) -> CapabilitySpec:
        return CapabilitySpec(
            key=self.key,
            title=self.title,
            description=self.description,
            stage=self.stage,
            order=self.order,
            keywords=self.keywords,
            visible=self.visible,
        )


class CapabilityCatalog:
    """Ordered source of truth for navigation, search, and activity creation."""

    def __init__(self, capabilities: Iterable[Capability] = ()) -> None:
        self._items: dict[str, Capability] = {}
        self._projection = CapabilityProjection()
        for capability in capabilities:
            self.register(capability)

    def register(self, capability: Capability) -> None:
        self._projection = self._projection.registered(capability.to_spec())
        self._items[capability.key] = capability

    def get(self, key: str) -> Capability:
        self._projection.get(key)
        return self._items[key]

    def find(self, key: str) -> Capability | None:
        return self._items.get(key)

    def visible(self) -> tuple[Capability, ...]:
        return tuple(self._items[item.key] for item in self._projection.visible())

    def grouped(self) -> tuple[tuple[ActivityStage, tuple[Capability, ...]], ...]:
        return tuple(
            (stage, tuple(self._items[item.key] for item in items))
            for stage, items in self._projection.grouped()
        )

    def search(self, query: str) -> tuple[Capability, ...]:
        return tuple(self._items[item.key] for item in self._projection.search(query))
