"""Pure project workbench state and presentation transitions."""

from __future__ import annotations

from dataclasses import dataclass, replace as evolve
from enum import StrEnum

from .layout_presentation import LayoutPresentation, layout_presentation


class WorkbenchEventKind(StrEnum):
    NAVIGATE = "navigate"
    ACTIVITY_MOUNTED = "activity_mounted"
    ROUTE_DEPTH_CHANGED = "route_depth_changed"
    INSPECTOR_TOGGLED = "inspector_toggled"
    INSPECTOR_CLOSED = "inspector_closed"
    VIEWPORT_RESIZED = "viewport_resized"
    TASKS_CHANGED = "tasks_changed"


class WorkbenchIntentKind(StrEnum):
    RESET_ROUTES = "reset_routes"
    SELECT_NAVIGATION = "select_navigation"
    PRESENT_ACTIVITY = "present_activity"
    UPDATE_ACTIVITY_HEADER = "update_activity_header"
    UPDATE_INSPECTOR = "update_inspector"
    UPDATE_LAYOUT = "update_layout"
    UPDATE_TASK_SUMMARY = "update_task_summary"


@dataclass(frozen=True, slots=True)
class WorkbenchEvent:
    kind: WorkbenchEventKind
    value: object = None


@dataclass(frozen=True, slots=True)
class WorkbenchIntent:
    kind: WorkbenchIntentKind
    value: object = None


@dataclass(frozen=True, slots=True)
class WorkbenchState:
    active_capability_key: str = "overview"
    route_depth: int = 0
    inspector_open: bool = False
    viewport_width: int = 1540
    active_task_count: int = 0
    activity_owns_header: bool = False

    @property
    def layout(self) -> LayoutPresentation:
        return layout_presentation(self.viewport_width)

    @property
    def activity_header_visible(self) -> bool:
        return self.route_depth == 0 and not self.activity_owns_header

    @property
    def task_strip_visible(self) -> bool:
        return self.active_task_count > 0


@dataclass(frozen=True, slots=True)
class WorkbenchTransition:
    state: WorkbenchState
    intents: tuple[WorkbenchIntent, ...] = ()


def transition_workbench(state: WorkbenchState, event: WorkbenchEvent) -> WorkbenchTransition:
    """Reduce one workbench event into immutable state and shell intents."""

    kind = WorkbenchEventKind(event.kind)
    if kind is WorkbenchEventKind.NAVIGATE:
        key = str(event.value or "")
        if not key:
            return WorkbenchTransition(state)
        next_state = evolve(
            state,
            active_capability_key=key,
            route_depth=0,
            activity_owns_header=False,
        )
        return WorkbenchTransition(
            next_state,
            (
                WorkbenchIntent(WorkbenchIntentKind.RESET_ROUTES),
                WorkbenchIntent(WorkbenchIntentKind.SELECT_NAVIGATION, key),
                WorkbenchIntent(WorkbenchIntentKind.PRESENT_ACTIVITY, key),
            ),
        )

    if kind is WorkbenchEventKind.ACTIVITY_MOUNTED:
        owns_header = bool(event.value)
        next_state = evolve(state, activity_owns_header=owns_header)
        return WorkbenchTransition(
            next_state,
            (WorkbenchIntent(WorkbenchIntentKind.UPDATE_ACTIVITY_HEADER, next_state.activity_header_visible),),
        )

    if kind is WorkbenchEventKind.ROUTE_DEPTH_CHANGED:
        depth = max(0, int(event.value or 0))
        next_state = evolve(state, route_depth=depth)
        return WorkbenchTransition(
            next_state,
            (WorkbenchIntent(WorkbenchIntentKind.UPDATE_ACTIVITY_HEADER, next_state.activity_header_visible),),
        )

    if kind in {WorkbenchEventKind.INSPECTOR_TOGGLED, WorkbenchEventKind.INSPECTOR_CLOSED}:
        opened = not state.inspector_open if kind is WorkbenchEventKind.INSPECTOR_TOGGLED else False
        next_state = evolve(state, inspector_open=opened)
        return WorkbenchTransition(
            next_state,
            (WorkbenchIntent(WorkbenchIntentKind.UPDATE_INSPECTOR, opened),),
        )

    if kind is WorkbenchEventKind.VIEWPORT_RESIZED:
        width = max(0, int(event.value or 0))
        previous_layout = state.layout
        next_state = evolve(state, viewport_width=width)
        intents: tuple[WorkbenchIntent, ...] = ()
        if next_state.layout != previous_layout:
            intents = (WorkbenchIntent(WorkbenchIntentKind.UPDATE_LAYOUT, next_state.layout),)
        return WorkbenchTransition(next_state, intents)

    if kind is WorkbenchEventKind.TASKS_CHANGED:
        active_count = max(0, int(event.value or 0))
        next_state = evolve(state, active_task_count=active_count)
        return WorkbenchTransition(
            next_state,
            (WorkbenchIntent(WorkbenchIntentKind.UPDATE_TASK_SUMMARY, active_count),),
        )

    return WorkbenchTransition(state)
