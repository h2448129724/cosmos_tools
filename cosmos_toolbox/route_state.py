"""Pure route-stack state and transitions.

The page router owns Qt widgets, while this module only models the route stack.
Keeping lifecycle transitions here makes the edge cases around asynchronous
dialogs explicit and testable without a QApplication.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Route:
    """A value describing one embedded page in the route stack."""

    route_id: int
    source_key: str
    is_dialog: bool = False


@dataclass(frozen=True, slots=True)
class RouteState:
    """Immutable route stack state."""

    routes: tuple[Route, ...] = ()
    next_id: int = 1


@dataclass(frozen=True, slots=True)
class RouteTransition:
    """Result of a route transition."""

    state: RouteState
    added: Route | None = None
    removed: tuple[Route, ...] = ()
    finished: tuple[Route, int] | None = None
    return_to_source: str | None = None

    @property
    def changed(self) -> bool:
        return bool(self.added or self.removed or self.finished)


def push(state: RouteState, source_key: str, *, is_dialog: bool = False) -> RouteTransition:
    """Push a route and return the new state."""

    route = Route(state.next_id, source_key, is_dialog)
    return RouteTransition(
        state=RouteState((*state.routes, route), state.next_id + 1),
        added=route,
    )


def finish_dialog(state: RouteState, route_id: int, result: int) -> RouteTransition:
    """Finish and remove a dialog, regardless of its stack position."""

    index, route = _find(state, route_id)
    if route is None or not route.is_dialog:
        return RouteTransition(state)
    routes = state.routes[:index] + state.routes[index + 1 :]
    return RouteTransition(
        state=RouteState(routes, state.next_id),
        removed=(route,),
        finished=(route, int(result)),
        return_to_source=route.source_key if not routes else None,
    )


def close_top(state: RouteState, *, rejected_result: int = 0) -> RouteTransition:
    """Close the top route, reporting rejection for an open dialog."""

    if not state.routes:
        return RouteTransition(state)
    route = state.routes[-1]
    routes = state.routes[:-1]
    return RouteTransition(
        state=RouteState(routes, state.next_id),
        removed=(route,),
        finished=(route, int(rejected_result)) if route.is_dialog else None,
        return_to_source=route.source_key if not routes else None,
    )


def _find(state: RouteState, route_id: int) -> tuple[int, Route | None]:
    for index, route in enumerate(state.routes):
        if route.route_id == route_id:
            return index, route
    return -1, None
