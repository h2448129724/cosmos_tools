"""Auto-discovered registry for project-specific tool hubs."""
from __future__ import annotations

import importlib
import pkgutil
from functools import lru_cache

from .project_tools.models import RegisteredProject


def _iter_project_modules():
    package = importlib.import_module("apps.labeling_ui.app.project_tools")
    for module_info in pkgutil.iter_modules(package.__path__, package.__name__ + "."):
        if module_info.name.endswith(".models"):
            continue
        yield importlib.import_module(module_info.name)


def _extract_projects(module) -> list[RegisteredProject]:
    projects: list[RegisteredProject] = []
    project = getattr(module, "PROJECT", None)
    if isinstance(project, RegisteredProject):
        projects.append(project)
    module_projects = getattr(module, "PROJECTS", ())
    if module_projects:
        projects.extend(item for item in module_projects if isinstance(item, RegisteredProject))
    return projects


@lru_cache(maxsize=1)
def get_registered_projects() -> tuple[RegisteredProject, ...]:
    projects: list[RegisteredProject] = []
    for module in _iter_project_modules():
        projects.extend(_extract_projects(module))
    return tuple(sorted(projects, key=lambda item: item.menu_title.lower()))
