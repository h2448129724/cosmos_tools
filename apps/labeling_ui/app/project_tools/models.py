"""Shared data models for project-specific tool registration."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RegisteredProjectTool:
    key: str
    title: str
    description: str
    launcher_name: str
    category: str = "工具"
    featured: bool = False


@dataclass(frozen=True)
class RegisteredProject:
    key: str
    menu_title: str
    display_name: str
    tools: tuple[RegisteredProjectTool, ...]
