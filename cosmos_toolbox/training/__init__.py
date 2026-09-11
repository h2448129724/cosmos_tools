"""Lazy training model discovery and compatibility contracts."""

from __future__ import annotations

from typing import Any


__all__ = ["ModelSpec", "create_model", "list_models", "resolve_model"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from . import model_registry

        value = getattr(model_registry, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
