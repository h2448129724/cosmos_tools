"""Training model discovery and compatibility contracts."""

from .model_registry import ModelSpec, create_model, list_models, resolve_model

__all__ = ["ModelSpec", "create_model", "list_models", "resolve_model"]
