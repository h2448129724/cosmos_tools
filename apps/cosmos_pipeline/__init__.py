"""Headless execution of the complete Cosmos inspection flow."""

from .runner import PipelineRequest, describe_config, main, validate_request

__all__ = ["PipelineRequest", "describe_config", "main", "validate_request"]
