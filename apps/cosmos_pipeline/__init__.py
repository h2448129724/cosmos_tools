"""Headless execution of the complete Cosmos inspection flow.

Runner exports stay lazy so importing the outcome core does not initialize
configuration, image, or execution adapters.
"""

__all__ = ["PipelineRequest", "describe_config", "main", "validate_request"]


def __getattr__(name: str):
    if name in __all__:
        from . import runner

        value = getattr(runner, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
