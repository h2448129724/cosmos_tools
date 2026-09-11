"""Personal Cosmos image, labeling, and training toolbox.

Importing the package is intentionally inert.  Executable shells opt into
path bootstrapping by requesting :func:`ensure_import_paths` explicitly.
"""

__all__ = ["ensure_import_paths"]


def __getattr__(name: str):
    if name == "ensure_import_paths":
        from .paths import ensure_import_paths

        return ensure_import_paths
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
