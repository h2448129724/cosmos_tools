"""Shared CAB-F domain modules with lazy public exports.

Keeping the package initializer inert lets callers import a functional-core
submodule without loading filesystem or image adapters from sibling modules.
"""

from __future__ import annotations


_EXPORTS = {
    "IMAGE_SUFFIX_ORDER": ".constants",
    "IMAGE_SUFFIXES": ".constants",
    "MASTER_SCHEMA_VERSION": ".constants",
    "POINT_LABEL_ALIASES": ".constants",
    "export_master_to_model_a": ".dataset",
    "export_master_to_model_b": ".dataset",
    "summarize_validation": ".dataset",
    "summarize_validation_findings": ".dataset",
    "validate_master_dataset": ".dataset",
    "iter_image_files": ".io",
    "iter_json_files": ".io",
    "read_image_bgr": ".io",
    "read_image_size": ".io",
    "read_json": ".io",
    "write_image": ".io",
    "write_json": ".io",
    "normalize_edges_for_editor": ".normalize",
    "normalize_master_annotation": ".normalize",
    "normalize_points_for_editor": ".normalize",
    "CABF_ROI_COORDINATE_SPACE": ".roi",
    "CabfRoiDocument": ".roi",
    "Rect": ".roi",
    "RoiCardinalityError": ".roi",
    "RoiField": ".roi",
    "RoiIssue": ".roi",
    "RoiTextPatchError": ".roi",
    "collect_roi_fields": ".roi",
    "get_value": ".roi",
    "normalize_rects": ".roi",
    "patch_roi_text": ".roi",
    "set_rects": ".roi",
    "convert_labelme_to_master": ".schema",
    "is_labelme_point_annotation": ".schema",
    "load_labelme_points": ".schema",
    "make_empty_master_annotation": ".schema",
    "master_to_labelme": ".schema",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from importlib import import_module

    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
