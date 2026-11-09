"""Compatibility re-export; the implementation lives in :mod:`img_tools.core.image_io`."""
from __future__ import annotations

from img_tools.core.image_io import IMAGE_SUFFIXES, iter_image_files, read_image, write_image

__all__ = ["IMAGE_SUFFIXES", "iter_image_files", "read_image", "write_image"]
