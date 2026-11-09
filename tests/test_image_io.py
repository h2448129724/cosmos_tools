from __future__ import annotations

import cv2
import numpy as np
import pytest
from PIL import Image

from img_tools.core.image_io import read_image, write_image


@pytest.fixture
def rotated_jpeg(tmp_path):
    """A 40x20 (w x h) JPEG whose EXIF says "rotate 90° clockwise to display"."""
    path = tmp_path / "图片.jpg"
    exif = Image.Exif()
    exif[0x0112] = 6
    Image.new("RGB", (40, 20), "white").save(path, exif=exif)
    return path


@pytest.mark.parametrize("flags", [cv2.IMREAD_UNCHANGED, cv2.IMREAD_COLOR])
def test_exif_orientation_is_applied_exactly_once(rotated_jpeg, flags):
    assert read_image(rotated_jpeg, flags).shape[:2] == (40, 20)


@pytest.mark.parametrize("flags", [cv2.IMREAD_UNCHANGED, cv2.IMREAD_COLOR])
def test_auto_orient_off_keeps_stored_pixels(rotated_jpeg, flags):
    assert read_image(rotated_jpeg, flags, auto_orient=False).shape[:2] == (20, 40)


def test_write_raises_instead_of_silently_dropping(tmp_path):
    with pytest.raises(ValueError):
        write_image(tmp_path / "out.unknown", np.zeros((2, 2, 3), np.uint8))
    target = tmp_path / "子目录" / "out.png"
    write_image(target, np.zeros((2, 2, 3), np.uint8))
    assert read_image(target).shape == (2, 2, 3)


def test_missing_or_empty_file_returns_none(tmp_path):
    assert read_image(tmp_path / "missing.png") is None
    (tmp_path / "empty.png").write_bytes(b"")
    assert read_image(tmp_path / "empty.png") is None
