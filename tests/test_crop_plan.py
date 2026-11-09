"""Tests for absolute and scaled ROI crop planning."""

from __future__ import annotations

import cv2
import numpy as np

from apps.data_tools.processing.batch_crop import crop_single_image
from apps.data_tools.processing.image_io import read_image
from img_tools.core.crop_plan import CropRegion, allocate_crop_names, build_image_crop_plan


def _plan(mode: str):
    return build_image_crop_plan(
        source_name="frame.jpg",
        image_size=(2000, 1500),
        regions=(CropRegion(100, 80, 612, 592, "roi"),),
        reference_size=(1000, 750),
        coordinate_mode=mode,
        scale_semantics="edges",
        suffix_mode="preserve",
    )


def test_absolute_mode_keeps_a_512_box_unscaled():
    plan = _plan("absolute")
    assert plan.rejections == ()
    assert plan.operations[0].box == (100, 80, 612, 592)
    width = plan.operations[0].box[2] - plan.operations[0].box[0]
    height = plan.operations[0].box[3] - plan.operations[0].box[1]
    assert (width, height) == (512, 512)


def test_allocate_crop_names_skips_files_that_already_exist():
    first = allocate_crop_names("frame", ".png", 2, ())
    assert first == ("frame_roi_1.png", "frame_roi_2.png")
    second = allocate_crop_names("frame", ".png", 2, first)
    assert second == ("frame_roi_3.png", "frame_roi_4.png")
    gap = allocate_crop_names("frame", ".png", 2, {"frame_roi_1.png", "frame_roi_3.png"})
    assert gap == ("frame_roi_2.png", "frame_roi_4.png")


def test_scaled_mode_still_maps_from_the_reference_size():
    plan = _plan("scaled")
    assert plan.operations[0].box == (200, 160, 1224, 1184)


def test_crop_single_image_absolute_ignores_reference_scale(tmp_path):
    source = tmp_path / "frame.png"
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    image[10:40, 10:30] = (0, 255, 0)
    assert cv2.imwrite(str(source), image)
    output = tmp_path / "out"

    total = crop_single_image(
        str(source),
        [(10, 10, 30, 40)],
        100,
        50,
        str(output),
        coordinate_mode="absolute",
    )
    assert total == 1
    cropped = read_image(str(output / "frame_roi_1.png"))
    assert cropped is not None
    assert cropped.shape[:2] == (30, 20)

    scaled_total = crop_single_image(
        str(source),
        [(10, 10, 30, 40)],
        100,
        50,
        str(output),
    )
    assert scaled_total == 1
    scaled = read_image(str(output / "frame_roi_1.png"))
    assert scaled is not None
    assert scaled.shape[:2] == (60, 40)
