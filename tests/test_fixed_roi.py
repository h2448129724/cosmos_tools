"""Tests for fixed-size per-image ROI geometry."""

from __future__ import annotations

import pytest

from img_tools.core.fixed_roi import (
    inherit_fixed_box,
    resize_fixed_box,
    resolve_fixed_box,
    resolve_fixed_boxes,
)


def _center(box: tuple[int, int, int, int]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2, (y1 + y2) / 2


def test_resize_keeps_center_when_the_box_stays_inside():
    box = (400, 300, 656, 556)
    resized = resize_fixed_box(box, (512, 512), (2000, 1600))
    assert resized == (272, 172, 784, 684)
    assert _center(resized) == _center(box)
    assert (resized[2] - resized[0], resized[3] - resized[1]) == (512, 512)


def test_resize_keeps_the_whole_box_inside_near_the_edge():
    resized = resize_fixed_box((900, 700, 1000, 800), (512, 512), (1000, 800))
    x1, y1, x2, y2 = resized
    assert (x2 - x1, y2 - y1) == (512, 512)
    assert x1 >= 0 and y1 >= 0
    assert x2 <= 1000 and y2 <= 800


def test_resize_shrinks_to_the_image_when_the_request_is_larger():
    resized = resize_fixed_box((10, 10, 110, 110), (512, 512), (300, 200))
    assert resized == (0, 0, 300, 200)


def test_inherit_keeps_origin_and_shifts_inward_at_the_edge():
    assert inherit_fixed_box((100, 80, 612, 592), (2000, 1600)) == (100, 80, 612, 592)
    assert inherit_fixed_box((1800, 100, 2312, 612), (2000, 2000)) == (1488, 100, 2000, 612)
    assert inherit_fixed_box((10, 20, 522, 532), (300, 200)) == (0, 0, 300, 200)


def test_unvisited_image_inherits_previous_position_at_the_requested_size():
    previous = (100, 80, 612, 592)
    resolved = resolve_fixed_box(
        saved=None,
        previous=previous,
        size=(512, 512),
        image_size=(2000, 1600),
    )
    assert resolved == previous


def test_remembered_box_wins_and_adopts_the_requested_size():
    saved = (400, 300, 656, 556)
    resolved = resolve_fixed_box(
        saved=saved,
        previous=(0, 0, 10, 10),
        size=(512, 512),
        image_size=(2000, 1600),
    )
    assert resolved == (272, 172, 784, 684)
    assert _center(resolved) == _center(saved)


def test_unvisited_image_inherits_every_previous_box():
    previous = ((100, 80, 612, 592), (700, 80, 812, 192))
    assert resolve_fixed_boxes(saved=None, previous=previous, image_size=(2000, 1600)) == previous
    assert resolve_fixed_boxes(saved=(), previous=previous, image_size=(2000, 1600)) == ()


def test_no_box_until_one_has_been_placed():
    assert resolve_fixed_box(
        saved=None,
        previous=None,
        size=(512, 512),
        image_size=(2000, 1600),
    ) is None


def test_invalid_geometry_is_rejected():
    with pytest.raises(ValueError):
        resize_fixed_box((0, 0, 10, 10), (0, 512), (100, 100))
    with pytest.raises(ValueError):
        inherit_fixed_box((0, 0, 10, 10), (0, 100))
    with pytest.raises(ValueError):
        resize_fixed_box((5, 5, 5, 20), (10, 10), (100, 100))
