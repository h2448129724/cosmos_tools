import pytest

from img_tools.core.geometry import clamp_roi, scale_roi
from img_tools.core.models import Roi


def test_roi_uses_exclusive_right_and_bottom_edges():
    roi = Roi(10, 20, 30, 40)

    assert roi.as_xyxy() == (10, 20, 40, 60)


def test_clamp_roi_keeps_intersection_only():
    result = clamp_roi(Roi(-5, 7, 20, 10), 12, 12)

    assert result == Roi(0, 7, 12, 5, "ROI")


def test_clamp_roi_returns_none_when_outside():
    assert clamp_roi(Roi(20, 20, 3, 3), 10, 10) is None


def test_scale_roi_uses_reference_dimensions():
    result = scale_roi(Roi(10, 20, 30, 40), (100, 100), (200, 50))

    assert result == Roi(20, 10, 60, 20, "ROI")


def test_roi_rejects_empty_rectangle():
    with pytest.raises(ValueError):
        Roi(0, 0, 0, 10)
