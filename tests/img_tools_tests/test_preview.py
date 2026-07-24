import numpy as np
from img_tools.core.preview import make_display_preview


def test_large_preview_limits_pixel_count():
    preview = make_display_preview(np.zeros((400, 500, 3), dtype=np.uint8), max_pixels=10_000)
    assert preview.shape[0] * preview.shape[1] <= 10_100
