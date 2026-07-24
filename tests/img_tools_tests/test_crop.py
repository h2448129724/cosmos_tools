from pathlib import Path

import numpy as np

from img_tools.core.crop import batch_crop, crop_image
from img_tools.core.image_io import read_image, write_image
from img_tools.core.models import Roi


def test_crop_image_matches_numpy_half_open_slice():
    image = np.arange(100, dtype=np.uint8).reshape(10, 10)

    crop, effective = crop_image(image, Roi(2, 3, 4, 2))

    assert effective == Roi(2, 3, 4, 2, "ROI")
    assert np.array_equal(crop, image[3:5, 2:6])


def test_batch_crop_writes_unicode_path_and_scales_roi(tmp_path: Path):
    source = tmp_path / "输入"
    output = tmp_path / "输出"
    source.mkdir()
    image = np.zeros((20, 40, 3), dtype=np.uint8)
    image[4:12, 10:30] = (1, 2, 3)
    write_image(source / "样本.png", image)

    result = batch_crop(
        source,
        output,
        [Roi(5, 2, 10, 4)],
        reference_size=(20, 10),
        coordinate_mode="scaled",
    )

    written = read_image(output / "样本_roi_1.png")
    assert result.processed_files == 1
    assert result.written_files == 1
    assert written is not None
    assert written.shape[:2] == (8, 20)
    assert tuple(written[0, 0]) == (1, 2, 3)


def test_batch_crop_never_overwrites_an_existing_output(tmp_path: Path):
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    image = np.full((4, 4), 9, dtype=np.uint8)
    write_image(source / "a.png", image)
    write_image(output / "a_roi_1.png", np.zeros((1, 1), dtype=np.uint8))

    result = batch_crop(source, output, [Roi(0, 0, 2, 2)], reference_size=(4, 4))

    assert result.written_files == 1
    assert read_image(output / "a_roi_1.png").shape == (1, 1)
    assert read_image(output / "a_roi_1_1.png").shape == (2, 2)


def test_batch_crop_stops_cleanly_when_cancelled(tmp_path: Path):
    source, output = tmp_path / "in", tmp_path / "out"
    source.mkdir()
    for name in ("a.png", "b.png"):
        write_image(source / name, np.zeros((4, 4), dtype=np.uint8))
    calls = 0

    def cancel_after_first():
        nonlocal calls
        calls += 1
        return calls > 1

    result = batch_crop(source, output, [Roi(0, 0, 2, 2)], reference_size=(4, 4), should_cancel=cancel_after_first)

    assert result.cancelled is True
    assert result.processed_files == 1
