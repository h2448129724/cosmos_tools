from pathlib import Path

import numpy as np

from img_tools.core.image_io import read_image, write_image
from img_tools.core.transform import batch_transform, transform_image


def test_transform_rotates_then_resizes_and_grayscales():
    image = np.zeros((2, 4, 3), dtype=np.uint8)
    image[:, :, 2] = 255

    result = transform_image(image, width=2, height=2, keep_aspect=True, rotation="cw90", grayscale=True)

    assert result.shape == (2, 1)
    assert result.dtype == np.uint8


def test_batch_transform_converts_format_and_reports_progress(tmp_path: Path):
    source, output = tmp_path / "input", tmp_path / "output"
    source.mkdir()
    write_image(source / "input.png", np.zeros((8, 12, 3), dtype=np.uint8))
    progress = []

    result = batch_transform(
        source, output, output_suffix=".jpg", width=6, progress=lambda current, total: progress.append((current, total))
    )

    converted = read_image(output / "input.jpg")
    assert (result.processed_files, result.written_files, result.errors, result.cancelled) == (1, 1, (), False)
    assert progress == [(1, 1)]
    assert converted is not None and converted.shape[:2] == (4, 6)


def test_batch_transform_can_be_cancelled_between_files(tmp_path: Path):
    source, output = tmp_path / "input", tmp_path / "output"
    source.mkdir()
    for name in ("a.png", "b.png"):
        write_image(source / name, np.zeros((2, 2), dtype=np.uint8))
    calls = 0

    def stop_after_one():
        nonlocal calls
        calls += 1
        return calls > 1

    result = batch_transform(source, output, output_suffix=".png", should_cancel=stop_after_one)

    assert result.cancelled is True
    assert result.processed_files == 1
