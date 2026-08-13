import numpy as np

from apps.data_tools.processing.auto_tile_crop import batch_tile_crop
from apps.data_tools.processing.image_io import read_image, write_image


def test_batch_tile_crop_pads_small_image_to_requested_size(tmp_path):
    source = tmp_path / "input"
    output = tmp_path / "output"
    source.mkdir()
    image = np.full((600, 800, 3), 127, dtype=np.uint8)
    write_image(str(source / "small.png"), image)

    count = batch_tile_crop(str(source), str(output), 1024, 1024, allow_overlap=True)
    tile = read_image(str(output / "small_tile_0_0.png"))

    assert count == 1
    assert tile is not None
    assert tile.shape == (1024, 1024, 3)
    assert np.array_equal(tile[:600, :800], image)
    assert np.all(tile[600:, :] == 0)
    assert np.all(tile[:, 800:] == 0)


def test_batch_tile_crop_only_pads_the_short_dimension(tmp_path):
    source = tmp_path / "input"
    output = tmp_path / "output"
    source.mkdir()
    image = np.full((1200, 800), 63, dtype=np.uint8)
    write_image(str(source / "narrow.png"), image)

    count = batch_tile_crop(str(source), str(output), 1024, 1024, allow_overlap=True)
    tiles = sorted(output.glob("*.png"))

    assert count == 2
    assert len(tiles) == 2
    for path in tiles:
        tile = read_image(str(path))
        assert tile is not None
        assert tile.shape == (1024, 1024)
        assert np.all(tile[:, :800] == 63)
        assert np.all(tile[:, 800:] == 0)
