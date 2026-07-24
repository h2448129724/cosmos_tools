import numpy as np

from img_tools.core.tile import batch_tile, compute_tiles
from img_tools.core.image_io import read_image, write_image


def test_shift_policy_covers_right_and_bottom_edges_with_full_tiles():
    tiles = compute_tiles(10, 9, 4, 4, edge_policy="shift")

    assert {(tile.x, tile.y) for tile in tiles} >= {(0, 0), (6, 5)}
    assert all((tile.width, tile.height) == (4, 4) for tile in tiles)


def test_pad_policy_keeps_requested_tile_size_for_small_image(tmp_path):
    source, output = tmp_path / "in", tmp_path / "out"
    source.mkdir()
    write_image(source / "small.png", np.ones((2, 3), dtype=np.uint8))

    result = batch_tile(source, output, 4, 4, edge_policy="pad")
    tile = read_image(output / "small_tile_0_0.png")

    assert result.written_files == 1
    assert tile is not None and tile.shape == (4, 4)
    assert np.all(tile[:2, :3] == 1)
