import numpy as np

from img_tools.core.pipeline import batch_pipeline, run_pipeline
from img_tools.core.image_io import write_image


def test_pipeline_composes_transform_and_enhancement():
    image = np.zeros((4, 4), dtype=np.uint8)
    result = run_pipeline(image, [{"operation": "transform", "options": {"width": 2}}, {"operation": "enhance", "options": {"brightness": 20}}])
    assert result.shape == (2, 2)
    assert result[0, 0] == 20


def test_pipeline_runs_on_folder(tmp_path):
    source, output = tmp_path / "in", tmp_path / "out"
    source.mkdir()
    write_image(source / "a.png", np.zeros((4, 4), dtype=np.uint8))
    result = batch_pipeline(source, output, [{"operation": "enhance", "options": {"brightness": 10}}])
    assert (result.processed_files, result.written_files) == (1, 1)
