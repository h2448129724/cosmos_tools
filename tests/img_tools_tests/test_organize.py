import numpy as np

from img_tools.core.image_io import write_image
from img_tools.core.organize import batch_rename, classify_by_keywords, find_exact_duplicates, find_similar_images


def test_keyword_classification_copies_matches_and_counts_unmatched(tmp_path):
    source, output = tmp_path / "in", tmp_path / "out"
    source.mkdir()
    write_image(source / "left_1.png", np.zeros((2, 2), dtype=np.uint8))
    write_image(source / "other.png", np.zeros((2, 2), dtype=np.uint8))

    result = classify_by_keywords(source, output, ["left"])

    assert (result.processed_files, result.changed_files, result.unmatched_files) == (2, 1, 1)
    assert (output / "left" / "left_1.png").exists()


def test_duplicate_and_rename_operations(tmp_path):
    source = tmp_path / "in"
    source.mkdir()
    image = np.zeros((2, 2), dtype=np.uint8)
    write_image(source / "a.png", image)
    write_image(source / "b.png", image)

    assert len(find_exact_duplicates(source)) == 1
    assert len(find_similar_images(source)) == 1
    renamed = batch_rename(source, "sample", start=7, digits=3)
    assert [target.name for _, target in renamed] == ["sample_007.png", "sample_008.png"]
