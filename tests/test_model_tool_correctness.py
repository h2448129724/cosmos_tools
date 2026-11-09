"""Synthetic correctness checks for the model utility scripts.

These tests deliberately replace the backend with a tiny fake runtime; no
checkpoint or GPU is required.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


TOOLBOX = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLBOX))

from modules.segmentation.tools import auto_label_folder, predict_single_image  # noqa: E402
from modules.sew_point_conntect.batch_predict import load_labelme_points  # noqa: E402


def test_recursive_output_paths_preserve_tree_and_reject_same_stem(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    output_root = tmp_path / "annotations"
    nested = image_root / "left" / "sample.png"
    nested.parent.mkdir(parents=True)
    nested.touch()
    assert auto_label_folder._annotation_output_path(str(nested), str(image_root), str(output_root)) == str(
        output_root / "left" / "sample.json"
    )

    other = image_root / "right" / "sample.jpg"
    other.parent.mkdir()
    other.touch()
    planned = auto_label_folder._plan_annotation_outputs([str(nested), str(other)], str(image_root), str(output_root))
    assert set(planned.values()) == {
        str(output_root / "left" / "sample.json"),
        str(output_root / "right" / "sample.json"),
    }

    same_dir = image_root / "same"
    same_dir.mkdir()
    first = same_dir / "sample.png"
    second = same_dir / "sample.jpg"
    first.touch()
    second.touch()
    with pytest.raises(ValueError, match="同一标注文件"):
        auto_label_folder._plan_annotation_outputs([str(first), str(second)], str(image_root), str(output_root))


@pytest.mark.parametrize('shape', [(1, 1), (17, 23), (32, 32), (33, 81), (91, 105)])
def test_padded_tiled_inference_writes_small_image(monkeypatch: pytest.MonkeyPatch, shape) -> None:
    class _Meta:
        def __init__(self, name: str) -> None:
            self.name = name

    class _Runtime:
        def get_inputs(self):
            return [_Meta("input")]

        def get_outputs(self):
            return [_Meta("output")]

        def run(self, _names, feed):
            batch, _channels, height, width = feed["input"].shape
            logits = np.zeros((batch, 2, height, width), dtype=np.float32)
            logits[:, 1] = 5.0
            return [logits]

    monkeypatch.setattr(
        predict_single_image,
        "_load_inference_backend",
        lambda _path, _name: ("onnx", _Runtime(), None),
    )
    mask = predict_single_image.run_inference_tiled(
        np.zeros((*shape, 3), dtype=np.uint8),
        "fake.onnx",
        tile_size=32,
        batch_size=1,
        thresh=0.5,
        center_size=24,
    )
    assert mask.shape == shape
    assert np.all(mask == 255)


def test_training_pairs_consume_mirrored_annotations(tmp_path):
    from modules.segmentation.datasets import collect_training_pairs
    images, labels = tmp_path / 'images', tmp_path / 'labels'
    for name in ('left', 'right'):
        (images / name).mkdir(parents=True)
        (labels / name).mkdir(parents=True)
        (images / name / 'same.png').touch()
        (labels / name / 'same.json').write_text('{}')
    pairs = collect_training_pairs(images, labels)
    assert len(pairs) == 2
    assert all(Path(image).parent.name == Path(label).parent.name for image, label in pairs)
    (images / 'left' / 'same.jpg').touch()
    with pytest.raises(ValueError, match='同一标注'):
        collect_training_pairs(images, labels)


def test_labelme_keypoint_alias_uses_canonical_sew_rule() -> None:
    points = load_labelme_points(
        {
            "shapes": [
                {"shape_type": "point", "label": "keypoint", "points": [[3, 4]]},
                {"shape_type": "point", "label": "other", "points": [[5, 6]]},
            ]
        }
    )
    assert [(point["x"], point["y"]) for point in points] == [(3.0, 4.0)]
