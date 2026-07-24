import json
import sys
from pathlib import Path

# ruff: noqa: E402  -- test inserts the standalone module path before imports

import numpy as np
import torch
from PIL import Image


SEGMENTATION_DIR = Path(__file__).resolve().parents[1] / "modules" / "segmentation"
sys.path.insert(0, str(SEGMENTATION_DIR))

from datasets import BinarySegmentationDataset
from losses import MultiLabelDiceLoss
from model_registry import get_model
from trainer import _scheduler_pct_start


def _write_sample_dataset(tmp_path: Path) -> tuple[list[str], list[str]]:
    image_path = tmp_path / "sample.png"
    ann_path = tmp_path / "sample.json"

    image = np.zeros((32, 32, 3), dtype=np.uint8)
    image[:, :, 0] = 80
    image[:, :, 1] = 120
    image[:, :, 2] = 160
    Image.fromarray(image).save(image_path)

    annotation = {
        "version": "2.4.2",
        "imagePath": image_path.name,
        "imageHeight": 32,
        "imageWidth": 32,
        "shapes": [
            {"label": "glue", "shape_type": "polygon", "points": [[2, 2], [12, 2], [12, 12], [2, 12]]},
            {"label": "ear", "shape_type": "polygon", "points": [[4, 4], [28, 4], [28, 28], [4, 28]]},
            {"label": "knife", "shape_type": "polygon", "points": [[6, 6], [10, 6], [10, 20], [6, 20]]},
            {"label": "circle", "shape_type": "polygon", "points": [[18, 18], [24, 18], [24, 24], [18, 24]]},
        ],
    }
    ann_path.write_text(json.dumps(annotation), encoding="utf-8")
    return [str(image_path)], [str(ann_path)]


def test_binary_dataset_stays_class_index_mask(tmp_path):
    img_paths, ann_paths = _write_sample_dataset(tmp_path)

    dataset = BinarySegmentationDataset(
        img_paths,
        ann_paths,
        augment=False,
        image_size=64,
        target_label="glue",
    )

    image, mask = dataset[0]

    assert image.shape == (3, 64, 64)
    assert mask.shape == (64, 64)
    assert mask.dtype == torch.long
    assert int((mask == 1).sum()) > 0


def test_multilabel_dataset_returns_one_mask_per_label(tmp_path):
    img_paths, ann_paths = _write_sample_dataset(tmp_path)

    dataset = BinarySegmentationDataset(
        img_paths,
        ann_paths,
        augment=False,
        image_size=64,
        task_type="multilabel",
        target_labels=["ear", "knife", "circle"],
    )

    image, masks = dataset[0]

    assert image.shape == (3, 64, 64)
    assert masks.shape == (3, 64, 64)
    assert masks.dtype == torch.float32
    assert torch.all(masks.sum(dim=(1, 2)) > 0)


def test_multilabel_model_and_loss_accept_three_output_channels(tmp_path):
    img_paths, ann_paths = _write_sample_dataset(tmp_path)
    dataset = BinarySegmentationDataset(
        img_paths,
        ann_paths,
        augment=False,
        image_size=64,
        task_type="multilabel",
        target_labels=["ear", "knife", "circle"],
    )
    image, masks = dataset[0]

    model = get_model("microunet_gn", in_channels=3, n_classes=3)
    logits = model(image.unsqueeze(0))
    loss = MultiLabelDiceLoss()(logits, masks.unsqueeze(0))

    assert logits.shape == (1, 3, 64, 64)
    assert torch.isfinite(loss)


def test_scheduler_pct_start_stays_inside_one_cycle_bounds():
    assert _scheduler_pct_start(warmup_epochs=1, epochs=1) == 0.95
    assert _scheduler_pct_start(warmup_epochs=10, epochs=100) == 0.1
    assert _scheduler_pct_start(warmup_epochs=0, epochs=100) == 0.01
