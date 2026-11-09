from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from img_tools.core.review_session import (  # noqa: E402
    ReviewItem,
    begin_review,
    plan_review_decision,
    restore_review_item,
    select_review_index,
)


def test_restore_reinserts_item_selects_it_and_reverts_count():
    items = tuple(ReviewItem(Path(f"{name}.png")) for name in "abc")
    session = select_review_index(begin_review(items), 1)
    transition = plan_review_decision(session, "removed", Path("trash"), decision="remove")
    restored = restore_review_item(transition.session, transition.item, transition.removed_index, "removed")
    assert restored.items == items
    assert restored.current_index == 1
    assert restored.removed_count == 0


@pytest.fixture(scope="module")
def app():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def test_page_undo_moves_image_and_annotation_back(app, tmp_path):
    from apps.labeling_ui.app.data_review import DatasetReviewPage

    images, labels = tmp_path / "img", tmp_path / "ann"
    images.mkdir()
    labels.mkdir()
    for name in "abc":
        cv2.imwrite(str(images / f"{name}.png"), np.zeros((8, 8, 3), np.uint8))
    (labels / "b.json").write_text(json.dumps({"shapes": [], "imageWidth": 8, "imageHeight": 8}))

    page = DatasetReviewPage()
    page.configure_paths(image_dir=str(images), label_dir=str(labels), auto_load=False)
    page.open_dataset()
    page.jump_to_index(1)
    page.move_current_to_trash()
    assert sorted(path.name for path in images.iterdir()) == ["a.png", "c.png"]
    assert page.btn_undo.isEnabled()

    page.undo_last_decision()
    assert [item.image_path.name for item in page.items] == ["a.png", "b.png", "c.png"]
    assert page.current_index == 1 and page.removed_count == 0
    assert (labels / "b.json").is_file()
    assert page.file_list.count() == 3
    assert not page.btn_undo.isEnabled()
