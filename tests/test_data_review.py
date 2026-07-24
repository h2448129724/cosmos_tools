from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QDialog

from apps.labeling_ui.app.data_review import DatasetReviewPage
from apps.labeling_ui.app.data_review.model import (
    ReviewSpec,
    build_review_result,
    collect_review_items,
)
from cosmos_toolbox.app import ToolboxWindow
from cosmos_toolbox.project_context import ProjectContext


def _app() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_review_model_supports_images_with_optional_annotations(tmp_path: Path) -> None:
    images = tmp_path / "images"
    annotations = tmp_path / "annotations"
    images.mkdir()
    annotations.mkdir()
    (images / "a.png").write_bytes(b"not decoded by the model")
    (images / "b.jpg").write_bytes(b"not decoded by the model")
    (images / "ignored.txt").write_text("x", encoding="utf-8")
    (annotations / "a.json").write_text("{}", encoding="utf-8")

    optional = collect_review_items(ReviewSpec(images, annotations))
    required = collect_review_items(ReviewSpec(images, annotations, require_annotation=True))

    assert [item.image_path.name for item in optional] == ["a.png", "b.jpg"]
    assert optional[0].annotation_path == annotations / "a.json"
    assert optional[1].annotation_path is None
    assert [item.image_path.name for item in required] == ["a.png"]


def test_review_result_does_not_embed_workflow_semantics(tmp_path: Path) -> None:
    source = tmp_path / "source"
    accepted = tmp_path / "accepted"
    spec = ReviewSpec(source, accepted_output=accepted)

    untouched = build_review_result(spec, accepted_count=0, removed_count=2, remaining_count=3)
    curated = build_review_result(spec, accepted_count=4, removed_count=1, remaining_count=0)

    assert untouched.active_image_dir == source
    assert curated.active_image_dir == accepted
    assert curated.accepted_count == 4
    assert not hasattr(curated, "cabf_step")


def test_review_page_has_compact_embedded_layout(tmp_path: Path) -> None:
    app = _app()
    window = ToolboxWindow(context=ProjectContext(tmp_path / "workspace.json"))
    window.resize(1100, 700)
    window.show()
    window.navigate("data.sample_review")
    app.processEvents()

    page = window._native_widgets["data.sample_review"]
    assert isinstance(page, DatasetReviewPage)
    assert not isinstance(page, QDialog)
    assert not page.isWindow()
    assert page.minimumSizeHint().width() <= page.width()
    assert page.minimumSizeHint().height() <= page.height()
    assert page.canvas.minimumWidth() == 320
    assert page.canvas.minimumHeight() == 240
    assert page.file_list.height() >= 120
    assert page.lbl_index.height() >= page.lbl_index.fontMetrics().height()
    for button in (page.btn_prev, page.btn_next, page.btn_accept, page.btn_remove, page.btn_complete):
        top_left = button.mapTo(window, button.rect().topLeft())
        bottom_right = button.mapTo(window, button.rect().bottomRight())
        assert window.rect().contains(top_left)
        assert window.rect().contains(bottom_right)

    window.close()
    app.processEvents()
