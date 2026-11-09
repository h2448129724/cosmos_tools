"""Tests for the per-image fixed-size crop page."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QWidget

from apps.data_tools.processing.image_io import read_image
from cosmos_toolbox.capability_catalog import (
    WorkspaceCapabilityFact,
    plan_default_capabilities,
)


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


class _Host(QWidget):
    def __init__(self):
        super().__init__()
        self.status = ""

    def show_status(self, message: str) -> None:
        self.status = message


def _write_image(folder, name: str) -> None:
    image = np.zeros((1600, 2000, 3), dtype=np.uint8)
    assert cv2.imwrite(str(folder / name), image)


def test_capability_catalog_exposes_single_crop():
    planned = plan_default_capabilities(
        (WorkspaceCapabilityFact("image", "图像", "图像工作区"),),
        include_project_capabilities=True,
    )
    entry = next(item for item in planned if item.spec.key == "data.single_crop")
    assert entry.page_factory_key == "labeling_tool:single_crop"
    assert entry.spec.title == "逐张裁剪"


def test_page_remembers_each_image_and_crops_the_current_one(app, tmp_path):
    from apps.labeling_ui.app.tools.single_crop_page import SingleCropPage

    source = tmp_path / "images"
    output = tmp_path / "crops"
    source.mkdir()
    _write_image(source, "frame_a.png")
    _write_image(source, "frame_b.png")

    host = _Host()
    page = SingleCropPage(host)
    page.resize(1280, 820)
    page.show()
    app.processEvents()
    page._in_entry.setText(str(source))
    page._out_entry.setText(str(output))
    page._load_folder()

    assert page._current_file_index == 0
    assert len(page._files) == 2
    app.processEvents()
    assert page._preview.height() >= 360
    assert page._zoom_in_button.height() >= 32
    assert page._crop_button.height() >= 32
    assert page._prev_button.height() >= 32
    zoomed = page._preview.zoom_factor()
    page._zoom_in()
    assert page._preview.zoom_factor() > zoomed
    page._fit_preview()
    page._fixed_width.setValue(256)
    page._fixed_height.setValue(256)
    page._preview.set_roi_rects([(400, 300, 656, 556)])
    page._fixed_width.setValue(512)
    page._fixed_height.setValue(512)
    assert page._preview.get_roi_rects() == [(272, 172, 784, 684)]

    QTest.keyClick(page._preview, Qt.Key.Key_D)
    app.processEvents()
    assert page._current_file_index == 1
    assert page._preview.get_roi_rects() == [(272, 172, 784, 684)]

    page._preview.set_roi_rects([(300, 200, 812, 712)])
    QTest.keyClick(page._preview, Qt.Key.Key_A)
    app.processEvents()
    assert page._current_file_index == 0
    assert page._preview.get_roi_rects() == [(272, 172, 784, 684)]

    page._crop_current(advance=False)
    cropped = read_image(str(output / "frame_a_roi_1.png"))
    assert cropped is not None
    assert cropped.shape[:2] == (512, 512)
    page._crop_current(advance=False)
    assert (output / "frame_a_roi_1.png").is_file()
    again = read_image(str(output / "frame_a_roi_2.png"))
    assert again is not None
    assert again.shape[:2] == (512, 512)
    assert page._file_list.item(0).text().startswith("已裁 ")
    assert page._current_file_index == 0

    page._crop_current(advance=True)
    assert page._current_file_index == 1
    assert page._preview.get_roi_rects() == [(300, 200, 812, 712)]
    page.close()
    host.close()


def test_several_regions_are_saved_under_new_names(app, tmp_path):
    from apps.labeling_ui.app.tools.single_crop_page import SingleCropPage

    source = tmp_path / "images"
    output = tmp_path / "crops"
    source.mkdir()
    _write_image(source, "frame_a.png")
    _write_image(source, "frame_b.png")

    host = _Host()
    page = SingleCropPage(host)
    page._in_entry.setText(str(source))
    page._out_entry.setText(str(output))
    page._load_folder()
    regions = [(10, 20, 110, 100), (200, 40, 300, 120)]
    page._preview.set_roi_rects(regions)
    page._crop_current(advance=False)

    first = read_image(str(output / "frame_a_roi_1.png"))
    second = read_image(str(output / "frame_a_roi_2.png"))
    assert first is not None and first.shape[:2] == (80, 100)
    assert second is not None and second.shape[:2] == (80, 100)

    page._crop_current(advance=False)
    assert (output / "frame_a_roi_1.png").is_file()
    assert (output / "frame_a_roi_3.png").is_file()
    assert (output / "frame_a_roi_4.png").is_file()
    kept = read_image(str(output / "frame_a_roi_1.png"))
    assert kept is not None and kept.shape == first.shape

    page._show_offset(1)
    assert page._preview.get_roi_rects() == regions
    page.close()
    host.close()


def test_clipped_crop_logs_the_actual_size(app, tmp_path):
    from apps.labeling_ui.app.tools.single_crop_page import SingleCropPage

    source = tmp_path / "images"
    output = tmp_path / "crops"
    source.mkdir()
    image = np.zeros((200, 300, 3), dtype=np.uint8)
    assert cv2.imwrite(str(source / "small.png"), image)

    host = _Host()
    page = SingleCropPage(host)
    page._in_entry.setText(str(source))
    page._out_entry.setText(str(output))
    page._load_folder()
    page._preview.set_roi_rects([(0, 0, 512, 512)])
    page._crop_current(advance=False)

    log = page._log.toPlainText()
    assert "300 x 200" in log
    assert "已夹到图片内" in log
    cropped = read_image(str(output / "small_roi_1.png"))
    assert cropped is not None
    assert cropped.shape[:2] == (200, 300)
    page.close()
    host.close()
