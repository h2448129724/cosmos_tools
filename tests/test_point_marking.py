"""Tests for the ordered point marking core and its Qt page adapter."""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from apps.point_marking.point_core import (
    append_point,
    build_payload,
    build_snapshot,
    clamp_point,
    describe_points,
    insert_point,
    move_point,
    normalize_point,
    parse_export_text,
    point_label,
    remove_point,
    render_export_text,
)
from cosmos_toolbox.capability_catalog import (
    WorkspaceCapabilityFact,
    plan_default_capabilities,
)


# --------------------------------------------------------------------- core
def test_points_keep_click_order():
    points = append_point(append_point((), 10, 20), 30, 40)
    assert points == ((10, 20), (30, 40))
    assert build_payload(points) == ((10, 20), (30, 40))


def test_export_payload_is_a_two_level_list():
    text = render_export_text(build_payload(((1, 2), (3, 4))))
    assert text == "[\n  [1, 2],\n  [3, 4]\n]"
    assert text.startswith("[")


def test_snapshot_carries_points_payload_and_text():
    snapshot = build_snapshot(((5, 6),))
    assert snapshot.count == 1
    assert snapshot.payload == ((5, 6),)
    assert snapshot.text == "[\n  [5, 6]\n]"


def test_insert_remove_and_move_keep_order():
    points = insert_point(((1, 1), (3, 3)), 1, 2, 2)
    assert points == ((1, 1), (2, 2), (3, 3))
    assert remove_point(points, 1) == ((1, 1), (3, 3))
    assert move_point(points, 0, 9, 9) == ((9, 9), (2, 2), (3, 3))
    assert remove_point(points, 99) == points
    assert move_point(points, -1, 9, 9) == points


def test_normalize_and_clamp_use_source_pixels():
    assert normalize_point(10.4, 20.6) == (10, 21)
    assert clamp_point((900, 900), 100, 80) == (99, 79)
    assert clamp_point((-5, -5), 100, 80) == (0, 0)
    assert clamp_point((5, 5), 0, 0) == (5, 5)


def test_parse_export_text_round_trip():
    points, error = parse_export_text(render_export_text(((1, 2), (3, 4))))
    assert error == ""
    assert points == ((1, 2), (3, 4))


@pytest.mark.parametrize(
    ("text", "fragment"),
    [
        ("", "内容为空"),
        ("{", "JSON 解析失败"),
        ('{"a": 1}', "顶层必须是一个列表"),
        ("[[1, 2, 3]]", "第 1 项必须是 [x, y]"),
        ('[["a", 2]]', "第 1 项必须是数字"),
        ("[[true, 2]]", "第 1 项必须是数字"),
    ],
)
def test_parse_export_text_rejects_invalid_documents(text, fragment):
    points, error = parse_export_text(text)
    assert points == ()
    assert fragment in error


def test_labels_and_description():
    assert point_label(0) == "1"
    assert "尚未标注任何点" in describe_points(())
    assert "共 2 个点" in describe_points(((1, 2), (3, 4)))


# -------------------------------------------------------------- capability
def test_capability_catalog_exposes_the_point_mark_activity():
    planned = plan_default_capabilities(
        (WorkspaceCapabilityFact("image", "图像", "图像工作区"),),
        include_project_capabilities=True,
    )
    entry = next(item for item in planned if item.spec.key == "annotation.point_mark")
    assert entry.page_factory_key == "labeling_tool:point_mark"
    assert entry.spec.visible is True


# --------------------------------------------------------------------- page
@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture()
def image_path(tmp_path):
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    image[:, :] = (40, 90, 160)
    path = tmp_path / "sample.png"
    assert cv2.imwrite(str(path), image)
    return path


def test_page_marks_points_in_click_order_and_exports_two_level_list(app, image_path):
    from apps.labeling_ui.app.tools.point_mark_page import PointMarkPage

    page = PointMarkPage(None)
    page.resize(1100, 700)
    page.show()
    app.processEvents()

    page._load_image(str(image_path))
    assert page._canvas.has_image
    assert page._canvas.source_size == (320, 240)

    page._add_point(10, 20)
    page._add_point(30, 40)
    assert page._points == ((10, 20), (30, 40))
    assert page._export_text() == "[\n  [10, 20],\n  [30, 40]\n]"
    assert page._table.rowCount() == 2
    assert page._table.item(0, 0).text() == "1"
    assert page._table.item(1, 1).text() == "30"

    page._delete_selected()
    assert page._points == ((10, 20),)

    page._undo()
    assert page._points == ((10, 20), (30, 40))

    page._clear_points()
    assert page._points == ()
    page.close()


def test_canvas_click_maps_to_source_pixel_and_zoom_works(app, image_path):
    from apps.labeling_ui.app.tools.point_mark_page import PointMarkPage

    page = PointMarkPage(None)
    page.resize(1100, 700)
    page.show()
    app.processEvents()
    page._load_image(str(image_path))
    page._canvas.resize(700, 500)
    page._canvas.fit_to_view()
    app.processEvents()

    canvas = page._canvas
    before = canvas.zoom
    QTest.mouseMove(canvas, QPoint(200, 150))
    QTest.mouseClick(canvas, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(200, 150))
    assert len(page._points) == 1
    x, y = page._points[0]
    assert 0 <= x < 320 and 0 <= y < 240

    screen = canvas._source_to_screen((x, y))
    QTest.mouseClick(
        canvas,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
        QPoint(int(screen.x()), int(screen.y())),
    )
    assert len(page._points) == 2
    assert page._points[1] == (x, y)

    canvas.zoom_by(1.2)
    assert canvas.zoom > before
    canvas.reset_zoom()
    assert canvas.zoom == 1.0
    page.close()


def test_right_click_removes_the_point_under_the_cursor(app, image_path):
    from apps.labeling_ui.app.tools.point_mark_page import PointMarkPage

    page = PointMarkPage(None)
    page.resize(1100, 700)
    page.show()
    app.processEvents()
    page._load_image(str(image_path))
    page._canvas.resize(700, 500)
    page._canvas.fit_to_view()
    app.processEvents()

    page._add_point(120, 90)
    screen = page._canvas._source_to_screen((120, 90))
    QTest.mouseClick(
        page._canvas,
        Qt.MouseButton.RightButton,
        Qt.KeyboardModifier.NoModifier,
        QPoint(int(screen.x()), int(screen.y())),
    )
    assert page._points == ()
    page.close()
