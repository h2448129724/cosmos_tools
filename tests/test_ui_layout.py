"""Layout and keyboard regressions found by the 2026-11 UI audit.

Every capability page is opened at the minimum supported window size and a
common desktop size; buttons must not be squeezed below their text, visible
controls must not overlap, and the page must fit (or scroll) in the space the
shell gives it.  Keyboard shortcuts must keep working while a list has focus
and after the editor reloads an image.
"""
from __future__ import annotations

import os
import time

import cv2
import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPoint, QRect, Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QAbstractButton,
    QApplication,
    QCheckBox,
    QComboBox,
    QLabel,
    QLineEdit,
    QTabWidget,
    QWidget,
)

SIZES = [(960, 640), (1280, 760)]


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(app, tmp_path, monkeypatch):
    from apps.labeling_ui.app.tools import workflow_page
    from cosmos_toolbox.app import ToolboxWindow
    from cosmos_toolbox.project_context import ProjectContext

    (tmp_path / "ds" / "images").mkdir(parents=True)
    (tmp_path / "ds" / "annotations").mkdir()
    monkeypatch.setattr(workflow_page, "CONFIG_PATH", tmp_path / "flow.json")
    context = ProjectContext(tmp_path / "state.json")
    context.set_dataset_root(tmp_path / "ds")
    shell = ToolboxWindow(context=context)
    shell.show()
    yield shell
    shell.close()


def _settle(app, condition=lambda: True, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if condition():
            return
        time.sleep(0.01)


def _clip(widget, root):
    rect = QRect(widget.mapTo(root, QPoint(0, 0)), widget.size())
    parent = widget.parentWidget()
    while parent is not None and parent is not root:
        rect = rect.intersected(QRect(parent.mapTo(root, QPoint(0, 0)), parent.size()))
        parent = parent.parentWidget()
    return rect


def _problems(window, page):
    problems = []
    controls = []
    for widget in page.findChildren(QWidget):
        if not widget.isVisibleTo(window) or widget.width() <= 0:
            continue
        if isinstance(widget, QAbstractButton) and not isinstance(widget, QCheckBox) and widget.text():
            if widget.width() + 2 < widget.sizeHint().width():
                problems.append(f"squeezed button {widget.text()!r}")
        if isinstance(widget, (QAbstractButton, QLabel, QLineEdit, QComboBox)):
            controls.append(widget)
    for index, first in enumerate(controls):
        first_rect = _clip(first, window)
        for second in controls[index + 1:]:
            if first.isAncestorOf(second) or second.isAncestorOf(first):
                continue
            overlap = first_rect.intersected(_clip(second, window))
            if overlap.width() > 4 and overlap.height() > 4:
                problems.append(f"overlap {first.objectName() or type(first).__name__}"
                                f"/{getattr(first, 'text', lambda: '')()!r} x "
                                f"{getattr(second, 'text', lambda: '')()!r}")
    hint = page.minimumSizeHint()
    if hint.height() > window.stack.height() + 2 or hint.width() > window.stack.width() + 2:
        problems.append(f"page minimum {hint.width()}x{hint.height()} exceeds "
                        f"{window.stack.width()}x{window.stack.height()}")
    return problems


def _views(page):
    yield "", lambda: None
    inner = getattr(page, "page", None)
    if inner is not None and hasattr(inner, "_go_to_step"):
        for step in range(1, inner.step_stack.count()):
            yield f" step {step + 1}", (lambda step=step: inner._go_to_step(step))
    for tabs in page.findChildren(QTabWidget):
        for index in range(1, tabs.count()):
            yield f" tab {tabs.tabText(index)}", (lambda tabs=tabs, index=index: tabs.setCurrentIndex(index))


@pytest.mark.parametrize(("width", "height"), SIZES)
def test_every_page_fits_without_squeezing_or_overlap(app, window, width, height):
    window.resize(width, height)
    _settle(app)
    failures = {}
    for capability in window.catalog.visible():
        window.navigate(capability.key)
        _settle(app)
        page = window.stack.currentWidget()
        for suffix, show in list(_views(page)):
            show()
            _settle(app)
            problems = _problems(window, page)
            if problems:
                failures[capability.key + suffix] = problems
    assert not failures, failures


def test_flow_layout_places_a_button_shown_after_the_bar(app):
    """The workflow's "标记完成" starts hidden and appears on later steps."""
    from PySide6.QtWidgets import QPushButton

    from cosmos_toolbox.ui.primitives import ActionBar

    bar = ActionBar()
    buttons = [bar.add_widget(QPushButton(text)) for text in ("上一步", "标记完成", "下一步")]
    buttons[1].hide()
    bar.resize(400, 40)
    bar.show()
    _settle(app)
    buttons[1].show()
    _settle(app)
    lefts = [button.geometry().left() for button in buttons]
    assert lefts == sorted(set(lefts)), lefts
    bar.close()


def test_compact_navigation_uses_distinct_readable_labels(app, window):
    window.resize(960, 640)
    _settle(app)
    rail = window.navigation
    assert rail.is_compact
    labels = [button.text() for button in rail._buttons]
    assert len(labels) == len(set(labels))
    assert all(len(label) >= 2 for label in labels)


@pytest.fixture
def images(tmp_path):
    folder = tmp_path / "images"
    folder.mkdir()
    for index in range(4):
        cv2.imwrite(str(folder / f"CAB_{index}.png"), np.zeros((40, 40, 3), np.uint8))
    return folder


def _press(app, window, key):
    QTest.keyClick(app.focusWidget() or window, key)


def test_review_shortcuts_work_while_file_list_has_focus(app, window, images):
    window.resize(1280, 760)
    window.navigate("data.sample_review")
    page = window.stack.currentWidget()
    page.configure_paths(image_dir=str(images), auto_load=False)
    page.open_dataset()
    window.activateWindow()
    QTest.qWaitForWindowActive(window)
    page.file_list.setFocus()
    _settle(app)
    _press(app, window, Qt.Key_D)
    _settle(app)
    assert page.current_index == 1
    line_edit = page.findChild(QLineEdit)
    line_edit.setFocus()
    _settle(app)
    before = line_edit.text()
    _press(app, window, Qt.Key_D)
    _settle(app)
    assert page.current_index == 1 and line_edit.text() == before + "d"


def test_editor_keeps_keyboard_focus_across_image_loads(app, window, images, tmp_path):
    window.resize(1280, 760)
    window.navigate("cabf.graph_editor")
    editor = window.stack.currentWidget()
    labels = tmp_path / "labels"
    labels.mkdir()
    editor.configure_paths(image_dir=str(images), label_dir=str(labels), output_dir=str(labels), auto_open=True)
    _settle(app, lambda: editor.isEnabled() and editor.current_index == 0)
    window.activateWindow()
    QTest.qWaitForWindowActive(window)
    editor.canvas.setFocus()
    for expected in (1, 2, 3):
        _press(app, window, Qt.Key_D)
        _settle(app, lambda: editor.isEnabled() and editor.current_index == expected)
        _settle(app, timeout=0.05)
        assert editor.current_index == expected
