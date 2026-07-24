import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtWidgets import QApplication, QDialog, QDockWidget, QListWidget

from img_tools.ui.main_window import MainWindow


def test_main_window_can_create_inspector_canvas():
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.canvas.set_image(np.zeros((32, 48, 3), dtype=np.uint8))
    app.processEvents()

    assert window.canvas.image_size == (48, 32)
    assert window.windowTitle() == "通用图像工具"
    window.close()


def test_roi_tool_is_hidden_until_user_opens_it():
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.show()
    app.processEvents()

    assert window.roi_dock.isHidden()
    assert not window.canvas.roi_mode
    window._show_roi_tool()
    app.processEvents()
    assert not window.roi_dock.isHidden()
    assert window.canvas.roi_mode
    window.roi_dock.hide()
    app.processEvents()
    assert not window.canvas.roi_mode
    assert window.canvas.preview_roi is None
    window.close()


def test_coordinate_copy_and_activity_log_are_available():
    app = QApplication.instance() or QApplication([])
    window = MainWindow()

    window._copy_coordinate(12, 34)

    assert app.clipboard().text() == "(12, 34)"
    assert "坐标已复制：(12, 34)" in window.log_box.toPlainText()
    window.close()


class _WorkspaceRouter:
    def __init__(self) -> None:
        self.pages: list[tuple[object, str, str, str]] = []

    def open_page(self, widget, *, title: str, source_key: str, subtitle: str = "") -> None:
        self.pages.append((widget, title, source_key, subtitle))


def test_task_history_routes_to_embedded_page_and_docks_cannot_float():
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    router = _WorkspaceRouter()
    window._state["task_history"] = [
        {"at": "2026-07-15 10:00", "task_type": "crop", "summary": "导出 2 张", "output_dir": "D:/output"}
    ]
    window.roi_dock.setFloating(True)
    assert window.roi_dock.isFloating()

    window.bind_workspace_router(router, "image")
    window._show_task_history()
    app.processEvents()

    assert len(router.pages) == 1
    page, title, source_key, subtitle = router.pages[0]
    assert title == "任务历史"
    assert source_key == "image"
    assert subtitle
    history = page.findChild(QListWidget, "imageTaskHistoryList")
    assert history is not None
    assert "导出 2 张" in history.item(0).text()
    for dock in window._workspace_docks():
        assert dock.parent() is window
        assert not dock.isFloating()
        assert not bool(dock.features() & QDockWidget.DockWidgetFeature.DockWidgetFloatable)
    window.close()


def test_task_history_remains_modal_when_started_standalone(monkeypatch):
    QApplication.instance() or QApplication([])
    window = MainWindow()
    executed: list[QDialog] = []

    monkeypatch.setattr(QDialog, "exec", lambda dialog: executed.append(dialog))
    window._show_task_history()

    assert len(executed) == 1
    history = executed[0].findChild(QListWidget, "imageTaskHistoryList")
    assert history is not None
    window.close()
