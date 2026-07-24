from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QDialog, QLabel, QStackedWidget, QWidget

from cosmos_toolbox.page_router import PageRouter


def get_qapp():
    return QApplication.instance() or QApplication([])


def test_nested_pages_return_in_lifo_order():
    get_qapp()
    stack = QStackedWidget()
    base = QWidget()
    stack.addWidget(base)
    returned = []
    router = PageRouter(stack, returned.append)

    first = QLabel("workflow")
    second = QLabel("editor")
    first_host = router.open_page(first, "流程", "labeling")
    second_host = router.open_page(second, "编辑", "labeling")

    assert router.depth == 2
    assert stack.currentWidget() is second_host
    router.close_current()
    assert stack.currentWidget() is first_host
    assert returned == []
    router.close_current()
    assert returned == ["labeling"]


def test_dialog_result_is_forwarded_before_returning():
    get_qapp()
    stack = QStackedWidget()
    stack.addWidget(QWidget())
    returned = []
    results = []
    router = PageRouter(stack, returned.append)
    dialog = QDialog()

    router.open_dialog(dialog, "设置", "training", on_finished=results.append)
    dialog.accept()

    assert results == [int(QDialog.DialogCode.Accepted)]
    assert returned == ["training"]
    assert router.depth == 0


def test_back_shuts_down_page_before_returning_to_source():
    get_qapp()
    stack = QStackedWidget()
    stack.addWidget(QWidget())
    events = []
    router = PageRouter(stack, lambda source: events.append(("return", source)))

    class ManagedPage(QWidget):
        def shutdown(self):
            events.append(("shutdown", "page"))

    router.open_page(ManagedPage(), "任务", "labeling")
    router.close_current()

    assert events == [("shutdown", "page"), ("return", "labeling")]


def test_back_rejects_dialog_and_reports_finished_exactly_once():
    get_qapp()
    stack = QStackedWidget()
    stack.addWidget(QWidget())
    returned = []
    results = []
    shutdowns = []
    router = PageRouter(stack, returned.append)

    class ManagedDialog(QDialog):
        def shutdown(self):
            shutdowns.append("shutdown")

    router.open_dialog(
        ManagedDialog(),
        "设置",
        "training",
        on_finished=results.append,
    )
    router.close_current()

    assert shutdowns == ["shutdown"]
    assert results == [int(QDialog.DialogCode.Rejected)]
    assert returned == ["training"]
    assert router.depth == 0


def test_reset_cancels_all_pages_without_return_navigation():
    get_qapp()
    stack = QStackedWidget()
    stack.addWidget(QWidget())
    cancelled = []
    returned = []
    router = PageRouter(stack, returned.append)

    class CancellablePage(QWidget):
        def __init__(self, name):
            super().__init__()
            self.name = name

        def cancel(self):
            cancelled.append(self.name)

    router.open_page(CancellablePage("first"), "一", "labeling")
    router.open_page(CancellablePage("second"), "二", "labeling")
    router.reset()

    assert cancelled == ["second", "first"]
    assert returned == []
    assert router.depth == 0
