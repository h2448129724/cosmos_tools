from __future__ import annotations

from PySide6.QtWidgets import QApplication

from apps.labeling_ui.app.main_window import MainWindow
from apps.labeling_ui.app.project_registry import get_registered_projects
from apps.labeling_ui.app.tools.workflow_page import StitchWorkflowPage


def get_qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _menu_titles(window: MainWindow) -> list[str]:
    return [action.text() for action in window.menuBar().actions()]


def test_main_window_defaults_to_generic_image_tools():
    get_qapp()

    window = MainWindow()

    page_types = tuple(type(page) for page in window._tool_pages)
    page_titles = [window._tool_list.item(i).text() for i in range(window._tool_list.count())]
    assert StitchWorkflowPage not in page_types
    assert page_titles[0] == "关键词划分"
    assert "缝纫点与连边" not in page_titles
    assert window._sidebar_title.text() == "通用"
    assert window.windowTitle().startswith("通用图像工具")


def test_main_window_exposes_cabf_as_project_adapter_option():
    get_qapp()

    window = MainWindow()

    assert "项目适配(&P)" in _menu_titles(window)
    project_action = next(action for action in window.menuBar().actions() if action.text() == "项目适配(&P)")
    project_menu = project_action.menu()
    assert [action.text() for action in project_menu.actions()] == ["CAB-F"]


def test_cabf_project_adapter_contains_workflow_option():
    project = next(item for item in get_registered_projects() if item.key == "cabf")

    tool_keys = [tool.key for tool in project.tools]
    assert tool_keys[0] == "cabf_workflow"


class _WorkspaceRouterSpy:
    def __init__(self):
        self.pages = []
        self.dialogs = []

    def open_page(self, widget, **kwargs):
        self.pages.append((widget, kwargs))
        return widget

    def open_dialog(self, dialog, **kwargs):
        self.dialogs.append((dialog, kwargs))
        return dialog


def test_sidebar_exposes_direct_cabf_entries():
    get_qapp()
    window = MainWindow()

    assert [
        window._project_tool_list.item(i).text()
        for i in range(window._project_tool_list.count())
    ] == ["CAB-F 流程", "点边标注", "数据筛选", "数据集导出"]


def test_workflow_uses_bound_workspace_router():
    get_qapp()
    window = MainWindow()
    router = _WorkspaceRouterSpy()
    window.bind_workspace_router(router)

    result = window._show_cabf_workflow_dialog()

    assert isinstance(result, StitchWorkflowPage)
    assert router.pages[0][1]["source_key"] == "labeling"
    assert router.pages[0][1]["title"] == "CAB-F 缝纫点与连边流程"


def test_bound_workspace_uses_shell_navigation_only():
    get_qapp()
    window = MainWindow()

    assert not window._sidebar.isHidden()
    assert not window._btn_toggle_sidebar.isHidden()
    assert not window.menuBar().isHidden()

    window.bind_workspace_router(_WorkspaceRouterSpy())

    assert window._sidebar.isHidden()
    assert window._btn_toggle_sidebar.isHidden()
    assert window.menuBar().isHidden()
    assert window.statusBar().isHidden()

    window._set_embedded_mode(False)
    assert not window._sidebar.isHidden()
    assert not window._btn_toggle_sidebar.isHidden()
    assert not window.menuBar().isHidden()
    assert not window.statusBar().isHidden()


def test_nested_workflow_dialog_uses_same_router():
    get_qapp()
    window = MainWindow()
    router = _WorkspaceRouterSpy()
    window.bind_workspace_router(router)
    page = StitchWorkflowPage(window)

    class FakeDialog:
        def exec(self):
            raise AssertionError("business dialog must stay in the main window")

    dialog = FakeDialog()
    page._open_business_dialog(dialog, "连边修正", "返回流程")

    assert router.dialogs[0][0] is dialog
    assert router.dialogs[0][1]["source_key"] == "labeling"
