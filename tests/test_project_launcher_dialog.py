from __future__ import annotations

from PySide6.QtWidgets import QApplication
from PySide6.QtWidgets import QFrame

from apps.labeling_ui.app.project_launcher_dialog import ProjectToolEntry, ProjectToolsHubDialog
from apps.labeling_ui.app.project_registry import get_registered_projects


def get_qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_project_hub_groups_tools_by_category():
    get_qapp()
    dialog = ProjectToolsHubDialog(
        "Demo",
        [
            ProjectToolEntry("workflow", "主流程", "Run all steps", lambda: None, category="主流程", featured=True),
            ProjectToolEntry("editor", "编辑", "Edit labels", lambda: None, category="标注修正"),
            ProjectToolEntry("export", "导出", "Export data", lambda: None, category="数据复核"),
        ],
    )

    assert dialog.category_titles() == ["主流程", "标注修正", "数据复核"]
    assert dialog.featured_tool_keys() == ["workflow"]


def test_project_hub_uses_targeted_card_styles():
    get_qapp()
    dialog = ProjectToolsHubDialog(
        "Demo",
        [
            ProjectToolEntry("workflow", "主流程", "Run all steps", lambda: None, category="主流程", featured=True),
            ProjectToolEntry("editor", "编辑", "Edit labels", lambda: None, category="标注修正"),
        ],
    )

    cards = {
        frame.property("tool_key"): frame
        for frame in dialog.findChildren(QFrame)
        if frame.property("tool_key")
    }

    assert cards["workflow"].objectName() == "featuredProjectToolCard"
    assert cards["editor"].objectName() == "projectToolCard"
    assert "QFrame{" not in cards["workflow"].styleSheet()
    assert "QFrame{" not in cards["editor"].styleSheet()


def test_cabf_registered_tools_have_project_categories():
    project = next(item for item in get_registered_projects() if item.key == "cabf")

    categories = {tool.key: tool.category for tool in project.tools}
    assert categories["cabf_workflow"] == "主流程"
    assert categories["cabf_stitch_editor"] == "标注修正"
    assert categories["cabf_point_filter"] == "数据复核"
    assert next(tool for tool in project.tools if tool.key == "cabf_workflow").featured is True
