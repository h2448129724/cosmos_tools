from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication, QCheckBox, QListWidget, QScrollArea, QStackedWidget, QWidget

from apps.labeling_ui.app.theme import APP_STYLESHEET as LABELING_STYLESHEET
from cosmos_toolbox.app import ToolboxWindow, _toolbox_stylesheet
from cosmos_toolbox.capabilities import ActivityStage, Capability, CapabilityCatalog
from cosmos_toolbox.project_context import ProjectContext
from cosmos_toolbox.workspaces import WorkspaceAdapter
from shared.conda_runtime import CondaEnvInfo
from trainer_gui.theme import build_app_stylesheet as build_trainer_stylesheet


class FakeWorkspace(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.context = None
        self.applied_states = []

    def bind_project_context(self, context):
        self.context = context
        self.apply_project_context(context.state)

    def apply_project_context(self, state):
        self.applied_states.append(state)


class ManagedWidget(QWidget):
    def __init__(self, stopped, name, parent=None):
        super().__init__(parent)
        self.stopped = stopped
        self.name = name

    def shutdown(self):
        self.stopped.append(self.name)


class FakeCondaManager:
    def __init__(self):
        self.environments = [
            CondaEnvInfo("onnx-gpu", r"C:\conda\envs\onnx-gpu", r"C:\conda\envs\onnx-gpu\python.exe", True),
            CondaEnvInfo("train", r"C:\conda\envs\train", r"C:\conda\envs\train\python.exe"),
        ]

    def list_envs(self):
        return list(self.environments)

    def find(self, name):
        return next((item for item in self.environments if item.name == name), None)


def _window(tmp_path):
    context = ProjectContext(tmp_path / "workspace.json")
    adapter = WorkspaceAdapter("fake", "测试工作区", "测试", lambda parent: FakeWorkspace(parent))
    return ToolboxWindow(context=context, workspace_adapters=(adapter,))


def test_toolbox_shell_constructs_with_project_overview(tmp_path):
    app = QApplication.instance() or QApplication([])
    window = _window(tmp_path)

    assert window.windowTitle() == "Cosmos 个人工具箱"
    assert window.centralWidget() is not None
    assert window.stack.currentWidget() is window.overview
    assert window.navigation.count() == 2
    assert window.activity_host.stack is window.stack
    assert window.inspector.isHidden()

    window.close()
    app.processEvents()


def test_runtime_selector_persists_environment_for_future_tasks(tmp_path):
    app = QApplication.instance() or QApplication([])
    context = ProjectContext(tmp_path / "workspace.json")
    adapter = WorkspaceAdapter("fake", "测试工作区", "测试", lambda parent: FakeWorkspace(parent))
    window = ToolboxWindow(
        context=context,
        workspace_adapters=(adapter,),
        conda_manager=FakeCondaManager(),
    )

    assert window.navigation.runtime_combo.currentData() == "onnx-gpu"
    train_index = window.navigation.runtime_combo.findData("train")
    window.navigation.runtime_combo.setCurrentIndex(train_index)
    app.processEvents()

    assert context.state.runtime_profile == "train"
    assert ProjectContext(tmp_path / "workspace.json").state.runtime_profile == "train"
    window.close()


def test_shell_visual_language_stays_plain_and_tool_oriented():
    stylesheets = (_toolbox_stylesheet(), LABELING_STYLESHEET, build_trainer_stylesheet())

    assert all("#111a2b" not in stylesheet for stylesheet in stylesheets)
    assert all("border-radius: 999px" not in stylesheet for stylesheet in stylesheets)
    assert all("border-radius: 14px" not in stylesheet for stylesheet in stylesheets)
    assert all("border-radius: 3px" in stylesheet for stylesheet in stylesheets)
    assert "border-left: 3px solid #356a9a" in stylesheets[0]
    assert "QWidget#workflowScrollViewport" in LABELING_STYLESHEET


def test_white_theme_keeps_checkbox_indicators_visible():
    for stylesheet in (_toolbox_stylesheet(), build_trainer_stylesheet()):
        assert "QCheckBox::indicator:unchecked" in stylesheet
        assert "#747d85" in stylesheet
        assert "QCheckBox::indicator:checked" in stylesheet
        assert "#356a9a" in stylesheet


def test_trainer_checkbox_indicator_renders_contrast_on_white():
    app = QApplication.instance() or QApplication([])
    checkbox = QCheckBox()
    checkbox.setStyleSheet(build_trainer_stylesheet() + "\nQCheckBox { background: #ffffff; }")
    checkbox.resize(40, 32)
    checkbox.show()

    def rendered_colors() -> list[tuple[int, int, int]]:
        app.processEvents()
        image = checkbox.grab().toImage()
        return [
            (
                (image.pixel(x, y) >> 16) & 0xFF,
                (image.pixel(x, y) >> 8) & 0xFF,
                image.pixel(x, y) & 0xFF,
            )
            for y in range(image.height())
            for x in range(image.width())
        ]

    unchecked_colors = rendered_colors()
    assert any(max(color) < 180 for color in unchecked_colors)
    checkbox.setChecked(True)
    checked_colors = rendered_colors()
    assert any(30 <= red <= 80 and 70 <= green <= 130 and 120 <= blue <= 180 for red, green, blue in checked_colors)
    checkbox.close()


def test_default_shell_projects_customizable_capabilities_into_one_navigation(tmp_path):
    app = QApplication.instance() or QApplication([])
    window = ToolboxWindow(context=ProjectContext(tmp_path / "workspace.json"))

    assert window.catalog.find("cabf.sew_point_connect") is not None
    assert window.catalog.find("cabf.workflow") is not None
    assert window.catalog.find("training.sew_point_conntect") is not None
    assert window.catalog.find("project.artifacts") is not None

    window.navigate("training.sew_point_conntect")
    app.processEvents()
    training = window._workspace_widgets["training"]
    assert training.current_feature.feature_name == "sew_point_conntect"
    assert training.nav_panel.isHidden()
    assert window.stack.currentWidget() is training

    window.close()
    app.processEvents()


def test_embedded_labeling_page_keeps_labeling_theme_after_reparent(tmp_path):
    app = QApplication.instance() or QApplication([])
    window = ToolboxWindow(context=ProjectContext(tmp_path / "workspace.json"))

    window.navigate("cabf.workflow")
    app.processEvents()

    page = window.page_router.current_page
    assert page is not None
    assert page.styleSheet() == LABELING_STYLESHEET

    window.close()
    app.processEvents()


def test_sew_point_connect_activity_links_to_cabf_workflow_inside_shell(tmp_path):
    app = QApplication.instance() or QApplication([])
    window = ToolboxWindow(context=ProjectContext(tmp_path / "workspace.json"))
    window.navigate("cabf.sew_point_connect")
    app.processEvents()

    assert window._current_capability_key == "cabf.sew_point_connect"
    window.navigate("cabf.workflow")
    app.processEvents()

    assert window.page_router.depth == 1
    assert window._workspace_widgets["labeling"]._workspace_router is window.page_router
    window.page_router.close_current()
    app.processEvents()
    assert window._current_capability_key == "cabf.sew_point_connect"
    assert window.stack.currentWidget() is window._native_widgets["cabf.sew_point_connect"]
    window.close()
    app.processEvents()


@pytest.mark.parametrize(
    "capability_key",
    (
        "cabf.workflow",
        "cabf.point_filter",
    ),
)
def test_compatibility_cabf_page_back_restores_previous_capability(capability_key, tmp_path):
    app = QApplication.instance() or QApplication([])
    window = ToolboxWindow(context=ProjectContext(tmp_path / "workspace.json"))
    origin_widget = window.overview

    window.navigate(capability_key)
    app.processEvents()
    assert window.page_router.depth == 1

    window.page_router.close_current()
    app.processEvents()

    assert window._current_capability_key == "overview"
    assert window.stack.currentWidget() is origin_widget
    assert window.navigation.tabData(window.navigation.currentIndex()) == "overview"
    assert window.activity_host.title.text() == window.catalog.get("overview").title
    window.close()


@pytest.mark.parametrize("capability_key", ("cabf.graph_editor", "cabf.dataset_export"))
def test_primary_cabf_tools_are_native_pages_without_back_route(capability_key, tmp_path):
    app = QApplication.instance() or QApplication([])
    window = ToolboxWindow(context=ProjectContext(tmp_path / "workspace.json"))
    window.resize(1100, 700)
    window.show()

    window.navigate(capability_key)
    app.processEvents()

    assert window._current_capability_key == capability_key
    assert window.page_router.depth == 0
    assert window.stack.currentWidget() is window._native_widgets[capability_key]
    assert not window.activity_host.header.isVisible()
    page = window._native_widgets[capability_key]
    if capability_key == "cabf.graph_editor":
        assert page.left_panel.horizontalScrollBar().maximum() == 0
        assert page.lbl_index.height() >= page.lbl_index.fontMetrics().height()
    else:
        scroll = page.findChild(QScrollArea)
        assert scroll is not None
        assert scroll.horizontalScrollBar().maximum() == 0
    window.close()
    app.processEvents()


def test_generic_sample_review_replaces_visible_cabf_filter_capability(tmp_path):
    app = QApplication.instance() or QApplication([])
    window = ToolboxWindow(context=ProjectContext(tmp_path / "workspace.json"))

    visible_keys = {capability.key for capability in window.catalog.visible()}

    assert "data.sample_review" in visible_keys
    assert "cabf.point_filter" not in visible_keys
    assert "data.keyword_split" in visible_keys
    assert "labeling" not in visible_keys
    window.close()
    app.processEvents()


def test_keyword_split_is_titled_as_its_own_capability(tmp_path):
    app = QApplication.instance() or QApplication([])
    window = ToolboxWindow(context=ProjectContext(tmp_path / "workspace.json"))
    window.show()

    window.navigate("data.keyword_split")
    app.processEvents()

    assert window._current_capability_key == "data.keyword_split"
    assert window.activity_host.title.text() == "关键字划分"
    assert "关键字划分" in window.project_header.activity_label.text()
    assert window.activity_host.header.isHidden()
    window.close()
    app.processEvents()


def test_task_center_is_rendered_as_a_native_activity(tmp_path):
    app = QApplication.instance() or QApplication([])
    window = ToolboxWindow(context=ProjectContext(tmp_path / "workspace.json"))
    window.task_center.start("run-1", "连边训练", "training.sew_point_conntect")
    window.task_center.log("run-1", "epoch 1")
    window.navigate("project.tasks")
    app.processEvents()

    page = window._native_widgets["project.tasks"]
    lists = page.findChildren(QListWidget)
    assert lists[0].count() == 1
    assert "连边训练" in lists[0].item(0).text()
    assert lists[1].item(0).text() == "epoch 1"

    window.close()


def test_artifact_and_task_pages_have_real_empty_states(tmp_path):
    app = QApplication.instance() or QApplication([])
    window = ToolboxWindow(context=ProjectContext(tmp_path / "workspace.json"))

    window.navigate("project.artifacts")
    app.processEvents()
    artifacts = window._native_widgets["project.artifacts"]
    artifact_stack = artifacts.findChild(QStackedWidget, "activityStateStack")
    assert artifact_stack.currentWidget().objectName() == "emptyState"

    window.navigate("project.tasks")
    app.processEvents()
    tasks = window._native_widgets["project.tasks"]
    task_stack = tasks.findChild(QStackedWidget, "activityStateStack")
    assert task_stack.currentWidget().objectName() == "emptyState"

    window.task_center.start("run-1", "训练", "training.sew_point")
    app.processEvents()
    assert task_stack.currentWidget().objectName() == "taskContent"
    window.close()
    app.processEvents()


def test_workspace_is_embedded_lazily_and_receives_shared_context(tmp_path):
    app = QApplication.instance() or QApplication([])
    window = _window(tmp_path)

    assert "fake" not in window._workspace_widgets
    window.navigate("fake")
    app.processEvents()

    workspace = window._workspace_widgets["fake"]
    assert isinstance(workspace, FakeWorkspace)
    assert workspace.parent() is window.stack
    assert window.stack.currentWidget() is workspace

    window.context.update(project_name="cabf", dataset_root=tmp_path)
    app.processEvents()
    assert workspace.applied_states[-1].project_name == "cabf"
    assert window.project_name_edit.text() == "cabf"

    window.close()


def test_closing_toolbox_shuts_down_native_pages_and_workspaces(tmp_path):
    app = QApplication.instance() or QApplication([])
    stopped = []
    adapter = WorkspaceAdapter(
        "fake",
        "测试工作区",
        "测试",
        lambda parent: ManagedWidget(stopped, "workspace", parent),
    )
    catalog = CapabilityCatalog(
        (
            Capability(
                key="overview",
                title="概览",
                description="测试",
                stage=ActivityStage.PROJECT,
                page_factory=lambda _runtime, parent: ManagedWidget(stopped, "native", parent),
            ),
            Capability(
                key="fake",
                title="工作区",
                description="测试",
                stage=ActivityStage.DATA,
                workspace_key="fake",
            ),
        )
    )
    window = ToolboxWindow(
        context=ProjectContext(tmp_path / "workspace.json"),
        workspace_adapters=(adapter,),
        capability_catalog=catalog,
    )
    window.navigate("fake")
    app.processEvents()

    window.close()
    app.processEvents()

    assert sorted(stopped) == ["native", "workspace"]


def test_default_workspaces_share_project_paths(tmp_path):
    app = QApplication.instance() or QApplication([])
    images = tmp_path / "images"
    annotations = tmp_path / "annotations"
    outputs = tmp_path / "outputs"
    for directory in (images, annotations, outputs):
        directory.mkdir()
    model = tmp_path / "point-model.pth"
    model.write_bytes(b"test")

    context = ProjectContext(tmp_path / "workspace.json")
    context.set_dataset_root(tmp_path)
    context.update(model_path=model)
    window = ToolboxWindow(context=context)
    for key in ("image", "labeling", "training"):
        window.navigate(key)
        app.processEvents()

    image = window._workspace_widgets["image"]
    labeling = window._workspace_widgets["labeling"]
    training = window._workspace_widgets["training"]
    assert image.parent() is window.stack
    assert image._project_image_dir == str(images.resolve())
    assert image.open_project_images.isEnabled()
    assert labeling.config_data["master_images_dir"] == str(images.resolve())
    assert labeling.config_data["master_annotations_dir"] == str(annotations.resolve())
    assert labeling.config_data["weights"]["sew_point_connector_pth"] == str(model.resolve())
    assert training.project_name_edit.text() == tmp_path.name
    assert image._workspace_router is window.page_router
    assert labeling._workspace_router is window.page_router
    assert training._workspace_router is window.page_router
    assert image.menuBar().isHidden()
    assert labeling.menuBar().isHidden()
    assert training.menuBar().isHidden()
    assert training._select_feature_action("sew_point_conntect", "train")
    training.apply_project_context(context.state)
    params = training.form_builder.values(training.current_action.schema)
    assert params["image_dir"] == str(images.resolve())
    assert params["annotation_dir"] == str(annotations.resolve())
    assert params["stage1_model_path"] == str(model.resolve())

    window.close()


def test_cabf_business_page_stays_inside_shell(tmp_path):
    app = QApplication.instance() or QApplication([])
    context = ProjectContext(tmp_path / "workspace.json")
    window = ToolboxWindow(context=context)
    window.show()
    window.navigate("labeling")
    app.processEvents()
    labeling = window._workspace_widgets["labeling"]

    labeling._show_cabf_workflow_dialog()
    app.processEvents()

    assert window.page_router.depth == 1
    assert window.stack.currentWidget().page is window.page_router.current_page
    visible_top_levels = [widget for widget in QApplication.topLevelWidgets() if widget.isVisible()]
    assert visible_top_levels == [window]

    window.page_router.close_current()
    assert window.stack.currentWidget() is labeling
    window.close()
