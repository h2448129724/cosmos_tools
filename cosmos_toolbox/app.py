from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence
from weakref import ref

from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication, QFileDialog, QHBoxLayout, QLabel, QMainWindow, QVBoxLayout, QWidget

from .capabilities import Capability, CapabilityCatalog, STAGE_TITLES
from .default_capabilities import build_capability_catalog
from .page_router import PageRouter
from .paths import TOOLBOX_ROOT, ensure_import_paths
from .project_context import ProjectContext, ProjectState
from .project_session import ArtifactKind, ProjectSession
from .shell_widgets import ActivityHost, NavigationRail, ProjectHeader, ProjectInspector
from .task_center import TaskCenter, TaskStatus
from .workspaces import WorkspaceAdapter, default_workspace_adapters, request_shutdown
from shared.conda_runtime import CondaEnvInfo, CondaEnvManager


class _ToolboxRuntime:
    def __init__(self, window: "ToolboxWindow") -> None:
        self._window_ref = ref(window)

    @property
    def window(self) -> "ToolboxWindow":
        window = self._window_ref()
        if window is None:
            raise RuntimeError("Toolbox window is no longer available")
        return window

    @property
    def project_context(self) -> ProjectContext:
        return self.window.context

    @property
    def project_session(self) -> ProjectSession:
        return self.window.project_session

    @property
    def page_router(self) -> PageRouter:
        return self.window.page_router

    @property
    def task_center(self) -> TaskCenter:
        return self.window.task_center

    @property
    def conda_environment(self) -> CondaEnvInfo | None:
        return self.window.conda_manager.find(self.window.context.state.runtime_profile)

    @property
    def conda_manager(self) -> CondaEnvManager:
        return self.window.conda_manager

    def open_capability(self, key: str) -> None:
        self.window.navigate(key)

    def workspace(self, key: str) -> QWidget | None:
        return self.window._workspace_widgets.get(key)


class ToolboxWindow(QMainWindow):
    def __init__(
        self,
        context: ProjectContext | None = None,
        workspace_adapters: Sequence[WorkspaceAdapter] | None = None,
        initial_workspace: str = "overview",
        capability_catalog: CapabilityCatalog | None = None,
        conda_manager: CondaEnvManager | None = None,
    ) -> None:
        super().__init__()
        using_default_workspaces = workspace_adapters is None
        self.context = context or ProjectContext(parent=self)
        self.workspace_adapters = tuple(workspace_adapters or default_workspace_adapters())
        self._adapters_by_key = {adapter.key: adapter for adapter in self.workspace_adapters}
        self.catalog = capability_catalog or build_capability_catalog(
            self.workspace_adapters,
            include_project_capabilities=using_default_workspaces,
        )
        session_path = self.context.state_path.with_name("project_session.json")
        self.project_session = ProjectSession(session_path, self)
        self.task_center = TaskCenter(self)
        self.conda_manager = conda_manager or CondaEnvManager()
        self._runtime = _ToolboxRuntime(self)
        self._workspace_widgets: dict[str, QWidget] = {}
        self._native_widgets: dict[str, QWidget] = {}
        self._stack_indexes: dict[str, int] = {}
        self._last_capability_for_workspace: dict[str, str] = {}
        self._current_capability_key = ""
        self._pending_return_capability_key: str | None = None

        self.setObjectName("toolboxShell")
        self.setWindowTitle("Cosmos 个人工具箱")
        self.resize(1540, 920)
        self.setMinimumSize(1100, 700)
        self.setStyleSheet(_toolbox_stylesheet())
        self.setCentralWidget(self._build_ui())

        self.context.changed.connect(self._on_context_changed)
        self.project_session.changed.connect(self._on_session_changed)
        self.task_center.task_finished.connect(self._on_task_finished)
        self._on_context_changed(self.context.state)
        self._on_session_changed()

        initial = initial_workspace if self.catalog.find(initial_workspace) else "overview"
        self.navigate(initial)

    def _build_ui(self) -> QWidget:
        root = QWidget()
        root.setObjectName("shellRoot")
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        self.project_header = ProjectHeader()
        self.project_header.settings_requested.connect(self._toggle_inspector)
        root_layout.addWidget(self.project_header)

        body = QWidget()
        body_layout = QHBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(0)
        self.navigation = NavigationRail(self.catalog)
        self._nav_rows = {
            str(self.navigation.tabData(index)): index
            for index in range(self.navigation.count())
        }
        self.navigation.currentChanged.connect(self._on_navigation_changed)
        self.navigation.runtimeProfileChanged.connect(self._on_runtime_profile_changed)
        self._load_runtime_environments()
        body_layout.addWidget(self.navigation)

        work_area = QWidget()
        work_layout = QVBoxLayout(work_area)
        work_layout.setContentsMargins(0, 0, 0, 0)
        work_layout.setSpacing(0)
        content = QWidget()
        content_layout = QHBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)

        self.activity_host = ActivityHost()
        self.stack = self.activity_host.stack
        content_layout.addWidget(self.activity_host, 1)

        self.inspector = ProjectInspector()
        self.inspector.close_requested.connect(self.inspector.hide)
        self.inspector.dataset_requested.connect(self._choose_dataset)
        self.inspector.model_requested.connect(self._choose_model)
        self.inspector.image_dir_requested.connect(lambda: self._choose_directory("image_dir"))
        self.inspector.annotation_dir_requested.connect(lambda: self._choose_directory("annotation_dir"))
        self.inspector.output_root_requested.connect(lambda: self._choose_directory("output_root"))
        self.inspector.hide()
        content_layout.addWidget(self.inspector)
        work_layout.addWidget(content, 1)

        # Task history is a first-class activity in the same content host.  A
        # compact bottom drawer can be added after the remaining QMainWindow
        # adapters have been removed; mixing another hidden text view with the
        # legacy trainer causes unstable Qt teardown on Windows.
        self.task_drawer = QWidget()
        self.task_drawer.hide()
        body_layout.addWidget(work_area, 1)
        root_layout.addWidget(body, 1)

        self.page_router = PageRouter(self.stack, self._return_to_workspace, self)
        self.page_router.depth_changed.connect(lambda depth: self.activity_host.set_nested(depth > 0))

        # Compatibility attributes remain available while legacy workspaces migrate.
        self.project_name_edit = self.inspector.project_name_edit
        self.dataset_edit = self.inspector.dataset_edit
        self.image_dir_edit = self.inspector.image_dir_edit
        self.annotation_dir_edit = self.inspector.annotation_dir_edit
        self.model_edit = self.inspector.model_edit
        self.output_root_edit = self.inspector.output_root_edit
        self.project_name_edit.editingFinished.connect(
            lambda: self.context.update(project_name=self.project_name_edit.text())
        )
        self.dataset_edit.editingFinished.connect(self._apply_dataset_text)
        self.image_dir_edit.editingFinished.connect(
            lambda: self.context.update(image_dir=self.image_dir_edit.text())
        )
        self.annotation_dir_edit.editingFinished.connect(
            lambda: self.context.update(annotation_dir=self.annotation_dir_edit.text())
        )
        self.model_edit.editingFinished.connect(
            lambda: self.context.update(model_path=self.model_edit.text())
        )
        self.output_root_edit.editingFinished.connect(
            lambda: self.context.update(output_root=self.output_root_edit.text())
        )
        self.context_status = QLabel()
        self.context_status.hide()

        self.overview = self._ensure_capability_widget(self.catalog.get("overview"))
        return root

    def navigate(self, key: str) -> None:
        capability = self.catalog.find(key)
        if capability is None:
            return
        self._capture_navigation_origin()
        self.page_router.reset()
        row = self._nav_rows.get(key, -1)
        if row >= 0 and self.navigation.currentIndex() != row:
            self.navigation.setCurrentIndex(row)
            return
        self._show_capability(capability)

    def _on_navigation_changed(self, row: int) -> None:
        key = self.navigation.tabData(row)
        capability = self.catalog.find(str(key)) if key else None
        if capability is None:
            return
        self._capture_navigation_origin()
        self.page_router.reset()
        self._show_capability(capability)

    def _show_capability(self, capability: Capability, *, activate: bool = True) -> None:
        return_capability_key = self._pending_return_capability_key
        self._pending_return_capability_key = None
        widget = self._ensure_capability_widget(capability)
        self.stack.setCurrentWidget(widget)
        self._current_capability_key = capability.key
        if capability.workspace_key:
            self._last_capability_for_workspace[capability.workspace_key] = capability.key
        stage_title = STAGE_TITLES[capability.stage]
        self.project_header.set_activity(stage_title, capability.title)
        self.activity_host.set_activity(capability.title, capability.description)
        self.activity_host.set_nested(False)
        if bool(widget.property("ownsPageHeader")):
            self.activity_host.header.hide()
        if activate and capability.activate is not None:
            with self.page_router.returning_to(return_capability_key):
                capability.activate(widget, self._runtime)

    def _capture_navigation_origin(self) -> None:
        if self._pending_return_capability_key is not None:
            return
        self._pending_return_capability_key = (
            self.page_router.current_return_target or self._current_capability_key or None
        )

    def _ensure_capability_widget(self, capability: Capability) -> QWidget:
        cache_key = capability.cache_key
        if cache_key in self._stack_indexes:
            return self.stack.widget(self._stack_indexes[cache_key])
        if capability.page_factory is not None:
            widget = capability.page_factory(self._runtime, self.stack)
            self._native_widgets[capability.key] = widget
        else:
            adapter = self._adapters_by_key.get(str(capability.workspace_key))
            if adapter is None:
                raise KeyError(f"No workspace adapter for capability: {capability.key}")
            widget = adapter.create(self.stack, self.context, self.page_router)
            self._workspace_widgets[adapter.key] = widget
            self._bind_workspace_tasks(adapter.key, widget)
        self._stack_indexes[cache_key] = self.stack.addWidget(widget)
        return widget

    def _bind_workspace_tasks(self, key: str, workspace: QWidget) -> None:
        if key == "training" and hasattr(workspace, "run_manager"):
            self.task_center.bind_training_manager(workspace.run_manager)

    def _return_to_workspace(self, key: str) -> None:
        capability_key = self._last_capability_for_workspace.get(key, key)
        capability = self.catalog.find(capability_key) or self.catalog.find(key) or self.catalog.get("overview")
        row = self._nav_rows.get(capability.key, -1)
        if row >= 0:
            blocked = self.navigation.blockSignals(True)
            self.navigation.setCurrentIndex(row)
            self.navigation.blockSignals(blocked)
        self._show_capability(capability, activate=False)

    def _show_workspace(self, key: str) -> None:
        """Compatibility entry used by older tests and callers."""
        capability = self.catalog.find(key)
        if capability is not None:
            self._show_capability(capability)

    def _toggle_inspector(self) -> None:
        self.inspector.setVisible(not self.inspector.isVisible())

    def show_status(self, message: str) -> None:
        self.statusBar().showMessage(message, 5000)

    def _choose_dataset(self) -> None:
        start = self.context.state.dataset_root or str(TOOLBOX_ROOT)
        path = QFileDialog.getExistingDirectory(self, "选择项目数据集根目录", start)
        if path:
            self.context.set_dataset_root(path)

    def _apply_dataset_text(self) -> None:
        value = self.dataset_edit.text().strip()
        if value:
            self.context.set_dataset_root(value)
        elif self.context.state.dataset_root:
            self.context.update(dataset_root="")

    def _choose_directory(self, field: str) -> None:
        start = getattr(self.context.state, field) or self.context.state.dataset_root or str(TOOLBOX_ROOT)
        path = QFileDialog.getExistingDirectory(self, "选择目录", start)
        if path:
            self.context.update(**{field: path})

    def _choose_model(self) -> None:
        start = self.context.state.model_path or self.context.state.output_root or str(TOOLBOX_ROOT)
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择当前模型",
            start,
            "模型 (*.onnx *.pth *.pt *.ckpt *.yaml);;所有文件 (*)",
        )
        if path:
            self.context.update(model_path=path)

    def _load_runtime_environments(self) -> None:
        environments = self.conda_manager.list_envs()
        selected_name = self.context.state.runtime_profile
        if environments and self.conda_manager.find(selected_name) is None:
            selected = next((item for item in environments if item.is_active), environments[0])
            selected_name = selected.name
            self.context.update(runtime_profile=selected_name)
        self.navigation.set_runtime_environments(environments, selected_name)

    def _on_runtime_profile_changed(self, name: str) -> None:
        if not name or self.conda_manager.find(name) is None:
            return
        self.context.update(runtime_profile=name)
        self.show_status(f"后续任务将使用 Conda 环境：{name}")

    def _on_context_changed(self, state: ProjectState) -> None:
        self.navigation.set_runtime_profile(state.runtime_profile)
        self.project_header.apply_state(state)
        self.inspector.apply_state(state)
        apply_overview = getattr(self.overview, "apply_project_context", None)
        if callable(apply_overview):
            apply_overview(state)
        for key, workspace in self._workspace_widgets.items():
            adapter = self._adapters_by_key.get(key)
            if adapter is not None:
                adapter.apply_context(workspace, state)
        for widget in self._native_widgets.values():
            apply = getattr(widget, "apply_project_context", None)
            if callable(apply) and widget is not self.overview:
                apply(state)
        self.context_status.setText(
            f"当前项目：{state.project_name or '未命名'} | 数据集：{state.dataset_root or '未选择'}"
        )

    def _on_session_changed(self) -> None:
        self.inspector.apply_artifacts(self.project_session.recent())

    def _on_task_finished(self, task) -> None:
        if task.status != TaskStatus.SUCCESS or not task.output_path:
            return
        if not Path(task.output_path).exists():
            return
        if any(item.source_task_id == task.task_id for item in self.project_session.artifacts):
            return
        self.project_session.register_artifact(
            kind=ArtifactKind.RUN_OUTPUT,
            name=task.title,
            path=task.output_path,
            source_capability=task.capability_key,
            source_task_id=task.task_id,
        )

    def closeEvent(self, event) -> None:
        self.page_router.reset()
        widgets = (*self._native_widgets.values(), *self._workspace_widgets.values())
        seen: set[int] = set()
        for widget in widgets:
            identity = id(widget)
            if identity in seen:
                continue
            seen.add(identity)
            request_shutdown(widget)
        super().closeEvent(event)


def _toolbox_stylesheet() -> str:
    return """
    QMainWindow#toolboxShell, QWidget#shellRoot { background: #ffffff; color: #202428; }
    QFrame#projectHeader { background: #ffffff; border-bottom: 1px solid #cfd3d7; }
    QLabel#projectHeaderTitle { color: #202428; font-size: 17px; font-weight: 700; }
    QLabel#projectHeaderBreadcrumb { color: #697077; font-size: 11px; }
    QLabel#contextChip { color: #4f565d; background: transparent; border: none; padding: 4px 8px; }
    QPushButton#headerSettingsButton { background: #ffffff; color: #202428; border: 1px solid #bfc5ca;
        border-radius: 3px; padding: 5px 12px; font-weight: 600; }

    QFrame#navigationRail { background: #eceeef; border-right: 1px solid #c9cdd1; }
    QWidget#navigationContent { background: #eceeef; }
    QLabel#navigationBrand { color: #202428; font-size: 17px; font-weight: 750; letter-spacing: 1px; }
    QLabel#navigationSubtitle { color: #747b82; font-size: 11px; }
    QLabel#navigationSection { color: #767d84; font-size: 10px; font-weight: 700;
        padding: 10px 8px 4px 8px; }
    QPushButton#navigationItem { text-align: left; color: #30363b; background: transparent;
        border: none; border-left: 3px solid transparent; border-radius: 0; min-height: 30px;
        padding: 3px 9px; font-weight: 500; }
    QPushButton#navigationItem:hover { background: #e2e5e7; color: #202428; }
    QPushButton#navigationItem:checked { background: #d8e3ee; color: #183f66;
        border-left: 3px solid #356a9a; font-weight: 700; }
    QLabel#runtimeLabel { color: #646b72; font-size: 10px; font-weight: 600; }
    QComboBox#runtimeSelector { color: #30363b; background: #ffffff; border: 1px solid #bfc5ca;
        border-radius: 3px; min-height: 26px; padding: 1px 6px; }
    QComboBox#runtimeSelector:disabled { color: #747b82; background: #e4e6e8; }

    QFrame#activityHost { background: #ffffff; }
    QFrame#activityTitleBar { background: transparent; }
    QLabel#activityTitle { color: #202428; font-size: 20px; font-weight: 700; }
    QLabel#activityDescription { color: #697077; font-size: 12px; }
    QStackedWidget#activityStack { background: transparent; }

    QFrame#projectInspector { background: #ffffff; border-left: 1px solid #cfd3d7; padding: 12px; }
    QLabel#inspectorTitle { color: #202428; font-size: 15px; font-weight: 700; }
    QLabel#inspectorSection { color: #202428; font-size: 12px; font-weight: 700; margin-top: 6px; }
    QLabel#fieldLabel { color: #646b72; font-size: 11px; font-weight: 600; }
    QPushButton#inspectorClose { border: 1px solid #cfd3d7; background: #ffffff; color: #555d64; border-radius: 3px; }
    QListWidget#artifactList, QListWidget#artifactsActivityList { background: #ffffff; border: 1px solid #cfd3d7;
        border-radius: 2px; padding: 3px; }

    QWidget#taskBar { background: #ffffff; border-top: 1px solid #e3e8ef; }
    QLabel#taskBarTitle { color: #26364d; font-weight: 700; }
    QPushButton#taskBarLink { color: #285b86; background: transparent; border: none; font-weight: 650; }
    QFrame#taskDrawerHeader { background: #ffffff; }
    QPushButton#taskDrawerToggle { border: none; background: transparent; color: #26364d; font-weight: 700;
        text-align: left; }
    QLabel#taskDrawerSummary { color: #718096; }
    QSplitter#taskDrawerBody { background: #f5f5f5; border-top: 1px solid #d7dadd; }

    QFrame#overviewHero, QFrame#pipelineBanner { background: #ffffff; border: 1px solid #cfd3d7; border-radius: 3px; }
    QLabel#overviewHeroTitle, QLabel#pipelineTitle { color: #202428; font-size: 18px; font-weight: 700; }
    QLabel#overviewHeroText, QLabel#pipelineSubtitle { color: #697077; font-size: 12px; }
    QLabel#readinessBadge, QLabel#pipelineStatus { color: #2f5b80; background: #e5edf4;
        border: 1px solid #c5d4e1; border-radius: 2px; padding: 4px 8px; font-weight: 700; }
    QLabel#sectionTitle { color: #202428; font-size: 15px; font-weight: 700; }
    QFrame#activityCard, QFrame#pipelineStep { background: #ffffff; border: 1px solid #cfd3d7;
        border-radius: 3px; }
    QLabel#activityCardNumber { color: #356a9a; font-size: 11px; font-weight: 700; }
    QLabel#activityCardTitle, QLabel#pipelineStepTitle { color: #202428; font-size: 15px; font-weight: 700; }
    QLabel#activityCardDescription, QLabel#pipelineStepDescription, QLabel#pipelineHint { color: #697077; }
    QLabel#pipelineStepNumber { color: #ffffff; background: #4b6f90; border-radius: 2px; font-weight: 700; }
    QLabel#pipelineStepState { color: #72551d; background: #f5eedc; border-radius: 2px; padding: 3px 6px; }
    QLabel#pipelineStepState[complete="true"] { color: #2f623f; background: #e3eee7; }
    QLabel#overviewSummary { color: #525960; background: #e9ebed; border-radius: 2px; padding: 9px; }

    QFrame#cabfCard { background: #ffffff; border: 1px solid #cfd3d7; border-radius: 3px; }
    QLabel#cabfCardTitle { color: #202428; font-size: 14px; font-weight: 700; }
    QLabel#cabfCardSubtitle, QLabel#cabfMuted { color: #697077; font-size: 11px; }
    QLabel#cabfSectionTitle { color: #202428; font-size: 12px; font-weight: 700; margin-top: 4px; }
    QLabel#cabfBadge { color: #2f5b80; background: #e5edf4; border: 1px solid #c5d4e1;
        border-radius: 2px; padding: 3px 7px; font-weight: 650; }
    QLabel#cabfStatus { color: #525960; background: #e9ebed; border-radius: 2px; padding: 7px 9px; }
    QLabel#cabfValidation { color: #525960; background: #eeeeee; border-radius: 2px; padding: 7px; }
    QLabel#cabfValidation[level="success"] { color: #2f623f; background: #e3eee7; }
    QLabel#cabfValidation[level="warning"] { color: #79521c; background: #f6ead8; }
    QWidget#cabfCanvas { background: #202428; border: 1px solid #4d5358; border-radius: 2px; }

    QFrame#embeddedPageHost { background: #ffffff; }
    QFrame#embeddedPageHeader { background: white; border: 1px solid #cfd3d7; border-radius: 3px; }
    QLabel#embeddedPageTitle { font-size: 17px; font-weight: 700; color: #202428; }
    QLabel#embeddedPageSubtitle { font-size: 12px; color: #697077; }
    QPushButton#embeddedBackButton { padding: 5px 10px; border: 1px solid #bfc5ca; border-radius: 3px;
        background: white; color: #30363b; }

    QMainWindow#toolboxShell QPushButton { min-height: 28px; padding: 3px 9px; border: 1px solid #bfc5ca;
        border-radius: 3px; background: #ffffff; color: #30363b; }
    QMainWindow#toolboxShell QPushButton:hover { background: #e9ebed; border-color: #9fa6ac; }
    QMainWindow#toolboxShell QPushButton[buttonRole="primary"] { background: #356a9a; color: white;
        border-color: #356a9a; }
    QMainWindow#toolboxShell QPushButton[buttonRole="secondary"] { background: #f4f4f4; color: #30363b; }
    QMainWindow#toolboxShell QPushButton[buttonRole="danger"] { background: #a83b32; color: white;
        border-color: #a83b32; }
    QMainWindow#toolboxShell QLineEdit, QMainWindow#toolboxShell QComboBox,
    QMainWindow#toolboxShell QSpinBox, QMainWindow#toolboxShell QDoubleSpinBox {
        min-height: 28px; border: 1px solid #bfc5ca; border-radius: 3px; background: white; padding: 2px 7px; }
    QMainWindow#toolboxShell QCheckBox { spacing: 7px; color: #202428; background: transparent; }
    QMainWindow#toolboxShell QCheckBox::indicator { width: 15px; height: 15px; border-radius: 3px; }
    QMainWindow#toolboxShell QCheckBox::indicator:unchecked {
        background: #ffffff; border: 1px solid #747d85; }
    QMainWindow#toolboxShell QCheckBox::indicator:unchecked:hover { border: 2px solid #356a9a; }
    QMainWindow#toolboxShell QCheckBox::indicator:checked {
        background: #356a9a; border: 1px solid #2c5d89; }
    QMainWindow#toolboxShell QCheckBox::indicator:disabled {
        background: #d7dadd; border: 1px solid #92989d; }
    QMainWindow#toolboxShell QCheckBox:disabled { color: #92989d; }
    QMainWindow#toolboxShell QGroupBox { border: 1px solid #cfd3d7; border-radius: 3px; margin-top: 9px;
        padding-top: 8px; background: #ffffff; }
    QMainWindow#toolboxShell QTabWidget::pane { border: 1px solid #cfd3d7; border-radius: 2px; background: white; }
    QMainWindow#toolboxShell QPushButton#navigationItem { text-align: left; color: #30363b;
        background: transparent; border: none; border-left: 3px solid transparent; border-radius: 0;
        min-height: 30px; padding: 3px 9px; font-weight: 500; }
    QMainWindow#toolboxShell QPushButton#navigationItem:hover { background: #e2e5e7; color: #202428; }
    QMainWindow#toolboxShell QPushButton#navigationItem:checked { background: #d8e3ee; color: #183f66;
        border-left: 3px solid #356a9a; font-weight: 700; }
    QMainWindow#toolboxShell QPushButton#headerSettingsButton { background: #ffffff; color: #202428;
        border: 1px solid #bfc5ca; border-radius: 3px; padding: 5px 12px; font-weight: 600; }
    QFrame#navigationRail QScrollBar:vertical { background: #eceeef; width: 7px; margin: 0; }
    QFrame#navigationRail QScrollBar::handle:vertical { background: #b5bbc0; min-height: 28px; border-radius: 2px; }
    QFrame#navigationRail QScrollBar::add-line:vertical, QFrame#navigationRail QScrollBar::sub-line:vertical {
        height: 0; background: transparent; }
    """


def main(workspace: str = "hub") -> int:
    ensure_import_paths()
    app = QApplication.instance()
    owns_app = app is None
    if app is None:
        app = QApplication(sys.argv)
    font = QFont(app.font())
    if font.pointSize() <= 0:
        font.setPointSize(10)
    app.setFont(font)
    app.setStyle("Fusion")

    initial_workspace = "overview" if workspace == "hub" else workspace
    window = ToolboxWindow(initial_workspace=initial_workspace)
    window.show()
    if owns_app:
        return int(app.exec())
    return 0
