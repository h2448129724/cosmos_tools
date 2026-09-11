from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence
from weakref import ref

from PySide6.QtGui import QFont, QResizeEvent
from PySide6.QtWidgets import QApplication, QFileDialog, QHBoxLayout, QLabel, QMainWindow, QVBoxLayout, QWidget

from apps.cosmos_pipeline.outcome import PipelineOutcome, should_register_artifact

from .capabilities import Capability, CapabilityCatalog, STAGE_TITLES
from .default_capabilities import build_capability_catalog
from .page_router import PageRouter
from .paths import TOOLBOX_ROOT, ensure_import_paths
from .project_context import ProjectContext, ProjectState
from .project_session import ArtifactKind, ProjectSession
from .shell_widgets import ActivityHost, NavigationRail, ProjectHeader, ProjectInspector, TaskStatusStrip
from .task_center import TaskCenter
from .ui.theme import build_toolbox_stylesheet
from .workbench_state import (
    WorkbenchEvent,
    WorkbenchEventKind,
    WorkbenchIntentKind,
    WorkbenchState,
    transition_workbench,
)
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
        self._workbench_state = WorkbenchState(viewport_width=1540)

        self.setObjectName("toolboxShell")
        self.setWindowTitle("Cosmos 个人工具箱")
        self.resize(1540, 920)
        self.setMinimumSize(960, 640)
        self.setStyleSheet(_toolbox_stylesheet())
        self.setCentralWidget(self._build_ui())

        self.context.changed.connect(self._on_context_changed)
        self.project_session.changed.connect(self._on_session_changed)
        self.task_center.changed.connect(self._on_tasks_changed)
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
        self.inspector.close_requested.connect(self._close_inspector)
        self.inspector.dataset_requested.connect(self._choose_dataset)
        self.inspector.model_requested.connect(self._choose_model)
        self.inspector.image_dir_requested.connect(lambda: self._choose_directory("image_dir"))
        self.inspector.annotation_dir_requested.connect(lambda: self._choose_directory("annotation_dir"))
        self.inspector.output_root_requested.connect(lambda: self._choose_directory("output_root"))
        self.inspector.hide()
        content_layout.addWidget(self.inspector)
        work_layout.addWidget(content, 1)

        self.task_status_strip = TaskStatusStrip()
        self.task_status_strip.open_requested.connect(lambda: self.navigate("project.tasks"))
        self.task_drawer = self.task_status_strip
        work_layout.addWidget(self.task_status_strip)
        body_layout.addWidget(work_area, 1)
        root_layout.addWidget(body, 1)

        self.page_router = PageRouter(self.stack, self._return_to_workspace, self)
        self.page_router.depth_changed.connect(self._on_route_depth_changed)

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
        self._apply_layout(self._workbench_state.layout)
        return root

    def navigate(self, key: str) -> None:
        capability = self.catalog.find(key)
        if capability is None:
            return
        self._capture_navigation_origin()
        self._dispatch_workbench(WorkbenchEvent(WorkbenchEventKind.NAVIGATE, key))

    def _on_navigation_changed(self, row: int) -> None:
        key = self.navigation.tabData(row)
        capability = self.catalog.find(str(key)) if key else None
        if capability is None:
            return
        self.navigate(capability.key)

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
        self._dispatch_workbench(
            WorkbenchEvent(WorkbenchEventKind.ACTIVITY_MOUNTED, bool(widget.property("ownsPageHeader")))
        )
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
        transition = transition_workbench(
            self._workbench_state,
            WorkbenchEvent(WorkbenchEventKind.NAVIGATE, capability.key),
        )
        self._workbench_state = transition.state
        self._show_capability(capability, activate=False)

    def _show_workspace(self, key: str) -> None:
        """Compatibility entry used by older tests and callers."""
        capability = self.catalog.find(key)
        if capability is not None:
            self._show_capability(capability)

    def _toggle_inspector(self) -> None:
        self._dispatch_workbench(WorkbenchEvent(WorkbenchEventKind.INSPECTOR_TOGGLED))

    def _close_inspector(self) -> None:
        self._dispatch_workbench(WorkbenchEvent(WorkbenchEventKind.INSPECTOR_CLOSED))

    def _on_route_depth_changed(self, depth: int) -> None:
        self._dispatch_workbench(WorkbenchEvent(WorkbenchEventKind.ROUTE_DEPTH_CHANGED, depth))

    def _on_tasks_changed(self) -> None:
        self._dispatch_workbench(
            WorkbenchEvent(WorkbenchEventKind.TASKS_CHANGED, self.task_center.active_count)
        )

    def _dispatch_workbench(self, event: WorkbenchEvent) -> None:
        transition = transition_workbench(self._workbench_state, event)
        self._workbench_state = transition.state
        for intent in transition.intents:
            if intent.kind is WorkbenchIntentKind.RESET_ROUTES:
                self.page_router.reset()
            elif intent.kind is WorkbenchIntentKind.SELECT_NAVIGATION:
                row = self._nav_rows.get(str(intent.value), -1)
                if row >= 0 and self.navigation.currentIndex() != row:
                    blocked = self.navigation.blockSignals(True)
                    self.navigation.setCurrentIndex(row)
                    self.navigation.blockSignals(blocked)
            elif intent.kind is WorkbenchIntentKind.PRESENT_ACTIVITY:
                capability = self.catalog.find(str(intent.value))
                if capability is not None:
                    self._show_capability(capability)
            elif intent.kind is WorkbenchIntentKind.UPDATE_ACTIVITY_HEADER:
                self.activity_host.header.setVisible(bool(intent.value))
            elif intent.kind is WorkbenchIntentKind.UPDATE_INSPECTOR:
                self.inspector.setVisible(bool(intent.value))
            elif intent.kind is WorkbenchIntentKind.UPDATE_LAYOUT:
                self._apply_layout(intent.value)
            elif intent.kind is WorkbenchIntentKind.UPDATE_TASK_SUMMARY:
                self.task_status_strip.apply_presentations(self.task_center.presentations)

    def _apply_layout(self, presentation) -> None:
        self.navigation.set_compact(presentation.navigation_compact)
        self.activity_host.apply_layout(presentation)
        self.inspector.apply_layout(presentation)

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        if hasattr(self, "_workbench_state") and hasattr(self, "navigation"):
            self._dispatch_workbench(
                WorkbenchEvent(WorkbenchEventKind.VIEWPORT_RESIZED, event.size().width())
            )

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
        try:
            outcome = PipelineOutcome(task.status.value)
        except (AttributeError, ValueError):
            return
        has_output = bool(task.output_path)
        output_exists = has_output and Path(task.output_path).exists()
        already_registered = any(
            item.source_task_id == task.task_id for item in self.project_session.artifacts
        )
        if not should_register_artifact(
            outcome,
            has_output=has_output,
            output_exists=output_exists,
            already_registered=already_registered,
        ):
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
    return build_toolbox_stylesheet()


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
