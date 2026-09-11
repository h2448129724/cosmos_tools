from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMainWindow, QWidget

from .paths import TOOLBOX_ROOT
from .project_context import ProjectContext, ProjectState


def request_shutdown(widget: QWidget) -> bool:
    """Invoke the toolbox lifecycle contract on a page or workspace.

    ``shutdown`` is the preferred full teardown hook. ``cancel`` is the
    lightweight fallback for pages that only expose task cancellation.
    """
    for method_name in ("shutdown", "cancel"):
        method = getattr(widget, method_name, None)
        if callable(method):
            method()
            return True
    return False


@dataclass(frozen=True, slots=True)
class WorkspaceAdapter:
    key: str
    title: str
    description: str
    factory: Callable[[QWidget], QWidget]

    def create(self, parent: QWidget, context: ProjectContext, router=None) -> QWidget:
        workspace = self.factory(parent)
        if workspace.parentWidget() is not parent:
            workspace.setParent(parent)
        # Native workspace pages are already ordinary widgets.  Window-flag
        # mutation belongs exclusively to the migration adapter for genuine
        # legacy QMainWindow implementations.
        if isinstance(workspace, QMainWindow):
            workspace.setWindowFlag(Qt.WindowType.Window, False)
            workspace.setWindowFlag(Qt.WindowType.Dialog, False)
            workspace.setWindowFlag(Qt.WindowType.Widget, True)
            workspace.menuBar().hide()
        binder = getattr(workspace, "bind_project_context", None)
        if callable(binder):
            binder(context)
        else:
            self.apply_context(workspace, context.state)
        router_binder = getattr(workspace, "bind_workspace_router", None)
        if router is not None and callable(router_binder):
            router_binder(router, workspace_key=self.key)
        return workspace

    @staticmethod
    def apply_context(workspace: QWidget, state: ProjectState) -> None:
        apply = getattr(workspace, "apply_project_context", None)
        if callable(apply):
            apply(state)

    @staticmethod
    def shutdown(workspace: QWidget) -> bool:
        return request_shutdown(workspace)


def default_workspace_adapters() -> tuple[WorkspaceAdapter, ...]:
    def image_factory(parent: QWidget) -> QWidget:
        from img_tools.ui.main_window import workspace_page

        return workspace_page(parent)

    def labeling_factory(parent: QWidget) -> QWidget:
        from apps.labeling_ui.app.workspace_page import LabelingWorkspacePage

        return LabelingWorkspacePage(parent=parent)

    def training_factory(parent: QWidget) -> QWidget:
        from trainer_gui.workspace_page import workspace_page

        return workspace_page(TOOLBOX_ROOT, parent=parent)

    return (
        WorkspaceAdapter("image", "素材与 ROI", "浏览图片、检查像素并编辑 ROI；批量处理请从能力导航进入。", image_factory),
        WorkspaceAdapter("labeling", "标注与数据", "CAB-F 标注、筛选、校验、数据导出和项目流程。", labeling_factory),
        WorkspaceAdapter("training", "训练与推理", "训练、推理、评估、ONNX 导出和运行历史。", training_factory),
    )
