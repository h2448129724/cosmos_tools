"""Embeddable labeling workspace.

The page owns the labeling tools and CAB-F launchers while deliberately
avoiding a second window chrome.  :class:`~.main_window.MainWindow` remains a
standalone adapter for the legacy entry point and delegates to this page when
it is hosted by the Cosmos workbench.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from apps.cabf_flow.config_model import load_config
from apps.cabf_flow.flow import CONFIG_PATH

from .project_registry import get_registered_projects
from .theme import APP_STYLESHEET
from .tools.batch_crop_page import BatchCropPage
from .tools.image_filter_page import ImageFilterPage
from .tools.inner_mask_page import InnerMaskPage
from .tools.keyword_split_page import KeywordSplitPage
from .tools.label_visualization_page import LabelVisualizationPage
from .tools.roi_editor_page import RoiConfigEditorPage
from .tools.tile_crop_page import AutoTileCropPage
from .tools.workflow_page import StitchWorkflowPage


class LabelingWorkspacePage(QWidget):
    """Business page that can be mounted directly in ``ToolboxWindow.stack``."""

    def __init__(self, parent: QWidget | None = None, config_path: Path | None = None):
        super().__init__(parent)
        self.config_path = Path(config_path or CONFIG_PATH)
        self.config_data = load_config(self.config_path)
        self._project_context = None
        self._project_state = None
        self._project_tool_registry = get_registered_projects()
        self._workspace_router = None
        self._workspace_key = "labeling"
        self._status_text = "就绪"
        self.setObjectName("labelingWorkspacePage")
        self.setStyleSheet(APP_STYLESHEET)

        # The registry is instantiated once here.  Standalone MainWindow and
        # embedded shell therefore share the exact same business page objects.
        self._tool_pages = [
            KeywordSplitPage(self),
            BatchCropPage(self),
            AutoTileCropPage(self),
            RoiConfigEditorPage(self),
            InnerMaskPage(self),
            ImageFilterPage(self),
            LabelVisualizationPage(self),
        ]
        self._project_tool_entries = (
            ("CAB-F 流程", "◎", "_show_cabf_workflow_dialog"),
            ("点边标注", "✎", "_show_graph_annotation_dialog"),
            ("数据筛选", "◇", "_show_point_filter_dialog"),
            ("数据集导出", "⇧", "_show_dataset_export_dialog"),
        )
        self._build_ui()
        self._tool_list.setCurrentRow(0)

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._tool_list = QListWidget(self)
        self._tool_list.hide()
        self._tool_list.currentRowChanged.connect(self._on_tool_selected)
        for page in self._tool_pages:
            item = QListWidgetItem(getattr(page, "tool_nav_title", page.tool_title))
            item.setData(Qt.UserRole, page.tool_key)
            self._tool_list.addItem(item)

        self.stack = QStackedWidget(self)
        for page in self._tool_pages:
            self.stack.addWidget(page)
        surface = QFrame(self)
        surface.setObjectName("surfaceCard")
        surface_layout = QVBoxLayout(surface)
        surface_layout.setContentsMargins(0, 0, 0, 0)
        surface_layout.addWidget(self.stack)
        root.addWidget(surface, 1)

        # Kept as a lightweight status sink for tool pages.  The shell owns
        # visible task/status chrome; standalone adapter mirrors this text.
        self.lbl_status = QLabel(self._status_text, self)
        self.lbl_status.hide()

    def _on_tool_selected(self, idx: int) -> None:
        if not 0 <= idx < len(self._tool_pages):
            return
        old = self.stack.currentIndex()
        if old != idx and 0 <= old < len(self._tool_pages):
            self._tool_pages[old].on_deactivated()
        self._tool_pages[idx].on_activated()
        self.stack.setCurrentIndex(idx)

    def show_status(self, msg: str) -> None:
        self._status_text = str(msg)
        self.lbl_status.setText(self._status_text)

    def bind_workspace_router(self, router, workspace_key: str = "labeling") -> None:
        self._workspace_router = router
        self._workspace_key = workspace_key

    def open_workspace_dialog(self, dialog: QDialog, title: str, subtitle: str = "", on_finished=None):
        if self._workspace_router is None:
            return dialog.exec()
        setter = getattr(dialog, "setStyleSheet", None)
        if callable(setter):
            setter(APP_STYLESHEET)
        return self._workspace_router.open_dialog(
            dialog, title=title, source_key=self._workspace_key,
            subtitle=subtitle, on_finished=on_finished,
        )

    def open_workspace_page(self, page: QWidget, title: str, subtitle: str = ""):
        if self._workspace_router is None:
            return None
        setter = getattr(page, "setStyleSheet", None)
        if callable(setter):
            setter(APP_STYLESHEET)
        return self._workspace_router.open_page(
            page, title=title, source_key=self._workspace_key, subtitle=subtitle,
        )

    def bind_project_context(self, context) -> None:
        self._project_context = context
        self.apply_project_context(context.state)

    def apply_project_context(self, state) -> None:
        self._project_state = state
        mapping = {
            "dataset_root": state.dataset_root,
            "master_images_dir": state.image_dir,
            "master_annotations_dir": state.annotation_dir,
        }
        for key, value in mapping.items():
            if value:
                self.config_data[key] = value
        if state.model_path:
            weights = self.config_data.setdefault("weights", {})
            if Path(state.model_path).suffix.lower() == ".onnx":
                weights["sew_point_onnx"] = state.model_path
            else:
                weights["sew_point_connector_pth"] = state.model_path
        if state.output_root:
            outputs = self.config_data.setdefault("outputs", {})
            outputs["sew_point_train_out"] = str(Path(state.output_root) / "sew_point")
            outputs["sew_point_conntect_train_out"] = str(Path(state.output_root) / "sew_point_connect")
        self.show_status(f"已同步项目：{state.project_name or '未命名'}")

    # ----------------------------------------------------------- CAB-F launchers
    def _show_graph_annotation_dialog(self):
        from .graph_annotation_dialog import StitchGraphEditorDialog
        dialog = StitchGraphEditorDialog(self)
        if self._project_state is not None:
            image_dir = self._project_state.image_dir or self._project_state.dataset_root
            annotation_dir = self._project_state.annotation_dir
            dialog.configure_paths(image_dir=image_dir, label_dir=annotation_dir,
                                   output_dir=annotation_dir,
                                   auto_open=bool(image_dir and annotation_dir))
            if self._project_state.selected_image and dialog.folder_items:
                selected = Path(self._project_state.selected_image).resolve()
                for index, item in enumerate(dialog.folder_items):
                    if item.image_path.resolve() == selected:
                        dialog.file_list.setCurrentRow(index)
                        break
        return self.open_workspace_dialog(dialog, "CAB-F 点边标注", "在同一工作台内修正缝纫点和连边关系。")

    def _show_point_annotation_dialog(self):
        return self._show_graph_annotation_dialog()

    def _show_point_filter_dialog(self):
        from .data_review import DatasetReviewDialog
        dialog = DatasetReviewDialog(self)
        if self._project_state is not None:
            dialog.configure_paths(image_dir=self._project_state.image_dir or self._project_state.dataset_root,
                                   label_dir=self._project_state.annotation_dir, auto_load=False)
        if self._project_context is not None:
            dialog.reviewCompleted.connect(
                lambda result: self._project_context.update(image_dir=str(result.active_image_dir))
            )
        return self.open_workspace_dialog(dialog, "样本审阅", "浏览图片和可选标注，保留有效样本或移除异常样本。")

    def _show_dataset_export_dialog(self):
        from .dataset_export_dialog import CabfDatasetToolDialog
        dialog = CabfDatasetToolDialog(self)
        if self._project_state is not None:
            image_dir = self._project_state.image_dir or self._project_state.dataset_root
            if image_dir:
                dialog.edit_image_dir.setText(image_dir)
            if self._project_state.annotation_dir:
                dialog.edit_annotation_dir.setText(self._project_state.annotation_dir)
            if self._project_state.output_root:
                root = Path(self._project_state.output_root)
                dialog.edit_model_a_output_dir.setText(str(root / "model_a"))
                dialog.edit_model_b_output_dir.setText(str(root / "model_b"))
        return self.open_workspace_dialog(dialog, "CAB-F 数据集导出", "校验母数据并导出训练数据集。")

    def _show_cabf_workflow_dialog(self):
        page = StitchWorkflowPage(self)
        if self._workspace_router is not None:
            return self.open_workspace_page(page, "CAB-F 缝纫点与连边流程", "从数据筛选到预测、修正、校验、导出和训练。")
        dlg = QDialog(self)
        dlg.setWindowTitle("CAB-F 缝纫点与连边流程")
        dlg.resize(1180, 760)
        layout = QVBoxLayout(dlg)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(page)
        return dlg.exec()

    def shutdown(self) -> None:
        for page in self._tool_pages:
            for method_name in ("shutdown", "cancel"):
                method = getattr(page, method_name, None)
                if callable(method):
                    method()
                    break
