"""Generic image tools main window."""
from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QApplication, QDialog, QFileDialog, QFrame, QHBoxLayout, QLabel,
    QListWidget, QListWidgetItem, QMainWindow, QMessageBox, QPushButton,
    QStatusBar, QVBoxLayout, QWidget,
)

from apps.cabf_flow.config_model import load_config
from apps.cabf_flow.flow import CONFIG_PATH

from .tools.workflow_page import StitchWorkflowPage
from .theme import APP_STYLESHEET, TOKENS
from .workspace_page import LabelingWorkspacePage
STYLESHEET = APP_STYLESHEET


class MainWindow(QMainWindow):
    def __init__(self, config_path: Path | None = None, parent: QWidget | None = None):
        super().__init__(parent)
        self.config_path = Path(config_path or CONFIG_PATH)
        self.config_data = load_config(self.config_path)
        self._project_context = None
        self._project_state = None
        self._project_tool_registry = ()
        self._sidebar_visible = True
        self._sidebar_expanded_width = 204
        self._sidebar_collapsed_width = 52
        self._workspace_router = None
        self._workspace_key = "labeling"
        self._embedded_mode = False

        self.setWindowTitle("通用图像工具")
        self.resize(1320, 820)
        self.setStyleSheet(APP_STYLESHEET)

        # Keep all business pages in one embeddable owner.  The standalone
        # adapter contributes only legacy sidebar/menu/status chrome.
        self._workspace_page = LabelingWorkspacePage(self, config_path=self.config_path)
        self._tool_pages = self._workspace_page._tool_pages
        self.config_data = self._workspace_page.config_data
        self._project_tool_registry = self._workspace_page._project_tool_registry

        self._build_ui()
        self._build_menu()
        self._tool_list.setCurrentRow(0)

    # ------------------------------------------------------------------- ui
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        sidebar = self._build_sidebar()
        self._sidebar = sidebar
        sidebar.setFixedWidth(self._sidebar_expanded_width)
        root.addWidget(sidebar)

        content_shell = QWidget()
        content_shell.setObjectName("contentShell")
        content_layout = QVBoxLayout(content_shell)
        self._content_layout = content_layout
        content_layout.setContentsMargins(12, 12, 12, 12)
        content_layout.setSpacing(6)

        topbar = QHBoxLayout()
        self._btn_toggle_sidebar = QPushButton("收起工具栏")
        self._btn_toggle_sidebar.setMinimumHeight(34)
        self._btn_toggle_sidebar.clicked.connect(lambda checked=False: self._toggle_sidebar())
        topbar.addWidget(self._btn_toggle_sidebar)
        topbar.addStretch(1)
        content_layout.addLayout(topbar)

        page_surface = QFrame()
        page_surface.setObjectName("surfaceCard")
        page_surface_layout = QVBoxLayout(page_surface)
        page_surface_layout.setContentsMargins(0, 0, 0, 0)

        self.stack = self._workspace_page.stack
        page_surface_layout.addWidget(self._workspace_page)
        content_layout.addWidget(page_surface, 1)
        root.addWidget(content_shell, 1)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.lbl_status = QLabel("就绪")
        self.lbl_status.setStyleSheet("color:#64748b;")
        self.status_bar.addWidget(self.lbl_status)

    # -------------------------------------------------------------- sidebar
    def _build_sidebar(self) -> QWidget:
        frame = QFrame()
        frame.setObjectName("sidebar")
        lay = QVBoxLayout(frame)
        lay.setContentsMargins(12, 14, 12, 10)
        lay.setSpacing(6)

        # app title row with collapse button
        title_row = QHBoxLayout()
        title_row.setSpacing(8)
        self._sidebar_title = QLabel("通用")
        self._sidebar_title.setStyleSheet(
            f"color:{TOKENS['text_main']}; font-size:22px; font-weight:700; letter-spacing:0.3px;"
        )
        title_row.addWidget(self._sidebar_title)
        title_row.addStretch()

        self._btn_collapse = QPushButton("◁")
        self._btn_collapse.setFixedSize(28, 28)
        self._btn_collapse.setToolTip("收起侧边栏")
        self._btn_collapse.setStyleSheet(
            f"QPushButton{{border:none;background:transparent;color:{TOKENS['text_secondary']};font-size:16px;}}"
            f"QPushButton:hover{{background:{TOKENS['sidebar_hover']};border-radius:6px;}}"
        )
        self._btn_collapse.clicked.connect(lambda checked=False: self._toggle_sidebar())
        title_row.addWidget(self._btn_collapse)
        lay.addLayout(title_row)

        self._sidebar_subtitle = QLabel("工具集")
        self._sidebar_subtitle.setStyleSheet(
            f"color:{TOKENS['text_secondary']}; font-size:11px; margin-bottom:6px; letter-spacing:0;"
        )
        lay.addWidget(self._sidebar_subtitle)

        # divider
        div = QFrame()
        div.setFixedHeight(1)
        div.setStyleSheet(f"background: {TOKENS['border_main']}; margin: 0 0 6px 0;")
        lay.addWidget(div)

        hdr = QLabel("工具")
        hdr.setObjectName("sidebarHeader")
        lay.addWidget(hdr)

        self._tool_list = QListWidget()
        self._tool_list.setObjectName("toolList")
        self._tool_list.currentRowChanged.connect(self._on_tool_selected)
        for page in self._tool_pages:
            nav_title = getattr(page, "tool_nav_title", page.tool_title)
            icon = getattr(page, "tool_icon", "")
            item = QListWidgetItem(nav_title)
            item.setData(Qt.UserRole, page.tool_key)
            item.setData(Qt.UserRole + 1, icon)
            self._tool_list.addItem(item)
        lay.addWidget(self._tool_list)

        self._project_header = QLabel("CAB-F")
        self._project_header.setObjectName("sidebarHeader")
        lay.addWidget(self._project_header)
        self._project_tool_entries = (
            ("CAB-F 流程", "◎", "_show_cabf_workflow_dialog"),
            ("点边标注", "✎", "_show_graph_annotation_dialog"),
            ("数据筛选", "◇", "_show_point_filter_dialog"),
            ("数据集导出", "⇧", "_show_dataset_export_dialog"),
        )
        self._project_tool_list = QListWidget()
        self._project_tool_list.setObjectName("toolList")
        self._project_tool_list.setFixedHeight(154)
        self._project_tool_list.currentRowChanged.connect(self._on_project_tool_selected)
        for title, icon, launcher_name in self._project_tool_entries:
            item = QListWidgetItem(title)
            item.setData(Qt.UserRole, launcher_name)
            item.setData(Qt.UserRole + 1, icon)
            self._project_tool_list.addItem(item)
        lay.addWidget(self._project_tool_list)

        self._sidebar_footnote = QLabel("")
        self._sidebar_footnote.setObjectName("sidebarFootnote")
        self._sidebar_footnote.setWordWrap(True)
        lay.addStretch(1)
        return frame

    def _toggle_sidebar(self, expanded: bool | None = None):
        if expanded is None:
            expanded = not self._sidebar_visible
        self._sidebar_visible = expanded
        if expanded:
            self._sidebar.setFixedWidth(self._sidebar_expanded_width)
            self._sidebar_title.setText("通用")
            self._sidebar_subtitle.show()
            self._btn_collapse.setText("◁")
            self._btn_collapse.setToolTip("收起侧边栏")
            self._btn_toggle_sidebar.setText("收起工具栏")
            self._project_header.setText("CAB-F")
            for i in range(self._tool_list.count()):
                item = self._tool_list.item(i)
                item.setText(getattr(self._tool_pages[i], "tool_nav_title", self._tool_pages[i].tool_title))
            self._tool_list.setStyleSheet("")
            for i, (title, _icon, _launcher) in enumerate(self._project_tool_entries):
                self._project_tool_list.item(i).setText(title)
            self._project_tool_list.setStyleSheet("")
        else:
            self._sidebar.setFixedWidth(self._sidebar_collapsed_width)
            self._sidebar_title.setText("G")
            self._sidebar_subtitle.hide()
            self._btn_collapse.setText("▷")
            self._btn_collapse.setToolTip("展开侧边栏")
            self._btn_toggle_sidebar.setText("展开工具栏")
            self._project_header.setText("C")
            for i in range(self._tool_list.count()):
                item = self._tool_list.item(i)
                icon = item.data(Qt.UserRole + 1) or "•"
                item.setText(icon)
            self._tool_list.setStyleSheet(
                "#toolList::item{padding:10px 0;text-align:center;font-size:18px;}"
            )
            for i in range(self._project_tool_list.count()):
                item = self._project_tool_list.item(i)
                item.setText(item.data(Qt.UserRole + 1) or "•")
            self._project_tool_list.setStyleSheet(
                "#toolList::item{padding:10px 0;text-align:center;font-size:18px;}"
            )

    def _on_tool_selected(self, idx):
        if 0 <= idx < len(self._tool_pages):
            old = self.stack.currentIndex()
            if old != idx and 0 <= old < len(self._tool_pages):
                self._tool_pages[old].on_deactivated()
            self._tool_pages[idx].on_activated()
            self.stack.setCurrentIndex(idx)
            self._refresh_overview(idx)

    def _refresh_overview(self, idx: int) -> None:
        if not 0 <= idx < len(self._tool_pages):
            return
        page = self._tool_pages[idx]
        self.setWindowTitle(f"通用图像工具 - {page.tool_title}")

    def _on_project_tool_selected(self, idx: int) -> None:
        if not 0 <= idx < self._project_tool_list.count():
            return
        item = self._project_tool_list.item(idx)
        launcher = getattr(self, item.data(Qt.UserRole), None)
        self._project_tool_list.blockSignals(True)
        self._project_tool_list.setCurrentRow(-1)
        self._project_tool_list.blockSignals(False)
        if callable(launcher):
            launcher()

    # ----------------------------------------------------------------- menu
    def _build_menu(self):
        menubar = self.menuBar()

        file_menu = menubar.addMenu("文件(&F)")
        open_cfg_act = file_menu.addAction("打开配置...")
        open_cfg_act.triggered.connect(self._open_config)
        file_menu.addSeparator()
        quit_act = file_menu.addAction("退出\tCtrl+Q")
        quit_act.triggered.connect(self.close)

        if self._project_tool_registry:
            project_menu = menubar.addMenu("项目适配(&P)")
            for project in self._project_tool_registry:
                project_act = project_menu.addAction(project.menu_title)
                project_act.triggered.connect(
                    lambda checked=False, pk=project.key: self._show_project_launcher_dialog(pk)
                )

        help_menu = menubar.addMenu("帮助(&H)")
        about_act = help_menu.addAction("关于")
        about_act.triggered.connect(self._show_about)

        # Global keyboard shortcuts
        for i, page in enumerate(self._tool_pages):
            shortcut = QKeySequence(f"Ctrl+{i + 1}")
            act = QAction(f"切换到 {page.tool_nav_title}", self)
            act.setShortcut(shortcut)
            act.triggered.connect(lambda checked=False, idx=i: self._tool_list.setCurrentRow(idx))
            self.addAction(act)

        save_act = QAction("保存配置", self)
        save_act.setShortcut(QKeySequence.Save)
        save_act.triggered.connect(self._save_current_tool)
        self.addAction(save_act)

    # ---------------------------------------------------------- shared helpers
    def show_status(self, msg: str):
        self.lbl_status.setText(msg)

    def bind_workspace_router(self, router, workspace_key: str = "labeling") -> None:
        self._workspace_router = router
        self._workspace_key = workspace_key
        self._workspace_page.bind_workspace_router(router, workspace_key=workspace_key)
        self._set_embedded_mode(True)

    def _set_embedded_mode(self, embedded: bool) -> None:
        """Let the Cosmos shell own navigation and page chrome when embedded."""
        self._embedded_mode = bool(embedded)
        show_internal_chrome = not self._embedded_mode
        self._sidebar.setVisible(show_internal_chrome)
        self._btn_toggle_sidebar.setVisible(show_internal_chrome)
        self.menuBar().setVisible(show_internal_chrome)
        self.statusBar().setVisible(show_internal_chrome)
        self._content_layout.setContentsMargins(
            0 if self._embedded_mode else 12,
            0 if self._embedded_mode else 12,
            0 if self._embedded_mode else 12,
            0 if self._embedded_mode else 12,
        )
        self._content_layout.setSpacing(0 if self._embedded_mode else 6)

    def open_workspace_dialog(self, dialog: QDialog, title: str, subtitle: str = "", on_finished=None):
        if self._workspace_router is None:
            return dialog.exec()
        set_stylesheet = getattr(dialog, "setStyleSheet", None)
        if callable(set_stylesheet):
            set_stylesheet(APP_STYLESHEET)
        return self._workspace_router.open_dialog(
            dialog,
            title=title,
            source_key=self._workspace_key,
            subtitle=subtitle,
            on_finished=on_finished,
        )

    def open_workspace_page(self, page: QWidget, title: str, subtitle: str = ""):
        if self._workspace_router is None:
            return None
        set_stylesheet = getattr(page, "setStyleSheet", None)
        if callable(set_stylesheet):
            set_stylesheet(APP_STYLESHEET)
        return self._workspace_router.open_page(
            page,
            title=title,
            source_key=self._workspace_key,
            subtitle=subtitle,
        )

    def refresh_tool_overview(self) -> None:
        if not hasattr(self, "stack"):
            return
        idx = self.stack.currentIndex()
        if idx >= 0:
            self._refresh_overview(idx)

    def _save_current_tool(self):
        idx = self.stack.currentIndex()
        if 0 <= idx < len(self._tool_pages):
            page = self._tool_pages[idx]
            if hasattr(page, "_sync_form"):
                page._sync_form()
                self.show_status("配置已保存")

    def _open_config(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "打开配置", str(self.config_path.parent), "JSON (*.json)"
        )
        if path:
            self.config_path = Path(path)
            self.config_data = load_config(self.config_path)

    # ----------------------------------------------------------- tool dialogs
    def _show_graph_annotation_dialog(self):
        from .graph_annotation_dialog import StitchGraphEditorDialog
        dialog = StitchGraphEditorDialog(self)
        if self._project_state is not None:
            image_dir = self._project_state.image_dir or self._project_state.dataset_root
            annotation_dir = self._project_state.annotation_dir
            dialog.configure_paths(
                image_dir=image_dir,
                label_dir=annotation_dir,
                output_dir=annotation_dir,
                auto_open=bool(image_dir and annotation_dir),
            )
            if self._project_state.selected_image and dialog.folder_items:
                selected = Path(self._project_state.selected_image).resolve()
                for index, item in enumerate(dialog.folder_items):
                    if item.image_path.resolve() == selected:
                        dialog.file_list.setCurrentRow(index)
                        break
        return self.open_workspace_dialog(
            dialog,
            "CAB-F 点边标注",
            "在同一工作台内修正缝纫点和连边关系。",
        )

    def _show_point_annotation_dialog(self):
        self._show_graph_annotation_dialog()

    def _show_point_filter_dialog(self):
        from .data_review import DatasetReviewDialog

        dialog = DatasetReviewDialog(self)
        if self._project_state is not None:
            dialog.configure_paths(
                image_dir=self._project_state.image_dir or self._project_state.dataset_root,
                label_dir=self._project_state.annotation_dir,
                auto_load=False,
            )
        if self._project_context is not None:
            dialog.reviewCompleted.connect(
                lambda result: self._project_context.update(image_dir=str(result.active_image_dir))
            )
        return self.open_workspace_dialog(
            dialog,
            "样本审阅",
            "浏览图片和可选标注，保留有效样本或移除异常样本。",
        )

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
        return self.open_workspace_dialog(
            dialog,
            "CAB-F 数据集导出",
            "校验母数据并导出训练数据集。",
        )

    def _show_cabf_workflow_dialog(self):
        page = StitchWorkflowPage(self)
        if self._workspace_router is not None:
            return self.open_workspace_page(
                page,
                "CAB-F 缝纫点与连边流程",
                "从数据筛选到预测、修正、校验、导出和训练。",
            )
        dlg = QDialog(self)
        dlg.setWindowTitle("CAB-F 缝纫点与连边流程")
        dlg.resize(1180, 760)
        layout = QVBoxLayout(dlg)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(page)
        return dlg.exec()

    def _show_project_launcher_dialog(self, project_key: str):
        project = next((item for item in self._project_tool_registry if item.key == project_key), None)
        if project is None:
            QMessageBox.warning(self, "提示", f"未找到项目工具配置: {project_key}")
            return
        from .project_launcher_dialog import ProjectToolsHubDialog, ProjectToolEntry
        entries = []
        for tool in project.tools:
            launch = getattr(self, tool.launcher_name, None)
            if callable(launch):
                entries.append(ProjectToolEntry(key=tool.key, title=tool.title,
                                                description=tool.description, launch=launch,
                                                category=tool.category, featured=tool.featured))
        dlg = ProjectToolsHubDialog(project.display_name, entries, self)
        return self.open_workspace_dialog(
            dlg,
            f"{project.menu_title} 项目工具",
            "选择项目流程或专项数据工具。",
        )

    def _show_about(self):
        QMessageBox.about(self, "关于", "通用图像工具\n\n通用裁剪、ROI、筛选、标注与项目适配入口。")

    def bind_project_context(self, context) -> None:
        self._project_context = context
        self.apply_project_context(context.state)
        self._workspace_page.bind_project_context(context)

    def apply_project_context(self, state) -> None:
        self._project_state = state
        self._workspace_page.apply_project_context(state)
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

    def shutdown(self) -> None:
        """Forward the toolbox lifecycle contract to persistent tool pages."""
        for page in self._tool_pages:
            for method_name in ("shutdown", "cancel"):
                method = getattr(page, method_name, None)
                if callable(method):
                    method()
                    break


def launch_standalone(config_path: Path | None = None) -> int:
    app = QApplication.instance()
    owns = app is None
    if app is None:
        app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(STYLESHEET)
    window = MainWindow(config_path=config_path)
    window.show()
    if owns:
        return int(app.exec())
    window.raise_()
    window.activateWindow()
    return 0
