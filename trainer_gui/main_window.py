from __future__ import annotations

import html
import re
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices, QTextCursor
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QScrollArea,
    QSplitter,
    QTabWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from modules.yolo.dataset_inspector import inspect_yolo_dataset
from .argparse_parser import ArgparseSchemaParser
from .conda_manager import CondaEnvManager
from .feature_scanner import FeatureScanner
from .history_manager import HistoryManager
from .models import CondaEnvInfo, FeatureAction, FeatureModule, RunRecord
from .run_manager import RunManager
from .settings_dialog import SettingsDialog
from .settings_manager import SettingsManager
from .theme import build_app_stylesheet, status_badge_stylesheet
from .widgets.form_builder import FormBuilder
from .widgets.navigation_panel import BasicConfigGroup, NavigationPanel
from .widgets.history_panel import HistoryPanel
from .widgets.log_panel import LogPanel
from .widgets.run_status_panel import RunStatusPanel
from .widgets.yolo_dataset_panel import YoloDatasetPanel


class MainWindow(QMainWindow):
    STATUS_META = {
        "idle": ("未开始", "#94a3b8", "#e2e8f0"),
        "running": ("运行中", "#2563eb", "#dbeafe"),
        "success": ("成功", "#15803d", "#dcfce7"),
        "failed": ("失败", "#b91c1c", "#fee2e2"),
        "stopped": ("已停止", "#c2410c", "#ffedd5"),
        "warning": ("警告", "#b45309", "#fef3c7"),
    }

    def __init__(self, project_root: str | Path, parent: QWidget | None = None):
        super().__init__(parent)
        self.project_root = Path(project_root).resolve()
        self.setWindowTitle("训练管理工作台")
        self.setMinimumSize(980, 640)
        self._resize_to_available_screen()
        self.setStyleSheet(build_app_stylesheet())

        self.settings_manager = SettingsManager(project_root)
        self.settings = self.settings_manager.load()
        self.history_manager = HistoryManager(self.project_root)
        self.conda_manager = CondaEnvManager()
        self.feature_scanner = FeatureScanner(self.project_root)
        self.schema_parser = ArgparseSchemaParser()
        self.run_manager = RunManager(self.project_root, self.history_manager)
        self.form_builder = FormBuilder(self)

        self.features: list[FeatureModule] = []
        self.conda_envs: list[CondaEnvInfo] = []
        self.current_feature: FeatureModule | None = None
        self.current_action: FeatureAction | None = None
        self.active_run_id: str | None = None
        self.active_record: RunRecord | None = None
        self.generated_command_text: str = ""
        self.active_log_lines: list[str] = []
        self._all_history: list[dict] = []
        self._project_context = None
        self._project_state = None
        self._workspace_router = None
        self._workspace_key = "training"
        self._embedded_mode = False

        self._build_ui()
        self._connect_signals()
        self._load_features()
        self._load_conda_envs()
        self._load_history()
        self._update_run_controls(False)
        self._apply_status_record(None)
        self._update_log_buttons()
        self._update_status_action_buttons()

    def _resize_to_available_screen(self) -> None:
        screen = QApplication.primaryScreen()
        if screen is None:
            self.resize(1360, 820)
            return

        available = screen.availableGeometry()
        target_width = min(1560, max(980, int(available.width() * 0.92)))
        target_height = min(940, max(640, int(available.height() * 0.88)))
        self.resize(target_width, target_height)

        frame = self.frameGeometry()
        frame.moveCenter(available.center())
        self.move(frame.topLeft())

    def _build_ui(self) -> None:
        central = QWidget(self)
        root_layout = QHBoxLayout(central)
        root_layout.setContentsMargins(14, 14, 14, 14)

        splitter = QSplitter(Qt.Horizontal, central)
        self.main_splitter = splitter
        root_layout.addWidget(splitter)

        # ---- 左栏：模块导航 ----
        self.nav_panel = NavigationPanel()
        self.feature_list = self.nav_panel.feature_list
        splitter.addWidget(self.nav_panel)

        # ---- 中栏：参数配置 ----
        config_scroll = QScrollArea()
        config_scroll.setWidgetResizable(True)
        config_scroll.setFrameShape(QFrame.Shape.NoFrame)
        config_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        config_widget = QWidget()
        config_layout = QVBoxLayout(config_widget)
        config_layout.setContentsMargins(0, 0, 0, 0)
        config_layout.setSpacing(12)

        self.feature_title = QLabel("选择一个训练功能")
        self.feature_title.setObjectName("titleLabel")
        self.feature_desc = QLabel("")
        self.feature_desc.setObjectName("mutedLabel")
        self.feature_desc.setWordWrap(True)
        self.summary_label = QLabel("扫描训练模块中...")
        self.summary_label.setObjectName("mutedLabel")
        self.summary_label.setWordWrap(True)

        self.basic_config = BasicConfigGroup(
            default_platform=self.settings.default_command_platform or "linux"
        )
        self.project_name_edit = self.basic_config.project_name_edit
        self.action_combo = self.basic_config.action_combo
        self.conda_env_combo = self.basic_config.conda_env_combo
        self.command_platform_combo = self.basic_config.command_platform_combo
        self.settings_button = self.basic_config.settings_button

        form_group = QGroupBox("参数配置")
        self.form_layout = QFormLayout(form_group)
        self.form_layout.setHorizontalSpacing(12)
        self.form_layout.setVerticalSpacing(10)
        self.form_layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        self.form_layout.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        self.yolo_dataset_panel = YoloDatasetPanel(self)

        button_row = QHBoxLayout()
        self.start_button = QPushButton("开始执行")
        self.start_button.setProperty("buttonRole", "primary")
        self.stop_button = QPushButton("停止当前任务")
        self.stop_button.setProperty("buttonRole", "danger")
        self.reset_form_button = QPushButton("恢复默认")
        self.reset_form_button.setProperty("buttonRole", "secondary")
        self.generate_command_button = QPushButton("查看服务命令")
        self.generate_command_button.setProperty("buttonRole", "secondary")
        button_row.addWidget(self.start_button)
        button_row.addWidget(self.stop_button)
        button_row.addWidget(self.reset_form_button)
        button_row.addWidget(self.generate_command_button)
        button_row.addStretch(1)

        config_layout.addWidget(self.feature_title)
        config_layout.addWidget(self.feature_desc)
        config_layout.addWidget(self.summary_label)
        config_layout.addWidget(self.basic_config)
        config_layout.addWidget(form_group, 1)
        config_layout.addWidget(self.yolo_dataset_panel)
        config_layout.addLayout(button_row)
        config_layout.addStretch(1)
        config_scroll.setWidget(config_widget)

        # ---- 右栏：运行监控（Tab 形式：运行+日志 / 历史 / 命令）----
        self.right_tabs = QTabWidget()

        # Tab 1: 运行状态 + 日志
        run_tab = QWidget()
        run_tab_layout = QVBoxLayout(run_tab)
        run_tab_layout.setContentsMargins(0, 0, 0, 0)
        run_tab_layout.setSpacing(8)

        self.run_status_panel = RunStatusPanel()
        self.status_badge = self.run_status_panel.status_badge
        self.status_run_id = self.run_status_panel.status_run_id
        self.status_project = self.run_status_panel.status_project
        self.status_feature = self.run_status_panel.status_feature
        self.status_action = self.run_status_panel.status_action
        self.status_start = self.run_status_panel.status_start
        self.status_end = self.run_status_panel.status_end
        self.status_duration = self.run_status_panel.status_duration
        self.status_exit = self.run_status_panel.status_exit
        self.status_output = self.run_status_panel.status_output
        self.status_artifacts = self.run_status_panel.status_artifacts
        self.status_cwd = self.run_status_panel.status_cwd
        self.status_python = self.run_status_panel.status_python
        self.status_conda = self.run_status_panel.status_conda
        self.status_repo_root = self.run_status_panel.status_repo_root
        self.status_command = self.run_status_panel.status_command
        self.open_output_button = self.run_status_panel.open_output_button
        self.copy_output_button = self.run_status_panel.copy_output_button
        self.copy_command_button = self.run_status_panel.copy_command_button
        self.view_full_log_button = self.run_status_panel.view_full_log_button

        self.log_panel = LogPanel()
        self.log_output = self.log_panel.log_output
        self.auto_scroll_checkbox = self.log_panel.auto_scroll_checkbox
        self.copy_log_button = self.log_panel.copy_log_button
        self.clear_log_button = self.log_panel.clear_log_button
        self.export_log_button = self.log_panel.export_log_button
        self.history_detail = self.log_panel.history_detail

        run_tab_layout.addWidget(self.run_status_panel, 0, Qt.AlignmentFlag.AlignTop)
        run_tab_layout.addWidget(self.log_panel, 1)

        # Tab 2: 历史记录
        self.history_panel = HistoryPanel()
        self.history_table = self.history_panel.table

        # Tab 3: 命令预览
        self.command_preview = QPlainTextEdit()
        self.command_preview.setReadOnly(True)
        self.command_preview.setPlaceholderText("配置参数后点击 \"查看服务命令\" 生成命令，或直接运行任务后查看。")

        self.right_tabs.addTab(run_tab, "运行")
        self.right_tabs.addTab(self.history_panel, "历史")
        self.right_tabs.addTab(self.command_preview, "命令")

        splitter.addWidget(config_scroll)
        splitter.addWidget(self.right_tabs)
        splitter.setSizes([280, 580, 680])

        self.setCentralWidget(central)


    def _connect_signals(self) -> None:
        self.feature_list.currentRowChanged.connect(self._on_feature_changed)
        self.action_combo.currentIndexChanged.connect(self._on_action_changed)
        self.start_button.clicked.connect(self._start_run)
        self.reset_form_button.clicked.connect(self._reset_form_defaults)
        self.generate_command_button.clicked.connect(self._generate_command)
        self.command_platform_combo.currentIndexChanged.connect(self._regenerate_command_if_needed)
        self.command_platform_combo.currentIndexChanged.connect(self._save_settings_state)
        self.conda_env_combo.currentIndexChanged.connect(self._save_settings_state)
        self.project_name_edit.textChanged.connect(self._refresh_idle_status_card)
        self.stop_button.clicked.connect(self._stop_run)
        self.settings_button.clicked.connect(self._open_settings)
        self.run_manager.log_received.connect(self._append_log)
        self.run_manager.run_started.connect(self._on_run_started)
        self.run_manager.run_finished.connect(self._on_run_finished)
        self.history_table.itemSelectionChanged.connect(self._show_history_detail)
        self.history_table.itemClicked.connect(self._apply_history_to_form_from_item)
        self.history_table.customContextMenuRequested.connect(self._show_history_menu)
        self.open_output_button.clicked.connect(self._open_active_output_dir)
        self.copy_output_button.clicked.connect(self._copy_active_output_dir)
        self.copy_command_button.clicked.connect(self._copy_active_command)
        self.view_full_log_button.clicked.connect(self._show_full_log_dialog)
        self.copy_log_button.clicked.connect(self._copy_log)
        self.clear_log_button.clicked.connect(self._clear_log)
        self.export_log_button.clicked.connect(self._export_log)
        # 历史过滤
        self.history_panel.status_filter.currentIndexChanged.connect(self._apply_history_filter)
        self.history_panel.search_edit.textChanged.connect(self._apply_history_filter)
        self.yolo_dataset_panel.inspect_requested.connect(self._inspect_yolo_dataset)

    def _load_features(self) -> None:
        self.features = [feature for feature in self.feature_scanner.scan() if feature.enabled]
        self.feature_list.clear()
        for feature in self.features:
            for action in feature.actions:
                self.schema_parser.parse_action(action)
            item = QListWidgetItem(feature.display_name)
            item.setToolTip(feature.description)
            self.feature_list.addItem(item)
        self.summary_label.setText(f"已发现 {len(self.features)} 个训练功能，可统一在这里配置和启动。")
        if not self.features:
            return

        restore_row = 0
        if self.settings.remember_last_selection and self.settings.last_feature_name:
            for index, feature in enumerate(self.features):
                if feature.feature_name == self.settings.last_feature_name:
                    restore_row = index
                    break
        self.feature_list.setCurrentRow(restore_row)

    def _load_conda_envs(self) -> None:
        self.conda_env_combo.clear()
        self.conda_envs = self.conda_manager.list_envs()
        self.conda_env_combo.addItem("当前环境 / PATH 默认值", None)
        for env in self.conda_envs:
            label = f"{env.name} ({env.prefix})"
            if env.is_active:
                label += " [active]"
            self.conda_env_combo.addItem(label, env)
        if self.settings.remember_last_selection and self.settings.last_conda_env_name:
            for index in range(self.conda_env_combo.count()):
                env = self.conda_env_combo.itemData(index)
                if isinstance(env, CondaEnvInfo) and env.name == self.settings.last_conda_env_name:
                    self.conda_env_combo.setCurrentIndex(index)
                    return
        if self.current_feature is not None:
            self._select_preferred_conda_env(self.current_feature.preferred_conda_env)

    def _load_history(self) -> None:
        self._all_history = self.history_manager.load_history()
        self._apply_history_filter()

    def _apply_history_filter(self) -> None:
        """根据当前过滤条件刷新历史表格。"""
        status_filter = self.history_panel.current_status_filter()
        search = self.history_panel.current_search_text()

        filtered = self._all_history
        if status_filter:
            filtered = [h for h in filtered if h.get("status") == status_filter]
        if search:
            filtered = [
                h for h in filtered
                if search in str(h.get("project_name", "")).lower()
                or search in str(h.get("feature_name", "")).lower()
                or search in str(h.get("action_name", "")).lower()
                or search in str(h.get("action_display_name", "")).lower()
            ]

        self.history_table.clearContents()
        self.history_table.setRowCount(len(filtered))
        for row, item in enumerate(filtered):
            project_item = QTableWidgetItem(str(item.get("project_name", "")))
            project_item.setData(Qt.ItemDataRole.UserRole, item)
            feature_item = QTableWidgetItem(str(item.get("feature_name", "")))
            action_item = QTableWidgetItem(str(item.get("action_display_name", item.get("action_name", ""))))
            start_item = QTableWidgetItem(self._format_time(item.get("start_time")))
            start_item.setToolTip(str(item.get("start_time", "")))
            duration_item = QTableWidgetItem(self._format_duration(item.get("duration_seconds")))
            output_dir = str(item.get("output_dir", "") or "")
            output_item = QTableWidgetItem(self._elide_middle(output_dir, 44))
            output_item.setToolTip(output_dir)
            output_item.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

            self.history_table.setItem(row, 0, project_item)
            self.history_table.setItem(row, 1, feature_item)
            self.history_table.setItem(row, 2, action_item)
            self.history_table.setItem(row, 4, start_item)
            self.history_table.setItem(row, 5, duration_item)
            self.history_table.setItem(row, 6, output_item)
            self.history_table.setCellWidget(row, 3, self._build_status_badge(str(item.get("status", "idle"))))
            self.history_table.setCellWidget(row, 7, self._build_history_actions(item))
            self.history_table.setRowHeight(row, 38)
        self.history_table.setColumnWidth(0, 110)
        self.history_table.setColumnWidth(1, 120)
        self.history_table.setColumnWidth(2, 120)
        self.history_table.setColumnWidth(3, 88)
        self.history_table.setColumnWidth(4, 148)
        self.history_table.setColumnWidth(5, 84)
        self.history_table.setColumnWidth(7, 260)
        if filtered:
            self.history_table.selectRow(0)

    def _on_feature_changed(self, row: int) -> None:
        if row < 0 or row >= len(self.features):
            return
        feature = self.features[row]
        self.current_feature = feature
        self.feature_title.setText(feature.display_name)
        self.feature_desc.setText(feature.description or "")
        self.project_name_edit.setText(self.settings.default_project_name or feature.default_project_name)
        self.action_combo.blockSignals(True)
        self.action_combo.clear()
        for action in feature.actions:
            self.action_combo.addItem(action.display_name, action)
        self.action_combo.blockSignals(False)
        if not feature.actions:
            return

        action_index = 0
        if self.settings.remember_last_selection and self.settings.last_feature_name == feature.feature_name and self.settings.last_action_name:
            for index, action in enumerate(feature.actions):
                if action.action_name == self.settings.last_action_name:
                    action_index = index
                    break
        self.action_combo.setCurrentIndex(action_index)
        self._apply_action(feature.actions[action_index])
        if self.active_record is None:
            self._apply_status_record(None)
        self._save_settings_state()

    def _on_action_changed(self, index: int) -> None:
        if index < 0:
            return
        action = self.action_combo.itemData(index)
        if isinstance(action, FeatureAction):
            self._apply_action(action)

    def _apply_action(self, action: FeatureAction) -> None:
        self.current_action = action
        self._hide_auto_output_fields(action)
        preferred_env = action.preferred_conda_env or (self.current_feature.preferred_conda_env if self.current_feature else None)
        if not (self.settings.remember_last_selection and self.settings.last_conda_env_name):
            self._select_preferred_conda_env(preferred_env)
        self.form_builder.build(self.form_layout, action.schema)
        self._apply_project_context_to_form()
        self._update_yolo_dataset_panel_visibility()
        feature_name = self.current_feature.feature_name if self.current_feature else ""
        self.summary_label.setText(
            f"当前功能: {feature_name} / {action.display_name} | 参数项: {len(action.schema)} | 推荐环境: {preferred_env or '当前环境'}"
        )
        self.generated_command_text = ""
        if self.active_record is None:
            self._apply_status_record(None)
        self._save_settings_state()

    def bind_project_context(self, context) -> None:
        self._project_context = context
        self.apply_project_context(context.state)

    def apply_project_context(self, state) -> None:
        self._project_state = state
        if state.project_name:
            self.project_name_edit.setText(state.project_name)
        self._select_preferred_conda_env(state.runtime_profile or "onnx-gpu")
        self._apply_project_context_to_form()

    def bind_workspace_router(self, router, workspace_key: str = "training") -> None:
        self._workspace_router = router
        self._workspace_key = workspace_key
        self._set_embedded_mode(True)

    def _set_embedded_mode(self, embedded: bool) -> None:
        """Let the project Shell own navigation while this page owns training work."""
        self._embedded_mode = bool(embedded)
        self.nav_panel.setVisible(not self._embedded_mode)
        self.feature_title.setVisible(not self._embedded_mode)
        self.feature_desc.setVisible(not self._embedded_mode)
        self.summary_label.setVisible(not self._embedded_mode)
        self.log_panel.setVisible(not self._embedded_mode)
        self.run_status_panel.set_compact_mode(self._embedded_mode)
        if self._embedded_mode:
            self.setMinimumSize(0, 0)
            self.main_splitter.setSizes([0, 760, 440])
        else:
            self.setMinimumSize(980, 640)
        central = self.centralWidget()
        if central is not None and central.layout() is not None:
            central.layout().setContentsMargins(0, 0, 0, 0) if self._embedded_mode else central.layout().setContentsMargins(14, 14, 14, 14)

    def open_feature(self, feature_name: str, action_name: str = "train") -> bool:
        """Public capability seam used by the Cosmos activity catalog."""
        selected = self._select_feature_action(feature_name, action_name)
        if selected:
            self._apply_project_context_to_form()
        return selected

    def _apply_project_context_to_form(self) -> None:
        if self._project_state is None or self.current_action is None:
            return
        state = self._project_state
        current = self.form_builder.values(self.current_action.schema)
        image_dir = state.image_dir or state.dataset_root
        candidates = {
            "image_dir": image_dir,
            "input_dir": image_dir,
            "source": image_dir,
            "annotation_dir": state.annotation_dir,
            "label_dir": state.annotation_dir,
            "model": state.model_path,
            "model_path": state.model_path,
            "custom_model": state.model_path,
            "stage1_model_path": state.model_path,
        }
        values = {
            name: value
            for name, value in candidates.items()
            if value and current.get(name) in (None, "")
        }
        if values:
            self.form_builder.set_values(self.current_action.schema, values)

    def _select_preferred_conda_env(self, preferred_name: str | None) -> None:
        if not preferred_name:
            self.conda_env_combo.setCurrentIndex(0)
            return
        for index in range(self.conda_env_combo.count()):
            env = self.conda_env_combo.itemData(index)
            if isinstance(env, CondaEnvInfo) and env.name == preferred_name:
                self.conda_env_combo.setCurrentIndex(index)
                return
        self.conda_env_combo.setCurrentIndex(0)

    def _update_yolo_dataset_panel_visibility(self) -> None:
        visible = self._is_yolo_train_action()
        self.yolo_dataset_panel.setVisible(visible)
        if visible:
            self.yolo_dataset_panel.reset()

    def _is_yolo_train_action(self) -> bool:
        return bool(
            self.current_feature is not None
            and self.current_feature.feature_name == "yolo"
            and self.current_action is not None
            and self.current_action.action_name == "train"
        )

    def _inspect_yolo_dataset(self) -> None:
        if not self._is_yolo_train_action():
            return
        params = self._collect_params(require_complete=False) or {}
        data_path = str(params.get("data") or "").strip()
        if not data_path:
            self.yolo_dataset_panel.set_error("请先选择 data.yaml。")
            return
        try:
            report = inspect_yolo_dataset(data_path)
        except Exception as exc:  # noqa: BLE001 - surface diagnostics in GUI
            self.yolo_dataset_panel.set_error(f"数据集检查失败: {exc}")
            return
        self.yolo_dataset_panel.set_report(report)

    def _start_run(self) -> None:
        if self.current_feature is None or self.current_action is None:
            QMessageBox.warning(self, "未选择功能", "请先选择一个训练功能。")
            return
        params = self._collect_params(require_complete=True)
        if params is None:
            return

        project_name = self.project_name_edit.text().strip() or self.current_feature.default_project_name
        conda_env = self.conda_env_combo.currentData()
        self._clear_log()
        try:
            record = self.run_manager.start_run(self.current_feature, self.current_action, project_name, params, conda_env)
        except Exception as exc:
            QMessageBox.critical(self, "启动失败", str(exc))
            return
        self.active_run_id = record.run_id
        self.active_record = record
        self._update_run_controls(True)

    def _generate_command(self) -> None:
        if self.current_feature is None or self.current_action is None:
            QMessageBox.warning(self, "未选择功能", "请先选择一个训练功能。")
            return
        params = self._collect_params(require_complete=True)
        if params is None:
            return

        conda_env = self.conda_env_combo.currentData()
        command_text = self.run_manager.build_export_command_text(
            self.current_feature,
            self.current_action,
            params,
            conda_env,
            self._selected_command_platform(),
        )
        self.generated_command_text = command_text
        self.command_preview.setPlainText(command_text)
        for i in range(self.right_tabs.count()):
            if self.right_tabs.tabText(i) == "命令":
                self.right_tabs.setCurrentIndex(i)
                break

    def _regenerate_command_if_needed(self) -> None:
        if self.generated_command_text.strip():
            self._generate_command()

    def _stop_run(self) -> None:
        if self.active_run_id:
            self.run_manager.stop_run(self.active_run_id)

    def _append_log(self, run_id: str, stream_name: str, text: str) -> None:
        if run_id != self.active_run_id:
            return
        if self._is_noisy_progress_line(text):
            return
        display_text = text
        if stream_name == "stderr" and self._looks_like_error(text):
            display_text = "[ERROR] " + text
        elif stream_name == "stderr":
            display_text = "[STDERR] " + text

        self.active_log_lines.append(display_text)
        self.log_panel.append_log_line(display_text, stream=stream_name)
        if self.auto_scroll_checkbox.isChecked():
            cursor = self.log_output.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.End)
            self.log_output.setTextCursor(cursor)
        self._update_log_buttons()
        self._update_status_action_buttons()

    def _on_run_started(self, record: RunRecord) -> None:
        self.active_record = record
        self.active_run_id = record.run_id
        self._update_run_controls(True)
        self._apply_status_record(record)
        self._show_run_output_detail(record, title="当前任务输出")
        self._update_status_action_buttons()

    def _on_run_finished(self, record: RunRecord) -> None:
        if record.run_id == self.active_run_id:
            self.active_record = record
            self._apply_status_record(record)
            self._show_run_output_detail(record, title="最近任务输出")
            self._update_run_controls(False)
            self._update_status_action_buttons()
        self._load_history()

    def _show_history_detail(self) -> None:
        payload = self._current_history_payload()
        if not payload:
            return
        output_dir = payload.get("output_dir")
        if not output_dir:
            self.history_detail.clear()
            return
        meta = self.history_manager.load_run_meta(output_dir)
        if meta:
            feature_name = str(meta.get("feature_name") or payload.get("feature_name") or "")
            action_name = str(meta.get("action_name") or payload.get("action_name") or "")
            artifacts_dir = str(meta.get("artifacts_dir") or output_dir)
            meta["discovered_artifacts"] = self._discover_yolo_artifacts(
                artifacts_dir, action_name, feature_name,
            )
            self.history_detail.setPlainText(self._format_detail_payload(meta, title="历史记录详情"))
            return
        config = self.history_manager.load_run_config(output_dir)
        self.history_detail.setPlainText(self._format_detail_payload(config, title="历史记录详情"))

    def _apply_history_to_form_from_item(self, item: QTableWidgetItem) -> None:
        payload = self.history_table.item(item.row(), 0).data(Qt.ItemDataRole.UserRole)
        if isinstance(payload, dict):
            self._apply_history_payload(payload)

    def _apply_history_payload(self, payload: dict) -> bool:
        output_dir = str(payload.get("output_dir", "") or "").strip()
        if not output_dir:
            return False

        config = self.history_manager.load_run_config(output_dir)
        params = config.get("params") if isinstance(config, dict) else None
        if not isinstance(params, dict):
            QMessageBox.warning(self, "无法回填参数", "这条训练历史没有可读取的参数快照。")
            return False

        feature_name = str(config.get("feature_name") or payload.get("feature_name") or "")
        action_name = str(config.get("action_name") or payload.get("action_name") or "")
        if not self._select_feature_action(feature_name, action_name):
            QMessageBox.warning(self, "无法回填参数", "当前工作台没有找到这条历史对应的功能或子功能。")
            return False

        project_name = str(config.get("project_name") or payload.get("project_name") or "").strip()
        if project_name:
            self.project_name_edit.setText(project_name)

        conda_env_name = str(config.get("conda_env_name") or "").strip()
        if conda_env_name:
            self._select_conda_env_name(conda_env_name)

        if self.current_action is not None:
            self.form_builder.set_values(self.current_action.schema, params)
            self.generated_command_text = ""
            self.summary_label.setText(f"已从训练历史回填参数: {feature_name} / {self.current_action.display_name}")
            return True
        return False

    def _select_feature_action(self, feature_name: str, action_name: str) -> bool:
        feature_index = next(
            (index for index, feature in enumerate(self.features) if feature.feature_name == feature_name),
            -1,
        )
        if feature_index < 0:
            return False

        if self.feature_list.currentRow() != feature_index:
            self.feature_list.setCurrentRow(feature_index)
        feature = self.features[feature_index]
        action_index = next(
            (index for index, action in enumerate(feature.actions) if action.action_name == action_name),
            -1,
        )
        if action_index < 0:
            return False

        if self.action_combo.currentIndex() != action_index:
            self.action_combo.setCurrentIndex(action_index)
        else:
            action = self.action_combo.itemData(action_index)
            if isinstance(action, FeatureAction):
                self._apply_action(action)
        return True

    def _select_conda_env_name(self, env_name: str) -> None:
        for index in range(self.conda_env_combo.count()):
            env = self.conda_env_combo.itemData(index)
            if isinstance(env, CondaEnvInfo) and env.name == env_name:
                self.conda_env_combo.setCurrentIndex(index)
                return

    def _collect_params(self, require_complete: bool) -> dict | None:
        if self.current_action is None:
            return None
        params = self.form_builder.values(self.current_action.schema)
        if not require_complete:
            return params

        self.form_builder.validate_visible()

        missing = [
            field.label or field.name
            for field in self.current_action.schema
            if field.required and not field.hidden and params.get(field.name) in (None, "")
        ]
        if missing:
            QMessageBox.warning(self, "参数不完整", f"以下必填参数未填写：\n{', '.join(missing)}")
            return None
        return params

    def _selected_command_platform(self) -> str:
        value = self.command_platform_combo.currentData()
        return str(value or "linux")

    def _open_settings(self) -> None:
        dialog = SettingsDialog(self.settings, self)
        if self._workspace_router is not None:
            self._workspace_router.open_dialog(
                dialog,
                title="训练设置",
                source_key=self._workspace_key,
                subtitle="默认项目、命令格式与选择记忆",
                on_finished=lambda result: self._finish_settings_dialog(dialog, result),
            )
            return

        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._apply_settings_dialog(dialog)

    def _finish_settings_dialog(self, dialog: SettingsDialog, result: int) -> None:
        if result == int(QDialog.DialogCode.Accepted):
            self._apply_settings_dialog(dialog)

    def _apply_settings_dialog(self, dialog: SettingsDialog) -> None:
        updated = dialog.get_settings()
        updated.last_feature_name = self.settings.last_feature_name
        updated.last_action_name = self.settings.last_action_name
        updated.last_conda_env_name = self.settings.last_conda_env_name
        self.settings = updated
        self.settings_manager.save(self.settings)

        platform_index = self.command_platform_combo.findData(self.settings.default_command_platform or "linux")
        self.command_platform_combo.setCurrentIndex(platform_index if platform_index >= 0 else 0)
        if self.current_feature is not None:
            self.project_name_edit.setText(self.settings.default_project_name or self.current_feature.default_project_name)

    def _reset_form_defaults(self) -> None:
        if self.current_action is None:
            return
        defaults = {field.name: field.default for field in self.current_action.schema}
        self.form_builder.set_values(self.current_action.schema, defaults)

    def _save_settings_state(self) -> None:
        self.settings.default_command_platform = self._selected_command_platform()
        if self.settings.remember_last_selection:
            self.settings.last_feature_name = self.current_feature.feature_name if self.current_feature else ""
            self.settings.last_action_name = self.current_action.action_name if self.current_action else ""
            env = self.conda_env_combo.currentData()
            self.settings.last_conda_env_name = env.name if isinstance(env, CondaEnvInfo) else ""
        else:
            self.settings.last_feature_name = ""
            self.settings.last_action_name = ""
            self.settings.last_conda_env_name = ""
        self.settings_manager.save(self.settings)

    @staticmethod
    def _hide_auto_output_fields(action: FeatureAction) -> None:
        output_names = {action.output.arg_name} if action.output.arg_name else set()
        output_names.update(extra.arg_name for extra in action.output.extra_outputs)
        for field in action.schema:
            if field.name in output_names:
                field.hidden = True
                field.required = False

    @staticmethod
    def _is_noisy_progress_line(text: str) -> bool:
        return bool(
            re.search(r"Epoch\s+\d+/\d+\s+\[(Train|Val)\]\s+step\s+\d+/\d+", text)
            or re.search(r"类别统计\s+step\s+\d+/\d+", text)
        )

    @staticmethod
    def _looks_like_error(text: str) -> bool:
        lowered = text.lower()
        error_markers = [
            "traceback",
            "error:",
            "exception",
            "runtimeerror",
            "valueerror",
            "filenotfounderror",
            "modulenotfounderror",
            "cuda out of memory",
            "oom",
            "failed",
            "permission denied",
            "no such file",
        ]
        return any(marker in lowered for marker in error_markers)

    def _show_history_menu(self, position) -> None:
        item = self.history_table.itemAt(position)
        if item is None:
            return
        self.history_table.selectRow(item.row())
        payload = self._current_history_payload()
        if not payload:
            return

        menu = QMenu(self)
        move_action = menu.addAction("移到垃圾桶")
        chosen = menu.exec(self.history_table.viewport().mapToGlobal(position))
        if chosen == move_action:
            self._move_history_to_trash(payload)

    def _current_history_payload(self) -> dict | None:
        row = self.history_table.currentRow()
        if row < 0:
            return None
        item = self.history_table.item(row, 0)
        if item is None:
            return None
        payload = item.data(Qt.ItemDataRole.UserRole)
        return payload if isinstance(payload, dict) else None

    def _move_history_to_trash(self, payload: dict) -> None:
        output_dir = str(payload.get("output_dir", "")).strip()
        run_id = str(payload.get("run_id", "")).strip()
        if not output_dir:
            return
        if run_id and run_id == self.active_run_id:
            QMessageBox.warning(self, "无法整理", "当前正在运行的任务不能移动到垃圾桶。")
            return

        reply = QMessageBox.question(
            self,
            "移到垃圾桶",
            "根据当前安全规则，这里不会真正删除训练历史，只会把记录目录移动到项目 .trash。是否继续？",
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        try:
            destination = self.history_manager.move_run_to_trash(output_dir)
        except Exception as exc:
            QMessageBox.critical(self, "操作失败", str(exc))
            return

        self.history_detail.clear()
        self._load_history()
        QMessageBox.information(self, "已移动", f"训练记录已移动到：\n{destination}")

    def _show_run_output_detail(self, record: RunRecord, title: str) -> None:
        payload = {
            "title": title,
            "run_id": record.run_id,
            "project_name": record.project_name,
            "feature_name": record.feature_name,
            "action_name": record.action_name,
            "action_display_name": record.action_display_name,
            "status": record.status,
            "output_dir": record.output_dir,
            "artifacts_dir": record.artifacts_dir,
            "stdout_log": record.stdout_log,
            "stderr_log": record.stderr_log,
            "config_path": record.config_path,
            "meta_path": record.meta_path,
            "start_time": record.start_time,
            "end_time": record.end_time,
            "duration_seconds": record.duration_seconds,
            "exit_code": record.exit_code,
            "conda_env_name": record.conda_env_name,
            "python_executable": record.python_executable,
            "command": record.command,
        }
        payload["discovered_artifacts"] = self._discover_yolo_artifacts(
            record.artifacts_dir, record.action_name, record.feature_name,
        )
        self.history_detail.setPlainText(self._format_detail_payload(payload, title=title))

    @staticmethod
    def _discover_yolo_artifacts(artifacts_dir: str, action_name: str, feature_name: str) -> list[str]:
        """Scan the run output directory for YOLO result files and return a summary list."""
        if feature_name != "yolo":
            return []

        base = Path(artifacts_dir)
        if not base.exists():
            return []

        lines: list[str] = []
        max_image_preview = 20

        if action_name == "train":
            train_artifacts = [
                "results.png", "results.csv",
                "confusion_matrix.png",
                "PR_curve.png", "F1_curve.png", "P_curve.png", "R_curve.png",
                "weights/best.pt", "weights/last.pt",
            ]
            for name in train_artifacts:
                p = base / name
                if p.exists():
                    size_kb = p.stat().st_size / 1024
                    lines.append(f"  {name}  ({size_kb:.1f} KB)")

        elif action_name == "predict":
            # Ultralytics may create a subdirectory named after output_dir.name inside the predict dir
            # Scan for images in the artifacts dir and its immediate subdirectories
            image_exts = {".jpg", ".jpeg", ".png", ".bmp"}
            images: list[Path] = []
            for ext in image_exts:
                images.extend(base.rglob(f"*{ext}"))
            # Also scan one level deeper (Ultralytics often saves to predict/ subfolder)
            for child in base.iterdir():
                if child.is_dir():
                    for ext in image_exts:
                        images.extend(child.rglob(f"*{ext}"))

            # Deduplicate
            seen = set()
            unique_images = []
            for img in images:
                if img not in seen:
                    seen.add(img)
                    unique_images.append(img)

            count = len(unique_images)
            shown = unique_images[:max_image_preview]
            for img in sorted(shown):
                rel = img.relative_to(base)
                lines.append(f"  {rel}  ({img.stat().st_size / 1024:.1f} KB)")
            if count > max_image_preview:
                lines.append(f"  ... 共 {count} 张图片，仅显示前 {max_image_preview} 张")

            # Labels
            label_files = list(base.rglob("labels/*.txt"))
            if label_files:
                lines.append(f"  labels/  ({len(label_files)} 个标签文件)")

            # Crops
            crops_dir = base / "crops"
            if crops_dir.exists() and crops_dir.is_dir():
                crop_count = sum(1 for _ in crops_dir.rglob("*") if _.is_file())
                lines.append(f"  crops/  ({crop_count} 个裁剪文件)")

        return lines

    @staticmethod
    def _format_detail_payload(payload: dict, title: str) -> str:
        def line(label: str, value) -> str:
            text = "" if value in (None, "", []) else str(value)
            return f"{label}: {text}"

        sections: list[str] = [title]
        sections.append(
            "\n".join(
                item
                for item in [
                    line("项目名", payload.get("project_name")),
                    line("功能", payload.get("feature_name")),
                    line("子功能", payload.get("action_display_name") or payload.get("action_name")),
                    line("状态", payload.get("status")),
                    line("运行 ID", payload.get("run_id")),
                ]
                if not item.endswith(": ")
            )
        )
        time_lines = [
            line("开始时间", payload.get("start_time")),
            line("结束时间", payload.get("end_time")),
            line("耗时(秒)", payload.get("duration_seconds")),
            line("退出码", payload.get("exit_code")),
        ]
        if any(not item.endswith(": ") for item in time_lines):
            sections.append("\n".join(item for item in time_lines if not item.endswith(": ")))

        path_lines = [
            line("输出目录", payload.get("output_dir")),
            line("结果目录", payload.get("artifacts_dir")),
            line("标准输出日志", payload.get("stdout_log")),
            line("错误日志", payload.get("stderr_log")),
            line("配置文件", payload.get("config_path")),
            line("元信息文件", payload.get("meta_path")),
        ]
        sections.append("\n".join(item for item in path_lines if not item.endswith(": ")))

        metric_keys = ["tp", "fp", "fn", "gt_edges", "pred_edges", "precision", "recall", "f1"]
        metric_lines = [line(key.upper() if len(key) <= 2 else key, payload.get(key)) for key in metric_keys]
        metric_lines = [item for item in metric_lines if not item.endswith(": ")]
        if metric_lines:
            sections.append("\n".join(metric_lines))

        env_lines = [
            line("Conda 环境", payload.get("conda_env_name")),
            line("Python", payload.get("python_executable")),
        ]
        env_lines = [item for item in env_lines if not item.endswith(": ")]
        if env_lines:
            sections.append("\n".join(env_lines))

        command = payload.get("command")
        if isinstance(command, list) and command:
            sections.append("命令:\n" + " ".join(str(part) for part in command))

        artifacts = payload.get("discovered_artifacts", [])
        if artifacts:
            sections.append("发现文件:\n" + "\n".join(artifacts))

        return "\n\n".join(section for section in sections if section.strip())

    def _build_status_badge(self, status: str) -> QLabel:
        text, fg, bg = self._status_style(status)
        label = QLabel(text)
        label.setObjectName("statusBadge")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet(status_badge_stylesheet(status))
        return label

    def _build_history_actions(self, payload: dict) -> QWidget:
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        for text, tooltip, handler in [
            ("日志", "查看日志", lambda p=payload: self._view_history_log(p)),
            ("目录", "打开目录", lambda p=payload: self._open_history_dir(p)),
            ("复制", "复制路径", lambda p=payload: self._copy_history_path(p)),
            ("重跑", "重新运行", lambda p=payload: self._rerun_history(p)),
        ]:
            button = QPushButton(text)
            button.setToolTip(tooltip)
            button.setProperty("buttonRole", "ghost")
            button.setProperty("compact", "true")
            button.clicked.connect(handler)
            layout.addWidget(button)
        return widget

    def _view_history_log(self, payload: dict) -> None:
        output_dir = str(payload.get("output_dir", "") or "").strip()
        if not output_dir:
            return
        meta = self.history_manager.load_run_meta(output_dir)
        stdout_path = Path(meta.get("stdout_log", Path(output_dir) / "stdout.log"))
        stderr_path = Path(meta.get("stderr_log", Path(output_dir) / "stderr.log"))
        content_parts = []
        for title, path in [("STDOUT", stdout_path), ("STDERR", stderr_path)]:
            if path.exists():
                content_parts.append(f"[{title}] {path}\n{path.read_text(encoding='utf-8', errors='replace')}")
        if not content_parts:
            QMessageBox.information(self, "暂无日志", "没有找到这条历史记录的日志文件。")
            return
        self._show_text_dialog("历史任务日志", "\n\n".join(content_parts))

    def _open_history_dir(self, payload: dict) -> None:
        output_dir = str(payload.get("output_dir", "") or "").strip()
        if output_dir:
            QDesktopServices.openUrl(QUrl.fromLocalFile(output_dir))

    def _copy_history_path(self, payload: dict) -> None:
        output_dir = str(payload.get("output_dir", "") or "").strip()
        if output_dir:
            QApplication.clipboard().setText(output_dir)

    def _rerun_history(self, payload: dict) -> None:
        if self.active_run_id and self.run_manager.active_runs.get(self.active_run_id):
            QMessageBox.warning(self, "任务运行中", "请先等待当前任务结束，或先停止当前任务。")
            return
        if self._apply_history_payload(payload):
            self._start_run()

    def _update_run_controls(self, is_running: bool) -> None:
        self.start_button.setEnabled(not is_running)
        self.stop_button.setEnabled(is_running)

    def _refresh_idle_status_card(self) -> None:
        if self.active_record is None:
            self._apply_status_record(None)

    def _apply_status_record(self, record: RunRecord | None) -> None:
        if record is None:
            self.status_badge.setText("未开始")
            self.status_badge.setStyleSheet(status_badge_stylesheet("idle"))
            self._set_label_value(self.status_run_id, "-")
            self._set_label_value(self.status_project, self.project_name_edit.text().strip() or self.settings.default_project_name or "default")
            self._set_label_value(self.status_feature, self.current_feature.feature_name if self.current_feature else "-")
            self._set_label_value(self.status_action, self.current_action.display_name if self.current_action else "-")
            for label in [self.status_start, self.status_end, self.status_duration, self.status_exit,
                          self.status_output, self.status_artifacts, self.status_cwd,
                          self.status_python, self.status_conda, self.status_repo_root, self.status_command]:
                self._set_label_value(label, "-")
            return

        text, fg, bg = self._status_style(record.status)
        self.status_badge.setText(text)
        self.status_badge.setStyleSheet(status_badge_stylesheet(record.status))
        # 主要信息
        self._set_label_value(self.status_run_id, record.run_id)
        self._set_label_value(self.status_action, record.action_display_name)
        self._set_label_value(self.status_start, self._format_time(record.start_time))
        self._set_label_value(self.status_duration, self._format_duration(record.duration_seconds))
        self._set_label_value(self.status_exit, "" if record.exit_code is None else str(record.exit_code))
        self._set_label_value(self.status_output, record.output_dir)
        # 次要信息（详情折叠区）
        self._set_label_value(self.status_project, record.project_name)
        self._set_label_value(self.status_feature, record.feature_name)
        self._set_label_value(self.status_end, self._format_time(record.end_time))
        self._set_label_value(self.status_artifacts, record.artifacts_dir)
        self._set_label_value(self.status_cwd, record.cwd)
        self._set_label_value(self.status_python, record.python_executable)
        conda_info = record.conda_env_name or "-"
        if record.conda_prefix:
            conda_info = f"{record.conda_env_name} ({record.conda_prefix})"
        self._set_label_value(self.status_conda, conda_info)
        self._set_label_value(self.status_repo_root, record.repo_root)
        self._set_label_value(self.status_command, " ".join(record.command))

    @staticmethod
    def _set_label_value(label: QLabel, value: str) -> None:
        text = value or "-"
        label.setText(text)
        label.setToolTip(text if text != "-" else "")

    @classmethod
    def _status_style(cls, status: str) -> tuple[str, str, str]:
        return cls.STATUS_META.get(status, cls.STATUS_META["warning"])

    @staticmethod
    def _format_time(value: str | None) -> str:
        if not value:
            return "-"
        try:
            dt = datetime.fromisoformat(value)
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except (ValueError, TypeError):
            return str(value)

    @staticmethod
    def _format_duration(value) -> str:
        if value in (None, ""):
            return "-"
        try:
            seconds = float(value)
        except (ValueError, TypeError):
            return str(value)
        if seconds < 60:
            return f"{seconds:.1f}s"
        minutes, rem = divmod(int(seconds), 60)
        hours, minutes = divmod(minutes, 60)
        if hours:
            return f"{hours}h {minutes}m"
        return f"{minutes}m {rem}s"

    def _highlight_log_text(self, text: str) -> str:
        escaped = html.escape(text)
        patterns = [
            (r"\b(success|valid|saved)\b", "#22c55e"),
            (r"\b(failed|error|exception|traceback)\b", "#f87171"),
            (r"\b(warning|tracerwarning)\b", "#fbbf24"),
            (r"\b(command|export|train)\b", "#93c5fd"),
        ]
        for pattern, color in patterns:
            escaped = re.sub(
                pattern,
                lambda m: f"<span style='color:{color}; font-weight:700;'>{m.group(0)}</span>",
                escaped,
                flags=re.IGNORECASE,
            )
        return f"<span style='white-space:pre; font-family:Consolas, Courier New, monospace;'>{escaped}</span>"

    def _copy_log(self) -> None:
        if not self.active_log_lines:
            return
        QApplication.clipboard().setText("\n".join(self.active_log_lines))

    def _clear_log(self) -> None:
        self.active_log_lines.clear()
        self.log_panel.clear_log()
        self._update_log_buttons()
        self._update_status_action_buttons()

    def _export_log(self) -> None:
        if not self.active_log_lines:
            QMessageBox.information(self, "暂无日志", "当前没有可导出的日志内容。")
            return
        default_name = f"{self.active_run_id or 'trainer'}_log.txt"
        path, _ = QFileDialog.getSaveFileName(self, "导出日志", str(self.project_root / default_name), "Text (*.txt)")
        if not path:
            return
        Path(path).write_text("\n".join(self.active_log_lines), encoding="utf-8")
        QMessageBox.information(self, "导出完成", f"日志已导出到：\n{path}")

    def _show_full_log_dialog(self) -> None:
        if not self.active_record:
            QMessageBox.information(self, "暂无日志", "当前还没有运行日志。")
            return
        content = []
        for title, path in [("STDOUT", self.active_record.stdout_log), ("STDERR", self.active_record.stderr_log)]:
            file_path = Path(path)
            if file_path.exists():
                content.append(f"[{title}] {path}\n{file_path.read_text(encoding='utf-8', errors='replace')}")
        if not content:
            content.append("\n".join(self.active_log_lines))
        self._show_text_dialog("完整日志", "\n\n".join(content))

    def _show_text_dialog(self, title: str, content: str) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle(title)
        dialog.resize(980, 620)
        layout = QVBoxLayout(dialog)
        text_edit = QPlainTextEdit(dialog)
        text_edit.setReadOnly(True)
        text_edit.setPlainText(content)
        layout.addWidget(text_edit)
        button_row = QHBoxLayout()
        copy_button = QPushButton("复制内容")
        copy_button.setProperty("buttonRole", "secondary")
        copy_button.clicked.connect(lambda: QApplication.clipboard().setText(text_edit.toPlainText()))
        close_button = QPushButton("关闭")
        close_button.setProperty("buttonRole", "secondary")
        close_button.clicked.connect(dialog.accept)
        button_row.addWidget(copy_button)
        button_row.addStretch(1)
        button_row.addWidget(close_button)
        layout.addLayout(button_row)
        if self._workspace_router is not None:
            self._workspace_router.open_dialog(
                dialog,
                title=title,
                source_key=self._workspace_key,
                subtitle="训练日志与文本详情",
            )
            return
        dialog.exec()

    def _open_active_output_dir(self) -> None:
        if self.active_record and self.active_record.output_dir:
            QDesktopServices.openUrl(QUrl.fromLocalFile(self.active_record.output_dir))

    def _copy_active_output_dir(self) -> None:
        if self.active_record and self.active_record.output_dir:
            QApplication.clipboard().setText(self.active_record.output_dir)

    def _copy_active_command(self) -> None:
        if self.active_record and self.active_record.command:
            QApplication.clipboard().setText(" ".join(self.active_record.command))

    def _update_log_buttons(self) -> None:
        has_logs = bool(self.active_log_lines)
        self.copy_log_button.setEnabled(has_logs)
        self.clear_log_button.setEnabled(has_logs)
        self.export_log_button.setEnabled(has_logs)

    def _update_status_action_buttons(self) -> None:
        has_record = self.active_record is not None
        has_output = bool(has_record and self.active_record.output_dir)
        has_command = bool(has_record and self.active_record.command)
        has_logs = bool(
            self.active_log_lines
            or (
                has_record
                and (
                    Path(self.active_record.stdout_log).exists()
                    or Path(self.active_record.stderr_log).exists()
                )
            )
        )
        self.open_output_button.setEnabled(has_output)
        self.copy_output_button.setEnabled(has_output)
        self.copy_command_button.setEnabled(has_command)
        self.view_full_log_button.setEnabled(has_logs)

    def shutdown(self) -> None:
        """Release training subprocesses when the embedding shell closes."""
        self.run_manager.shutdown()

    def _elide_middle(self, text: str, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        head = max_chars // 2 - 2
        tail = max_chars - head - 3
        return f"{text[:head]}...{text[-tail:]}"
