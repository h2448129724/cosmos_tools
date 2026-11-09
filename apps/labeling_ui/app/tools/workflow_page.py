"""缝纫点与连边标注数据处理 — 四步流程工具页面。"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

from PySide6.QtCore import QSettings, Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QScrollArea,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from apps.cabf_flow.config_model import (
    FIELDS,
    get_nested,
    load_config,
    save_config,
    set_nested,
)
from apps.cabf_flow.flow import CONFIG_PATH
from shared.project_paths import repo_root

from cosmos_toolbox.ui import (
    ActionBar,
    PageHeader,
    SectionSurface,
    StatusBanner,
    set_tone,
    set_ui_role,
)

from .base import BaseToolPage, make_log_card, make_log_box, set_primary


class StepInfo:
    __slots__ = ("key", "title", "desc")

    def __init__(self, key: str, title: str, desc: str):
        self.key = key
        self.title = title
        self.desc = desc


STEPS = [
    StepInfo("config", "数据集准备", "初始化或检查数据集目录结构"),
    StepInfo("filter_data", "数据筛选", "先筛掉不符合要求的裁剪图，再开始标注"),
    StepInfo("annotate", "点边标注", "标注或修正缝纫点，以及点之间的连边"),
    StepInfo("export", "校验与导出", "校验母标注，导出点数据集与连边数据集"),
]

PENDING, ACTIVE, COMPLETED, ERROR = range(4)
_MANUAL_KEYS = {"filter_data", "annotate"}
_STATUS_TEXT = {PENDING: "待处理", ACTIVE: "进行中", COMPLETED: "已完成", ERROR: "异常"}

class _StepIndicator(QWidget):
    """Compact clickable step list with explicit text labels."""

    stepClicked = Signal(int)

    def __init__(self, steps: list[StepInfo], parent=None):
        super().__init__(parent)
        self._steps = steps
        self._statuses: list[int] = [ACTIVE] + [PENDING] * (len(steps) - 1)
        self._current = 0
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.setAccessibleName("CAB-F 流程步骤")
        self._buttons: list[QPushButton] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        for index, step in enumerate(steps):
            btn = QPushButton()
            set_ui_role(btn, "stepButton")
            btn.setAccessibleName(step.title)
            btn.setAccessibleDescription(step.desc)
            btn.setCursor(Qt.PointingHandCursor)
            btn.setFlat(True)
            btn.setMinimumHeight(34)
            btn.clicked.connect(lambda checked=False, idx=index: self.stepClicked.emit(idx))
            self._buttons.append(btn)
            layout.addWidget(btn)
        layout.addStretch(1)
        self._apply_styles()

    def set_statuses(self, statuses: list[int], current: int):
        self._statuses = statuses
        self._current = current
        self._apply_styles()

    def _apply_styles(self) -> None:
        for index, btn in enumerate(self._buttons):
            step = self._steps[index]
            state = self._statuses[index] if index < len(self._statuses) else PENDING
            is_current = index == self._current
            symbol = "✓" if state == COMPLETED else str(index + 1)
            btn.setText(f"{symbol}  {step.title}")
            btn.setProperty("stepState", _STATUS_TEXT.get(state, _STATUS_TEXT[PENDING]))
            btn.setProperty("current", is_current)
            btn.setProperty("buttonRole", "primary" if is_current else "ghost")
            btn.style().unpolish(btn)
            btn.style().polish(btn)
            btn.setToolTip(step.desc)


def _make_output_box() -> QPlainTextEdit:
    return make_log_box("输出日志...", height=96)


class StitchWorkflowPage(BaseToolPage):
    tool_key = "workflow_page"
    tool_title = "缝纫点与连边标注数据处理"
    tool_nav_title = "缝纫点与连边"
    tool_icon = "◉"
    tool_summary = "数据集准备、筛选、点边标注、校验与导出四个步骤。"
    tool_tags = ("全流程", "点边标注", "数据集导出")

    def __init__(self, main_window, parent=None):
        super().__init__(main_window, parent)
        self.config_path = main_window.config_path
        self.config_data = main_window.config_data
        self.step_status: list[int] = [ACTIVE] + [PENDING] * (len(STEPS) - 1)
        self.current_step = 0
        self._build_ui()
        self._go_to_step(0)

    def shutdown(self) -> None:
        """Let an in-flight validation/export finish before teardown."""
        panel = getattr(self, "_export_panel", None)
        if panel is not None:
            panel.shutdown()

    def _build_ui(self):
        self.setProperty("ownsPageHeader", True)
        self.setAccessibleName(self.tool_title)
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(10)

        sidebar = self._build_step_sidebar()
        sidebar.setMinimumWidth(150)
        sidebar.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        root.addWidget(sidebar)

        right = QVBoxLayout()
        right.setContentsMargins(0, 0, 0, 0)
        right.setSpacing(10)
        right.addWidget(self._build_page_header())
        self._workflow_status = StatusBanner("就绪", "neutral")
        self._workflow_status.setAccessibleName("流程状态")
        right.addWidget(self._workflow_status)

        self.step_stack = QStackedWidget()
        for builder in (
            self._build_step_config,
            self._build_step_filter_data,
            self._build_step_annotate,
            self._build_step_export,
        ):
            self.step_stack.addWidget(builder())
        self._step_scroll = QScrollArea()
        self._step_scroll.setObjectName("workflowStepScroll")
        self._step_scroll.setWidgetResizable(True)
        self._step_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._step_scroll.setFrameShape(QFrame.NoFrame)
        self._step_scroll.setWidget(self.step_stack)
        right.addWidget(self._step_scroll, 1)
        right.addWidget(self._build_nav())
        root.addLayout(right, 1)

    def _build_step_sidebar(self) -> QWidget:
        frame = SectionSurface("流程步骤", "按顺序完成准备、筛选、标注与导出。")
        frame.setObjectName("stepSidebar")
        self._step_indicator = _StepIndicator(STEPS)
        self._step_indicator.stepClicked.connect(self._go_to_step)
        frame.body_layout.addWidget(self._step_indicator)
        frame.body_layout.addStretch(1)
        return frame

    def _build_page_header(self) -> QWidget:
        header = PageHeader(self.tool_title, self.tool_summary, parent=self)
        self._page_header = header
        self._page_title = header.title_label
        self._page_desc = header.description_label
        self._page_status = header.status_label
        self._page_title.setAccessibleName("当前步骤标题")
        self._page_desc.setAccessibleName("当前步骤说明")
        return header

    def _build_step_body_card(self) -> tuple[QWidget, QVBoxLayout]:
        card = SectionSurface(parent=self)
        card.setObjectName("stepContentCard")
        card.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        card.body_layout.setAlignment(Qt.AlignTop)
        return card, card.body_layout

    def _build_log_card(self, box: QPlainTextEdit) -> QWidget:
        card = make_log_card(box)
        card.setObjectName("logCard")
        card.setAccessibleName("运行日志面板")
        return card

    def _build_nav(self) -> QWidget:
        bar = ActionBar()
        bar.setAccessibleName("流程导航操作")
        self._btn_prev = QPushButton("< 上一步")
        self._btn_prev.setAccessibleName("上一步")
        self._btn_prev.clicked.connect(self._prev_step)
        self._btn_done = QPushButton("标记完成")
        self._btn_done.setAccessibleName("标记当前步骤完成")
        self._btn_done.clicked.connect(self._mark_done)
        self._btn_next = QPushButton("下一步 >")
        self._btn_next.setAccessibleName("下一步")
        set_primary(self._btn_next)
        self._btn_next.clicked.connect(self._next_step)
        bar.add_widget(self._btn_prev)
        bar.add_widget(self._btn_done)
        bar.add_widget(self._btn_next)
        return bar

    def _refresh_step_sidebar(self):
        self._step_indicator.set_statuses(self.step_status, self.current_step)

    def _refresh_step_summary(self):
        step = STEPS[self.current_step]
        self._page_title.setText(step.title if step.key != "config" else "数据集初始化")
        self._page_desc.setText(
            "选择裁剪图目录和数据集根目录，程序会自动创建子目录并保存配置。"
            if step.key == "config" else step.desc
        )
        counts = self._path_counts() if step.key == "config" else None
        if counts is not None:
            valid, missing, total = counts
            status_text = f"当前状态：还差 {missing} 项" if missing else f"当前状态：{valid}/{total} 已就绪"
            tone = "warning" if missing else "success"
        else:
            text, level = self.status_badge()
            status_text = f"当前状态：{text}"
            tone = level
        self._page_header.set_status(status_text, tone)
        self._workflow_status.set_status(status_text, tone)

    def _go_to_step(self, idx: int):
        if not 0 <= idx < len(STEPS):
            return
        self._sync_form_to_memory()
        self.current_step = idx
        self.step_stack.setCurrentIndex(idx)
        self._refresh_step_sidebar()
        self._refresh_step_summary()
        self._sync_step_fields(idx)
        self._btn_prev.setEnabled(idx > 0)
        self._btn_next.setEnabled(idx < len(STEPS) - 1)
        self._btn_done.setVisible(STEPS[idx].key in _MANUAL_KEYS)
        if hasattr(self._mw, "refresh_tool_overview"):
            self._mw.refresh_tool_overview()

    def _sync_step_fields(self, idx: int):
        """Auto-fill step form entries from config when entering a step."""
        if STEPS[idx].key == "export" and hasattr(self, "_export_panel"):
            panel = self._export_panel
            panel.edit_image_dir.setText(self._cfg("master_images_dir"))
            panel.edit_annotation_dir.setText(self._cfg("master_annotations_dir"))
            panel.edit_model_a_output_dir.setText(self._cfg("model_a_export_root"))
            panel.edit_model_b_output_dir.setText(self._cfg("model_b_export_root"))

    def _next_step(self):
        self._go_to_step(self.current_step + 1)

    def _prev_step(self):
        self._go_to_step(self.current_step - 1)

    def _mark_done(self):
        self.step_status[self.current_step] = COMPLETED
        self._refresh_step_sidebar()
        if hasattr(self._mw, "refresh_tool_overview"):
            self._mw.refresh_tool_overview()
        if self.current_step < len(STEPS) - 1:
            self._next_step()

    def _cfg(self, key: str) -> str:
        return get_nested(self.config_data, key)

    def _set_cfg_text(self, key: str, value: str) -> None:
        entry = self._cfg_entries.get(key) if hasattr(self, "_cfg_entries") else None
        if entry is not None and entry.text() != value:
            entry.setText(value)
        set_nested(self.config_data, key, value)

    def current_step_summary(self) -> str:
        return f"{self.current_step + 1}/{len(STEPS)}"

    def _path_counts(self) -> tuple[int, int, int] | None:
        if not hasattr(self, "_dataset_root_entry"):
            return None
        dataset_root_text = self._dataset_root_entry.text().strip()
        mode = self._dataset_mode.currentData() if hasattr(self, "_dataset_mode") else "init"
        if mode == "init":
            if not hasattr(self, "_dataset_source_entry"):
                return None
            values = [
                self._dataset_source_entry.text().strip(),
                dataset_root_text,
            ]
            valid = sum(1 for val in values if val and Path(val).is_dir())
            missing = len(values) - valid
            return valid, missing, len(values)

        if not dataset_root_text:
            return 0, 9, 9
        paths = self._expected_dataset_paths(Path(dataset_root_text))
        values = [dataset_root_text, *[str(path) for path in paths.values()]]
        valid = sum(1 for val in values if val and Path(val).is_dir())
        missing = len(values) - valid
        return valid, missing, len(values)

    def path_check_summary(self) -> str:
        counts = self._path_counts()
        if counts is None:
            return "--"
        valid, _missing, total = counts
        return f"{valid}/{total}"

    def missing_path_summary(self) -> str:
        counts = self._path_counts()
        if counts is None:
            return "--"
        _valid, missing, _total = counts
        return str(missing)

    def status_badge(self) -> tuple[str, str]:
        state = self.step_status[self.current_step]
        if state == COMPLETED:
            return _STATUS_TEXT[state], "success"
        if state == ERROR:
            return _STATUS_TEXT[state], "danger"
        if state == ACTIVE:
            return _STATUS_TEXT[state], "info"
        return _STATUS_TEXT[state], "neutral"

    def _sync_form_to_memory(self) -> None:
        """Copy visible form values into the in-memory config without I/O."""
        for key, _label, _kind in FIELDS:
            entry = self._cfg_entries.get(key)
            if entry is not None:
                set_nested(self.config_data, key, entry.text().strip())

    def _sync_form(self) -> None:
        """Persist the current form after an explicit save or commit action."""
        self._sync_form_to_memory()
        save_config(self.config_path, self.config_data)

    def _expected_dataset_paths(self, dataset_root: Path) -> dict[str, Path]:
        return {
            "pending_filter_dir": dataset_root / "pending_filter",
            "filtered_keep_dir": dataset_root / "filtered_keep",
            "master_images_dir": dataset_root / "pending_filter",
            "master_annotations_dir": dataset_root / "master_annotations" / "annotations",
            "model_a_export_root": dataset_root / "model_a_export",
            "model_b_export_root": dataset_root / "model_b_export",
        }

    def _apply_dataset_root_paths(self, dataset_root: Path) -> dict[str, Path]:
        paths = self._expected_dataset_paths(dataset_root)
        active_image_dir = self._resolve_active_image_dir(dataset_root, paths)
        self._cfg_entries["repo_root"].setText(str(repo_root()))
        self._cfg_entries["dataset_root"].setText(str(dataset_root))
        self._cfg_entries["pending_filter_dir"].setText(str(paths["pending_filter_dir"]))
        self._cfg_entries["filtered_keep_dir"].setText(str(paths["filtered_keep_dir"]))
        self._cfg_entries["master_images_dir"].setText(str(active_image_dir))
        self._cfg_entries["master_annotations_dir"].setText(str(paths["master_annotations_dir"]))
        self._cfg_entries["model_a_export_root"].setText(str(paths["model_a_export_root"]))
        self._cfg_entries["model_b_export_root"].setText(str(paths["model_b_export_root"]))
        set_nested(self.config_data, "pending_filter_dir", str(paths["pending_filter_dir"]))
        set_nested(self.config_data, "filtered_keep_dir", str(paths["filtered_keep_dir"]))
        set_nested(self.config_data, "master_images_dir", str(active_image_dir))
        self._refresh_generated_preview()
        self._refresh_filter_step_status()
        return paths

    def _resolve_active_image_dir(self, dataset_root: Path, paths: dict[str, Path]) -> Path:
        current = Path(self._cfg("master_images_dir")) if self._cfg("master_images_dir").strip() else None
        allowed = {paths["pending_filter_dir"].resolve(), paths["filtered_keep_dir"].resolve()}
        if current is not None:
            try:
                current_resolved = current.resolve()
            except Exception:
                current_resolved = None
            if current_resolved in allowed:
                return current
        return paths["pending_filter_dir"]

    def _refresh_generated_preview(self) -> None:
        if not hasattr(self, "_preview_entries"):
            return
        dataset_root_text = self._dataset_root_entry.text().strip() if hasattr(self, "_dataset_root_entry") else ""
        values = {
            "dataset_root": dataset_root_text,
            "pending_filter_dir": "",
            "filtered_keep_dir": "",
            "master_images_dir": "",
            "master_annotations_dir": "",
            "model_a_export_root": "",
            "model_b_export_root": "",
        }
        if dataset_root_text:
            paths = self._expected_dataset_paths(Path(dataset_root_text))
            active_dir = self._resolve_active_image_dir(Path(dataset_root_text), paths)
            values.update({
                "pending_filter_dir": str(paths["pending_filter_dir"]),
                "filtered_keep_dir": str(paths["filtered_keep_dir"]),
                "master_images_dir": str(active_dir),
                "master_annotations_dir": str(paths["master_annotations_dir"]),
                "model_a_export_root": str(paths["model_a_export_root"]),
                "model_b_export_root": str(paths["model_b_export_root"]),
            })
        for key, entry in self._preview_entries.items():
            entry.setText(values.get(key, ""))
            entry.setToolTip(values.get(key, ""))

    def _move_file_unique(self, source: Path, dest_dir: Path) -> Path:
        dest_dir.mkdir(parents=True, exist_ok=True)
        target = dest_dir / source.name
        if target.exists():
            index = 1
            while True:
                candidate = dest_dir / f"{source.stem}_{index}{source.suffix}"
                if not candidate.exists():
                    target = candidate
                    break
                index += 1
        shutil.move(str(source), str(target))
        return target

    def _init_dataset_from_root(self):
        source_dir_text = self._dataset_source_entry.text().strip()
        dataset_root_text = self._dataset_root_entry.text().strip()
        if not source_dir_text or not Path(source_dir_text).is_dir():
            QMessageBox.warning(self._mw, "提示", "请先选择有效的裁剪图目录。")
            return
        if not dataset_root_text:
            QMessageBox.warning(self._mw, "提示", "请先选择数据集根目录。")
            return

        source_dir = Path(source_dir_text)
        dataset_root = Path(dataset_root_text)
        paths = self._expected_dataset_paths(dataset_root)

        for path in (
            dataset_root,
            *paths.values(),
        ):
            path.mkdir(parents=True, exist_ok=True)

        image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
        moved = 0
        for path in sorted(source_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in image_exts:
                self._move_file_unique(path, paths["pending_filter_dir"])
                moved += 1

        self._apply_dataset_root_paths(dataset_root)
        self._sync_form()
        self._check_all()
        self._cfg_box.appendPlainText(f"已初始化数据集根目录并转移 {moved} 张图片到 {paths['pending_filter_dir']}")

    def _check_existing_dataset(self):
        dataset_root_text = self._dataset_root_entry.text().strip()
        if not dataset_root_text or not Path(dataset_root_text).is_dir():
            QMessageBox.warning(self._mw, "提示", "请先选择有效的数据集根目录。")
            return
        dataset_root = Path(dataset_root_text)
        paths = self._apply_dataset_root_paths(dataset_root)
        self._sync_form()
        self._check_all()
        missing = [str(path) for path in paths.values() if not path.is_dir()]
        if missing:
            self._cfg_box.appendPlainText("已有数据集检查完成，以下目录缺失：")
            for item in missing:
                self._cfg_box.appendPlainText(f"- {item}")
        else:
            self._cfg_box.appendPlainText("已有数据集检查完成，目录结构完整。")

    def _on_dataset_mode_changed(self):
        mode = self._dataset_mode.currentData()
        is_init = mode == "init"
        self._init_title.setText("数据集初始化" if is_init else "已有数据集检查")
        self._init_desc.setText(
            "选择裁剪图目录和数据集根目录，程序会自动创建子目录并保存配置。"
            if is_init else
            "选择已有数据集根目录。程序会自动推导并检查筛选、母标注和导出等目录。"
        )
        self._dataset_source_entry.setEnabled(is_init)
        self._btn_src.setEnabled(is_init)
        self._src_label.setEnabled(is_init)
        self._btn_init.setVisible(is_init)
        self._btn_check_existing.setVisible(not is_init)
        self._check_all()

    def _choose_dir(self, entry: QLineEdit):
        settings = QSettings("CABF", "img_tools")
        last_dir = settings.value("last_dir", entry.text() or ".")
        d = QFileDialog.getExistingDirectory(self._mw, "选择目录", last_dir)
        if d:
            entry.setText(d)
            settings.setValue("last_dir", d)

    def _open_path(self, p: str):
        if not p:
            return
        resolved = Path(p).resolve()
        if not resolved.exists():
            return
        os.startfile(str(resolved))

    @staticmethod
    def _set_badge(label: QLabel, text: str, tone: str = "neutral") -> None:
        label.setText(text)
        set_ui_role(label, "statusBadge")
        set_tone(label, tone)

    def _check_all(self):
        counts = self._path_counts()
        valid, _missing, total = counts if counts is not None else (0, 0, 0)
        mode = self._dataset_mode.currentData() if hasattr(self, "_dataset_mode") else "init"
        self._cfg_box.clear()
        if mode == "init":
            self._cfg_box.appendPlainText("当前模式：初始化数据集。检查裁剪图目录和数据集根目录。")
        else:
            self._cfg_box.appendPlainText("当前模式：检查已有数据集。检查数据集根目录及自动推导的目录结构。")
        self._cfg_box.appendPlainText(f"{valid}/{total} 项有效")
        self._set_badge(
            self._config_summary_badge,
            f"{valid}/{total} 有效",
            "success" if valid == total else "warning" if valid > 0 else "neutral",
        )
        self.step_status[0] = COMPLETED if valid == total else ACTIVE
        self._refresh_step_sidebar()
        self._refresh_step_summary()
        self._refresh_generated_preview()
        if hasattr(self._mw, "refresh_tool_overview"):
            self._mw.refresh_tool_overview()

    def _build_step_config(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QScrollArea()
        scroll.setObjectName("workflowScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.viewport().setObjectName("workflowScrollViewport")
        outer.addWidget(scroll)

        content = QWidget()
        content.setObjectName("workflowScrollContent")
        scroll.setWidget(content)
        lay = QVBoxLayout(content)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(10)

        init_card, init_lay = self._build_step_body_card()
        status_row = QHBoxLayout()
        status_row.setContentsMargins(0, 0, 0, 0)
        self._config_summary_badge = QLabel("待检查")
        self._set_badge(self._config_summary_badge, "待检查")
        status_row.addWidget(self._config_summary_badge, 0, Qt.AlignLeft)
        status_row.addStretch(1)
        init_lay.addLayout(status_row)

        self._init_title = QLabel("数据集初始化")
        set_ui_role(self._init_title, "sectionTitle")
        self._init_desc = QLabel("选择裁剪图目录和数据集根目录，程序会自动创建子目录并保存配置。")
        self._init_desc.setWordWrap(True)
        set_ui_role(self._init_desc, "muted")
        init_lay.addWidget(self._init_title)
        init_lay.addWidget(self._init_desc)

        init_lay.setSpacing(9)

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("检查模式"))
        self._dataset_mode = QComboBox()
        self._dataset_mode.setAccessibleName("检查模式")
        self._dataset_mode.addItem("初始化数据集", "init")
        self._dataset_mode.addItem("检查已有数据集", "existing")
        self._dataset_mode.currentIndexChanged.connect(lambda _=0: self._on_dataset_mode_changed())
        mode_row.addWidget(self._dataset_mode)
        mode_row.addStretch(1)
        init_lay.addLayout(mode_row)

        src_row = QHBoxLayout()
        self._src_label = QLabel("裁剪图目录")
        src_row.addWidget(self._src_label)
        self._dataset_source_entry = QLineEdit()
        self._dataset_source_entry.setAccessibleName("裁剪图目录")
        self._dataset_source_entry.editingFinished.connect(self._check_all)
        self._dataset_source_entry.textChanged.connect(lambda _=None: self._refresh_generated_preview())
        src_row.addWidget(self._dataset_source_entry, 1)
        self._btn_src = QPushButton("浏览")
        self._btn_src.setAccessibleName("选择裁剪图目录")
        self._btn_src.clicked.connect(lambda: self._choose_dir(self._dataset_source_entry))
        src_row.addWidget(self._btn_src)
        init_lay.addLayout(src_row)

        root_row = QHBoxLayout()
        root_row.addWidget(QLabel("数据集根目录"))
        self._dataset_root_entry = QLineEdit(self._cfg("dataset_root"))
        self._dataset_root_entry.setAccessibleName("数据集根目录")
        self._dataset_root_entry.editingFinished.connect(self._check_all)
        self._dataset_root_entry.textChanged.connect(lambda _=None: self._refresh_generated_preview())
        root_row.addWidget(self._dataset_root_entry, 1)
        btn_root = QPushButton("浏览")
        btn_root.setAccessibleName("选择数据集根目录")
        btn_root.clicked.connect(lambda: self._choose_dir(self._dataset_root_entry))
        root_row.addWidget(btn_root)
        init_lay.addLayout(root_row)

        preview_card = QFrame()
        preview_card.setObjectName("configSection")
        preview_lay = QVBoxLayout(preview_card)
        preview_lay.setContentsMargins(0, 10, 0, 4)
        preview_lay.setSpacing(7)
        preview_title = QLabel("自动配置预览")
        set_ui_role(preview_title, "sectionTitle")
        preview_desc = QLabel("以下目录会根据数据集根目录自动生成并写回配置。")
        preview_desc.setWordWrap(True)
        set_ui_role(preview_desc, "muted")
        preview_lay.addWidget(preview_title)
        preview_lay.addWidget(preview_desc)

        preview_grid = QGridLayout()
        preview_grid.setHorizontalSpacing(10)
        preview_grid.setVerticalSpacing(6)
        preview_grid.setColumnStretch(1, 1)
        self._preview_entries: dict[str, QLineEdit] = {}
        preview_rows = [
            ("pending_filter_dir", "待筛选目录"),
            ("filtered_keep_dir", "筛选保留目录"),
            ("master_images_dir", "母图目录"),
            ("master_annotations_dir", "母标注目录"),
            ("model_a_export_root", "点数据集导出"),
            ("model_b_export_root", "连边数据集导出"),
        ]
        for row_index, (key, label_text) in enumerate(preview_rows):
            label = QLabel(label_text)
            set_ui_role(label, "fieldLabel")
            entry = QLineEdit()
            entry.setReadOnly(True)
            entry.setPlaceholderText("等待选择数据集根目录")
            preview_grid.addWidget(label, row_index, 0)
            preview_grid.addWidget(entry, row_index, 1)
            self._preview_entries[key] = entry
        preview_lay.addLayout(preview_grid)
        init_lay.addWidget(preview_card)

        init_btn_row = ActionBar()
        self._btn_init = QPushButton("初始化数据集")
        self._btn_init.setAccessibleName("初始化数据集")
        set_primary(self._btn_init)
        self._btn_init.clicked.connect(self._init_dataset_from_root)
        init_btn_row.add_widget(self._btn_init)
        self._btn_check_existing = QPushButton("检查已有数据集")
        self._btn_check_existing.setAccessibleName("检查已有数据集")
        set_primary(self._btn_check_existing)
        self._btn_check_existing.clicked.connect(self._check_existing_dataset)
        init_btn_row.add_widget(self._btn_check_existing)
        btn_save = QPushButton("保存")
        btn_save.setAccessibleName("保存配置")
        btn_save.clicked.connect(lambda: (self._sync_form(), QMessageBox.information(self._mw, "", "已保存")))
        init_btn_row.add_widget(btn_save)
        btn_open = QPushButton("打开配置目录")
        btn_open.setAccessibleName("打开配置目录")
        btn_open.clicked.connect(lambda: self._open_path(str(self.config_path.parent)))
        init_btn_row.add_widget(btn_open)
        init_lay.addWidget(init_btn_row)
        lay.addWidget(init_card)

        self._cfg_entries: dict[str, QLineEdit] = {}
        for key, label, kind in FIELDS:
            entry = QLineEdit(self._cfg(key))
            self._cfg_entries[key] = entry

        self._cfg_entries["dataset_root"] = self._dataset_root_entry
        self._cfg_box = _make_output_box()
        lay.addWidget(self._build_log_card(self._cfg_box))
        self._on_dataset_mode_changed()
        self._check_all()
        return page

    def _build_step_filter_data(self) -> QWidget:
        page = QWidget()
        page_lay = QVBoxLayout(page)
        page_lay.setContentsMargins(0, 0, 0, 0)
        page_lay.setSpacing(10)
        page_lay.setAlignment(Qt.AlignTop)

        card, lay = self._build_step_body_card()
        desc = QLabel(
            "此步骤用于在标注前先筛除不符合要求的裁剪图。\n\n"
            "打开样本审阅后逐张浏览待筛选目录：S 保留到筛选保留目录，W 移除，Z 撤销。"
            " 点“完成审阅”后，保留下来的图片会作为后续标注和导出的母图目录。"
        )
        desc.setWordWrap(True)
        set_ui_role(desc, "muted")
        lay.addWidget(desc)

        note = QLabel("提示：不需要筛选时可以点“不需要筛选，直接继续”，后续步骤会直接使用待筛选目录。")
        note.setWordWrap(True)
        set_ui_role(note, "muted")
        lay.addWidget(note)

        status_box = QFrame()
        status_box.setObjectName("hintPanel")
        status_lay = QVBoxLayout(status_box)
        status_lay.setContentsMargins(12, 10, 12, 10)
        status_lay.setSpacing(6)
        status_title = QLabel("当前目录状态")
        set_ui_role(status_title, "sectionTitle")
        status_lay.addWidget(status_title)
        self._filter_pending_label = QLabel("待筛选目录：-")
        self._filter_keep_label = QLabel("筛选保留目录：-")
        self._filter_active_label = QLabel("当前生效目录：-")
        for lbl in (self._filter_pending_label, self._filter_keep_label, self._filter_active_label):
            lbl.setWordWrap(True)
            set_ui_role(lbl, "muted")
            status_lay.addWidget(lbl)
        lay.addWidget(status_box)

        row = ActionBar()
        btn_skip = QPushButton("不需要筛选，直接继续")
        btn_skip.setAccessibleName("跳过数据筛选")
        btn_skip.clicked.connect(self._use_pending_filter_dir)
        row.add_widget(btn_skip)
        btn = QPushButton("打开数据筛选器")
        btn.setAccessibleName("打开数据筛选器")
        set_primary(btn)
        btn.clicked.connect(self._open_filter_data)
        row.add_widget(btn)
        lay.addWidget(row)

        page_lay.addWidget(card)
        page_lay.addStretch(1)
        self._refresh_filter_step_status()
        return page

    def _open_filter_data(self):
        from ..point_filter_dialog import StitchPointFilterDialog

        dlg = StitchPointFilterDialog(self._mw)
        dataset_root = self._cfg("dataset_root")
        keep_dir = self._cfg("filtered_keep_dir") or (str(Path(dataset_root) / "filtered_keep") if dataset_root else "")
        dlg.reviewCompleted.connect(self._apply_review_result)
        dlg.configure_paths(
            mode="unlabeled",
            image_dir=self._cfg("pending_filter_dir") or self._cfg("master_images_dir"),
            label_dir=self._cfg("master_annotations_dir"),
            save_dir=keep_dir,
            auto_load=True,
        )
        self._open_business_dialog(
            dlg,
            "CAB-F 数据筛选",
            "完成后返回当前 CAB-F 流程并继续后续步骤。",
        )

    def _apply_review_result(self, result):
        """CAB-F adapter: map a generic review result into this flow's roles."""
        self._apply_filtered_images_dir(str(result.active_image_dir))

    def _open_business_dialog(self, dialog, title: str, subtitle: str = ""):
        opener = getattr(self._mw, "open_workspace_dialog", None)
        if callable(opener):
            return opener(dialog, title, subtitle)
        return dialog.exec()

    def _apply_filtered_images_dir(self, filtered_dir: str):
        filtered_path = Path(filtered_dir)
        self._cfg_entries["filtered_keep_dir"].setText(str(filtered_path))
        self._cfg_entries["master_images_dir"].setText(str(filtered_path))
        self.step_status[1] = COMPLETED
        self._refresh_step_sidebar()
        self._refresh_step_summary()
        self._refresh_filter_step_status()
        self._sync_form()
        if hasattr(self, "_cfg_box"):
            self._cfg_box.appendPlainText(f"已将筛选后的图片目录应用到后续流程: {filtered_path}")
        if hasattr(self._mw, "show_status"):
            self._mw.show_status("已应用筛选结果到后续流程")
        if hasattr(self._mw, "refresh_tool_overview"):
            self._mw.refresh_tool_overview()

    def _use_pending_filter_dir(self):
        pending_dir = self._cfg("pending_filter_dir")
        if not pending_dir:
            dataset_root = self._cfg("dataset_root")
            if dataset_root:
                pending_dir = str(self._expected_dataset_paths(Path(dataset_root))["pending_filter_dir"])
        if not pending_dir:
            QMessageBox.warning(self._mw, "提示", "当前没有可用的待筛选目录，请先完成数据集初始化。")
            return
        self._cfg_entries["master_images_dir"].setText(pending_dir)
        self.step_status[1] = COMPLETED
        self._refresh_step_sidebar()
        self._refresh_step_summary()
        self._refresh_filter_step_status()
        self._sync_form()
        if hasattr(self, "_cfg_box"):
            self._cfg_box.appendPlainText(f"已选择跳过筛选，后续流程继续使用待筛选目录: {pending_dir}")
        if hasattr(self._mw, "show_status"):
            self._mw.show_status("已跳过筛选，继续后续流程")
        if hasattr(self._mw, "refresh_tool_overview"):
            self._mw.refresh_tool_overview()

    def _refresh_filter_step_status(self):
        if not hasattr(self, "_filter_pending_label"):
            return
        pending_dir = self._cfg_entries.get("pending_filter_dir").text().strip() if self._cfg_entries.get("pending_filter_dir") else self._cfg("pending_filter_dir")
        keep_dir = self._cfg_entries.get("filtered_keep_dir").text().strip() if self._cfg_entries.get("filtered_keep_dir") else self._cfg("filtered_keep_dir")
        active_dir = self._cfg_entries.get("master_images_dir").text().strip() if self._cfg_entries.get("master_images_dir") else self._cfg("master_images_dir")
        self._filter_pending_label.setText(f"待筛选目录：{pending_dir or '-'}")
        self._filter_keep_label.setText(f"筛选保留目录：{keep_dir or '-'}")
        self._filter_active_label.setText(f"当前生效目录：{active_dir or '-'}")

    def _build_step_annotate(self) -> QWidget:
        page = QWidget()
        page_lay = QVBoxLayout(page)
        page_lay.setContentsMargins(0, 0, 0, 0)
        page_lay.setSpacing(10)
        page_lay.setAlignment(Qt.AlignTop)
        card, lay = self._build_step_body_card()
        desc = QLabel(
            "此步骤需要手动操作。\n\n点击下方按钮打开点边一体标注器，标注或修正缝纫点，"
            "并删除错误连边、补充缺失连边。\n会自动带入当前母图目录，标注读取并保存到母标注目录。"
        )
        desc.setWordWrap(True)
        set_ui_role(desc, "muted")
        lay.addWidget(desc)
        row = ActionBar()
        btn = QPushButton("打开点边一体标注器")
        btn.setAccessibleName("打开点边标注器")
        set_primary(btn)
        btn.clicked.connect(self._open_annotator)
        row.add_widget(btn)
        lay.addWidget(row)
        page_lay.addWidget(card)
        page_lay.addStretch(1)
        return page

    def _open_annotator(self):
        from ..graph_annotation_dialog import StitchGraphEditorDialog

        self._sync_form_to_memory()
        annotation_dir = self._cfg("master_annotations_dir")
        if annotation_dir:
            Path(annotation_dir).mkdir(parents=True, exist_ok=True)
        dlg = StitchGraphEditorDialog(self._mw)
        dlg.configure_paths(
            image_dir=self._cfg("master_images_dir"),
            label_dir=annotation_dir,
            output_dir=annotation_dir,
            overwrite_source=True,
            auto_open=True,
        )
        self._open_business_dialog(
            dlg,
            "CAB-F 点边标注",
            "保存后返回当前 CAB-F 流程。",
        )

    def _build_step_export(self) -> QWidget:
        from ..dataset_export_panel import CabfDatasetExportPanel

        self._export_panel = CabfDatasetExportPanel(self)
        self._export_panel.operationFinished.connect(self._on_export_operation)
        return self._export_panel

    def _on_export_operation(self, title: str, ok: bool) -> None:
        index = next(i for i, step in enumerate(STEPS) if step.key == "export")
        if not ok:
            self.step_status[index] = ERROR
        elif title != "校验":
            self.step_status[index] = COMPLETED
        self._refresh_step_sidebar()
        self._refresh_step_summary()
        # Remember edited export destinations for the next session.
        panel = self._export_panel
        set_nested(self.config_data, "model_a_export_root", panel.edit_model_a_output_dir.text().strip())
        set_nested(self.config_data, "model_b_export_root", panel.edit_model_b_output_dir.text().strip())
        save_config(self.config_path, self.config_data)


class CabfWorkflowActivity(QWidget):
    """Toolbox page hosting the CAB-F flow without a legacy labeling workspace.

    It supplies the small host interface the flow expects: persisted flow
    config, status text, and opening nested editors through the page router.
    """

    capability_key = "cabf.workflow"

    def __init__(self, runtime, parent: QWidget | None = None, config_path: Path | None = None) -> None:
        super().__init__(parent)
        self.runtime = runtime
        self.config_path = Path(config_path or CONFIG_PATH)
        self.config_data = load_config(self.config_path)
        self.setProperty("ownsPageHeader", True)
        self._apply_state_to_config(runtime.project_context.state)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.page = StitchWorkflowPage(self, self)
        layout.addWidget(self.page)

    def _apply_state_to_config(self, state) -> None:
        for key, value in (
            ("dataset_root", state.dataset_root),
            ("master_images_dir", state.image_dir),
            ("master_annotations_dir", state.annotation_dir),
        ):
            if value and not self.config_data.get(key):
                self.config_data[key] = value

    def apply_project_context(self, state) -> None:
        self._apply_state_to_config(state)

    def show_status(self, message: str) -> None:
        show = getattr(self.runtime.window, "show_status", None)
        if callable(show):
            show(message)

    def open_workspace_dialog(self, dialog, title: str, subtitle: str = ""):
        return self.runtime.page_router.open_dialog(
            dialog, title=title, source_key=self.capability_key, subtitle=subtitle,
        )

    def shutdown(self) -> None:
        self.page.shutdown()
