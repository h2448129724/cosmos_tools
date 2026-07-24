"""缝纫点与连边标注数据处理 — 9步流程工具页面。"""
from __future__ import annotations

import locale
import os
import shutil
import subprocess
from pathlib import Path

from PySide6.QtCore import QSettings, QThread, Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
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
    cabf_flow_script,
    default_train_modules_root,
    get_nested,
    save_config,
    set_nested,
)
from shared.project_paths import repo_root
from shared.conda_runtime import CondaEnvManager

from ..theme import TOKENS, badge_style
from .base import BaseToolPage, set_primary


class StepInfo:
    __slots__ = ("key", "title", "desc")

    def __init__(self, key: str, title: str, desc: str):
        self.key = key
        self.title = title
        self.desc = desc


STEPS = [
    StepInfo("config", "配置检查", "校验路径与模型权重配置"),
    StepInfo("filter_data", "数据筛选", "先筛掉不符合要求的裁剪图，再进入预测流程"),
    StepInfo("predict_points", "缝纫点预测", "运行缝纫点批量推理"),
    StepInfo("edit_points", "缝纫点修正", "修正缺失、错误或偏移的缝纫点"),
    StepInfo("predict_edges", "连边预测", "运行连边批量推理"),
    StepInfo("edit_edges", "连边修正", "删除错误连边、补充缺失连边"),
    StepInfo("validate", "数据校验", "校验母标注完整性"),
    StepInfo("export", "数据导出", "导出模型 A/B 训练数据集"),
    StepInfo("train", "模型训练", "训练缝纫点检测与连边模型"),
]

PENDING, ACTIVE, COMPLETED, ERROR = range(4)
_AUTO_KEYS = {"predict_points", "predict_edges", "validate", "export", "train"}
_MANUAL_KEYS = {"filter_data", "edit_points", "edit_edges"}
_STATUS_TEXT = {PENDING: "待处理", ACTIVE: "进行中", COMPLETED: "已完成", ERROR: "异常"}

_STEP_COLORS = {
    PENDING: ("#9AA0A5", "#596168"),
    ACTIVE: ("#356A9A", "#28577F"),
    COMPLETED: ("#4D735A", "#356848"),
    ERROR: ("#A83B32", "#8E332C"),
}


class _StepIndicator(QWidget):
    """Compact clickable step list with explicit text labels."""

    stepClicked = Signal(int)

    def __init__(self, steps: list[StepInfo], parent=None):
        super().__init__(parent)
        self._steps = steps
        self._statuses: list[int] = [ACTIVE] + [PENDING] * (len(steps) - 1)
        self._current = 0
        self.setFixedWidth(176)
        self._buttons: list[QPushButton] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        for index, step in enumerate(steps):
            btn = QPushButton()
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
            accent, text_color = _STEP_COLORS.get(state, _STEP_COLORS[PENDING])
            is_current = index == self._current
            symbol = "✓" if state == COMPLETED else str(index + 1)
            bg = "#D8E3EE" if is_current else "transparent"
            border = "#356A9A" if is_current else "transparent"
            title_color = "#202428" if is_current else text_color
            btn.setText(f"{symbol}  {step.title}")
            btn.setStyleSheet(
                "QPushButton{"
                f"text-align:left;padding:4px 8px;border-radius:0;"
                f"border:none;border-left:3px solid {border};background:{bg};"
                f"color:{title_color};font-size:12px;font-weight:{'700' if is_current else '500'};"
                "}"
                "QPushButton:hover{background:#E7E9EB;}"
            )
            btn.setToolTip(step.desc)


def _field_help(kind: str) -> str:
    return "目录路径" if kind == "dir" else "文件路径"


def _make_output_box() -> QPlainTextEdit:
    box = QPlainTextEdit()
    box.setReadOnly(True)
    box.setFixedHeight(120)
    box.setPlaceholderText("输出日志...")
    return box


class _CmdWorker(QThread):
    finished = Signal(int, str, str)

    def __init__(self, args: list[str], cwd: str, environment: dict[str, str] | None = None, parent=None):
        super().__init__(parent)
        self._args = args
        self._cwd = cwd
        self._environment = environment
        self._process: subprocess.Popen | None = None
        self._cancel_requested = False

    @staticmethod
    def _decode_output(data: bytes | str | None) -> str:
        if not data:
            return ""
        if isinstance(data, str):
            return data
        encodings = []
        preferred = locale.getpreferredencoding(False)
        if preferred:
            encodings.append(preferred)
        encodings.extend(["utf-8", "gbk"])
        tried: set[str] = set()
        for encoding in encodings:
            if encoding in tried:
                continue
            tried.add(encoding)
            try:
                return data.decode(encoding)
            except UnicodeDecodeError:
                continue
        return data.decode(preferred or "utf-8", errors="replace")

    def run(self):
        try:
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
            self._process = subprocess.Popen(
                self._args,
                cwd=self._cwd,
                env=self._environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=False,
                creationflags=creationflags,
            )
            try:
                stdout, stderr = self._process.communicate(timeout=3600)
            except subprocess.TimeoutExpired:
                self.cancel()
                stdout, stderr = self._finish_terminated_process()
                self.finished.emit(-1, self._decode_output(stdout), "命令超时（超过 3600 秒）")
                return
            if self._cancel_requested:
                self.finished.emit(-1, self._decode_output(stdout), "命令已取消")
                return
            self.finished.emit(
                int(self._process.returncode or 0),
                self._decode_output(stdout),
                self._decode_output(stderr),
            )
        except Exception as exc:
            self.finished.emit(-1, "", str(exc))
        finally:
            self._process = None

    def cancel(self) -> None:
        self._cancel_requested = True
        process = self._process
        if process is None or process.poll() is not None:
            return
        if os.name == "nt":
            try:
                subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                if process.poll() is not None:
                    return
            except OSError:
                pass
        process.terminate()

    def _finish_terminated_process(self) -> tuple[bytes | str | None, bytes | str | None]:
        process = self._process
        if process is None:
            return b"", b""
        try:
            return process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            return process.communicate()


class StitchWorkflowPage(BaseToolPage):
    tool_key = "workflow_page"
    tool_title = "缝纫点与连边标注数据处理"
    tool_nav_title = "缝纫点与连边"
    tool_icon = "◉"
    tool_summary = "覆盖配置检查、筛选、预测、人工修正、校验、导出到训练的完整 9 步流程。"
    tool_tags = ("全流程", "标注修正", "训练导出")
    _PRESET_MAP = {"均衡": "balanced", "激进": "aggressive", "保守": "conservative"}

    def __init__(self, main_window, parent=None):
        super().__init__(main_window, parent)
        self.config_path = main_window.config_path
        self.config_data = main_window.config_data
        self.step_status: list[int] = [ACTIVE] + [PENDING] * (len(STEPS) - 1)
        self.current_step = 0
        self._worker: _CmdWorker | None = None
        self._conda_manager = CondaEnvManager()
        self._build_ui()
        self._go_to_step(0)

    def shutdown(self) -> None:
        """Cancel an in-flight workflow command before the page is destroyed."""
        worker = self._worker
        if worker is None or not worker.isRunning():
            return
        worker.cancel()
        worker.wait(10000)

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(12)

        sidebar = self._build_step_sidebar()
        sidebar.setFixedWidth(176)
        root.addWidget(sidebar)

        right = QVBoxLayout()
        right.setContentsMargins(0, 0, 0, 0)
        right.setSpacing(10)
        right.addWidget(self._build_page_header())

        self.step_stack = QStackedWidget()
        for builder in (
            self._build_step_config,
            self._build_step_filter_data,
            self._build_step_predict_points,
            self._build_step_edit_points,
            self._build_step_predict_edges,
            self._build_step_edit_edges,
            self._build_step_validate,
            self._build_step_export,
            self._build_step_train,
        ):
            self.step_stack.addWidget(builder())
        right.addWidget(self.step_stack, 1)
        right.addWidget(self._build_nav())
        root.addLayout(right, 1)

    def _build_step_sidebar(self) -> QWidget:
        frame = QFrame()
        frame.setObjectName("stepSidebar")
        lay = QVBoxLayout(frame)
        lay.setContentsMargins(8, 10, 8, 10)
        lay.setSpacing(4)

        hdr = QLabel("流程步骤")
        hdr.setStyleSheet("font-size:12px;font-weight:700;color:#596168;")
        lay.addWidget(hdr)

        self._step_indicator = _StepIndicator(STEPS)
        self._step_indicator.stepClicked.connect(self._go_to_step)
        lay.addWidget(self._step_indicator)
        lay.addStretch(1)
        return frame

    def _build_page_header(self) -> QWidget:
        header = QWidget()
        lay = QVBoxLayout(header)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)

        self._page_title = QLabel("")
        self._page_title.setStyleSheet("font-size:17px;font-weight:700;color:#202428;")
        lay.addWidget(self._page_title)

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(10)
        self._page_desc = QLabel("")
        self._page_desc.setWordWrap(True)
        self._page_desc.setStyleSheet("font-size:12px;color:#697077;")
        row.addWidget(self._page_desc, 1)
        self._page_status = QLabel("")
        self._page_status.setStyleSheet(badge_style("neutral"))
        row.addWidget(self._page_status, 0, Qt.AlignTop)
        lay.addLayout(row)
        return header

    def _build_step_body_card(self) -> tuple[QWidget, QVBoxLayout]:
        card = QFrame()
        card.setObjectName("stepContentCard")
        card.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        lay = QVBoxLayout(card)
        lay.setContentsMargins(16, 14, 16, 14)
        lay.setSpacing(9)
        lay.setAlignment(Qt.AlignTop)
        return card, lay

    def _build_log_card(self, box: QPlainTextEdit) -> QWidget:
        card = QFrame()
        card.setObjectName("logCard")
        card_height = box.height() + 52
        card.setFixedHeight(card_height)
        card.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        lay = QVBoxLayout(card)
        lay.setContentsMargins(12, 10, 12, 10)
        lay.setSpacing(6)
        lay.setAlignment(Qt.AlignTop)
        title = QLabel("运行日志")
        title.setStyleSheet("font-size:12px;font-weight:600;color:#202428;")
        lay.addWidget(title)
        lay.addWidget(box)
        return card

    def _build_nav(self) -> QWidget:
        bar = QFrame()
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(12)

        self._btn_prev = QPushButton("< 上一步")
        self._btn_prev.clicked.connect(self._prev_step)
        self._btn_done = QPushButton("标记完成")
        self._btn_done.clicked.connect(self._mark_done)
        self._btn_next = QPushButton("下一步 >")
        set_primary(self._btn_next)
        self._btn_next.clicked.connect(self._next_step)

        lay.addWidget(self._btn_prev)
        lay.addStretch(1)
        lay.addWidget(self._btn_done)
        lay.addWidget(self._btn_next)
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
            self._page_status.setText(f"当前状态：还差 {missing} 项" if missing else f"当前状态：{valid}/{total} 已就绪")
            self._page_status.setStyleSheet(badge_style("warning" if missing else "success"))
        else:
            text, level = self.status_badge()
            self._page_status.setText(f"当前状态：{text}")
            self._page_status.setStyleSheet(badge_style(level))

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
        key = STEPS[idx].key
        cfg = self._cfg
        if key == "predict_points" and hasattr(self, "_pp_in"):
            self._pp_in.setText(cfg("master_images_dir"))
            self._pp_out.setText(cfg("point_predictions_dir"))
            self._pp_model.setText(cfg("weights.sew_point_onnx"))
            self._pp_thr.setValue(float(cfg("predict.point_threshold") or 0.3))
            self._pp_dist.setValue(float(cfg("predict.point_distance_threshold") or 3.0))
        elif key == "predict_edges" and hasattr(self, "_pe_img"):
            self._pe_img.setText(cfg("master_images_dir"))
            self._pe_ann.setText(cfg("master_annotations_dir"))
            self._pe_out.setText(cfg("edge_predictions_dir"))
            self._pe_model.setText(cfg("weights.sew_point_connector_pth"))
        elif key == "validate" and hasattr(self, "_val_img"):
            self._val_img.setText(cfg("master_images_dir"))
            self._val_ann.setText(cfg("master_annotations_dir"))
        elif key == "export" and hasattr(self, "_ex_img"):
            self._ex_img.setText(cfg("master_images_dir"))
            self._ex_ann.setText(cfg("master_annotations_dir"))
            self._ex_a.setText(cfg("model_a_export_root"))
            self._ex_b.setText(cfg("model_b_export_root"))
        elif key == "train" and hasattr(self, "_tr_a"):
            self._tr_a.setText(cfg("outputs.sew_point_train_out"))
            self._tr_b.setText(cfg("outputs.sew_point_conntect_train_out"))

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
        if hasattr(self, "_tr_a"):
            self._set_cfg_text("outputs.sew_point_train_out", self._tr_a.text().strip())
        if hasattr(self, "_tr_b"):
            self._set_cfg_text("outputs.sew_point_conntect_train_out", self._tr_b.text().strip())
        for key, _label, _kind in FIELDS:
            entry = self._cfg_entries.get(key)
            if entry is not None:
                set_nested(self.config_data, key, entry.text().strip())
        if hasattr(self, "_pp_thr"):
            set_nested(self.config_data, "predict.point_threshold", f"{self._pp_thr.value():.2f}")
        if hasattr(self, "_pp_dist"):
            set_nested(self.config_data, "predict.point_distance_threshold", f"{self._pp_dist.value():.2f}")

    def _sync_form(self) -> None:
        """Persist the current form after an explicit save or commit action."""
        self._sync_form_to_memory()
        save_config(self.config_path, self.config_data)

    def _expected_dataset_paths(self, dataset_root: Path) -> dict[str, Path]:
        runs_root = dataset_root / "runs"
        return {
            "pending_filter_dir": dataset_root / "pending_filter",
            "filtered_keep_dir": dataset_root / "filtered_keep",
            "master_images_dir": dataset_root / "pending_filter",
            "master_annotations_dir": dataset_root / "master_annotations" / "annotations",
            "point_predictions_dir": dataset_root / "predictions" / "points",
            "edge_predictions_dir": dataset_root / "predictions" / "edges",
            "model_a_export_root": dataset_root / "model_a_export",
            "model_b_export_root": dataset_root / "model_b_export",
            "point_train_out": runs_root / "sew_point_train",
            "edge_train_out": runs_root / "sew_point_conntect_train",
        }

    def _apply_dataset_root_paths(self, dataset_root: Path) -> dict[str, Path]:
        paths = self._expected_dataset_paths(dataset_root)
        active_image_dir = self._resolve_active_image_dir(dataset_root, paths)
        self._cfg_entries["img_tools_root"].setText(str(repo_root()))
        self._cfg_entries["dataset_root"].setText(str(dataset_root))
        self._cfg_entries["train_model_modules_root"].setText(default_train_modules_root())
        self._cfg_entries["pending_filter_dir"].setText(str(paths["pending_filter_dir"]))
        self._cfg_entries["filtered_keep_dir"].setText(str(paths["filtered_keep_dir"]))
        self._cfg_entries["master_images_dir"].setText(str(active_image_dir))
        self._cfg_entries["master_annotations_dir"].setText(str(paths["master_annotations_dir"]))
        self._cfg_entries["point_predictions_dir"].setText(str(paths["point_predictions_dir"]))
        self._cfg_entries["edge_predictions_dir"].setText(str(paths["edge_predictions_dir"]))
        self._cfg_entries["model_a_export_root"].setText(str(paths["model_a_export_root"]))
        self._cfg_entries["model_b_export_root"].setText(str(paths["model_b_export_root"]))
        self._cfg_entries["outputs.sew_point_train_out"].setText(str(paths["point_train_out"]))
        self._cfg_entries["outputs.sew_point_conntect_train_out"].setText(str(paths["edge_train_out"]))
        if hasattr(self, "_tr_a"):
            self._tr_a.setText(str(paths["point_train_out"]))
        if hasattr(self, "_tr_b"):
            self._tr_b.setText(str(paths["edge_train_out"]))
        set_nested(self.config_data, "outputs.sew_point_train_out", str(paths["point_train_out"]))
        set_nested(self.config_data, "outputs.sew_point_conntect_train_out", str(paths["edge_train_out"]))
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
        root = str(repo_root())
        values = {
            "img_tools_root": root,
            "train_model_modules_root": default_train_modules_root(),
            "dataset_root": dataset_root_text,
            "pending_filter_dir": "",
            "filtered_keep_dir": "",
            "master_images_dir": "",
            "master_annotations_dir": "",
            "point_predictions_dir": "",
            "edge_predictions_dir": "",
            "model_a_export_root": "",
            "model_b_export_root": "",
            "point_train_out": "",
            "edge_train_out": "",
        }
        if dataset_root_text:
            paths = self._expected_dataset_paths(Path(dataset_root_text))
            active_dir = self._resolve_active_image_dir(Path(dataset_root_text), paths)
            values.update({
                "pending_filter_dir": str(paths["pending_filter_dir"]),
                "filtered_keep_dir": str(paths["filtered_keep_dir"]),
                "master_images_dir": str(active_dir),
                "master_annotations_dir": str(paths["master_annotations_dir"]),
                "point_predictions_dir": str(paths["point_predictions_dir"]),
                "edge_predictions_dir": str(paths["edge_predictions_dir"]),
                "model_a_export_root": str(paths["model_a_export_root"]),
                "model_b_export_root": str(paths["model_b_export_root"]),
                "point_train_out": str(paths["point_train_out"]),
                "edge_train_out": str(paths["edge_train_out"]),
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
            "选择已有数据集根目录。程序会自动推导并检查 master、predictions、export、runs 等目录。"
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

    def _choose_file(self, entry: QLineEdit):
        settings = QSettings("CABF", "img_tools")
        last_file = settings.value("last_file", entry.text() or ".")
        f, _ = QFileDialog.getOpenFileName(self._mw, "选择文件", last_file)
        if f:
            entry.setText(f)
            settings.setValue("last_file", f)

    def _run_flow(self, args: list[str], box: QPlainTextEdit):
        if self._worker and self._worker.isRunning():
            box.appendPlainText("[忙碌] 上一个命令仍在运行")
            return
        self._sync_form()
        cwd = self._cfg("img_tools_root") or str(repo_root())
        state = getattr(self._mw, "_project_state", None)
        runtime_profile = str(getattr(state, "runtime_profile", "") or "")
        python_executable = self._conda_manager.python_executable(runtime_profile)
        environment = self._conda_manager.process_environment(runtime_profile)
        full = [python_executable, str(cabf_flow_script()), "--config", str(self.config_path), *args]
        box.clear()
        box.appendPlainText("$ " + " ".join(full))
        self._worker = _CmdWorker(full, cwd, environment, self)
        self._worker.finished.connect(lambda rc, out, err: self._on_done(rc, out, err, box))
        self._worker.start()

    def _on_done(self, rc: int, stdout: str, stderr: str, box: QPlainTextEdit):
        text = stdout.strip()
        if stderr.strip():
            text = f"{text}\n[stderr]\n{stderr.strip()}" if text else f"[stderr]\n{stderr.strip()}"
        box.appendPlainText(text)
        box.appendPlainText(f"\nexit={rc}")
        if rc == 0 and STEPS[self.current_step].key in _AUTO_KEYS:
            self.step_status[self.current_step] = COMPLETED
        elif rc != 0:
            self.step_status[self.current_step] = ERROR
        self._refresh_step_sidebar()
        if hasattr(self._mw, "refresh_tool_overview"):
            self._mw.refresh_tool_overview()

    def _open_path(self, p: str):
        if not p:
            return
        resolved = Path(p).resolve()
        if not resolved.exists():
            return
        os.startfile(str(resolved))

    def _header_label(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet("font-size:12px;color:#697077;font-weight:600;")
        return lbl

    def _build_config_row(self, key: str, label: str, kind: str) -> QWidget:
        row = QFrame()
        row.setObjectName("configRow")
        lay = QGridLayout(row)
        lay.setContentsMargins(0, 8, 0, 8)
        lay.setHorizontalSpacing(12)
        lay.setVerticalSpacing(4)
        lay.setColumnStretch(1, 1)
        lay.setColumnMinimumWidth(0, 150)
        lay.setColumnMinimumWidth(2, 72)
        lay.setColumnMinimumWidth(3, 84)

        name = QLabel(label)
        name.setObjectName("configLabel")
        lay.addWidget(name, 0, 0)

        entry = QLineEdit(self._cfg(key))
        entry.setPlaceholderText(_field_help(kind))
        entry.setToolTip(self._cfg(key))
        entry.textChanged.connect(lambda text, e=entry: e.setToolTip(text))
        lay.addWidget(entry, 0, 1)

        btn = QPushButton("浏览")
        btn.setFixedWidth(72)
        btn.clicked.connect(lambda: self._choose_dir(entry) if kind == "dir" else self._choose_file(entry))
        lay.addWidget(btn, 0, 2)

        badge = QLabel("未配置")
        badge.setAlignment(Qt.AlignCenter)
        badge.setFixedWidth(76)
        badge.setStyleSheet(badge_style("neutral"))
        lay.addWidget(badge, 0, 3)

        hint = QLabel(_field_help(kind))
        hint.setStyleSheet("font-size:11px;color:#92989D;")
        lay.addWidget(hint, 1, 0, 1, 2)

        self._cfg_entries[key] = entry
        self._cfg_labels[key] = badge
        self._cfg_kinds[key] = kind
        return row

    def _check_field(self, key: str, kind: str) -> bool:
        val = self._cfg_entries[key].text().strip()
        lbl = self._cfg_labels[key]
        if not val:
            lbl.setText("未配置")
            lbl.setStyleSheet(badge_style("neutral"))
            return False
        p = Path(val)
        ok = p.is_dir() if kind == "dir" else p.is_file()
        lbl.setText("有效" if ok else "缺失")
        lbl.setStyleSheet(badge_style("success" if ok else "danger"))
        return ok

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
        self._config_summary_badge.setText(f"{valid}/{total} 有效")
        self._config_summary_badge.setStyleSheet(
            badge_style("success" if valid == total else "warning" if valid > 0 else "neutral")
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
        self._config_summary_badge.setStyleSheet(badge_style("neutral"))
        status_row.addWidget(self._config_summary_badge, 0, Qt.AlignLeft)
        status_row.addStretch(1)
        init_lay.addLayout(status_row)

        self._init_title = QLabel("数据集初始化")
        self._init_title.setStyleSheet("font-size:15px;font-weight:600;color:#202428;")
        self._init_desc = QLabel("选择裁剪图目录和数据集根目录，程序会自动创建子目录并保存配置。")
        self._init_desc.setWordWrap(True)
        self._init_desc.setStyleSheet("font-size:12px;color:#697077;")
        init_lay.addWidget(self._init_title)
        init_lay.addWidget(self._init_desc)

        init_lay.setSpacing(9)

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("检查模式"))
        self._dataset_mode = QComboBox()
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
        self._dataset_source_entry.editingFinished.connect(self._check_all)
        self._dataset_source_entry.textChanged.connect(lambda _=None: self._refresh_generated_preview())
        src_row.addWidget(self._dataset_source_entry, 1)
        self._btn_src = QPushButton("浏览")
        self._btn_src.setFixedWidth(72)
        self._btn_src.clicked.connect(lambda: self._choose_dir(self._dataset_source_entry))
        src_row.addWidget(self._btn_src)
        init_lay.addLayout(src_row)

        root_row = QHBoxLayout()
        root_row.addWidget(QLabel("数据集根目录"))
        self._dataset_root_entry = QLineEdit(self._cfg("dataset_root"))
        self._dataset_root_entry.editingFinished.connect(self._check_all)
        self._dataset_root_entry.textChanged.connect(lambda _=None: self._refresh_generated_preview())
        root_row.addWidget(self._dataset_root_entry, 1)
        btn_root = QPushButton("浏览")
        btn_root.setFixedWidth(72)
        btn_root.clicked.connect(lambda: self._choose_dir(self._dataset_root_entry))
        root_row.addWidget(btn_root)
        init_lay.addLayout(root_row)

        preview_card = QFrame()
        preview_card.setObjectName("configSection")
        preview_lay = QVBoxLayout(preview_card)
        preview_lay.setContentsMargins(0, 10, 0, 4)
        preview_lay.setSpacing(7)
        preview_title = QLabel("自动配置预览")
        preview_title.setStyleSheet("font-size:14px;font-weight:600;color:#202428;")
        preview_desc = QLabel("以下目录会根据数据集根目录自动生成并写回配置。")
        preview_desc.setWordWrap(True)
        preview_desc.setStyleSheet("font-size:11px;color:#697077;")
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
            ("point_predictions_dir", "点预测目录"),
            ("edge_predictions_dir", "边预测目录"),
            ("model_a_export_root", "模型A导出"),
            ("model_b_export_root", "模型B导出"),
            ("point_train_out", "点训练输出"),
            ("edge_train_out", "边训练输出"),
        ]
        for row_index, (key, label_text) in enumerate(preview_rows):
            label = QLabel(label_text)
            label.setStyleSheet("font-size:11px;color:#697077;")
            entry = QLineEdit()
            entry.setReadOnly(True)
            entry.setPlaceholderText("等待选择数据集根目录")
            preview_grid.addWidget(label, row_index, 0)
            preview_grid.addWidget(entry, row_index, 1)
            self._preview_entries[key] = entry
        preview_lay.addLayout(preview_grid)
        init_lay.addWidget(preview_card)

        divider = QFrame()
        divider.setFixedHeight(1)
        divider.setStyleSheet(f"background:{TOKENS['border_weak']};")
        init_lay.addWidget(divider)

        init_btn_row = QHBoxLayout()
        self._btn_init = QPushButton("初始化数据集")
        set_primary(self._btn_init)
        self._btn_init.clicked.connect(self._init_dataset_from_root)
        init_btn_row.addWidget(self._btn_init)
        self._btn_check_existing = QPushButton("检查已有数据集")
        set_primary(self._btn_check_existing)
        self._btn_check_existing.clicked.connect(self._check_existing_dataset)
        init_btn_row.addWidget(self._btn_check_existing)
        btn_save = QPushButton("保存")
        btn_save.clicked.connect(lambda: (self._sync_form(), QMessageBox.information(self._mw, "", "已保存")))
        init_btn_row.addWidget(btn_save)
        btn_open = QPushButton("打开配置目录")
        btn_open.clicked.connect(lambda: self._open_path(str(self.config_path.parent)))
        init_btn_row.addWidget(btn_open)
        init_btn_row.addStretch(1)
        init_lay.addLayout(init_btn_row)
        lay.addWidget(init_card)

        self._cfg_entries: dict[str, QLineEdit] = {}
        self._cfg_labels: dict[str, QLabel] = {}
        self._cfg_kinds: dict[str, str] = {}
        for key, label, kind in FIELDS:
            entry = QLineEdit(self._cfg(key))
            self._cfg_entries[key] = entry
            self._cfg_kinds[key] = kind

        self._cfg_entries["dataset_root"] = self._dataset_root_entry
        self._cfg_box = _make_output_box()
        lay.addWidget(self._build_log_card(self._cfg_box))
        self._on_dataset_mode_changed()
        self._check_all()
        return page

    def _build_form_step(
        self, rows: list[tuple[str, str, str]], buttons: list[QPushButton]
    ) -> tuple[QWidget, list[QLineEdit], QPlainTextEdit]:
        page = QWidget()
        page_lay = QVBoxLayout(page)
        page_lay.setContentsMargins(0, 0, 0, 0)
        page_lay.setSpacing(10)
        page_lay.setAlignment(Qt.AlignTop)

        card, lay = self._build_step_body_card()
        built_entries: list[QLineEdit] = []
        for label, value, kind in rows:
            entry, row = self._form_row(label, value, kind)
            built_entries.append(entry)
            lay.addLayout(row)
        button_row = QHBoxLayout()
        button_row.setSpacing(10)
        for btn in buttons:
            button_row.addWidget(btn)
        button_row.addStretch(1)
        lay.addLayout(button_row)
        page_lay.addWidget(card)

        box = _make_output_box()
        page_lay.addWidget(self._build_log_card(box))
        page_lay.addStretch(1)
        return page, built_entries, box

    def _form_row(self, label: str, value: str, kind: str) -> tuple[QLineEdit, QHBoxLayout]:
        row = QHBoxLayout()
        row.setSpacing(10)
        lbl = QLabel(label)
        lbl.setFixedWidth(108)
        row.addWidget(lbl)
        entry = QLineEdit(value)
        entry.setPlaceholderText(_field_help(kind))
        entry.setToolTip(value)
        entry.textChanged.connect(lambda text, e=entry: e.setToolTip(text))
        row.addWidget(entry, 1)
        btn = QPushButton("浏览")
        btn.setFixedWidth(72)
        btn.clicked.connect(lambda: self._choose_dir(entry) if kind == "dir" else self._choose_file(entry))
        row.addWidget(btn)
        return entry, row

    def _build_step_predict_points(self) -> QWidget:
        btn_run = QPushButton("执行")
        set_primary(btn_run)
        btn_run.clicked.connect(lambda: self._run_pp(False))
        btn_dry = QPushButton("试运行")
        btn_dry.clicked.connect(lambda: self._run_pp(True))
        page, entries, self._pp_box = self._build_form_step(
            [
                ("输入图片目录", self._cfg("master_images_dir"), "dir"),
                ("输出（缝纫点）", self._cfg("point_predictions_dir"), "dir"),
                ("模型（ONNX）", self._cfg("weights.sew_point_onnx"), "file"),
            ],
            [btn_run, btn_dry],
        )
        self._pp_in, self._pp_out, self._pp_model = entries
        body = page.layout().itemAt(0).widget().layout()
        tr = QHBoxLayout()
        tr.addWidget(QLabel("置信度阈值"))
        self._pp_thr = QDoubleSpinBox()
        self._pp_thr.setRange(0, 1)
        self._pp_thr.setSingleStep(0.05)
        self._pp_thr.setValue(0.3)
        self._pp_thr.setDecimals(2)
        tr.addWidget(self._pp_thr)
        tr.addSpacing(16)
        tr.addWidget(QLabel("距离阈值"))
        self._pp_dist = QDoubleSpinBox()
        self._pp_dist.setRange(0, 9999)
        self._pp_dist.setSingleStep(1.0)
        self._pp_dist.setDecimals(2)
        self._pp_dist.setValue(float(self._cfg("predict.point_distance_threshold") or 3.0))
        tr.addWidget(self._pp_dist)
        tr.addStretch(1)
        body.insertLayout(3, tr)
        return page

    def _build_step_edit_points(self) -> QWidget:
        page = QWidget()
        page_lay = QVBoxLayout(page)
        page_lay.setContentsMargins(0, 0, 0, 0)
        page_lay.setSpacing(10)
        page_lay.setAlignment(Qt.AlignTop)
        card, lay = self._build_step_body_card()
        desc = QLabel(
            "此步骤需要手动操作。\n\n点击下方按钮打开点边一体标注器，直接修正缝纫点。"
            "\n会自动带入当前图片目录、点预测目录和母标注输出目录。"
        )
        desc.setStyleSheet("color:#6B7280;")
        lay.addWidget(desc)
        row = QHBoxLayout()
        btn = QPushButton("打开点边一体标注器")
        set_primary(btn)
        btn.clicked.connect(self._open_pt)
        row.addWidget(btn)
        row.addStretch(1)
        lay.addLayout(row)
        page_lay.addWidget(card)
        page_lay.addStretch(1)
        return page

    def _build_step_filter_data(self) -> QWidget:
        page = QWidget()
        page_lay = QVBoxLayout(page)
        page_lay.setContentsMargins(0, 0, 0, 0)
        page_lay.setSpacing(10)
        page_lay.setAlignment(Qt.AlignTop)

        card, lay = self._build_step_body_card()
        desc = QLabel(
            "此步骤用于在缝纫点预测前先筛除不符合要求的裁剪图。\n\n"
            "默认按无标签模式打开筛选器，直接浏览当前母图目录。"
            " 建议把不合格样本移到垃圾桶，保留下来的图片会继续进入后续预测流程。"
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("color:#6B7280;")
        lay.addWidget(desc)

        note = QLabel("提示：如果只是想剔除不合格图片，优先使用“移到垃圾桶”，这样后续步骤会继续使用当前目录中的剩余样本。")
        note.setWordWrap(True)
        note.setStyleSheet("font-size:12px;color:#64748b;")
        lay.addWidget(note)

        status_box = QFrame()
        status_box.setObjectName("hintPanel")
        status_lay = QVBoxLayout(status_box)
        status_lay.setContentsMargins(12, 10, 12, 10)
        status_lay.setSpacing(6)
        status_title = QLabel("当前目录状态")
        status_title.setStyleSheet("color:#111827;font-size:13px;font-weight:700;")
        status_lay.addWidget(status_title)
        self._filter_pending_label = QLabel("待筛选目录：-")
        self._filter_keep_label = QLabel("筛选保留目录：-")
        self._filter_active_label = QLabel("当前生效目录：-")
        for lbl in (self._filter_pending_label, self._filter_keep_label, self._filter_active_label):
            lbl.setWordWrap(True)
            lbl.setStyleSheet("color:#475569;font-size:12px;")
            status_lay.addWidget(lbl)
        lay.addWidget(status_box)

        row = QHBoxLayout()
        btn_skip = QPushButton("不需要筛选，直接继续")
        btn_skip.clicked.connect(self._use_pending_filter_dir)
        row.addWidget(btn_skip)
        btn = QPushButton("打开数据筛选器")
        set_primary(btn)
        btn.clicked.connect(self._open_filter_data)
        row.addWidget(btn)
        row.addStretch(1)
        lay.addLayout(row)

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
        if hasattr(self, "_pp_in"):
            self._pp_in.setText(str(filtered_path))
        if hasattr(self, "_pe_img"):
            self._pe_img.setText(str(filtered_path))
        if hasattr(self, "_val_img"):
            self._val_img.setText(str(filtered_path))
        if hasattr(self, "_ex_img"):
            self._ex_img.setText(str(filtered_path))
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
        if hasattr(self, "_pp_in"):
            self._pp_in.setText(pending_dir)
        if hasattr(self, "_pe_img"):
            self._pe_img.setText(pending_dir)
        if hasattr(self, "_val_img"):
            self._val_img.setText(pending_dir)
        if hasattr(self, "_ex_img"):
            self._ex_img.setText(pending_dir)
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

    def _open_pt(self):
        from ..graph_annotation_dialog import StitchGraphEditorDialog

        dlg = StitchGraphEditorDialog(self._mw)
        dlg.configure_paths(
            image_dir=self._pp_in.text().strip() if hasattr(self, "_pp_in") else self._cfg("master_images_dir"),
            label_dir=self._pp_out.text().strip() if hasattr(self, "_pp_out") else self._cfg("point_predictions_dir"),
            output_dir=self._cfg("master_annotations_dir"),
            overwrite_source=False,
            auto_open=True,
        )
        self._open_business_dialog(
            dlg,
            "CAB-F 缝纫点修正",
            "保存后返回当前 CAB-F 流程。",
        )

    def _build_step_predict_edges(self) -> QWidget:
        btn_run = QPushButton("执行")
        set_primary(btn_run)
        btn_run.clicked.connect(lambda: self._run_pe(False))
        btn_dry = QPushButton("试运行")
        btn_dry.clicked.connect(lambda: self._run_pe(True))
        page, entries, self._pe_box = self._build_form_step(
            [
                ("图片目录", self._cfg("master_images_dir"), "dir"),
                ("标注目录", self._cfg("master_annotations_dir"), "dir"),
                ("输出（连边）", self._cfg("edge_predictions_dir"), "dir"),
                ("模型（PTH）", self._cfg("weights.sew_point_connector_pth"), "file"),
            ],
            [btn_run, btn_dry],
        )
        self._pe_img, self._pe_ann, self._pe_out, self._pe_model = entries
        body = page.layout().itemAt(0).widget().layout()
        pr = QHBoxLayout()
        pr.addWidget(QLabel("后处理策略"))
        self._pe_preset = QComboBox()
        self._pe_preset.addItems(["均衡", "激进", "保守"])
        pr.addWidget(self._pe_preset)
        pr.addStretch(1)
        body.insertLayout(4, pr)
        edge_hint = QLabel("默认读取母标注目录；如果其中没有可用 JSON，会自动回退到点预测目录。")
        edge_hint.setWordWrap(True)
        edge_hint.setStyleSheet("font-size:12px;color:#6B7280;")
        body.insertWidget(5, edge_hint)
        return page

    def _build_step_edit_edges(self) -> QWidget:
        page = QWidget()
        page_lay = QVBoxLayout(page)
        page_lay.setContentsMargins(0, 0, 0, 0)
        page_lay.setSpacing(10)
        page_lay.setAlignment(Qt.AlignTop)
        card, lay = self._build_step_body_card()
        desc = QLabel(
            "此步骤需要手动操作。\n\n点击下方按钮打开点边一体标注器，删除错误连边、补充缺失连边，"
            "\n会自动带入当前图片目录、连边预测目录和母标注输出目录。"
        )
        desc.setStyleSheet("color:#6B7280;")
        lay.addWidget(desc)
        row = QHBoxLayout()
        btn = QPushButton("打开点边一体标注器")
        set_primary(btn)
        btn.clicked.connect(self._open_eg)
        row.addWidget(btn)
        row.addStretch(1)
        lay.addLayout(row)
        page_lay.addWidget(card)
        page_lay.addStretch(1)
        return page

    def _open_eg(self):
        from ..graph_annotation_dialog import StitchGraphEditorDialog

        dlg = StitchGraphEditorDialog(self._mw)
        dlg.configure_paths(
            image_dir=self._pe_img.text().strip() if hasattr(self, "_pe_img") else self._cfg("master_images_dir"),
            label_dir=self._pe_out.text().strip() if hasattr(self, "_pe_out") else self._cfg("edge_predictions_dir"),
            output_dir=self._cfg("master_annotations_dir"),
            overwrite_source=False,
            auto_open=True,
        )
        self._open_business_dialog(
            dlg,
            "CAB-F 连边修正",
            "保存后返回当前 CAB-F 流程。",
        )

    def _build_step_validate(self) -> QWidget:
        btn = QPushButton("执行校验")
        set_primary(btn)
        btn.clicked.connect(self._run_val)
        page, entries, self._val_box = self._build_form_step(
            [
                ("图片目录", self._cfg("master_images_dir"), "dir"),
                ("标注目录", self._cfg("master_annotations_dir"), "dir"),
            ],
            [btn],
        )
        self._val_img, self._val_ann = entries
        return page

    def _run_val(self):
        self._run_flow(
            [
                "validate",
                "--image-dir",
                self._val_img.text().strip(),
                "--annotation-dir",
                self._val_ann.text().strip(),
                "--show-samples",
            ],
            self._val_box,
        )

    def _build_step_export(self) -> QWidget:
        btn = QPushButton("导出全部")
        set_primary(btn)
        btn.clicked.connect(self._run_export)
        page, entries, self._ex_box = self._build_form_step(
            [
                ("图片目录", self._cfg("master_images_dir"), "dir"),
                ("标注目录", self._cfg("master_annotations_dir"), "dir"),
                ("模型 A 导出目录", self._cfg("model_a_export_root"), "dir"),
                ("模型 B 导出目录", self._cfg("model_b_export_root"), "dir"),
            ],
            [btn],
        )
        self._ex_img, self._ex_ann, self._ex_a, self._ex_b = entries
        return page

    def _run_export(self):
        self._run_flow(
            [
                "export",
                "--image-dir",
                self._ex_img.text().strip(),
                "--annotation-dir",
                self._ex_ann.text().strip(),
                "--model-a-output",
                self._ex_a.text().strip(),
                "--model-b-output",
                self._ex_b.text().strip(),
            ],
            self._ex_box,
        )

    def _build_step_train(self) -> QWidget:
        btn_train = QPushButton("训练")
        set_primary(btn_train)
        btn_train.clicked.connect(lambda: self._run_tr(False))
        btn_dry = QPushButton("试运行")
        btn_dry.clicked.connect(lambda: self._run_tr(True))
        page, entries, self._tr_box = self._build_form_step(
            [
                ("缝纫点模型输出", self._cfg("outputs.sew_point_train_out"), "dir"),
                ("连边模型输出", self._cfg("outputs.sew_point_conntect_train_out"), "dir"),
            ],
            [btn_train, btn_dry],
        )
        self._tr_a, self._tr_b = entries
        return page

    def _run_pp(self, dry: bool):
        args = [
            "predict-points",
            "--image-dir",
            self._pp_in.text().strip(),
            "--output-dir",
            self._pp_out.text().strip(),
            "--model",
            self._pp_model.text().strip(),
            "--threshold",
            str(self._pp_thr.value()),
        ]
        if self._pp_dist.value() > 0:
            args.extend(["--cluster_dist", str(self._pp_dist.value())])
        if dry:
            args.append("--dry-run")
        self._run_flow(args, self._pp_box)

    def _run_pe(self, dry: bool):
        preset = self._PRESET_MAP.get(self._pe_preset.currentText(), "balanced")
        args = [
            "predict-edges",
            "--image-dir",
            self._pe_img.text().strip(),
            "--annotation-dir",
            self._pe_ann.text().strip(),
            "--output-dir",
            self._pe_out.text().strip(),
            "--model",
            self._pe_model.text().strip(),
            "--postprocess-preset",
            preset,
        ]
        if dry:
            args.append("--dry-run")
        self._run_flow(args, self._pe_box)

    def _run_tr(self, dry: bool):
        args = [
            "train",
            "--model-a-out",
            self._tr_a.text().strip(),
            "--model-b-out",
            self._tr_b.text().strip(),
        ]
        if dry:
            args.append("--dry-run")
        self._run_flow(args, self._tr_box)
