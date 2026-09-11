from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import yaml
from PySide6.QtCore import QProcess, QProcessEnvironment, Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QLabel,
    QLineEdit,
    QLayout,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from apps.cosmos_pipeline.outcome import PipelineOutcome, outcome_from_process_exit
from apps.cosmos_pipeline.runner import DEFAULT_BACKEND_CONFIG, PipelineRequest, describe_config, validate_request

from .capabilities import CapabilityRuntime
from .paths import COSMOS_ROOT, TOOLBOX_ROOT, runtime_env
from .project_context import ProjectState
from .task_center import TaskStatus
from .task_presentation import TaskPresentation, present_task
from .ui.primitives import ActionBar, CollapsibleLogPanel, PathField, SectionSurface, StatusBanner


def build_pipeline_command(request: PipelineRequest, python_executable: str | None = None) -> list[str]:
    command = [
        python_executable or sys.executable,
        "-u",
        str(TOOLBOX_ROOT / "scripts" / "cosmos_pipeline_test.py"),
        "--config",
        str(request.config_path),
        "--backend-config",
        str(request.backend_config_path),
        "--input-manifest",
        str(request.input_manifest_path),
        "--output-dir",
        str(request.output_dir),
        "--engine",
        request.engine,
    ]
    if request.dry_run:
        command.append("--dry-run")
    if request.persist_db:
        command.append("--persist-db")
    command.append("--fail-on-ng" if request.fail_on_ng else "--allow-ng")
    return command


def _card(title: str, subtitle: str = "") -> tuple[QFrame, QVBoxLayout]:
    surface = SectionSurface(title, subtitle)
    return surface, surface.body_layout


class CosmosPipelineActivity(QWidget):
    capability_key = "cosmos.pipeline_test"

    def __init__(self, runtime: CapabilityRuntime, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.runtime = runtime
        self._process: QProcess | None = None
        self._task_id = ""
        self._task_finished = True
        self._cancelling = False
        self._stdout_buffer = ""
        self._stderr_buffer = ""
        self._output_auto = True
        self._ephemeral_manifest: Path | None = None
        self._build_ui()
        self.apply_project_context(runtime.project_context.state)

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        self.scroll = QScrollArea()
        self.scroll.setObjectName("cosmosPipelineScroll")
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        layout.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
        self.scroll.setWidget(content)
        outer.addWidget(self.scroll, 1)
        config_card, config_layout = _card(
            "配置 → 输入 → 检测 → 规则 → 结果",
            "根据产品 YAML 自动识别 CAB、CAB-F、DAB-* 或 OS-DAB，并执行对应的 Cosmos 业务链。",
        )
        self.config_edit = self._path_row(
            config_layout, "产品 YAML", "选择 conf 下的产品配置", self._pick_config
        )
        self.config_edit.editingFinished.connect(self._describe)
        self.backend_edit = self._path_row(
            config_layout, "后端配置", "assets/config/backend_config.yaml", self._pick_backend
        )
        self.manifest_edit = self._path_row(
            config_layout, "输入清单", "YAML：case 与各输入槽文件", self._pick_manifest
        )
        self.output_edit = self._path_row(
            config_layout, "结果目录", "独立的测试输出目录", self._pick_output
        )
        self.output_edit.textEdited.connect(lambda _text: setattr(self, "_output_auto", False))
        self.description_label = QLabel("选择产品配置后显示项目、执行引擎和输入槽。")
        self.description_label.setObjectName("cabfMuted")
        self.description_label.setWordWrap(True)
        config_layout.addWidget(self.description_label)
        options = ActionBar()
        options.add_widget(QLabel("执行引擎"))
        self.engine_combo = QComboBox()
        self.engine_combo.addItem("自动（与生产一致）", "auto")
        self.engine_combo.addItem("本地算法", "local")
        self.engine_combo.addItem("算法服务", "service")
        self.engine_combo.setAccessibleName("执行引擎")
        options.add_widget(self.engine_combo)
        self.dry_run_check = QCheckBox("只检查，不加载模型")
        options.add_widget(self.dry_run_check)
        self.persist_db_check = QCheckBox("写入 Cosmos 数据库")
        options.add_widget(self.persist_db_check)
        self.fail_on_ng_check = QCheckBox("NG 返回非零退出码（默认）")
        self.fail_on_ng_check.setChecked(True)
        options.add_widget(self.fail_on_ng_check)
        config_layout.addWidget(options)
        layout.addWidget(config_card)

        run_card, run_layout = _card(
            "运行状态",
            "Headless 测试不启动生产页面、相机、PLC、Socket 或同步线程；数据库默认关闭。",
        )
        toolbar = ActionBar()
        self.status_banner = StatusBanner("等待运行")
        self.status_label = self.status_banner.label
        toolbar.add_widget(self.status_banner)
        self.start_button = QPushButton("运行完整流程")
        self.start_button.setProperty("buttonRole", "primary")
        self.start_button.clicked.connect(self.start)
        self.start_button.setAccessibleName("运行完整 Cosmos 流程")
        toolbar.add_widget(self.start_button)
        self.cancel_button = QPushButton("停止")
        self.cancel_button.setProperty("buttonRole", "secondary")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel)
        toolbar.add_widget(self.cancel_button)
        run_layout.addWidget(toolbar)
        self.error_label = QLabel("")
        self.error_label.setObjectName("cabfMuted")
        self.error_label.setWordWrap(True)
        run_layout.addWidget(self.error_label)
        self.log_panel = CollapsibleLogPanel("结构化运行日志")
        self.log = self.log_panel.log
        self.log.setMinimumHeight(120)
        self.log.setPlaceholderText("结构化运行日志会显示在这里。")
        run_layout.addWidget(self.log_panel, 1)
        layout.addWidget(run_card, 1)

    def _path_row(self, layout, label: str, placeholder: str, callback) -> QLineEdit:
        field = PathField(label, placeholder=placeholder)
        field.browse_requested.connect(callback)
        if isinstance(layout, QFormLayout):
            layout.addRow(field)
        else:
            layout.addWidget(field)
        return field.line_edit

    def _set_status(self, text: str, tone: str = "neutral") -> None:
        self.status_banner.set_status(text, tone)

    def _task_presentation(self) -> TaskPresentation | None:
        """Read the current task through the shared, pure UI projection."""

        if not self._task_id:
            return None
        task_center = self.runtime.task_center
        presentation_for = getattr(task_center, "presentation", None)
        if callable(presentation_for):
            presentation = presentation_for(self._task_id)
            if presentation is not None:
                return presentation
        # Lightweight runtime doubles may expose only the record query.  They
        # still flow through the same functional projection as TaskCenter.
        get_task = getattr(task_center, "get", None)
        record = get_task(self._task_id) if callable(get_task) else None
        return present_task(record) if record is not None else None

    def _sync_task_presentation(self, detail: str = "") -> TaskPresentation | None:
        presentation = self._task_presentation()
        if presentation is None:
            return None
        text = presentation.status_text
        if detail:
            text = f"{text} · {detail}"
        self._set_status(text, presentation.tone)
        self.start_button.setEnabled(not presentation.active)
        self.cancel_button.setEnabled(presentation.cancellable)
        return presentation

    def apply_project_context(self, state: ProjectState) -> None:
        config = state.pipeline_config_path or state.cabf_config_path
        if not self.config_edit.text().strip() and config:
            self.config_edit.setText(config)
        if not self.backend_edit.text().strip():
            self.backend_edit.setText(state.pipeline_backend_config_path or str(DEFAULT_BACKEND_CONFIG))
        if not self.manifest_edit.text().strip() and state.pipeline_input_manifest_path:
            self.manifest_edit.setText(state.pipeline_input_manifest_path)
        if self._output_auto and self._process is None:
            root = Path(state.output_root) if state.output_root else TOOLBOX_ROOT / "artifacts"
            self.output_edit.setText(str(root / "cosmos_pipeline" / datetime.now().strftime("%Y%m%d_%H%M%S")))
        self._describe()

    def _pick_config(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择 Cosmos 产品配置",
            self.config_edit.text().strip() or str(COSMOS_ROOT / "conf"),
            "YAML (*.yaml *.yml)",
        )
        if path:
            self.config_edit.setText(path)
            self._describe()

    def _pick_backend(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择后端配置",
            self.backend_edit.text().strip() or str(DEFAULT_BACKEND_CONFIG),
            "YAML (*.yaml *.yml)",
        )
        if path:
            self.backend_edit.setText(path)

    def _pick_manifest(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择 Pipeline 输入清单",
            self.manifest_edit.text().strip() or self.runtime.project_context.state.dataset_root or str(TOOLBOX_ROOT),
            "YAML (*.yaml *.yml)",
        )
        if path:
            self.manifest_edit.setText(path)

    def _pick_output(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self,
            "选择结果目录",
            self.output_edit.text().strip() or self.runtime.project_context.state.output_root or str(TOOLBOX_ROOT),
        )
        if path:
            self._output_auto = False
            self.output_edit.setText(path)

    def _describe(self) -> None:
        path = self.config_edit.text().strip()
        if not path or not Path(path).is_file():
            return
        try:
            descriptor = describe_config(path)
            slots = "，".join(
                f"{slot.name}（{slot.min_files} 文件）" for slot in descriptor.slots
            )
            self.description_label.setText(
                f"识别结果：{descriptor.project} / {descriptor.product or '未命名'}；"
                f"生产引擎：{descriptor.engine_default}；输入槽：{slots}。"
            )
            self.error_label.setText("")
        except Exception as exc:
            self.description_label.setText("无法识别该产品配置。")
            self.error_label.setText(str(exc))

    def _auto_manifest(self) -> Path | None:
        state = self.runtime.project_context.state
        if not state.cabf_reference_top or not state.cabf_reference_bottom:
            return None
        descriptor = describe_config(self.config_edit.text().strip())
        if descriptor.project != "CAB-F":
            return None
        path = Path(tempfile.gettempdir()) / f"cosmos-pipeline-input-{uuid4().hex}.yaml"
        payload = {
            "color_space": "rgb",
            "cases": [
                {
                    "id": "selected-reference",
                    "inputs": {
                        "top": [state.cabf_reference_top],
                        "bottom": [state.cabf_reference_bottom],
                    },
                }
            ],
        }
        path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
        self._ephemeral_manifest = path
        return path

    def _request(self) -> PipelineRequest:
        output = Path(self.output_edit.text().strip()).expanduser().resolve()
        manifest_text = self.manifest_edit.text().strip()
        manifest = Path(manifest_text).expanduser().resolve() if manifest_text else self._auto_manifest()
        if manifest is None:
            raise ValueError("请选择输入清单；CAB-F 也可以先在配置页生成并保存 TOP/BOTTOM 校准基准图")
        return PipelineRequest(
            Path(self.config_edit.text().strip()).expanduser().resolve(),
            Path(self.backend_edit.text().strip()).expanduser().resolve(),
            manifest,
            output,
            str(self.engine_combo.currentData()),
            self.dry_run_check.isChecked(),
            self.persist_db_check.isChecked(),
            self.fail_on_ng_check.isChecked(),
        )

    def start(self) -> None:
        if self._process is not None:
            return
        try:
            request = self._request()
            descriptor, cases = validate_request(request)
        except Exception as exc:
            if self._ephemeral_manifest is not None:
                self._ephemeral_manifest.unlink(missing_ok=True)
                self._ephemeral_manifest = None
            self._set_status("配置有误", "danger")
            self.error_label.setText(str(exc))
            return
        context_updates = {
            "pipeline_config_path": request.config_path,
            "pipeline_backend_config_path": request.backend_config_path,
        }
        if self._ephemeral_manifest is None:
            context_updates["pipeline_input_manifest_path"] = request.input_manifest_path
        self.runtime.project_context.update(
            **context_updates,
        )
        self.error_label.setText("")
        self.log.clear()
        self.log.appendPlainText(f"{descriptor.project} / {descriptor.product} · {len(cases)} 个测试 case")
        runtime_profile = self.runtime.project_context.state.runtime_profile
        conda_manager = getattr(self.runtime, "conda_manager", None)
        python_executable = (
            conda_manager.python_executable(runtime_profile)
            if conda_manager is not None
            else sys.executable
        )
        self.log.appendPlainText(f"运行环境：{runtime_profile or '当前解释器'} · {python_executable}")
        command = build_pipeline_command(request, python_executable)
        process = QProcess(self)
        process.setWorkingDirectory(str(COSMOS_ROOT))
        environment = QProcessEnvironment.systemEnvironment()
        if conda_manager is not None:
            process_environment = conda_manager.process_environment(runtime_profile)
            for key, value in process_environment.items():
                environment.insert(key, value)
        for key, value in runtime_env().items():
            environment.insert(key, value)
        environment.insert("PYTHONUNBUFFERED", "1")
        process.setProcessEnvironment(environment)
        process.setProgram(command[0])
        process.setArguments(command[1:])
        process.readyReadStandardOutput.connect(self._read_stdout)
        process.readyReadStandardError.connect(self._read_stderr)
        process.finished.connect(self._on_finished)
        process.errorOccurred.connect(self._on_process_error)
        self._process = process
        self._task_id = f"cosmos-pipeline-{uuid4().hex[:12]}"
        self._task_finished = False
        self._cancelling = False
        self._stdout_buffer = ""
        self._stderr_buffer = ""
        output_path = "" if request.dry_run else str(request.output_dir)
        self.runtime.task_center.start(
            self._task_id,
            f"Cosmos 完整流程 · {descriptor.product or descriptor.project}",
            self.capability_key,
            output_path,
            cancel=self._terminate_process,
        )
        self._sync_task_presentation()
        process.start()

    def _read_stdout(self) -> None:
        if self._process is not None:
            chunk = bytes(self._process.readAllStandardOutput()).decode("utf-8", errors="replace")
            self._stdout_buffer = self._consume(self._stdout_buffer + chunk, "stdout")

    def _read_stderr(self) -> None:
        if self._process is not None:
            chunk = bytes(self._process.readAllStandardError()).decode("utf-8", errors="replace")
            self._stderr_buffer = self._consume(self._stderr_buffer + chunk, "stderr")

    def _consume(self, text: str, stream: str) -> str:
        lines = text.splitlines(keepends=True)
        remainder = ""
        if lines and not lines[-1].endswith(("\n", "\r")):
            remainder = lines.pop()
        for line in lines:
            self._append_line(line.rstrip("\r\n"), stream)
        return remainder

    def _append_line(self, line: str, stream: str) -> None:
        if not line:
            return
        display = f"[stderr] {line}" if stream == "stderr" else line
        self.log.appendPlainText(display)
        if self._task_id:
            self.runtime.task_center.log(self._task_id, line, stream)
        if not line.startswith("COSMOS_EVENT "):
            return
        try:
            event = json.loads(line.removeprefix("COSMOS_EVENT "))
        except json.JSONDecodeError:
            return
        if event.get("type") == "progress" and self._task_id:
            self.runtime.task_center.progress(self._task_id, int(event["current"]), int(event["total"]))
        elif event.get("type") == "business_result":
            self._sync_task_presentation(f"{event.get('slot')} {event.get('outcome')}")

    def cancel(self) -> None:
        if not self._task_id:
            return
        self.runtime.task_center.cancel(self._task_id)
        self._sync_task_presentation()

    def _terminate_process(self) -> None:
        process = self._process
        if process is None or process.state() == QProcess.ProcessState.NotRunning:
            return
        self._cancelling = True
        pid = int(process.processId())
        if os.name == "nt" and pid > 0:
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        else:
            process.terminate()

    def _on_process_error(self, error: QProcess.ProcessError) -> None:
        if error == QProcess.ProcessError.FailedToStart:
            self._append_line(f"无法启动：{self._process.errorString() if self._process else error}", "stderr")
            self._finish(TaskStatus(PipelineOutcome.TECHNICAL_FAILURE.value))

    def _on_finished(self, exit_code: int, _status: QProcess.ExitStatus) -> None:
        if self._stdout_buffer:
            self._append_line(self._stdout_buffer, "stdout")
        if self._stderr_buffer:
            self._append_line(self._stderr_buffer, "stderr")
        outcome = outcome_from_process_exit(
            exit_code,
            cancelled=self._cancelling,
            crashed=_status == QProcess.ExitStatus.CrashExit,
        )
        self._finish(TaskStatus(outcome.value))

    def _finish(self, status: TaskStatus) -> None:
        if self._task_finished:
            return
        self._task_finished = True
        if self._task_id:
            self.runtime.task_center.finish(self._task_id, status)
        self._sync_task_presentation()
        process = self._process
        self._process = None
        if process is not None:
            process.deleteLater()
        self._task_id = ""
        if self._ephemeral_manifest is not None:
            self._ephemeral_manifest.unlink(missing_ok=True)
            self._ephemeral_manifest = None

    def shutdown(self) -> None:
        process = self._process
        self.cancel()
        if process is not None and process.state() != QProcess.ProcessState.NotRunning:
            if not process.waitForFinished(5000):
                process.kill()
                process.waitForFinished(2000)
