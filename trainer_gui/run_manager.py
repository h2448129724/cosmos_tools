from __future__ import annotations

import os
import shlex
import subprocess
import threading
from pathlib import Path

from PySide6.QtCore import QObject, Signal

from .history_manager import HistoryManager
from .models import CondaEnvInfo, FeatureAction, FeatureModule, RunRecord

WINDOWS_NO_WINDOW_FLAGS = 0
WINDOWS_STARTUPINFO = None

if os.name == "nt":
    WINDOWS_NO_WINDOW_FLAGS = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startupinfo.wShowWindow = 0
    WINDOWS_STARTUPINFO = startupinfo


class RunManager(QObject):
    log_received = Signal(str, str, str)
    run_started = Signal(object)
    run_finished = Signal(object)

    def __init__(self, project_root: str | Path, history_manager: HistoryManager):
        super().__init__()
        self.project_root = Path(project_root).resolve()
        self.history_manager = history_manager
        self.active_runs: dict[str, subprocess.Popen] = {}
        self.stopped_runs: set[str] = set()

    def start_run(
        self,
        feature: FeatureModule,
        action: FeatureAction,
        project_name: str,
        params: dict,
        conda_env: CondaEnvInfo | None = None,
    ) -> RunRecord:
        path_info = self.history_manager.create_run_paths(feature, action, project_name)
        normalized_project = path_info["project_name"].name
        run_dir = path_info["run_dir"]
        artifacts_dir = path_info["artifacts_dir"]

        params = dict(params)
        effective_artifacts_dir = artifacts_dir
        if action.output.arg_name:
            primary_artifacts_dir = self._resolve_output_target(
                action=action,
                arg_name=action.output.arg_name,
                params=params,
                artifacts_dir=artifacts_dir,
                default_file_name=action.output.default_file_name,
                use_artifacts_subdir=action.output.use_artifacts_subdir,
            )
            if primary_artifacts_dir is not None:
                effective_artifacts_dir = primary_artifacts_dir
        for extra_output in action.output.extra_outputs:
            self._resolve_output_target(
                action=action,
                arg_name=extra_output.arg_name,
                params=params,
                artifacts_dir=artifacts_dir,
                default_file_name=extra_output.default_file_name,
                use_artifacts_subdir=action.output.use_artifacts_subdir,
            )

        command, cwd, process_env, python_executable = self._build_command(feature, action, params, conda_env)
        record = self.history_manager.init_run_record(
            feature=feature,
            action=action,
            project_name=normalized_project,
            command=command,
            cwd=cwd,
            run_dir=run_dir,
            artifacts_dir=effective_artifacts_dir,
            conda_env_name=conda_env.name if conda_env else None,
            conda_prefix=conda_env.prefix if conda_env else None,
            python_executable=python_executable,
        )
        self.history_manager.write_config(
            record.config_path,
            {
                "project_name": normalized_project,
                "feature_name": feature.feature_name,
                "action_name": action.action_name,
                "action_display_name": action.display_name,
                "conda_env_name": conda_env.name if conda_env else None,
                "conda_prefix": conda_env.prefix if conda_env else None,
                "python_executable": python_executable,
                "params": params,
            },
        )
        self.history_manager.write_meta(record)

        stdout_path = Path(record.stdout_log)
        stderr_path = Path(record.stderr_log)
        stdout_handle = stdout_path.open("w", encoding="utf-8")
        stderr_handle = stderr_path.open("w", encoding="utf-8")

        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            env=process_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            startupinfo=WINDOWS_STARTUPINFO,
            creationflags=WINDOWS_NO_WINDOW_FLAGS,
        )
        self.active_runs[record.run_id] = process
        self.run_started.emit(record)

        self._start_stream_thread(process.stdout, stdout_handle, record.run_id, "stdout")
        self._start_stream_thread(process.stderr, stderr_handle, record.run_id, "stderr")

        watcher = threading.Thread(
            target=self._wait_for_process,
            args=(record, process, stdout_handle, stderr_handle),
            daemon=True,
        )
        watcher.start()
        return record

    def stop_run(self, run_id: str) -> None:
        process = self.active_runs.get(run_id)
        if process and process.poll() is None:
            self.stopped_runs.add(run_id)
            self._terminate_process_tree(process)

    def shutdown(self) -> None:
        """Request cancellation of every process still owned by the manager."""
        for run_id in tuple(self.active_runs):
            self.stop_run(run_id)

    @staticmethod
    def _terminate_process_tree(process: subprocess.Popen) -> None:
        """Terminate descendants on Windows and safely fall back elsewhere."""
        if os.name == "nt":
            try:
                result = subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=WINDOWS_NO_WINDOW_FLAGS,
                )
                if result.returncode == 0:
                    return
            except OSError:
                pass
        process.terminate()

    def build_export_command_text(
        self,
        feature: FeatureModule,
        action: FeatureAction,
        params: dict,
        conda_env: CondaEnvInfo | None = None,
        target_platform: str = "linux",
    ) -> str:
        if feature.feature_name == "yolo" and action.action_name in ("train", "predict", "export_onnx"):
            return self._build_yolo_export_command_text(action, params, conda_env, target_platform)

        command = self._build_export_command_args(feature, action, params)
        lines: list[str] = []
        self._append_activate_line(lines, conda_env, target_platform)
        command_text = self._format_command(command, target_platform)

        if action.entry.type == "module":
            pythonpath = self._format_pythonpath_for_export(target_platform)
            if target_platform == "windows":
                lines.append(f"$env:PYTHONPATH = '{pythonpath}'")
                lines.append(command_text)
            else:
                lines.append(f"PYTHONPATH={pythonpath} {command_text}")
        else:
            module_rel_path = feature.module_dir.relative_to(self.project_root)
            if target_platform == "windows":
                lines.append(f"Set-Location {self._quote_windows(module_rel_path.as_posix().replace('/', chr(92)))}")
            else:
                lines.append(f"cd {shlex.quote(module_rel_path.as_posix())}")
            lines.append(command_text)
        return "\n".join(lines)

    def _build_yolo_export_command_text(
        self,
        action: FeatureAction,
        params: dict,
        conda_env: CondaEnvInfo | None,
        target_platform: str,
    ) -> str:
        lines: list[str] = []
        self._append_activate_line(lines, conda_env, target_platform)

        if action.action_name == "train":
            task = str(params.get("task") or "detect")
            model_name = self._resolve_yolo_model_name(params)
            command = [
                "yolo",
                task,
                "train",
                f"data={params.get('data')}",
                f"model={model_name}",
            ]
            for name in ["epochs", "imgsz", "batch", "workers", "patience", "seed"]:
                value = params.get(name)
                if value not in (None, ""):
                    command.append(f"{name}={value}")

            if params.get("device") not in (None, ""):
                command.append(f"device={params.get('device')}")
            if params.get("cache"):
                command.append("cache=True")
            if params.get("amp"):
                command.append("amp=True")

            save_dir = str(params.get("save_dir", "") or "").strip()
            run_name = str(params.get("run_name", "") or "").strip()
            if save_dir:
                save_path = Path(save_dir)
                project = save_path.parent.as_posix()
                name = run_name or save_path.name
                command.append(f"project={project}")
                command.append(f"name={name}")
            elif run_name:
                command.append(f"name={run_name}")

            lines.append(self._format_command(command, target_platform))
            return "\n".join(lines)

        if action.action_name == "export_onnx":
            command = [
                "yolo",
                "export",
                f"model={params.get('model')}",
                "format=onnx",
            ]
            for name in ["imgsz", "opset"]:
                value = params.get(name)
                if value not in (None, ""):
                    command.append(f"{name}={value}")
            if params.get("simplify"):
                command.append("simplify=True")
            if params.get("dynamic"):
                command.append("dynamic=True")
            if params.get("half"):
                command.append("half=True")
            if params.get("device") not in (None, ""):
                command.append(f"device={params.get('device')}")

            output = str(params.get("output", "") or "").strip()
            if output:
                output_path = Path(output)
                command.append(f"project={output_path.parent.as_posix()}")
                command.append(f"name={output_path.stem}")

            lines.append(self._format_command(command, target_platform))
            return "\n".join(lines)

        if action.action_name == "predict":
            task = str(params.get("task") or "detect")
            command = [
                "yolo",
                task,
                "predict",
                f"model={params.get('model')}",
                f"source={params.get('source')}",
            ]
            for name in ["imgsz", "conf", "iou"]:
                value = params.get(name)
                if value not in (None, ""):
                    command.append(f"{name}={value}")
            if params.get("device") not in (None, ""):
                command.append(f"device={params.get('device')}")
            if "save" in params:
                command.append(f"save={bool(params.get('save'))}")
            if params.get("save_txt"):
                command.append("save_txt=True")
            if params.get("save_conf"):
                command.append("save_conf=True")
            if params.get("save_crop"):
                command.append("save_crop=True")

            output_dir = str(params.get("output_dir", "") or "").strip()
            if output_dir:
                output_path = Path(output_dir)
                command.append(f"project={output_path.parent.as_posix()}")
                command.append(f"name={output_path.name}")

            lines.append(self._format_command(command, target_platform))
            return "\n".join(lines)

        return "\n".join(lines)

    def _build_command(
        self,
        feature: FeatureModule,
        action: FeatureAction,
        params: dict,
        conda_env: CondaEnvInfo | None,
    ) -> tuple[list[str], Path, dict, str]:
        process_env = self._build_process_env(conda_env, self.project_root)
        if feature.feature_name == "yolo" and action.action_name in ("train", "predict"):
            command = self._build_yolo_runtime_command(action, params)
            return command, self.project_root, process_env, command[0]

        python_executable = conda_env.python_executable if conda_env else "python"
        if action.entry.type == "module":
            command = [python_executable, "-m", action.entry.value]
            cwd = self.project_root
        else:
            command = [python_executable, action.entry.value]
            cwd = feature.module_dir

        for field in action.schema:
            if field.name not in params:
                continue
            value = params[field.name]
            if field.action == "store_true":
                if value:
                    command.append(field.cli_flag)
                continue
            if field.action == "store_false":
                if not value:
                    command.append(field.cli_flag)
                continue
            if value in (None, ""):
                continue
            command.extend([field.cli_flag, str(value)])
        return command, cwd, process_env, python_executable

    @staticmethod
    def _build_export_command_args(feature: FeatureModule, action: FeatureAction, params: dict) -> list[str]:
        if action.entry.type == "module":
            command = ["python", "-m", action.entry.value]
        else:
            command = ["python", action.entry.value]

        for field in action.schema:
            if field.name not in params:
                continue
            value = params[field.name]
            if field.action == "store_true":
                if value:
                    command.append(field.cli_flag)
                continue
            if field.action == "store_false":
                if not value:
                    command.append(field.cli_flag)
                continue
            if value in (None, ""):
                continue
            command.extend([field.cli_flag, str(value)])
        return command

    def _build_yolo_runtime_command(self, action: FeatureAction, params: dict) -> list[str]:
        if action.action_name == "train":
            task = str(params.get("task") or "detect")
            model_name = self._resolve_yolo_model_name(params)
            command = [
                "yolo",
                task,
                "train",
                f"data={params.get('data')}",
                f"model={model_name}",
            ]
            for name in ["epochs", "imgsz", "batch", "workers", "patience", "seed"]:
                value = params.get(name)
                if value not in (None, ""):
                    command.append(f"{name}={value}")
            if params.get("device") not in (None, ""):
                command.append(f"device={params.get('device')}")
            if params.get("cache"):
                command.append("cache=True")
            if params.get("amp"):
                command.append("amp=True")

            save_dir = str(params.get("save_dir", "") or "").strip()
            run_name = str(params.get("run_name", "") or "").strip()
            if save_dir:
                save_path = Path(save_dir)
                command.append(f"project={save_path.parent}")
                command.append(f"name={run_name or save_path.name}")
            elif run_name:
                command.append(f"name={run_name}")
            return command

        if action.action_name == "export_onnx":
            command = [
                "yolo",
                "export",
                f"model={params.get('model')}",
                "format=onnx",
            ]
            for name in ["imgsz", "opset"]:
                value = params.get(name)
                if value not in (None, ""):
                    command.append(f"{name}={value}")
            if params.get("simplify"):
                command.append("simplify=True")
            if params.get("dynamic"):
                command.append("dynamic=True")
            if params.get("half"):
                command.append("half=True")
            if params.get("device") not in (None, ""):
                command.append(f"device={params.get('device')}")

            output = str(params.get("output", "") or "").strip()
            if output:
                output_path = Path(output)
                command.append(f"project={output_path.parent}")
                command.append(f"name={output_path.stem}")
            return command

        if action.action_name == "predict":
            task = str(params.get("task") or "detect")
            command = [
                "yolo",
                task,
                "predict",
                f"model={params.get('model')}",
                f"source={params.get('source')}",
            ]
            for name in ["imgsz", "conf", "iou"]:
                value = params.get(name)
                if value not in (None, ""):
                    command.append(f"{name}={value}")
            if params.get("device") not in (None, ""):
                command.append(f"device={params.get('device')}")
            if "save" in params:
                command.append(f"save={bool(params.get('save'))}")
            if params.get("save_txt"):
                command.append("save_txt=True")
            if params.get("save_conf"):
                command.append("save_conf=True")
            if params.get("save_crop"):
                command.append("save_crop=True")

            output_dir = str(params.get("output_dir", "") or "").strip()
            if output_dir:
                output_path = Path(output_dir)
                command.append(f"project={output_path.parent}")
                command.append(f"name={output_path.name}")
            return command

        return ["yolo"]

    @staticmethod
    def _is_directory_output_arg(arg_name: str) -> bool:
        return arg_name.endswith("_dir") or arg_name in {"save_dir", "output_dir", "out_dir"}

    def _resolve_output_target(
        self,
        action: FeatureAction,
        arg_name: str,
        params: dict,
        artifacts_dir: Path,
        default_file_name: str | None,
        use_artifacts_subdir: bool,
    ) -> Path | None:
        user_value = str(params.get(arg_name, "") or "").strip()
        field = next((item for item in action.schema if item.name == arg_name), None)
        is_directory = self._is_directory_output_target(arg_name, field.path_mode if field else None)
        force_managed_output = bool(field and field.hidden)

        if force_managed_output:
            user_value = ""
            params[arg_name] = ""

        if user_value:
            output_path = Path(user_value).expanduser().resolve()
            if is_directory:
                output_path.mkdir(parents=True, exist_ok=True)
                params[arg_name] = str(output_path)
                return output_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            params[arg_name] = str(output_path)
            return output_path.parent

        if not use_artifacts_subdir:
            return None

        if is_directory:
            params[arg_name] = str(artifacts_dir)
            return artifacts_dir

        if not default_file_name:
            return None

        default_path = artifacts_dir / default_file_name
        default_path.parent.mkdir(parents=True, exist_ok=True)
        params[arg_name] = str(default_path)
        return default_path.parent

    @staticmethod
    def _is_directory_output_target(arg_name: str, path_mode: str | None) -> bool:
        if path_mode == "dir":
            return True
        if path_mode in {"save_file", "open_file"}:
            return False
        return RunManager._is_directory_output_arg(arg_name)

    @staticmethod
    def _resolve_yolo_model_name(params: dict) -> str:
        custom_model = str(params.get("custom_model", "") or "").strip()
        if custom_model:
            return custom_model

        task = str(params.get("task") or "detect")
        series = str(params.get("model_series") or "yolo11")
        size = str(params.get("model_size") or "n")
        base = f"{series}{size}"
        if task == "segment":
            return f"{base}-seg.pt"
        if task == "classify":
            return f"{base}-cls.pt"
        if task == "pose":
            return f"{base}-pose.pt"
        return f"{base}.pt"

    @staticmethod
    def _format_command(command: list[str], target_platform: str) -> str:
        if target_platform == "windows":
            return subprocess.list2cmdline(command)
        return " ".join(shlex.quote(part) for part in command)

    @staticmethod
    def _quote_windows(text: str) -> str:
        return subprocess.list2cmdline([text])

    def _append_activate_line(
        self,
        lines: list[str],
        conda_env: CondaEnvInfo | None,
        target_platform: str,
    ) -> None:
        if conda_env is None:
            return
        if target_platform == "windows":
            lines.append(f"conda activate {self._quote_windows(conda_env.name)}")
        else:
            lines.append(f"conda activate {shlex.quote(conda_env.name)}")

    @staticmethod
    def _module_pythonpath_entries(project_root: Path) -> list[Path]:
        return [
            project_root / "modules",
            project_root / "shared" / "cabf_common",
        ]

    def _format_pythonpath_for_export(self, target_platform: str) -> str:
        entries = [
            entry.relative_to(self.project_root).as_posix()
            for entry in self._module_pythonpath_entries(self.project_root)
        ]
        if target_platform == "windows":
            return ";".join(entry.replace("/", "\\") for entry in entries)
        return ":".join(entries)

    @staticmethod
    def _build_process_env(conda_env: CondaEnvInfo | None, project_root: Path | None = None) -> dict:
        env = dict(os.environ)
        if project_root is not None:
            pythonpath_entries = [
                str(entry.resolve())
                for entry in RunManager._module_pythonpath_entries(project_root)
            ]
            current_pythonpath = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = os.pathsep.join(part for part in [*pythonpath_entries, current_pythonpath] if part)

        if conda_env is None:
            return env

        prefix = Path(conda_env.prefix)
        path_parts = [
            str(prefix),
            str(prefix / "Scripts"),
            str(prefix / "Library" / "bin"),
            env.get("PATH", ""),
        ]
        env["PATH"] = ";".join(part for part in path_parts if part)
        env["CONDA_DEFAULT_ENV"] = conda_env.name
        env["CONDA_PREFIX"] = conda_env.prefix
        return env

    def _start_stream_thread(self, stream, handle, run_id: str, stream_name: str) -> None:
        thread = threading.Thread(
            target=self._stream_reader,
            args=(stream, handle, run_id, stream_name),
            daemon=True,
        )
        thread.start()

    def _stream_reader(self, stream, handle, run_id: str, stream_name: str) -> None:
        try:
            for line in iter(stream.readline, ""):
                handle.write(line)
                handle.flush()
                self.log_received.emit(run_id, stream_name, line.rstrip("\n"))
        finally:
            stream.close()

    def _wait_for_process(
        self,
        record: RunRecord,
        process: subprocess.Popen,
        stdout_handle,
        stderr_handle,
    ) -> None:
        exit_code = process.wait()
        stdout_handle.close()
        stderr_handle.close()
        self.active_runs.pop(record.run_id, None)
        if record.run_id in self.stopped_runs:
            self.stopped_runs.discard(record.run_id)
            status = "stopped"
        else:
            status = "success" if exit_code == 0 else "failed"
        final_record = self.history_manager.finalize_run(record, status=status, exit_code=exit_code)
        self.run_finished.emit(final_record)
