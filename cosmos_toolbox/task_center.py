from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Callable
from weakref import ref

from PySide6.QtCore import QObject, Signal


class TaskStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    BUSINESS_NG = "business_ng"
    FAILED = "failed"
    STOPPED = "stopped"


@dataclass(slots=True)
class TaskRecord:
    task_id: str
    title: str
    capability_key: str
    status: TaskStatus = TaskStatus.PENDING
    progress_current: int = 0
    progress_total: int = 0
    output_path: str = ""
    logs: list[str] = field(default_factory=list)
    cancellable: bool = False

    @property
    def progress_text(self) -> str:
        if self.progress_total > 0:
            return f"{self.progress_current}/{self.progress_total}"
        return ""


class TaskCenter(QObject):
    """One task interface for progress, logs, history, cancellation, and artifacts."""

    changed = Signal()
    task_started = Signal(object)
    task_finished = Signal(object)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._tasks: dict[str, TaskRecord] = {}
        self._bound_managers: set[int] = set()
        self._cancel_callbacks: dict[str, Callable[[], None]] = {}

    @property
    def tasks(self) -> tuple[TaskRecord, ...]:
        return tuple(reversed(tuple(self._tasks.values())))

    @property
    def active_count(self) -> int:
        return sum(item.status == TaskStatus.RUNNING for item in self._tasks.values())

    def get(self, task_id: str) -> TaskRecord | None:
        return self._tasks.get(task_id)

    def start(
        self,
        task_id: str,
        title: str,
        capability_key: str,
        output_path: str = "",
        cancel: Callable[[], None] | None = None,
    ) -> TaskRecord:
        task = TaskRecord(
            task_id,
            title,
            capability_key,
            TaskStatus.RUNNING,
            output_path=output_path,
            cancellable=cancel is not None,
        )
        self._tasks[task_id] = task
        if cancel is not None:
            self._cancel_callbacks[task_id] = cancel
        self.changed.emit()
        self.task_started.emit(task)
        return task

    def cancel(self, task_id: str) -> bool:
        task = self._tasks.get(task_id)
        callback = self._cancel_callbacks.get(task_id)
        if task is None or task.status != TaskStatus.RUNNING or callback is None:
            return False
        callback()
        return True

    def progress(self, task_id: str, current: int, total: int) -> None:
        task = self._tasks.get(task_id)
        if task is None:
            return
        task.progress_current = max(0, int(current))
        task.progress_total = max(0, int(total))
        self.changed.emit()

    def log(self, task_id: str, line: str, stream: str = "stdout") -> None:
        task = self._tasks.get(task_id)
        if task is None:
            return
        prefix = "[stderr] " if stream == "stderr" else ""
        task.logs.append(prefix + line.rstrip())
        if len(task.logs) > 2000:
            del task.logs[:-2000]
        self.changed.emit()

    def finish(self, task_id: str, status: TaskStatus, output_path: str = "") -> None:
        task = self._tasks.get(task_id)
        if task is None:
            return
        task.status = status
        task.cancellable = False
        self._cancel_callbacks.pop(task_id, None)
        if output_path:
            task.output_path = output_path
        self.changed.emit()
        self.task_finished.emit(task)

    def bind_training_manager(
        self,
        manager: QObject,
        capability_key: Callable[[object], str] | None = None,
    ) -> None:
        """Adapt trainer RunManager signals into the shared task interface."""

        identity = id(manager)
        if identity in self._bound_managers:
            return
        self._bound_managers.add(identity)
        key_for = capability_key or (lambda record: f"training.{getattr(record, 'feature_name', 'unknown')}")
        center_ref = ref(self)

        def started(record) -> None:
            center = center_ref()
            if center is None:
                return
            run_id = str(record.run_id)
            title = f"{getattr(record, 'action_display_name', '运行')} · {getattr(record, 'feature_name', '')}"
            stop_run = getattr(manager, "stop_run", None)
            cancel = (lambda target=run_id: stop_run(target)) if callable(stop_run) else None
            center.start(
                run_id,
                title,
                key_for(record),
                str(getattr(record, "artifacts_dir", "") or getattr(record, "output_dir", "")),
                cancel=cancel,
            )

        def logged(run_id: str, stream: str, line: str) -> None:
            center = center_ref()
            if center is not None:
                center.log(str(run_id), line, stream)

        def finished(record) -> None:
            center = center_ref()
            if center is None:
                return
            raw_status = str(getattr(record, "status", "failed"))
            status = {
                "success": TaskStatus.SUCCESS,
                "business_ng": TaskStatus.BUSINESS_NG,
                "stopped": TaskStatus.STOPPED,
                "failed": TaskStatus.FAILED,
            }.get(raw_status, TaskStatus.FAILED)
            output = str(getattr(record, "artifacts_dir", "") or getattr(record, "output_dir", ""))
            center.finish(str(record.run_id), status, output)

        manager.run_started.connect(started)
        manager.log_received.connect(logged)
        manager.run_finished.connect(finished)
