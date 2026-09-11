"""Pure task state transitions used by :mod:`task_center`.

The ledger deliberately contains only values.  It has no knowledge of Qt,
threads, processes, clocks, or persistence; the TaskCenter shell owns those
effects and replaces its ledger after each transition.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Iterable


class TaskStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    CANCELLING = "cancelling"
    SUCCESS = "success"
    BUSINESS_NG = "business_ng"
    FAILED = "failed"
    STOPPED = "stopped"


@dataclass(frozen=True, slots=True, init=False)
class TaskRecord:
    """Immutable snapshot of one task.

    Logs are stored as a tuple so a snapshot remains stable after a later
    transition.  ``logs`` returns a fresh list for compatibility with the
    original mutable record API.
    """

    task_id: str
    title: str
    capability_key: str
    status: TaskStatus = TaskStatus.PENDING
    progress_current: int = 0
    progress_total: int = 0
    output_path: str = ""
    _logs: tuple[str, ...] = ()
    cancellable: bool = False

    def __init__(
        self,
        task_id: str,
        title: str,
        capability_key: str,
        status: TaskStatus = TaskStatus.PENDING,
        progress_current: int = 0,
        progress_total: int = 0,
        output_path: str = "",
        logs: Iterable[str] = (),
        cancellable: bool = False,
    ) -> None:
        # Keep the original positional order and ``logs=`` keyword while
        # normalising the value-object internals.
        object.__setattr__(self, "task_id", task_id)
        object.__setattr__(self, "title", title)
        object.__setattr__(self, "capability_key", capability_key)
        object.__setattr__(self, "status", TaskStatus(status))
        object.__setattr__(self, "progress_current", progress_current)
        object.__setattr__(self, "progress_total", progress_total)
        object.__setattr__(self, "output_path", output_path)
        object.__setattr__(self, "_logs", tuple(logs))
        object.__setattr__(self, "cancellable", cancellable)

    @property
    def logs(self) -> list[str]:
        return list(self._logs)

    @property
    def progress_text(self) -> str:
        if self.progress_total > 0:
            return f"{self.progress_current}/{self.progress_total}"
        return ""

    @classmethod
    def create(
        cls,
        task_id: str,
        title: str,
        capability_key: str,
        *,
        status: TaskStatus = TaskStatus.PENDING,
        progress_current: int = 0,
        progress_total: int = 0,
        output_path: str = "",
        logs: Iterable[str] = (),
        cancellable: bool = False,
    ) -> "TaskRecord":
        return cls(
            task_id=str(task_id),
            title=str(title),
            capability_key=str(capability_key),
            status=TaskStatus(status),
            progress_current=int(progress_current),
            progress_total=int(progress_total),
            output_path=str(output_path),
            logs=logs,
            cancellable=bool(cancellable),
        )


@dataclass(frozen=True, slots=True)
class TaskLedger:
    """Persistent value object containing task snapshots in insertion order."""

    _records: tuple[TaskRecord, ...] = ()

    @classmethod
    def empty(cls) -> "TaskLedger":
        return cls()

    @property
    def tasks(self) -> tuple[TaskRecord, ...]:
        # The UI historically displayed newest tasks first.
        return tuple(reversed(self._records))

    @property
    def active_count(self) -> int:
        return sum(record.status in (TaskStatus.RUNNING, TaskStatus.CANCELLING) for record in self._records)

    def get(self, task_id: str) -> TaskRecord | None:
        for record in self._records:
            if record.task_id == task_id:
                return record
        return None

    def can_cancel(self, task_id: str) -> bool:
        """Return whether a task is eligible for a shell cancellation effect."""

        task = self.get(task_id)
        return task is not None and task.status is TaskStatus.RUNNING and task.cancellable

    def request_cancel(self, task_id: str) -> "TaskLedger":
        """Move a cancellable running task into its stopping state.

        The transition is deliberately separate from the cancellation effect:
        the imperative shell can publish this state before invoking a process
        or thread stop callback.  A second request is therefore a no-op.
        """

        task = self.get(task_id)
        if task is None or task.status is not TaskStatus.RUNNING or not task.cancellable:
            return self
        return self._replace(
            TaskRecord(
                task.task_id,
                task.title,
                task.capability_key,
                TaskStatus.CANCELLING,
                task.progress_current,
                task.progress_total,
                task.output_path,
                task._logs,
                False,
            )
        )

    # ``cancel`` is kept as a compact query/transition spelling for callers
    # that use the ledger directly; the shell's ``TaskCenter.cancel`` remains
    # the operation that performs the external effect.
    def cancel(self, task_id: str) -> "TaskLedger":
        return self.request_cancel(task_id)

    def _replace(self, record: TaskRecord) -> "TaskLedger":
        for index, current in enumerate(self._records):
            if current.task_id == record.task_id:
                values = self._records[:index] + (record,) + self._records[index + 1 :]
                return TaskLedger(values)
        return TaskLedger(self._records + (record,))

    def start(
        self,
        task_id: str,
        title: str,
        capability_key: str,
        output_path: str = "",
        *,
        cancellable: bool = False,
    ) -> "TaskLedger":
        return self._replace(
            TaskRecord.create(
                task_id,
                title,
                capability_key,
                status=TaskStatus.RUNNING,
                output_path=output_path,
                cancellable=cancellable,
            )
        )

    def progress(self, task_id: str, current: int, total: int) -> "TaskLedger":
        task = self.get(task_id)
        if task is None:
            return self
        updated = TaskRecord(
            task.task_id,
            task.title,
            task.capability_key,
            task.status,
            max(0, int(current)),
            max(0, int(total)),
            task.output_path,
            task._logs,
            task.cancellable,
        )
        return self._replace(updated)

    def log(self, task_id: str, line: str, stream: str = "stdout") -> "TaskLedger":
        task = self.get(task_id)
        if task is None:
            return self
        prefix = "[stderr] " if stream == "stderr" else ""
        logs = task._logs + (prefix + line.rstrip(),)
        if len(logs) > 2000:
            logs = logs[-2000:]
        updated = TaskRecord(
            task.task_id,
            task.title,
            task.capability_key,
            task.status,
            task.progress_current,
            task.progress_total,
            task.output_path,
            logs,
            task.cancellable,
        )
        return self._replace(updated)

    def finish(
        self,
        task_id: str,
        status: TaskStatus,
        output_path: str = "",
    ) -> "TaskLedger":
        task = self.get(task_id)
        if task is None:
            return self
        updated = TaskRecord(
            task.task_id,
            task.title,
            task.capability_key,
            TaskStatus(status),
            task.progress_current,
            task.progress_total,
            output_path if output_path else task.output_path,
            task._logs,
            False,
        )
        return self._replace(updated)
