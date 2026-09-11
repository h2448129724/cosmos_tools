from __future__ import annotations

from typing import Callable
from weakref import ref

from PySide6.QtCore import QObject, Signal, Slot

from .task_ledger import TaskLedger, TaskRecord, TaskStatus
from .task_presentation import TaskPresentation, present_task

__all__ = ["TaskCenter", "TaskLedger", "TaskRecord", "TaskPresentation", "TaskStatus"]


class _TrainingTaskBridge(QObject):
    """Queue RunManager facts onto the TaskCenter object's Qt thread."""

    def __init__(
        self,
        center: "TaskCenter",
        manager: QObject,
        key_for: Callable[[object], str],
    ) -> None:
        super().__init__(center)
        self._center_ref = ref(center)
        self._manager_ref = ref(manager)
        self._key_for = key_for

    @Slot(object)
    def started(self, record: object) -> None:
        center = self._center_ref()
        manager = self._manager_ref()
        if center is None or manager is None:
            return
        run_id = str(getattr(record, "run_id"))
        title = (
            f"{getattr(record, 'action_display_name', '运行')} · "
            f"{getattr(record, 'feature_name', '')}"
        )
        stop_run = getattr(manager, "stop_run", None)

        def cancel(target: str = run_id, manager_ref=self._manager_ref) -> None:
            current_manager = manager_ref()
            current_stop = getattr(current_manager, "stop_run", None)
            if callable(current_stop):
                current_stop(target)

        center.start(
            run_id,
            title,
            self._key_for(record),
            str(getattr(record, "artifacts_dir", "") or getattr(record, "output_dir", "")),
            cancel=cancel if callable(stop_run) else None,
        )

    @Slot(str, str, str)
    def logged(self, run_id: str, stream: str, line: str) -> None:
        center = self._center_ref()
        if center is not None:
            center.log(str(run_id), line, stream)

    @Slot(object)
    def finished(self, record: object) -> None:
        center = self._center_ref()
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
        center.finish(str(getattr(record, "run_id")), status, output)


class TaskCenter(QObject):
    """Imperative shell around the pure :class:`TaskLedger`.

    Qt signals and cancellation callbacks are deliberately kept here.  Every
    state mutation replaces the immutable ledger with a new value, so records
    previously handed to consumers remain stable snapshots.
    """

    changed = Signal()
    task_started = Signal(object)
    task_finished = Signal(object)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._ledger = TaskLedger.empty()
        self._training_bridges: dict[int, _TrainingTaskBridge] = {}
        self._cancel_callbacks: dict[str, Callable[[], None]] = {}

    @property
    def tasks(self) -> tuple[TaskRecord, ...]:
        return self._ledger.tasks

    @property
    def active_count(self) -> int:
        return self._ledger.active_count

    @property
    def presentations(self) -> tuple[TaskPresentation, ...]:
        """UI-ready projections in the same newest-first order as ``tasks``."""

        return tuple(present_task(task) for task in self.tasks)

    def get(self, task_id: str) -> TaskRecord | None:
        return self._ledger.get(task_id)

    def presentation(self, task_id: str) -> TaskPresentation | None:
        task = self.get(task_id)
        return present_task(task) if task is not None else None

    def start(
        self,
        task_id: str,
        title: str,
        capability_key: str,
        output_path: str = "",
        cancel: Callable[[], None] | None = None,
    ) -> TaskRecord:
        # A repeated ID replaces the record in-place.  Explicitly clearing the
        # callback when cancel is omitted avoids stale cancellation handles.
        self._cancel_callbacks.pop(task_id, None)
        if cancel is not None:
            self._cancel_callbacks[task_id] = cancel
        self._ledger = self._ledger.start(
            task_id,
            title,
            capability_key,
            output_path,
            cancellable=cancel is not None,
        )
        task = self._ledger.get(task_id)
        assert task is not None
        self.changed.emit()
        self.task_started.emit(task)
        return task

    def cancel(self, task_id: str) -> bool:
        callback = self._cancel_callbacks.get(task_id)
        if not self._ledger.can_cancel(task_id) or callback is None:
            return False
        # Consume the callback and publish CANCELLING before invoking any
        # external effect.  This makes repeated clicks harmless, and lets the
        # UI disable its stop action immediately even if the bridge is slow.
        self._cancel_callbacks.pop(task_id, None)
        self._ledger = self._ledger.request_cancel(task_id)
        self.changed.emit()
        callback()
        return True

    def progress(self, task_id: str, current: int, total: int) -> None:
        if self._ledger.get(task_id) is None:
            return
        self._ledger = self._ledger.progress(task_id, current, total)
        self.changed.emit()

    def log(self, task_id: str, line: str, stream: str = "stdout") -> None:
        if self._ledger.get(task_id) is None:
            return
        self._ledger = self._ledger.log(task_id, line, stream)
        self.changed.emit()

    def finish(self, task_id: str, status: TaskStatus, output_path: str = "") -> None:
        if self._ledger.get(task_id) is None:
            return
        self._ledger = self._ledger.finish(task_id, status, output_path)
        self._cancel_callbacks.pop(task_id, None)
        task = self._ledger.get(task_id)
        assert task is not None
        self.changed.emit()
        self.task_finished.emit(task)

    def bind_training_manager(
        self,
        manager: QObject,
        capability_key: Callable[[object], str] | None = None,
    ) -> None:
        """Adapt trainer RunManager signals into the shared task interface."""

        identity = id(manager)
        if identity in self._training_bridges:
            return
        key_for = capability_key or (lambda record: f"training.{getattr(record, 'feature_name', 'unknown')}")
        bridge = _TrainingTaskBridge(self, manager, key_for)
        self._training_bridges[identity] = bridge
        manager.run_started.connect(bridge.started)
        manager.log_received.connect(bridge.logged)
        manager.run_finished.connect(bridge.finished)
