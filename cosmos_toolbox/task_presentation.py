"""Pure, UI-facing projections of task records.

The task ledger owns state transitions while this module owns the vocabulary
and display semantics shared by the task center and embedded workspaces.  It
contains no Qt imports, making the projection straightforward to test and to
reuse from non-Qt renderers.
"""

from __future__ import annotations

from dataclasses import dataclass

from .task_ledger import TaskRecord, TaskStatus

__all__ = ["TaskPresentation", "present_task", "task_presentation", "status_label", "status_tone"]


_STATUS_LABELS: dict[TaskStatus, str] = {
    TaskStatus.PENDING: "等待",
    TaskStatus.RUNNING: "运行中",
    TaskStatus.CANCELLING: "正在停止",
    TaskStatus.SUCCESS: "完成",
    TaskStatus.BUSINESS_NG: "业务 NG",
    TaskStatus.FAILED: "失败",
    TaskStatus.STOPPED: "已停止",
}

_STATUS_TONES: dict[TaskStatus, str] = {
    TaskStatus.PENDING: "neutral",
    TaskStatus.RUNNING: "info",
    TaskStatus.CANCELLING: "warning",
    TaskStatus.SUCCESS: "success",
    TaskStatus.BUSINESS_NG: "warning",
    TaskStatus.FAILED: "danger",
    TaskStatus.STOPPED: "neutral",
}


def status_label(status: TaskStatus) -> str:
    """Return the stable Chinese label for a task status."""

    return _STATUS_LABELS[TaskStatus(status)]


def status_tone(status: TaskStatus) -> str:
    """Return a semantic tone name, independent of a widget toolkit/theme."""

    return _STATUS_TONES[TaskStatus(status)]


@dataclass(frozen=True, slots=True)
class TaskPresentation:
    """Immutable display projection consumed by task UIs.

    ``progress`` is an integer percentage (0--100), or ``None`` when a task
    has no determinate total.  Current/total values remain available for
    renderers that prefer a ``2/8`` label.
    """

    task_id: str
    title: str
    status: TaskStatus
    status_text: str
    tone: str
    progress: int | None
    progress_current: int
    progress_total: int
    progress_text: str
    cancellable: bool
    button_label: str
    summary: str

    @property
    def label(self) -> str:
        """Alias useful to compact list renderers."""

        return self.status_text

    @property
    def status_label(self) -> str:
        return self.status_text

    @property
    def can_cancel(self) -> bool:
        return self.cancellable

    @property
    def cancel_eligible(self) -> bool:
        """Long-form alias used by renderers that expose eligibility."""

        return self.cancellable

    @property
    def active(self) -> bool:
        """Whether the task remains in-flight from a UI perspective."""

        return self.status in (TaskStatus.RUNNING, TaskStatus.CANCELLING)

    @property
    def is_active(self) -> bool:
        """Explicit alias for renderers that prefer predicate naming."""

        return self.active

    @property
    def action_label(self) -> str:
        return self.button_label

    @property
    def progress_value(self) -> int | None:
        """Alias for widget APIs that call the percentage a value."""

        return self.progress

    @property
    def progress_ratio(self) -> float | None:
        if self.progress is None:
            return None
        return self.progress / 100

    @classmethod
    def from_record(cls, task: TaskRecord) -> "TaskPresentation":
        return present_task(task)


def present_task(task: TaskRecord) -> TaskPresentation:
    """Project one immutable ledger record into shared UI vocabulary."""

    status = TaskStatus(task.status)
    current = max(0, int(task.progress_current))
    total = max(0, int(task.progress_total))
    if total:
        progress = min(100, max(0, round(current * 100 / total)))
        progress_text = f"{current}/{total}"
    else:
        progress = None
        progress_text = ""

    label = status_label(status)
    cancellable = bool(status is TaskStatus.RUNNING and task.cancellable)
    button_label = "停止任务" if cancellable else ("正在停止" if status is TaskStatus.CANCELLING else "")
    summary_parts = [str(task.title), label]
    if progress_text:
        summary_parts.append(progress_text)
    return TaskPresentation(
        task_id=str(task.task_id),
        title=str(task.title),
        status=status,
        status_text=label,
        tone=status_tone(status),
        progress=progress,
        progress_current=current,
        progress_total=total,
        progress_text=progress_text,
        cancellable=cancellable,
        button_label=button_label,
        summary=" · ".join(part for part in summary_parts if part),
    )


# Short functional alias for callers that treat projections as a query.
task_presentation = present_task
