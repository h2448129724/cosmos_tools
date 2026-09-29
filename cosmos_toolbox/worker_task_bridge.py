"""Publish page worker facts to the existing task center on the Qt UI thread."""
from __future__ import annotations

import json
from uuid import uuid4
from pathlib import Path

from PySide6.QtCore import QObject, Slot

from .task_center import TaskStatus


class WorkerTaskBridge(QObject):
    def __init__(self, page, center, worker, title, capability, output, stop, session=None):
        super().__init__(page)
        self.center = center
        self.task_id = f'{capability}-{uuid4().hex[:12]}'
        self.control = worker.control
        self.result = None
        self.error = None
        self.output = output if not worker.scan_only else ''
        self.session = session
        self.title, self.capability = title, capability
        options = getattr(worker, 'options', {})
        self.inputs = tuple(options.get('sources') or [options.get('source') or options.get('database')
                                                     or options.get('plan', {}).get('database') or ''])
        center.start(self.task_id, title, capability, self.output,
                     cancel=stop if not worker.scan_only else None)
        worker.progress.connect(self.progress)
        worker.result.connect(self.received_result)
        worker.error.connect(self.received_error)
        worker.finished.connect(self.finished)

    @Slot(dict)
    def progress(self, fact):
        current = fact.get('current', fact.get('done', fact.get('processed', 0)))
        total = fact.get('total', 0)
        if isinstance(current, int) and isinstance(total, int) and total > 0:
            self.center.progress(self.task_id, current, total)
        self.center.log(self.task_id, str(fact.get('message') or json.dumps(fact, ensure_ascii=False)))

    @Slot(dict)
    def received_result(self, result):
        self.result = result
        self.center.log(self.task_id, json.dumps(result, ensure_ascii=False, default=str))

    @Slot(str)
    def received_error(self, message):
        self.error = message
        self.center.log(self.task_id, message, 'stderr')

    def note(self, message):
        self.center.log(self.task_id, message)

    @Slot()
    def finished(self):
        # Business result arrives before QThread.finished. Do not publish a
        # terminal task until the worker has actually stopped using resources.
        result = self.result or {}
        runs = result.get('results', []) if result.get('batch') else [result]
        failed = bool(self.error or self.result is None or result.get('errors'))
        failed = failed or any(run.get('failed_model_jobs', 0) or run.get('failed', 0)
                               or run.get('export', {}).get('errors')
                               or run.get('export', {}).get('preview_errors') for run in runs)
        status = TaskStatus.FAILED if failed else (
            TaskStatus.STOPPED if result.get('stopped') or self.control.stopped.is_set() else TaskStatus.SUCCESS)
        if self.session is not None and self.output:
            from .project_session import ArtifactKind
            for run in runs:
                path = run.get('export', {}).get('directory') or run.get('output')
                if path and Path(path).is_dir():
                    try:
                        self.session.register_artifact(
                            kind=ArtifactKind.EXPORT if run.get('export') else ArtifactKind.RUN_OUTPUT,
                            name=self.title, path=path, source_capability=self.capability,
                            source_task_id=self.task_id, inputs=tuple(p for p in self.inputs if p),
                            metadata={'run_id': run.get('run_id'), 'status': status.value,
                                      'scope': run.get('export', {}).get('scope')})
                    except Exception as exc:
                        self.center.log(self.task_id, f'产物登记失败（输出已保留）：{exc}', 'stderr')
        self.center.finish(self.task_id, status, self.output)


def bind_worker_task(page, title, capability):
    center = getattr(page, 'task_center', None)
    if center is None:
        return
    previous = getattr(page, '_task_bridge', None)
    if previous is not None:
        previous.deleteLater()
    page._task_bridge = WorkerTaskBridge(page, center, page.worker, title, capability,
                                         page.output.text().strip(), page._stop,
                                         session=getattr(page, 'project_session', None))
