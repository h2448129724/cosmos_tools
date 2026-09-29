import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from types import SimpleNamespace
import threading

import pytest
from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QApplication, QWidget

from cosmos_toolbox.task_center import TaskCenter, TaskStatus
from cosmos_toolbox.worker_task_bridge import WorkerTaskBridge


class Worker(QObject):
    progress = Signal(dict)
    result = Signal(dict)
    error = Signal(str)
    finished = Signal()


@pytest.mark.parametrize('result,status', [
    ({'completed_model_jobs': 2}, TaskStatus.SUCCESS),
    ({'batch': True, 'results': [{'failed_model_jobs': 1}]}, TaskStatus.FAILED),
    ({'stopped': True}, TaskStatus.STOPPED),
    ({'failed': 1}, TaskStatus.FAILED),
])
def test_result_is_not_terminal_until_thread_finished(result, status):
    app = QApplication.instance() or QApplication([])
    page, center, worker = QWidget(), TaskCenter(), Worker()
    worker.scan_only = False
    worker.control = SimpleNamespace(stopped=threading.Event())
    bridge = WorkerTaskBridge(page, center, worker, 'test', 'cabf.field_dataset', '', lambda: None)
    worker.progress.emit({'current': 1, 'total': 2, 'message': 'progress'})
    worker.result.emit(result)
    assert center.active_count == 1
    worker.finished.emit()
    assert center.get(bridge.task_id).status == status
    assert center.active_count == 0
    page.close()


def test_cancel_is_cooperative_and_only_once():
    app = QApplication.instance() or QApplication([])
    page, center, worker = QWidget(), TaskCenter(), Worker()
    worker.scan_only = False
    worker.control = SimpleNamespace(stopped=threading.Event())
    calls = []
    bridge = WorkerTaskBridge(page, center, worker, 'test', 'cabf.db_ng_export', '', lambda: calls.append(1))
    assert center.cancel(bridge.task_id)
    assert not center.cancel(bridge.task_id)
    assert calls == [1]
    assert center.get(bridge.task_id).status == TaskStatus.CANCELLING
    worker.result.emit({'stopped': True})
    worker.finished.emit()
    assert center.get(bridge.task_id).status == TaskStatus.STOPPED
    page.close()


def test_artifact_lineage_is_registered_at_actual_finish(tmp_path):
    from cosmos_toolbox.project_session import ProjectSession
    app = QApplication.instance() or QApplication([])
    page, center, worker = QWidget(), TaskCenter(), Worker()
    worker.scan_only = False
    worker.control = SimpleNamespace(stopped=threading.Event())
    worker.options = {'sources': ['source-a']}
    session = ProjectSession(tmp_path / 'session.json')
    export = tmp_path / 'export'
    export.mkdir()
    bridge = WorkerTaskBridge(page, center, worker, 'test', 'cabf.field_dataset', str(tmp_path),
                              lambda: None, session=session)
    worker.result.emit({'run_id': 'run-1', 'export': {'directory': str(export), 'scope': 'current'}})
    assert not session.artifacts
    worker.finished.emit()
    artifact = session.artifacts[0]
    assert artifact.source_task_id == bridge.task_id
    assert artifact.inputs == ('source-a',)
    assert artifact.metadata['run_id'] == 'run-1'
    page.close()
