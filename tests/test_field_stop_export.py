import os
import threading
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6.QtWidgets import QApplication
from cosmos_toolbox.field_dataset_runtime import control_state, apply_control_state
from cosmos_toolbox.field_dataset_ui import FieldDatasetPage, DatasetWorker


def control():
    return SimpleNamespace(paused=threading.Event(), stopped=threading.Event(), export_on_stop=threading.Event())


def test_mailbox_stop_export_and_pause():
    parent, child = control(), control()
    parent.paused.set()
    parent.export_on_stop.set()
    parent.stopped.set()
    apply_control_state(child, control_state(parent))
    assert child.stopped.is_set() and child.export_on_stop.is_set()
    assert not child.paused.is_set()
    legacy = SimpleNamespace(paused=threading.Event(), stopped=threading.Event())
    legacy.stopped.set()
    child = control()
    apply_control_state(child, control_state(legacy))
    assert child.stopped.is_set() and not child.export_on_stop.is_set()


def test_button_unpauses_and_reports_export():
    app = QApplication.instance() or QApplication([])
    page = FieldDatasetPage()
    page._set_busy(True)
    assert page.stop_export_button.isEnabled()
    page.control.paused.set()
    page._stop_and_export()
    assert page.control.export_on_stop.is_set() and page.control.stopped.is_set()
    assert not page.control.paused.is_set()
    assert not page.stop_export_button.isEnabled()
    page._on_progress({'phase': 'export', 'message': '正在导出测试'})
    assert page.status.text() == '正在导出测试'
    page._on_result({'batch': True, 'stopped': True, 'results': [{'export': {'directory': 'test-output'}}]})
    assert '1 个导出目录' in page.status.text()
    assert 'test-output' in page.log.toPlainText()
    page._set_busy(True, scan_only=True)
    assert not page.stop_export_button.isEnabled()
    page.close()


def test_batch_does_not_start_next_folder_after_stop_export(tmp_path):
    app = QApplication.instance() or QApplication([])
    state = control()
    worker = DatasetWorker({'sources': ['one', 'two'], 'output': str(tmp_path/'out')}, state)
    calls, results = [], []
    def run_one(options):
        calls.append(options['source'])
        state.export_on_stop.set()
        state.stopped.set()
        return {'stopped': True, 'export': {'directory': 'partial'}}
    worker.result.connect(results.append)
    with patch.object(worker, '_run_one', side_effect=run_one):
        worker.run()
    assert calls == ['one']
    assert results[0]['stopped']
    assert results[0]['results'][0]['export']['directory'] == 'partial'
