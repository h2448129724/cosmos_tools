import copy
import json
import os
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6.QtWidgets import QApplication, QListWidgetItem, QMessageBox
from PySide6.QtCore import Qt
from cosmos_toolbox.field_dataset_ui import DatasetWorker, FieldDatasetPage
from cosmos_toolbox.field_history import jobs_from_records, new_batch, read_batch, discover_batches


def control():
    return SimpleNamespace(stopped=threading.Event(), paused=threading.Event(), export_on_stop=threading.Event())


def record(tmp_path, name, product='D01-L'):
    return dict(path=str(tmp_path/name/'runs/123.json'), data={'environment': name+'-env'}, missing=[],
        options=dict(source=str(tmp_path/name/'input'), output=str(tmp_path/name/'output'),
            product=product, product_config=None, selected=['roi_detector'], face='all', limit=None,
            mode='auto', ng_only=False, export_scope='current'))


def test_multiselect_keeps_individual_settings_and_rejects_overlap(tmp_path):
    first, second = record(tmp_path, 'one'), record(tmp_path, 'two', 'D01-R')
    jobs = jobs_from_records([first, second], 'fallback')
    assert [job['environment'] for job in jobs] == ['one-env', 'two-env']
    assert [job['options']['product'] for job in jobs] == ['D01-L', 'D01-R']
    assert [job['options']['resume_from'] for job in jobs] == [first['path'], second['path']]
    with pytest.raises(ValueError, match='同一工作区'):
        jobs_from_records([first, copy.deepcopy(first)], 'fallback')


def test_batch_persisted_before_work_stop_retains_pending(tmp_path):
    app = QApplication.instance() or QApplication([])
    state = control()
    worker = DatasetWorker(dict(sources=['one', 'two', 'three'], output=str(tmp_path)), state)
    results = []
    worker.result.connect(results.append)
    def run_one(options):
        path = next((tmp_path/'batches').glob('*.json'))
        saved = json.loads(path.read_text())
        assert len(saved['jobs']) == 3
        assert saved['jobs'][0]['status'] == 'running'
        state.stopped.set()
        return dict(stopped=True, run_id='123', output=options['output'])
    with patch.object(worker, '_run_one', side_effect=run_one) as runner:
        worker.run()
    assert runner.call_count == 1
    manifest = Path(results[0]['batch_manifest'])
    saved = read_batch(manifest)
    assert saved['statuses'] == ['stopped', 'pending', 'pending']
    assert saved['jobs'][0]['options']['resume_from'].endswith('runs\\123.json') or saved['jobs'][0]['options']['resume_from'].endswith('runs/123.json')
    assert 'resume_from' not in saved['jobs'][1]['options']
    assert discover_batches(tmp_path)[0][0]['statuses'] == saved['statuses']


def test_batch_failure_isolated_and_original_manifest_not_rewritten(tmp_path):
    app = QApplication.instance() or QApplication([])
    jobs = [{'options': record(tmp_path, name)['options'], 'environment': name+'-env'} for name in ('one', 'two')]
    path, _ = new_batch(tmp_path, jobs)
    before = path.read_bytes()
    restored = read_batch(path)['jobs']
    worker = DatasetWorker(dict(jobs=restored, output=str(tmp_path)), control())
    calls, results = [], []
    worker.result.connect(results.append)
    def run_one(options):
        calls.append((options['product'], worker.environment_name))
        if len(calls) == 1:
            raise ValueError('incompatible model')
        return dict(completed_model_jobs=1)
    with patch.object(worker, '_run_one', side_effect=run_one):
        worker.run()
    assert calls == [('D01-L', 'one-env'), ('D01-L', 'two-env')]
    assert len(results[0]['errors']) == 1 and len(results[0]['results']) == 1
    assert path.read_bytes() == before


def test_changed_pending_yaml_is_blocked(tmp_path):
    app = QApplication.instance() or QApplication([])
    config = tmp_path/'config.yaml'
    config.write_text('original')
    options = record(tmp_path, 'one')['options']
    options['product_config'] = str(config)
    path, _ = new_batch(tmp_path, [{'options': options, 'environment': 'onnx-gpu'}])
    config.write_text('changed')
    worker = DatasetWorker(dict(jobs=read_batch(path)['jobs'], output=str(tmp_path)), control())
    results = []
    worker.result.connect(results.append)
    with patch.object(worker, '_run_one') as runner:
        worker.run()
    runner.assert_not_called()
    assert '配置文件' in results[0]['errors'][0]['error']


def test_ui_multiselect_recovery_locks_shared_fields(tmp_path):
    app = QApplication.instance() or QApplication([])
    page = FieldDatasetPage()
    page.output.setText(str(tmp_path))
    for name in ('one', 'two'):
        item = QListWidgetItem(name)
        item.setData(Qt.ItemDataRole.UserRole, record(tmp_path, name))
        page.history_list.addItem(item)
        item.setSelected(True)
    with patch.object(QMessageBox, 'question', return_value=QMessageBox.StandardButton.Yes):
        page._restore_history()
    assert len(page._resume_jobs) == 2
    assert not page.product_config.isEnabled()
    assert not page.source.isEnabled()
    assert '2' in page.start_button.text()
    page._new_task()
    assert page._resume_jobs is None
    assert page.source.isEnabled()
    page.close()


def test_real_batch_stop_and_resume_includes_unstarted_folder(tmp_path, monkeypatch):
    import sys
    import cv2
    import numpy as np
    class Models:
        snapshot = {'weights': 'test'}
        calls = 0
        def __init__(self, product):
            self.errors = {}
        def generate(self, image, selected, **kwargs):
            Models.calls += 1
            return {name: [{'image': image, 'shapes': []}] for name in selected}
    monkeypatch.setitem(sys.modules, 'cosmos_toolbox.field_models', SimpleNamespace(FieldModels=Models))
    app = QApplication.instance() or QApplication([])
    sources = []
    for name in ('one', 'two'):
        folder = tmp_path/name
        folder.mkdir()
        cv2.imwrite(str(folder/'sample_top.png'), np.zeros((12, 12, 3), np.uint8))
        sources.append(str(folder))
    output = tmp_path/'output'
    state = control()
    worker = DatasetWorker(dict(sources=sources, output=str(output), product='D01-R', selected=['roi_detector'],
                                face='all', mode='auto', limit=None, ng_only=False, export_scope='current'), state)
    received = []
    worker.result.connect(received.append)
    def stop_after_one(fact):
        if fact.get('completed') == 1:
            state.export_on_stop.set()
            state.stopped.set()
    worker.progress.connect(stop_after_one)
    worker.run()
    batch = read_batch(received[0]['batch_manifest'])
    assert batch['statuses'] == ['stopped', 'pending']
    assert Models.calls == 1
    assert received[0]['results'][0]['export']['samples'] == 1
    resumed = DatasetWorker(dict(jobs=batch['jobs'], output=str(output)), control())
    resumed.result.connect(received.append)
    resumed.run()
    assert not received[-1]['errors']
    assert len(received[-1]['results']) == 2
    assert received[-1]['results'][0]['skipped_model_jobs'] == 1
    assert received[-1]['results'][1]['completed_model_jobs'] == 1
    assert Models.calls == 2


def test_interrupted_batch_recovers_only_unambiguous_created_run(tmp_path):
    import sqlite3
    options = record(tmp_path, 'one')['options']
    workspace = Path(options['output'])
    (workspace/'runs').mkdir(parents=True)
    with sqlite3.connect(workspace/'run.db'):
        pass
    path, batch = new_batch(tmp_path, [{'options': options}])
    batch['jobs'][0].update(status='running', known_runs=[])
    path.write_text(json.dumps(batch))
    snapshot = dict(run_id='123', version='v1', source=options['source'], product=options['product'],
                    selected=options['selected'], mode='auto', options=options)
    (workspace/'runs/123.json').write_text(json.dumps(snapshot))
    assert read_batch(path)['jobs'][0]['options']['resume_from'].endswith('123.json')
    snapshot['run_id'] = '124'
    (workspace/'runs/124.json').write_text(json.dumps(snapshot))
    with pytest.raises(ValueError, match='多个未关联运行'):
        read_batch(path)
