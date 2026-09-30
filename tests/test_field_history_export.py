import json
import os
import sqlite3
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6.QtWidgets import QApplication, QListWidgetItem, QMessageBox
from PySide6.QtCore import Qt
from cosmos_toolbox.field_history import read_run, jobs_from_records, export_history_jobs
from cosmos_toolbox.field_dataset_ui import DatasetWorker, FieldDatasetPage
from test_field_export import sample


def control():
    return SimpleNamespace(paused=threading.Event(), stopped=threading.Event(), export_on_stop=threading.Event())


def history(root):
    root.mkdir(parents=True)
    folder = sample(root)
    with sqlite3.connect(root/'run.db') as db:
        db.execute('CREATE TABLE run_items(run_id TEXT,item_key TEXT)')
        db.execute("INSERT INTO run_items VALUES ('100','fixture')")
        db.execute("INSERT INTO items SELECT 'other',model,status,details FROM items WHERE key='fixture'")
        db.execute("INSERT INTO run_items VALUES ('101','other')")
    (root/'runs').mkdir()
    snapshot = dict(run_id='100', version='old-algorithm-version', source=str(root/'missing-originals'),
                    product='D01-R', selected=['roi_detector'], mode='auto',
                    models={'product_config_path': str(root/'missing.yaml')})
    path = root/'runs/100.json'
    path.write_text(json.dumps(snapshot), encoding='utf-8')
    return read_run(path), folder


def test_export_only_exact_run_without_models_config_or_originals(tmp_path):
    app = QApplication.instance() or QApplication([])
    record, folder = history(tmp_path/'workspace')
    root = Path(record['workspace'])
    before = {str(path): path.read_bytes() for path in [root/'run.db', Path(record['path']), folder/'image.json']}
    jobs = jobs_from_records([record], None)
    worker = DatasetWorker(dict(export_only=True, jobs=jobs), control(), environment_name='nonexistent-environment')
    results = []
    worker.result.connect(results.append)
    with patch.object(worker, '_run_one', side_effect=AssertionError('must not generate')), \
         patch('cosmos_toolbox.field_dataset_runtime.run_in_environment', side_effect=AssertionError('must not launch GPU')):
        worker.run()
    assert not results[0]['errors']
    report = results[0]['results'][0]['export']
    assert report['samples'] == 1 and report['run_id'] == '100'
    assert report['preview_samples'] == 1 and report['valid']
    assert all(Path(path).read_bytes() == content for path, content in before.items())
    second = export_history_jobs(jobs, control(), lambda fact: None)
    assert second['results'][0]['export']['directory'] != report['directory']


def test_export_pending_empty_missing_isolated(tmp_path):
    good, _ = history(tmp_path/'good')
    broken, _ = history(tmp_path/'broken')
    jobs = jobs_from_records([broken, good], None)
    Path(broken['path']).unlink()
    jobs.append({'options': {'source': 'unstarted', 'output': str(tmp_path/'unstarted')}})
    result = export_history_jobs(jobs, control(), lambda fact: None)
    assert len(result['errors']) == 1
    assert len(result['results']) == 1
    assert len(result['skipped_exports']) == 1
    assert not (tmp_path/'unstarted').exists()


def test_no_complete_rows_does_not_create_export(tmp_path):
    record, _ = history(tmp_path/'empty')
    with sqlite3.connect(Path(record['workspace'])/'run.db') as db:
        db.execute("UPDATE items SET status='failed'")
    result = export_history_jobs(jobs_from_records([record], None), control(), lambda fact: None)
    assert not result['results'] and len(result['skipped_exports']) == 1
    assert not (Path(record['workspace'])/'exports').exists()


def test_stop_between_exports(tmp_path):
    records = [history(tmp_path/name)[0] for name in ('one', 'two')]
    state = control()
    def stop(fact):
        if str(fact.get('message', '')).startswith('导出目录：'):
            state.stopped.set()
    result = export_history_jobs(jobs_from_records(records, None), state, stop)
    assert result['stopped'] and len(result['results']) == 1
    assert not (tmp_path/'two/exports').exists()


def test_active_workspace_rejected(tmp_path):
    import msvcrt
    record, _ = history(tmp_path/'locked')
    with (Path(record['workspace'])/'.running.lock').open('w+b') as lock:
        lock.write(b'0')
        lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            result = export_history_jobs(jobs_from_records([record], None), control(), lambda fact: None)
            assert not result['results']
            assert '正在生成或导出' in result['errors'][0]['error']
        finally:
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)


def test_ui_export_selected_without_restoring_settings(tmp_path):
    app = QApplication.instance() or QApplication([])
    page = FieldDatasetPage()
    record, _ = history(tmp_path/'workspace')
    page.product_config.setText('invalid-config')
    page.source.setPlainText('invalid-source')
    item = QListWidgetItem('history')
    item.setData(Qt.ItemDataRole.UserRole, record)
    page.history_list.addItem(item)
    item.setSelected(True)
    with patch.object(QMessageBox, 'question', return_value=QMessageBox.StandardButton.Yes), \
         patch.object(page, '_launch_worker') as launch:
        page._export_history()
    options, scan_only, environment = launch.call_args.args
    assert options['export_only'] and not scan_only and environment is None
    assert options['jobs'][0]['options']['resume_from'] == record['path']
    page._on_result({'export_only': True, 'results': [{'export': {'directory': 'output', 'samples': 2}}],
                     'errors': [], 'skipped_exports': []})
    assert '样本 2' in page.status.text()
    assert page.tabs.currentWidget() is page.log
    page.close()
