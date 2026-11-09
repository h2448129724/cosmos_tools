"""Exercise the native page and the complete DB → Excel → originals workflow."""
import os
import sqlite3
import time
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import pytest
from PySide6.QtCore import QEvent
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QApplication, QWidget

from cosmos_toolbox.capability_catalog import plan_default_capabilities
from cosmos_toolbox.data_transfer_ui import DataTransferPage, DataTransferWorker
from cosmos_toolbox.default_capabilities import _page_factory_for
from cosmos_toolbox.task_center import TaskCenter, TaskStatus


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def database(tmp_path):
    path = tmp_path / 'cosmos.db'
    with sqlite3.connect(path) as db:
        db.executescript('''
            CREATE TABLE inspection_result (id INTEGER, ok INTEGER, create_time TEXT);
            CREATE TABLE image_file (id INTEGER, result_id INTEGER, filename TEXT);
            CREATE TABLE ng_error (result_id INTEGER, file_id INTEGER);
            INSERT INTO inspection_result VALUES (1, 0, '2026-10-11 23:59:59');
            INSERT INTO inspection_result VALUES (2, 1, '2026-10-11 12:00:00');
            INSERT INTO image_file VALUES (11, 1, 'E:/top.png');
            INSERT INTO image_file VALUES (12, 1, 'E:/bottom.png');
            INSERT INTO image_file VALUES (21, 2, 'E:/ok.png');
            INSERT INTO ng_error VALUES (1, 11);
        ''')
    (tmp_path / 'source').mkdir()
    for name in ('top.png', 'bottom.png', 'ok.png'):
        (tmp_path / 'source' / name).write_bytes(name.encode())
    return path


def run_worker(operation, options):
    worker = DataTransferWorker(operation, options)
    results, errors = [], []
    worker.result.connect(results.append)
    worker.error.connect(errors.append)
    worker.run()
    assert not errors, errors
    assert len(results) == 1
    return results[0]


def image_options(tmp_path, **kwargs):
    return dict(excel=str(tmp_path / 'cosmos.xlsx'), output=str(tmp_path / 'images'),
                source_root=str(tmp_path / 'source'), source_drive='E:', only_ng_faces=False,
                start_date='2026-10-11', end_date='2026-10-11', dry_run=False, report=None) | kwargs


def test_export_check_copy_and_conflicts(app, database, tmp_path):
    before = database.read_bytes()
    run_worker('export', dict(database=database, output=tmp_path / 'cosmos.xlsx'))
    assert database.read_bytes() == before
    checked = run_worker('check', image_options(tmp_path, dry_run=True))
    assert checked['summary'] == {'ready': 2}
    assert not (tmp_path / 'images').exists()
    copied = run_worker('copy', image_options(tmp_path))
    assert copied['summary'] == {'copied': 2}
    assert Path(copied['report']).is_file()
    assert not (tmp_path / 'images' / 'ok.png').exists()
    assert (tmp_path / 'images' / 'top.png').read_bytes() == b'top.png'
    (tmp_path / 'images' / 'top.png').write_bytes(b'keep')
    conflict = run_worker('copy', image_options(tmp_path, only_ng_faces=True))
    assert conflict['failed'] == 1
    assert conflict['summary'] == {'conflict': 1}
    assert (tmp_path / 'images' / 'top.png').read_bytes() == b'keep'


def test_invalid_report_prevents_copy(app, database, tmp_path):
    run_worker('export', dict(database=database, output=tmp_path / 'cosmos.xlsx'))
    existing = tmp_path / 'report.csv'
    existing.write_text('keep')
    worker = DataTransferWorker('copy', image_options(tmp_path, report=str(existing)))
    errors = []
    worker.error.connect(errors.append)
    worker.run()
    assert errors and '尚不存在' in errors[0]
    assert not (tmp_path / 'images').exists()
    assert existing.read_text() == 'keep'


def test_stopped_copy_reports_unprocessed_rows(app, database, tmp_path):
    run_worker('export', dict(database=database, output=tmp_path / 'cosmos.xlsx'))
    worker = DataTransferWorker('copy', image_options(tmp_path))
    worker.stopped.set()
    results = []
    worker.result.connect(results.append)
    worker.run()
    assert results[0]['stopped']
    assert results[0]['summary'] == {'pending': 2}
    assert not (tmp_path / 'images' / 'top.png').exists()
    assert 'pending' in Path(results[0]['report']).read_text(encoding='utf-8-sig')


def wait_page(app, page):
    deadline = time.monotonic() + 15
    while page.worker is not None and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    assert page.worker is None


def test_registered_page_real_threads_and_path_handoff(app, database, tmp_path):
    plan = plan_default_capabilities((), include_project_capabilities=True)
    capability = next(item for item in plan if item.spec.key == 'data.ng_extract')
    window = QWidget()
    center = TaskCenter()
    host = _page_factory_for(capability.page_factory_key)(
        SimpleNamespace(window=window, task_center=center, project_session=None), window)
    page = host.excel_source
    assert isinstance(page, DataTransferPage)
    assert page.task_center is center
    page.database.setText(str(database))
    page.excel_output.setText(str(tmp_path / 'cosmos.xlsx'))
    page.export_button.click()
    assert not page.image_settings.isEnabled()
    assert not page.stop_button.isEnabled()
    closing = QCloseEvent()
    assert closing.type() == QEvent.Type.Close
    assert page.eventFilter(window, closing)
    assert not closing.isAccepted()
    wait_page(app, page)
    assert page.excel.text() == str(tmp_path / 'cosmos.xlsx')
    assert page.tabs.currentIndex() == 1
    assert page.output.text() == str(tmp_path / 'NG_images')
    page.source_root.setText(str(tmp_path / 'source'))
    page.copy_button.click()
    wait_page(app, page)
    assert page._result['summary'] == {'copied': 2}
    assert page.preview.rowCount() == 2
    assert page.open_button.isEnabled()
    assert len(center.tasks) == 2
    assert all(task.status == TaskStatus.SUCCESS for task in center.tasks)
    page.close()
    window.close()
