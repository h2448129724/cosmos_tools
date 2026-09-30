import copy
import os
from pathlib import Path
from types import SimpleNamespace
import pytest
from cosmos_toolbox.field_history import validate_snapshot, COMPATIBLE_CORE_PAIRS, discover


def test_discovery_never_creates_database(tmp_path):
    assert discover(tmp_path) == ([], [])
    assert not list(tmp_path.iterdir())


def test_batch_and_ng_history_discovery(tmp_path):
    import json
    import sqlite3
    for name, ng in [('folder_one', False), ('folder_two', True)]:
        workspace = tmp_path / name
        if ng:
            workspace /= 'ng_only'
        (workspace / 'runs').mkdir(parents=True)
        with sqlite3.connect(workspace/'run.db') as db:
            db.execute('CREATE TABLE items(key TEXT,status TEXT)')
            db.execute('CREATE TABLE run_items(run_id TEXT,item_key TEXT)')
            db.execute("INSERT INTO items VALUES ('item','complete')")
            db.execute("INSERT INTO run_items VALUES ('123','item')")
        (workspace/'runs/123.json').write_text(json.dumps(dict(run_id='123', source='input', version='v1',
            product='D01-R', selected=['roi_detector'], mode='auto', ng_only=ng)), encoding='utf-8')
    records, errors = discover(tmp_path)
    assert len(records) == 2 and not errors
    for record in records:
        assert record['counts']['complete'] == 1
        assert Path(record['options']['output']).name in ('folder_one', 'folder_two')


def test_migration_targets_match_current_core_file():
    import hashlib
    from cosmos_toolbox import field_dataset
    data = Path(field_dataset.__file__).read_bytes().replace(b'\r\n', b'\n')
    accepted = {new for _, new in COMPATIBLE_CORE_PAIRS}
    assert hashlib.sha256(data).hexdigest() in accepted
    assert hashlib.sha256(data.replace(b'\n', b'\r\n')).hexdigest() in accepted


def test_snapshot_allows_only_audited_core_and_ui_changes():
    old_core, new_core = next(iter(COMPATIBLE_CORE_PAIRS))
    old = dict(models={'weights': 'original'}, tool_sources={'field_dataset.py': old_core, 'field_models.py': 'adapter',
                                                          'field_dataset_ui.py': 'old-ui'})
    new = copy.deepcopy(old)
    new['tool_sources'].update({'field_dataset.py': new_core, 'field_dataset_ui.py': 'new-ui', 'field_history.py': 'new'})
    validate_snapshot(old, new)
    new['tool_sources']['field_dataset.py'] = 'arbitrary-future-core'
    with pytest.raises(ValueError, match='生成源码不兼容'):
        validate_snapshot(old, new)
    new = copy.deepcopy(old)
    new['models']['weights'] = 'changed'
    with pytest.raises(ValueError, match='快照不兼容'):
        validate_snapshot(old, new)


def test_restore_ui_and_explicit_new_task(tmp_path):
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from PySide6.QtWidgets import QApplication
    from cosmos_toolbox.field_dataset_ui import FieldDatasetPage
    app = QApplication.instance() or QApplication([])
    page = FieldDatasetPage()
    record = dict(path='run.json', data={'environment': 'cabf'}, missing=[], options={
        'source': str(tmp_path/'input'), 'output': str(tmp_path/'output'), 'product_config': None,
        'face': 'bottom', 'mode': 'images', 'limit': 7, 'export_scope': 'current',
        'ng_only': True, 'selected': ['hook_detector']})
    page._restore_record(record)
    assert page.source.toPlainText() == record['options']['source']
    assert page.face.currentData() == 'bottom'
    assert page.limit.value() == 7
    assert page.ng_only.isChecked()
    assert page.environment.currentText() == 'cabf'
    assert [key for key, value in page.checks.items() if value.isChecked()] == ['hook_detector']
    assert page.start_button.text() == '继续此任务'
    page._new_task()
    assert page._resume_record is None
    assert page.start_button.text() == '开始生成'
    page.close()
