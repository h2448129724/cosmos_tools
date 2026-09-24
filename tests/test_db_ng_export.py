import json
import sqlite3
import threading
from types import SimpleNamespace

import pytest

from cosmos_toolbox.db_ng_export import scan, copy_plan


@pytest.fixture
def database(tmp_path):
    path = tmp_path / 'logs.db'
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'original.jpg').write_bytes(b'fixture image bytes')
    with sqlite3.connect(path) as db:
        db.executescript('''
            CREATE TABLE inspection_result(id INTEGER,project TEXT,product TEXT,product_id TEXT,ok INTEGER,is_misjudged INTEGER,create_time TEXT);
            CREATE TABLE image_file(id INTEGER,result_id INTEGER,filename TEXT,raw_image TEXT,result_image TEXT);
            CREATE TABLE ng_error(result_id INTEGER,file_id INTEGER,error_code TEXT);
            INSERT INTO inspection_result VALUES(1,'CAB-F','D01-L','a',0,0,'2026-09-24 10:00:00');
            INSERT INTO inspection_result VALUES(2,'CAB-F','D01-R','b',1,0,'2026-09-24 11:00:00');
            INSERT INTO inspection_result VALUES(3,'CAB-F','D01-L','c',0,1,'2026-09-25 10:00:00');
            INSERT INTO inspection_result VALUES(4,'CAB','D01-L','d',0,0,'2026-09-24 10:00:00');
            INSERT INTO ng_error VALUES(1,0,'hook NG');
        ''')
        db.execute('INSERT INTO image_file VALUES(1,1,?,?,?)', ('original.jpg', str(source/'original.jpg'), str(source/'missing.jpg')))
        db.execute('INSERT INTO image_file VALUES(2,1,?,?,?)', ('original.jpg', str(source/'original.jpg'), ''))
    return path


def test_readonly_scan_filters_dedup_and_missing(database):
    before = database.read_bytes()
    plan = scan(database, kinds=('raw', 'result'))
    assert plan['summary']['ng_records'] == 2
    assert plan['summary']['image_records'] == 2
    assert plan['summary']['ready_files'] == 1
    assert plan['summary']['missing_files'] == 2
    assert plan['summary']['unlinked_errors'] == 1
    assert len(plan['missing_records']) == 1
    assert len(plan['files'][0]['associations']) == 2
    assert database.read_bytes() == before
    assert scan(database, end='2026-09-25')['summary']['ng_records'] == 1
    assert scan(database, exclude_misjudged=True)['summary']['ng_records'] == 1
    assert scan(database, product="' OR 1=1--")['summary']['ng_records'] == 0


def test_copy_resume_and_nonoverwrite(database, tmp_path):
    plan = scan(database)
    output = tmp_path/'out'
    source = plan['files'][0]['source']
    assert copy_plan(plan, output)['copied'] == 1
    assert copy_plan(plan, output)['skipped'] == 1
    image = next((output/'raw').glob('*.jpg'))
    assert image.read_bytes() == b'fixture image bytes'
    record = json.loads((output/'manifest.jsonl').read_text(encoding='utf-8'))
    assert record['source'] == source and len(record['associations']) == 2
    image.write_bytes(b'edited by user')
    assert copy_plan(plan, output)['failed'] == 1
    assert image.read_bytes() == b'edited by user'


def test_recovery_after_rename_before_journal(database, tmp_path):
    plan = scan(database)
    output = tmp_path/'out'
    copy_plan(plan, output)
    (output/'copy_ledger.jsonl').write_text('{partial', encoding='utf-8')
    assert copy_plan(plan, output)['skipped'] == 1
    assert copy_plan(plan, output)['skipped'] == 1


def test_source_changed_and_cancel(database, tmp_path):
    plan = scan(database)
    control = SimpleNamespace(paused=threading.Event(), stopped=threading.Event())
    control.stopped.set()
    report = copy_plan(plan, tmp_path/'out', control)
    assert report['stopped'] and report['copied'] == 0
    (tmp_path/'source/original.jpg').write_bytes(b'changed')
    assert copy_plan(plan, tmp_path/'out')['failed'] == 1


def test_reject_unrelated_nonempty_output(database, tmp_path):
    output = tmp_path/'out'
    output.mkdir()
    (output/'manifest.jsonl').write_text('user content', encoding='utf-8')
    with pytest.raises(ValueError, match='非空'):
        copy_plan(scan(database), output)
    assert (output/'manifest.jsonl').read_text(encoding='utf-8') == 'user content'


def test_path_mapping_and_output_protection(database, tmp_path):
    mapped = tmp_path/'mapped'
    mapped.mkdir()
    (mapped/'original.jpg').write_bytes(b'mapped')
    plan = scan(database, old_root=str(tmp_path/'source'), new_root=str(mapped))
    assert plan['files'][0]['source'] == str(mapped/'original.jpg')
    with pytest.raises(ValueError):
        copy_plan(plan, mapped)
    with pytest.raises(ValueError):
        scan(database, old_root=str(mapped))
    with pytest.raises(ValueError):
        scan(database, start='2026-09-25', end='2026-09-24')


def test_stop_mid_copy_cleans_partial_then_resumes(database, tmp_path, monkeypatch):
    import cosmos_toolbox.db_ng_export as module
    plan = scan(database)
    original = module._checkpoint
    control = SimpleNamespace(paused=threading.Event(), stopped=threading.Event())
    calls = 0
    def checkpoint(value):
        nonlocal calls
        calls += 1
        if calls == 2:
            control.stopped.set()
        return original(value)
    monkeypatch.setattr(module, '_checkpoint', checkpoint)
    output = tmp_path/'out'
    report = copy_plan(plan, output, control)
    assert report['stopped'] and report['copied'] == 0
    assert not list(output.rglob('*.partial'))
    assert not list(output.rglob('*.jpg'))
    monkeypatch.setattr(module, '_checkpoint', original)
    control.stopped.clear()
    assert copy_plan(plan, output, control)['copied'] == 1


def test_missing_files_are_reported_not_copied(database, tmp_path):
    report = copy_plan(scan(database, kinds=('result',)), tmp_path/'out')
    assert report['missing'] == 2 and report['copied'] == 0
    assert len(json.loads((tmp_path/'out/missing.json').read_text(encoding='utf-8'))) == 2
