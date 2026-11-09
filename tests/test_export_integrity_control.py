import json
import sqlite3
import threading
import pytest
from pathlib import Path
from types import SimpleNamespace

from cosmos_toolbox.field_dataset import _intact
from cosmos_toolbox.field_dataset_export import export
from cosmos_toolbox.field_batch import run_batch
from test_field_export import sample


def control():
    return SimpleNamespace(paused=threading.Event(), stopped=threading.Event())


def test_truncated_image_is_not_resumed_or_exported(tmp_path):
    folder = sample(tmp_path)
    image = folder / 'image.png'
    image.write_bytes(image.read_bytes()[:24])
    before = image.read_bytes()
    assert not _intact([{'directory': str(folder)}])
    report = export(tmp_path, scope='history')
    assert not report['valid'] and report['samples'] == 0
    assert report['status'] == 'failed'
    assert (Path(report['directory']) / 'INCOMPLETE.txt').is_file()
    assert image.read_bytes() == before


@pytest.mark.parametrize('damage', ['iend', 'crc'])
def test_png_container_damage_is_rejected(tmp_path, damage):
    folder = sample(tmp_path)
    image = folder / 'image.png'
    content = image.read_bytes()
    image.write_bytes(content[:-12] if damage == 'iend' else content[:-1] + bytes([content[-1] ^ 1]))
    assert not _intact([{'directory': str(folder)}])
    assert not export(tmp_path, scope='history')['valid']


def test_valid_human_edits_remain_intact(tmp_path):
    folder = sample(tmp_path)
    path = folder / 'image.json'
    annotation = json.loads(path.read_text())
    annotation['shapes'] = []
    path.write_text(json.dumps(annotation))
    before = path.read_bytes()
    assert _intact([{'directory': str(folder)}])
    report = export(tmp_path, scope='history')
    assert report['valid'] and report['status'] == 'complete'
    assert not (Path(report['directory']) / 'INCOMPLETE.txt').exists()
    assert path.read_bytes() == before


def test_export_stops_between_samples_and_preserves_ledger(tmp_path):
    sample(tmp_path)
    with sqlite3.connect(tmp_path / 'run.db') as db:
        db.execute("INSERT INTO items SELECT 'two',model,status,details FROM items")
    before = (tmp_path / 'run.db').read_bytes()
    state = control()
    def progress(fact):
        if fact.get('export_samples') == 1:
            state.stopped.set()
    report = export(tmp_path, scope='history', control=state, on_progress=progress)
    assert report['stopped'] and not report['valid'] and report['samples'] == 1
    root = Path(report['directory'])
    assert (root / 'INCOMPLETE.txt').exists()
    assert json.loads((root / 'export_state.json').read_text())['status'] == 'stopped'
    assert (tmp_path / 'run.db').read_bytes() == before


def test_pause_waits_then_stop_leaves_explicit_partial(tmp_path):
    sample(tmp_path)
    state = control()
    state.paused.set()
    timer = threading.Timer(.1, state.stopped.set)
    timer.start()
    try:
        report = export(tmp_path, scope='history', control=state)
    finally:
        timer.join()
    assert report['stopped'] and report['samples'] == 0


def test_export_write_failure_is_not_published(tmp_path, monkeypatch):
    from cosmos_toolbox import field_dataset_export
    sample(tmp_path)
    before = (tmp_path / 'run.db').read_bytes()
    def disk_full(*args):
        raise OSError('disk full')
    monkeypatch.setattr(field_dataset_export, '_copy', disk_full)
    report = export(tmp_path, scope='history')
    assert report['status'] == 'failed' and not report['valid']
    assert (Path(report['directory']) / 'INCOMPLETE.txt').exists()
    assert (tmp_path / 'run.db').read_bytes() == before


def test_export_pause_resumes(tmp_path):
    sample(tmp_path)
    state = control()
    state.paused.set()
    timer = threading.Timer(.1, state.paused.clear)
    timer.start()
    try:
        report = export(tmp_path, scope='history', control=state)
    finally:
        timer.join()
    assert report['status'] == 'complete' and report['samples'] == 1


def test_batch_execution_without_qt_preserves_queue(tmp_path):
    state = control()
    seen = []
    def execute(options, environment):
        seen.append((options, environment))
        state.stopped.set()
        return {'stopped': True}
    result = run_batch({'sources': ['one', 'two'], 'output': str(tmp_path)},
                       'onnx-gpu', state, lambda fact: None, execute)
    assert len(seen) == 1 and seen[0][1] == 'onnx-gpu'
    batch = json.loads(Path(result['batch_manifest']).read_text())
    assert [j['status'] for j in batch['jobs']] == ['stopped', 'pending']
