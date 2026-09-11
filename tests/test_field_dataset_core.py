import json
import sys
import threading
import types

import cv2
import numpy as np
import pytest

from cosmos_toolbox import field_dataset as core


@pytest.fixture
def fake_models(monkeypatch):
    class Models:
        calls = 0
        snapshot = {'model': 'fixture-v1'}
        def __init__(self, product):
            self.errors = {}
        def generate(self, image, selected, **kwargs):
            Models.calls += 1
            return {name: [{'image': image, 'shapes': []}] for name in selected}
    monkeypatch.setitem(sys.modules, 'cosmos_toolbox.field_models', types.SimpleNamespace(FieldModels=Models))
    return Models


def source_image(tmp_path):
    source = tmp_path / 'input'
    source.mkdir()
    cv2.imwrite(str(source / 'CAB-F_D01-R_20260907_081340149_top.png'), np.zeros((12, 12, 3), np.uint8))
    return source


def test_multi_model_one_decode_resume_and_preserve_annotation(tmp_path, fake_models, monkeypatch):
    source = source_image(tmp_path)
    output = tmp_path / 'output'
    original = cv2.imdecode
    calls = []
    monkeypatch.setattr(cv2, 'imdecode', lambda *a: (calls.append(1), original(*a))[1])
    args = dict(source=source, output=output, product='D01-R', selected=['tail_roi_detector', 'roi_detector'])
    assert core.run(**args)['completed_model_jobs'] == 2
    annotation = next((output / 'datasets').rglob('image.json'))
    payload = json.loads(annotation.read_text(encoding='utf-8'))
    payload['flags']['human'] = True
    annotation.write_text(json.dumps(payload), encoding='utf-8')
    assert core.run(**args)['skipped_model_jobs'] == 2
    assert len(calls) == 1
    assert json.loads(annotation.read_text())['flags']['human'] is True


def test_stop_before_decode(tmp_path, fake_models):
    source = source_image(tmp_path)
    control = types.SimpleNamespace(paused=threading.Event(), stopped=threading.Event())
    control.stopped.set()
    result = core.run(source, tmp_path / 'out', 'D01-R', ['roi_detector'], control=control)
    assert result['stopped'] and fake_models.calls == 0


def test_reject_nested_output(tmp_path, fake_models):
    source = source_image(tmp_path)
    with pytest.raises(ValueError):
        core.run(source, source / 'out', 'D01-R', ['roi_detector'])


def test_invalid_coordinates_never_commit(tmp_path, fake_models, monkeypatch):
    monkeypatch.setattr(fake_models, 'generate', lambda *a, **kw: {'roi_detector': [{
        'image': np.zeros((12, 12, 3), np.uint8),
        'shapes': [{'label': 'tail', 'shape_type': 'rectangle', 'points': [[0, 0], [99, 99]]}]}]})
    source = source_image(tmp_path)
    output = tmp_path / 'out'
    result = core.run(source, output, 'D01-R', ['roi_detector'])
    assert result['failed_model_jobs'] == 1
    assert not list((output / 'datasets').rglob('commit.json'))


def test_corrupt_image_isolated_and_reported(tmp_path, fake_models):
    source = source_image(tmp_path)
    (source / 'CAB-F_D01-R_20260907_081341149_bottom.png').write_bytes(b'broken')
    output = tmp_path / 'out'
    result = core.run(source, output, 'D01-R', ['roi_detector'])
    assert result['completed_model_jobs'] == 1
    assert result['failed_model_jobs'] == 1
    assert len(json.loads((output / 'failures.json').read_text(encoding='utf-8'))) == 1


def test_missing_label_does_not_overwrite_other_committed_files(tmp_path, fake_models):
    source = source_image(tmp_path)
    output = tmp_path / 'out'
    args = dict(source=source, output=output, product='D01-R', selected=['roi_detector'])
    core.run(**args)
    label = next((output / 'datasets').rglob('image.txt'))
    label.unlink()
    result = core.run(**args)
    assert result['failed_model_jobs'] == 1
    assert not label.exists()


@pytest.mark.parametrize('filename,product', [
    ('CAB-F_TEST_D01-L_20260806_151746595_bottom.jpg', 'D01-R'),
    ('CAB-F_TEST_D01-R_20260806_161601062_bottom.jpg', 'D01-L'),
])
def test_selected_product_overrides_filename(tmp_path, fake_models, filename, product):
    source = tmp_path / 'input'
    source.mkdir()
    cv2.imwrite(str(source / filename), np.zeros((12, 12, 3), np.uint8))
    output = tmp_path / 'out'
    result = core.run(source, output, product, ['roi_detector'])
    assert result['completed_model_jobs'] == 1
    assert result['failed_model_jobs'] == 0
    assert list((output / 'datasets' / 'roi_detector' / product).rglob('commit.json'))
    assert filename in (output / 'manifest.jsonl').read_text(encoding='utf-8')
