import json
import re
import sqlite3
from pathlib import Path

import cv2
import numpy as np
import pytest

from functools import partial
from cosmos_toolbox.field_dataset_export import export as default_export

# Retain regression coverage for explicit legacy multi-format exports.
export = partial(default_export, xany_only=False)


@pytest.mark.parametrize('empty', [True, False])
def test_default_export_is_one_xany_pair_per_sample(tmp_path, empty):
    sample(tmp_path, shapes=[] if empty else None)
    report = default_export(tmp_path)
    root = Path(report['directory'])
    assert report['valid'] and report['samples'] == 1
    assert 'training_dataset' not in report
    assert len(list(root.rglob('*.png'))) == 1
    expected = root / ('_review/roi_detector' if empty else 'roi_detector')
    image = next(expected.glob('*.png'))
    annotation = json.loads(image.with_suffix('.json').read_text(encoding='utf-8'))
    assert annotation['imagePath'] == image.name
    assert annotation['field_metadata']['product'] == 'D01-R'
    assert not (root / 'training_datasets').exists()
    assert not any(p.name in {'train', 'val'} for p in root.rglob('*') if p.is_dir())
    assert not list(root.rglob('data.yaml'))
    assert not list(root.rglob('mask_*.png'))


def sample(tmp_path, model='roi_detector', shapes=None, metadata=None, text='0 0.5 0.5 0.5 0.5'):
    folder = tmp_path / 'datasets' / 'sample'
    folder.mkdir(parents=True)
    cv2.imwrite(str(folder / 'image.png'), np.zeros((20, 40, 3), np.uint8))
    if shapes is None:
        shapes = [dict(label='tail', shape_type='rectangle', points=[[10, 5], [30, 15]])]
    annotation = dict(imageWidth=40, imageHeight=20, shapes=shapes, field_metadata=metadata or {})
    (folder / 'image.json').write_text(json.dumps(annotation), encoding='utf-8')
    (folder / 'image.txt').write_text(text, encoding='utf-8')
    (folder / 'classes.json').write_text(json.dumps(['qr'] if model == 'qr_yolo_cut' else ['tail']))
    record = dict(directory=str(folder), product='D01-R', face='top', split='train')
    with sqlite3.connect(tmp_path / 'run.db') as db:
        db.execute('CREATE TABLE items (key TEXT, model TEXT, status TEXT, details TEXT)')
        db.execute('INSERT INTO items VALUES (?,?,?,?)', ('fixture', model, 'complete', json.dumps([record])))
    return folder


def test_detection_snapshot_preserves_previous_export_and_sources(tmp_path):
    folder = sample(tmp_path)
    before = (folder / 'image.json').read_bytes()
    first = export(tmp_path)
    assert first['valid'] and first['samples'] == 1
    root = Path(first['directory'])
    assert re.fullmatch(r'dataset_candidates_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_\d+', root.name)
    label = next(root.glob('*/D01-R/top/yolo_candidates/labels/train/*.txt'))
    original_label = label.read_bytes()
    second = export(tmp_path)
    assert second['directory'] != first['directory']
    assert label.read_bytes() == original_label
    assert (folder / 'image.json').read_bytes() == before
    annotation = next(root.glob('*/D01-R/top/xanylabeling/train/*.json'))
    data = json.loads(annotation.read_text())
    assert data['field_metadata']['annotation_status'] == 'review'
    assert (annotation.parent / data['imagePath']).is_file()
    flat = Path(first['training_dataset']['directory'])
    assert (flat / 'roi_detector/data.yaml').is_file()
    assert len(list((flat / 'roi_detector/images/train').glob('*.png'))) == 1
    assert len(list((flat / 'roi_detector/labels/train').glob('*.txt'))) == 1
    assert not (flat / 'roi_detector/D01-R').exists()
    assert len(list((flat / '_review/roi_detector/train').glob('*.json'))) == 1


def test_readable_source_names_are_stable_and_unique(tmp_path):
    import shutil
    folder = sample(tmp_path)
    second = folder.parent / 'second_crop'
    shutil.copytree(folder, second)
    records = [dict(directory=str(p), source=str(tmp_path / sub / '原图_top.jpg'),
                    product='D01-R', face='top', split='train')
               for p, sub in [(folder, 'a'), (second, 'b')]]
    with sqlite3.connect(tmp_path / 'run.db') as db:
        db.execute('UPDATE items SET details=?', (json.dumps(records),))
    first = default_export(tmp_path, source=tmp_path / '0907')
    second_run = default_export(tmp_path, source=tmp_path / '0907')
    assert Path(first['directory']).name.startswith('0907_candidates_')
    names = {p.name for p in Path(first['directory']).rglob('*.png')}
    assert len(names) == 2
    assert all(name.startswith('原图_top__crop_') for name in names)
    assert names == {p.name for p in Path(second_run['directory']).rglob('*.png')}
    for path in Path(first['directory']).rglob('*.png'):
        assert json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))['imagePath'] == path.name


def test_empty_detection_kept_only_for_review(tmp_path):
    sample(tmp_path, text='', shapes=[])
    report = export(tmp_path)
    assert report['valid']
    root = Path(report['directory'])
    assert len(list(root.glob('*/D01-R/top/xanylabeling/train/*.png'))) == 1
    assert not list(root.glob('*/D01-R/top/yolo_candidates/images/train/*.png'))


def test_invalid_json_coordinate_rejected(tmp_path):
    sample(tmp_path, shapes=[dict(label='tail', shape_type='rectangle', points=[[-1, 2], [10, 10]])])
    report = export(tmp_path)
    assert not report['valid'] and report['samples'] == 0


def test_detection_uses_edited_json_not_stale_txt(tmp_path):
    folder = sample(tmp_path)
    annotation = json.loads((folder / 'image.json').read_text())
    annotation['shapes'][0]['points'] = [[2, 4], [10, 8]]
    (folder / 'image.json').write_text(json.dumps(annotation))
    report = export(tmp_path)
    assert report['valid'], report
    labels = list(Path(report['directory']).glob('*/D01-R/top/yolo_candidates/labels/train/*.txt'))
    assert list(map(float, labels[0].read_text().split())) == pytest.approx([0, .15, .3, .2, .2])
    assert (folder / 'image.txt').read_text() == '0 0.5 0.5 0.5 0.5'
    annotation['shapes'] = []
    (folder / 'image.json').write_text(json.dumps(annotation))
    deleted = export(tmp_path)
    assert deleted['valid']
    assert not list(Path(deleted['directory']).glob('*/D01-R/top/yolo_candidates/labels/train/*.txt'))


def test_qr_rotated_polygon_exports_nine_column_obb_from_json(tmp_path):
    points = [[10, 1], [30, 5], [28, 15], [8, 11]]
    sample(tmp_path, model='qr_yolo_cut', shapes=[dict(label='qr', shape_type='polygon', points=points)],
           metadata={'yolo_task': 'obb'}, text='')
    report = export(tmp_path)
    assert report['valid'], report
    root = Path(report['directory'])
    label = next(root.glob('*/D01-R/top/yolo_candidates/labels/train/*.txt'))
    values = label.read_text().split()
    assert len(values) == 9 and values[0] == '0'
    assert list(map(float, values[1:])) == pytest.approx([v for x, y in points for v in (x/40, y/20)])
    assert 'task: obb' in (label.parents[2] / 'data.yaml').read_text()


@pytest.mark.parametrize('points', [
    [[1, 1], [10, 10], [1, 10], [10, 1]],
    [[1, 1], [2, 2], [3, 3], [4, 4]],
    [[-1, 1], [10, 1], [10, 10], [1, 10]],
])
def test_invalid_obb_rejected(tmp_path, points):
    sample(tmp_path, model='qr_yolo_cut', shapes=[dict(label='qr', shape_type='polygon', points=points)],
           metadata={'yolo_task': 'obb'}, text='')
    report = export(tmp_path)
    assert not report['valid'] and report['samples'] == 0
