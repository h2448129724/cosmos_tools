"""Resumable, append-only field dataset collection; images are decoded once per run item."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import shutil
import time
from pathlib import Path

MODEL_IDS = dict(zip(
    ('glue_segment roi_detector ear_placement_classifier knife_segment tail_roi_detector '
     'tail_placement_classifier tail_cloth_roi_detector tail_cloth_seam_classifier hook_detector '
     'sew_point_detector sew_point_connector reinforcement_placement_detector error_detector qr_yolo_cut').split(),
    ('胶路分割 公共区域 耳片位置 耳片分割 尾部区域 拉带位置 拉带布细节 缝线类型 钩子方向 '
     '针点 针点连接 加强布 布面异常 二维码').split()))
CLASSES = {
    'roi_detector': ['ear', 'tail', 'dm'], 'tail_roi_detector': ['cloth', 'hook', 'placement'],
    'tail_cloth_roi_detector': ['point', 'seam', 'line'], 'qr_yolo_cut': ['qr'],
    'error_detector': ['defective_fabric', 'defect_mark_line', 'stain', 'pulled_thread',
                       'loose_thread', 'broken_thread', 'skipped_stitch', 'foreign_object'],
}
MODEL_DEPENDENCIES = {
    'glue_segment': [], 'roi_detector': [],
    'ear_placement_classifier': ['roi_detector'], 'knife_segment': ['roi_detector'],
    'tail_roi_detector': ['roi_detector'],
    'tail_placement_classifier': ['roi_detector', 'tail_roi_detector'],
    'tail_cloth_roi_detector': ['roi_detector', 'tail_roi_detector'],
    'tail_cloth_seam_classifier': ['roi_detector', 'tail_roi_detector', 'tail_cloth_roi_detector'],
    'hook_detector': ['roi_detector', 'tail_roi_detector'],
    'sew_point_detector': ['roi_detector', 'tail_roi_detector'],
    'sew_point_connector': ['roi_detector', 'tail_roi_detector', 'sew_point_detector'],
    'reinforcement_placement_detector': ['glue_segment'],
    'error_detector': [], 'qr_yolo_cut': ['glue_segment'],
}


def scan(source, face='all'):
    root = Path(source).resolve()
    if not root.is_dir():
        raise ValueError('原图目录不存在')
    return sorted(p for p in root.rglob('*') if p.is_file()
                  and p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
                  and (face == 'all' or p.stem.lower().endswith('_' + face)))


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    os.replace(temp, path)


def _intact(records):
    return bool(records) and all(all((Path(r['directory']) / name).is_file()
        for name in r.get('files', ['image.png', 'image.json'])) for r in records)


def _save_sample(folder, sample, identity, model, mode):
    import cv2
    import numpy as np
    image = sample['image']
    h, w = image.shape[:2]
    shapes = sample.get('shapes', [])
    for shape in shapes:
        for x, y in shape.get('points', []):
            if not (np.isfinite(x) and np.isfinite(y) and 0 <= x <= w and 0 <= y <= h):
                raise ValueError('标注坐标越界或非有限值')
    folder.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode('.png', image)
    if not ok:
        raise ValueError('裁片编码失败')
    encoded.tofile(str(folder / 'image.png'))
    metadata = {**identity, **sample.get('metadata', {}), 'model': model, 'mode': mode,
                'annotation_status': 'unlabeled' if mode == 'images' else 'review',
                'classification': sample.get('classification')}
    _json(folder / 'image.json', {'version': '5.5.0', 'flags': {}, 'shapes': shapes,
          'imagePath': 'image.png', 'imageData': None, 'imageHeight': h, 'imageWidth': w,
          'field_metadata': metadata})
    names = CLASSES.get(model)
    if names and mode != 'images':
        lines = []
        for shape in shapes:
            if shape.get('shape_type') != 'rectangle':
                continue
            label = shape['label']
            if label not in names:
                raise ValueError(f'未知类别 {label}')
            (x1, y1), (x2, y2) = shape['points']
            x1, x2 = sorted((x1, x2)); y1, y2 = sorted((y1, y2))
            if x2 <= x1 or y2 <= y1:
                raise ValueError('零面积框')
            lines.append(f'{names.index(label)} {(x1+x2)/2/w:.8f} {(y1+y2)/2/h:.8f} {(x2-x1)/w:.8f} {(y2-y1)/h:.8f}')
        (folder / 'image.txt').write_text('\n'.join(lines), encoding='utf-8')
        _json(folder / 'classes.json', names)
    masks = sample.get('masks', {})
    for label, mask in masks.items():
        if Path(str(label)).name != str(label):
            raise ValueError('无效 mask 类别名')
        ok, data = cv2.imencode('.png', np.asarray(mask, dtype=np.uint8))
        if not ok:
            raise ValueError('mask 编码失败')
        data.tofile(str(folder / f'mask_{label}.png'))
    return {'directory': str(folder), **metadata, 'shape_count': len(shapes), 'width': w, 'height': h,
            'files': sorted(p.name for p in folder.iterdir() if p.is_file())}


def run(source, output, product, selected, face='all', mode='auto', limit=None, control=None, on_progress=None,
        product_config=None):
    from .paths import ensure_import_paths
    ensure_import_paths()
    import cv2
    import numpy as np
    from .field_models import FieldModels
    selected = list(dict.fromkeys(selected))
    if product not in {'D01-L', 'D01-R'} or mode not in {'auto', 'images'} or face not in {'all', 'top', 'bottom'}:
        raise ValueError('产品、生成方式或正反面无效')
    if not selected or set(selected) - MODEL_IDS.keys():
        raise ValueError('请选择有效模型')
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError('输入与输出目录不能互相包含')
    paths = scan(source, face)
    # The operator's selection owns product identity; filenames are provenance only.
    if limit is not None:
        paths = paths[:limit]
    output.mkdir(parents=True, exist_ok=True)
    lock_path = output / '.running.lock'
    lock = open(lock_path, 'a+b')
    # OS lock is released after process death; a leftover filename does not block recovery.
    import msvcrt
    lock.seek(0); lock.write(b'0'); lock.flush(); lock.seek(0)
    try:
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        lock.close()
        raise RuntimeError('此输出目录已有生成任务运行')
    db = sqlite3.connect(output / 'run.db')
    db.execute('CREATE TABLE IF NOT EXISTS items (key TEXT PRIMARY KEY, source TEXT, model TEXT, status TEXT, details TEXT)')
    models = None
    processed = skipped = failures = 0
    try:
        models = FieldModels(product, product_config=product_config) if product_config else FieldModels(product)
        if on_progress and getattr(models, 'product_config_path', None):
            on_progress({'message': f'实际产品配置：{models.product_config_path}'})
        snapshot = getattr(models, 'snapshot', {'product': product})
        if callable(snapshot):
            snapshot = snapshot()
        from . import field_models
        snapshot = {**snapshot, 'adapter_sha256': hashlib.sha256(Path(field_models.__file__).read_bytes()).hexdigest()
                    if getattr(field_models, '__file__', None) else 'test'}
        version = hashlib.sha256(json.dumps(snapshot, sort_keys=True, default=str).encode()).hexdigest()[:12]
        _json(output / 'runs' / f'{time.time_ns()}.json', {'product': product, 'selected': selected,
              'mode': mode, 'source': str(source), 'models': snapshot, 'version': version})
        for index, path in enumerate(paths):
            if shutil.disk_usage(output).free < 2 * 1024**3:
                raise RuntimeError('输出磁盘可用空间不足 2 GiB，已提交进度保存在 run.db')
            if control:
                while control.paused.is_set() and not control.stopped.is_set():
                    time.sleep(.1)
                if control.stopped.is_set():
                    break
            stat = path.stat()
            identity_text = f'{path}|{stat.st_size}|{stat.st_mtime_ns}|{version}|{mode}'
            sid = hashlib.sha256(identity_text.encode()).hexdigest()[:20]
            pending = []
            for model in selected:
                found = db.execute('SELECT status,details FROM items WHERE key=?', (sid + model,)).fetchone()
                intact = found and found[0] == 'complete' and _intact(json.loads(found[1]))
                if intact:
                    skipped += 1
                else:
                    pending.append(model)
            if not pending:
                continue
            match = re.search(r'_(\d{8})_(\d{6})\d{3}_(top|bottom)$', path.stem)
            group = str(path.parent.relative_to(source)) + '/' + (
                '_'.join(match.group(1, 2)) if match else path.stem)
            split = 'val' if int(hashlib.sha256((product + group + '42').encode()).hexdigest()[:8], 16) % 5 == 0 else 'train'
            identity = {'source': str(path), 'product': product, 'face': path.stem.rsplit('_', 1)[-1],
                        'split': split, 'sample_id': sid, 'model_version': version}
            try:
                image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError('原图解码失败')
                results = models.generate(image, pending, mode=mode, face=identity['face'])
                for model in pending:
                    key = sid + model
                    try:
                        if model not in results:
                            raise RuntimeError(getattr(models, 'errors', {}).get(model, '没有生成结果'))
                        samples = results[model]
                        if not samples:
                            raise RuntimeError('没有可导出的候选裁片，需要复核上游定位')
                        target = output / 'datasets' / model / product / identity['face'] / sid
                        if target.exists():
                            # Recover a completed directory that was committed before the DB transaction.
                            marker = target / 'commit.json'
                            if not marker.exists():
                                raise RuntimeError('已有未确认目录，需要人工检查')
                            records = json.loads(marker.read_text(encoding='utf-8'))
                            if not _intact(records):
                                raise RuntimeError('已提交样本文件缺失；保留目录以免覆盖人工修改，请使用新输出目录重建')
                        else:
                            staging = output / 'staging' / f'{sid}_{model}_{time.time_ns()}'
                            records = [_save_sample(staging / str(i), s, identity, model, mode)
                                       for i, s in enumerate(samples)]
                            for i, record in enumerate(records):
                                record['directory'] = str(target / str(i))
                            _json(staging / 'commit.json', records)
                            target.parent.mkdir(parents=True, exist_ok=True)
                            os.replace(staging, target)
                        db.execute('INSERT OR REPLACE INTO items VALUES (?,?,?,?,?)',
                                   (key, str(path), model, 'complete', json.dumps(records, ensure_ascii=False)))
                        processed += 1
                    except Exception as exc:
                        failures += 1
                        db.execute('INSERT OR REPLACE INTO items VALUES (?,?,?,?,?)',
                                   (key, str(path), model, 'failed', json.dumps({'error': str(exc)}, ensure_ascii=False)))
                    db.commit()
                del image, results
            except Exception as exc:
                for model in pending:
                    db.execute('INSERT OR REPLACE INTO items VALUES (?,?,?,?,?)',
                               (sid + model, str(path), model, 'failed', json.dumps({'error': str(exc)}, ensure_ascii=False)))
                    failures += 1
                db.commit()
            if on_progress:
                on_progress({'current': index + 1, 'total': len(paths), 'source': str(path),
                             'completed': processed, 'failed': failures, 'skipped': skipped})
        rows = db.execute('SELECT source,model,status,details FROM items ORDER BY key').fetchall()
        summary = {'scanned': len(paths), 'completed_model_jobs': processed, 'failed_model_jobs': failures,
                   'skipped_model_jobs': skipped, 'total_model_jobs': len(rows),
                   'stopped': bool(control and control.stopped.is_set())}
        summary['models'] = {model: {
            'complete': sum(r[1] == model and r[2] == 'complete' for r in rows),
            'failed': sum(r[1] == model and r[2] == 'failed' for r in rows),
            'samples': sum(len(json.loads(r[3])) for r in rows if r[1] == model and r[2] == 'complete'),
        } for model in selected}
        _json(output / 'failures.json', [{'source': r[0], 'model': r[1], **json.loads(r[3])}
                                       for r in rows if r[2] == 'failed'])
        _json(output / 'summary.json', summary)
        temp = output / 'manifest.jsonl.tmp'
        with temp.open('w', encoding='utf-8') as stream:
            for src, model, status, details in rows:
                stream.write(json.dumps({'source': src, 'model': model, 'status': status,
                                         'details': json.loads(details)}, ensure_ascii=False) + '\n')
        os.replace(temp, output / 'manifest.jsonl')
        if rows and not summary['stopped']:
            from .field_dataset_export import export
            summary['export'] = export(output, selected)
            _json(output / 'summary.json', summary)
        return summary
    finally:
        db.close()
        lock.seek(0); msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1); lock.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True); parser.add_argument('--output', required=True)
    parser.add_argument('--product', choices=['D01-L', 'D01-R'], required=True)
    parser.add_argument('--models', nargs='+', choices=list(MODEL_IDS), required=True)
    parser.add_argument('--face', default='all', choices=['all', 'top', 'bottom'])
    parser.add_argument('--mode', default='auto', choices=['auto', 'images'])
    parser.add_argument('--limit', type=int)
    parser.add_argument('--product-config', help='Optional product YAML; defaults to the selected product .yaml')
    args = vars(parser.parse_args()); args['selected'] = args.pop('models')
    print(run(**args, on_progress=lambda p: print(json.dumps(p, ensure_ascii=False), flush=True)))


if __name__ == '__main__':
    main()
