"""Portable, versioned candidate exports from committed field collection samples."""
from __future__ import annotations

import hashlib
import json
import math
import shutil
import sqlite3
import struct
import time
from pathlib import Path


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def _part(value):
    value = str(value)
    if not value or value in {'.', '..'} or any(c in value for c in '/\\<>:"|?*'):
        raise ValueError(f'Invalid export path component: {value!r}')
    return value


def _dimensions(path):
    with path.open('rb') as stream:
        header = stream.read(24)
    if len(header) != 24 or header[:8] != b'\x89PNG\r\n\x1a\n' or header[12:16] != b'IHDR':
        raise ValueError(f'Invalid PNG header: {path.name}')
    width, height = struct.unpack('>II', header[16:24])
    if not width or not height:
        raise ValueError('Empty image dimensions')
    return width, height


def _copy(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)


def export(output, models=None):
    """Export a fresh snapshot; never overwrite earlier exports or source annotations.

    The returned report describes structural validation, not annotation accuracy.
    Empty detections are available for review only, never automatic negatives.
    """
    output = Path(output).resolve()
    database = output / 'run.db'
    if not database.is_file():
        raise ValueError('run.db does not exist')
    destination = output / 'exports' / ('candidates_' + str(time.time_ns()))
    destination.mkdir(parents=True, exist_ok=False)
    report = {'directory': str(destination), 'annotation_status': 'review',
              'verified_ground_truth': False, 'samples': 0, 'models': {}, 'errors': [],
              'validation_scope': 'PNG headers, dimensions, label classes/coordinates, pairing; no model accuracy claim'}
    selected = set(models) if models is not None else None
    with sqlite3.connect(database.as_uri() + '?mode=ro', uri=True) as db:
        rows = db.execute("SELECT model,details FROM items WHERE status='complete' ORDER BY key").fetchall()
    manifests = []
    detection_classes = {}
    detection_tasks = {}
    for model, details in rows:
        if selected is not None and model not in selected:
            continue
        try:
            records = json.loads(details)
            if not isinstance(records, list):
                raise ValueError('Committed details must be a sample list')
        except Exception as exc:
            report['errors'].append({'model': model, 'error': str(exc)})
            continue
        for record in records:
            if not isinstance(record, dict):
                report['errors'].append({'model': model, 'error': 'Invalid sample record'})
                continue
            try:
                folder = Path(record['directory']).resolve()
                if output not in folder.parents:
                    raise ValueError('Sample directory is outside the output root')
                image = folder / 'image.png'
                width, height = _dimensions(image)
                annotation = json.loads((folder / 'image.json').read_text(encoding='utf-8'))
                if (annotation['imageWidth'], annotation['imageHeight']) != (width, height):
                    raise ValueError('JSON/image dimension mismatch')
                for shape in annotation.get('shapes', []):
                    for x, y in shape.get('points', []):
                        if not (math.isfinite(x) and math.isfinite(y) and 0 <= x <= width and 0 <= y <= height):
                            raise ValueError('Out-of-range annotation coordinates')
                split = record.get('split', 'train')
                if split not in {'train', 'val'}:
                    raise ValueError('Invalid split')
                relative = Path(_part(model)) / _part(record['product']) / _part(record['face'])
                root = destination / relative
                name = hashlib.sha256(str(folder.relative_to(output)).encode()).hexdigest()[:24]
                text = folder / 'image.txt'
                classes = None
                lines = []
                metadata = annotation.get('field_metadata', {})
                obb = model == 'qr_yolo_cut' and metadata.get('yolo_task') == 'obb'
                if text.exists() or (obb and metadata.get('mode', record.get('mode')) != 'images'):
                    classes_file = folder / 'classes.json'
                    classes = json.loads(classes_file.read_text(encoding='utf-8')) if classes_file.exists() else ['qr']
                    if not isinstance(classes, list) or not classes or len(set(classes)) != len(classes):
                        raise ValueError('Invalid classes list')
                    previous = detection_classes.get(relative)
                    if previous is not None and previous != classes:
                        raise ValueError('Class mapping changed within one model/product/face')
                    task = 'obb' if obb else 'detect'
                    if relative in detection_tasks and detection_tasks[relative] != task:
                        raise ValueError('Mixed YOLO task types in one dataset')
                    if obb:
                        # Core rectangle labels cannot represent rotated QR boxes. Use the
                        # four ordered polygon vertices, preserving their actual rotation.
                        for shape in annotation.get('shapes', []):
                            points = shape.get('points', [])
                            if shape.get('shape_type') != 'polygon' or len(points) != 4:
                                raise ValueError('OBB annotations require four polygon vertices')
                            if shape.get('label') not in classes:
                                raise ValueError('Unknown OBB class')
                            crosses = []
                            for i in range(4):
                                a, b, c = points[i], points[(i+1) % 4], points[(i+2) % 4]
                                crosses.append((b[0]-a[0])*(c[1]-b[1])-(b[1]-a[1])*(c[0]-b[0]))
                            if not (all(c > 1e-8 for c in crosses) or all(c < -1e-8 for c in crosses)):
                                raise ValueError('Degenerate or self-intersecting OBB polygon')
                            coordinates = [v for x, y in points for v in (x / width, y / height)]
                            lines.append(str(classes.index(shape['label'])) + ' ' +
                                         ' '.join(f'{v:.8f}' for v in coordinates))
                    else:
                        # JSON is the editable annotation source. The committed TXT may
                        # predate a user's X-AnyLabeling corrections, including deletion.
                        for shape in annotation.get('shapes', []):
                            points = shape.get('points', [])
                            if shape.get('shape_type') != 'rectangle' or len(points) != 2:
                                raise ValueError('Detection annotations require two rectangle vertices')
                            if shape.get('label') not in classes:
                                raise ValueError('Unknown detection class')
                            (x1, y1), (x2, y2) = points
                            x1, x2 = sorted((x1, x2))
                            y1, y2 = sorted((y1, y2))
                            if x2 <= x1 or y2 <= y1:
                                raise ValueError('Zero-area detection rectangle')
                            values = ((x1+x2)/2/width, (y1+y2)/2/height,
                                      (x2-x1)/width, (y2-y1)/height)
                            lines.append(str(classes.index(shape['label'])) + ' ' +
                                         ' '.join(f'{v:.8f}' for v in values))
                    for line in ([] if obb else lines):
                        values = line.split()
                        if len(values) != 5:
                            raise ValueError('Invalid YOLO row')
                        category = int(values[0])
                        cx, cy, bw, bh = map(float, values[1:])
                        if not (0 <= category < len(classes) and all(math.isfinite(v) for v in (cx, cy, bw, bh))
                                and bw > 0 and bh > 0 and cx-bw/2 >= -1e-7 and cy-bh/2 >= -1e-7
                                and cx+bw/2 <= 1+1e-7 and cy+bh/2 <= 1+1e-7):
                            raise ValueError('Invalid YOLO class/coordinates')
                    detection_classes[relative] = classes
                    detection_tasks[relative] = task
                masks = sorted(folder.glob('mask_*.png'))
                for mask in masks:
                    if _dimensions(mask) != (width, height):
                        raise ValueError('Mask/image dimension mismatch')
                classification = annotation.get('field_metadata', {}).get('classification')
                class_name = _part(classification) if classification is not None else None
                graph = annotation.get('field_metadata', {}).get('graph')
                if graph is not None:
                    identifiers = set()
                    for point in graph.get('points', []):
                        x, y = point['x'], point['y']
                        if not (math.isfinite(x) and math.isfinite(y) and 0 <= x <= width and 0 <= y <= height):
                            raise ValueError('Invalid graph point coordinate')
                        if point['id'] in identifiers:
                            raise ValueError('Duplicate graph point ID')
                        identifiers.add(point['id'])
                    for edge in graph.get('edges', []):
                        if edge['src'] not in identifiers or edge['dst'] not in identifiers:
                            raise ValueError('Graph edge refers to absent point')
                # All samples remain review candidates regardless of prior metadata.
                annotation['imagePath'] = name + '.png'
                annotation['imageData'] = None
                annotation.setdefault('field_metadata', {}).update(
                    annotation_status='review', verified_ground_truth=False,
                    original_annotation_status=record.get('annotation_status'), export_sample_id=name)
                _copy(image, root / 'xanylabeling' / split / (name + '.png'))
                _write(root / 'xanylabeling' / split / (name + '.json'), annotation)
                if lines:
                    _copy(image, root / 'yolo_candidates' / 'images' / split / (name + '.png'))
                    label_path = root / 'yolo_candidates' / 'labels' / split / (name + '.txt')
                    label_path.parent.mkdir(parents=True, exist_ok=True)
                    label_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
                if class_name is not None:
                    _copy(image, root / 'classification_candidates' / split / class_name / (name + '.png'))
                if masks:
                    _copy(image, root / 'segmentation_candidates' / 'images' / split / (name + '.png'))
                    for mask in masks:
                        _copy(mask, root / 'segmentation_candidates' / 'masks' / mask.stem[5:] / split / (name + '.png'))
                if graph is not None:
                    _write(root / 'point_edge_candidates' / split / (name + '.json'),
                           {**graph, 'imagePath': name + '.png', 'imageWidth': width,
                            'imageHeight': height, 'field_metadata': annotation['field_metadata']})
                    _copy(image, root / 'point_edge_candidates' / split / (name + '.png'))
                stats = report['models'].setdefault(str(relative),
                    {'samples': 0, 'detection_images': 0, 'boxes': 0, 'classification_images': 0,
                     'segmentation_images': 0, 'empty_detection_review': 0})
                stats['samples'] += 1
                stats['detection_images'] += bool(lines)
                stats['boxes'] += len(lines)
                stats['classification_images'] += class_name is not None
                stats['segmentation_images'] += bool(masks)
                stats['empty_detection_review'] += classes is not None and not lines
                report['samples'] += 1
                manifests.append({**record, 'directory': str(relative / 'xanylabeling' / split),
                                  'image': name + '.png', 'annotation_status': 'review',
                                  'original_sample_directory': str(folder), 'verified_ground_truth': False})
            except Exception as exc:
                report['errors'].append({'model': model, 'sample': record.get('directory'), 'error': str(exc)})
    for relative, classes in detection_classes.items():
        root = destination / relative / 'yolo_candidates'
        for split in ('train', 'val'):
            (root / 'images' / split).mkdir(parents=True, exist_ok=True)
            (root / 'labels' / split).mkdir(parents=True, exist_ok=True)
        # Relative entries are relocatable; intentionally omit an absolute `path`.
        yaml = f'task: {detection_tasks[relative]}\ntrain: images/train\nval: images/val\nnames:\n'
        yaml += ''.join(f'  {i}: {json.dumps(name, ensure_ascii=False)}\n' for i, name in enumerate(classes))
        (root / 'data.yaml').write_text(yaml, encoding='utf-8')
    for root in destination.glob('*/*/*'):
        if not root.is_dir():
            continue
        for split in ('train', 'val'):
            xany = root / 'xanylabeling' / split
            images = {p.stem for p in xany.glob('*.png')}
            labels = {p.stem for p in xany.glob('*.json')}
            if images != labels:
                report['errors'].append({'directory': str(xany), 'error': 'Orphan image/JSON'})
            yolo = root / 'yolo_candidates'
            if {p.stem for p in (yolo/'images'/split).glob('*.png')} != {p.stem for p in (yolo/'labels'/split).glob('*.txt')}:
                report['errors'].append({'directory': str(yolo), 'error': 'Orphan YOLO image/label'})
    with (destination / 'manifest.jsonl').open('w', encoding='utf-8') as stream:
        for record in manifests:
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
    (destination / 'README.txt').write_text(
        'All exported annotations are unverified candidates for human review.\n'
        'YOLO empty detections are excluded from training folders and retained in X-AnyLabeling.\n'
        'Classification folders and segmentation masks contain pseudo-labels, not human ground truth.\n'
        'Original sources and model versions are recorded in manifest.jsonl.\n', encoding='utf-8')
    report['valid'] = not report['errors']
    _write(destination / 'validation.json', report)
    return report
