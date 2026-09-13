"""Portable, versioned candidate exports from committed field collection samples."""
from __future__ import annotations

import hashlib
import json
import math
import shutil
import sqlite3
import struct
import time
from datetime import datetime
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


def flatten_snapshot(snapshot, destination=None):
    """Create portable per-model training folders from a validated candidate export."""
    snapshot = Path(snapshot).resolve()
    validation = json.loads((snapshot / 'validation.json').read_text(encoding='utf-8'))
    if not validation.get('valid'):
        raise ValueError('Candidate export must pass validation before flattening')
    destination = Path(destination).resolve() if destination else snapshot / 'training_datasets'
    destination.mkdir(parents=True, exist_ok=False)
    mappings = {'yolo_candidates': '', 'classification_candidates': '',
                'segmentation_candidates': '', 'point_edge_candidates': ''}
    schemas = {}
    counts = {}
    for model in sorted(snapshot.iterdir()):
        if not model.is_dir() or model == destination:
            continue
        for face_root in sorted(model.glob('*/*')):
            if not face_root.is_dir():
                continue
            for format_name in (*mappings, 'xanylabeling'):
                source = face_root / format_name
                if not source.is_dir():
                    continue
                root = (destination / '_review' / model.name if format_name == 'xanylabeling'
                        else destination / model.name)
                if format_name != 'xanylabeling':
                    previous = schemas.setdefault(model.name, format_name)
                    if previous != format_name:
                        raise ValueError(f'Mixed training formats for {model.name}')
                for path in sorted(source.rglob('*')):
                    relative = path.relative_to(source)
                    target = root / relative
                    if path.is_dir():
                        target.mkdir(parents=True, exist_ok=True)
                        continue
                    if target.exists():
                        if relative == Path('data.yaml') and target.read_bytes() == path.read_bytes():
                            continue
                        raise ValueError(f'Duplicate export target: {target}')
                    _copy(path, target)
                    if path.suffix == '.png' and format_name != 'xanylabeling' and 'masks' not in relative.parts:
                        counts[model.name] = counts.get(model.name, 0) + 1
    # Keep product, face, original path and split as metadata, not directory levels.
    with (destination / 'manifest.jsonl').open('w', encoding='utf-8') as stream:
        for line in (snapshot / 'manifest.jsonl').read_text(encoding='utf-8').splitlines():
            record = json.loads(line)
            original = Path(record['directory'])
            model = original.parts[0]
            record['directory'] = str(Path('_review') / model / record.get('split', 'train'))
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
    (destination / 'README.txt').write_text(
        'Each model folder is one dataset; product and face are retained in manifest.jsonl.\n'
        'Detection: images/{train,val}, labels/{train,val}, data.yaml.\n'
        'Classification: {train,val}/class. Segmentation: images/{train,val}, masks/class/{train,val}.\n'
        'Points/connections: {train,val} with paired PNG and JSON.\n'
        '_review contains X-AnyLabeling image/JSON pairs, including empty detections.\n'
        'Labels are unverified model candidates, not human ground truth.\n', encoding='utf-8')
    report = {'directory': str(destination), 'models': counts, 'formats': schemas,
              'verified_ground_truth': False}
    _write(destination / 'report.json', report)
    return report


def export(output, models=None, *, xany_only=True):
    """Export a fresh snapshot; never overwrite earlier exports or source annotations.

    The returned report describes structural validation, not annotation accuracy.
    Empty detections are available for review only, never automatic negatives.
    """
    output = Path(output).resolve()
    database = output / 'run.db'
    if not database.is_file():
        raise ValueError('run.db does not exist')
    # Local wall time is readable on site; nanoseconds retain collision resistance.
    stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    destination = output / 'exports' / f'candidates_{stamp}_{time.time_ns()}'
    destination.mkdir(parents=True, exist_ok=False)
    report = {'directory': str(destination), 'annotation_status': 'review',
              'format': 'xanylabeling' if xany_only else 'multi_format',
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
                if xany_only:
                    empty_detection = classes is not None and not lines
                    annotation_relative = ((Path('_review') / _part(model)) if empty_detection
                                           else Path(_part(model)))
                    annotation['field_metadata'].update(product=record['product'], face=record['face'],
                        split=split, source=record.get('source'), empty_detection=empty_detection)
                    if class_name is not None:
                        annotation.setdefault('flags', {})[class_name] = True
                else:
                    annotation_relative = relative / 'xanylabeling' / split
                _copy(image, destination / annotation_relative / (name + '.png'))
                _write(destination / annotation_relative / (name + '.json'), annotation)
                if lines and not xany_only:
                    _copy(image, root / 'yolo_candidates' / 'images' / split / (name + '.png'))
                    label_path = root / 'yolo_candidates' / 'labels' / split / (name + '.txt')
                    label_path.parent.mkdir(parents=True, exist_ok=True)
                    label_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
                if class_name is not None and not xany_only:
                    _copy(image, root / 'classification_candidates' / split / class_name / (name + '.png'))
                if masks and not xany_only:
                    _copy(image, root / 'segmentation_candidates' / 'images' / split / (name + '.png'))
                    for mask in masks:
                        _copy(mask, root / 'segmentation_candidates' / 'masks' / mask.stem[5:] / split / (name + '.png'))
                if graph is not None and not xany_only:
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
                manifests.append({**record, 'directory': str(annotation_relative),
                                  'image': name + '.png', 'annotation_status': 'review',
                                  'original_sample_directory': str(folder), 'verified_ground_truth': False})
            except Exception as exc:
                report['errors'].append({'model': model, 'sample': record.get('directory'), 'error': str(exc)})
    for relative, classes in detection_classes.items():
        if xany_only:
            continue
        root = destination / relative / 'yolo_candidates'
        for split in ('train', 'val'):
            (root / 'images' / split).mkdir(parents=True, exist_ok=True)
            (root / 'labels' / split).mkdir(parents=True, exist_ok=True)
        # Relative entries are relocatable; intentionally omit an absolute `path`.
        yaml = f'task: {detection_tasks[relative]}\ntrain: images/train\nval: images/val\nnames:\n'
        yaml += ''.join(f'  {i}: {json.dumps(name, ensure_ascii=False)}\n' for i, name in enumerate(classes))
        (root / 'data.yaml').write_text(yaml, encoding='utf-8')
    if xany_only:
        for relative in {record['directory'] for record in manifests}:
            folder = destination / relative
            if {p.stem for p in folder.glob('*.png')} != {p.stem for p in folder.glob('*.json')}:
                report['errors'].append({'directory': str(folder), 'error': 'Orphan image/JSON'})
    for root in ([] if xany_only else destination.glob('*/*/*')):
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
        ('One PNG and matching X-AnyLabeling JSON per sample; no duplicate training export.\n'
         'Each model folder contains labeled candidates; _review/model contains empty detections. No train/val directories.\n'
         'Segmentation polygons are in shapes; classification is in flags/field_metadata; graphs in field_metadata.\n'
         'All labels are unverified. Source, product, face and split are preserved in manifest.jsonl.\n'
         'Exact raster masks remain in the original working dataset; polygon annotations are not lossless masks.\n') if xany_only else (
        'All exported annotations are unverified candidates for human review.\n'
        'YOLO empty detections are excluded from training folders and retained in X-AnyLabeling.\n'
        'Classification folders and segmentation masks contain pseudo-labels, not human ground truth.\n'
        'Original sources and model versions are recorded in manifest.jsonl.\n'), encoding='utf-8')
    report['valid'] = not report['errors']
    _write(destination / 'validation.json', report)
    if report['valid'] and not xany_only:
        report['training_dataset'] = flatten_snapshot(destination)
        _write(destination / 'validation.json', report)
    return report
