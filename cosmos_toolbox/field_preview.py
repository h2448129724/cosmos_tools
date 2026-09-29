"""Human-review overlays, separate from training images and editable labels."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def _font(size):
    for name in ('C:/Windows/Fonts/msyh.ttc', 'DejaVuSans.ttf'):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def render_preview(source, destination, annotation, *, model, product, face, ng_filter=None):
    """Draw the current JSON, never stale TXT/masks or source-image predictions.

    The image is downscaled to at most 1600px on its longest side. The separate
    header is not part of the annotation coordinate space. No model is loaded.
    """
    header = 112
    font = _font(16)
    with Image.open(source) as opened:
        original_size = opened.size
        frame = opened.convert('RGB')
    frame.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
    sx, sy = frame.width / original_size[0], frame.height / original_size[1]
    canvas = Image.new('RGB', (max(640, frame.width), frame.height + header), '#202a35')
    canvas.paste(frame, (0, header))
    draw = ImageDraw.Draw(canvas, 'RGBA')

    def fit(text, width):
        text = str(text).replace('\n', ' ')
        if draw.textlength(text, font=font) <= width:
            return text
        while text and draw.textlength(text + '…', font=font) > width:
            text = text[:-1]
        return text + '…'

    metadata = annotation.get('field_metadata', {})
    classification = metadata.get('classification')
    shapes = annotation.get('shapes', [])
    decision = ng_filter or metadata.get('ng_filter') or {}
    checks = [f"{item.get('item', '')}: {item.get('state', '')}" for item in decision.get('checks', [])]
    titles = [
        f'{model} | {product} | {face} | PREVIEW / 待人工复核',
        f'类别: {classification}' if classification is not None else f'标注数: {len(shapes)}' + ('（无标注，不能视为负样本）' if not shapes else ''),
        '检查: ' + ('; '.join(checks) if checks else '未启用 NG 筛选 / 无检查记录'),
        '原图: ' + str(metadata.get('source') or Path(source).name),
    ]
    for index, text in enumerate(titles):
        draw.text((10, 5 + index * 26), fit(text, canvas.width - 20), font=font, fill='white')
    palette = [(255, 193, 7), (0, 220, 175), (255, 108, 120), (109, 180, 255), (208, 150, 255)]
    labels = sorted({str(shape.get('label', '')) for shape in shapes})
    if classification is None and labels:
        legend_x = int(draw.textlength(titles[1], font=font)) + 28
        for label in labels:
            color = palette[int(hashlib.sha256(label.encode()).hexdigest()[:8], 16) % len(palette)]
            width = int(draw.textlength(label, font=font)) + 20
            if legend_x + width > canvas.width - 10:
                break
            draw.text((legend_x, 31), label, font=font, fill=(*color, 255))
            legend_x += width
    for shape in shapes:
        points = [(max(0, min(frame.width - 1, round(x * sx))),
                   header + max(0, min(frame.height - 1, round(y * sy))))
                  for x, y in shape.get('points', [])]
        if not points:
            continue
        label = str(shape.get('label', ''))
        color = palette[int(hashlib.sha256(label.encode()).hexdigest()[:8], 16) % len(palette)]
        kind = shape.get('shape_type', 'polygon')
        if kind == 'rectangle' and len(points) == 2:
            a, b = points
            draw.rectangle((min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1])), outline=(*color, 255), width=3)
        elif kind == 'polygon' and len(points) >= 3:
            draw.polygon(points, fill=(*color, 55))
            draw.line(points + [points[0]], fill=(*color, 255), width=2)
        elif kind in ('line', 'linestrip') and len(points) >= 2:
            draw.line(points, fill=(*color, 255), width=2)
        elif kind == 'point':
            x, y = points[0]
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=(*color, 255))
        # Avoid obscuring dense point/edge datasets with a label at every node.
        if kind not in ('point', 'line', 'linestrip') and not (kind == 'polygon' and len(shapes) > 20):
            x, y = points[0]
            label = fit(label, canvas.width - 12)
            x = min(x, max(0, canvas.width - int(draw.textlength(label, font=font)) - 8))
            y = max(header, y - 22)
            bounds = draw.textbbox((x + 3, y), label, font=font)
            draw.rectangle((x, y, bounds[2] + 3, bounds[3] + 2), fill=(0, 0, 0, 190))
            draw.text((x + 3, y), label, font=font, fill=(*color, 255))
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, 'JPEG', quality=90)
    return {'width': canvas.width, 'height': canvas.height, 'scale': [sx, sy]}


def backfill_previews(snapshot, *, overwrite=False):
    """Add previews to an existing export without changing images, labels or manifest."""
    root = Path(snapshot).resolve()
    report = {'generated': 0, 'skipped': 0, 'errors': [], 'files': []}
    for line in (root / 'manifest.jsonl').read_text(encoding='utf-8').splitlines():
        record = json.loads(line)
        try:
            source = (root / record['directory'] / record['image']).resolve()
            source.relative_to(root)
            parts = Path(record['directory']).parts
            model = record.get('model') or parts[1 if parts[0] == '_review' else 0]
            if not model or model in ('.', '..') or any(c in model for c in '/\\<>:"|?*'):
                raise ValueError('Invalid model name')
            target = (root / 'preview' / model / (source.stem + '.jpg')).resolve()
            target.relative_to(root)
            if target.exists() and not overwrite:
                report['skipped'] += 1
                continue
            label = source.with_suffix('.json')
            if label.is_file():
                annotation = json.loads(label.read_text(encoding='utf-8'))
            elif model == 'hook_detector' and record.get('classification') in ('up', 'down'):
                annotation = {'shapes': [], 'field_metadata': record}
            else:
                raise ValueError('Missing annotation; cannot create an accurate preview')
            render_preview(source, target, annotation, model=model, product=record['product'],
                           face=record['face'], ng_filter=record.get('ng_filter'))
            report['generated'] += 1
            report['files'].append(str(target.relative_to(root)))
        except Exception as exc:
            report['errors'].append({'image': record.get('image'), 'error': str(exc)})
    preview = (root / 'preview').resolve()
    preview.relative_to(root)
    preview.mkdir(exist_ok=True)
    (preview / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return report


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Add review overlays to an existing dataset export')
    parser.add_argument('snapshot')
    args = parser.parse_args()
    print(json.dumps(backfill_previews(args.snapshot), ensure_ascii=False, indent=2))
