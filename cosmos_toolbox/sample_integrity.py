"""Read-only validation of committed images and editable annotations."""
from __future__ import annotations

import json
import math
import struct
import zlib
from pathlib import Path


def image_dimensions(path):
    """Decode every pixel; a valid header alone does not prove image integrity."""
    import cv2
    import numpy as np
    path = Path(path)
    if path.suffix.lower() == '.png':
        _verify_png(path)
    try:
        image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    except cv2.error as exc:
        raise ValueError(f'图片解码失败：{path}') from exc
    if image is None or not image.size:
        raise ValueError(f'图片损坏或无法完整解码：{path}')
    height, width = image.shape[:2]
    return width, height


def _verify_png(path):
    """Check the complete container as decoders may tolerate a missing IEND."""
    with path.open('rb') as stream:
        if stream.read(8) != b'\x89PNG\r\n\x1a\n':
            raise ValueError(f'Invalid PNG signature: {path}')
        first, seen_data = True, False
        while True:
            prefix = stream.read(8)
            if len(prefix) != 8:
                raise ValueError(f'Incomplete PNG: {path}')
            size, kind = struct.unpack('>I4s', prefix)
            if first and (kind != b'IHDR' or size != 13):
                raise ValueError('Missing PNG IHDR')
            first = False
            checksum = zlib.crc32(kind)
            remaining = size
            while remaining:
                chunk = stream.read(min(remaining, 1024 * 1024))
                if not chunk:
                    raise ValueError(f'Incomplete PNG chunk: {path}')
                remaining -= len(chunk)
                checksum = zlib.crc32(chunk, checksum)
            crc = stream.read(4)
            if len(crc) != 4 or struct.unpack('>I', crc)[0] != checksum & 0xffffffff:
                raise ValueError(f'Invalid PNG CRC: {path}')
            seen_data = seen_data or kind == b'IDAT'
            if kind == b'IEND':
                if size or not seen_data or stream.read(1):
                    raise ValueError(f'Invalid PNG ending: {path}')
                return


def validate_sample(record):
    folder = Path(record['directory'])
    for name in record.get('files', ['image.png', 'image.json']):
        path = folder / name
        if not path.is_file():
            raise ValueError(f'样本缺少文件：{path}')
        if path.suffix.lower() == '.json':
            json.loads(path.read_text(encoding='utf-8'))
    width, height = image_dimensions(folder / 'image.png')
    annotation = json.loads((folder / 'image.json').read_text(encoding='utf-8'))
    if (annotation['imageWidth'], annotation['imageHeight']) != (width, height):
        raise ValueError('JSON/image dimension mismatch')
    for shape in annotation.get('shapes', []):
        for x, y in shape.get('points', []):
            if not (math.isfinite(x) and math.isfinite(y) and 0 <= x <= width and 0 <= y <= height):
                raise ValueError('Out-of-range annotation coordinates')
    for mask in folder.glob('mask_*.png'):
        if image_dimensions(mask) != (width, height):
            raise ValueError('Mask/image dimension mismatch')
    return width, height


def samples_intact(records):
    try:
        if not records:
            return False
        for record in records:
            validate_sample(record)
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False
