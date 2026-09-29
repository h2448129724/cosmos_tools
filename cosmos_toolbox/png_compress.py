"""PNG compression shell: preserve decoded pixels and every non-IDAT chunk.

The low-memory path uses only zlib; optional refiltering uses existing OpenCV.
Unsupported animation, unknown critical chunks and malformed files are refused.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
import struct
import tempfile
import time
import zlib

SIGNATURE = b'\x89PNG\r\n\x1a\n'
BLOCK = 1024 * 1024


class Cancelled(Exception):
    pass


def checkpoint(control):
    if control is None:
        return
    while control.paused.is_set():
        if control.stopped.wait(.1):
            raise Cancelled()
    if control.stopped.is_set():
        raise Cancelled()


def scan_folders(folders, recursive=True, control=None):
    found = {}
    for folder in folders:
        root = Path(folder).resolve(strict=True)
        if not root.is_dir():
            raise ValueError(f'不是文件夹：{root}')
        for path in root.rglob('*') if recursive else root.iterdir():
            checkpoint(control)
            if path.suffix.lower() == '.png' and path.is_file() and not path.is_symlink():
                found[str(path.resolve()).casefold()] = path
    return sorted(found.values())


def _chunk(output, kind, data):
    output.write(struct.pack('>I', len(data)) + kind + data)
    output.write(struct.pack('>I', zlib.crc32(kind + data) & 0xffffffff))


def _transcode(source, output=None, level=6, control=None):
    pixels, metadata = hashlib.sha256(), hashlib.sha256()
    decoder = zlib.decompressobj()
    encoder = zlib.compressobj(level) if output else None
    seen_idat = ended_idat = False
    header = None
    raw_size = 0
    with open(source, 'rb') as stream:
        if stream.read(8) != SIGNATURE:
            raise ValueError('不是有效 PNG')
        if output:
            output.write(SIGNATURE)
        while True:
            checkpoint(control)
            prefix = stream.read(8)
            if len(prefix) != 8:
                raise ValueError('PNG 不完整')
            size, kind = struct.unpack('>I4s', prefix)
            if size > 64 * BLOCK:
                raise ValueError('暂不支持超过 64 MiB 的单个 PNG chunk')
            data = stream.read(size)
            crc = stream.read(4)
            if len(data) != size or len(crc) != 4 or struct.unpack('>I', crc)[0] != zlib.crc32(kind + data) & 0xffffffff:
                raise ValueError('PNG chunk CRC 校验失败')
            if header is None:
                if kind != b'IHDR' or size != 13:
                    raise ValueError('缺少 PNG IHDR')
                header = data
            elif kind == b'IHDR':
                raise ValueError('重复 IHDR')
            if kind in (b'acTL', b'fcTL', b'fdAT'):
                raise ValueError('暂不支持动画 PNG，原文件保留')
            if not kind[0] & 32 and kind not in (b'IHDR', b'PLTE', b'IDAT', b'IEND'):
                raise ValueError('未知 PNG 关键 chunk')
            if kind == b'IDAT':
                if ended_idat:
                    raise ValueError('IDAT 不连续')
                seen_idat = True
                pending = data
                while pending:
                    checkpoint(control)
                    raw = decoder.decompress(pending, BLOCK)
                    pending = decoder.unconsumed_tail
                    pixels.update(raw)
                    raw_size += len(raw)
                    if decoder.unused_data:
                        raise ValueError('IDAT 含额外压缩流')
                    if encoder:
                        packed = encoder.compress(raw)
                        if packed:
                            _chunk(output, b'IDAT', packed)
            else:
                if seen_idat and not ended_idat:
                    if not decoder.eof:
                        raise ValueError('IDAT 压缩流不完整')
                    if encoder:
                        _chunk(output, b'IDAT', encoder.flush())
                    ended_idat = True
                metadata.update(prefix + data + crc)
                if output:
                    output.write(prefix + data + crc)
            if kind == b'IEND':
                if size or not seen_idat or stream.read(1):
                    raise ValueError('PNG 结束结构异常')
                break
    width, height, depth, color, comp, filt, interlace = struct.unpack('>IIBBBBB', header)
    channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}.get(color)
    allowed = {0: (1, 2, 4, 8, 16), 2: (8, 16), 3: (1, 2, 4, 8), 4: (8, 16), 6: (8, 16)}
    if not width or not height or depth not in allowed.get(color, ()) or comp or filt or interlace:
        raise ValueError('不支持的 PNG 头部或交错 PNG，原文件保留')
    if raw_size != height * (1 + (width * channels * depth + 7) // 8):
        raise ValueError('PNG 像素流长度不匹配')
    return pixels.digest(), metadata.digest(), (width, height, depth, color)


def _decoded_hash(path):
    import cv2
    import numpy as np
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError('图片解码失败')
    return image, (image.shape, image.dtype.str, hashlib.sha256(image).digest())


def _refilter(source, output, level, control):
    """Reuse source metadata verbatim; replace only IDAT with OpenCV encoding."""
    import cv2
    image, digest = _decoded_hash(source)
    checkpoint(control)
    ok, packed = cv2.imencode('.png', image, [cv2.IMWRITE_PNG_COMPRESSION, level])
    del image
    if not ok:
        raise ValueError('PNG 编码失败')
    checkpoint(control)
    encoded = memoryview(packed)
    encoded_header = bytes(encoded[16:29])
    output.write(SIGNATURE)
    inserted = False
    with open(source, 'rb') as stream:
        stream.read(8)
        while True:
            checkpoint(control)
            prefix = stream.read(8)
            if len(prefix) != 8:
                raise ValueError('源文件发生变化')
            size, kind = struct.unpack('>I4s', prefix)
            if size > 64 * BLOCK:
                raise ValueError('源 chunk 过大')
            data, crc = stream.read(size), stream.read(4)
            if len(data) != size or len(crc) != 4:
                raise ValueError('源文件发生变化')
            if kind == b'IHDR' and data != encoded_header:
                raise ValueError('编码器改变了 PNG 色彩类型或位深')
            if kind == b'IDAT':
                if not inserted:
                    position = 8
                    while position < len(encoded):
                        checkpoint(control)
                        count = struct.unpack('>I', encoded[position:position + 4])[0]
                        if bytes(encoded[position + 4:position + 8]) == b'IDAT':
                            output.write(encoded[position:position + count + 12])
                        position += count + 12
                    inserted = True
            else:
                output.write(prefix + data + crc)
            if kind == b'IEND':
                break
    return digest


def compress_file(path, *, level=6, control=None, optimize=False):
    """Stage beside source, verify, replace only if smaller. No backup retained."""
    path = Path(path)
    if path.is_symlink() or path.suffix.lower() != '.png':
        raise ValueError('只支持普通 PNG 文件')
    before = path.stat()
    if before.st_nlink > 1:
        raise ValueError('不处理具有硬链接的文件')
    if level not in (1, 6, 9):
        raise ValueError('压缩级别必须为 1、6 或 9')
    if shutil.disk_usage(path.parent).free < before.st_size + 16 * BLOCK:
        raise OSError('剩余空间不足以安全生成临时文件')
    started = time.monotonic()
    fd, name = tempfile.mkstemp(prefix='.cosmos-png-', suffix='.tmp', dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, 'wb') as output:
            expected = _transcode(path, None if optimize else output, level, control)
            # Indexed and low-bit grayscale retain their exact original encoding.
            refilter = optimize and expected[2][2] >= 8 and expected[2][3] in (0, 2, 6)
            if refilter:
                decoded_expected = _refilter(path, output, level, control)
            elif optimize:
                _transcode(path, output, level, control)
            output.flush()
            os.fsync(output.fileno())
        actual = _transcode(temporary, control=control)
        mismatch = actual[1:] != expected[1:] if refilter else actual != expected
        if mismatch:
            raise ValueError('压缩后的像素流或 PNG 元数据校验失败')
        if refilter:
            decoded, decoded_actual = _decoded_hash(temporary)
            del decoded
            if decoded_actual != decoded_expected:
                raise ValueError('重新编码后逐像素校验失败')
        checkpoint(control)
        current = path.stat()
        if (current.st_size, current.st_mtime_ns, current.st_ino) != (before.st_size, before.st_mtime_ns, before.st_ino):
            raise RuntimeError('处理期间原文件已改变，不替换')
        after = temporary.stat().st_size
        replaced = after < before.st_size
        if replaced:
            shutil.copystat(path, temporary)
            os.replace(temporary, path)
        return dict(path=str(path), status='compressed' if replaced else 'unchanged',
                    before=before.st_size, after=after if replaced else before.st_size,
                    saved=before.st_size - after if replaced else 0,
                    seconds=round(time.monotonic() - started, 3), image=expected[2])
    finally:
        temporary.unlink(missing_ok=True)


def _publish_jpeg(temporary, target):
    # Windows rename refuses an existing destination (including on SMB).
    # POSIX rename would overwrite, so use exclusive hard-link publication there.
    if os.name == 'nt':
        os.rename(temporary, target)
    else:
        os.link(temporary, target)
        temporary.unlink()


def convert_jpeg(path, *, quality=95, delete_source=False, control=None):
    """Publish a verified JPEG without overwriting existing files, then optionally delete PNG."""
    import cv2
    import numpy as np

    path = Path(path)
    if path.suffix.lower() != '.png' or path.is_symlink():
        raise ValueError('只支持普通 PNG 文件')
    if not 1 <= quality <= 100:
        raise ValueError('JPG 质量必须为 1–100')
    before = path.stat()
    if before.st_nlink > 1:
        raise ValueError('不处理具有硬链接的文件')
    target = path.with_suffix('.jpg')
    if target.exists() or target.is_symlink():
        raise FileExistsError(f'目标已存在，不覆盖也不删除原图：{target}')
    if shutil.disk_usage(path.parent).free < before.st_size + 16 * BLOCK:
        raise OSError('剩余空间不足以安全转换')
    started = time.monotonic()
    info = _transcode(path, control=control)[2]
    if info[2] != 8 or info[3] not in (0, 2):
        raise ValueError('JPG 转换仅支持 8 位灰度/RGB PNG；不自动降低位深或丢弃透明度')
    image, _ = _decoded_hash(path)
    if image.dtype != np.uint8 or (image.ndim == 3 and image.shape[2] != 3):
        raise ValueError('图片含透明通道或不支持的像素格式')
    shape = image.shape
    checkpoint(control)
    params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    if hasattr(cv2, 'IMWRITE_JPEG_SAMPLING_FACTOR'):
        params += [cv2.IMWRITE_JPEG_SAMPLING_FACTOR, cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444]
    ok, packed = cv2.imencode('.jpg', image, params)
    del image
    if not ok:
        raise ValueError('JPG 编码失败')
    checkpoint(control)
    fd, name = tempfile.mkstemp(prefix='.cosmos-jpg-', suffix='.tmp', dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, 'wb') as output:
            output.write(memoryview(packed))
            output.flush()
            os.fsync(output.fileno())
        del packed
        decoded, _ = _decoded_hash(temporary)
        if decoded.shape != shape or decoded.dtype != np.uint8:
            raise ValueError('JPG 解码或尺寸校验失败')
        del decoded
        checkpoint(control)
        current = path.stat()
        identity = lambda stat: (stat.st_size, stat.st_mtime_ns, stat.st_ino)
        if identity(current) != identity(before):
            raise RuntimeError('处理期间源 PNG 已变化，不发布 JPG')
        after = temporary.stat().st_size
        if after >= before.st_size:
            return dict(path=str(path), status='unchanged', before=before.st_size, after=before.st_size,
                        saved=0, potential_saved=0, source_deleted=False)
        shutil.copystat(path, temporary)
        _publish_jpeg(temporary, target)
        deleted = False
        if delete_source:
            checkpoint(control)
            if identity(path.stat()) != identity(before):
                raise RuntimeError(f'PNG 已变化，已保留 PNG 和 JPG：{target}')
            path.unlink()
            deleted = True
        return dict(path=str(path), output=str(target), status='converted', quality=quality,
                    before=before.st_size, after=after, potential_saved=before.st_size-after,
                    saved=before.st_size-after if deleted else 0, source_deleted=deleted,
                    seconds=round(time.monotonic()-started, 3), metadata_preserved=False)
    finally:
        temporary.unlink(missing_ok=True)
