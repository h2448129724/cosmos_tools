import hashlib
import struct
import threading
import zlib
from types import SimpleNamespace

import pytest

from cosmos_toolbox.png_compress import SIGNATURE, Cancelled, _chunk, compress_file, scan_folders


def make_png(path, *, level=0, animation=False, depth=8, color=2):
    channels = {0: 1, 2: 3, 6: 4}[color]
    with path.open('wb') as stream:
        stream.write(SIGNATURE)
        _chunk(stream, b'IHDR', struct.pack('>IIBBBBB', 100, 100, depth, color, 0, 0, 0))
        _chunk(stream, b'tEXt', b'author\x00metadata preserved')
        if animation:
            _chunk(stream, b'acTL', struct.pack('>II', 1, 0))
        packed = zlib.compress((b'\x00' + b'\x30' * (100 * channels * depth // 8)) * 100, level)
        for start in range(0, len(packed), 512):
            _chunk(stream, b'IDAT', packed[start:start + 512])
        _chunk(stream, b'IEND', b'')


@pytest.mark.parametrize('depth,color', [(8, 2), (16, 2), (16, 0), (8, 6)])
@pytest.mark.parametrize('optimize', [True, False])
def test_lossless(tmp_path, depth, color, optimize):
    from PIL import Image
    path = tmp_path / 'source.png'
    make_png(path, depth=depth, color=color)
    before = path.stat()
    with Image.open(path) as image:
        pixels = image.tobytes()
    result = compress_file(path, optimize=optimize)
    assert result['saved'] > 0
    assert path.stat().st_mtime_ns == before.st_mtime_ns
    with Image.open(path) as image:
        assert image.tobytes() == pixels
        assert image.info['author'] == 'metadata preserved'
    assert list(tmp_path.iterdir()) == [path]


def test_no_growth(tmp_path):
    path = tmp_path / 'source.png'
    make_png(path)
    compress_file(path, level=9)
    before = path.read_bytes()
    result = compress_file(path, level=1)
    assert result['status'] == 'unchanged'
    assert path.read_bytes() == before


@pytest.mark.parametrize('failure', ['crc', 'animation', 'cancel', 'replace', 'changed'])
def test_failure_preserves_original(tmp_path, monkeypatch, failure):
    import cosmos_toolbox.png_compress as module
    path = tmp_path / 'source.png'
    make_png(path, animation=failure == 'animation')
    if failure == 'crc':
        path.write_bytes(path.read_bytes()[:-1] + b'!')
    control = SimpleNamespace(paused=threading.Event(), stopped=threading.Event())
    if failure == 'cancel':
        control.stopped.set()
    if failure == 'replace':
        monkeypatch.setattr(module.os, 'replace', lambda *args: (_ for _ in ()).throw(OSError('locked')))
    if failure == 'changed':
        original = module._transcode
        def change(*args, **kwargs):
            result = original(*args, **kwargs)
            if str(args[0]) != str(path):
                module.os.utime(path, ns=(path.stat().st_atime_ns, path.stat().st_mtime_ns + 1000000))
            return result
        monkeypatch.setattr(module, '_transcode', change)
    digest = hashlib.sha256(path.read_bytes()).digest()
    with pytest.raises((ValueError, OSError, Cancelled, RuntimeError)):
        compress_file(path, control=control)
    assert hashlib.sha256(path.read_bytes()).digest() == digest
    assert list(tmp_path.iterdir()) == [path]


def test_scan_deduplicates(tmp_path):
    nested = tmp_path / 'child'
    nested.mkdir()
    path = nested / 'UPPER.PNG'
    make_png(path)
    assert scan_folders([tmp_path, nested]) == [path]
    assert not scan_folders([tmp_path], recursive=False)


def test_palette_keeps_palette_and_transparency(tmp_path):
    from PIL import Image
    path = tmp_path / 'palette.png'
    image = Image.new('P', (128, 128), 1)
    image.putpalette([0, 0, 0, 255, 0, 0] + [0] * 762)
    image.save(path, transparency=0, compress_level=0)
    compress_file(path, optimize=True)
    with Image.open(path) as result:
        assert result.mode == 'P'
        assert result.info['transparency'] == 0
        assert result.convert('RGB').tobytes() == image.convert('RGB').tobytes()


def test_refilter_verification_failure_does_not_replace(tmp_path, monkeypatch):
    import cosmos_toolbox.png_compress as module
    path = tmp_path / 'image.png'
    make_png(path)
    original_bytes = path.read_bytes()
    real = module._decoded_hash
    def wrong_pixels(candidate):
        image, digest = real(candidate)
        if str(candidate) != str(path):
            digest = (digest[0], digest[1], b'wrong')
        return image, digest
    monkeypatch.setattr(module, '_decoded_hash', wrong_pixels)
    with pytest.raises(ValueError, match='逐像素'):
        compress_file(path, optimize=True)
    assert path.read_bytes() == original_bytes
    assert list(tmp_path.iterdir()) == [path]
