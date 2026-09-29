import pytest
from PIL import Image
from cosmos_toolbox.png_compress import convert_jpeg


@pytest.mark.parametrize('delete', [False, True])
def test_convert(tmp_path, delete):
    source = tmp_path / '中文.png'
    Image.new('RGB', (512, 512), (220, 30, 10)).save(source, compress_level=0)
    result = convert_jpeg(source, delete_source=delete)
    assert source.exists() != delete
    assert result['potential_saved'] > 0
    assert bool(result['saved']) == delete
    with Image.open(source.with_suffix('.jpg')) as image:
        assert image.size == (512, 512)
        r, g, b = image.getpixel((100, 100))
        assert r > 200 and b < 20


def test_existing_not_overwritten(tmp_path):
    source = tmp_path / 'image.png'
    Image.new('RGB', (512, 512)).save(source)
    target = source.with_suffix('.jpg')
    target.write_bytes(b'existing')
    with pytest.raises(FileExistsError):
        convert_jpeg(source, delete_source=True)
    assert source.exists()
    assert target.read_bytes() == b'existing'


@pytest.mark.parametrize('mode', ['RGBA', 'I;16'])
def test_unsupported_preserved(tmp_path, mode):
    source = tmp_path / 'image.png'
    Image.new(mode, (128, 128)).save(source)
    with pytest.raises(ValueError):
        convert_jpeg(source, delete_source=True)
    assert source.exists()
    assert not source.with_suffix('.jpg').exists()


def test_publish_failure_preserves_png(tmp_path, monkeypatch):
    import cosmos_toolbox.png_compress as module
    source = tmp_path / 'image.png'
    Image.new('RGB', (512, 512)).save(source, compress_level=0)
    def fail(*args):
        raise OSError('unsupported link')
    monkeypatch.setattr(module, '_publish_jpeg', fail)
    with pytest.raises(OSError):
        convert_jpeg(source, delete_source=True)
    assert list(tmp_path.iterdir()) == [source]


def test_publish_race_does_not_overwrite(tmp_path):
    from cosmos_toolbox.png_compress import _publish_jpeg
    temporary = tmp_path / 'temp'
    target = tmp_path / 'existing.jpg'
    temporary.write_bytes(b'new')
    target.write_bytes(b'existing')
    with pytest.raises(OSError):
        _publish_jpeg(temporary, target)
    assert target.read_bytes() == b'existing'
    assert temporary.read_bytes() == b'new'
