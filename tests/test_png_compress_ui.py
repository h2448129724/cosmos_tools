import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from PySide6.QtWidgets import QApplication
from cosmos_toolbox.png_compress_ui import PngCompressPage, PngWorker


def test_page_and_scan(tmp_path):
    app = QApplication.instance() or QApplication([])
    page = PngCompressPage()
    page.resize(960, 640)
    page.show()
    app.processEvents()
    assert page.level.currentData() == 6
    assert page.recursive.isChecked()
    assert not page.stop.isEnabled()
    assert page.mode.currentData() == 'jpg'
    assert page.quality.value() == 95
    assert not page.delete_source.isChecked()
    assert not page.optimize.isVisible()
    assert page.quality.isVisible()
    page.mode.setCurrentIndex(0)
    assert page.start.geometry().height() > 10
    results = []
    worker = PngWorker(dict(sources=[str(tmp_path)], recursive=True, level=6), True)
    worker.result.connect(results.append)
    worker.run()
    assert results[0]['count'] == 0
    assert results[0]['failed'] == 0
    page.close()


def test_multiple_folders_and_quality(tmp_path):
    app = QApplication.instance() or QApplication([])
    page = PngCompressPage()
    first, second = str(tmp_path / 'first'), str(tmp_path / 'second')
    page._append_sources([first])
    page._append_sources([first, second])
    assert page.sources.toPlainText().splitlines() == [first, second]
    for quality in (1, 70, 80, 85, 90, 95, 100):
        page.quality.setValue(quality)
        assert page.quality.value() == quality
    page.close()


def test_jpg_worker_uses_selected_quality_and_all_folders(tmp_path, monkeypatch):
    from cosmos_toolbox import png_compress
    folders = [tmp_path / 'one', tmp_path / 'two']
    for folder in folders:
        folder.mkdir()
        (folder / 'sample.png').write_bytes(b'fixture')
    calls = []
    def convert(path, **options):
        calls.append((path, options))
        return dict(status='converted', saved=0, potential_saved=1)
    monkeypatch.setattr(png_compress, 'convert_jpeg', convert)
    worker = PngWorker(dict(sources=[str(folder) for folder in folders], recursive=True,
                            mode='jpg', quality=85, delete_source=False), False)
    worker.run()
    assert len(calls) == 2
    assert all(options['quality'] == 85 and options['delete_source'] is False for _, options in calls)


def test_worker_continues_after_bad_file(tmp_path):
    (tmp_path / 'bad.png').write_bytes(b'bad')
    worker = PngWorker(dict(sources=[str(tmp_path)], recursive=True, level=6), False)
    results = []
    worker.result.connect(results.append)
    worker.run()
    assert results[0]['failed'] == 1
    assert results[0]['processed'] == 1
