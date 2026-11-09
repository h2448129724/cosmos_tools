from __future__ import annotations

import os
import threading
import time

import cv2
import numpy as np
import pytest

from img_tools.core.image_cache import ImagePrefetcher, neighbours


def _write(path, value):
    cv2.imwrite(str(path), np.full((4, 4, 3), value, np.uint8))


def test_prefetched_image_is_decoded_once_and_read_only(tmp_path):
    calls = []
    paths = [tmp_path / f"{index}.png" for index in range(3)]
    for index, path in enumerate(paths):
        _write(path, index)

    def loader(path):
        calls.append(path.name)
        return cv2.imread(str(path))

    cache = ImagePrefetcher(loader)
    cache.prefetch([paths[1]])
    image = cache.get(paths[1])
    assert calls == ["1.png"]
    assert image[0, 0, 0] == 1
    assert not image.flags.writeable
    cache.close()


def test_changed_file_is_decoded_again(tmp_path):
    path = tmp_path / "a.png"
    _write(path, 10)
    cache = ImagePrefetcher(lambda p: cv2.imread(str(p)))
    assert cache.get(path)[0, 0, 0] == 10
    _write(path, 20)
    stamp = time.time() + 5
    os.utime(path, (stamp, stamp))
    assert cache.get(path)[0, 0, 0] == 20
    cache.close()


def test_capacity_evicts_oldest(tmp_path):
    calls = []
    cache = ImagePrefetcher(lambda p: calls.append(p.name) or np.zeros((1, 1, 3), np.uint8), capacity=2)
    paths = [tmp_path / f"{i}.png" for i in range(3)]
    for path in paths:
        path.write_bytes(b"x")
        cache.get(path)
    cache.get(paths[0])
    assert calls == ["0.png", "1.png", "2.png", "0.png"]
    cache.close()


def test_get_waits_for_inflight_prefetch_instead_of_decoding_twice(tmp_path):
    path = tmp_path / "slow.png"
    path.write_bytes(b"x")
    started, release, calls = threading.Event(), threading.Event(), []

    def loader(p):
        calls.append(p)
        started.set()
        release.wait(5)
        return np.zeros((1, 1, 3), np.uint8)

    cache = ImagePrefetcher(loader)
    cache.prefetch([path])
    assert started.wait(5)
    threading.Timer(0.05, release.set).start()
    assert cache.get(path) is not None
    assert len(calls) == 1
    cache.close()


@pytest.mark.parametrize(("index", "expected"), [(0, [1]), (2, [3, 1]), (4, [3])])
def test_neighbours_prefer_forward(index, expected):
    assert neighbours([0, 1, 2, 3, 4], index) == expected


def test_widget_hide_releases_cache(tmp_path):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication, QWidget

    from img_tools.ui.qt_image import bind_cache_to_widget

    QApplication.instance() or QApplication([])
    path = tmp_path / "a.png"
    _write(path, 1)
    calls = []
    widget = QWidget()
    cache = bind_cache_to_widget(widget, ImagePrefetcher(lambda p: calls.append(p) or cv2.imread(str(p))))
    widget.show()
    cache.get(path)
    cache.get(path)
    widget.hide()
    cache.get(path)
    assert len(calls) == 2
    widget.deleteLater()
