"""Small decoded-image cache with background prefetch for one-by-one browsing.

Pages that step through a folder (A/D, next/previous) ask for the current
image with :meth:`ImagePrefetcher.get` and then call :meth:`prefetch` with the
neighbours.  Decoding runs on worker threads (OpenCV releases the GIL), so the
next image is usually already decoded when the user moves on.

Entries are keyed by path and validated against the file's size and mtime, so
an image replaced or moved on disk is decoded again rather than served stale.
Cached arrays are read-only; callers that draw on an image must copy it.
"""
from __future__ import annotations

import os
import threading
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from .image_io import read_image

Loader = Callable[[Path], "np.ndarray | None"]


def _signature(path: Path) -> tuple[int, int] | None:
    try:
        stat = os.stat(path)
    except OSError:
        return None
    return stat.st_size, stat.st_mtime_ns


def _load_frozen(loader: Loader, path: Path) -> tuple[tuple[int, int] | None, np.ndarray | None]:
    signature = _signature(path)
    image = loader(path)
    if image is not None:
        image.setflags(write=False)
    return signature, image


class ImagePrefetcher:
    def __init__(self, loader: Loader | None = None, *, capacity: int = 3, workers: int = 2) -> None:
        self._loader: Loader = loader or (lambda path: read_image(path))
        self._capacity = max(1, int(capacity))
        self._executor = ThreadPoolExecutor(max_workers=max(1, int(workers)), thread_name_prefix="image-prefetch")
        self._entries: OrderedDict[Path, Future] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, path: str | Path) -> np.ndarray | None:
        """Return the decoded image, waiting for an in-flight prefetch if any."""
        key = Path(path)
        with self._lock:
            future = self._entries.get(key)
            if future is not None:
                self._entries.move_to_end(key)
        if future is not None:
            try:
                signature, image = future.result()
            except Exception:
                signature, image = None, None
            if image is not None and signature == _signature(key):
                return image
        signature, image = _load_frozen(self._loader, key)
        done: Future = Future()
        done.set_result((signature, image))
        self._store(key, done)
        return image

    def prefetch(self, paths: Iterable[str | Path]) -> None:
        """Start decoding ``paths`` in the background (most important first)."""
        for path in paths:
            key = Path(path)
            with self._lock:
                if key in self._entries:
                    self._entries.move_to_end(key)
                    continue
            self._store(key, self._executor.submit(_load_frozen, self._loader, key))

    def clear(self) -> None:
        with self._lock:
            entries = list(self._entries.values())
            self._entries.clear()
        for future in entries:
            future.cancel()

    def close(self) -> None:
        self.clear()
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _store(self, key: Path, future: Future) -> None:
        with self._lock:
            self._entries[key] = future
            self._entries.move_to_end(key)
            while len(self._entries) > self._capacity:
                _old_key, old = self._entries.popitem(last=False)
                old.cancel()


def neighbours(paths: list, index: int, radius: int = 1) -> list:
    """Paths around ``index``, nearest (and forward) first."""
    order = []
    for step in range(1, radius + 1):
        for candidate in (index + step, index - step):
            if 0 <= candidate < len(paths):
                order.append(paths[candidate])
    return order


__all__ = ["ImagePrefetcher", "neighbours"]
