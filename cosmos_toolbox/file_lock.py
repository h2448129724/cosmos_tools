"""Cross-platform, non-blocking exclusive lock on a marker file.

The OS releases the lock when the process dies, so a leftover lock file never
blocks recovery; only a live holder does.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


class LockBusyError(RuntimeError):
    """Another live process or handle already holds the lock."""


if os.name == "nt":
    import msvcrt

    def _acquire(fd: int) -> None:
        msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)

    def _release(fd: int) -> None:
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
else:
    import fcntl

    def _acquire(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def _release(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_UN)


@contextmanager
def exclusive_lock(path: str | Path, busy_message: str) -> Iterator[None]:
    """Hold an exclusive lock on ``path`` or raise :class:`LockBusyError`."""
    stream = open(path, "a+b")
    try:
        if stream.tell() == 0:
            # msvcrt locks a byte range, so the file needs one byte to lock.
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            _acquire(stream.fileno())
        except OSError as exc:
            raise LockBusyError(busy_message) from exc
        try:
            yield
        finally:
            stream.seek(0)
            _release(stream.fileno())
    finally:
        stream.close()
