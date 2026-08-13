from __future__ import annotations

import struct
import zlib
from pathlib import Path

from PySide6.QtGui import QImageReader

from cosmos_toolbox.cabf_activity import _load_preview_pixmap


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    checksum = zlib.crc32(kind)
    checksum = zlib.crc32(payload, checksum)
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", checksum)


def _write_large_compressible_rgb_png(path: Path, width: int, height: int) -> None:
    compressor = zlib.compressobj(level=1)
    compressed = bytearray()
    row = b"\0" * (1 + width * 3)
    for _ in range(height):
        compressed.extend(compressor.compress(row))
    compressed.extend(compressor.flush())
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", header)
        + _png_chunk(b"IDAT", bytes(compressed))
        + _png_chunk(b"IEND", b"")
    )


def test_cabf_preview_loads_image_above_qt_default_allocation_limit(tmp_path: Path):
    image_path = tmp_path / "large-source.png"
    _write_large_compressible_rgb_png(image_path, width=16_384, height=4_097)
    previous_limit = QImageReader.allocationLimit()
    QImageReader.setAllocationLimit(256)

    try:
        pixmap, source_size = _load_preview_pixmap(image_path)

        assert source_size == (16_384, 4_097)
        assert pixmap.width() == 4_096
        assert pixmap.height() == 1_024
        assert QImageReader.allocationLimit() == 256
    finally:
        QImageReader.setAllocationLimit(previous_limit)
