from __future__ import annotations

import pytest

from cosmos_toolbox.file_lock import LockBusyError, exclusive_lock


def test_second_holder_is_rejected_and_lock_file_stays_one_byte(tmp_path):
    path = tmp_path / ".lock"
    with exclusive_lock(path, "busy"):
        with pytest.raises(LockBusyError, match="busy"):
            with exclusive_lock(path, "busy"):
                pass
    with exclusive_lock(path, "busy"):
        pass
    assert path.read_bytes() == b"0"
