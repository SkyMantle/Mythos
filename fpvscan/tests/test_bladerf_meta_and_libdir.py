"""bladeRF metadata layout and library search — USB overflows and 'DLL not found'."""
from __future__ import annotations

import inspect
import os

from fpvscan.sdr.bladerf import (
    META_STATUS_OVERRUN,
    BladeRF,
    _Metadata,
    _search_dirs,
)


def test_metadata_status_is_a_separate_word_from_flags():
    """libbladeRF fills timestamp, flags, then status. Overrun is
    `status & 0x1`. If the ctypes struct swaps flags and status, the
    console's зривів counter stays 0 while USB is actually dropping,
    and the operator keeps blaming gain.
    """
    names = [n for n, _ in _Metadata._fields_]
    assert names[:4] == ["timestamp", "flags", "status", "actual_count"]
    assert _Metadata.flags.offset == 8
    assert _Metadata.status.offset == 12
    src = inspect.getsource(BladeRF.read)
    assert "meta.status & META_STATUS_OVERRUN" in src
    assert META_STATUS_OVERRUN == 1 << 0


def test_bladerf_lib_dir_env_is_searched_first(tmp_path, monkeypatch):
    """Custom installs (non-standard prefix on Pi, unzipped SDK on
    Windows) only work if BLADERF_LIB_DIR wins over PATH. Doctor then
    reports 'не знайдено модуль' for a library that is sitting right
    there.
    """
    monkeypatch.setenv("BLADERF_LIB_DIR", str(tmp_path))
    dirs = _search_dirs()
    assert dirs, "search must return the env dir when it exists"
    assert os.path.normpath(dirs[0]) == os.path.normpath(str(tmp_path))
