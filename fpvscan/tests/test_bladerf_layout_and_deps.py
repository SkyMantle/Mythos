"""bladeRF metadata tail and Windows-only DLL deps — USB and doctor lies."""
from __future__ import annotations

import ctypes
import os

from fpvscan.sdr.bladerf import _Metadata, _search_dirs, missing_deps


def test_metadata_has_32_byte_reserved_tail():
    """libbladeRF's bladerf_metadata ends with a reserved blob. If the
    ctypes struct omits it, actual_count / status are still at the
    right offsets for the first read, but the next stack object is
    overwritten and overrun counting (and then the picture) goes
    haywire after one USB transfer.
    """
    names = [n for n, typ in _Metadata._fields_]
    types = dict(_Metadata._fields_)
    assert names[-1] == "reserved"
    assert names[-2] == "actual_count"
    assert ctypes.sizeof(types["reserved"]) == 32
    assert ctypes.sizeof(_Metadata) >= 52


def test_missing_deps_empty_off_windows(tmp_path):
    """doctor.py prints 'бракує libusb' from missing_deps(). On Linux
    those DLLs do not exist next to libbladeRF.so, so the check must
    be a no-op or every Pi boot looks like a broken Windows install.
    """
    fake = tmp_path / "libbladeRF.so"
    fake.write_bytes(b"")
    assert missing_deps(str(fake)) == []


def test_missing_lib_dir_env_is_skipped(tmp_path, monkeypatch):
    """_search_dirs only yields directories that exist. A stale
    BLADERF_LIB_DIR pointing at a removed unzip would otherwise sit
    first in the list; doctor then reports it as a search location
    for a library that cannot be there.
    """
    gone = tmp_path / "nope"
    monkeypatch.setenv("BLADERF_LIB_DIR", str(gone))
    dirs = _search_dirs()
    norm = os.path.normpath(str(gone))
    assert all(os.path.normpath(d) != norm for d in dirs)
