"""Windows libbladeRF discovery — the usual 'DLL not found' traps."""
from __future__ import annotations

from pathlib import Path

from fpvscan.sdr import bladerf as B


SRC = (Path(__file__).resolve().parents[1]
       / "fpvscan" / "sdr" / "bladerf.py").read_text(encoding="utf-8")


def test_windows_search_lists_program_files_and_nuand():
    """The Nuand installer puts bladeRF.dll under Program Files\\bladeRF\\x64,
    not on PATH. Dropping those bases makes doctor/load_lib say «not
    installed» on a machine where bladeRF-cli -e info already works.
    `_search_dirs()` skips missing directories, so this pins the source
    list — those paths are absent on a Linux CI image.
    """
    body = SRC[SRC.index("def _search_dirs"):SRC.index("def _lib_names")]
    assert "Program Files" in body
    assert "ProgramFiles(x86)" in body
    assert "Nuand" in body
    assert "x64" in body
    assert "_win_reg_dirs" in body
    names = SRC[SRC.index("def _lib_names"):SRC.index("def find_lib_files")]
    assert 'return ["bladeRF.dll"]' in names


def test_try_load_adds_dll_directory_for_python38():
    """Since Python 3.8, ctypes no longer walks PATH for dependent DLLs.
    libusb-1.0.dll sits next to bladeRF.dll; without add_dll_directory
    Windows reports «module not found» for a library that is on disk.
    winmode=0 is the fallback that restores the old search.
    """
    body = SRC[SRC.index("def _try_load"):SRC.index("def load_lib")]
    assert "add_dll_directory" in body
    assert "winmode" in body


def test_load_failure_hint_names_libusb_and_bitness():
    """The «found on disk but will not load» path is not the same as
    `_not_installed_msg`. Operators copy the first paragraph into chat.
    If it does not name libusb / 64-bit / VC++, they reinstall the SDK.
    """
    body = SRC[SRC.index("def load_lib"):SRC.index("def _not_installed_msg")]
    assert "libusb-1.0.dll" in body
    assert "64-бітний" in body
    assert "Visual C++" in body


def test_windows_not_installed_hint_names_nuand_installer():
    """Linux hint (hostedxa4) is already pinned. The Windows branch must
    name the Nuand installer and the exact --lib path, otherwise a Pi
    package name is what the operator googles on a laptop.
    """
    body = SRC[SRC.index("def _not_installed_msg"):SRC.index("def lib_path")]
    assert "Nuand" in body
    assert "bladeRF-cli" in body
    assert r"C:\\Program Files\\bladeRF\\x64\\bladeRF.dll" in body


def test_load_lib_returns_the_cached_handle():
    """`_LIB` is process-global. A later --lib / BLADERF_LIB_DIR must not
    silently keep the first handle — but today it does, so doctor --lib
    after a previous import is a no-op. Pin the cache so a change here
    is deliberate (and so tests that stub _LIB can restore it).
    """
    sentinel = object()
    old_lib, old_path = B._LIB, B._LIB_PATH
    try:
        B._LIB = sentinel
        B._LIB_PATH = "/already/loaded"
        assert B.load_lib("/some/other/bladeRF.dll") is sentinel
        assert B._LIB_PATH == "/already/loaded"
    finally:
        B._LIB, B._LIB_PATH = old_lib, old_path


def test_win_reg_dirs_is_empty_off_windows():
    """_win_reg_dirs imports winreg. On Linux that ImportError must stay
    an empty list, not leak into _search_dirs if someone moves the call
    outside the win32 branch.
    """
    import sys
    if sys.platform.startswith("win"):
        return
    assert B._win_reg_dirs() == []
