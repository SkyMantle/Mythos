"""Pi 5 / Ubuntu multiarch is where libbladeRF actually lives."""
from __future__ import annotations

import sys
from pathlib import Path

from fpvscan.sdr import bladerf as B


SRC = (Path(__file__).resolve().parents[1]
       / "fpvscan" / "sdr" / "bladerf.py").read_text(encoding="utf-8")


def test_search_dirs_include_debian_multiarch():
    """apt on Pi 5 (aarch64) and Ubuntu x86_64 put libbladeRF.so.2 under
    the multiarch triplet, not /usr/lib. Dropping those paths makes
    doctor/load_lib say 'not installed' on a board that bladeRF-cli sees.
    `_search_dirs()` skips missing directories, so this pins the source
    list — the aarch64 path is absent on an x86 CI image.
    """
    assert "/usr/lib/aarch64-linux-gnu" in SRC
    assert "/usr/lib/x86_64-linux-gnu" in SRC
    assert "/usr/local/lib" in SRC
    body = SRC[SRC.index("def _search_dirs"):SRC.index("def _lib_names")]
    assert "BLADERF_LIB_DIR" in body


def test_linux_soname_is_libbladerf_so_2_first():
    """The apt package ships the SONAME symlink. Searching only
    libbladeRF.so misses a default install_pi.sh image.
    """
    if sys.platform.startswith("win"):
        assert B._lib_names() == ["bladeRF.dll"]
        return
    assert B._lib_names()[0] == "libbladeRF.so.2"
    assert "libbladeRF.so" in B._lib_names()


def test_linux_not_installed_hint_names_the_fpga_package():
    """An xA4 without hostedxA4 enumerates then every sync_rx fails.
    The hint must name the FPGA package, not just libbladerf2.
    """
    if sys.platform.startswith("win"):
        return
    msg = B._not_installed_msg()
    assert "libbladerf2" in msg
    assert "bladerf-fpga-hostedxa4" in msg
    assert "libbladeRF.so.2" in msg
