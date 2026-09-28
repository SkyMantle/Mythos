"""doctor.py step 3 must name libusb — the usual Windows 'DLL not found'."""
from __future__ import annotations

from pathlib import Path


SRC = (Path(__file__).resolve().parents[1]
       / "scripts" / "doctor.py").read_text(encoding="utf-8")


def test_missing_deps_names_libusb_on_windows():
    """Windows reports 'module not found' for bladeRF.dll when
    libusb-1.0.dll is the file that is actually missing. If step 3
    drops that name, the operator reinstalls Python or the Nuand
    package and never copies the dependency next to the DLL.
    Linux has no WIN_DEPS check — the hint stays behind the
    Windows guard so Pi doctor output does not send people to
    a .dll.
    """
    assert "libusb-1.0.dll" in SRC
    assert "missing_deps" in SRC
    assert 'if miss and sys.platform.startswith("win")' in SRC
    assert "бракує" in SRC
