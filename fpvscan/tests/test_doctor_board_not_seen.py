"""doctor.py step 5 must blame the cable/port, not the Python binding."""
from __future__ import annotations

from pathlib import Path


SRC = (Path(__file__).resolve().parents[1]
       / "scripts" / "doctor.py").read_text(encoding="utf-8")


def test_open_fail_names_usb3_and_a_held_device():
    """bladerf_open < 0 after a successful load_lib is almost never
    ctypes. If the hint drops USB 3.0 / Nuand / bladeRF-cli, the
    next hour is spent reinstalling Python instead of moving the
    cable off a hub or killing the other process that owns the
    board.
    """
    start = SRC.index('print("\\n5. плата")')
    end = SRC.index('print("\\n6. FPGA")')
    body = SRC[start:end]
    assert "USB 3.0" in body
    assert "Nuand" in body
    assert "bladeRF-cli" in body
    assert "return 1" in body
