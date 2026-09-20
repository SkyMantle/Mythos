"""headless default runtime and doctor.py's 32-bit stop."""
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_headless_default_secs_is_long_enough_for_one_sweep():
    """`--secs` default 0 returns before the first spectrum. 45 s is the
    window that covers a 400 MHz–6 GHz pass at 35 Msps on the xA4.
    """
    src = (ROOT / "scripts" / "headless.py").read_text(encoding="utf-8")
    assert "default=45.0" in src
    assert "while time.time() - t0 < a.secs:" in src


def test_doctor_refuses_32bit_python_before_loading_the_dll():
    """x64 bladeRF.dll on 32-bit Python is 'module not found' with no
    useful hint. doctor must stop on word size before find_lib_files.
    """
    src = (ROOT / "scripts" / "doctor.py").read_text(encoding="utf-8")
    assert "ctypes.sizeof(ctypes.c_void_p) == 8" in src
    assert "32-бітний Python" in src
    assert "files = [a.lib] if a.lib else B.find_lib_files()" in src
