"""bringup.py is the first script on a live stand. MHz vs Hz here
tunes the bladeRF to 5.8 kHz and the operator thinks the VTx is dead.
"""
from __future__ import annotations

from pathlib import Path


SRC = (Path(__file__).resolve().parents[1]
       / "scripts" / "bringup.py").read_text(encoding="utf-8")


def test_bringup_freq_rate_and_scan_are_mhz():
    assert "fc = a.freq * 1e6 if a.freq else (a.scan[0] + a.scan[1]) / 2 * 1e6" in SRC
    assert "fs = a.rate * 1e6" in SRC
    assert "scan_band(src, a.scan[0] * 1e6, a.scan[1] * 1e6, fs)" in SRC
    assert 'help="МГц"' in SRC
    assert "Мвідл/с" in SRC


def test_bringup_file_uses_file_source_not_bladerf():
    """`--file caps/x.cf32` must replay the capture. Opening BladeRF
    here fails on a laptop and hides the offline debug path.
    """
    assert "from fpvscan.sdr.file import FileSource" in SRC
    assert "src = FileSource(a.file)" in SRC
    assert "from fpvscan.sdr.bladerf import BladeRF" in SRC


def test_bringup_grab_dumps_pll_and_level_strips_dc():
    """First milliseconds after retune are the LO transient. Measuring
    peak including the DC hump picks a gain that then clips on LOCK.
    """
    assert "src.read(int(fs * 0.02))" in SRC
    assert "d = x - np.mean(x)" in SRC
