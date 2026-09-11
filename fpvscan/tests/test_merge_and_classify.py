"""Merge tolerance must not swallow neighboring 5.8 GHz channels."""
from __future__ import annotations

from pathlib import Path

from fpvscan.bands import BAND_B, BAND_F
from fpvscan.dsp import demod
from fpvscan.engine import Engine


def test_merge_tol_is_narrower_than_band_b_and_f_spacing():
    """Fatshark F is 20 MHz, Band B is 19 MHz. If MERGE_TOL_HZ were
    widened to "fix duplicates", two live VTx would collapse into one
    hit and LOCK would sit between them.
    """
    b_mhz = [BAND_B[f"B{i}"] for i in range(1, 9)]
    f_mhz = [BAND_F[f"F{i}"] for i in range(1, 9)]
    b_gap = min(abs(a - b) for a, b in zip(b_mhz, b_mhz[1:])) * 1e6
    f_gap = min(abs(a - b) for a, b in zip(f_mhz, f_mhz[1:])) * 1e6
    assert b_gap == 19e6
    assert f_gap == 20e6
    assert Engine.MERGE_TOL_HZ < b_gap
    assert Engine.MERGE_TOL_HZ < f_gap
    assert Engine.MERGE_TOL_HZ == 6e6


def test_classify_search_window_covers_pal_and_ntsc():
    """Window 15.0–16.2 kHz is the only place analog line-rate is
    accepted. Tightening it to a PAL-only slice would drop NTSC; opening
    it past ~20 kHz would start scoring harmonics as the fundamental.
    Decimation to ~400 kHz must keep 16.2 kHz below Nyquist.
    """
    src = Path(demod.__file__).read_text(encoding="utf-8")
    assert "freqs > 15.0e3" in src
    assert "freqs < 16.2e3" in src
    assert "int(fs / 400e3)" in src
    assert 15.0e3 < demod.LINE_PAL < 16.2e3
    assert 15.0e3 < demod.LINE_NTSC < 16.2e3
    fs = 20e6
    dec = max(1, int(fs / 400e3))
    assert (fs / dec) / 2 > 16.2e3
