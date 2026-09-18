"""Waterfall scroll direction, and the quick-tune blob the ctypes call writes into."""
from __future__ import annotations

from pathlib import Path

from fpvscan.sdr import bladerf


def _html() -> str:
    return (Path(__file__).resolve().parents[1]
            / "fpvscan" / "web" / "static" / "index.html").read_text(encoding="utf-8")


def test_waterfall_scrolls_old_rows_down():
    """pushSpectrum copies H-1 rows to y=1 and paints the new row at y=0.
    putImageData(prev, 0, 0) overwrites the history; y=-1 throws and the
    whole onmessage handler dies — ribbon and hits freeze mid-sweep.
    """
    html = _html()
    start = html.index("function pushSpectrum")
    end = html.index("function renderHits")
    body = html[start:end]
    assert "fctx.getImageData(0, 0, W, H - 1)" in body
    assert "fctx.putImageData(prev, 0, 1)" in body
    assert "putImageData(prev, 0, 0)" not in body


def test_quick_tune_blob_is_oversized_so_xA4_profile_fits():
    """prime_quick_tune allocates QUICK_TUNE_BYTES and hands the pointer
    to libbladeRF. The xA4 RFIC profile is smaller, but a short buffer
    is a write past the Python bytes object — USB hangs look like a
    dead radio. 128 is the spare the driver comments call for.
    """
    assert bladerf.QUICK_TUNE_BYTES == 128
    assert bladerf.QUICK_TUNE_BYTES >= 64
    assert "msvcr120.dll" in bladerf.WIN_DEPS
    assert "msvcp120.dll" in bladerf.WIN_DEPS
