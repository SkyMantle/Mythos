"""Sweep hops whose span crosses 400 MHz or 6 GHz must not paint off-canvas."""
from __future__ import annotations

from pathlib import Path


HTML = (Path(__file__).resolve().parents[1]
        / "fpvscan" / "web" / "static" / "index.html").read_text(encoding="utf-8")


def test_push_spectrum_skips_bins_outside_ribbon():
    """First hop is start+fs/2 with span=fs, so bins sit on 400 MHz.
    Last hop crosses 6 GHz. Without the F0/F1 guard, x is -1 or
    >=RW and the ribbon either throws or writes the last pixel as
    if 6 GHz were occupied — a false 5.8 hit the operator then locks.
    """
    start = HTML.index("function pushSpectrum")
    end = HTML.index("function renderHits")
    body = HTML[start:end]
    assert "if (f < F0 || f >= F1) continue" in body
    assert "Math.floor((f - F0) / (F1 - F0) * RW)" in body
